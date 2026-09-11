// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Run the cyclic schedule on the device and compare it with the fixtures.
//
// cyclic_schedule_test.cpp checks the schedule headers on the host. This runs
// the same headers compiled for RISC-V, on the core grid the relay will use,
// and reads the result back. It is the first device step of the port and it
// carries no protocol: no core-to-core traffic, no semaphores, nothing that
// waits. A failure here is a placement, runtime-argument or DRAM-addressing
// mistake, which is worth separating from the protocol bugs that come next.
//
// What it establishes:
//   * the logical-core-to-CoreCoord placement is what parity_snake.hpp says,
//     because each core is told only its own logical id and the host checks
//     the page that core wrote;
//   * a persistent kernel can run the whole t = 0..T loop;
//   * the schedule headers evaluate identically on device and on host, and
//     agree with the Python reference where a fixture exists.

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <cstdint>
#include <vector>

#include "autograd/auto_context.hpp"
#include "cyclic_schedule_golden.hpp"
#include "metal/ops/cyclic_sdpa_bw/device/parity_snake.hpp"

namespace {

using namespace ttml::metal::ops::cyclic_sdpa_bw;
namespace golden = ttml::metal::ops::cyclic_sdpa_bw::golden;
namespace tt_dist = tt::tt_metal::distributed;

constexpr uint32_t kFields = 8;  // must match the kernel
constexpr const char* kKernelPath =
    "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/dataflow/cyclic_schedule_probe.cpp";

// One timestep of one core, as the kernel writes it.
struct Record {
    uint32_t i;
    uint32_t j;
    uint32_t producer_core;
    uint32_t producer_internal;
    uint32_t next_consumer;
    uint32_t later_streak_start;
    uint32_t endpoint_threshold;
    uint32_t spill_endpoint;
};

// Run the probe for C cores on a grid_w x grid_h region and return the
// records, indexed [(c - 1) * (T + 1) + t].
std::vector<Record> run_probe(uint32_t C, uint32_t grid_w, uint32_t grid_h) {
    using namespace tt::tt_metal;

    auto& mesh = ttml::autograd::ctx().get_device();
    const uint32_t timesteps = 2u * C + 1u;
    const uint32_t page_bytes = kFields * sizeof(uint32_t) * timesteps;

    const auto local_config = tt_dist::DeviceLocalBufferConfig{
        .page_size = page_bytes,
        .buffer_type = BufferType::DRAM,
    };
    const auto buffer_config = tt_dist::ReplicatedBufferConfig{.size = page_bytes * C};
    auto out = tt_dist::MeshBuffer::create(buffer_config, local_config, &mesh);

    auto program = CreateProgram();
    const auto region = CoreRange(CoreCoord{0, 0}, CoreCoord{grid_w - 1, grid_h - 1});

    // Scratch for one page. The kernel writes it directly rather than going
    // through the CB protocol; only the address is used.
    CreateCircularBuffer(
        program,
        region,
        CircularBufferConfig(page_bytes, {{tt::CBIndex::c_0, tt::DataFormat::UInt32}})
            .set_page_size(tt::CBIndex::c_0, page_bytes));

    std::vector<uint32_t> compile_args = {C};
    tt::tt_metal::TensorAccessorArgs(*out->get_reference_buffer()).append_to(compile_args);
    const auto kernel = CreateKernel(
        program,
        kKernelPath,
        region,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_args});

    // Each core learns only its own logical id; where it sits is the
    // placement's business.
    for (uint32_t c = 1; c <= C; ++c) {
        const auto xy = placement_of(C, grid_w, c);
        SetRuntimeArgs(program, kernel, CoreCoord{xy.x, xy.y}, {out->address(), c});
    }

    auto workload = tt_dist::MeshWorkload();
    workload.add_program(tt_dist::MeshCoordinateRange(mesh.shape()), std::move(program));
    tt_dist::EnqueueMeshWorkload(mesh.mesh_command_queue(), workload, /*blocking=*/true);

    std::vector<uint32_t> raw(kFields * timesteps * C, 0xDEADBEEFu);
    tt_dist::ReadShard(
        mesh.mesh_command_queue(), raw, out, tt_dist::MeshCoordinate(0, 0), /*blocking=*/true);

    std::vector<Record> records(timesteps * C);
    for (size_t n = 0; n < records.size(); ++n) {
        const uint32_t* f = raw.data() + n * kFields;
        records[n] = Record{f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]};
    }
    return records;
}

const golden::Config* find_config(uint32_t C) {
    for (uint32_t k = 0; k < golden::kNumConfigs; ++k) {
        if (golden::kConfigs[k].C == C) {
            return &golden::kConfigs[k];
        }
    }
    return nullptr;
}

// Check the device's records against the headers, and against the full table
// from the Python reference where one exists.
void check_probe(uint32_t C, const uint32_t* golden_pairs) {
    const auto* cfg = find_config(C);
    ASSERT_NE(cfg, nullptr) << "no golden config for C=" << C;

    auto& mesh = ttml::autograd::ctx().get_device();
    const auto grid = mesh.compute_with_storage_grid_size();
    if (cfg->grid_w > grid.x || cfg->grid_h > grid.y) {
        GTEST_SKIP() << "C=" << C << " needs a " << cfg->grid_w << "x" << cfg->grid_h
                     << " region; device offers " << grid.x << "x" << grid.y;
    }

    const CyclicSchedule s(C);
    const uint32_t timesteps = s.T() + 1u;
    const auto records = run_probe(C, cfg->grid_w, cfg->grid_h);

    for (uint32_t c = 1; c <= C; ++c) {
        for (uint32_t t = 0; t < timesteps; ++t) {
            const Record& r = records[(c - 1u) * timesteps + t];
            const auto expected = s.pair(c, t);
            const auto producer = s.producer(c, t);
            const bool later = s.is_later_streak_start(expected.i, t);

            // A wrong placement shows up here: the core at placement_of(c)
            // would have reported a different core's schedule.
            EXPECT_EQ(r.i, expected.i) << "C=" << C << " c=" << c << " t=" << t << " row";
            EXPECT_EQ(r.j, expected.j) << "C=" << C << " c=" << c << " t=" << t << " column";
            EXPECT_EQ(r.producer_core, producer.core) << "C=" << C << " c=" << c << " t=" << t;
            EXPECT_EQ(r.producer_internal, producer.internal ? 1u : 0u)
                << "C=" << C << " c=" << c << " t=" << t;
            EXPECT_EQ(r.next_consumer, s.next_consumer(expected.i, t))
                << "C=" << C << " c=" << c << " t=" << t;
            EXPECT_EQ(r.later_streak_start, later ? 1u : 0u)
                << "C=" << C << " c=" << c << " t=" << t;
            EXPECT_EQ(r.endpoint_threshold, later ? s.endpoint_threshold(expected.i, t) : 0u)
                << "C=" << C << " c=" << c << " t=" << t;
            EXPECT_EQ(r.spill_endpoint, later ? s.spill_endpoint(expected.i, t) : 0u)
                << "C=" << C << " c=" << c << " t=" << t;

            if (golden_pairs != nullptr) {
                const uint32_t base = 2u * ((c - 1u) * timesteps + t);
                EXPECT_EQ(r.i, golden_pairs[base]) << "C=" << C << " c=" << c << " t=" << t;
                EXPECT_EQ(r.j, golden_pairs[base + 1u]) << "C=" << C << " c=" << c << " t=" << t;
            }
        }
    }
}

}  // namespace

TEST(CyclicScheduleDeviceTest, FirstLightFourCoresOnTwoByTwo) {
    check_probe(4, golden::kPairs_C4);
}

TEST(CyclicScheduleDeviceTest, EightCoresOnFourByTwo) {
    check_probe(8, golden::kPairs_C8);
}

TEST(CyclicScheduleDeviceTest, SixteenCoresOnFourByFour) {
    check_probe(16, golden::kPairs_C16);
}

TEST(CyclicScheduleDeviceTest, ThirtyTwoCoresOnEightByFour) {
    check_probe(32, nullptr);
}

TEST(CyclicScheduleDeviceTest, SixtyFourCoresOnEightByEight) {
    check_probe(64, nullptr);
}
