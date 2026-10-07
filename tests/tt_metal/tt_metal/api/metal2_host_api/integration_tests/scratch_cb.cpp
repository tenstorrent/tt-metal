// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// End-to-end scratch CB sync (experimental/scratch_cb_api.h) on Blackhole silicon: DM <-> compute handoffs on
// the stream counters of CB 62 (channel 0) and CB 63 (channel 1), through caller-owned L1 rings.

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>

#include "impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/program_spec_hw_fixture.hpp"
namespace tt::tt_metal::experimental {
namespace scratch_cb_test {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecHWTest;

// ============================================================================
// Scratch CB: one DM kernel (test_scratch_cb_dm.cpp, producer or consumer by `role`) and one compute kernel
// (test_scratch_cb_compute.cpp), three patterns selected by the `pattern` compile-time arg:
//   A (0)  datacopy DM -> compute -> DM through two caller-owned rings, both channels live at once
//   B (1)  bounded producer: DM producer against a slow UNPACK consumer, in-flight high-water mark
//   C (2)  ping-pong: one DM kernel owns both channels and sends every compute result back as the next input
// ============================================================================

constexpr std::uint32_t kPatternA = 0;
constexpr std::uint32_t kPatternB = 1;
constexpr std::uint32_t kPatternC = 2;

const NodeRange kNode{{0, 0}, {0, 0}};
constexpr std::uint32_t kNumTiles = 64;            // pattern A payload; num_iters repeats it in rounds
constexpr std::uint32_t kTileBytes = 32 * 32 * 2;  // Float16_b 32x32
constexpr std::uint32_t kTileWords = kTileBytes / 4;
constexpr std::uint32_t kSlots = 2;  // largest pattern A ring; smaller rings guard their actual capacity
constexpr std::uint32_t kGuardBytes = 64;
constexpr std::uint32_t kGuardValue = 0xA5C39E71u;
// Fixed L1 regions, well above the kernel-config region (as in compute_semaphore.cpp).
constexpr std::uint32_t kInAddr = 100 * 1024 + 0x40000;
constexpr std::uint32_t kRingAAddr = kInAddr + kNumTiles * kTileBytes + kGuardBytes;
constexpr std::uint32_t kRingBAddr = kRingAAddr + kSlots * kTileBytes + kGuardBytes;
constexpr std::uint32_t kOutAddr = kRingBAddr + kSlots * kTileBytes + kGuardBytes;
constexpr std::uint32_t kReportAddr = kOutAddr + kNumTiles * kTileBytes + kGuardBytes;
constexpr std::uint32_t kReportWords = 2;

// Pattern B ring depth, passed to the kernels as the scratch capacity.
constexpr std::uint32_t kBoundedDepth = 4;

// Real CBs on every ID the CB API allows; mirrors the DM kernel's kRealCbs.
constexpr std::uint32_t kRealCbs = 62;
constexpr std::uint32_t kRealCbEntries = 2;
constexpr std::uint32_t kRealCbEntryBytes = 64;

// Mirrors the DM kernel: the producer stamps word 0 of transfer `sequence` with this tag.
std::uint32_t SequenceTag(std::uint32_t sequence) {
    return ((0x4000u | ((sequence >> 8) & 0x1fffu)) << 16) | (0x3f00u | (sequence & 0xffu));
}

struct ScratchCbRun {
    std::uint32_t pattern = kPatternA;
    std::uint32_t capacity = 2;
    std::uint32_t num_iters = kNumTiles;
    std::uint32_t nosync = 0;  // negative control: drop reserve/wait
    std::uint32_t batch = 1;   // pattern B: pages per reserve_back/push_back
    std::uint32_t real_cbs = 0;
};

// Pattern A: producer (BRISC), compute and consumer (NCRISC). Patterns B and C: producer and compute only.
Program MakeScratchCbProgram(distributed::MeshDevice& mesh_device, const ScratchCbRun& run, const NodeRange& nodes) {
    const std::vector<std::string> dm_args = {
        "num_iters", "num_tiles", "tile_bytes", "in_addr", "ring_addr", "ring_b_addr", "out_addr", "report_addr"};
    auto make_dm = [&](const char* name, DataMovementProcessor processor, std::uint32_t role) {
        auto dm = MakeMinimalGen1DMKernel(name, processor);
        dm.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/test_scratch_cb_dm.cpp";
        dm.runtime_arg_schema.runtime_arg_names = dm_args;
        dm.compile_time_args = {
            {"pattern", run.pattern},
            {"role", role},
            {"capacity", run.capacity},
            {"nosync", run.nosync},
            {"batch", run.batch},
            {"real_cbs", run.real_cbs}};
        return dm;
    };
    auto producer = make_dm("producer", DataMovementProcessor::RISCV_0, 0);
    auto consumer = make_dm("consumer", DataMovementProcessor::RISCV_1, 1);

    auto compute = MakeMinimalGen1ComputeKernel("compute");
    compute.source = "tests/tt_metal/tt_metal/test_kernels/compute/test_scratch_cb_compute.cpp";
    compute.runtime_arg_schema.runtime_arg_names = {"num_iters", "tile_bytes", "ring_a_addr", "ring_b_addr"};
    compute.compile_time_args = {{"pattern", run.pattern}, {"capacity", run.capacity}, {"nosync", run.nosync}};

    std::vector<DataflowBufferSpec> real_cbs;
    if (run.real_cbs) {
        // Each DFB takes the lowest free CB ID on its nodes, so spec order gives IDs 0-61.
        for (std::uint32_t id = 0; id < kRealCbs; ++id) {
            const std::string name = "real_cb_" + std::to_string(id);
            auto dfb = MakeMinimalDFB(name, kRealCbEntryBytes, kRealCbEntries);
            dfb.data_format_metadata = tt::DataFormat::Float16_b;
            producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{name}, name));
            consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{name}, name));
            real_cbs.push_back(std::move(dfb));
        }
    }

    const bool datacopy = run.pattern == kPatternA;
    std::vector<KernelSpec> kernels = {producer, compute};
    std::vector<std::string> kernel_names = {"producer", "compute"};
    if (datacopy) {
        kernels.push_back(consumer);
        kernel_names.push_back("consumer");
    }
    ProgramSpec spec{
        .name = "scratch_cb",
        .kernels = kernels,
        .dataflow_buffers = real_cbs,
        .work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", nodes, kernel_names)},
    };
    Program program = MakeProgramFromSpec(mesh_device, spec);

    ProgramRunArgs::KernelRunArgs producer_args{.kernel = KernelSpecName{"producer"}};
    ProgramRunArgs::KernelRunArgs compute_args{.kernel = KernelSpecName{"compute"}};
    ProgramRunArgs::KernelRunArgs consumer_args{.kernel = KernelSpecName{"consumer"}};
    for (const auto& node : nodes) {
        for (auto [dm_args, ring_addr] :
             {std::pair{&producer_args, kRingAAddr}, std::pair{&consumer_args, kRingBAddr}}) {
            AddRuntimeArgsForNode(
                dm_args->runtime_arg_values,
                node,
                {{"num_iters", run.num_iters},
                 {"num_tiles", kNumTiles},
                 {"tile_bytes", kTileBytes},
                 {"in_addr", kInAddr},
                 {"ring_addr", ring_addr},
                 {"ring_b_addr", kRingBAddr},
                 {"out_addr", kOutAddr},
                 {"report_addr", kReportAddr}});
        }
        AddRuntimeArgsForNode(
            compute_args.runtime_arg_values,
            node,
            {{"num_iters", run.num_iters},
             {"tile_bytes", kTileBytes},
             {"ring_a_addr", kRingAAddr},
             {"ring_b_addr", kRingBAddr}});
    }
    ProgramRunArgs args;
    args.kernel_run_args = {producer_args, compute_args};
    if (datacopy) {
        args.kernel_run_args.push_back(consumer_args);
    }
    SetProgramRunArgs(program, args);
    return program;
}

// Tile t, datum k = bf16 0x4000 + (t << 7) + (k & 0x7F), XOR'd with seed in the mantissa: normal positive
// values (exact through a bf16 datacopy) and distinct per tile, so a stale slot read is detected.
std::vector<std::uint32_t> MakeInputTiles(std::uint32_t seed) {
    std::vector<std::uint32_t> in(kNumTiles * kTileWords);
    for (std::uint32_t t = 0; t < kNumTiles; ++t) {
        for (std::uint32_t w = 0; w < kTileWords; ++w) {
            const std::uint32_t lo = (0x4000u + (t << 7) + ((2 * w) & 0x7Fu)) ^ seed;
            const std::uint32_t hi = (0x4000u + (t << 7) + ((2 * w + 1) & 0x7Fu)) ^ seed;
            in[t * kTileWords + w] = (hi << 16) | lo;
        }
    }
    return in;
}

struct DatacopyResult {
    std::uint32_t mismatched_tiles = 0;  // final output tiles that differ from the input
    std::uint32_t scratch_errors = 0;    // transfers the consumer saw stale, reordered or corrupt
    std::uint32_t real_cb_errors = 0;
};

// PATTERN A on every node: writes distinct input tiles (varied by `seed` and the node), poisons the rings and
// the output, guards both ends of every region, launches once, and totals the result over all nodes.
DatacopyResult RunScratchCbDatacopy(
    distributed::MeshDevice& mesh_device, const ScratchCbRun& run, std::uint32_t seed, const NodeRange& nodes = kNode) {
    EXPECT_EQ(run.num_iters % kNumTiles, 0u);
    std::vector<std::uint32_t> poison((kReportAddr + kReportWords * 4 - kRingAAddr) / 4, 0xDEADBEEFu);
    std::vector<std::uint32_t> guard(kGuardBytes / 4, kGuardValue);
    const std::vector<std::uint32_t> guard_addresses = {
        kRingAAddr - kGuardBytes,
        kRingAAddr + run.capacity * kTileBytes,
        kRingBAddr - kGuardBytes,
        kRingBAddr + run.capacity * kTileBytes,
        kOutAddr + kNumTiles * kTileBytes};

    // The node index stays below 0x80, so the XOR only touches the low mantissa bits.
    std::vector<std::vector<std::uint32_t>> inputs;
    for (const auto& node : nodes) {
        inputs.push_back(MakeInputTiles(seed ^ (inputs.size() & 0x7Fu)));
        slow_dispatch::WriteToL1(mesh_device, node, kInAddr, inputs.back());
        slow_dispatch::WriteToL1(mesh_device, node, kRingAAddr, poison);  // covers both rings, out and report
        for (auto address : guard_addresses) {
            slow_dispatch::WriteToL1(mesh_device, node, address, guard);
        }
    }
    LaunchProgram(mesh_device, MakeScratchCbProgram(mesh_device, run, nodes));

    DatacopyResult result;
    const std::uint32_t last_round = run.num_iters / kNumTiles - 1;
    auto in = inputs.begin();
    for (const auto& node : nodes) {
        SCOPED_TRACE("node=" + node.str());
        for (auto address : guard_addresses) {
            std::vector<std::uint32_t> actual;
            slow_dispatch::ReadFromL1(mesh_device, node, address, kGuardBytes, actual);
            EXPECT_EQ(actual, guard) << "L1 guard overwritten at " << address;
        }
        std::vector<std::uint32_t> report;
        slow_dispatch::ReadFromL1(mesh_device, node, kReportAddr, kReportWords * sizeof(std::uint32_t), report);
        result.scratch_errors += report.at(0);
        result.real_cb_errors += report.at(1);

        for (std::uint32_t t = 0; t < kNumTiles; ++t) {
            (*in)[t * kTileWords] = SequenceTag(last_round * kNumTiles + t);
        }
        std::vector<std::uint32_t> out;
        slow_dispatch::ReadFromL1(mesh_device, node, kOutAddr, kNumTiles * kTileBytes, out);
        for (std::uint32_t t = 0; t < kNumTiles; ++t) {
            if (!std::equal(
                    out.begin() + t * kTileWords, out.begin() + (t + 1) * kTileWords, in->begin() + t * kTileWords)) {
                ++result.mismatched_tiles;
            }
        }
        ++in;
    }
    return result;
}

void ExpectDatacopyCorrect(const DatacopyResult& r) {
    EXPECT_EQ(r.scratch_errors, 0u) << "a transfer was stale, reordered, overwritten or corrupt";
    EXPECT_EQ(r.real_cb_errors, 0u) << "a real CB transfer was corrupt";
    EXPECT_EQ(r.mismatched_tiles, 0u) << "output != input";
}

// PATTERN B: returns {pages produced, in-flight high-water mark}.
std::vector<std::uint32_t> RunScratchCbBoundedProducer(distributed::MeshDevice& mesh_device, const ScratchCbRun& run) {
    std::vector<std::uint32_t> zero_report(kReportWords, 0u);
    slow_dispatch::WriteToL1(mesh_device, kNode.start_coord, kReportAddr, zero_report);
    LaunchProgram(mesh_device, MakeScratchCbProgram(mesh_device, run, kNode));
    std::vector<std::uint32_t> r;
    slow_dispatch::ReadFromL1(mesh_device, kNode.start_coord, kReportAddr, kReportWords * sizeof(std::uint32_t), r);
    return r;
}

// PATTERN A: DM -> compute -> DM through ping/pong rings, both channels live at once. Relaunching the same
// program checks that counters start each kernel at 0.
TEST_F(ProgramSpecHWTest, ScratchCbDatacopy) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only";
    }
    for (std::uint32_t capacity : {1u, 2u}) {
        SCOPED_TRACE("capacity=" + std::to_string(capacity));
        ExpectDatacopyCorrect(RunScratchCbDatacopy(*mesh_device, {.capacity = capacity}, 0x00u));
        ExpectDatacopyCorrect(RunScratchCbDatacopy(*mesh_device, {.capacity = capacity}, 0x15u));
    }
}

// More than 65536 handoffs per channel: the 16-bit counters wrap mid-kernel.
TEST_F(ProgramSpecHWTest, ScratchCbDatacopyCounterWrap) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only";
    }
    ExpectDatacopyCorrect(RunScratchCbDatacopy(*mesh_device, {.num_iters = 1025 * kNumTiles}, 0x25u));
}

// Negative control: without reserve/wait, compute and the consumer read slots before they are written, so the
// output must NOT equal the input. If this passes the positive tests above are not detectors.
TEST_F(ProgramSpecHWTest, ScratchCbDatacopyNoSyncControl) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only";
    }
    const auto r = RunScratchCbDatacopy(*mesh_device, {.nosync = 1}, 0x00u);
    GTEST_LOG_(INFO) << "no-sync control: mismatched_tiles=" << r.mismatched_tiles << " / " << kNumTiles;
    EXPECT_GT(r.mismatched_tiles, 0u) << "unsynchronized datacopy came out correct -- the positive tests cannot "
                                         "distinguish a working scratch CB from no synchronization";
}

// PATTERN B: the DM producer gates every push with reserve_back against a capacity of kBoundedDepth; the slow
// UNPACK consumer keeps it full. Pages in flight must never exceed the capacity.
TEST_F(ProgramSpecHWTest, ScratchCbBoundedProducer) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only";
    }
    constexpr std::uint32_t num_iters = 1024;
    for (std::uint32_t batch : {1u, 2u}) {
        SCOPED_TRACE("batch=" + std::to_string(batch));
        const auto r = RunScratchCbBoundedProducer(
            *mesh_device, {.pattern = kPatternB, .capacity = kBoundedDepth, .num_iters = num_iters, .batch = batch});
        GTEST_LOG_(INFO) << "bounded producer, batch " << batch << ": produced=" << r[0] << " high_water=" << r[1];
        EXPECT_EQ(r[0], num_iters);
        EXPECT_LE(r[1], kBoundedDepth) << "producer ran past the capacity -- reserve_back did not hold it";
        EXPECT_GE(r[1], 2u) << "consumer was never behind -- the test did not exercise back-pressure";
    }
}

// Negative control: same run without reserve_back. The producer must run past the capacity.
TEST_F(ProgramSpecHWTest, ScratchCbBoundedProducerNoWaitControl) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only";
    }
    constexpr std::uint32_t num_iters = 64;
    const auto r = RunScratchCbBoundedProducer(
        *mesh_device, {.pattern = kPatternB, .capacity = kBoundedDepth, .num_iters = num_iters, .nosync = 1});
    GTEST_LOG_(INFO) << "bounded producer no-wait control: produced=" << r[0] << " high_water=" << r[1];
    EXPECT_GT(r[1], kBoundedDepth) << "ungated producer never exceeded the capacity -- the positive test cannot "
                                      "tell reserve_back from nothing";
}

// PATTERN C: one DM kernel produces channel 0 and consumes channel 1. Every round sends compute's last result
// back through ring A, so the two sides strictly alternate and one stale or early read is carried into every
// later round. More than 65536 rounds also wraps both counters.
TEST_F(ProgramSpecHWTest, ScratchCbPingPong) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only";
    }
    constexpr std::uint32_t num_iters = 1025 * kNumTiles;
    auto in = MakeInputTiles(0x2Bu);
    std::vector<std::uint32_t> poison((kReportAddr + kReportWords * 4 - kRingAAddr) / 4, 0xDEADBEEFu);
    slow_dispatch::WriteToL1(*mesh_device, kNode.start_coord, kInAddr, in);
    slow_dispatch::WriteToL1(*mesh_device, kNode.start_coord, kRingAAddr, poison);  // covers both rings, out and report
    LaunchProgram(
        *mesh_device,
        MakeScratchCbProgram(*mesh_device, {.pattern = kPatternC, .capacity = 1, .num_iters = num_iters}, kNode));

    std::vector<std::uint32_t> report;
    slow_dispatch::ReadFromL1(
        *mesh_device, kNode.start_coord, kReportAddr, kReportWords * sizeof(std::uint32_t), report);
    EXPECT_EQ(report.at(0), 0u) << "a round came back stale, early or corrupt";

    std::vector<std::uint32_t> expected(in.begin(), in.begin() + kTileWords);
    expected[0] = SequenceTag(num_iters - 1);
    std::vector<std::uint32_t> out;
    slow_dispatch::ReadFromL1(*mesh_device, kNode.start_coord, kOutAddr, kTileBytes, out);
    EXPECT_EQ(out, expected) << "final ping-pong tile != input";
}

// CB IDs 62 and 63 belong to scratch: the CB API rejects them, real CBs fill 0-61, and firmware setting up
// those CBs every launch must not disturb scratch's counters (nor scratch traffic any real CB).
TEST_F(ProgramSpecHWTest, ScratchCbAlongsideRealCbs) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only";
    }
    EXPECT_EQ(hal::get_num_dataflow_buffers(), kRealCbs);
    EXPECT_NO_THROW((CircularBufferConfig(kTileBytes, {{kRealCbs - 1, tt::DataFormat::Float16_b}})));
    for (std::uint32_t id : {62u, 63u}) {
        EXPECT_THROW((CircularBufferConfig(kTileBytes, {{id, tt::DataFormat::Float16_b}})), std::runtime_error);
    }

    const ScratchCbRun run{.num_iters = 1025 * kNumTiles, .real_cbs = 1};
    Program program = MakeScratchCbProgram(*mesh_device, run, kNode);
    std::vector<std::uint32_t> ids;
    for (const auto& dfb : program.impl().dataflow_buffers()) {
        ids.push_back(dfb->device_slot);
    }
    std::sort(ids.begin(), ids.end());
    std::vector<std::uint32_t> expected(kRealCbs);
    std::iota(expected.begin(), expected.end(), 0u);
    EXPECT_EQ(ids, expected);
    // The real CBs are allocated from the bottom of L1 and must stay clear of the fixed test regions.
    EXPECT_LT(
        mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1) +
            kRealCbs * kRealCbEntries * kRealCbEntryBytes,
        kInAddr);

    ExpectDatacopyCorrect(RunScratchCbDatacopy(*mesh_device, run, 0x31u));
}

// Every worker node runs its own producer/compute/consumer at once, each on its own data.
TEST_F(ProgramSpecHWTest, ScratchCbAllNodes) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only";
    }
    const auto grid = mesh_device->compute_with_storage_grid_size();
    const NodeRange all_nodes{{0, 0}, {grid.x - 1, grid.y - 1}};
    ExpectDatacopyCorrect(RunScratchCbDatacopy(*mesh_device, {.num_iters = 4 * kNumTiles}, 0x19u, all_nodes));
}

}  // namespace scratch_cb_test
}  // namespace tt::tt_metal::experimental
