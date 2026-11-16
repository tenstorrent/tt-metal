// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar iDMA tilize, fp32: one DM thread rewrites a row-major tensor in L1 into tilized face
// order with one iDMA copy per face row (kernels/idma_tilize_dm.cpp). The output is checked datum
// for datum, and the NEO0 wall-clock cycles per tile are logged.

#include <cstdint>
#include <string>
#include <vector>
#include <tt-logger/tt-logger.hpp>
#include "device_fixture.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

namespace tt::tt_metal {

namespace unit_tests::dm::quasar_idma_tilize {

constexpr std::uint32_t TILE_DIM = 32;
constexpr std::uint32_t FACE_DIM = 16;
constexpr std::uint32_t DATUM_BYTES = 4;
constexpr std::uint32_t TILE_BYTES = TILE_DIM * TILE_DIM * DATUM_BYTES;
constexpr CoreCoord CORE = {0, 0};
// wall, dm_setup, dm_issue, dm_drain (layout in kernels/idma_tilize_dm.cpp).
constexpr std::uint32_t RESULT_WORDS = 4;

// Each datum is its row-major index, tagged so zero is never a valid value.
std::uint32_t datum_at(std::uint32_t r, std::uint32_t c, std::uint32_t width) { return 0x5A000000u | (r * width + c); }

std::shared_ptr<distributed::MeshBuffer> make_l1_buffer(distributed::MeshDevice& mesh_device, std::uint32_t bytes) {
    distributed::DeviceLocalBufferConfig local{.page_size = bytes, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig cfg{.size = bytes};
    return distributed::MeshBuffer::create(cfg, local, &mesh_device);
}

}  // namespace unit_tests::dm::quasar_idma_tilize

class QuasarIdmaTilize : public QuasarMeshDeviceSingleCardFixture {};

TEST_F(QuasarIdmaTilize, Fp32) {
    using namespace unit_tests::dm::quasar_idma_tilize;
    if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::QUASAR ||
        std::getenv("TT_METAL_SIMULATOR") == nullptr) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    constexpr std::uint32_t RT_DIM = 4;
    constexpr std::uint32_t CT_DIM = 8;
    constexpr std::uint32_t NUM_TILES = RT_DIM * CT_DIM;
    constexpr std::uint32_t WIDTH = CT_DIM * TILE_DIM;
    constexpr std::uint32_t TENSOR_WORDS = NUM_TILES * TILE_DIM * TILE_DIM;
    auto& mesh_device = this->device();

    std::vector<std::uint32_t> row_major(TENSOR_WORDS);
    for (std::uint32_t r = 0; r < RT_DIM * TILE_DIM; r++) {
        for (std::uint32_t c = 0; c < WIDTH; c++) {
            row_major[r * WIDTH + c] = datum_at(r, c, WIDTH);
        }
    }
    std::vector<std::uint32_t> tilized;
    tilized.reserve(TENSOR_WORDS);
    for (std::uint32_t tr = 0; tr < RT_DIM; tr++) {
        for (std::uint32_t tc = 0; tc < CT_DIM; tc++) {
            for (std::uint32_t f = 0; f < 4; f++) {
                for (std::uint32_t fr = 0; fr < FACE_DIM; fr++) {
                    for (std::uint32_t fc = 0; fc < FACE_DIM; fc++) {
                        tilized.push_back(datum_at(
                            tr * TILE_DIM + (f / 2) * FACE_DIM + fr, tc * TILE_DIM + (f % 2) * FACE_DIM + fc, WIDTH));
                    }
                }
            }
        }
    }

    const auto src = make_l1_buffer(mesh_device, NUM_TILES * TILE_BYTES);
    const auto dst = make_l1_buffer(mesh_device, NUM_TILES * TILE_BYTES);
    const auto result = make_l1_buffer(mesh_device, RESULT_WORDS * sizeof(std::uint32_t));
    const auto addr = [](const auto& b) { return static_cast<std::uint32_t>(b->address()); };
    ASSERT_TRUE(slow_dispatch::WriteToL1(mesh_device, CORE, addr(src), row_major));
    std::vector<std::uint32_t> dst_init(TENSOR_WORDS, 0);
    std::vector<std::uint32_t> result_init(RESULT_WORDS, 0);
    ASSERT_TRUE(slow_dispatch::WriteToL1(mesh_device, CORE, addr(dst), dst_init));
    ASSERT_TRUE(slow_dispatch::WriteToL1(mesh_device, CORE, addr(result), result_init));

    const experimental::NodeCoord node{0, 0};
    const experimental::KernelSpecName KERNEL{"idma_tilize"};
    experimental::ProgramSpec spec{
        .name = "idma_tilize",
        .kernels = {{
            .unique_id = KERNEL,
            .source = "tests/tt_metal/tt_metal/data_movement/quasar_idma_tilize/kernels/idma_tilize_dm.cpp",
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "dst_addr", "rt_dim", "ct_dim", "result_addr"}},
            .hw_config = experimental::DataMovementHardwareConfig{},
        }},
        .work_units = {{.name = "main", .kernels = {KERNEL}, .target_nodes = node}},
    };
    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {{
        .kernel = KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node,
            {{"src_addr", addr(src)},
             {"dst_addr", addr(dst)},
             {"rt_dim", RT_DIM},
             {"ct_dim", CT_DIM},
             {"result_addr", addr(result)}}),
    }};
    experimental::SetProgramRunArgs(program, params);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);

    std::vector<std::uint32_t> out;
    ASSERT_TRUE(slow_dispatch::ReadFromL1(mesh_device, CORE, addr(dst), NUM_TILES * TILE_BYTES, out));
    out.resize(TENSOR_WORDS);
    EXPECT_TRUE(out == tilized) << "tilized output mismatch";

    std::vector<std::uint32_t> t;
    ASSERT_TRUE(slow_dispatch::ReadFromL1(mesh_device, CORE, addr(result), RESULT_WORDS * sizeof(std::uint32_t), t));
    ASSERT_GE(t.size(), RESULT_WORDS);
    const std::uint32_t wall = t[0];
    const std::uint32_t dm_setup = t[1];
    const std::uint32_t dm_issue = t[2];
    const std::uint32_t dm_drain = t[3];
    EXPECT_NE(wall, 0u) << "timing record not written";

    constexpr std::uint32_t COPY_BYTES = FACE_DIM * DATUM_BYTES;
    constexpr double NUM_COPIES = NUM_TILES * 4 * FACE_DIM;
    log_info(
        tt::LogTest,
        "iDMA tilize, fp32, 1 DM thread, cmdbuf 0 + addrgen 0, all iDMA VCs\n"
        "  shape                {} x {} tiles ({} tiles), {} copies of {} B (one per face row)\n"
        "  wall clock (NEO0)    total {} cycles | {:.1f} per tile | {:.2f} per copy | {:.1f} B/cycle\n"
        "  DM cycles            setup {} | issue {} | drain {} | total {}\n"
        "  DM issue rate        {:.2f} DM cycles per push + issue",
        RT_DIM,
        CT_DIM,
        NUM_TILES,
        static_cast<std::uint32_t>(NUM_COPIES),
        COPY_BYTES,
        wall,
        wall / static_cast<double>(NUM_TILES),
        wall / NUM_COPIES,
        NUM_TILES * TILE_BYTES / static_cast<double>(wall),
        dm_setup,
        dm_issue,
        dm_drain,
        dm_setup + dm_issue + dm_drain,
        dm_issue / NUM_COPIES);
}

}  // namespace tt::tt_metal
