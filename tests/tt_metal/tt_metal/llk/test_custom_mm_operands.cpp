// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/debug/watcher_server.hpp"
#include "llk_device_fixture.hpp"
#include "mesh_dispatch_fixture.hpp"

namespace tt::tt_metal {
namespace {

void run_custom_mm_operands(
    distributed::MeshDevice& mesh, std::uint32_t rows, bool mixed, std::uint32_t mode, bool full_sync) {
    constexpr std::uint32_t blocks = 3;
    const std::uint32_t input_block_tiles = mode == 2 ? 5 : 3;
    const std::uint32_t input_tiles = blocks * input_block_tiles;
    const Tile tiny({rows, 32});
    const Tile full;
    const auto activation_format = tt::DataFormat::Float16_b;
    const auto weight_format = mode == 1 ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
    const auto a_bytes = tiny.get_tile_size(activation_format);
    const auto b_bytes = full.get_tile_size(weight_format);
    const CoreCoord core{0, 0};
    Program program = CreateProgram();
    auto buffer = [&](std::uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            {.page_size = bytes, .buffer_type = BufferType::DRAM},
            &mesh);
    };
    auto a_buffer = buffer(input_tiles * a_bytes);
    auto b_buffer = buffer(input_tiles * b_bytes);
    auto result_buffer = buffer(blocks * a_bytes);
    auto make_cb = [&](tt::CBIndex index, tt::DataFormat format, const Tile& tile, std::uint32_t depth) {
        const auto bytes = tile.get_tile_size(format);
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(depth * bytes, {{index, format}})
                .set_page_size(index, bytes)
                .set_tile_dims(index, tile));
    };
    // Two blocks of capacity force successive executions to use different L1
    // bases, then wrap the CB. Static input metadata stays unchanged.
    make_cb(tt::CBIndex::c_0, activation_format, tiny, 2 * input_block_tiles);
    make_cb(tt::CBIndex::c_1, weight_format, full, 2 * input_block_tiles);
    make_cb(tt::CBIndex::c_16, activation_format, tiny, 2);
    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/custom_mm_operands.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = full_sync,
            .math_approx_mode = false,
            .compile_args = {blocks, rows, static_cast<std::uint32_t>(mixed), mode}});

    std::vector<bfloat16> activations(input_tiles * rows * 32);
    std::vector<bfloat16> weights(input_tiles * 1024, bfloat16(0.0f));
    std::vector<bfloat16> golden(blocks * rows * 32);
    for (std::uint32_t block = 0; block < blocks; ++block) {
        for (std::uint32_t tile = 0; tile < input_block_tiles; ++tile) {
            for (std::uint32_t row = 0; row < rows; ++row) {
                for (std::uint32_t col = 0; col < 32; ++col) {
                    const auto index =
                        (block * input_block_tiles + tile) * rows * 32 + (col / 16) * rows * 16 + row * 16 + col % 16;
                    const float value =
                        tile == 1 || tile == 2 ? static_cast<float>(block + row + col % 7 + tile) / 8.0f : 64.0f;
                    activations[index] = bfloat16(value);
                }
            }
            for (std::uint32_t d = 0; d < 32; ++d) {
                const auto index = (block * input_block_tiles + tile) * 1024 + (d / 16) * 3 * 256 + (d % 16) * 17;
                // The two-column producer uses [I, 2I; 2I, I]. The reuse
                // consumer reads weight tiles 2 and 4, exercising a K stride of 2.
                float weight = 2.0f;
                if (tile == 0) {
                    weight = 32.0f;
                } else if (tile == 1 || tile == 4) {
                    weight = 1.0f;
                }
                weights[index] = bfloat16(weight);
            }
        }
        for (std::uint32_t i = 0; i < rows * 32; ++i) {
            const auto a0 = static_cast<float>(activations[(block * input_block_tiles + 1) * rows * 32 + i]);
            const auto a1 = static_cast<float>(activations[(block * input_block_tiles + 2) * rows * 32 + i]);
            // Reuse consumes both populated DEST tiles:
            // 2 * (a0 + 2*a1) + (2*a0 + a1) = 4*a0 + 5*a1.
            golden[block * rows * 32 + i] = bfloat16(mode == 2 ? 4.0f * a0 + 5.0f * a1 : a0 + 2.0f * a1);
        }
    }
    auto& cq = mesh.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, a_buffer, pack_bfloat16_vec_into_uint32_vec(activations), true);
    auto packed_weights =
        mode == 1 ? pack_as_bfp8_tiles<bfloat16>(weights, false, false) : pack_bfloat16_vec_into_uint32_vec(weights);
    distributed::EnqueueWriteMeshBuffer(cq, b_buffer, packed_weights, true);
    SetRuntimeArgs(program, reader, core, {a_buffer->address(), 0, b_buffer->address(), 0, input_tiles});
    SetRuntimeArgs(program, writer, core, {result_buffer->address(), 0, blocks});
    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    std::vector<std::uint32_t> result;
    distributed::EnqueueReadMeshBuffer(cq, result, result_buffer, true);
    EXPECT_EQ(result, pack_bfloat16_vec_into_uint32_vec(golden));
}

std::string wait_for_watcher_exception() {
    constexpr auto timeout = std::chrono::milliseconds(5000);
    constexpr auto poll_interval = std::chrono::milliseconds(100);
    const auto start = std::chrono::steady_clock::now();
    std::string exception;
    while (std::chrono::steady_clock::now() - start < timeout) {
        exception = MetalContext::instance().watcher_server()->exception_message();
        if (!exception.empty()) {
            break;
        }
        std::this_thread::sleep_for(poll_interval);
    }
    return exception;
}

// A [16, 32] activation tile is a legal CB tile but outside the custom_mm family's
// [1|2|4|8, 32] in0 contract. The typed LLKOperand overloads reject it at compile
// time; the CB-id overloads read the face height and face grid at run time, so their
// only guard is the LLK asserts this exercises. Without them the init completes with no
// diagnostic and the product is silently wrong: 16x16 faces run the 1..4-row program on
// a 16-row operand and drop rows 8..15 of each face; 8x16 faces in a 2x2 grid run the
// 8-row program and read faces 2/3 as the next K tile. (The product is not run here:
// a product kernel whose init asserts leaves the pack thread in a tile-register wait,
// which blocks the core's re-initialization.)
constexpr std::uint32_t kGuardRows = 16;

// mode: 0 custom_mm, 1 compressed_custom_mm. thread: 0 unpack init (TRISC0), 1 math init (TRISC1).
// in0_tile: the [kGuardRows, 32] in0 CB tile, 16x16 faces or an 8x16 face grid.
void run_custom_mm_rows_guard(
    distributed::MeshDevice& mesh, std::uint32_t mode, std::uint32_t thread, const Tile& in0_tile) {
    // A legal CB-id height must not trip the guard.
    run_custom_mm_operands(mesh, 8 /*rows*/, true /*mixed*/, mode, false /*full_sync*/);
    EXPECT_TRUE(MetalContext::instance().watcher_server()->exception_message().empty())
        << "rows=8 tripped the watcher: " << MetalContext::instance().watcher_server()->exception_message();

    // Init-only kernel, one thread, on the [kGuardRows, 32] in0 CB: that init must assert.
    const Tile& tiny = in0_tile;
    const Tile full;
    const auto weight_format = mode == 1 ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
    const CoreCoord core{0, 0};
    Program program = CreateProgram();
    auto make_cb = [&](tt::CBIndex index, tt::DataFormat format, const Tile& tile) {
        const auto bytes = tile.get_tile_size(format);
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(bytes, {{index, format}}).set_page_size(index, bytes).set_tile_dims(index, tile));
    };
    make_cb(tt::CBIndex::c_0, tt::DataFormat::Float16_b, tiny);
    make_cb(tt::CBIndex::c_1, weight_format, full);
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/custom_mm_operands_guard.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {mode, thread}});
    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    auto& cq = mesh.mesh_command_queue();
    // The wait aborts once the watcher reports the assert, but the program may complete first (no
    // guard), so the throw is tolerated here and the watcher state is what is checked.
    try {
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("Aborting wait due to watcher error"), std::string::npos) << e.what();
    }
    const std::string exception = wait_for_watcher_exception();
    ASSERT_FALSE(exception.empty()) << "Timeout waiting for the watcher to report the in0 tile height assert";
    EXPECT_NE(exception.find("tripped an assert"), std::string::npos) << exception;
    EXPECT_NE(exception.find("custom_mm_operands_guard"), std::string::npos) << exception;
    // The watcher records the first assert per core; only the chosen thread ran an init.
    EXPECT_NE(exception.find(thread == 0 ? "TRISC0" : "TRISC1"), std::string::npos) << exception;
}

}  // namespace

// Opens one Blackhole device per test with the watcher and LLK asserts enabled, so a device-side
// LLK_ASSERT surfaces as a watcher exception instead of a silent hang. Mirrors MeshWatcherFixture
// (tests/tt_metal/tt_metal/debug_tools/debug_tools_fixture.hpp); the options must be set before the
// device opens, and a tripped assert parks a TRISC, so the device is reopened for every test.
class LLKBlackholeAssertFixture : public MeshDispatchFixture {
public:
    static constexpr int kWatcherIntervalMs = 250;

    // No suite-shared device: the options below change the firmware build, so the device must open
    // after they are set, once per test.
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

protected:
    void SetUp() override {
        // Capture before any early exit: TearDown still runs after a skip or a failed assertion
        // here, and must restore the real previous values, not the zero-initialised members.
        auto& rtoptions = MetalContext::instance().rtoptions();
        watcher_previous_enabled_ = rtoptions.get_watcher_enabled();
        watcher_previous_interval_ = rtoptions.get_watcher_interval();
        test_mode_previous_ = rtoptions.get_test_mode_enabled();
        llk_asserts_previous_ = rtoptions.get_llk_asserts();

        if (detail::detect_arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "This test can only be run on Blackhole cards";
        }
        rtoptions.set_watcher_enabled(true);
        rtoptions.set_watcher_interval(kWatcherIntervalMs);
        // In test mode a watcher error stops the server and fails the wait instead of ending the process.
        rtoptions.set_test_mode_enabled(true);
        rtoptions.set_llk_asserts(true);

        const ChipId mmio_device_id = *MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        auto id_to_device = distributed::MeshDevice::create_unit_meshes(
            {mmio_device_id},
            l1_small_size_,
            trace_region_size_,
            1,
            MetalContext::instance().resolve_dispatch_core_config());
        devices_ = {id_to_device.at(mmio_device_id)};
        arch_ = tt::ARCH::BLACKHOLE;
        MetalContext::instance().watcher_server()->clear_log();
    }

    void TearDown() override {
        for (auto& device : devices_) {
            device->close();
        }
        devices_.clear();
        auto& rtoptions = MetalContext::instance().rtoptions();
        rtoptions.set_llk_asserts(llk_asserts_previous_);
        rtoptions.set_test_mode_enabled(test_mode_previous_);
        rtoptions.set_watcher_interval(watcher_previous_interval_);
        rtoptions.set_watcher_enabled(watcher_previous_enabled_);
    }

private:
    bool watcher_previous_enabled_{};
    int watcher_previous_interval_{};
    bool test_mode_previous_{};
    bool llk_asserts_previous_{};
};

TEST_F(LLKBlackholeSingleCardFixture, CustomMatmulOperandOverloads) {
    for (const auto rows : {1u, 2u, 4u, 8u}) {
        for (const bool mixed : {false, true}) {
            for (const auto mode : {0u, 1u, 2u}) {
                for (const bool full_sync : {false, true}) {
                    SCOPED_TRACE(
                        ::testing::Message()
                        << "rows=" << rows << ", mixed=" << mixed << ", mode=" << mode << ", full_sync=" << full_sync);
                    run_custom_mm_operands(*devices_.at(0), rows, mixed, mode, full_sync);
                }
            }
        }
    }
}

TEST_F(LLKBlackholeAssertFixture, CustomMatmulUnpackInitRejectsSixteenRowActivation) {
    run_custom_mm_rows_guard(*devices_.at(0), 0 /*mode*/, 0 /*thread*/, Tile({kGuardRows, 32}));
}

TEST_F(LLKBlackholeAssertFixture, CustomMatmulMathInitRejectsSixteenRowActivation) {
    run_custom_mm_rows_guard(*devices_.at(0), 0 /*mode*/, 1 /*thread*/, Tile({kGuardRows, 32}));
}

TEST_F(LLKBlackholeAssertFixture, CompressedCustomMatmulUnpackInitRejectsSixteenRowActivation) {
    run_custom_mm_rows_guard(*devices_.at(0), 1 /*mode*/, 0 /*thread*/, Tile({kGuardRows, 32}));
}

TEST_F(LLKBlackholeAssertFixture, CompressedCustomMatmulMathInitRejectsSixteenRowActivation) {
    run_custom_mm_rows_guard(*devices_.at(0), 1 /*mode*/, 1 /*thread*/, Tile({kGuardRows, 32}));
}

// Same height from 8x16 faces in a 2x2 grid: the face height passes, the face grid must not.
TEST_F(LLKBlackholeAssertFixture, CustomMatmulUnpackInitRejectsFourFaceActivation) {
    run_custom_mm_rows_guard(*devices_.at(0), 0 /*mode*/, 0 /*thread*/, Tile::from_face_grid(2, 2, {8, 16}));
}

TEST_F(LLKBlackholeAssertFixture, CompressedCustomMatmulUnpackInitRejectsFourFaceActivation) {
    run_custom_mm_rows_guard(*devices_.at(0), 1 /*mode*/, 0 /*thread*/, Tile::from_face_grid(2, 2, {8, 16}));
}

}  // namespace tt::tt_metal
