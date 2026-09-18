// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "llk_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include "test_golden_impls.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include <umd/device/types/arch.hpp>
#include "single_core_compute_runners.hpp"

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::compute::pack_rows {

struct TestConfig {
    // Whether or not to sync full/half DST between MATH and PACK:
    bool dst_full_sync_en = false;
    // Number of rows to pack from DEST (1-64):
    std::uint32_t num_rows;
};

void run_single_core_pack_rows_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const TestConfig& test_config) {
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = tt::tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    CoreCoord core = {0, 0};

    // Input: 1 tile (32x32 = 1024 bfloat16 = 2048 bytes)
    std::uint32_t input_single_tile_size = 2 * 1024;
    // Output: num_rows * 16 datums * 2 bytes each
    std::uint32_t output_size = test_config.num_rows * 16 * 2;

    auto src0_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = input_single_tile_size},
        {.page_size = input_single_tile_size, .buffer_type = tt_metal::BufferType::DRAM},
        mesh_device.get());
    std::uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();

    auto dst_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = output_size},
        {.page_size = output_size, .buffer_type = tt_metal::BufferType::DRAM},
        mesh_device.get());
    std::uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    std::uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(input_single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, input_single_tile_size);
    tt_metal::CreateCircularBuffer(program_, core, cb_src0_config);

    std::uint32_t output_cb_index = tt::CBIndex::c_16;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(output_size, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, output_size);
    tt_metal::CreateCircularBuffer(program_, core, cb_output_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_n.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<std::uint32_t> compute_kernel_args = {
        std::uint32_t(test_config.num_rows),
    };

    tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/pack_rows.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = test_config.dst_full_sync_en,
            .compile_args = compute_kernel_args});

    // Generate tilized input data with sequential values
    std::vector<std::uint32_t> src0_vec;
    for (std::uint32_t i = 0; i < input_single_tile_size / sizeof(std::uint32_t); i++) {
        bfloat16 val1(static_cast<float>(i * 2));
        bfloat16 val2(static_cast<float>((i * 2) + 1));
        src0_vec.push_back(pack_two_bfloat16_into_uint32({val1, val2}));
    }
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, /*blocking=*/true);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {dram_buffer_src0_addr,
         (std::uint32_t)0,  // dram bank id
         (std::uint32_t)1,  // num_tiles
         src0_cb_index,
         (std::uint32_t)1,  // block_size
         false});

    tt_metal::SetRuntimeArgs(
        program_, unary_writer_kernel, core, {dram_buffer_dst_addr, (std::uint32_t)0, (std::uint32_t)1});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<std::uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);

    ::unit_tests::compute::PackRowsConfig golden_config = {
        .num_rows = static_cast<int>(test_config.num_rows),
    };
    vector<std::uint32_t> golden = ::unit_tests::compute::gold_standard_pack_rows(src0_vec, golden_config);

    EXPECT_EQ(golden.size(), result_vec.size());
    EXPECT_EQ(golden, result_vec);

    log_info(
        tt::LogTest,
        "Done running test with: num_rows = {}, DstSyncFull = {}",
        test_config.num_rows,
        test_config.dst_full_sync_en);
}

}  // namespace unit_tests::compute::pack_rows

/**************************************
Following tests are for pack_rows
***************************************/

// Parameterized test fixture that combines LLKMeshDeviceFixture with test parameters
class TensixComputePackRowsTest : public LLKMeshDeviceFixture,
                                  public testing::WithParamInterface<unit_tests::compute::pack_rows::TestConfig> {};

// The actual parameterized test
TEST_P(TensixComputePackRowsTest, PackRows) {
    auto test_config = GetParam();
    unit_tests::compute::pack_rows::run_single_core_pack_rows_program(this->devices_.at(0), test_config);
}

// Define all test parameter combinations
INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    TensixComputePackRowsTest,
    testing::Values(
        // num_rows = 1
        unit_tests::compute::pack_rows::TestConfig{.dst_full_sync_en = false, .num_rows = 1},
        unit_tests::compute::pack_rows::TestConfig{.dst_full_sync_en = true, .num_rows = 1},
        // num_rows = 8
        unit_tests::compute::pack_rows::TestConfig{.dst_full_sync_en = false, .num_rows = 8},
        unit_tests::compute::pack_rows::TestConfig{.dst_full_sync_en = true, .num_rows = 8},
        // num_rows = 16
        unit_tests::compute::pack_rows::TestConfig{.dst_full_sync_en = false, .num_rows = 16},
        unit_tests::compute::pack_rows::TestConfig{.dst_full_sync_en = true, .num_rows = 16},
        // num_rows = 32
        unit_tests::compute::pack_rows::TestConfig{.dst_full_sync_en = false, .num_rows = 32},
        unit_tests::compute::pack_rows::TestConfig{.dst_full_sync_en = true, .num_rows = 32},
        // num_rows = 48
        unit_tests::compute::pack_rows::TestConfig{.dst_full_sync_en = false, .num_rows = 48},
        unit_tests::compute::pack_rows::TestConfig{.dst_full_sync_en = true, .num_rows = 48},
        // num_rows = 64
        unit_tests::compute::pack_rows::TestConfig{.dst_full_sync_en = false, .num_rows = 64},
        unit_tests::compute::pack_rows::TestConfig{.dst_full_sync_en = true, .num_rows = 64}),
    // Custom name generator for better test output
    [](const testing::TestParamInfo<TensixComputePackRowsTest::ParamType>& info) {
        return fmt::format("NumRows_{}_DstSync_{}", info.param.num_rows, info.param.dst_full_sync_en ? "On" : "Off");
    });

// ============================================================================
// Id-free (2.0) pack_rows: copies one Float16_b tile into DST then row-packs the full tile (num_rows = 64)
// row-major to c_16. Validated against the gold_standard_pack_rows host golden (exact). Runs on Blackhole
// (BH-only API). num_rows == 64 writes the whole output tile, so no uninitialized bytes.
// ============================================================================
TEST_F(LLKBlackholeSingleCardFixture, TensixPackRowsIdFreeGolden) {
    constexpr std::uint32_t num_tiles = 1;
    constexpr int num_rows = 64;  // full tile (matches pack_rows_2_0.cpp)
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result = unit_tests::llk::single_core::run_unary(
        *this->devices_.at(0),
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b,
        src_vec,
        num_tiles,
        /*fp32_dest_acc_en=*/false,
        "tests/tt_metal/tt_metal/test_kernels/compute/pack_rows_2_0.cpp");

    vector<std::uint32_t> golden = ::unit_tests::compute::gold_standard_pack_rows(
        src_vec, ::unit_tests::compute::PackRowsConfig{.num_rows = num_rows});
    EXPECT_EQ(golden.size(), result.size());
    EXPECT_EQ(golden, result);
}

}  // namespace tt::tt_metal
