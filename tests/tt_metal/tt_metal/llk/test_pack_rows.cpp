// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
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
#include <tt-metalium/data_types.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include "test_golden_impls.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include <umd/device/types/arch.hpp>

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
    uint32_t num_rows;
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
    auto* device = mesh_device->get_devices()[0];

    CoreCoord core = {0, 0};

    // Input: 1 tile (32x32 = 1024 bfloat16 = 2048 bytes)
    uint32_t input_single_tile_size = 2 * 1024;
    // Output: num_rows * 16 datums * 2 bytes each
    uint32_t output_size = test_config.num_rows * 16 * 2;

    tt_metal::InterleavedBufferConfig input_dram_config{
        .device = device,
        .size = input_single_tile_size,
        .page_size = input_single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig output_dram_config{
        .device = device, .size = output_size, .page_size = output_size, .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt_metal::Buffer> src0_dram_buffer = CreateBuffer(input_dram_config);
    uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();

    std::shared_ptr<tt_metal::Buffer> dst_dram_buffer = CreateBuffer(output_dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(input_single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, input_single_tile_size);
    tt_metal::CreateCircularBuffer(program_, core, cb_src0_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(output_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, output_size);
    tt_metal::CreateCircularBuffer(program_, core, cb_output_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_n.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program_,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {
        uint(test_config.num_rows),
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
    std::vector<uint32_t> src0_vec;
    for (uint32_t i = 0; i < input_single_tile_size / sizeof(uint32_t); i++) {
        bfloat16 val1(static_cast<float>(i * 2));
        bfloat16 val2(static_cast<float>(i * 2 + 1));
        src0_vec.push_back(pack_two_bfloat16_into_uint32({val1, val2}));
    }
    tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_vec);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {dram_buffer_src0_addr,
         (uint32_t)0,  // dram bank id
         (uint32_t)1,  // num_tiles
         src0_cb_index,
         (uint32_t)1,  // block_size
         false});

    tt_metal::SetRuntimeArgs(program_, unary_writer_kernel, core, {dram_buffer_dst_addr, (uint32_t)0, (uint32_t)1});

    distributed::EnqueueMeshWorkload(cq, workload, false);

    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    ::unit_tests::compute::PackRowsConfig golden_config = {
        .num_rows = static_cast<int>(test_config.num_rows),
    };
    vector<uint32_t> golden = ::unit_tests::compute::gold_standard_pack_rows(src0_vec, golden_config);

    bool pass = true;
    pass &= (golden.size() == result_vec.size());
    pass &= (golden == result_vec);

    if (not pass) {
        std::cout << "GOLDEN " << std::endl;
        print_vector(unpack_vector<bfloat16, uint32_t>(golden));
        std::cout << "RESULTS " << std::endl;
        print_vector(unpack_vector<bfloat16, uint32_t>(result_vec));
    }

    log_info(
        tt::LogTest,
        "Done running test with: num_rows = {}, DstSyncFull = {}, pass = {}",
        test_config.num_rows,
        test_config.dst_full_sync_en,
        pass);
    ASSERT_TRUE(pass);
}

}  // namespace unit_tests::compute::pack_rows

/**************************************
Following tests are for pack_rows
***************************************/

TEST_F(MeshDeviceFixture, TensixComputePackRows) {
    vector<uint32_t> num_rows_values = {1, 8, 16, 32, 48, 64};
    for (auto num_rows : num_rows_values) {
        for (bool dst_full_sync_en : {true, false}) {
            unit_tests::compute::pack_rows::TestConfig test_config = {
                .dst_full_sync_en = dst_full_sync_en,
                .num_rows = num_rows,
            };
            unit_tests::compute::pack_rows::run_single_core_pack_rows_program(this->devices_.at(0), test_config);
        }
    }
}

}  // namespace tt::tt_metal
