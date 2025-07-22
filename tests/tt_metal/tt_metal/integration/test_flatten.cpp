// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "dispatch_fixture.hpp"
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal {

using std::vector;
using namespace tt;

namespace test_flatten {

uint32_t prod(vector<uint32_t>& shape) {
    uint32_t shape_prod = 1;

    for (uint32_t shape_i : shape) {
        shape_prod *= shape_i;
    }

    return shape_prod;
}

inline std::vector<uint32_t> gold_standard_flatten(std::vector<uint32_t> src_vec, vector<uint32_t> shape) {
    int numel_in_tensor = prod(shape) / 2;
    int idx = 0;
    std::vector<uint32_t> expected_dst_vec;

    uint32_t num_tile_rows = shape.at(shape.size() - 2) / 32;
    uint32_t num_tile_cols = shape.at(shape.size() - 1) / 32;

    uint32_t start_dram_addr_offset_for_tensor_row = 0;

    for (int i = 0; i < num_tile_rows; i++) {
        for (uint32_t j = 0; j < 32; j++) {
            uint32_t src_addr_ = start_dram_addr_offset_for_tensor_row;
            for (uint32_t k = 0; k < num_tile_cols; k++) {
                // Copy a row
                for (uint32_t l = 0; l < 16; l++) {
                    uint32_t src_addr = src_addr_ + l;
                    expected_dst_vec.push_back(src_vec.at(src_addr_ + l));
                }

                // Zero padding
                for (uint32_t l = 0; l < 31 * 16; l++) {
                    expected_dst_vec.push_back(0);
                }
                src_addr_ += 32 * 16;
            }
            start_dram_addr_offset_for_tensor_row += 16;
        }
        start_dram_addr_offset_for_tensor_row += num_tile_cols * 16;
    }

    TT_FATAL(
        expected_dst_vec.size() == (num_tile_rows * 32) * (num_tile_cols * 16) * 32,
        "Unexpected dst vec size {}.",
        expected_dst_vec.size());
    return expected_dst_vec;
}

bool flatten(
    tt_metal::DispatchFixture* fixture, tt_metal::IDevice* device, uint32_t num_tiles_r = 5, uint32_t num_tiles_c = 5) {
    bool pass = true;

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;

    uint32_t num_tiles = num_tiles_r * num_tiles_c;
    uint32_t num_bytes_per_tensor_row = num_tiles_c * 64;
    uint32_t num_bytes_per_tile = num_tiles * single_tile_size;

    uint32_t dram_buffer_size =
        single_tile_size * num_tiles * 32;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};
    auto src_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();
    auto dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
    // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel,
    // input CB and reader
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 8;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 1;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    auto flatten_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/flatten.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {
        num_tiles * 32  // per_core_tile_cnt
    };

    auto eltwise_unary_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<uint32_t> golden = gold_standard_flatten(src_vec, {num_tiles_r * 32, num_tiles_c * 32});

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    fixture->WriteBuffer(device, src_dram_buffer, src_vec);

    tt_metal::SetRuntimeArgs(
        program, flatten_kernel, core, {dram_buffer_src_addr, 0, num_tiles_r, num_tiles_c, num_bytes_per_tensor_row});

    tt_metal::SetRuntimeArgs(program, unary_writer_kernel, core, {dram_buffer_dst_addr, 0, num_tiles * 32});

    fixture->RunProgram(device, program);

    std::vector<uint32_t> result_vec;
    fixture->ReadBuffer(device, dst_dram_buffer, result_vec);

    ////////////////////////////////////////////////////////////////////////////
    //                      Validation & Teardown
    ////////////////////////////////////////////////////////////////////////////

    TT_FATAL(
        golden.size() == result_vec.size(),
        "Size mismatch between golden {} and result vec {}.",
        golden.size(),
        result_vec.size());
    pass &= (golden == result_vec);

    if (not pass) {
        std::cout << "GOLDEN" << std::endl;
        print_vec_of_uint32_as_packed_bfloat16(golden, num_tiles * 32);

        std::cout << "RESULT" << std::endl;
        print_vec_of_uint32_as_packed_bfloat16(result_vec, num_tiles * 32);
    }
    return pass;
}

bool flatten_stress(tt_metal::IDevice* device, uint32_t num_tiles_r = 5, uint32_t num_tiles_c = 5) {
    // Test Simulating Program Caching with Async Command Queues
    bool pass = true;
    // Create a program used across all loops
    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;

    uint32_t num_tiles = num_tiles_r * num_tiles_c;
    uint32_t num_bytes_per_tensor_row = num_tiles_c * 64;
    uint32_t num_bytes_per_tile = num_tiles * single_tile_size;

    uint32_t dram_buffer_size = single_tile_size * num_tiles * 32;

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 8;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = 16;
    uint32_t num_output_tiles = 1;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    auto cb_output = CreateCircularBuffer(program, core, cb_output_config);

    auto flatten_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/flatten.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {num_tiles * 32};

    auto eltwise_unary_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

    // Inside the loop, run async runtime functions
    for (int i = 0; i < 1000; i++) {
        // Create Device Buffers Asynchronously
        auto src_dram_buffer = CreateBuffer(dram_config);
        auto dst_dram_buffer = CreateBuffer(dram_config);

        // Create the source vector
        std::shared_ptr<std::vector<uint32_t>> src_vec =
            std::make_shared<std::vector<uint32_t>>(create_random_vector_of_bfloat16(
                dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count()));

        std::vector<uint32_t> golden = gold_standard_flatten(*src_vec, {num_tiles_r * 32, num_tiles_c * 32});
        // Set the runtime args asynchronously
        std::shared_ptr<tt_metal::RuntimeArgs> writer_runtime_args = std::make_shared<tt_metal::RuntimeArgs>();
        std::shared_ptr<tt_metal::RuntimeArgs> compute_runtime_args = std::make_shared<tt_metal::RuntimeArgs>();
        *compute_runtime_args = {
            src_dram_buffer.get(), (uint32_t)0, num_tiles_r, num_tiles_c, num_bytes_per_tensor_row};
        *writer_runtime_args = {dst_dram_buffer.get(), (uint32_t)0, num_tiles * 32};

        SetRuntimeArgs(device, tt_metal::detail::GetKernel(program, flatten_kernel), core, compute_runtime_args);

        SetRuntimeArgs(device, tt_metal::detail::GetKernel(program, unary_writer_kernel), core, writer_runtime_args);
        // Async write input
        EnqueueWriteBuffer(device->command_queue(), src_dram_buffer, src_vec, false);
        // Share ownership of buffer with program
        AssignGlobalBufferToProgram(src_dram_buffer, program);
        // Main thread gives up ownership of buffer and src data (this is what python does)
        src_dram_buffer.reset();
        src_vec.reset();
        // Queue up program
        EnqueueProgram(device->command_queue(), program, false);
        // Blocking read
        std::vector<uint32_t> result_vec;
        EnqueueReadBuffer(device->command_queue(), dst_dram_buffer, result_vec, true);

        // Validation of data
        TT_FATAL(
            golden.size() == result_vec.size(),
            "Size mismatch between golden {} and result vec {}.",
            golden.size(),
            result_vec.size());
        pass &= (golden == result_vec);

        if (not pass) {
            std::cout << "GOLDEN" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(golden, num_tiles * 32);

            std::cout << "RESULT" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(result_vec, num_tiles * 32);
        }
    }
    return pass;
}

}  // namespace test_flatten

TEST_F(DispatchFixture, TensixFlatten) {
    // TODO: Re-enable when #7264 is fixed
    GTEST_SKIP();
    uint32_t num_tiles_r = 2;
    uint32_t num_tiles_c = 2;
    if (!this->IsSlowDispatch()) {
        log_info(LogTest, "Flatten running with num_tiles_r=1, num_tiles_c=1");
        num_tiles_r = 1;
        num_tiles_c = 1;
    }
    for (unsigned int id = 0; id < devices_.size(); id++) {
        // TODO: #6097, fix this for fast dispatch remote device.
        if (!this->IsSlowDispatch() && id > 0) {
            continue;
        }
        ASSERT_TRUE(test_flatten::flatten(this, this->devices_.at(id), num_tiles_r, num_tiles_c));
    }
}

}  // namespace tt::tt_metal
