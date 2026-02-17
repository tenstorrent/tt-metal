// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <functional>
#include <random>
#include <map>
#include <memory>
#include <ostream>
#include <utility>
#include <variant>
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
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

namespace tt::tt_metal {

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::matmul {

std::vector<int8_t> generate_uniform_int8(size_t num_elements, int8_t min, int8_t max, uint32_t seed = 0) {
    std::vector<int8_t> random_array(num_elements);

    std::mt19937 gen(seed);
    std::uniform_int_distribution<int16_t> distrib(min, max);

    for (size_t i = 0; i < num_elements; i++) {
        random_array[i] = static_cast<int8_t>(distrib(gen));
    }

    return random_array;
}

void convert_to_sign_mag(std::vector<int8_t>& vec) {
    for (signed char & i : vec) {
        int8_t temp = i;
        if (temp < 0) {
            if (temp == -128) {
                temp = -127;
            }
            temp = (~temp) + 1;
            temp = temp | 0x80;
            i = temp;
        }
    }
}

int get_output_coordinate(int x, int y) {
    int offset = ((x < 16) ? 0 : 256) + ((y < 16) ? 0 : 512);
    return offset + ((y % 16) * 16) + (x % 16);
}

bool single_tile_matmul_int8(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    bool pass = true;

    CoreCoord core(0, 0);
    const uint32_t in0_cb_index = 0;
    const uint32_t in1_cb_index = 1;
    const uint32_t out_cb_index = 16;
    const size_t tile_size = 1 * 1 * 32 * 32;
    const size_t byte_size = 1 * tile_size;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;

    auto local_buffer_config =
        distributed::DeviceLocalBufferConfig{.page_size = tile_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
    auto distributed_buffer_config = distributed::ShardedBufferConfig{
        .global_size = tile_size,
        .global_buffer_shape = {32, 32},
        .shard_shape = {32, 32},
        .shard_orientation = ShardOrientation::ROW_MAJOR};

    workload.add_program(device_range, tt_metal::CreateProgram());
    auto& program = workload.get_programs().at(device_range);

    auto input0_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
    const uint32_t in0_dram_addr = input0_dram_buffer->address();
    auto input1_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
    const uint32_t in1_dram_addr = input1_dram_buffer->address();
    auto output_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
    const uint32_t out_dram_addr = output_dram_buffer->address();

    tt_metal::CircularBufferConfig l1_input0_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{in0_cb_index, tt::DataFormat::Int8}})
            .set_page_size(in0_cb_index, byte_size);
    tt_metal::CreateCircularBuffer(program, core, l1_input0_cb_config);

    tt_metal::CircularBufferConfig l1_input1_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{in1_cb_index, tt::DataFormat::Int8}})
            .set_page_size(in1_cb_index, byte_size);
    tt_metal::CreateCircularBuffer(program, core, l1_input1_cb_config);

    tt_metal::CircularBufferConfig l1_output_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{out_cb_index, tt::DataFormat::Int8}})
            .set_page_size(out_cb_index, byte_size);
    tt_metal::CreateCircularBuffer(program, core, l1_output_cb_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {in0_cb_index, in1_cb_index}});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {out_cb_index}});

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/single_tile_compute.cpp",
        core,
        tt_metal::ComputeConfig{.fp32_dest_acc_en = true, .compile_args = {in0_cb_index, in1_cb_index, out_cb_index}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    int8_t min = -5, max = 5;
    std::vector<int8_t> input_0 = generate_uniform_int8(tile_size, min, max, 0 /*seed*/);
    std::vector<int8_t> input_1 = generate_uniform_int8(tile_size, min, max, 1 /*different seed*/);

    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<int32_t> golden_output_int32(tile_size, 0);
    std::vector<int8_t> golden_output(tile_size, 0);
    // matmul
    for (int x = 0; x < 32; x++) {
        for (int y = 0; y < 32; y++) {
            for (int z = 0; z < 32; z++) {
                golden_output_int32[get_output_coordinate(x, y)] +=
                    (int32_t)input_0[get_output_coordinate(z, y)] * (int32_t)input_1[get_output_coordinate(x, z)];
            }
        }
    }
    // clamping to int8
    for (int i = 0; i < 1024; i++) {
        if (golden_output_int32[i] > 127) {
            golden_output_int32[i] = 127;
        } else if (golden_output_int32[i] < -127) {
            golden_output_int32[i] = -127;
        }
        golden_output[i] = static_cast<int8_t>(golden_output_int32[i]);
    }
    convert_to_sign_mag(golden_output);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    convert_to_sign_mag(input_0);
    distributed::EnqueueWriteMeshBuffer(cq, input0_dram_buffer, input_0);
    convert_to_sign_mag(input_1);
    distributed::EnqueueWriteMeshBuffer(cq, input1_dram_buffer, input_1);

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)in0_dram_addr,
            (uint32_t)0,  // in_0 dram bank id
            (uint32_t)in1_dram_addr,
            (uint32_t)0,
            (uint32_t)1,  // num_tiles
        });
    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)out_dram_addr,
            (uint32_t)0,
            (uint32_t)1,  // num_tiles
        });

    distributed::EnqueueMeshWorkload(cq, workload, false);

    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<int8_t> dest_buffer_data;
    distributed::EnqueueReadMeshBuffer(cq, dest_buffer_data, output_dram_buffer);
    pass = dest_buffer_data == golden_output;

    for (int i = 0; i < 1024; i++) {
        if (dest_buffer_data[i] != golden_output[i]) {
            std::cout << "element " << i << ": " << (uint32_t)dest_buffer_data[i] << " vs. "
                      << (uint32_t)golden_output[i] << " \n";
        }
    }

    return pass;
}

}  // namespace unit_tests::compute::matmul

TEST_F(MeshDeviceFixture, TensixTestSingleCoreSingleTileComputeMatmulInt8) {
    ASSERT_TRUE(unit_tests::compute::matmul::single_tile_matmul_int8(this->devices_.at(0)));
}

}  // namespace tt::tt_metal
