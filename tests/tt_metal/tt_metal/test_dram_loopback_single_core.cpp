// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-logger/tt-logger.hpp>

using namespace tt;
using namespace tt::tt_metal;

//////////////////////////////////////////////////////////////////////////////////////////
// 1. Host writes data to buffer in DRAM
// 2. dram_copy kernel on logical core {0, 0} BRISC copies data from buffer
//      in step 1. to buffer in L1 and back to another buffer in DRAM
// 4. Host reads from buffer written to in step 2.
//////////////////////////////////////////////////////////////////////////////////////////
TEST_F(MeshDeviceSingleCardFixture, DramLoopbackSingleCore) {
    IDevice* dev = devices_[0]->get_devices()[0];
    Program program = CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 50;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    uint32_t l1_buffer_addr = 400 * 1024;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
    auto input_dram_buffer = CreateBuffer(dram_config);
    uint32_t input_dram_buffer_addr = input_dram_buffer->address();

    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t output_dram_buffer_addr = output_dram_buffer->address();

    auto dram_copy_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Execute
    std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    detail::WriteToBuffer(input_dram_buffer, input_vec);

    SetRuntimeArgs(
        program,
        dram_copy_kernel,
        core,
        {l1_buffer_addr, input_dram_buffer_addr, 0, output_dram_buffer_addr, 0, dram_buffer_size});

    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(output_dram_buffer, result_vec);

    // Validation
    EXPECT_EQ(input_vec, result_vec);
}
