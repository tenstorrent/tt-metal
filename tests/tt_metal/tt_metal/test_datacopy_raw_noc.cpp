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
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-logger/tt-logger.hpp>

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

TEST_F(MeshDeviceSingleCardFixture, DatacopyRawNoc) {
    IDevice* dev = devices_[0]->get_devices()[0];
    Program program = CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
    auto src_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();
    auto dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    // Allocate 1KB L1 buffer for NOC command serialization experiment
    constexpr uint32_t noc_cmd_buffer_size = 1024;
    InterleavedBufferConfig l1_config{
        .device = dev, .size = noc_cmd_buffer_size, .page_size = noc_cmd_buffer_size, .buffer_type = BufferType::L1};
    auto noc_cmd_l1_buffer = CreateBuffer(l1_config);
    uint32_t noc_cmd_buffer_addr = noc_cmd_l1_buffer->address();
    log_info(tt::LogTest, "Allocated 1KB L1 buffer for NOC commands at address: 0x{:08x}", noc_cmd_buffer_addr);

    // Allocate second 1KB L1 buffer
    auto noc_cmd_l1_buffer_2 = CreateBuffer(l1_config);
    uint32_t noc_cmd_buffer_addr_2 = noc_cmd_l1_buffer_2->address();
    log_info(tt::LogTest, "Allocated second 1KB L1 buffer at address: 0x{:08x}", noc_cmd_buffer_addr_2);

    // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input
    // CB CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math
    // kernel, input CB and reader
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 8;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    auto unary_reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Use the raw NOC write kernel instead of experimental API
    auto unary_writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {ouput_cb_index}});  // Pass CB index as compile arg

    vector<uint32_t> compute_kernel_args = {uint(num_tiles)};

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args});

    // Execute
    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    detail::WriteToBuffer(src_dram_buffer, src_vec);

    SetRuntimeArgs(program, unary_reader_kernel, core, {dram_buffer_src_addr, 0, num_tiles});
    SetRuntimeArgs(
        program,
        unary_writer_kernel,
        core,
        {dram_buffer_dst_addr, 0, num_tiles, noc_cmd_buffer_addr, noc_cmd_buffer_addr_2});

    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    // Read back the NOC command buffer to see what was serialized
    std::vector<uint32_t> noc_cmd_buffer_contents;
    detail::ReadFromBuffer(noc_cmd_l1_buffer, noc_cmd_buffer_contents);
    log_info(tt::LogTest, "=== NOC Command Buffer 1 Contents (first 16 words) ===");
    for (uint32_t i = 0; i < std::min(16u, (uint32_t)noc_cmd_buffer_contents.size()); i++) {
        log_info(tt::LogTest, "  [0x{:02x}] = 0x{:08x}", i * 4, noc_cmd_buffer_contents[i]);
    }

    // Read back the second NOC command buffer
    std::vector<uint32_t> noc_cmd_buffer_contents_2;
    detail::ReadFromBuffer(noc_cmd_l1_buffer_2, noc_cmd_buffer_contents_2);
    log_info(tt::LogTest, "=== NOC Command Buffer 2 Contents (first 16 words) ===");
    for (uint32_t i = 0; i < std::min(16u, (uint32_t)noc_cmd_buffer_contents_2.size()); i++) {
        log_info(tt::LogTest, "  [0x{:02x}] = 0x{:08x}", i * 4, noc_cmd_buffer_contents_2[i]);
    }

    // Print first few elements for debugging
    log_debug(tt::LogTest, "=== Source Buffer (first 32 elements) ===");
    for (uint32_t i = 0; i < std::min(32u, (uint32_t)src_vec.size()); i++) {
        log_debug(tt::LogTest, "src[{}] = 0x{:08x}", i, src_vec[i]);
    }

    log_debug(tt::LogTest, "=== Destination Buffer (first 32 elements) ===");
    for (uint32_t i = 0; i < std::min(32u, (uint32_t)result_vec.size()); i++) {
        log_debug(tt::LogTest, "dst[{}] = 0x{:08x}", i, result_vec[i]);
    }

    // Validation
    log_info(tt::LogTest, "Checking {} tiles ({} bytes)...", num_tiles, dram_buffer_size);
    EXPECT_EQ(src_vec, result_vec);
    log_info(tt::LogTest, "✓ Data correctness verified!");
}
