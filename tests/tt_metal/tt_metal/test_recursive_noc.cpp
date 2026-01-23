// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-logger/tt-logger.hpp>

using namespace tt;
using namespace tt::tt_metal;

TEST_F(MeshDeviceSingleCardFixture, RecursiveNocWrite) {
    IDevice* dev = devices_[0]->get_devices()[0];
    Program program = CreateProgram();

    CoreCoord core = {0, 0};
    constexpr uint32_t buffer_size = 256;

    // Create input, output, and cmd circular buffers
    uint32_t input_cb_index = tt::CBIndex::c_0;
    uint32_t output_cb_index = tt::CBIndex::c_1;
    uint32_t cmd_cb_index = tt::CBIndex::c_2;

    CircularBufferConfig input_cb_config =
        CircularBufferConfig(buffer_size, {{input_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(input_cb_index, buffer_size);
    CreateCircularBuffer(program, core, input_cb_config);

    CircularBufferConfig output_cb_config =
        CircularBufferConfig(buffer_size, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, buffer_size);
    CreateCircularBuffer(program, core, output_cb_config);

    // cmd_cb needs to hold 17 words (68 bytes) for serialized NOC command
    CircularBufferConfig cmd_cb_config =
        CircularBufferConfig(128, {{cmd_cb_index, tt::DataFormat::Float16_b}}).set_page_size(cmd_cb_index, 128);
    CreateCircularBuffer(program, core, cmd_cb_config);

    // Create kernel with CB indices as compile args
    auto writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/recursive_noc_writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {input_cb_index, output_cb_index, cmd_cb_index}});

    // Run the program
    log_info(tt::LogTest, "Launching kernel...");
    detail::LaunchProgram(dev, program);
    log_info(tt::LogTest, "Kernel complete - check DPRINT output");
}
