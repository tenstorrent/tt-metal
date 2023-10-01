// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <thread>
#include <unistd.h>

#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/executor.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"

using namespace tt::tt_metal;
using namespace tt::test_utils;

namespace concurrent_tests {

struct DatacopyProgramConfig {
    size_t num_tiles = 1;
    size_t page_size_bytes = 2 * 32 * 32;
    BufferType input_buffer_type = BufferType::DRAM;
    BufferType output_buffer_type = BufferType::DRAM;
    CoreCoord logical_core = CoreCoord({.x = 0, .y = 0});
    tt::DataFormat l1_data_format = tt::DataFormat::Float16_b;
    bool fast_dispatch = false;
};

bool reader_datacopy_writer(Device* device, const DatacopyProgramConfig& cfg) {
    bool pass = true;

    const uint32_t input0_cb_index = 0;
    const uint32_t output_cb_index = 16;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = Program();
    size_t size_bytes = cfg.num_tiles * cfg.page_size_bytes;
    auto input_buffer = CreateBuffer(device, size_bytes, cfg.page_size_bytes, cfg.input_buffer_type);
    auto output_buffer = CreateBuffer(device, size_bytes, cfg.page_size_bytes, cfg.output_buffer_type);

    constexpr uint32_t num_pages_cb = 1;
    CircularBufferConfig l1_input_cb_config = CircularBufferConfig(cfg.page_size_bytes, {{input0_cb_index, cfg.l1_data_format}})
        .set_page_size(input0_cb_index, cfg.page_size_bytes);
    auto l1_input_cb = CreateCircularBuffer(program, cfg.logical_core, l1_input_cb_config);

    CircularBufferConfig l1_output_cb_config = CircularBufferConfig(cfg.page_size_bytes, {{output_cb_index, cfg.l1_data_format}})
        .set_page_size(output_cb_index, cfg.page_size_bytes);
    auto l1_output_cb = CreateCircularBuffer(program, cfg.logical_core, l1_output_cb_config);

    bool input_is_dram = cfg.input_buffer_type == BufferType::DRAM;
    bool output_is_dram = cfg.output_buffer_type == BufferType::DRAM;

    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/banked_reader.cpp",
        cfg.logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {input0_cb_index, uint32_t(input_buffer.page_size()), (uint32_t)input_is_dram}});

    auto writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/banked_writer.cpp",
        cfg.logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {output_cb_index, uint32_t(output_buffer.page_size()), (uint32_t)output_is_dram}});

    vector<uint32_t> compute_kernel_args = {
        uint(cfg.num_tiles)  // per_core_tile_cnt
    };
    auto datacopy_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
        cfg.logical_core,
        ComputeConfig{.compile_args = compute_kernel_args});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> input_packed = generate_packed_uniform_random_vector<uint32_t, df::bfloat16>(
        -1.0f, 1.0f, size_bytes / df::bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    if (cfg.fast_dispatch) {
        EnqueueWriteBuffer(*detail::GLOBAL_CQ, input_buffer, input_packed, /*blocking=*/false);
    } else {
        WriteToBuffer(input_buffer, input_packed);
    }

    SetRuntimeArgs(
        program,
        reader_kernel,
        cfg.logical_core,
        {
            (uint32_t)input_buffer.address(),
            (uint32_t)cfg.num_tiles,
        }
    );
    SetRuntimeArgs(
        program,
        writer_kernel,
        cfg.logical_core,
        {
            (uint32_t)output_buffer.address(),
            (uint32_t)cfg.num_tiles,
        }
    );

    if (cfg.fast_dispatch) {
        EnqueueProgram(*detail::GLOBAL_CQ, program, false);
        Finish(*detail::GLOBAL_CQ);
    } else {
        LaunchProgram(device, program);
    }

    std::vector<uint32_t> dest_buffer_data;
    if (cfg.fast_dispatch) {
        EnqueueReadBuffer(*detail::GLOBAL_CQ, output_buffer, dest_buffer_data, true);
    } else {
        ReadFromBuffer(output_buffer, dest_buffer_data);
    }

    EXPECT_EQ(input_packed, dest_buffer_data);
    pass &= input_packed == dest_buffer_data;

    return pass;
}

}   // namespace concurrent_tests
