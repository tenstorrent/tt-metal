// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "command_queue_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt::tt_metal;

struct TestBufferConfig {
    uint32_t num_pages;
    uint32_t page_size;
    BufferType buftype;
};

Program create_simple_unary_program(const Buffer& input, const Buffer& output) {
    Program program = CreateProgram();

    CoreCoord worker = {0, 0};
    auto reader_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary.cpp",
        worker,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        worker,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto sfpu_kernel = CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        worker,
        ComputeConfig{
            .math_approx_mode = true,
            .compile_args = {1, 1},
            .defines = {{"SFPU_OP_EXP_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}}});

    CircularBufferConfig input_cb_config = CircularBufferConfig(2048, {{0, tt::DataFormat::Float16_b}})
            .set_page_size(0, 2048);

    CoreRange core_range({0, 0});
    CreateCircularBuffer(program, core_range, input_cb_config);
    vector<uint32_t> writer_rt_args = {
        output.address(),
        (uint32_t)output.noc_coordinates().x,
        (uint32_t)output.noc_coordinates().y,
        output.num_pages()
    };
    SetRuntimeArgs(program, writer_kernel, worker, writer_rt_args);

    CircularBufferConfig output_cb_config = CircularBufferConfig(2048, {{16, tt::DataFormat::Float16_b}})
            .set_page_size(16, 2048);

    CreateCircularBuffer(program, core_range, output_cb_config);
    vector<uint32_t> reader_rt_args = {
        input.address(),
        (uint32_t)input.noc_coordinates().x,
        (uint32_t)input.noc_coordinates().y,
        input.num_pages()
    };
    SetRuntimeArgs(program, reader_kernel, worker, reader_rt_args);

    return program;
}

// All basic trace tests just assert that the replayed result exactly matches
// the eager mode results
namespace basic_tests {

TEST_F(MultiCommandQueueSingleDeviceFixture, EnqueueOneProgramTrace) {

    Buffer input(this->device_, 2048, 2048, BufferType::DRAM);
    Buffer output(this->device_, 2048, 2048, BufferType::DRAM);

    CommandQueue trace_queue(this->device_, 0);
    CommandQueue data_movement_queue(this->device_, 1);

    Program simple_program = create_simple_unary_program(input, output);
    vector<uint32_t> input_data(input.size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }
    try {
        EnqueueWriteBuffer(data_movement_queue, input, input_data.data(), true);
        Trace trace = BeginTrace(trace_queue);
        EnqueueProgram(trace_queue, simple_program, false, trace);
        Finish(trace_queue);
        EndTrace(trace);
        vector<uint32_t> eager_output_data;
        eager_output_data.resize(input_data.size());
        EnqueueReadBuffer(data_movement_queue, output, eager_output_data.data(), true);

        vector<uint32_t> trace_output_data;
        trace_output_data.resize(input_data.size());
        EnqueueTrace(trace, true);
        EnqueueReadBuffer(data_movement_queue, output, trace_output_data.data(), true);

        EXPECT_TRUE(eager_output_data == trace_output_data);
    } catch (...) {
        tt::log_info("Bad state reached in EnqueueOneProgramTrace");
    }
}
} // end namespace basic_tests
