// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/common/scoped_timer.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"

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

TEST_F(SingleDeviceTraceFixture, EnqueueOneProgramTrace) {
    Setup(2048, 2);
    Buffer input(this->device_, 2048, 2048, BufferType::DRAM);
    Buffer output(this->device_, 2048, 2048, BufferType::DRAM);

    CommandQueue& command_queue = this->device_->command_queue(0);
    CommandQueue& data_movement_queue = this->device_->command_queue(1);

    Program simple_program = create_simple_unary_program(input, output);
    vector<uint32_t> input_data(input.size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    // Eager mode
    vector<uint32_t> eager_output_data;
    eager_output_data.resize(input_data.size());

    EnqueueWriteBuffer(data_movement_queue, input, input_data.data(), true);
    EnqueueProgram(command_queue, simple_program, true);
    EnqueueReadBuffer(data_movement_queue, output, eager_output_data.data(), true);

    // Trace mode
    vector<uint32_t> trace_output_data;
    trace_output_data.resize(input_data.size());

    EnqueueWriteBuffer(data_movement_queue, input, input_data.data(), true);

    uint32_t tid = BeginTraceCapture(this->device_, command_queue.id());
    EnqueueProgram(command_queue, simple_program, false);
    EndTraceCapture(this->device_, command_queue.id(), tid);

    EnqueueTrace(command_queue, tid, true);
    EnqueueReadBuffer(data_movement_queue, output, trace_output_data.data(), true);
    EXPECT_TRUE(eager_output_data == trace_output_data);

    // Done
    Finish(command_queue);
    ReleaseTrace(this->device_, tid);
}

TEST_F(SingleDeviceTraceFixture, EnqueueOneProgramTraceLoops) {
    Setup(4096, 2);
    Buffer input(this->device_, 2048, 2048, BufferType::DRAM);
    Buffer output(this->device_, 2048, 2048, BufferType::DRAM);

    CommandQueue& command_queue = this->device_->command_queue(0);
    CommandQueue& data_movement_queue = this->device_->command_queue(1);

    Program simple_program = create_simple_unary_program(input, output);
    vector<uint32_t> input_data(input.size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    // Trace mode output
    uint32_t num_loops = 10;
    vector<vector<uint32_t>> trace_outputs;

    for (auto i = 0; i < num_loops; i++) {
        trace_outputs.push_back({});
        trace_outputs[i].resize(input_data.size());
    }

    // Compile
    EnqueueProgram(command_queue, simple_program, true);

    // Trace mode execution
    uint32_t trace_id = 0;
    bool trace_captured = false;
    for (auto i = 0; i < num_loops; i++) {
        EnqueueWriteBuffer(data_movement_queue, input, input_data.data(), true);

        if (not trace_captured) {
            trace_id = BeginTraceCapture(this->device_, command_queue.id());
            EnqueueProgram(command_queue, simple_program, false);
            EndTraceCapture(this->device_, command_queue.id(), trace_id);
            trace_captured = true;
        }

        EnqueueTrace(command_queue, trace_id, false);
        EnqueueReadBuffer(data_movement_queue, output, trace_outputs[i].data(), true);

        // Expect same output across all loops
        EXPECT_TRUE(trace_outputs[i] == trace_outputs[0]);
    }

    // Done
    Finish(command_queue);
    ReleaseTrace(this->device_, trace_id);
}

TEST_F(SingleDeviceTraceFixture, EnqueueOneProgramTraceBenchmark) {
    Setup(6144, 2);
    Buffer input(this->device_, 2048, 2048, BufferType::DRAM);
    Buffer output(this->device_, 2048, 2048, BufferType::DRAM);

    constexpr bool kBlocking = true;
    constexpr bool kNonBlocking = false;
    vector<bool> blocking_flags = {kBlocking, kNonBlocking};

    // Single Q for data and commands
    // Keep this queue in passthrough mode for now
    CommandQueue& command_queue = this->device_->command_queue(0);

    auto simple_program = create_simple_unary_program(input, output);
    vector<uint32_t> input_data(input.size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    // Trace mode output
    uint32_t num_loops = 10;
    vector<vector<uint32_t>> trace_outputs;

    for (auto i = 0; i < num_loops; i++) {
        trace_outputs.push_back({});
        trace_outputs[i].resize(input_data.size());
    }

    // Eager mode
    vector<uint32_t> expected_output_data;
    vector<uint32_t> eager_output_data;
    expected_output_data.resize(input_data.size());
    eager_output_data.resize(input_data.size());

    // Warm up and use the eager blocking run as the expected output
    EnqueueWriteBuffer(command_queue, input, input_data.data(), kBlocking);
    EnqueueProgram(command_queue, simple_program, kBlocking);
    EnqueueReadBuffer(command_queue, output, expected_output_data.data(), kBlocking);
    Finish(command_queue);

    for (bool blocking : blocking_flags) {
        std::string mode = blocking ? "Eager-B" : "Eager-NB";
        for (auto i = 0; i < num_loops; i++) {
            tt::ScopedTimer timer(mode + " loop " + std::to_string(i));
            EnqueueWriteBuffer(command_queue, input, input_data.data(), blocking);
            EnqueueProgram(command_queue, simple_program, blocking);
            EnqueueReadBuffer(command_queue, output, eager_output_data.data(), blocking);
        }
        if (not blocking) {
            // (Optional) wait for the last non-blocking command to finish
            Finish(command_queue);
        }
        EXPECT_TRUE(eager_output_data == expected_output_data);
    }

    // Capture trace on a trace queue
    uint32_t tid = BeginTraceCapture(this->device_, command_queue.id());
    EnqueueProgram(command_queue, simple_program, false);
    EndTraceCapture(this->device_, command_queue.id(), tid);

    // Trace mode execution
    for (auto i = 0; i < num_loops; i++) {
        tt::ScopedTimer timer("Trace loop " + std::to_string(i));
        EnqueueWriteBuffer(command_queue, input, input_data.data(), kNonBlocking);
        EnqueueTrace(command_queue, tid, kNonBlocking);
        EnqueueReadBuffer(command_queue, output, trace_outputs[i].data(), kNonBlocking);
    }
    Finish(command_queue);

    // Expect same output across all loops
    for (auto i = 0; i < num_loops; i++) {
        EXPECT_TRUE(trace_outputs[i] == trace_outputs[0]);
    }
    ReleaseTrace(this->device_, tid);
}

} // end namespace basic_tests
