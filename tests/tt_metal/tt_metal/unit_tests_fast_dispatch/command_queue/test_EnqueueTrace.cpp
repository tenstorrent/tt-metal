// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include "command_queue_fixture.hpp"
#include "detail/tt_metal.hpp"
#include "tt_metal/common/env_lib.hpp"
#include "gtest/gtest.h"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/program/program.hpp"
#include "tt_metal/impl/program/program_pool.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/common/scoped_timer.hpp"
#include "tt_metal/host_api.hpp"


using namespace tt;
using namespace tt::tt_metal;

struct TestBufferConfig {
    uint32_t num_pages;
    uint32_t page_size;
    BufferType buftype;
};

ScopedProgramHandle create_simple_unary_program(Buffer& input, Buffer& output) {
    auto program = CreateScopedProgram();
    Device* device = input.device();
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
    std::shared_ptr<RuntimeArgs> writer_runtime_args = std::make_shared<RuntimeArgs>();
    std::shared_ptr<RuntimeArgs> reader_runtime_args = std::make_shared<RuntimeArgs>();

    *writer_runtime_args = {
        &output,
        (uint32_t)output.noc_coordinates().x,
        (uint32_t)output.noc_coordinates().y,
        output.num_pages()
    };

    *reader_runtime_args = {
        &input,
        (uint32_t)input.noc_coordinates().x,
        (uint32_t)input.noc_coordinates().y,
        input.num_pages()
    };

    auto* program_ptr = ProgramPool::instance().get_program(program);
    SetRuntimeArgs(device, detail::GetKernel(*program_ptr, writer_kernel), worker, writer_runtime_args);
    SetRuntimeArgs(device, detail::GetKernel(*program_ptr, reader_kernel), worker, reader_runtime_args);

    CircularBufferConfig output_cb_config = CircularBufferConfig(2048, {{16, tt::DataFormat::Float16_b}})
            .set_page_size(16, 2048);

    CreateCircularBuffer(program, core_range, output_cb_config);
    return std::move(program);
}

// All basic trace tests just assert that the replayed result exactly matches
// the eager mode results
namespace basic_tests {

constexpr bool kBlocking = true;
constexpr bool kNonBlocking = false;
vector<bool> blocking_flags = {kBlocking, kNonBlocking};

TEST_F(SingleDeviceTraceFixture, InstantiateTraceSanity) {
    Setup(2048);
    CommandQueue& command_queue = this->device_->command_queue();

    Buffer input(this->device_, 2048, 2048, BufferType::DRAM);
    vector<uint32_t> input_data(input.size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }
    Buffer output(this->device_, 2048, 2048, BufferType::DRAM);
    auto simple_program = create_simple_unary_program(input, output);
    EnqueueProgram(command_queue, simple_program, true);
    uint32_t tid = BeginTraceCapture(this->device_, command_queue.id());
    EnqueueProgram(command_queue, simple_program, kNonBlocking);
    EndTraceCapture(this->device_, command_queue.id(), tid);

    // Instantiate a trace on a device bound command queue
    auto trace_inst = this->device_->get_trace(tid);
    vector<uint32_t> data_fd, data_bd;

    // Backdoor read the trace buffer
    ::detail::ReadFromBuffer(trace_inst->buffer, data_bd);

    // Frontdoor reaad the trace buffer
    data_fd.resize(trace_inst->buffer->size() / sizeof(uint32_t));
    EnqueueReadBuffer(command_queue, trace_inst->buffer, data_fd.data(), kBlocking);
    EXPECT_EQ(data_fd, data_bd);

    log_trace(LogTest, "Trace buffer content: {}", data_fd);
    ReleaseTrace(this->device_, tid);
}

TEST_F(SingleDeviceTraceFixture, EnqueueProgramTraceCapture) {
    Setup(2048);
    Buffer input(this->device_, 2048, 2048, BufferType::DRAM);
    Buffer output(this->device_, 2048, 2048, BufferType::DRAM);

    CommandQueue& command_queue = this->device_->command_queue();

    auto simple_program = create_simple_unary_program(input, output);
    vector<uint32_t> input_data(input.size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    vector<uint32_t> eager_output_data;
    eager_output_data.resize(input_data.size());
    vector<uint32_t> trace_output_data;
    trace_output_data.resize(input_data.size());

    EnqueueWriteBuffer(command_queue, input, input_data.data(), true);
    EnqueueProgram(command_queue, simple_program, true);
    EnqueueReadBuffer(command_queue, output, eager_output_data.data(), true);

    EnqueueWriteBuffer(command_queue, input, input_data.data(), true);

    uint32_t tid = BeginTraceCapture(this->device_, command_queue.id());
    EnqueueProgram(command_queue, simple_program, false);
    EndTraceCapture(this->device_, command_queue.id(), tid);
    // Create and Enqueue a Program with a live trace to ensure that a warning is generated
    Buffer input_temp(this->device_, 2048, 2048, BufferType::DRAM);
    Buffer output_temp(this->device_, 2048, 2048, BufferType::DRAM);
    auto simple_program_temp = create_simple_unary_program(input_temp, output_temp);
    EnqueueProgram(command_queue, simple_program_temp, true);
    // Run trace that can clobber the temporary buffers created above
    EnqueueProgram(command_queue, simple_program, false);
    EnqueueTrace(command_queue, tid, true);
    EnqueueReadBuffer(command_queue, output, trace_output_data.data(), true);
    EXPECT_TRUE(eager_output_data == trace_output_data);

    // Done
    Finish(command_queue);
    ReleaseTrace(this->device_, tid);
}

TEST_F(SingleDeviceTraceFixture, EnqueueProgramDeviceCapture) {
    Setup(2048);
    Buffer input(this->device_, 2048, 2048, BufferType::DRAM);
    Buffer output(this->device_, 2048, 2048, BufferType::DRAM);

    CommandQueue& command_queue = this->device_->command_queue();

    vector<uint32_t> input_data(input.size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    vector<uint32_t> eager_output_data;
    eager_output_data.resize(input_data.size());
    vector<uint32_t> trace_output_data;
    trace_output_data.resize(input_data.size());

    bool has_eager = true;
    ScopedProgramHandle simple_program;
    // EAGER MODE EXECUTION
    if (has_eager) {
        simple_program = create_simple_unary_program(input, output);
        EnqueueWriteBuffer(command_queue, input, input_data.data(), true);
        EnqueueProgram(command_queue, simple_program, true);
        EnqueueReadBuffer(command_queue, output, eager_output_data.data(), true);
    }

    // THIS->DEVICE_ CAPTURE AND REPLAY MODE
    bool has_trace = false;
    uint32_t tid = 0;
    for (int i = 0; i < 1; i++) {
        EnqueueWriteBuffer(command_queue, input, input_data.data(), true);

        if (!has_trace) {
            // Program must be cached first
            tid = BeginTraceCapture(this->device_, command_queue.id());
            EnqueueProgram(command_queue, simple_program, false);
            EndTraceCapture(this->device_, command_queue.id(), tid);
            has_trace = true;
        }
        ReplayTrace(this->device_, command_queue.id(), tid, true);

        EnqueueReadBuffer(command_queue, output, trace_output_data.data(), true);
        if (has_eager) EXPECT_TRUE(eager_output_data == trace_output_data);
    }

    // Done
    Finish(command_queue);
    ReleaseTrace(this->device_, tid);
}

TEST_F(SingleDeviceTraceFixture, EnqueueTwoProgramTrace) {
    Setup(6144);
    // Get command queue from device for this test, since its running in async mode
    CommandQueue& command_queue = this->device_->command_queue();

    Buffer input(this->device_, 2048, 2048, BufferType::DRAM);
    Buffer interm(this->device_, 2048, 2048, BufferType::DRAM);
    Buffer output(this->device_, 2048, 2048, BufferType::DRAM);

    auto op0 = create_simple_unary_program(input, interm);
    auto op1 = create_simple_unary_program(interm, output);
    vector<uint32_t> input_data(input.size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    // Trace mode output
    uint32_t num_loops = parse_env<int>("TT_METAL_TRACE_LOOPS", 5);
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
    EnqueueProgram(command_queue, op0, kBlocking);
    EnqueueProgram(command_queue, op1, kBlocking);
    EnqueueReadBuffer(command_queue, output, expected_output_data.data(), kBlocking);
    Finish(command_queue);

    for (bool blocking : blocking_flags) {
        std::string mode = blocking ? "Eager-B" : "Eager-NB";
        for (auto i = 0; i < num_loops; i++) {
            ScopedTimer timer(mode + " loop " + std::to_string(i));
            EnqueueWriteBuffer(command_queue, input, input_data.data(), blocking);
            EnqueueProgram(command_queue, op0, blocking);
            EnqueueProgram(command_queue, op1, blocking);
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
    EnqueueProgram(command_queue, op0, kNonBlocking);
    EnqueueProgram(command_queue, op1, kNonBlocking);
    EndTraceCapture(this->device_, command_queue.id(), tid);

    // Trace mode execution
    for (auto i = 0; i < num_loops; i++) {
        ScopedTimer timer("Trace loop " + std::to_string(i));
        EnqueueWriteBuffer(command_queue, input, input_data.data(), kNonBlocking);
        EnqueueTrace(command_queue, tid, kNonBlocking);
        EnqueueReadBuffer(command_queue, output, trace_outputs[i].data(), kNonBlocking);
    }
    Finish(command_queue);
    ReleaseTrace(this->device_, tid);

    // Expect same output across all loops
    for (auto i = 0; i < num_loops; i++) {
        EXPECT_TRUE(trace_outputs[i] == trace_outputs[0]);
    }
}

TEST_F(SingleDeviceTraceFixture, EnqueueMultiProgramTraceBenchmark) {
    Setup(6144);
    CommandQueue& command_queue = this->device_->command_queue();

    std::shared_ptr<Buffer> input = std::make_shared<Buffer>(this->device_, 2048, 2048, BufferType::DRAM);
    std::shared_ptr<Buffer> output = std::make_shared<Buffer>(this->device_, 2048, 2048, BufferType::DRAM);

    uint32_t num_loops = parse_env<int>("TT_METAL_TRACE_LOOPS", 4);
    uint32_t num_programs = parse_env<int>("TT_METAL_TRACE_PROGRAMS", 4);
    std::vector<std::shared_ptr<Buffer>> interm_buffers;
    std::vector<ScopedProgramHandle> programs;

    vector<uint32_t> input_data(input->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    for (int i = 0; i < num_programs; i++) {
        interm_buffers.push_back(std::make_shared<Buffer>(this->device_, 2048, 2048, BufferType::DRAM));
        if (i == 0) {
            programs.emplace_back(create_simple_unary_program(*input, *(interm_buffers[i])));
        } else if (i == (num_programs - 1)) {
            programs.emplace_back(create_simple_unary_program(*(interm_buffers[i - 1]), *output));
        } else {
            programs.emplace_back(create_simple_unary_program(*(interm_buffers[i - 1]), *(interm_buffers[i])));
        }
    }

    // Eager mode
    vector<uint32_t> eager_output_data;
    eager_output_data.resize(input_data.size());

    // Trace mode output
    vector<vector<uint32_t>> trace_outputs;

    for (uint32_t i = 0; i < num_loops; i++) {
        trace_outputs.push_back({});
        trace_outputs[i].resize(input_data.size());
    }

    for (bool blocking : blocking_flags) {
        std::string mode = blocking ? "Eager-B" : "Eager-NB";
        log_info(LogTest, "Starting {} profiling with {} programs", mode, num_programs);
        for (uint32_t iter = 0; iter < num_loops; iter++) {
            ScopedTimer timer(mode + " loop " + std::to_string(iter));
            EnqueueWriteBuffer(command_queue, input, input_data.data(), blocking);
            for (uint32_t i = 0; i < num_programs; i++) {
                EnqueueProgram(command_queue, programs[i], blocking);
            }
            EnqueueReadBuffer(command_queue, output, eager_output_data.data(), blocking);
        }
        if (not blocking) {
            // (Optional) wait for the last non-blocking command to finish
            Finish(command_queue);
        }
    }

    // Capture trace on a trace queue
    uint32_t tid = BeginTraceCapture(this->device_, command_queue.id());
    for (uint32_t i = 0; i < num_programs; i++) {
        EnqueueProgram(command_queue, programs[i], kNonBlocking);
    }
    EndTraceCapture(this->device_, command_queue.id(), tid);

    // Trace mode execution
    for (auto i = 0; i < num_loops; i++) {
        ScopedTimer timer("Trace loop " + std::to_string(i));
        EnqueueWriteBuffer(command_queue, input, input_data.data(), kNonBlocking);
        EnqueueTrace(command_queue, tid, kNonBlocking);
        EnqueueReadBuffer(command_queue, output, trace_outputs[i].data(), kNonBlocking);
    }
    Finish(command_queue);
    ReleaseTrace(this->device_, tid);
}

} // end namespace basic_tests
