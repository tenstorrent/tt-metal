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

Program create_simple_unary_program(Buffer& input, Buffer& output) {
    Program program = CreateProgram();
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

    SetRuntimeArgs(device, detail::GetKernel(program, writer_kernel), worker, writer_runtime_args);
    SetRuntimeArgs(device, detail::GetKernel(program, reader_kernel), worker, reader_runtime_args);

    CircularBufferConfig output_cb_config = CircularBufferConfig(2048, {{16, tt::DataFormat::Float16_b}})
            .set_page_size(16, 2048);

    CreateCircularBuffer(program, core_range, output_cb_config);
    return program;
}

// All basic trace tests just assert that the replayed result exactly matches
// the eager mode results
namespace basic_tests {

constexpr bool kBlocking = true;
constexpr bool kNonBlocking = false;
vector<bool> blocking_flags = {kBlocking, kNonBlocking};

TEST_F(CommandQueueFixture, TraceInstanceManagement) {
    CommandQueue& cq = this->device_->command_queue();
    vector<uint64_t> trace_size = {32*1024, 32};
    vector<uint64_t> page_size = {DeviceCommand::PROGRAM_PAGE_SIZE, 32};
    vector<uint64_t> buf_size_per_bank;

    for (int i=0; i<trace_size.size(); i++) {
        int banks = cq.device()->num_banks(BufferType::DRAM);
        int pages = trace_size.at(i) / page_size.at(i);
        int pages_per_bank = pages / banks + (pages % banks ? 1 : 0);
        buf_size_per_bank.push_back(pages_per_bank * page_size.at(i));
    }

    auto mem_idle = cq.device()->get_memory_allocation_statistics(BufferType::DRAM);
    log_debug(LogTest, "DRAM usage before trace buffer allocation: {}, {}, {}",
        mem_idle.total_allocatable_size_bytes,
        mem_idle.total_free_bytes,
        mem_idle.total_allocated_bytes);

    // Add instances scope, trace buffers go out of scope yet remain cached in memory
    {
        TraceBuffer trace_buffer0 = {{}, std::make_shared<Buffer>(
            cq.device(), trace_size.at(0), page_size.at(0), BufferType::DRAM, TensorMemoryLayout::INTERLEAVED)};
        TraceBuffer trace_buffer1 = {{}, std::make_shared<Buffer>(
            cq.device(), trace_size.at(1), page_size.at(1), BufferType::DRAM, TensorMemoryLayout::INTERLEAVED)};
        auto mem_multi_trace = cq.device()->get_memory_allocation_statistics(BufferType::DRAM);
        log_debug(
            LogTest,
            "DRAM usage post trace buffer allocation: {}, {}, {}",
            mem_multi_trace.total_allocatable_size_bytes,
            mem_multi_trace.total_free_bytes,
            mem_multi_trace.total_allocated_bytes);

        // Cache the trace buffer in memory via instance pinning calls
        Trace::add_instance(0, trace_buffer0);
        Trace::add_instance(1, trace_buffer1);
    }

    // Some user interaction with traces, unimportant... check that traces are still cached
    auto mem_multi_trace = cq.device()->get_memory_allocation_statistics(BufferType::DRAM);
    EXPECT_EQ(mem_idle.total_allocated_bytes, mem_multi_trace.total_allocated_bytes - buf_size_per_bank.at(0) - buf_size_per_bank.at(1));
    EXPECT_EQ(mem_idle.total_free_bytes, mem_multi_trace.total_free_bytes + buf_size_per_bank.at(0) + buf_size_per_bank.at(1));

    // Release instances scope, trace buffers remain cached in memory until released by user
    {
        ReleaseTrace(1);
        auto mem_release_one = cq.device()->get_memory_allocation_statistics(BufferType::DRAM);
        EXPECT_EQ(mem_idle.total_allocated_bytes, mem_release_one.total_allocated_bytes - buf_size_per_bank.at(0));
        EXPECT_EQ(mem_idle.total_free_bytes, mem_release_one.total_free_bytes + buf_size_per_bank.at(0));

        ReleaseTrace(0);
        auto mem_release_two = cq.device()->get_memory_allocation_statistics(BufferType::DRAM);
        EXPECT_EQ(mem_idle.total_allocatable_size_bytes, mem_release_two.total_allocatable_size_bytes);
        EXPECT_EQ(mem_idle.total_free_bytes, mem_release_two.total_free_bytes);
        EXPECT_EQ(mem_idle.total_allocated_bytes, mem_release_two.total_allocated_bytes);
    }

    // Add instances scope, trace buffers go out of scope yet remain cached in memory
    {
        TraceBuffer trace_buffer0 = {{}, std::make_shared<Buffer>(
            cq.device(), trace_size.at(0), page_size.at(0), BufferType::DRAM, TensorMemoryLayout::INTERLEAVED)};
        TraceBuffer trace_buffer1 = {{}, std::make_shared<Buffer>(
            cq.device(), trace_size.at(1), page_size.at(1), BufferType::DRAM, TensorMemoryLayout::INTERLEAVED)};
        auto mem_multi_trace = cq.device()->get_memory_allocation_statistics(BufferType::DRAM);

        // Cache the trace buffer in memory via instance pinning calls
        Trace::add_instance(0, trace_buffer0);
        Trace::add_instance(1, trace_buffer1);
    }

    ReleaseTrace(-1);
    auto mem_release_all = cq.device()->get_memory_allocation_statistics(BufferType::DRAM);
    EXPECT_EQ(mem_idle.total_allocatable_size_bytes, mem_release_all.total_allocatable_size_bytes);
    EXPECT_EQ(mem_idle.total_free_bytes, mem_release_all.total_free_bytes);
    EXPECT_EQ(mem_idle.total_allocated_bytes, mem_release_all.total_allocated_bytes);
}

TEST_F(CommandQueueFixture, InstantiateTraceSanity) {
    CommandQueue& command_queue = this->device_->command_queue();

    Buffer input(this->device_, 2048, 2048, BufferType::DRAM);
    vector<uint32_t> input_data(input.size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    // Capture trace on a trace queue
    Trace trace;
    BeginTrace(trace);
    EnqueueWriteBuffer(trace.queue(), input, input_data.data(), kNonBlocking);
    EnqueueWriteBuffer(trace.queue(), input, input_data.data(), kNonBlocking);
    EndTrace(trace);

    // Instantiate a trace on a device bound command queue
    uint32_t trace_id = InstantiateTrace(trace, command_queue);
    auto trace_inst = Trace::get_instance(trace_id);
    vector<uint32_t> data_fd, data_bd;

    // Backdoor read the trace buffer
    ::detail::ReadFromBuffer(trace_inst.buffer, data_bd);

    // Frontdoor reaad the trace buffer
    data_fd.resize(trace_inst.buffer->size() / sizeof(uint32_t));
    EnqueueReadBuffer(command_queue, trace_inst.buffer, data_fd.data(), kBlocking);
    EXPECT_EQ(data_fd, data_bd);

    // Check for content correctness in the trace buffer
    // The following commands are expected based on the trace capture
    CQPrefetchCmd* p_cmd;
    CQDispatchCmd* d_cmd;
    size_t p_size = (sizeof(CQPrefetchCmd) / sizeof(uint32_t));
    size_t d_size = (sizeof(CQDispatchCmd) / sizeof(uint32_t));
    size_t offset = 0;
    p_cmd = (CQPrefetchCmd*)(data_fd.data() + offset);
    offset += p_size;
    EXPECT_EQ(p_cmd->base.cmd_id, CQ_PREFETCH_CMD_RELAY_INLINE);

    d_cmd = (CQDispatchCmd*)(data_fd.data() + offset);
    offset += d_size;
    EXPECT_EQ(d_cmd->base.cmd_id, CQ_DISPATCH_CMD_WAIT);

    p_cmd = (CQPrefetchCmd*)(data_fd.data() + offset);
    offset += p_size;
    EXPECT_EQ(p_cmd->base.cmd_id, CQ_PREFETCH_CMD_RELAY_INLINE);

    d_cmd = (CQDispatchCmd*)(data_fd.data() + offset);
    offset += d_size;
    EXPECT_EQ(d_cmd->base.cmd_id, CQ_DISPATCH_CMD_WRITE_PAGED);
    EXPECT_EQ(d_cmd->write_paged.is_dram, true);
    EXPECT_EQ(d_cmd->write_paged.page_size, 2048);

    log_trace(LogTest, "Trace buffer content: {}", data_fd);
    ReleaseTrace(trace_id);
}

TEST_F(CommandQueueFixture, EnqueueTraceWriteBufferCommand) {
    CommandQueue& command_queue = this->device_->command_queue();

    Buffer input(this->device_, 2048, 2048, BufferType::DRAM);
    vector<uint32_t> input_first(input.size() / sizeof(uint32_t), 0xfaceface);
    vector<uint32_t> input_last(input.size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_last.size(); i++) {
        input_last[i] = i;
    }

    // TRACE CAPTURE & INSTANTIATE MODE
    // Capture trace on a trace queue
    Trace trace;
    BeginTrace(trace);
    EnqueueWriteBuffer(trace.queue(), input, input_first.data(), kNonBlocking);
    EnqueueWriteBuffer(trace.queue(), input, input_last.data(), kNonBlocking);
    EndTrace(trace);

    // Instantiate a trace on a device bound command queue
    uint32_t trace_id = InstantiateTrace(trace, command_queue);

    // Repeat traces, check that last write occurs correctly during each iteration
    vector<uint32_t> readback(input.size() / sizeof(uint32_t), 0);
    for (int i = 0; i < 10; i++) {
        EnqueueTrace(command_queue, trace_id, true);
        EnqueueReadBuffer(command_queue, input, readback.data(), kBlocking);
        EXPECT_EQ(input_last, readback);
    }

    ReleaseTrace(trace_id);
}

TEST_F(CommandQueueFixture, EnqueueTraceWriteBufferCommandViaDevice) {
    CommandQueue& command_queue = this->device_->command_queue();

    Buffer input(this->device_, 2048, 2048, BufferType::DRAM);
    vector<uint32_t> input_first(input.size() / sizeof(uint32_t), 0xfaceface);
    vector<uint32_t> input_last(input.size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_last.size(); i++) {
        input_last[i] = i;
    }

    // DEVICE CAPTURE AND REPLAY MODE
    // Capture trace on a device rather than a trace objet
    detail::BeginTraceCapture(this->device_);
    EnqueueWriteBuffer(command_queue, input, input_first.data(), kNonBlocking);
    EnqueueWriteBuffer(command_queue, input, input_last.data(), kNonBlocking);
    detail::EndTraceCapture(this->device_);

    // Repeat traces, check that last write occurs correctly during each iteration
    vector<uint32_t> readback(input.size() / sizeof(uint32_t), 0);
    for (int i = 0; i < 10; i++) {
        detail::ExecuteLastTrace(this->device_, true);
        EnqueueReadBuffer(command_queue, input, readback.data(), kBlocking);
        EXPECT_EQ(input_last, readback);
    }
}

TEST_F(CommandQueueFixture, EnqueueProgramTraceCapture) {
    Buffer input(this->device_, 2048, 2048, BufferType::DRAM);
    Buffer output(this->device_, 2048, 2048, BufferType::DRAM);

    CommandQueue& command_queue = this->device_->command_queue();

    Program simple_program = create_simple_unary_program(input, output);
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

    // TRACE CAPTURE & INSTANTIATE MODE
    Trace trace;
    EnqueueWriteBuffer(command_queue, input, input_data.data(), true);

    BeginTrace(trace);
    EnqueueProgram(trace.queue(), simple_program, false);
    EndTrace(trace);

    // Instantiate a trace on a device queue
    uint32_t trace_id = InstantiateTrace(trace, command_queue);

    EnqueueTrace(command_queue, trace_id, true);
    EnqueueReadBuffer(command_queue, output, trace_output_data.data(), true);
    EXPECT_TRUE(eager_output_data == trace_output_data);

    // Done
    Finish(command_queue);
}

TEST_F(CommandQueueFixture, EnqueueProgramDeviceCapture) {
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
    // EAGER MODE EXECUTION
    if (has_eager) {
        Program simple_program = create_simple_unary_program(input, output);
        EnqueueWriteBuffer(command_queue, input, input_data.data(), true);
        EnqueueProgram(command_queue, simple_program, true);
        EnqueueReadBuffer(command_queue, output, eager_output_data.data(), true);
    }

    // DEVICE CAPTURE AND REPLAY MODE
    bool has_trace = false;
    for (int i = 0; i < 1; i++) {
        EnqueueWriteBuffer(command_queue, input, input_data.data(), true);

        if (!has_trace) {
            detail::BeginTraceCapture(this->device_);
            EnqueueProgram(command_queue, std::make_shared<Program>(create_simple_unary_program(input, output)), true);
            detail::EndTraceCapture(this->device_);
            has_trace = true;
        }
        detail::ExecuteLastTrace(this->device_, true);

        EnqueueReadBuffer(command_queue, output, trace_output_data.data(), true);
        if (has_eager) EXPECT_TRUE(eager_output_data == trace_output_data);
    }

    // Done
    Finish(command_queue);
}

TEST_F(CommandQueueFixture, EnqueueTwoProgramTrace) {
    // Get command queue from device for this test, since its running in async mode
    CommandQueue& command_queue = this->device_->command_queue();

    Buffer input(this->device_, 2048, 2048, BufferType::DRAM);
    Buffer interm(this->device_, 2048, 2048, BufferType::DRAM);
    Buffer output(this->device_, 2048, 2048, BufferType::DRAM);

    Program op0 = create_simple_unary_program(input, interm);
    Program op1 = create_simple_unary_program(interm, output);
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
    Trace trace;
    CommandQueue& trace_queue = BeginTrace(trace);
    EnqueueProgram(trace_queue, op0, kNonBlocking);
    EnqueueProgram(trace_queue, op1, kNonBlocking);
    EndTrace(trace);

    // Instantiate a trace on a device bound command queue
    uint32_t trace_id = InstantiateTrace(trace, command_queue);

    // Trace mode execution
    for (auto i = 0; i < num_loops; i++) {
        ScopedTimer timer("Trace loop " + std::to_string(i));
        EnqueueWriteBuffer(command_queue, input, input_data.data(), kNonBlocking);
        EnqueueTrace(command_queue, trace_id, kNonBlocking);
        EnqueueReadBuffer(command_queue, output, trace_outputs[i].data(), kNonBlocking);
    }
    Finish(command_queue);
    ReleaseTrace(trace_id);

    // Expect same output across all loops
    for (auto i = 0; i < num_loops; i++) {
        EXPECT_TRUE(trace_outputs[i] == trace_outputs[0]);
    }
}

TEST_F(CommandQueueFixture, EnqueueMultiProgramTraceBenchmark) {
    CommandQueue& command_queue = this->device_->command_queue();

    std::shared_ptr<Buffer> input = std::make_shared<Buffer>(this->device_, 2048, 2048, BufferType::DRAM);
    std::shared_ptr<Buffer> output = std::make_shared<Buffer>(this->device_, 2048, 2048, BufferType::DRAM);

    uint32_t num_loops = parse_env<int>("TT_METAL_TRACE_LOOPS", 4);
    uint32_t num_programs = parse_env<int>("TT_METAL_TRACE_PROGRAMS", 4);
    vector<std::shared_ptr<Buffer>> interm_buffers;
    vector<Program> programs;

    vector<uint32_t> input_data(input->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    for (int i = 0; i < num_programs; i++) {
        interm_buffers.push_back(std::make_shared<Buffer>(this->device_, 2048, 2048, BufferType::DRAM));
        if (i == 0) {
            programs.push_back(create_simple_unary_program(*input, *(interm_buffers[i])));
        } else if (i == (num_programs - 1)) {
            programs.push_back(create_simple_unary_program(*(interm_buffers[i - 1]), *output));
        } else {
            programs.push_back(create_simple_unary_program(*(interm_buffers[i - 1]), *(interm_buffers[i])));
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
    Trace trace;
    CommandQueue& trace_queue = BeginTrace(trace);
    for (uint32_t i = 0; i < num_programs; i++) {
        EnqueueProgram(trace_queue, programs[i], kNonBlocking);
    }
    EndTrace(trace);

    // Instantiate a trace on a device bound command queue
    uint32_t trace_id = InstantiateTrace(trace, command_queue);

    // Trace mode execution
    for (auto i = 0; i < num_loops; i++) {
        ScopedTimer timer("Trace loop " + std::to_string(i));
        EnqueueWriteBuffer(command_queue, input, input_data.data(), kNonBlocking);
        EnqueueTrace(command_queue, trace_id, kNonBlocking);
        EnqueueReadBuffer(command_queue, output, trace_outputs[i].data(), kNonBlocking);
    }
    Finish(command_queue);
    ReleaseTrace(trace_id);
}

} // end namespace basic_tests
