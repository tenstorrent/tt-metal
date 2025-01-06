// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>

#include "multi_command_queue_fixture.hpp"
#include "random_program_fixture.hpp"
#include "dispatch_test_utils.hpp"
#include "detail/tt_metal.hpp"
#include "tt_metal/common/env_lib.hpp"
#include "gtest/gtest.h"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/program/program.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/common/scoped_timer.hpp"
#include "tt_metal/host_api.hpp"

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

Program create_simple_unary_program(Buffer& input, Buffer& output) {
    Program program = CreateProgram();
    Device* device = input.device();
    CoreCoord worker = {0, 0};
    auto reader_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary.cpp",
        worker,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        worker,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto sfpu_kernel = CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        worker,
        ComputeConfig{
            .math_approx_mode = true,
            .compile_args = {1, 1},
            .defines = {{"SFPU_OP_EXP_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}}});

    CircularBufferConfig input_cb_config = CircularBufferConfig(2048, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
                                               .set_page_size(tt::CBIndex::c_0, 2048);

    CoreRange core_range({0, 0});
    CreateCircularBuffer(program, core_range, input_cb_config);
    std::shared_ptr<RuntimeArgs> writer_runtime_args = std::make_shared<RuntimeArgs>();
    std::shared_ptr<RuntimeArgs> reader_runtime_args = std::make_shared<RuntimeArgs>();

    *writer_runtime_args = {
        &output,
        (uint32_t)0,
        output.num_pages()
    };

    *reader_runtime_args = {
        &input,
        (uint32_t)0,
        input.num_pages()
    };

    SetRuntimeArgs(device, detail::GetKernel(program, writer_kernel), worker, writer_runtime_args);
    SetRuntimeArgs(device, detail::GetKernel(program, reader_kernel), worker, reader_runtime_args);

    CircularBufferConfig output_cb_config = CircularBufferConfig(2048, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
                                                .set_page_size(tt::CBIndex::c_16, 2048);

    CreateCircularBuffer(program, core_range, output_cb_config);
    return program;
}

// All basic trace tests just assert that the replayed result exactly matches
// the eager mode results
namespace basic_tests {

constexpr bool kBlocking = true;
constexpr bool kNonBlocking = false;
vector<bool> blocking_flags = {kBlocking, kNonBlocking};

TEST_F(MultiCommandQueueSingleDeviceTraceFixture, TensixEnqueueOneProgramTrace) {
    CreateDevice(2048);
    auto input = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);
    auto output = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);

    CommandQueue& command_queue = this->device_->command_queue(0);
    CommandQueue& data_movement_queue = this->device_->command_queue(1);

    Program simple_program = create_simple_unary_program(*input, *output);
    vector<uint32_t> input_data(input->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    // Eager mode
    vector<uint32_t> eager_output_data;
    eager_output_data.resize(input_data.size());

    EnqueueWriteBuffer(data_movement_queue, *input, input_data.data(), true);
    EnqueueProgram(command_queue, simple_program, true);
    EnqueueReadBuffer(data_movement_queue, output, eager_output_data.data(), true);

    // Trace mode
    vector<uint32_t> trace_output_data;
    trace_output_data.resize(input_data.size());

    EnqueueWriteBuffer(data_movement_queue, *input, input_data.data(), true);

    uint32_t tid = BeginTraceCapture(this->device_, command_queue.id());
    EnqueueProgram(command_queue, simple_program, false);
    EndTraceCapture(this->device_, command_queue.id(), tid);

    EnqueueTrace(command_queue, tid, true);
    EnqueueReadBuffer(data_movement_queue, *output, trace_output_data.data(), true);
    EXPECT_TRUE(eager_output_data == trace_output_data);

    // Done
    Finish(command_queue);
    ReleaseTrace(this->device_, tid);
}

TEST_F(MultiCommandQueueSingleDeviceTraceFixture, TensixEnqueueOneProgramTraceLoops) {
    CreateDevice(4096);
    auto input = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);
    auto output = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);

    CommandQueue& command_queue = this->device_->command_queue(0);
    CommandQueue& data_movement_queue = this->device_->command_queue(1);

    Program simple_program = create_simple_unary_program(*input, *output);
    vector<uint32_t> input_data(input->size() / sizeof(uint32_t), 0);
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
        EnqueueWriteBuffer(data_movement_queue, *input, input_data.data(), true);

        if (not trace_captured) {
            trace_id = BeginTraceCapture(this->device_, command_queue.id());
            EnqueueProgram(command_queue, simple_program, false);
            EndTraceCapture(this->device_, command_queue.id(), trace_id);
            trace_captured = true;
        }

        EnqueueTrace(command_queue, trace_id, false);
        EnqueueReadBuffer(data_movement_queue, *output, trace_outputs[i].data(), true);

        // Expect same output across all loops
        EXPECT_TRUE(trace_outputs[i] == trace_outputs[0]);
    }

    // Done
    Finish(command_queue);
    ReleaseTrace(this->device_, trace_id);
}

TEST_F(MultiCommandQueueSingleDeviceTraceFixture, TensixEnqueueOneProgramTraceBenchmark) {
    CreateDevice(6144);
    auto input = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);
    auto output = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);

    constexpr bool kBlocking = true;
    constexpr bool kNonBlocking = false;
    vector<bool> blocking_flags = {kBlocking, kNonBlocking};

    // Single Q for data and commands
    // Keep this queue in passthrough mode for now
    CommandQueue& command_queue = this->device_->command_queue(0);

    auto simple_program = create_simple_unary_program(*input, *output);
    vector<uint32_t> input_data(input->size() / sizeof(uint32_t), 0);
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
    EnqueueWriteBuffer(command_queue, *input, input_data.data(), kBlocking);
    EnqueueProgram(command_queue, simple_program, kBlocking);
    EnqueueReadBuffer(command_queue, *output, expected_output_data.data(), kBlocking);
    Finish(command_queue);

    for (bool blocking : blocking_flags) {
        std::string mode = blocking ? "Eager-B" : "Eager-NB";
        for (auto i = 0; i < num_loops; i++) {
            tt::ScopedTimer timer(mode + " loop " + std::to_string(i));
            EnqueueWriteBuffer(command_queue, *input, input_data.data(), blocking);
            EnqueueProgram(command_queue, simple_program, blocking);
            EnqueueReadBuffer(command_queue, *output, eager_output_data.data(), blocking);
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
        EnqueueWriteBuffer(command_queue, *input, input_data.data(), kNonBlocking);
        EnqueueTrace(command_queue, tid, kNonBlocking);
        EnqueueReadBuffer(command_queue, *output, trace_outputs[i].data(), kNonBlocking);
    }
    Finish(command_queue);

    // Expect same output across all loops
    for (auto i = 0; i < num_loops; i++) {
        EXPECT_TRUE(trace_outputs[i] == trace_outputs[0]);
    }
    ReleaseTrace(this->device_, tid);
}

TEST_F(CommandQueueTraceFixture, TensixInstantiateTraceSanity) {
    CreateDevice(2048);
    CommandQueue& command_queue = this->device_->command_queue();

    auto input = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);
    vector<uint32_t> input_data(input->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }
    auto output = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);
    auto simple_program = std::make_shared<Program>(create_simple_unary_program(*input, *output));
    EnqueueProgram(command_queue, *simple_program, true);
    uint32_t tid = BeginTraceCapture(this->device_, command_queue.id());
    EnqueueProgram(command_queue, *simple_program, kNonBlocking);
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

TEST_F(CommandQueueTraceFixture, TensixEnqueueProgramTraceCapture) {
    CreateDevice(2048);
    auto input = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);
    auto output = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);

    CommandQueue& command_queue = this->device_->command_queue();

    Program simple_program = create_simple_unary_program(*input, *output);
    vector<uint32_t> input_data(input->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    vector<uint32_t> eager_output_data;
    eager_output_data.resize(input_data.size());
    vector<uint32_t> trace_output_data;
    trace_output_data.resize(input_data.size());

    EnqueueWriteBuffer(command_queue, *input, input_data.data(), true);
    EnqueueProgram(command_queue, simple_program, true);
    EnqueueReadBuffer(command_queue, *output, eager_output_data.data(), true);

    EnqueueWriteBuffer(command_queue, *input, input_data.data(), true);

    uint32_t tid = BeginTraceCapture(this->device_, command_queue.id());
    EnqueueProgram(command_queue, simple_program, false);
    EndTraceCapture(this->device_, command_queue.id(), tid);
    // Create and Enqueue a Program with a live trace to ensure that a warning is generated
    auto input_temp = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);
    auto output_temp = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);
    Program simple_program_temp = create_simple_unary_program(*input_temp, *output_temp);
    EnqueueProgram(command_queue, simple_program_temp, true);
    // Run trace that can clobber the temporary buffers created above
    EnqueueProgram(command_queue, simple_program, false);
    EnqueueTrace(command_queue, tid, true);
    EnqueueReadBuffer(command_queue, *output, trace_output_data.data(), true);
    EXPECT_TRUE(eager_output_data == trace_output_data);

    // Done
    Finish(command_queue);
    ReleaseTrace(this->device_, tid);
}

TEST_F(CommandQueueTraceFixture, TensixEnqueueProgramDeviceCapture) {
    CreateDevice(2048);
    auto input = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);
    auto output = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);

    CommandQueue& command_queue = this->device_->command_queue();

    vector<uint32_t> input_data(input->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    vector<uint32_t> eager_output_data;
    eager_output_data.resize(input_data.size());
    vector<uint32_t> trace_output_data;
    trace_output_data.resize(input_data.size());

    bool has_eager = true;
    std::shared_ptr<Program> simple_program;
    // EAGER MODE EXECUTION
    if (has_eager) {
        simple_program = std::make_shared<Program>(create_simple_unary_program(*input, *output));
        EnqueueWriteBuffer(command_queue, *input, input_data.data(), true);
        EnqueueProgram(command_queue, *simple_program, true);
        EnqueueReadBuffer(command_queue, *output, eager_output_data.data(), true);
    }

    // THIS->DEVICE_ CAPTURE AND REPLAY MODE
    bool has_trace = false;
    uint32_t tid = 0;
    for (int i = 0; i < 1; i++) {
        EnqueueWriteBuffer(command_queue, *input, input_data.data(), true);

        if (!has_trace) {
            // Program must be cached first
            tid = BeginTraceCapture(this->device_, command_queue.id());
            EnqueueProgram(command_queue, *simple_program, false);
            EndTraceCapture(this->device_, command_queue.id(), tid);
            has_trace = true;
        }
        ReplayTrace(this->device_, command_queue.id(), tid, true);

        EnqueueReadBuffer(command_queue, *output, trace_output_data.data(), true);
        if (has_eager) {
            EXPECT_TRUE(eager_output_data == trace_output_data);
        }
    }

    // Done
    Finish(command_queue);
    ReleaseTrace(this->device_, tid);
}

TEST_F(CommandQueueTraceFixture, TensixEnqueueTwoProgramTrace) {
    CreateDevice(6144);
    // Get command queue from device for this test, since its running in async mode
    CommandQueue& command_queue = this->device_->command_queue();

    auto input = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);
    auto interm = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);
    auto output = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);

    Program op0 = create_simple_unary_program(*input, *interm);
    Program op1 = create_simple_unary_program(*interm, *output);
    vector<uint32_t> input_data(input->size() / sizeof(uint32_t), 0);
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
    EnqueueWriteBuffer(command_queue, *input, input_data.data(), kBlocking);
    EnqueueProgram(command_queue, op0, kBlocking);
    EnqueueProgram(command_queue, op1, kBlocking);
    EnqueueReadBuffer(command_queue, *output, expected_output_data.data(), kBlocking);
    Finish(command_queue);

    for (bool blocking : blocking_flags) {
        std::string mode = blocking ? "Eager-B" : "Eager-NB";
        for (auto i = 0; i < num_loops; i++) {
            ScopedTimer timer(mode + " loop " + std::to_string(i));
            EnqueueWriteBuffer(command_queue, *input, input_data.data(), blocking);
            EnqueueProgram(command_queue, op0, blocking);
            EnqueueProgram(command_queue, op1, blocking);
            EnqueueReadBuffer(command_queue, *output, eager_output_data.data(), blocking);
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
        EnqueueWriteBuffer(command_queue, *input, input_data.data(), kNonBlocking);
        EnqueueTrace(command_queue, tid, kNonBlocking);
        EnqueueReadBuffer(command_queue, *output, trace_outputs[i].data(), kNonBlocking);
    }
    Finish(command_queue);
    ReleaseTrace(this->device_, tid);

    // Expect same output across all loops
    for (auto i = 0; i < num_loops; i++) {
        EXPECT_TRUE(trace_outputs[i] == trace_outputs[0]);
    }
}

TEST_F(CommandQueueTraceFixture, TensixEnqueueMultiProgramTraceBenchmark) {
    CreateDevice(6144);
    CommandQueue& command_queue = this->device_->command_queue();

    auto input = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);
    auto output = Buffer::create(this->device_, 2048, 2048, BufferType::DRAM);

    uint32_t num_loops = parse_env<int>("TT_METAL_TRACE_LOOPS", 4);
    uint32_t num_programs = parse_env<int>("TT_METAL_TRACE_PROGRAMS", 4);
    vector<std::shared_ptr<Buffer>> interm_buffers;
    vector<Program> programs;

    vector<uint32_t> input_data(input->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    for (int i = 0; i < num_programs; i++) {
        interm_buffers.push_back(Buffer::create(this->device_, 2048, 2048, BufferType::DRAM));
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

}  // end namespace basic_tests

TEST_F(RandomProgramTraceFixture, TensixTestSimpleProgramsTrace) {
    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        this->programs[i] = CreateProgram();
        Program& program = this->programs[i];
        this->create_kernel(program, CoreType::WORKER, true);
        EnqueueProgram(this->device_->command_queue(), program, false);
    }

    const uint32_t trace_id = this->trace_programs();

    Finish(this->device_->command_queue());
    ReleaseTrace(this->device_, trace_id);
}

TEST_F(RandomProgramTraceFixture, ActiveEthTestSimpleProgramsTrace) {
    if (!does_device_have_active_eth_cores(this->device_)) {
        GTEST_SKIP() << "Skipping test because device " << this->device_->id()
                     << " does not have any active ethernet cores";
    }

    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        this->programs[i] = CreateProgram();
        Program& program = this->programs[i];
        this->create_kernel(program, CoreType::ETH, true);
        EnqueueProgram(this->device_->command_queue(), program, false);
    }

    const uint32_t trace_id = this->trace_programs();

    Finish(this->device_->command_queue());
    ReleaseTrace(this->device_, trace_id);
}

TEST_F(RandomProgramTraceFixture, TensixActiveEthTestSimpleProgramsTrace) {
    if (!does_device_have_active_eth_cores(this->device_)) {
        GTEST_SKIP() << "Skipping test because device " << this->device_->id()
                     << " does not have any active ethernet cores";
    }

    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        this->programs[i] = CreateProgram();
        Program& program = this->programs[i];

        bool eth_kernel_added_to_program = false;
        if (rand() % 2 == 0) {
            this->create_kernel(program, CoreType::ETH, true);
            eth_kernel_added_to_program = true;
        }
        if (rand() % 2 == 0 || !eth_kernel_added_to_program) {
            this->create_kernel(program, CoreType::WORKER, true);
        }

        EnqueueProgram(this->device_->command_queue(), program, false);
    }

    const uint32_t trace_id = this->trace_programs();

    Finish(this->device_->command_queue());
    ReleaseTrace(this->device_, trace_id);
}

TEST_F(RandomProgramTraceFixture, TensixTestProgramsTrace) {
    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        this->programs[i] = CreateProgram();
        Program& program = this->programs[i];
        this->create_kernel(program, CoreType::WORKER);
        EnqueueProgram(this->device_->command_queue(), program, false);
    }

    const uint32_t trace_id = this->trace_programs();

    Finish(this->device_->command_queue());
    ReleaseTrace(this->device_, trace_id);
}

TEST_F(RandomProgramTraceFixture, ActiveEthTestProgramsTrace) {
    if (!does_device_have_active_eth_cores(this->device_)) {
        GTEST_SKIP() << "Skipping test because device " << this->device_->id()
                     << " does not have any active ethernet cores";
    }

    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        this->programs[i] = CreateProgram();
        Program& program = this->programs[i];
        // Large eth kernels currently don't fit in the ring buffer, so we're reducing the max number of RTAs
        // and the max kernel size to ensure that the kernel can fit in the ring buffer
        KernelProperties kernel_properties;
        kernel_properties.max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES / 2;
        kernel_properties.max_num_rt_args = MAX_NUM_RUNTIME_ARGS / 4;
        this->create_kernel(program, CoreType::ETH, false, kernel_properties);
        EnqueueProgram(this->device_->command_queue(), program, false);
    }

    const uint32_t trace_id = this->trace_programs();

    Finish(this->device_->command_queue());
    ReleaseTrace(this->device_, trace_id);
}

TEST_F(RandomProgramTraceFixture, TensixActiveEthTestProgramsTrace) {
    if (!does_device_have_active_eth_cores(this->device_)) {
        GTEST_SKIP() << "Skipping test because device " << this->device_->id()
                     << " does not have any active ethernet cores";
    }

    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        this->programs[i] = CreateProgram();
        Program& program = this->programs[i];

        bool eth_kernel_added_to_program = false;
        if (rand() % 2 == 0) {
            // Large eth kernels currently don't fit in the ring buffer, so we're reducing the max number of RTAs
            // and the max kernel size to ensure that the kernel can fit in the ring buffer
            KernelProperties kernel_properties;
            kernel_properties.max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES / 2;
            kernel_properties.max_num_rt_args = MAX_NUM_RUNTIME_ARGS / 4;
            kernel_properties.max_num_sems = MAX_NUM_SEMS / 2;
            this->create_kernel(program, CoreType::ETH, false, kernel_properties);
            eth_kernel_added_to_program = true;
        }
        if (rand() % 2 == 0 || !eth_kernel_added_to_program) {
            KernelProperties kernel_properties;
            kernel_properties.max_num_sems = MAX_NUM_SEMS / 2;
            this->create_kernel(program, CoreType::WORKER, false, kernel_properties);
        }
        program.set_runtime_id(i);

        EnqueueProgram(this->device_->command_queue(), program, false);
    }

    const uint32_t trace_id = this->trace_programs();

    Finish(this->device_->command_queue());
    ReleaseTrace(this->device_, trace_id);
}

TEST_F(RandomProgramTraceFixture, TensixTestAlternatingLargeAndSmallProgramsTrace) {
    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        this->programs[i] = CreateProgram();
        Program& program = this->programs[i];

        KernelProperties kernel_properties;
        if (i % 2 == 0) {
            kernel_properties = this->get_large_kernel_properties();
        } else {
            kernel_properties = this->get_small_kernel_properties();
        }

        this->create_kernel(program, CoreType::WORKER, false, kernel_properties);
        EnqueueProgram(this->device_->command_queue(), program, false);
    }

    const uint32_t trace_id = this->trace_programs();

    Finish(this->device_->command_queue());
    ReleaseTrace(this->device_, trace_id);
}

TEST_F(RandomProgramTraceFixture, TensixTestLargeProgramFollowedBySmallProgramsTrace) {
    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        this->programs[i] = CreateProgram();
        Program& program = this->programs[i];

        KernelProperties kernel_properties;
        if (i == 0) {
            kernel_properties = this->get_large_kernel_properties();
        } else {
            kernel_properties = this->get_small_kernel_properties();
        }

        this->create_kernel(program, CoreType::WORKER, false, kernel_properties);
        EnqueueProgram(this->device_->command_queue(), program, false);
    }

    const uint32_t trace_id = this->trace_programs();

    Finish(this->device_->command_queue());
    ReleaseTrace(this->device_, trace_id);
}

TEST_F(RandomProgramTraceFixture, TensixTestLargeProgramInBetweenFiveSmallProgramsTrace) {
    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        this->programs[i] = CreateProgram();
        Program& program = this->programs[i];

        KernelProperties kernel_properties;
        if (i % 6 == 0) {
            kernel_properties = this->get_large_kernel_properties();
        } else {
            kernel_properties = this->get_small_kernel_properties();
        }

        this->create_kernel(program, CoreType::WORKER, false, kernel_properties);
        EnqueueProgram(this->device_->command_queue(), program, false);
    }

    const uint32_t trace_id = this->trace_programs();

    Finish(this->device_->command_queue());
    ReleaseTrace(this->device_, trace_id);
}

TEST_F(RandomProgramTraceFixture, TensixTestProgramsTraceAndNoTrace) {
    std::vector<uint32_t> trace_ids;
    std::unordered_map<uint64_t, uint32_t> program_ids_to_trace_ids;
    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        this->programs[i] = CreateProgram();
        Program& program = this->programs[i];
        this->create_kernel(program, CoreType::WORKER);

        const bool use_trace = (rand() % 2) == 0;
        if (use_trace) {
            EnqueueProgram(this->device_->command_queue(), program, false);
            const uint32_t trace_id = BeginTraceCapture(this->device_, this->device_->command_queue().id());
            EnqueueProgram(this->device_->command_queue(), program, false);
            EndTraceCapture(this->device_, this->device_->command_queue().id(), trace_id);
            trace_ids.push_back(trace_id);
            program_ids_to_trace_ids.emplace(program.get_id(), trace_id);
        }
    }
    Finish(this->device_->command_queue());

    for (Program& program : this->programs) {
        const uint64_t program_id = program.get_id();
        const bool use_trace = program_ids_to_trace_ids.contains(program_id);
        if (use_trace) {
            const uint32_t trace_id = program_ids_to_trace_ids[program_id];
            EnqueueTrace(this->device_->command_queue(), trace_id, false);
        }
        EnqueueProgram(this->device_->command_queue(), program, false);
    }

    Finish(this->device_->command_queue());
    for (const uint32_t trace_id : trace_ids) {
        ReleaseTrace(this->device_, trace_id);
    }
}

TEST_F(RandomProgramTraceFixture, ActiveEthTestProgramsTraceAndNoTrace) {
    if (!does_device_have_active_eth_cores(this->device_)) {
        GTEST_SKIP() << "Skipping test because device " << this->device_->id()
                     << " does not have any active ethernet cores";
    }

    std::vector<uint32_t> trace_ids;
    std::unordered_map<uint64_t, uint32_t> program_ids_to_trace_ids;
    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        this->programs[i] = CreateProgram();
        Program& program = this->programs[i];
        // Large eth kernels currently don't fit in the ring buffer, so we're reducing the max number of RTAs
        // and the max kernel size to ensure that the kernel can fit in the ring buffer
        KernelProperties kernel_properties;
        kernel_properties.max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES / 2;
        kernel_properties.max_num_rt_args = MAX_NUM_RUNTIME_ARGS / 4;
        this->create_kernel(program, CoreType::ETH, false, kernel_properties);

        const bool use_trace = (rand() % 2) == 0;
        if (use_trace) {
            EnqueueProgram(this->device_->command_queue(), program, false);
            const uint32_t trace_id = BeginTraceCapture(this->device_, this->device_->command_queue().id());
            EnqueueProgram(this->device_->command_queue(), program, false);
            EndTraceCapture(this->device_, this->device_->command_queue().id(), trace_id);
            trace_ids.push_back(trace_id);
            program_ids_to_trace_ids.emplace(program.get_id(), trace_id);
        }
    }
    Finish(this->device_->command_queue());

    for (Program& program : this->programs) {
        const uint64_t program_id = program.get_id();
        const bool use_trace = program_ids_to_trace_ids.contains(program_id);
        if (use_trace) {
            const uint32_t trace_id = program_ids_to_trace_ids[program_id];
            EnqueueTrace(this->device_->command_queue(), trace_id, false);
        }
        EnqueueProgram(this->device_->command_queue(), program, false);
    }

    Finish(this->device_->command_queue());
    for (const uint32_t trace_id : trace_ids) {
        ReleaseTrace(this->device_, trace_id);
    }
}

TEST_F(RandomProgramTraceFixture, TensixActiveEthTestProgramsTraceAndNoTrace) {
    if (!does_device_have_active_eth_cores(this->device_)) {
        GTEST_SKIP() << "Skipping test because device " << this->device_->id()
                     << " does not have any active ethernet cores";
    }

    std::vector<uint32_t> trace_ids;
    std::unordered_map<uint64_t, uint32_t> program_ids_to_trace_ids;
    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        this->programs[i] = CreateProgram();
        Program& program = this->programs[i];

        bool eth_kernel_added_to_program = false;
        if (rand() % 2 == 0) {
            // Large eth kernels currently don't fit in the ring buffer, so we're reducing the max number of RTAs
            // and the max kernel size to ensure that the kernel can fit in the ring buffer
            KernelProperties kernel_properties;
            kernel_properties.max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES / 2;
            kernel_properties.max_num_rt_args = MAX_NUM_RUNTIME_ARGS / 4;
            kernel_properties.max_num_sems = MAX_NUM_SEMS / 2;
            this->create_kernel(program, CoreType::ETH, false, kernel_properties);
            eth_kernel_added_to_program = true;
        }
        if (rand() % 2 == 0 || !eth_kernel_added_to_program) {
            KernelProperties kernel_properties;
            kernel_properties.max_num_sems = MAX_NUM_SEMS / 2;
            this->create_kernel(program, CoreType::WORKER, false, kernel_properties);
        }

        const bool use_trace = (rand() % 2) == 0;
        if (use_trace) {
            EnqueueProgram(this->device_->command_queue(), program, false);
            const uint32_t trace_id = BeginTraceCapture(this->device_, this->device_->command_queue().id());
            EnqueueProgram(this->device_->command_queue(), program, false);
            EndTraceCapture(this->device_, this->device_->command_queue().id(), trace_id);
            trace_ids.push_back(trace_id);
            program_ids_to_trace_ids.emplace(program.get_id(), trace_id);
        }
    }
    Finish(this->device_->command_queue());

    for (Program& program : this->programs) {
        const uint64_t program_id = program.get_id();
        const bool use_trace = program_ids_to_trace_ids.contains(program_id);
        if (use_trace) {
            const uint32_t trace_id = program_ids_to_trace_ids[program_id];
            EnqueueTrace(this->device_->command_queue(), trace_id, false);
        }
        EnqueueProgram(this->device_->command_queue(), program, false);
    }

    Finish(this->device_->command_queue());
    for (const uint32_t trace_id : trace_ids) {
        ReleaseTrace(this->device_, trace_id);
    }
}
