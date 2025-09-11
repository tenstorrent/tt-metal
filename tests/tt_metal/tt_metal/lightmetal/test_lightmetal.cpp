// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Issue #24955: Enable after Light-Metal rearchitecture

// #include <fmt/base.h>
// #include <stddef.h>
// #include <tt-metalium/command_queue.hpp>
// #include <tt-metalium/host_api.hpp>
// #include <tt-logger/tt-logger.hpp>
// #include <tt-metalium/program.hpp>
// #include <algorithm>
// #include <cstdint>
// #include <iostream>
// #include <map>
// #include <memory>
// #include <set>
// #include <string>
// #include <utility>
// #include <variant>
// #include <vector>

// #include <tt-metalium/assert.hpp>
// #include <tt-metalium/buffer.hpp>
// #include <tt-metalium/buffer_types.hpp>
// #include <tt-metalium/circular_buffer_config.hpp>
// #include <tt-metalium/constants.hpp>
// #include <tt-metalium/core_coord.hpp>
// #include <tt-metalium/data_types.hpp>
// #include <tt-metalium/device.hpp>
// #include <tt-metalium/tensor_accessor_args.hpp>
// #include "gtest/gtest.h"
// #include "hostdevcommon/kernel_structs.h"
// #include <tt-metalium/kernel_types.hpp>
// #include "lightmetal/host_api_capture_helpers.hpp"
// #include <tt-metalium/lightmetal_capture_utils.hpp>
// #include "lightmetal_fixture.hpp"
// #include <tt_stl/span.hpp>
// #include <tt-metalium/tt_backend_api_types.hpp>
// #include "tt_metal/test_utils/stimulus.hpp"

// // Access to internal API: ProgramImpl::get_kernel
// #include "impl/program/program_impl.hpp"

// using std::vector;
// using namespace tt;
// using namespace tt::tt_metal;

// namespace tt::tt_metal {
// namespace {

// struct L1Config {
//     uint32_t num_elements = 8 * tt::constants::TILE_HW;
//     uint32_t element_size = 2;
//     uint32_t size_bytes = element_size * num_elements;
//     uint32_t page_size_bytes = tt::constants::TILE_HW * element_size;
//     tt::DataFormat l1_data_format = tt::DataFormat::Float16_b;

//     BufferShardingArgs sharding_args;
// };

// // Inspired heavily from test_sharded_l1_buffer.cpp
// bool l1_buffer_read_write_test(IDevice* device, const L1Config& test_config) {
//     TT_FATAL(device != nullptr, "Device not setup");

//     bool pass = true;
//     CommandQueue& cq = device->command_queue();

//     uint32_t num_loops = 5;
//     std::vector<std::shared_ptr<Buffer>> buffers_vec;

//     for (uint32_t loop_idx = 0; loop_idx < num_loops; loop_idx++) {
//         log_debug(tt::LogTest, "Running loop: {}", loop_idx);

//         auto buffer = Buffer::create(
//             device, test_config.size_bytes, test_config.page_size_bytes, BufferType::L1, test_config.sharding_args);

//         if (loop_idx > 1) {
//             buffers_vec.push_back(buffer);
//         }

//         auto input =
//             tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, test_config.size_bytes / sizeof(uint32_t));

//         vector<uint32_t> output;
//         output.resize(input.size());

//         // Write data to buffer, then read outputs and verify against expected.
//         EnqueueWriteBuffer(cq, *buffer, input.data(), /*blocking=*/true);

//         // This will do read, and verify that readback matches between capture + replay
//         LightMetalCompareToCapture(cq, *buffer, output.data());

//         pass &= (output == input);

//         // For dev/debug go ahead and print the results. Had a replay bug, was seeing wrong data.
//         for (size_t i = 0; i < output.size(); i++) {
//             log_debug(tt::LogMetalTrace, "loop_idx: {} rd_data i: {:3d} => data: {}", loop_idx, i, output[i]);
//         }

//         if (!pass) {
//             if (input.size() != output.size()) {
//                 std::cout << "Different size of input and output, input.size() = " << input.size() << " output.size() "
//                           << output.size() << std::endl;
//             }
//             int smaller_size = std::min<int>(input.size(), output.size());
//             auto entries_per_page = test_config.page_size_bytes / (sizeof(uint32_t));
//             for (int i = 0; i < smaller_size; i++) {
//                 if (input[i] != output[i]) {
//                     std::cout << "mismatch on page: " << i / entries_per_page
//                               << " entry index: " << i % entries_per_page << " with input being " << std::hex
//                               << input[i] << " and output being " << output[i] << std::dec << std::endl;
//                 }
//             }
//         }
//     }

//     // If any Buffers were kept alive for testing, Deallocate them now to exercise that path for capture/replay.
//     if (buffers_vec.size() > 0) {
//         log_info(tt::LogTest, "Explicitly deallocating {} buffers now.", buffers_vec.size());
//         for (auto& buffer : buffers_vec) {
//             DeallocateBuffer(*buffer);
//         }
//     }

//     return pass;
// }

// // Single RISC, no CB's here. Very simple.
// Program create_simple_datamovement_program(
//     const Buffer& input, const Buffer& output, const Buffer& l1_buffer, bool rt_arg_per_core_vec = false) {
//     Program program = Program();  // Verify Program constructor can be used.
//     constexpr CoreCoord core = {0, 0};

//     std::vector<uint32_t> compile_time_args;
//     TensorAccessorArgs(input).append_to(compile_time_args);
//     TensorAccessorArgs(output).append_to(compile_time_args);
//     KernelHandle dram_copy_kernel_id = CreateKernel(
//         program,
//         "tt_metal/programming_examples/loopback/kernels/loopback_dram_copy.cpp",
//         core,
//         DataMovementConfig{
//             .processor = DataMovementProcessor::RISCV_0,
//             .noc = NOC::RISCV_0_default,
//             .compile_args = compile_time_args});

//     // Since all interleaved buffers have size == page_size, they are entirely contained in the first DRAM bank

//     // Handle Runtime Args
//     const std::vector<uint32_t> runtime_args = {
//         l1_buffer.address(), input.address(), output.address(), input.num_pages()};

//     // Very minimal testing/usage of other SetRuntimeArgs API that TTNN uses for ops here, j
//     // just to see it go through the light-metal capture + replay flow.
//     if (rt_arg_per_core_vec) {
//         const std::vector<std::vector<uint32_t>> runtime_args_per_core = {runtime_args};
//         SetRuntimeArgs(program, dram_copy_kernel_id, {core}, runtime_args_per_core);
//     } else {
//         // Note - this interface doesn't take Buffer, just data.
//         SetRuntimeArgs(program, dram_copy_kernel_id, core, runtime_args);
//     }

//     return program;
// }

// // Copied from test_EnqueueTrace.cpp
// Program create_simple_unary_program(Buffer& input, Buffer& output, Buffer* cb_input_buffer = nullptr) {
//     Program program = CreateProgram();
//     CoreCoord worker = {0, 0};
//     auto reader_kernel = CreateKernel(
//         program,
//         "tt_metal/kernels/dataflow/reader_unary.cpp",
//         worker,
//         DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

//     auto writer_kernel = CreateKernel(
//         program,
//         "tt_metal/kernels/dataflow/writer_unary.cpp",
//         worker,
//         DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

//     CreateKernel(
//         program,
//         "tt_metal/kernels/compute/eltwise_sfpu.cpp",
//         worker,
//         ComputeConfig{
//             .math_approx_mode = true,
//             .compile_args = {1, 1},
//             .defines = {{"SFPU_OP_EXP_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}}});

//     CircularBufferConfig input_cb_config = CircularBufferConfig(2048, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
//                                                .set_page_size(tt::CBIndex::c_0, 2048);

//     // For testing dynamic CB for which CB config has a shadow buffer ptr to test.
//     if (cb_input_buffer) {
//         input_cb_config.set_globally_allocated_address(*cb_input_buffer);
//     }

//     CoreRange core_range({0, 0});
//     CreateCircularBuffer(program, core_range, input_cb_config);

//     auto writer_runtime_args = {output.address(), uint32_t(0), output.num_pages()};
//     auto reader_runtime_args = {input.address(), uint32_t(0), input.num_pages()};

//     SetRuntimeArgs(program, writer_kernel, worker, writer_runtime_args);
//     SetRuntimeArgs(program, reader_kernel, worker, reader_runtime_args);

//     CircularBufferConfig output_cb_config = CircularBufferConfig(2048, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
//                                                 .set_page_size(tt::CBIndex::c_16, 2048);

//     CreateCircularBuffer(program, core_range, output_cb_config);
//     return program;
// }

// void write_junk_to_buffer(CommandQueue& command_queue, Buffer& buffer) {
//     vector<uint32_t> dummy_write_data(buffer.size() / sizeof(uint32_t), 0xDEADBEEF);
//     vector<uint32_t> dummy_read_data(buffer.size() / sizeof(uint32_t), 0);
//     EnqueueWriteBuffer(command_queue, buffer, dummy_write_data.data(), true);
//     EnqueueReadBuffer(command_queue, buffer, dummy_read_data.data(), true);
//     for (size_t i = 0; i < dummy_read_data.size(); i++) {
//         log_trace(tt::LogMetalTrace, "i: {:3d} output: {:x} after write+read of dummy data", i, dummy_read_data[i]);
//     }
//     EXPECT_TRUE(dummy_write_data == dummy_read_data);
// }

// // TODO (kmabee) - consider looping over blocking_flags in some/all tests once stable.
// constexpr bool kBlocking = true;
// constexpr bool kNonBlocking = false;
// vector<bool> blocking_flags = {kBlocking, kNonBlocking};

// using LightMetalBasicTest = SingleDeviceLightMetalFixture;

// // Test that create buffer, write, readback, and verify works when traced + replayed.
// TEST_F(LightMetalBasicTest, CreateBufferInterleavedEnqueueWriteRead) {
//     CreateDeviceAndBeginCapture(4096);
//     L1Config test_config;
//     EXPECT_TRUE(l1_buffer_read_write_test(device_, test_config));
// }

// TEST_F(LightMetalBasicTest, CreateBufferHeightShardEnqueueWriteRead) {
//     CreateDeviceAndBeginCapture(4096);
//     L1Config test_config;
//     test_config.sharding_args = BufferShardingArgs(
//         ShardSpecBuffer(
//             CoreRangeSet(std::set<CoreRange>({CoreRange(CoreCoord(0, 0), CoreCoord(0, 1))})),
//             {2 * tt::constants::TILE_HEIGHT, 2 * tt::constants::TILE_WIDTH},
//             ShardOrientation::ROW_MAJOR,
//             {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
//             {4, 2}),
//         TensorMemoryLayout::HEIGHT_SHARDED);
//     EXPECT_TRUE(l1_buffer_read_write_test(device_, test_config));
// }

// TEST_F(LightMetalBasicTest, CreateBufferWidthShardEnqueueWriteRead) {
//     CreateDeviceAndBeginCapture(4096);
//     L1Config test_config;
//     test_config.sharding_args = BufferShardingArgs(
//         ShardSpecBuffer(
//             CoreRangeSet(std::set<CoreRange>({CoreRange(CoreCoord(0, 0), CoreCoord(0, 1))})),
//             {2 * tt::constants::TILE_HEIGHT, 2 * tt::constants::TILE_WIDTH},
//             ShardOrientation::ROW_MAJOR,
//             {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
//             {2, 4}),
//         TensorMemoryLayout::WIDTH_SHARDED);
//     EXPECT_TRUE(l1_buffer_read_write_test(device_, test_config));
// }

// TEST_F(LightMetalBasicTest, CreateBufferNdShardEnqueueWriteRead) {
//     CreateDeviceAndBeginCapture(4096);
//     L1Config test_config;
//     test_config.sharding_args = BufferShardingArgs(BufferDistributionSpec(
//         Shape({2, 2, 2}),
//         Shape({1, 1, 1}),
//         CoreRangeSet(std::set<CoreRange>({CoreRange(CoreCoord(0, 0), CoreCoord(0, 1))})),
//         ShardOrientation::ROW_MAJOR));
//     EXPECT_TRUE(l1_buffer_read_write_test(device_, test_config));
// }

// // Test with large number of buffers, ensure Buffers are deallocated when going out of scope during replay
// TEST_F(LightMetalBasicTest, BufferDeallocationsScope) {
//     CreateDeviceAndBeginCapture(4096);

//     const uint32_t num_buffers = 100;
//     for (uint32_t i = 0; i < num_buffers; i++) {
//         const size_t size_bytes = 1024 * 256;  // 256 KB
//         auto buf = Buffer::create(device_, size_bytes, size_bytes, BufferType::L1);
//     }
// }

// // Test that we can create buffers and deallocate them, and new buffers don't collide in capture time object map.
// TEST_F(LightMetalBasicTest, CreateBufferAndDeallocate) {
//     CreateDeviceAndBeginCapture(4096);

//     const uint32_t num_buffers = 5;
//     for (uint32_t i = 0; i < num_buffers; i++) {
//         auto buf = Buffer::create(device_, 64, 64, BufferType::DRAM);
//         DeallocateBuffer(*buf);
//     }
// }

// void SingleRISCDataMovement_test(tt::tt_metal::IDevice* device, bool rt_arg_per_core_vec) {
//     uint32_t size_bytes = 32 * 32 * 2;
//     uint32_t page_size = 32 * 32 * 2;

//     // For extra coverage, use Buffer::create (now support for light metal capture/replay)
//     auto input = Buffer::create(device, size_bytes, page_size, BufferType::DRAM);
//     auto output = Buffer::create(device, size_bytes, page_size, BufferType::DRAM);
//     auto l1_buffer = Buffer::create(device, size_bytes, page_size, BufferType::L1);

//     log_debug(
//         tt::LogTest,
//         "Created 3 Buffers. input: 0x{:x} output: 0x{:x} l1_buffer: 0x{:x}",
//         input->address(),
//         output->address(),
//         l1_buffer->address());

//     CommandQueue& command_queue = device->command_queue();

//     Program simple_program = create_simple_datamovement_program(*input, *output, *l1_buffer, rt_arg_per_core_vec);
//     vector<uint32_t> input_data(input->size() / sizeof(uint32_t), 0);
//     for (uint32_t i = 0; i < input_data.size(); i++) {
//         input_data[i] = i;
//     }

//     vector<uint32_t> eager_output_data;
//     eager_output_data.resize(input_data.size());

//     // Write data to buffer, enqueue program, then read outputs and verify against expected.
//     EnqueueWriteBuffer(command_queue, *input, input_data.data(), /*blocking=*/true);
//     EnqueueProgram(command_queue, simple_program, /*blocking=*/true);
//     // This will verify that outputs matches between capture + replay
//     LightMetalCompareToCapture(command_queue, *output, eager_output_data.data());

//     EXPECT_TRUE(eager_output_data == input_data);

//     // For dev/debug go ahead and print the results
//     for (size_t i = 0; i < eager_output_data.size(); i++) {
//         log_debug(tt::LogMetalTrace, "i: {:3d} input: {} output: {}", i, input_data[i], eager_output_data[i]);
//     }

//     Finish(command_queue);
// }

// // Test simple case of single datamovement program on single RISC works for trace + replay.
// TEST_F(LightMetalBasicTest, SingleRISCDataMovement) {
//     CreateDeviceAndBeginCapture(4096);
//     SingleRISCDataMovement_test(device_, false);
// }

// // Same as above but with SetRuntimeArgs API that uses vec of CoreCoord and vec of vec rtargs.
// TEST_F(LightMetalBasicTest, SingleRISCDataMovementRtArgsPerCoreVec) {
//     CreateDeviceAndBeginCapture(4096);
//     SingleRISCDataMovement_test(device_, true);
// }

// // Same as above but let replay library manage open/close device instead of user (test), for coverage.
// TEST_F(LightMetalBasicTest, SingleRISCDataMovementReplayManageDevice) {
//     const bool replay_manage_device = true;
//     CreateDeviceAndBeginCapture(4096, replay_manage_device);
//     SingleRISCDataMovement_test(device_, false);
// }

// void three_risc_data_movement_compute_test(IDevice* device, bool dynamic_cb, bool dealloc_cb_buf_early) {
//     uint32_t buf_size_bytes = 64;  // 16 elements.
//     uint32_t cb_size_bytes = 2048;

//     CommandQueue& command_queue = device->command_queue();

//     auto input = CreateBuffer(InterleavedBufferConfig{device, buf_size_bytes, buf_size_bytes, BufferType::DRAM});
//     auto output = CreateBuffer(InterleavedBufferConfig{device, buf_size_bytes, buf_size_bytes, BufferType::DRAM});
//     std::shared_ptr<Buffer> cb_in_buf;
//     Program program;

//     if (dynamic_cb) {
//         cb_in_buf = CreateBuffer(InterleavedBufferConfig{device, cb_size_bytes, cb_size_bytes, BufferType::L1});
//         if (dealloc_cb_buf_early) {
//             DeallocateBuffer(*cb_in_buf);
//         }
//         program = create_simple_unary_program(*input, *output, cb_in_buf.get());
//     } else {
//         program = create_simple_unary_program(*input, *output);
//     }

//     vector<uint32_t> input_data(input->size() / sizeof(uint32_t), 0);
//     for (uint32_t i = 0; i < input_data.size(); i++) {
//         input_data[i] = i;
//     }

//     // Write data to buffer, enqueue program, then read outputs.
//     EnqueueWriteBuffer(command_queue, *input, input_data.data(), /*blocking=*/true);
//     EnqueueProgram(command_queue, program, /*blocking=*/true);
//     // This will verify that outputs matches between capture + replay
//     LightMetalCompareToCapture(command_queue, *output);  // No read return

//     Finish(command_queue);
// }

// // Test simple case of 3 riscs used for datamovement and compute works for trace + replay.
// TEST_F(LightMetalBasicTest, ThreeRISCDataMovementCompute) {
//     CreateDeviceAndBeginCapture(4096);
//     three_risc_data_movement_compute_test(device_, false, false);
// }

// // Test simple case of 3 riscs used for datamovement and compute works for trace + replay. Also include dynamic CB.
// TEST_F(LightMetalBasicTest, ThreeRISCDataMovementComputeDynamicCB) {
//     CreateDeviceAndBeginCapture(4096);
//     three_risc_data_movement_compute_test(device_, true, false);
// }

// // Same as previous test but deallocate the Buffer before CB uses it (like Move op, which exposed bug)
// TEST_F(LightMetalBasicTest, ThreeRISCDataMovementComputeDynamicCBDeallocEarly) {
//     CreateDeviceAndBeginCapture(4096);
//     three_risc_data_movement_compute_test(device_, true, true);
// }

// // Test simple compute test with metal trace, but no explicit trace replay (added automatically by light metal trace).
// // Test currently not supported due to Trace API deprecation. See Issue #24955
// TEST_F(LightMetalBasicTest, DISABLED_SingleProgramTraceCapture) {
//     CreateDeviceAndBeginCapture(4096);

//     uint32_t size_bytes = 64;  // 16 elements. Was 2048 in original test.
//     auto input = CreateBuffer(InterleavedBufferConfig{device_, size_bytes, size_bytes, BufferType::DRAM});
//     auto output = CreateBuffer(InterleavedBufferConfig{device_, size_bytes, size_bytes, BufferType::DRAM});

//     CommandQueue& command_queue = device_->command_queue();
//     Program simple_program = create_simple_unary_program(*input, *output);

//     // Setup input data for program with some simple values.
//     vector<uint32_t> input_data(input->size() / sizeof(uint32_t), 0);
//     for (uint32_t i = 0; i < input_data.size(); i++) {
//         input_data[i] = i;
//     }

//     std::vector<uint32_t> eager_output_data(input_data.size());

//     // Initial run w/o trace. Preloads binary cache, and captures golden output.
//     EnqueueWriteBuffer(command_queue, *input, input_data.data(), /*blocking=*/true);
//     EnqueueProgram(command_queue, simple_program, /*blocking=*/true);
//     // This will verify that outputs matches between capture + replay.
//     // LightMetalCompareToCapture(command_queue, *output, eager_output_data.data());

//     // Write junk to output buffer to help make sure trace run from standalone binary works.
//     write_junk_to_buffer(command_queue, *output);

//     // Now enable Metal Trace and run program again for capture.
//     // uint32_t tid = BeginTraceCapture(device_, command_queue.id());
//     EnqueueProgram(command_queue, simple_program, false);
//     // EndTraceCapture(device_, command_queue.id(), tid);

//     // Verify trace output during replay matches expected output from original capture.
//     // LightMetalCompareToGolden(command_queue, *output, eager_output_data.data());

//     // Done
//     Finish(command_queue);
//     // ReleaseTrace(device_, tid);
// }

// // Test simple compute test with metal trace, but no explicit trace replay (added automatically by light metal trace).
// // Test currently not supported due to Trace API deprecation. See Issue #24955
// TEST_F(LightMetalBasicTest, DISABLED_TwoProgramTraceCapture) {
//     CreateDeviceAndBeginCapture(4096);

//     uint32_t size_bytes = 64;  // 16 elements. Was 2048 in original test.
//     auto input = CreateBuffer(InterleavedBufferConfig{device_, size_bytes, size_bytes, BufferType::DRAM});
//     auto interm = CreateBuffer(InterleavedBufferConfig{device_, size_bytes, size_bytes, BufferType::DRAM});
//     auto output = CreateBuffer(InterleavedBufferConfig{device_, size_bytes, size_bytes, BufferType::DRAM});

//     CommandQueue& command_queue = device_->command_queue();

//     Program op0 = create_simple_unary_program(*input, *interm);
//     Program op1 = create_simple_unary_program(*interm, *output);

//     // Setup input data for program with some simple values.
//     vector<uint32_t> input_data(input->size() / sizeof(uint32_t), 0);
//     for (uint32_t i = 0; i < input_data.size(); i++) {
//         input_data[i] = i;
//     }

//     std::vector<uint32_t> eager_output_data(input_data.size());

//     // Initial run w/o trace. Preloads binary cache, and captures golden output.
//     EnqueueWriteBuffer(command_queue, *input, input_data.data(), /*blocking=*/true);
//     EnqueueProgram(command_queue, op0, /*blocking=*/true);
//     EnqueueProgram(command_queue, op1, /*blocking=*/true);
//     // This will verify that outputs matches between capture + replay.
//     // LightMetalCompareToCapture(command_queue, *output, eager_output_data.data());
//     Finish(command_queue);

//     // Write junk to output buffer to help make sure trace run from standalone binary works.
//     write_junk_to_buffer(command_queue, *output);

//     // Now enable Metal Trace and run program again for capture.
//     // uint32_t tid = BeginTraceCapture(device_, command_queue.id());
//     EnqueueProgram(command_queue, op0, false);
//     EnqueueProgram(command_queue, op1, false);
//     // EndTraceCapture(device_, command_queue.id(), tid);

//     // Verify trace output during replay matches expected output from original capture.
//     // LightMetalCompareToGolden(command_queue, *output, eager_output_data.data());

//     // Done
//     Finish(command_queue);
//     // ReleaseTrace(device_, tid);
// }

// }  // namespace
// }  // namespace tt::tt_metal
