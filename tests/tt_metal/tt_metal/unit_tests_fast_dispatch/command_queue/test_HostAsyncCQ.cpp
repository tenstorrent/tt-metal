// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "command_queue_fixture.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "impl/buffers/buffer.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/scoped_timer.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"

using namespace tt::tt_metal;

namespace host_cq_test_utils {
// Utility functions for Async Queue Flatten stress test
// Add more utils here for testing other ops/workloads
uint32_t prod(vector<uint32_t> &shape) {
    uint32_t shape_prod = 1;

    for (uint32_t shape_i: shape) {
        shape_prod *= shape_i;
    }

    return shape_prod;
}

inline std::vector<uint32_t> gold_standard_flatten(std::vector<uint32_t> src_vec, vector<uint32_t> shape) {

    int numel_in_tensor = prod(shape) / 2;
    int idx = 0;
    std::vector<uint32_t> expected_dst_vec;

    uint32_t num_tile_rows = shape.at(shape.size() - 2) / 32;
    uint32_t num_tile_cols = shape.at(shape.size() - 1) / 32;

    uint32_t start_dram_addr_offset_for_tensor_row = 0;

    for (int i = 0; i < num_tile_rows; i++) {
        for (uint32_t j = 0; j < 32; j++) {
            uint32_t src_addr_ = start_dram_addr_offset_for_tensor_row;
            for (uint32_t k = 0; k < num_tile_cols; k++) {

                // Copy a row
                for (uint32_t l = 0; l < 16; l++) {
                    uint32_t src_addr = src_addr_ + l;
                    expected_dst_vec.push_back(src_vec.at(src_addr_ + l));
                }

                // Zero padding
                for (uint32_t l = 0; l < 31 * 16; l++) {
                    expected_dst_vec.push_back(0);
                }
                src_addr_ += 32 * 16;
            }
            start_dram_addr_offset_for_tensor_row += 16;
        }
        start_dram_addr_offset_for_tensor_row += num_tile_cols * 16;
    }

    TT_FATAL(expected_dst_vec.size() == (num_tile_rows * 32) * (num_tile_cols * 16) * 32);
    return expected_dst_vec;
}

bool flatten(Device *device, uint32_t num_tiles_r = 5, uint32_t num_tiles_c = 5) {
    // Test Simulating Program Caching with Async Command Queues
    bool pass = true;
    // Create a program used across all loops
    Program program = CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;

    uint32_t num_tiles = num_tiles_r * num_tiles_c;
    uint32_t num_bytes_per_tensor_row = num_tiles_c * 64;
    uint32_t num_bytes_per_tile = num_tiles * single_tile_size;

    uint32_t dram_buffer_size = single_tile_size * num_tiles * 32;


    InterleavedBufferConfig dram_config{
                .device=device,
                .size = dram_buffer_size,
                .page_size = dram_buffer_size,
                .buffer_type = BufferType::DRAM
                };
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 8;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = 16;
    uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(ouput_cb_index, single_tile_size);
    auto cb_output = CreateCircularBuffer(program, core, cb_output_config);

    auto flatten_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/flatten.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto unary_writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {
        num_tiles * 32
    };

    auto eltwise_unary_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args}
    );

    // Inside the loop, run async runtime functions
    for (int i = 0; i < 1000; i++) {
        // Create Device Buffers Asynchronously
        auto src_dram_buffer = CreateBuffer(dram_config);
        auto dst_dram_buffer = CreateBuffer(dram_config);

        auto dram_src_noc_xy = src_dram_buffer->noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();
        // Create the source vector
        std::shared_ptr<std::vector<uint32_t>> src_vec = std::make_shared<std::vector<uint32_t>>(create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count()));

        std::vector<uint32_t> golden = gold_standard_flatten(*src_vec, {num_tiles_r * 32, num_tiles_c * 32});
        // Set the runtime args asynchronously
        std::shared_ptr<RuntimeArgs> writer_runtime_args = std::make_shared<RuntimeArgs>();
        std::shared_ptr<RuntimeArgs> compute_runtime_args = std::make_shared<RuntimeArgs>();
        *compute_runtime_args = {
            src_dram_buffer.get(),
            (std::uint32_t)dram_src_noc_xy.x,
            (std::uint32_t)dram_src_noc_xy.y,
            num_tiles_r,
            num_tiles_c,
            num_bytes_per_tensor_row
        };
        *writer_runtime_args = {
            dst_dram_buffer.get(),
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            num_tiles * 32
        };

        SetRuntimeArgs(
            device,
            detail::GetKernel(program, flatten_kernel),
            core,
            compute_runtime_args);

        SetRuntimeArgs(
            device,
            detail::GetKernel(program, unary_writer_kernel),
            core,
            writer_runtime_args);
        // Async write input
        EnqueueWriteBuffer(device->command_queue(), src_dram_buffer, src_vec, false);
        // Share ownership of buffer with program
        AssignGlobalBufferToProgram(src_dram_buffer, &program);
        // Main thread gives up ownership of buffer and src data (this is what python does)
        src_dram_buffer.reset();
        src_vec.reset();
        // Queue up program
        EnqueueProgram(device->command_queue(), &program, false);
        // Blocking read
        std::vector<uint32_t> result_vec;
        EnqueueReadBuffer(device->command_queue(), dst_dram_buffer, result_vec, true);

        // Validation of data
        TT_FATAL(golden.size() == result_vec.size());
        pass &= (golden == result_vec);

        if (not pass) {
            std::cout << "GOLDEN" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(golden, num_tiles * 32);

            std::cout << "RESULT" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(result_vec, num_tiles * 32);
        }
    }
    return pass;
}
}

namespace host_command_queue_tests {

TEST_F(CommandQueueFixture, TestAsyncCommandQueueSanityAndProfile) {
    auto& command_queue = this->device_->command_queue();
    auto current_mode = CommandQueue::default_mode();
    command_queue.set_mode(CommandQueue::CommandQueueMode::ASYNC);
    Program program;

    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});
    // Add an NCRISC blank manually, but in compile program, the BRISC blank will be
    // added separately
    auto dummy_reader_kernel = CreateKernel(
        program, "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/arbiter_hang.cpp", cr_set, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    // Use scoper timer to benchmark time for pushing 2 commands
    {
        tt::ScopedTimer timer("AsyncCommandQueue");
        EnqueueProgram(command_queue, &program, false);
        Finish(command_queue);
    }
    command_queue.set_mode(current_mode);
}

TEST_F(CommandQueueFixture, DISABLED_TestAsyncBufferRW) {
    // Test Async Enqueue Read and Write + Get Addr + Buffer Allocation and Deallocation
    auto& command_queue = this->device_->command_queue();
    auto current_mode = CommandQueue::default_mode();
    command_queue.set_mode(CommandQueue::CommandQueueMode::ASYNC);
    Program program; /* Dummy program that helps keep track of buffers */
    std::vector<Buffer> buffer_objects;
    for (int j = 0; j < 10; j++) {
        // Asynchronously initialize a buffer on device
        uint32_t first_buf_value = j + 1;
        uint32_t second_buf_value = j + 2;
        uint32_t first_buf_size = 4096;
        uint32_t second_buf_size = 2048;
        // Asynchronously allocate buffer on device
        std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(this->device_, first_buf_size, first_buf_size, BufferType::DRAM);
        // Copy the host side buffer structure to an object (i.e. reallocate on device). After reallocation the addresses should not match
        buffer_objects.push_back(*buffer);
        // Confirm that the addresses do not match after copy/reallocation. Send async calls to get addrs, which will return post dealloc addr
        std::shared_ptr<uint32_t> allocated_buffer_address = std::make_shared<uint32_t>();
        EnqueueGetBufferAddr(this->device_->command_queue(), allocated_buffer_address.get(), buffer.get(), true);
        std::shared_ptr<uint32_t> allocated_buffer_address_2 = std::make_shared<uint32_t>();
        EnqueueGetBufferAddr(this->device_->command_queue(), allocated_buffer_address_2.get(), &(buffer_objects.back()), true);
        EXPECT_NE(*allocated_buffer_address_2, *allocated_buffer_address);
        // Ensure returned addrs are correct
        EXPECT_EQ((*allocated_buffer_address), buffer->address());
        EXPECT_EQ((*allocated_buffer_address_2), buffer_objects.back().address());
        // Deallocate the second device side buffer
        detail::DeallocateBuffer(&(buffer_objects.back()));
        // Make the buffer_object address and the buffer address identical with a blocking call. buffer_object and buffer are now the same device side buffer
        buffer_objects.back().set_address(*allocated_buffer_address);

        std::shared_ptr<std::vector<uint32_t>> vec = std::make_shared<std::vector<uint32_t>>(first_buf_size / 4, first_buf_value);
        std::vector<uint32_t> readback_vec_1 = {};
        std::vector<uint32_t> readback_vec_2 = {};
        // Write first vector to existing on device buffer.
        EnqueueWriteBuffer(this->device_->command_queue(), buffer, vec, false);
        // Reallocate the vector in the main thread after asynchronously pushing it (ensure that worker still has access to this data)
        vec = std::make_shared<std::vector<uint32_t>>(second_buf_size / 4, second_buf_value);
        // Simulate what tt-eager does: Share buffer ownership with program
        AssignGlobalBufferToProgram(buffer, &program);
        // Reallocate buffer (this is safe, since the program also owns the existing buffer, which will not be deallocated)
        buffer = std::make_shared<Buffer>(this->device_, second_buf_size, second_buf_size, BufferType::DRAM);
        // Write second vector to second buffer
        EnqueueWriteBuffer(this->device_->command_queue(), buffer, vec, false);
        // Have main thread give up ownership immediately after writing
        vec.reset();
        // Read both buffer and ensure data is correct
        EnqueueReadBuffer(this->device_->command_queue(), buffer_objects.back(), readback_vec_1, false);
        EnqueueReadBuffer(this->device_->command_queue(), buffer, readback_vec_2, true);
        for (int i = 0; i < readback_vec_1.size(); i++) {
            EXPECT_EQ(readback_vec_1[i], first_buf_value);
        }
        for (int i = 0; i < readback_vec_2.size(); i++) {
            EXPECT_EQ(readback_vec_2[i], second_buf_value);
        }
    }
    command_queue.set_mode(current_mode);
}

TEST_F(CommandQueueFixture, DISABLED_TestAsyncCBAllocation) {
    // Test asynchronous allocation of buffers and their assignment to CBs
    auto& command_queue = this->device_->command_queue();
    auto current_mode = CommandQueue::default_mode();
    command_queue.set_mode(CommandQueue::CommandQueueMode::ASYNC);
    Program program;

    const uint32_t num_pages = 1;
    const uint32_t page_size = detail::TileSize(tt::DataFormat::Float16_b);
    const tt::DataFormat data_format = tt::DataFormat::Float16_b;

    auto buffer_size = page_size;
    tt::tt_metal::InterleavedBufferConfig buff_config{
                    .device=this->device_,
                    .size = buffer_size,
                    .page_size = buffer_size,
                    .buffer_type = tt::tt_metal::BufferType::L1
        };
    // Asynchronously allocate an L1 Buffer
    auto l1_buffer = CreateBuffer(buff_config);
    CoreRange cr({0, 0}, {0, 2});
    CoreRangeSet cr_set({cr});
    std::vector<uint8_t> buffer_indices = {16, 24};

    CircularBufferConfig config1 = CircularBufferConfig(page_size, {{buffer_indices[0], data_format}, {buffer_indices[1], data_format}}, *l1_buffer)
        .set_page_size(buffer_indices[0], page_size)
        .set_page_size(buffer_indices[1], page_size);
    // Asynchronously assign the L1 Buffer to the CB
    auto multi_core_cb = CreateCircularBuffer(program, cr_set, config1);
    auto cb_ptr = detail::GetCircularBuffer(program, multi_core_cb);
    Finish(this->device_->command_queue());
    // Addresses should match
    EXPECT_EQ(cb_ptr->address(), l1_buffer->address());
    // Asynchronously allocate a new L1 buffer
    auto l1_buffer_2 = CreateBuffer(buff_config);
    // Asynchronously update CB address to match new L1 buffer
    UpdateDynamicCircularBufferAddress(program, multi_core_cb, *l1_buffer_2);
    Finish(this->device_->command_queue());
    // Addresses should match
    EXPECT_EQ(cb_ptr->address(), l1_buffer_2->address());
    command_queue.set_mode(current_mode);
}

TEST_F(CommandQueueFixture, DISABLED_TestAsyncAssertForDeprecatedAPI) {
    auto& command_queue = this->device_->command_queue();
    auto current_mode = CommandQueue::default_mode();
    command_queue.set_mode(CommandQueue::CommandQueueMode::ASYNC);
    Program program;
    CoreCoord core = {0, 0};
    uint32_t buf_size = 4096;
    uint32_t page_size = 4096;
    auto dummy_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    auto src0 = std::make_shared<Buffer>(this->device_, buf_size, page_size, BufferType::DRAM);
    std::vector<uint32_t> runtime_args = {src0->address()};
    try {
        SetRuntimeArgs(program, dummy_kernel, core, runtime_args);
    }
    catch (std::runtime_error &e) {
        std::string expected = "This variant of SetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for Fast Dispatch.";
        const string error = string(e.what());
        EXPECT_TRUE(error.find(expected) != std::string::npos);
    }
    command_queue.set_mode(current_mode);
}

TEST_F(CommandQueueFixture, DISABLED_TestAsyncFlattenStress){
    auto& command_queue = this->device_->command_queue();
    auto current_mode = CommandQueue::default_mode();
    command_queue.set_mode(CommandQueue::CommandQueueMode::ASYNC);
    uint32_t num_tiles_r = 2;
    uint32_t num_tiles_c = 2;
    if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")){
        num_tiles_r = 1;
        num_tiles_c = 1;
    }
    ASSERT_TRUE(host_cq_test_utils::flatten(this->device_, num_tiles_r, num_tiles_c));
    command_queue.set_mode(current_mode);
}
}
