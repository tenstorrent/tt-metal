// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-logger/tt-logger.hpp>
#include <vector>
#include "device_fixture.hpp"
#include "gtest/gtest.h"
#include "host_api.hpp"
#include "tt_metal.hpp"
#include "circular_buffer.hpp"

namespace tt::tt_metal {

/**
 *
 * This test checks that the cb_reserve_back and cb_wait_front will wait correctly when the internal counters of CB
 * overflows at the PACKER core.
 *
 * To dive into a bit more detail, CB is implemented with 2 counters, received and acked, where received counter
 * indicates the amount of pages pushed into CB, and acked counter indicates the amount of pages popped from CB. The
 * calculation of received and acked counter is done in 16 bits. Naturally, the received counter will be bigger than the
 * acked counter, which would be reverted when received counter overflows. Prior to #26536, the calculation for the
 * amount of free pages left in the system does not handle the overflow correctly, resulting in cb_reserve_back
 * returning prematurely.
 *
 * The test is setup with:
 * A writer kernel, and a reader kernel;
 * A CB with 64 pages (1024 bytes with 16 bytes per page)
 *
 * We define a "step" as 32 pages, so the CB holds 2 "steps", all the calls to cb_reserve_back, cb_push_back,
 * cb_wait_front and cb_pop_front will be called with a multiple of "step" pages.
 *
 */

static constexpr auto CB_ID = tt::CBIndex::c_0;
static constexpr CoreCoord WORKER_CORE = {0, 0};

using DataT = std::uint32_t;
static constexpr auto DATA_FORMAT = DataFormat::UInt32;

// CB have 64 pages.
static constexpr std::size_t CB_SIZE = 1024;
static constexpr std::size_t CB_PAGE_SIZE = 16;

// Result buffer is used to beam small (1-4) number of data to the host.
// Buffer is exactly 1 page due to the small amount of data we will be transfering.
static constexpr std::size_t RESULT_BUFFER_PAGE_SIZE = CB_PAGE_SIZE;
static constexpr std::size_t RESULT_BUFFER_SIZE = RESULT_BUFFER_PAGE_SIZE;
static constexpr auto RESULT_BUFFER_TYPE = BufferType::L1;

// Helper function that creates and zero-initializes a result buffer.
std::shared_ptr<Buffer> create_result_buffer(IDevice* device) {
    auto result_buffer = Buffer::create(device, RESULT_BUFFER_PAGE_SIZE, RESULT_BUFFER_SIZE, RESULT_BUFFER_TYPE);
    std::vector<DataT> init_data(RESULT_BUFFER_SIZE / sizeof(DataT), 0);
    detail::WriteToDeviceL1(device, WORKER_CORE, result_buffer->address(), init_data);
    return result_buffer;
}

/**
 * Testing for blocking reserve_back and wait_front:
 *
 * Because we cannot directly test if functions are hanging correctly, we test for data corruption instead. If
 * cb_reserve_back or cb_wait_front does not handle overflow correctly when calculating the amount of free pages left/
 * amount of pages pending, they will return prematurely, resulting in writing data to non-freed pages, which would
 * cause data corruption. If no data corruption is detected, it means that the overflow is handled correctly.
 *
 * Step by step of how the test works:
 * 1. Reader + Writer churns through the received and acked counter till the acked and received counter 2 steps till
 * overflow.
 * 2. Reader enters a spin loop, stop consuming from the CB.
 * 3. Writer fills the CB with 2 steps of data of value A, now the CB should be full of value A.
 * 4. Writer calls cb_reserve_back and attempts to write 1 steps of data B.
 *    If cb_reserve_back handles the overflow correctly, it will hang and wait for reader.
 *    If cb_reserve_back handles the overflow incorrectly, it will be incorrectly overwritten the first step of A with
 *    B.
 * 5. Reader exits spin, reads all pending data (3 steps worth of data)
 *    If cb_reserve_back handles the overflow correctly, it should hang writer till reader exits spin, resulting in BBA.
 *    If cb_reserve_back handles the overflow incorrectly, premature write will happen before reader exists spin,
 *    resulting in ABA (first step of pages is overwritten with A).
 *
 */

// Values we write to the areas that could be overwritten by incorrect reserve calls.
static constexpr DataT WRAP_WRITE_VALUE = 0xAAAA;
// Values used to overwrite the buffer in the last few pages.
static constexpr DataT WRITE_OVER_VALUE = 0xBBBB;
// Expected result of the test.
static const std::vector<DataT> EXPECTED_RESULT = {WRAP_WRITE_VALUE, WRAP_WRITE_VALUE, WRITE_OVER_VALUE};

/**
 * This tests blocking reserve back and wait front between Packer at Compute Kernel and BRSIC.
 * Here, cb_reserve_back is implemented in compute kernel API while cb_wait_front is implemented in dataflow API.
 */
TEST_F(DeviceFixture, TensixTestCircularBufferWrappingBlockingToWriter) {
    auto device = devices_.at(0);
    Program program;
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_wrapping_test_blocking_writer.cpp",
        WORKER_CORE,
        ComputeConfig{});

    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_wrapping_test_blocking_reader.cpp",
        WORKER_CORE,
        WriterDataMovementConfig{});

    CreateCircularBuffer(
        program, WORKER_CORE, CircularBufferConfig{CB_SIZE, {{CB_ID, DATA_FORMAT}}}.set_page_size(CB_ID, CB_PAGE_SIZE));

    auto result_buffer = Buffer::create(device, RESULT_BUFFER_PAGE_SIZE, RESULT_BUFFER_SIZE, RESULT_BUFFER_TYPE);
    SetRuntimeArgs(program, reader_kernel, WORKER_CORE, {result_buffer->address()});

    EnqueueProgram(device->command_queue(), program, true);

    std::vector<DataT> host_buffer;
    auto expected_result_size = EXPECTED_RESULT.size() * sizeof(DataT);
    detail::ReadFromDeviceL1(device, WORKER_CORE, result_buffer->address(), expected_result_size, host_buffer);

    EXPECT_EQ(host_buffer, EXPECTED_RESULT) << "Page corruption detected.";
}

/**
 * This tests blocking reserve back and wait_front between BRSIC and Unpacker at Compute Kernel.
 * Here, cb_reserve_back is implemented in dataflow API while cb_wait_front is implemented in Compute Kernel API.
 */
TEST_F(DeviceFixture, TensixTestCircularBufferWrappingBlockingToCompute) {
    auto device = devices_.at(0);
    Program program;
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_wrapping_test_blocking_writer.cpp",
        WORKER_CORE,
        ReaderDataMovementConfig{});

    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_wrapping_test_blocking_reader.cpp",
        WORKER_CORE,
        ComputeConfig{});

    CreateCircularBuffer(
        program, WORKER_CORE, CircularBufferConfig{CB_SIZE, {{CB_ID, DATA_FORMAT}}}.set_page_size(CB_ID, CB_PAGE_SIZE));

    auto result_buffer = Buffer::create(device, RESULT_BUFFER_PAGE_SIZE, RESULT_BUFFER_SIZE, RESULT_BUFFER_TYPE);
    SetRuntimeArgs(program, reader_kernel, WORKER_CORE, {result_buffer->address()});

    EnqueueProgram(device->command_queue(), program, true);

    std::vector<DataT> host_buffer;
    auto expected_result_size = EXPECTED_RESULT.size() * sizeof(DataT);
    detail::ReadFromDeviceL1(device, WORKER_CORE, result_buffer->address(), expected_result_size, host_buffer);

    EXPECT_EQ(host_buffer, EXPECTED_RESULT) << "Page corruption detected.";
}

/**
 * Testing for nonblocking APIs: reservable at back and available at front.
 *
 * We use a similar setup as the blocking test, but test directly if the nonblocking APIs returns correctly.
 */

/**
 * This tests if available at front is working correctly.
 *
 * Here, writer is at Compute Kernel, and reader is at a Dataflow Kernel.
 *
 * The sequence of event is:
 * 1. Writer and Reader churn through the received and acked counter till the acked and received counter 2 steps till
 * overflow.
 * 2. Writer stops writing
 * 3. Reader asks for available pages, should be false.
 */
TEST_F(DeviceFixture, TensixTestCircularBufferWrappingNonBlockingFront) {
    static constexpr DataT SUCCESS_TOKEN = 0xC0FFEE;

    auto device = devices_.at(0);
    Program program;
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_wrapping_test_non_blocking_writer.cpp",
        WORKER_CORE,
        ComputeConfig{});

    WriterDataMovementConfig reader_config;
    reader_config.defines["CHECK_FRONT"] = "1";

    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_wrapping_test_non_blocking_reader.cpp",
        WORKER_CORE,
        reader_config);

    CreateCircularBuffer(
        program, WORKER_CORE, CircularBufferConfig{CB_SIZE, {{CB_ID, DATA_FORMAT}}}.set_page_size(CB_ID, CB_PAGE_SIZE));

    auto result_buffer = create_result_buffer(device);
    SetRuntimeArgs(program, reader_kernel, WORKER_CORE, {result_buffer->address(), SUCCESS_TOKEN});

    EnqueueProgram(device->command_queue(), program, true);

    std::vector<DataT> host_buffer;
    detail::ReadFromDeviceL1(device, WORKER_CORE, result_buffer->address(), sizeof(DataT), host_buffer);
    EXPECT_EQ(host_buffer.front(), SUCCESS_TOKEN) << "Reader should have detected that the CB is full.";
}

/**
 * This tests if reservable at back is working correctly.
 *
 * Here, writer is at a Dataflow Kernel, and reader is at Compute Kernel.
 *
 * The sequence of event is:
 * 1. Writer and Reader churn through the received and acked counter till the acked and received counter 2 steps till
 * overflow.
 * 2. Writer writes 2 steps of data, filling the CB.
 * 3. Writer asks if there's any reserable pages, should be false.
 */
TEST_F(DeviceFixture, TensixTestCircularBufferWrappingNonBlockingBack) {
    static constexpr DataT SUCCESS_TOKEN = 0xBABE;

    auto device = devices_.at(0);
    Program program;

    ReaderDataMovementConfig writer_config;
    writer_config.defines["CHECK_BACK"] = "1";

    auto writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_wrapping_test_non_blocking_writer.cpp",
        WORKER_CORE,
        writer_config);

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_wrapping_test_non_blocking_reader.cpp",
        WORKER_CORE,
        ComputeConfig{});

    CreateCircularBuffer(
        program, WORKER_CORE, CircularBufferConfig{CB_SIZE, {{CB_ID, DATA_FORMAT}}}.set_page_size(CB_ID, CB_PAGE_SIZE));

    auto result_buffer = create_result_buffer(device);
    SetRuntimeArgs(program, writer_kernel, WORKER_CORE, {result_buffer->address(), SUCCESS_TOKEN});

    EnqueueProgram(device->command_queue(), program, true);

    std::vector<DataT> host_buffer;
    detail::ReadFromDeviceL1(device, WORKER_CORE, result_buffer->address(), sizeof(DataT), host_buffer);
    EXPECT_EQ(host_buffer.front(), SUCCESS_TOKEN) << "Writer should have detected that the CB is full.";
}

}  // namespace tt::tt_metal
