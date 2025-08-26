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
 * Because we cannot directly test if functions are hanging correctly, we test for data corruption instead. If
 * cb_reserve_back or cb_wait_front does not handle overflow correctly when calculating the amount of free pages left/
 * amount of pages pending, they will return prematurely, resulting in writing data to non-freed pages, which would
 * cause data corruption. If no data corruption is detected, it means that the overflow is handled correctly.
 *
 * The test is setup with:
 * A writer kernel at PACK core, and a reader kernel at NOC1 core;
 * A CB with 64 pages (1024 bytes with 16 bytes per page)
 *
 * We define a "step" as 32 pages, so the CB holds 2 "steps", all the calls to cb_reserve_back, cb_push_back,
 * cb_wait_front and cb_pop_front will be called with a multiple of "step" pages.
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

static constexpr auto CB_ID = tt::CBIndex::c_0;
static constexpr CoreCoord WORKER_CORE = {0, 0};

using DataT = std::uint32_t;
static constexpr auto DATA_FORMAT = DataFormat::UInt32;

// CB have 64 pages.
static constexpr std::size_t CB_SIZE = 1024;
static constexpr std::size_t CB_PAGE_SIZE = 16;

// Values we write to the areas that could be overwritten by incorrect reserve calls.
static constexpr DataT WRAP_WRITE_VALUE = 0xAAAA;
// Values used to overwrite the buffer in the last few pages.
static constexpr DataT WRITE_OVER_VALUE = 0xBBBB;
// Expected result of the test.
static const std::vector<DataT> EXPECTED_RESULT = {WRAP_WRITE_VALUE, WRAP_WRITE_VALUE, WRITE_OVER_VALUE};

TEST_F(DeviceFixture, TensixTestCircularBufferWrapping) {
    auto device = devices_.at(0);
    Program program;
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_wrapping_test_writer.cpp",
        WORKER_CORE,
        ComputeConfig{});

    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_wrapping_test_reader.cpp",
        WORKER_CORE,
        WriterDataMovementConfig{});

    CreateCircularBuffer(
        program, WORKER_CORE, CircularBufferConfig{CB_SIZE, {{CB_ID, DATA_FORMAT}}}.set_page_size(CB_ID, CB_PAGE_SIZE));

    auto result_buffer = Buffer::create(device, CB_PAGE_SIZE, CB_PAGE_SIZE, BufferType::L1);
    SetRuntimeArgs(program, reader_kernel, WORKER_CORE, {result_buffer->address()});

    EnqueueProgram(device->command_queue(), program, true);

    std::vector<DataT> host_buffer;
    auto expected_result_size = EXPECTED_RESULT.size() * sizeof(DataT);
    detail::ReadFromDeviceL1(device, WORKER_CORE, result_buffer->address(), expected_result_size, host_buffer);

    EXPECT_EQ(host_buffer, EXPECTED_RESULT) << "Page corruption detected.";
}

}  // namespace tt::tt_metal
