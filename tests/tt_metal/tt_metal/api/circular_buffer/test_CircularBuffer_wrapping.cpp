// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-logger/tt-logger.hpp>
#include <vector>
#include "device_fixture.hpp"
#include "gtest/gtest.h"
#include "host_api.hpp"
#include "tt_metal.hpp"
#include "circular_buffer.hpp"

using namespace tt;
using namespace tt::tt_metal;

constexpr CoreCoord worker_core = {0, 0};

// CB have 64 pages.
constexpr size_t cb_size = 1024;
constexpr size_t cb_page_size = 16;

/**
 *
 * This test checks that the cb_reserve_back will wait correctly when the received and acked counter overflows.
 * Because we cannot directly test the overflow, we test for data corruption instead.
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
 * 5. Writer exits spin, reads all pending data (3 steps worth of data)
 *    If cb_reserve_back handles the overflow correctly, it should hang writer till reader exits spin, resulting in BBA.
 *    If cb_reserve_back handles the overflow incorrectly, premature write will happen before reader exists spin,
 *    resulting in ABA.
 *
 */

TEST_F(DeviceFixture, TensixTestCircularBufferWrapping) {
    auto device = devices_.at(0);
    Program program;
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_wrapping_test_writer.cpp",
        worker_core,
        ComputeConfig{});

    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_wrapping_test_reader.cpp",
        worker_core,
        WriterDataMovementConfig{});

    CreateCircularBuffer(
        program,
        worker_core,
        CircularBufferConfig{cb_size, {{CBIndex::c_0, DataFormat::UInt32}}}.set_page_size(CBIndex::c_0, cb_page_size));

    // We really only need to put 2 values in, but here the size is cb_page_size to ensure alignment.
    auto result_buffer = Buffer::create(device, cb_page_size, cb_page_size, BufferType::L1);
    log_info(tt::LogTest, "Result Buffer: {:X}", result_buffer->address());
    SetRuntimeArgs(program, reader_kernel, worker_core, {result_buffer->address()});

    EnqueueProgram(device->command_queue(), program, true);

    std::vector<uint32_t> host_buffer;
    detail::ReadFromDeviceL1(device, worker_core, result_buffer->address(), 3 * sizeof(uint32_t), host_buffer);

    const static std::vector<uint32_t> expected_result = {0xAAAA, 0xBBBB, 0xAAAA};
    EXPECT_EQ(host_buffer, expected_result) << "Page coruption detected.";
}
