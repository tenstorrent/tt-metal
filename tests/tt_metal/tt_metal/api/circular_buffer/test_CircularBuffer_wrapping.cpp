// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include "device_fixture.hpp"
#include "gtest/gtest.h"
#include "host_api.hpp"
#include "tt_metal.hpp"

using namespace tt;
using namespace tt::tt_metal;

constexpr CoreCoord worker_core = {0, 0};

// CB have 64 pages.
constexpr size_t cb_size = 1024;
constexpr size_t cb_page_size = 16;

/**
 *  This test checks that the cb_reserve_back will wait correctly when the buffer is full.
 *  There's two ways to test this:
 *  1. Fill the buffer up till it wraps around, and then check if this operation corupts any data in DRAM.
 */

TEST_F(DeviceFixture, TensixTestCircularBufferWrapping) {
    auto device = devices_.at(0);
    Program program;
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_wrapping_test_writer.cpp",
        worker_core,
        ComputeConfig{});

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_wrapping_test_reader.cpp",
        worker_core,
        WriterDataMovementConfig{});

    CreateCircularBuffer(
        program,
        worker_core,
        CircularBufferConfig{cb_size, {{CBIndex::c_0, DataFormat::UInt32}}}.set_page_size(CBIndex::c_0, cb_page_size));

    detail::CompileProgram(device, program);
    detail::LaunchProgram(device, program, true);
}
