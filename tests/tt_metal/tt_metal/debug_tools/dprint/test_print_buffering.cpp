// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include <tt-metalium/device.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "umd/device/types/xy_pair.h"
#include <tt-metalium/utils.hpp>

////////////////////////////////////////////////////////////////////////////////
// A test for checking that prints are properly buffered before being displayed to the user.
////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
const std::vector<string>& golden_output = {
    "(0,0): This is a large DPRINT message that should not be interleaved with other DPRINT messages. (0,0): Adding \
the alphabet to extend the size of this message: ABCDEFGHIJKLMNOPQRSTUVWXYZ. (0,0): Now, in reverse, to make it \
even longer: ZYXWVUTSRQPONMLKJIHGFEDCBA.",
    "(0,1): This is a large DPRINT message that should not be interleaved with other DPRINT messages. (0,1): Adding \
the alphabet to extend the size of this message: ABCDEFGHIJKLMNOPQRSTUVWXYZ. (0,1): Now, in reverse, to make it \
even longer: ZYXWVUTSRQPONMLKJIHGFEDCBA.",
    "(0,2): This is a large DPRINT message that should not be interleaved with other DPRINT messages. (0,2): Adding \
the alphabet to extend the size of this message: ABCDEFGHIJKLMNOPQRSTUVWXYZ. (0,2): Now, in reverse, to make it \
even longer: ZYXWVUTSRQPONMLKJIHGFEDCBA.",
    "(0,0): Once upon a time, in a small village, there was a little mouse named Tim. Tim wasn't like other mice. He \
was brave and curious, always venturing into places others wouldn't dare. One day, while exploring the forest, he \
found a big cheese trapped in a cage. Tim knew he had to help. Using his sharp teeth, he gnawed through the bars \
and set the cheese free. To his surprise, a kind old owl had been watching and offered him a gift - the ability \
to talk to all creatures. From that day on, Tim helped others, becoming a hero in the animal kingdom. And so, the \
little mouse learned that bravery and kindness can change the world.",
    "(0,1): Once upon a time, in a small village, there was a little mouse named Tim. Tim wasn't like other mice. He \
was brave and curious, always venturing into places others wouldn't dare. One day, while exploring the forest, he \
found a big cheese trapped in a cage. Tim knew he had to help. Using his sharp teeth, he gnawed through the bars \
and set the cheese free. To his surprise, a kind old owl had been watching and offered him a gift - the ability \
to talk to all creatures. From that day on, Tim helped others, becoming a hero in the animal kingdom. And so, the \
little mouse learned that bravery and kindness can change the world.",
    "(0,2): Once upon a time, in a small village, there was a little mouse named Tim. Tim wasn't like other mice. He \
was brave and curious, always venturing into places others wouldn't dare. One day, while exploring the forest, he \
found a big cheese trapped in a cage. Tim knew he had to help. Using his sharp teeth, he gnawed through the bars \
and set the cheese free. To his surprise, a kind old owl had been watching and offered him a gift - the ability \
to talk to all creatures. From that day on, Tim helped others, becoming a hero in the animal kingdom. And so, the \
little mouse learned that bravery and kindness can change the world.",
    "(0,0): This DPRINT message",
    "contains several newline characters",
    "and should be displayed over multiple lines.",
    "(0,1): This DPRINT message",
    "contains several newline characters",
    "and should be displayed over multiple lines.",
    "(0,2): This DPRINT message",
    "contains several newline characters",
    "and should be displayed over multiple lines."};

void RunTest(DPrintFixture* fixture, IDevice* device) {
    std::vector<CoreCoord> cores;
    cores.emplace_back(0, 0);
    cores.emplace_back(0, 1);
    cores.emplace_back(0, 2);

    Program program = Program();

    for (const CoreCoord& core : cores) {
        KernelHandle kernel_id = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/print_buffering.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        SetRuntimeArgs(program, kernel_id, core, {core.x, core.y});

        log_info(tt::LogTest, "Running test on core {}:({},{})", device->id(), core.x, core.y);
    }

    fixture->RunProgram(device, program);

    // Check the print log against golden output.
    EXPECT_TRUE(FileContainsAllStrings(DPrintFixture::dprint_file_name, golden_output));
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TEST_F(DPrintFixture, TensixTestPrintBuffering) {
    for (IDevice* device : this->devices_) {
        this->RunTestOnDevice(
            [](DPrintFixture* fixture, IDevice* device) { CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, device); }, device);
    }
}
