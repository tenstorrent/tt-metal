// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "llrt/llrt.hpp"

namespace tt::llrt {
namespace {

// get_risc_binary() keeps an in-memory cache keyed by path. A read that throws (here: a missing
// ELF) used to leave an empty entry behind, and every later load of the same path waited on it
// forever. The next load must read the file again, and fail again, instead of hanging.
TEST(BinaryCacheTest, FailedReadDoesNotBlockTheNextLoad) {
    const std::string path =
        (std::filesystem::temp_directory_path() / ("tt-binary-cache-test-" + std::to_string(getpid()) + "-missing.elf"))
            .string();
    std::filesystem::remove(path);

    // Both loads run in the death-test child, so a hang there ends on the alarm instead of leaving
    // a thread blocked in this process.
    EXPECT_EXIT(
        {
            alarm(10);
            try {
                get_risc_binary(path);
            } catch (...) {
            }
            try {
                get_risc_binary(path);
            } catch (...) {
                std::_Exit(0);
            }
            std::_Exit(1);
        },
        ::testing::ExitedWithCode(0),
        "");
}

}  // namespace
}  // namespace tt::llrt
