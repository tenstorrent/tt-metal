// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include "debug_tools_fixture.hpp"

using namespace tt::tt_metal;

namespace {

constexpr int kClearLogCalls = 20;
constexpr const char* kLegendLine = "Legend:";

size_t count_open_descriptors_of(const std::filesystem::path& target) {
    size_t count = 0;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator("/proc/self/fd", ec)) {
        std::error_code link_ec;
        const auto linked = std::filesystem::read_symlink(entry.path(), link_ec);
        if (!link_ec && linked == target) {
            count++;
        }
    }
    return count;
}

size_t count_lines_equal_to(const std::filesystem::path& file, const std::string& wanted) {
    std::ifstream in(file);
    size_t count = 0;
    for (std::string line; std::getline(in, line);) {
        if (line == wanted) {
            count++;
        }
    }
    return count;
}

}  // namespace

TEST_F(MeshWatcherFixture, TestWatcherClearLogKeepsOneOpenLogFile) {
    if (MetalContext::instance().rtoptions().get_watcher_skip_logging()) {
        GTEST_SKIP() << "Watcher logging is disabled";
    }
    auto& watcher_server = *MetalContext::instance().watcher_server();
    const auto log_file = std::filesystem::weakly_canonical(watcher_server.log_file_name());
    for (int i = 0; i < kClearLogCalls; i++) {
        watcher_server.clear_log();
    }
    EXPECT_EQ(count_open_descriptors_of(log_file), 1u);
    EXPECT_EQ(count_lines_equal_to(log_file, kLegendLine), 1u);
}
