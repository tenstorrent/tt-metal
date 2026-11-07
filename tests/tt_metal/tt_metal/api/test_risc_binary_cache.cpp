// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>

#include "llrt.hpp"

namespace {

constexpr unsigned kTimeoutSeconds = 30;
constexpr auto kSettle = std::chrono::milliseconds(200);
constexpr mode_t kFifoMode = 0600;

bool load_throws(const std::string& path) {
    try {
        tt::llrt::get_risc_binary(path);
    } catch (...) {
        return true;
    }
    return false;
}

void release_reader(const std::string& fifo) { close(open(fifo.c_str(), O_WRONLY)); }

TEST(RiscBinaryCache, CPU_FailedLoadDoesNotBlockLaterRequests) {
    const std::string path = "/nonexistent/risc_binary_cache_test/brisc.elf";
    auto run = [&] {
        alarm(kTimeoutSeconds);
        const bool all_threw = load_throws(path) && load_throws(path) && load_throws(path);
        std::_Exit(all_threw ? 0 : 1);
    };
    EXPECT_EXIT(run(), ::testing::ExitedWithCode(0), "");
}

TEST(RiscBinaryCache, CPU_WaiterReloadsAfterFailedLoad) {
    const auto fifo =
        (std::filesystem::temp_directory_path() / ("risc_binary_cache_" + std::to_string(getpid()) + ".elf")).string();
    std::filesystem::remove(fifo);
    ASSERT_EQ(mkfifo(fifo.c_str(), kFifoMode), 0);
    auto run = [&] {
        alarm(kTimeoutSeconds);
        bool loader_threw = false;
        bool waiter_threw = false;
        std::thread loader([&] { loader_threw = load_throws(fifo); });
        std::this_thread::sleep_for(kSettle);
        std::thread waiter([&] { waiter_threw = load_throws(fifo); });
        std::this_thread::sleep_for(kSettle);
        release_reader(fifo);
        loader.join();
        release_reader(fifo);
        waiter.join();
        std::_Exit(loader_threw && waiter_threw ? 0 : 1);
    };
    EXPECT_EXIT(run(), ::testing::ExitedWithCode(0), "");
    std::filesystem::remove(fifo);
}

}  // namespace
