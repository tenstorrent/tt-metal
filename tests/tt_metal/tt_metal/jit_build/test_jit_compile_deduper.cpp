// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "impl/jit_server/in_flight_compile_deduper.hpp"
#include "impl/jit_server/types.hpp"

namespace tt::tt_metal::jit_server {

TEST(JitCompileDeduperTest, DeduplicatesConcurrentIdenticalRequests) {
    InFlightCompileDeduper<CompileResponse> deduper;

    std::atomic<int> compile_invocations{0};
    constexpr int kNumThreads = 8;
    std::vector<std::future<CompileResponse>> futures;
    futures.reserve(kNumThreads);

    for (int i = 0; i < kNumThreads; ++i) {
        futures.push_back(std::async(std::launch::async, [&]() {
            return deduper.run("build:kernel", [&]() {
                ++compile_invocations;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                CompileResponse response;
                response.success = true;
                return response;
            });
        }));
    }

    for (auto& future : futures) {
        auto response = future.get();
        EXPECT_TRUE(response.success);
    }

    EXPECT_EQ(compile_invocations.load(), 1);
}

TEST(JitCompileDeduperTest, FailedBuildDoesNotPoisonFutureRequests) {
    InFlightCompileDeduper<CompileResponse> deduper;
    std::atomic<int> invocation_count{0};

    EXPECT_THROW(
        deduper.run(
            "retry-key",
            [&]() -> CompileResponse {
                ++invocation_count;
                throw std::runtime_error("simulated compile failure");
            }),
        std::runtime_error);

    auto retry_response = deduper.run("retry-key", [&]() {
        ++invocation_count;
        CompileResponse response;
        response.success = true;
        return response;
    });

    EXPECT_TRUE(retry_response.success);
    EXPECT_EQ(invocation_count.load(), 2);
}

}  // namespace tt::tt_metal::jit_server
