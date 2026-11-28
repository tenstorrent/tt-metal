// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <future>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "impl/context/context_types.hpp"
#include "impl/jit_server/remote_compile_coordinator.hpp"

namespace tt::tt_metal {

namespace {

using SubmitResult = RemoteCompileCoordinator::SubmitResult;

constexpr std::size_t kKernelHash = 1;
constexpr std::size_t kInFlightKernelHash = 2;

// Nothing listens here. These tests cover the submissions that send nothing, so no session is opened.
const std::vector<std::string> kUnusedEndpoints = {"localhost:1"};

RemoteCompileCoordinator make_coordinator() {
    return RemoteCompileCoordinator(kUnusedEndpoints, DEFAULT_CONTEXT_ID, /*device_build_id=*/0, /*build_key=*/0);
}

}  // namespace

// The dedup cache is process-wide, so every test starts with it empty. In the named namespace:
// TEST_F classes derive from it with external linkage, and an anonymous-namespace base trips
// -Werror=subobject-linkage under gcc Unity builds.
class RemoteCompileCoordinatorTest : public ::testing::Test {
protected:
    void SetUp() override { RemoteCompileCoordinator::clear_dedup_cache(); }
};

TEST_F(RemoteCompileCoordinatorTest, KernelWithNothingToCompileIsReusedThenDeduplicated) {
    int descriptor_calls = 0;
    const auto nothing_to_compile = [&]() -> std::optional<KernelCompileDescriptor> {
        ++descriptor_calls;
        return std::nullopt;
    };

    auto coordinator = make_coordinator();
    EXPECT_EQ(coordinator.submit(kKernelHash, nothing_to_compile), SubmitResult::Reused);
    EXPECT_EQ(coordinator.submit(kKernelHash, nothing_to_compile), SubmitResult::Deduplicated);
    EXPECT_NO_THROW(coordinator.finish());

    // A later Program compile in the same process does not check the kernel again.
    auto later = make_coordinator();
    EXPECT_EQ(later.submit(kKernelHash, nothing_to_compile), SubmitResult::Deduplicated);
    EXPECT_NO_THROW(later.finish());

    EXPECT_EQ(descriptor_calls, 1);
}

TEST_F(RemoteCompileCoordinatorTest, ClearDedupCacheForgetsFinishedKernelsButKeepsInFlightOnes) {
    const auto nothing_to_compile = []() -> std::optional<KernelCompileDescriptor> { return std::nullopt; };

    auto finished = make_coordinator();
    ASSERT_EQ(finished.submit(kKernelHash, nothing_to_compile), SubmitResult::Reused);

    // A second kernel whose owner is still inside make_descriptor().
    std::promise<void> started;
    std::promise<void> release;
    auto release_future = release.get_future();
    std::thread owner([&] {
        auto in_flight = make_coordinator();
        in_flight.submit(kInFlightKernelHash, [&]() -> std::optional<KernelCompileDescriptor> {
            started.set_value();
            release_future.wait();
            return std::nullopt;
        });
    });
    started.get_future().wait();

    RemoteCompileCoordinator::clear_dedup_cache();

    auto after_clear = make_coordinator();
    EXPECT_EQ(after_clear.submit(kKernelHash, nothing_to_compile), SubmitResult::Reused);
    EXPECT_EQ(after_clear.submit(kInFlightKernelHash, nothing_to_compile), SubmitResult::Deduplicated);

    release.set_value();
    owner.join();
    EXPECT_NO_THROW(after_clear.finish());
}

}  // namespace tt::tt_metal
