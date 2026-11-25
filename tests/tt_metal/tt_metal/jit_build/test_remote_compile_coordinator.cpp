// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "impl/context/context_types.hpp"
#include "impl/jit_server/remote_compile_coordinator.hpp"

namespace tt::tt_metal {

namespace {

using SubmitResult = RemoteCompileCoordinator::SubmitResult;

// Nothing listens here. These tests cover the submissions that send nothing, so no session is opened.
const std::vector<std::string> kUnusedEndpoints = {"localhost:1"};

RemoteCompileCoordinator make_coordinator() {
    return RemoteCompileCoordinator(kUnusedEndpoints, DEFAULT_CONTEXT_ID, /*device_build_id=*/0, /*build_key=*/0);
}

}  // namespace

// The dedup cache is process-wide, so each test uses its own kernel hash.

TEST(RemoteCompileCoordinatorTest, KernelWithNothingToCompileIsReusedThenDeduplicated) {
    constexpr std::size_t kKernelHash = 0x5eed0001;
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

TEST(RemoteCompileCoordinatorTest, ClearDedupCacheMakesTheNextSubmissionOwnTheKernelAgain) {
    constexpr std::size_t kKernelHash = 0x5eed0002;
    int descriptor_calls = 0;
    const auto nothing_to_compile = [&]() -> std::optional<KernelCompileDescriptor> {
        ++descriptor_calls;
        return std::nullopt;
    };

    auto first = make_coordinator();
    EXPECT_EQ(first.submit(kKernelHash, nothing_to_compile), SubmitResult::Reused);
    EXPECT_NO_THROW(first.finish());

    RemoteCompileCoordinator::clear_dedup_cache();

    auto second = make_coordinator();
    EXPECT_EQ(second.submit(kKernelHash, nothing_to_compile), SubmitResult::Reused);
    EXPECT_NO_THROW(second.finish());

    EXPECT_EQ(descriptor_calls, 2);
}

TEST(RemoteCompileCoordinatorTest, FailedDescriptorDoesNotPoisonLaterSubmissions) {
    constexpr std::size_t kKernelHash = 0x5eed0003;

    auto failing = make_coordinator();
    EXPECT_THROW(
        failing.submit(
            kKernelHash,
            []() -> std::optional<KernelCompileDescriptor> { throw std::runtime_error("simulated genfiles failure"); }),
        std::runtime_error);

    auto retry = make_coordinator();
    EXPECT_EQ(
        retry.submit(kKernelHash, []() -> std::optional<KernelCompileDescriptor> { return std::nullopt; }),
        SubmitResult::Reused);
    EXPECT_NO_THROW(retry.finish());
}

}  // namespace tt::tt_metal
