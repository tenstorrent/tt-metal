// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Correctness and throughput for HostMeshSocket. World size 2: rank 0 sends.

#include <gtest/gtest.h>

#include <tt-metalium/distributed_context.hpp>

#include "host_socket_test_utils.hpp"

namespace tt::tt_metal::distributed::host_socket_test {
namespace {

void require_two_ranks() {
    const auto& context = multihost::DistributedContext::get_current_world();
    ASSERT_NE(context, nullptr) << "DistributedContext is not initialized";
    ASSERT_EQ(*context->size(), 2) << "HostMeshSocket tests need exactly 2 ranks";
}

TEST(HostSocketTest, SingleCoreCorrectness) {
    require_two_ranks();
    Params params = params_from_env();
    params.num_cores = 1;
    run_transfer(params, /*verify=*/true);
}

TEST(HostSocketTest, MultiCoreCorrectness) {
    require_two_ranks();
    Params params = params_from_env(Params{.num_cores = 8});
    run_transfer(params, /*verify=*/true);
}

// Includes non-power-of-two sizes: the FIFO pointer arithmetic must be modular.
TEST(HostSocketTest, PageSizeSweepCorrectness) {
    require_two_ranks();
    for (uint32_t page_size : {2048u, 4096u, 14336u, 32768u}) {
        Params params = params_from_env();
        params.page_size = page_size;
        params.num_cores = 2;
        run_transfer(params, /*verify=*/true);
        if (::testing::Test::HasFatalFailure()) {
            return;
        }
    }
}

// Many laps, so pointer and counter wrap are exercised.
TEST(HostSocketTest, RingWrapCorrectness) {
    require_two_ranks();
    Params params = params_from_env();
    params.num_cores = 2;
    params.iterations = 64;
    run_transfer(params, /*verify=*/true);
}

TEST(HostSocketTest, Throughput) {
    require_two_ranks();
    // Enough volume that setup is not measured.
    Params params = params_from_env(Params{
        .page_size = 14336,
        .fifo_pages = 64,
        .num_cores = 1,
        .bytes_per_core = 14336ull * 64 * 64,
        .iterations = 16,
    });
    double gbps = 0.0;
    run_transfer(params, /*verify=*/false, &gbps);

    const auto& context = multihost::DistributedContext::get_current_world();
    if (context->rank() == kSenderRank) {
        GTEST_LOG_(INFO) << "HostMeshSocket throughput: " << gbps << " GB/s at " << params.page_size
                         << " B pages across " << params.num_cores << " core(s)";
        record_result(params, gbps);
        // Informational unless a floor is set.
        const char* floor_env = std::getenv("TT_HOST_SOCKET_MIN_GBPS");
        if (floor_env != nullptr && *floor_env != '\0') {
            EXPECT_GE(gbps, std::strtod(floor_env, nullptr));
        }
    }
}

// Inert unless TT_HOST_SOCKET_SOAK_SECONDS is set.
TEST(HostSocketTest, Soak) {
    require_two_ranks();
    Params params = params_from_env(Params{.num_cores = 4});
    if (params.min_seconds <= 0.0) {
        GTEST_SKIP() << "set TT_HOST_SOCKET_SOAK_SECONDS to run the soak";
    }
    double gbps = 0.0;
    run_transfer(params, /*verify=*/true, &gbps);
    const auto& context = multihost::DistributedContext::get_current_world();
    if (context->rank() == kSenderRank) {
        GTEST_LOG_(INFO) << "soak sustained " << gbps << " GB/s over " << params.min_seconds << " s";
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::host_socket_test
