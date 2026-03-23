// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <memory>
#include <thread>
#include <type_traits>

#include <tt-metalium/distributed_context.hpp>
#include <fmt/format.h>
namespace multihost::common {

using ContextPtr = std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>;
using ReduceOp = tt::tt_metal::distributed::multihost::ReduceOp;
using tt::tt_metal::distributed::multihost::dtype_of_v;
using tt::tt_metal::distributed::multihost::is_supported_dtype_v;
//----------------------------------------------------------------------
//  internal helpers
//----------------------------------------------------------------------

template <typename T>
inline tt::stl::Span<std::byte> as_bytes_single(T& obj) noexcept {
    return {reinterpret_cast<std::byte*>(std::addressof(obj)), sizeof(T)};
}

//  all_reduce wrapper on a single value (send == recv for MAX / MIN)

template <typename T>
    requires is_supported_dtype_v<T>
T all_reduce_value(const T& local, ReduceOp op, const ContextPtr& ctx) {
    T send = local;
    T recv{};
    ctx->all_reduce(as_bytes_single(send), as_bytes_single(recv), op, dtype_of_v<T>);
    return recv;
}

template <typename T>
void expect_equal_all_ranks(const T& local, const ContextPtr& ctx) {
    const auto global_max = all_reduce_value(local, ReduceOp::MAX, ctx);
    const auto global_min = all_reduce_value(local, ReduceOp::MIN, ctx);
    EXPECT_EQ(global_max, global_min) << "Rank " << *ctx->rank() << " sees " << local
                                      << ", but global min=" << global_min << " max=" << global_max;
}

template <typename T>
void assert_equal_all_ranks(const T& local, const ContextPtr& ctx) {
    const auto global_max = all_reduce_value(local, ReduceOp::MAX, ctx);
    const auto global_min = all_reduce_value(local, ReduceOp::MIN, ctx);
    ASSERT_EQ(global_max, global_min) << "Rank " << *ctx->rank() << " sees " << local
                                      << ", but global min=" << global_min << " max=" << global_max;
}

template <typename T>
void expect_near_all_ranks(const T& local, const ContextPtr& ctx, T abs_tol) {
    static_assert(std::is_floating_point_v<T>, "EXPECT_NEAR_ALL_RANKS requires float/double");
    const auto global_max = all_reduce_value(local, ReduceOp::MAX, ctx);
    const auto global_min = all_reduce_value(local, ReduceOp::MIN, ctx);
    EXPECT_NEAR(global_max, global_min, abs_tol)
        << "Rank " << *ctx->rank() << " sees " << local << ", but global min=" << global_min << " max=" << global_max;
}

template <typename BoolLike>
void assert_true_all_ranks(const BoolLike& cond, const ContextPtr& ctx) {
    int local = static_cast<int>(cond ? 1 : 0);
    int global = 0;
    ctx->all_reduce(
        as_bytes_single(local),
        as_bytes_single(global),
        ReduceOp::LAND,
        tt::tt_metal::distributed::multihost::DType::INT32);
    ASSERT_EQ(global, 1) << "Rank " << *ctx->rank() << " violated collective TRUE condition.";
}

inline void barrier(const ContextPtr& ctx) { ctx->barrier(); }

//----------------------------------------------------------------------
//  Google‑Test macro sugar
//----------------------------------------------------------------------

#define EXPECT_EQ_ALL_RANKS(val, ctx) ::multihost::common::expect_equal_all_ranks((val), (ctx))
#define ASSERT_EQ_ALL_RANKS(val, ctx) ::multihost::common::assert_equal_all_ranks((val), (ctx))
#define EXPECT_NEAR_ALL_RANKS(val, ctx, tol) ::multihost::common::expect_near_all_ranks((val), (ctx), (tol))
#define ASSERT_TRUE_ALL_RANKS(cond, ctx) ::multihost::common::assert_true_all_ranks((cond), (ctx))
#define BARRIER(ctx) ::multihost::common::barrier((ctx))
#define RANK0_PRINT(ctx, ...)         \
    \ do {                            \
        if ((ctx)->rank().get() == 0) \
            fmt::print(__VA_ARGS__);  \
    }                                 \
    while (0)

// ---------------------------------------------------------------------------
//  Fault-tolerance test helpers
//
//  kill_rank_and_recover():
//    Encapsulates the recurring pattern in fault-tolerance tests:
//      1. The victim rank sleeps briefly then calls raise(SIGKILL).
//      2. Surviving ranks execute a collective (barrier), catch any
//         DistributedException (including MPIRankFailureException), and
//         call revoke_and_shrink() to produce a healthy shrunken communicator.
//      3. After recovery, `post_recovery(ctx)` is invoked on all survivors.
//
//    Use this for tests whose interesting assertions come AFTER the shrink.
//    For tests that must inspect the caught exception itself (e.g.
//    MPIRankFailureExceptionCarriesContext), keep a manual catch block.
// ---------------------------------------------------------------------------

template <typename Fn>
inline void kill_rank_and_recover(
    const ContextPtr& ctx,
    int victim_rank,
    Fn&& post_recovery,
    std::chrono::milliseconds kill_delay = std::chrono::milliseconds(200)) {
    const int rank = *ctx->rank();
    if (rank == victim_rank) {
        std::this_thread::sleep_for(kill_delay);
        raise(SIGKILL);  // never returns; only this process exits
    }
    try {
        ctx->barrier();
    } catch (const tt::tt_metal::distributed::multihost::DistributedException&) {
        ctx->revoke_and_shrink();
    }
    std::forward<Fn>(post_recovery)(ctx);
}

// ---------------------------------------------------------------------------
//  Fault-tolerant test macros (Section 4.4 of ulfm-rank-reinit-architecture.md)
//
//  EXPECT_EQ_SURVIVING_RANKS / ASSERT_EQ_SURVIVING_RANKS:
//    Like the ALL_RANKS variants but semantically indicate that the
//    communicator has been through revoke_and_shrink() and may have
//    fewer ranks than the original world.  Functionally identical to
//    the ALL_RANKS macros (all_reduce on the *current* communicator),
//    but named differently for readability in fault-tolerance tests.
//
//  SKIP_IF_NO_ULFM(ctx):
//    Skip the current test if ULFM support is not available in this build.
//    Use at the start of any test that exercises ULFM-specific APIs
//    (revoke, shrink, agree, etc.).
// ---------------------------------------------------------------------------

#define EXPECT_EQ_SURVIVING_RANKS(val, ctx) ::multihost::common::expect_equal_all_ranks((val), (ctx))
#define ASSERT_EQ_SURVIVING_RANKS(val, ctx) ::multihost::common::assert_equal_all_ranks((val), (ctx))

#define SKIP_IF_NO_ULFM(ctx) \
    do { \
        if (!(ctx)->supports_fault_tolerance()) { \
            GTEST_SKIP() << "ULFM support not available in this build"; \
        } \
    } while (0)

// ----------------------------------------------------------------------
//  multihost test main function
// ----------------------------------------------------------------------
inline int multihost_main(int argc, char** argv) {
    tt::tt_metal::distributed::multihost::DistributedContext::create(argc, argv);

    // Parse argv/env into gtest flags before reading --output (GTEST_OUTPUT).
    ::testing::InitGoogleTest(&argc, argv);

    // If GTEST_OUTPUT is set to a directory, add the rank to the path to make it unique
    std::string gtest_output_str = GTEST_FLAG_GET(output);
    // Docker / shell sometimes leaves wrapping quotes in the value, which breaks the
    // "xml:path" prefix and triggers "unrecognized output format" warnings.
    while (!gtest_output_str.empty() && gtest_output_str.front() == '"') {
        gtest_output_str.erase(0, 1);
    }
    while (!gtest_output_str.empty() && gtest_output_str.back() == '"') {
        gtest_output_str.pop_back();
    }
    if (!gtest_output_str.empty()) {
        const size_t colon = gtest_output_str.find(':');

        // Split into prefix (up to and including colon) and path
        std::string prefix;
        std::string path_part;
        if (colon != std::string::npos) {
            prefix = gtest_output_str.substr(0, colon + 1);  // includes colon
            path_part = gtest_output_str.substr(colon + 1);
        } else {
            path_part = gtest_output_str;
        }

        std::filesystem::path path(path_part);
        bool is_dir_like =
            std::filesystem::exists(path) ? std::filesystem::is_directory(path) : path.extension().empty();
        if (is_dir_like) {
            path /=
                std::to_string(*tt::tt_metal::distributed::multihost::DistributedContext::get_current_world()->rank());
        }

        // Prepend the prefix (e.g., "xml:") back to the final path
        std::string final_output = prefix + path.string() + "/";
        GTEST_FLAG_SET(output, final_output);
    }

    const auto& ctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();

    // Skip all tests if lacking fault tolerance.
    if (!ctx->supports_fault_tolerance()) {
        fmt::println(
            "Fault tolerance support is not available in this build. Skipping fault tolerance tests. "
            "Fault tolerance support via ULFM is consistently available in builds of OpenMPI from version 5.0. "
            "If your distribution does not have OpenMPI 5.0 packaged, you may be able to obtain a known-good build "
            "by running install_dependencies.sh with --distributed passed and building with"
            "build_metal.sh --enable-distributed.\n");
        return 0;
    }

    // Run tests on every rank
    int local_rc = RUN_ALL_TESTS();

    // need to make sure that  we get context after the tests, old one could be revoked
    const auto& context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    fmt::print("Rank {}: local rc = {}\n", *context->rank(), local_rc);
    // Propagate the worst return code to all ranks

    ASSERT_EQ_ALL_RANKS(local_rc, context);

    return local_rc;
}

}  // namespace multihost::common
