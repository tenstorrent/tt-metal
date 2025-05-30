// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>
#include <memory>
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

// ----------------------------------------------------------------------
//  multihost test main function
// ----------------------------------------------------------------------
inline int multihost_main(int argc, char** argv) {
    tt::tt_metal::distributed::multihost::DistributedContext::create(argc, argv);
    using Rank = tt::tt_metal::distributed::multihost::Rank;
    using Tag = tt::tt_metal::distributed::multihost::Tag;

    ::testing::InitGoogleTest(&argc, argv);

    auto ctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();

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
    auto context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    fmt::print("Rank {}: local rc = {}\n", *context->rank(), local_rc);
    // Propagate the worst return code to all ranks

    ASSERT_EQ_ALL_RANKS(local_rc, context);

    return local_rc;
}

}  // namespace multihost::common
