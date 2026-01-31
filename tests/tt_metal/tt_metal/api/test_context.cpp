// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/context.hpp>
#include <tt-metalium/experimental/runtime.hpp>
#include <gtest/gtest.h>

#define ASSERT_ALL_INACTIVE(system_query, metal_instance) \
    ASSERT_FALSE(system_query.is_initialized());          \
    ASSERT_FALSE(metal_instance.has_bound_context());

TEST(ClusterQuery, TestInit) {
    // Open and close multiple times.
    auto& system_query = tt::tt_metal::experimental::ClusterQuery::instance();
    auto& metal_instance = tt::tt_metal::experimental::Runtime::instance();

    ASSERT_FALSE(metal_instance.has_bound_context());

    constexpr int num_iterations = 5;
    for (int i = 0; i < num_iterations; ++i) {
        ASSERT_FALSE(system_query.is_initialized());
        ASSERT_TRUE(system_query.initialize());
        ASSERT_TRUE(system_query.is_initialized());

        // Cannot initialize multiple times
        ASSERT_FALSE(system_query.initialize());
        ASSERT_TRUE(system_query.is_initialized());

        // Teardown
        ASSERT_TRUE(system_query.teardown());
        ASSERT_FALSE(system_query.is_initialized());
    }
}

TEST(ClusterQuery, TestTeardownBeforeInit) {
    auto& system_query = tt::tt_metal::experimental::ClusterQuery::instance();
    ASSERT_FALSE(system_query.is_initialized());
    ASSERT_FALSE(system_query.teardown());

    ASSERT_TRUE(system_query.initialize());
    ASSERT_TRUE(system_query.is_initialized());
    ASSERT_TRUE(system_query.teardown());
    ASSERT_FALSE(system_query.is_initialized());
}

TEST(ClusterQuery, TestBasicClusterQueries) {
    auto& system_query = tt::tt_metal::experimental::ClusterQuery::instance();
    ASSERT_TRUE(system_query.initialize());
    ASSERT_TRUE(system_query.is_initialized());
    ASSERT_GE(system_query.get_num_visible_devices(), 0);
    ASSERT_GE(system_query.get_num_pcie_devices(), 0);

    ASSERT_TRUE(system_query.teardown());
}

TEST(ClusterQuery, TestTeardownWithActiveContextInRuntime) {
    // Cannot teardown the ClusterQuery while a context is bound in the Runtime
    auto& system_query = tt::tt_metal::experimental::ClusterQuery::instance();
    auto& metal_instance = tt::tt_metal::experimental::Runtime::instance();

    ASSERT_TRUE(system_query.initialize());
    ASSERT_TRUE(system_query.is_initialized());
    ASSERT_TRUE(metal_instance.bind_context(
        std::make_shared<tt::tt_metal::experimental::Context>(1, 1, 1, 1, tt::tt_metal::DispatchCoreConfig{})));
    ASSERT_TRUE(metal_instance.has_bound_context());

    ASSERT_FALSE(system_query.teardown());

    // Unbind the context and try to teardown the query object
    ASSERT_TRUE(metal_instance.unbind_context());
    ASSERT_FALSE(metal_instance.has_bound_context());
    ASSERT_TRUE(system_query.teardown());
    ASSERT_FALSE(system_query.is_initialized());
}

TEST(Runtime, TestBindUnsuccessful) {
    // Context cannot be bound before the ClusterQuery is initialized
    auto& system_query = tt::tt_metal::experimental::ClusterQuery::instance();
    auto& metal_instance = tt::tt_metal::experimental::Runtime::instance();

    auto context =
        std::make_shared<tt::tt_metal::experimental::Context>(1, 1, 1, 1, tt::tt_metal::DispatchCoreConfig{});
    ASSERT_FALSE(metal_instance.bind_context(context));
    ASSERT_FALSE(metal_instance.has_bound_context());

    ASSERT_ALL_INACTIVE(system_query, metal_instance);
}

TEST(Runtime, TestUnbindUnsuccessful) {
    // Unbind without any active context
    auto& system_query = tt::tt_metal::experimental::ClusterQuery::instance();
    auto& metal_instance = tt::tt_metal::experimental::Runtime::instance();
    ASSERT_FALSE(metal_instance.unbind_context());
    ASSERT_ALL_INACTIVE(system_query, metal_instance);
}

TEST(Runtime, TestBindContext) {
    // Context can be bound only while the ClusterQuery is initialized
    auto& system_query = tt::tt_metal::experimental::ClusterQuery::instance();
    auto& metal_instance = tt::tt_metal::experimental::Runtime::instance();

    ASSERT_FALSE(system_query.is_initialized());
    system_query.initialize();
    ASSERT_TRUE(system_query.is_initialized());
    auto context =
        std::make_shared<tt::tt_metal::experimental::Context>(1, 1, 1, 1, tt::tt_metal::DispatchCoreConfig{});
    ASSERT_TRUE(metal_instance.bind_context(context));
    ASSERT_TRUE(metal_instance.has_bound_context());
    ASSERT_EQ(metal_instance.get_context(), context);
    ASSERT_TRUE(metal_instance.unbind_context());
    ASSERT_FALSE(metal_instance.has_bound_context());
    system_query.teardown();

    ASSERT_ALL_INACTIVE(system_query, metal_instance);
}

TEST(Runtime, TestUnbindNoBoundContext) {
    auto& system_query = tt::tt_metal::experimental::ClusterQuery::instance();
    auto& metal_instance = tt::tt_metal::experimental::Runtime::instance();

    ASSERT_FALSE(metal_instance.unbind_context());
    ASSERT_ALL_INACTIVE(system_query, metal_instance);
}
