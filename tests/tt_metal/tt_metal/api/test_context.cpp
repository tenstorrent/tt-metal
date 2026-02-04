// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/context_descriptor.hpp>
#include <tt-metalium/experimental/runtime.hpp>
#include <gtest/gtest.h>
#include <hostdevcommon/common_values.hpp>

#define ASSERT_ALL_INACTIVE(system_query, metal_instance) \
    ASSERT_FALSE(system_query.is_initialized());          \
    ASSERT_FALSE(metal_instance.has_descriptor());

TEST(MetaliumObject, TestInit) {
    // Open and close multiple times.
    auto& system_query = tt::tt_metal::experimental::MetaliumObject::instance();
    auto& metal_instance = tt::tt_metal::experimental::Context::instance();

    ASSERT_FALSE(metal_instance.has_descriptor());

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

TEST(MetaliumObject, TestTeardownBeforeInit) {
    auto& system_query = tt::tt_metal::experimental::MetaliumObject::instance();
    ASSERT_FALSE(system_query.is_initialized());
    ASSERT_FALSE(system_query.teardown());

    ASSERT_TRUE(system_query.initialize());
    ASSERT_TRUE(system_query.is_initialized());
    ASSERT_TRUE(system_query.teardown());
    ASSERT_FALSE(system_query.is_initialized());
}

TEST(MetaliumObject, TestBasicClusterQueries) {
    auto& system_query = tt::tt_metal::experimental::MetaliumObject::instance();
    ASSERT_TRUE(system_query.initialize());
    ASSERT_TRUE(system_query.is_initialized());
    ASSERT_GE(system_query.get_num_visible_devices(), 0);
    ASSERT_GE(system_query.get_num_pcie_devices(), 0);

    ASSERT_TRUE(system_query.teardown());
}

TEST(MetaliumObject, TestTeardownWithActiveContextInRuntime) {
    // Cannot teardown the MetaliumObject while a context is bound in the Context
    auto& system_query = tt::tt_metal::experimental::MetaliumObject::instance();
    auto& metal_instance = tt::tt_metal::experimental::Context::instance();

    ASSERT_TRUE(system_query.initialize());
    ASSERT_TRUE(system_query.is_initialized());
    auto context = std::make_shared<tt::tt_metal::experimental::ContextDescriptor>(/*num_cqs=*/1);
    ASSERT_TRUE(metal_instance.set_descriptor(context));
    ASSERT_TRUE(metal_instance.has_descriptor());

    ASSERT_FALSE(system_query.teardown());

    // Unbind the context and try to teardown the query object
    ASSERT_TRUE(metal_instance.remove_descriptor());
    ASSERT_FALSE(metal_instance.has_descriptor());
    ASSERT_TRUE(system_query.teardown());
    ASSERT_FALSE(system_query.is_initialized());
}

TEST(Context, TestBindUnsuccessful) {
    // ContextDescriptor cannot be bound before the MetaliumObject is initialized
    auto& system_query = tt::tt_metal::experimental::MetaliumObject::instance();
    auto& metal_instance = tt::tt_metal::experimental::Context::instance();

    auto context = std::make_shared<tt::tt_metal::experimental::ContextDescriptor>(/*num_cqs=*/1);
    ASSERT_FALSE(metal_instance.set_descriptor(context));
    ASSERT_FALSE(metal_instance.has_descriptor());

    ASSERT_ALL_INACTIVE(system_query, metal_instance);
}

TEST(Context, TestUnbindUnsuccessful) {
    // Unbind without any active context
    auto& system_query = tt::tt_metal::experimental::MetaliumObject::instance();
    auto& metal_instance = tt::tt_metal::experimental::Context::instance();
    ASSERT_FALSE(metal_instance.remove_descriptor());
    ASSERT_ALL_INACTIVE(system_query, metal_instance);
}

TEST(Context, TestBindContext) {
    // ContextDescriptor can be bound only while the MetaliumObject is initialized
    auto& system_query = tt::tt_metal::experimental::MetaliumObject::instance();
    auto& metal_instance = tt::tt_metal::experimental::Context::instance();

    ASSERT_FALSE(system_query.is_initialized());
    system_query.initialize();
    ASSERT_TRUE(system_query.is_initialized());
    auto context = std::make_shared<tt::tt_metal::experimental::ContextDescriptor>(/*num_cqs=*/1);
    ASSERT_TRUE(metal_instance.set_descriptor(context));
    ASSERT_TRUE(metal_instance.has_descriptor());
    ASSERT_TRUE(context->is_bound());
    ASSERT_EQ(metal_instance.get_descriptor(), context);
    ASSERT_TRUE(metal_instance.remove_descriptor());
    ASSERT_FALSE(metal_instance.has_descriptor());
    ASSERT_FALSE(context->is_bound());
    system_query.teardown();

    ASSERT_ALL_INACTIVE(system_query, metal_instance);
}

TEST(Context, TestUnbindNoBoundContext) {
    auto& system_query = tt::tt_metal::experimental::MetaliumObject::instance();
    auto& metal_instance = tt::tt_metal::experimental::Context::instance();

    ASSERT_FALSE(metal_instance.remove_descriptor());
    ASSERT_ALL_INACTIVE(system_query, metal_instance);
}

TEST(Context, TestBindContextWithFabricConfig) {
    auto& system_query = tt::tt_metal::experimental::MetaliumObject::instance();
    auto& metal_instance = tt::tt_metal::experimental::Context::instance();

    ASSERT_TRUE(system_query.initialize());

    auto context = std::make_shared<tt::tt_metal::experimental::ContextDescriptor>(
        1,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        DEFAULT_WORKER_L1_SIZE,
        tt::tt_metal::DispatchCoreConfig{},
        tt::tt_fabric::FabricConfig::FABRIC_1D);
    ASSERT_TRUE(metal_instance.set_descriptor(context));
    ASSERT_TRUE(metal_instance.has_descriptor());

    ASSERT_TRUE(system_query.teardown());
}
