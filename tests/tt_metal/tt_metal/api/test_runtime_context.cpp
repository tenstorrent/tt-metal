// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hostdevcommon/common_values.hpp"
#include "impl/context/experimental/runtime.hpp"
#include "impl/context/experimental/context.hpp"
#include <gtest/gtest.h>

TEST(MetaliumObject, CreateAndDestroy) {
    static constexpr int num_iterations = 10;

    // This will hang if the previous object is not destroyed properly
    for (int i = 0; i < num_iterations; i++) {
        auto obj = tt::tt_metal::experimental::MetaliumObject::create();
    }
}

TEST(Context, CreateAndDestroySiliconContext) {
    auto metalium = tt::tt_metal::experimental::MetaliumObject::create();
    using Context = tt::tt_metal::experimental::Context;

    auto descriptor = std::make_shared<tt::tt_metal::experimental::ContextDescriptor>();

    // User creates and owns the silicon context, passing MetaliumObject as dependency
    Context ctx(descriptor, metalium);
    EXPECT_FALSE(ctx.is_mock_device());
    EXPECT_NE(ctx.get_metalium_object(), nullptr);
    EXPECT_EQ(ctx.get_descriptor(), descriptor);
}

TEST(Context, CreateAndDestroyMockDeviceContext) {
    using Context = tt::tt_metal::experimental::Context;

    auto descriptor = std::make_shared<tt::tt_metal::experimental::ContextDescriptor>(
        /*num_cqs=*/1,
        /*l1_small_size=*/DEFAULT_L1_SMALL_SIZE,
        /*trace_region_size=*/DEFAULT_TRACE_REGION_SIZE,
        /*worker_l1_size=*/DEFAULT_WORKER_L1_SIZE,
        /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig{},
        /*is_mock_device=*/true);

    // Mock context does not need MetaliumObject
    Context ctx(descriptor);
    EXPECT_TRUE(ctx.is_mock_device());
    EXPECT_EQ(ctx.get_metalium_object(), nullptr);
}

TEST(Context, MultipleMockContextsAllowed) {
    using Context = tt::tt_metal::experimental::Context;

    auto mock_descriptor = std::make_shared<tt::tt_metal::experimental::ContextDescriptor>(
        /*num_cqs=*/1,
        /*l1_small_size=*/DEFAULT_L1_SMALL_SIZE,
        /*trace_region_size=*/DEFAULT_TRACE_REGION_SIZE,
        /*worker_l1_size=*/DEFAULT_WORKER_L1_SIZE,
        /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig{},
        /*is_mock_device=*/true);

    // User creates and owns multiple mock contexts directly
    Context ctx1(mock_descriptor);
    Context ctx2(mock_descriptor);
    Context ctx3(mock_descriptor);

    EXPECT_TRUE(ctx1.is_mock_device());
    EXPECT_TRUE(ctx2.is_mock_device());
    EXPECT_TRUE(ctx3.is_mock_device());
}

TEST(Context, SiliconAndMockContextsSimultaneously) {
    auto metalium = tt::tt_metal::experimental::MetaliumObject::create();
    using Context = tt::tt_metal::experimental::Context;

    // Create a silicon context with explicit MetaliumObject dependency
    auto silicon_descriptor = std::make_shared<tt::tt_metal::experimental::ContextDescriptor>();
    Context silicon_ctx(silicon_descriptor, metalium);

    // Create multiple mock device contexts - they don't need MetaliumObject
    auto mock_descriptor = std::make_shared<tt::tt_metal::experimental::ContextDescriptor>(
        /*num_cqs=*/1,
        /*l1_small_size=*/DEFAULT_L1_SMALL_SIZE,
        /*trace_region_size=*/DEFAULT_TRACE_REGION_SIZE,
        /*worker_l1_size=*/DEFAULT_WORKER_L1_SIZE,
        /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig{},
        /*is_mock_device=*/true);

    Context mock_ctx1(mock_descriptor);
    Context mock_ctx2(mock_descriptor);

    // Verify all contexts have correct types
    EXPECT_FALSE(silicon_ctx.is_mock_device());
    EXPECT_TRUE(mock_ctx1.is_mock_device());
    EXPECT_TRUE(mock_ctx2.is_mock_device());

    // Silicon context should have access to MetaliumObject
    EXPECT_NE(silicon_ctx.get_metalium_object(), nullptr);
    EXPECT_EQ(mock_ctx1.get_metalium_object(), nullptr);
    EXPECT_EQ(mock_ctx2.get_metalium_object(), nullptr);
}
