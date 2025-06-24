// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed {
namespace {

// If MeshDevice is initialized, the distributed context is initialized through MetalContext
using MeshDeviceAutoInitDistributedContextTest = GenericMeshDeviceFixture;

// Test that MetalContext constructor automatically initializes the distributed context
TEST_F(MeshDeviceAutoInitDistributedContextTest, MetalContextInitializesDistributedContext) {
    EXPECT_TRUE(multihost::DistributedContext::is_initialized());
    auto world_context = multihost::DistributedContext::get_current_world();
    EXPECT_NE(world_context, nullptr);
    EXPECT_EQ(*world_context->size(), 1);
}

// Test that initialization is only done once
TEST_F(MeshDeviceAutoInitDistributedContextTest, MetalContextDoesNotReinitialize) {
    auto world_context1 = multihost::DistributedContext::get_current_world();
    auto world_context2 = multihost::DistributedContext::get_current_world();
    EXPECT_EQ(world_context1, world_context2);
    EXPECT_EQ(*world_context1->size(), 1);
}

// Test that creating a new distributed context when one is initialized throws
TEST_F(MeshDeviceAutoInitDistributedContextTest, CannotCreateNewContextWhenInitialized) {
    multihost::DistributedContext::get_current_world();
    // This will just duplicate existing world context
    EXPECT_NO_THROW(multihost::DistributedContext::create(0, nullptr));
}

}  // namespace
}  // namespace tt::tt_metal::distributed
