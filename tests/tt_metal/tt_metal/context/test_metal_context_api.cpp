// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "device/mock_device_common.hpp"
#include "impl/context/metallium_object.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/context_id.hpp"
#include "tt_cluster.hpp"

namespace tt::tt_metal {

class MetalContextTest : public ::testing::Test {
protected:
    void TearDown() override { MetalContext::destroy_all_instances(); }
};

TEST_F(MetalContextTest, CreateSiliconInstance) {
    auto descriptor = std::make_shared<MetalliumObjectDescriptor>();
    ContextId context_id = MetalContext::create_instance(descriptor);
    EXPECT_EQ(context_id, SILICON_CONTEXT_ID);
}

TEST_F(MetalContextTest, MultipleSiliconInstances) {
    auto descriptor = std::make_shared<MetalliumObjectDescriptor>();
    ContextId context_id = MetalContext::create_instance(descriptor);
    EXPECT_EQ(context_id, SILICON_CONTEXT_ID);
    // Only one silicon instance is allowed
    EXPECT_THROW(MetalContext::create_instance(descriptor), std::runtime_error);
}

TEST_F(MetalContextTest, LegacyImplicitSiliconInstance) {
    // Implicit init to support legacy behaviour
    auto& context = MetalContext::instance();
    EXPECT_FALSE(context.rtoptions().get_mock_enabled());
}

TEST_F(MetalContextTest, CreateMockInstances) {
    auto descriptor = std::make_shared<MetalliumObjectDescriptor>(
        experimental::get_mock_cluster_desc_for_config(tt::ARCH::WORMHOLE_B0, 1).value());
    ContextId context_id_wh = MetalContext::create_instance(descriptor);
    EXPECT_EQ(context_id_wh, 1);

    descriptor = std::make_shared<MetalliumObjectDescriptor>(
        experimental::get_mock_cluster_desc_for_config(tt::ARCH::BLACKHOLE, 1).value());
    ContextId context_id_bh = MetalContext::create_instance(descriptor);
    EXPECT_EQ(context_id_bh, 2);
}

TEST_F(MetalContextTest, CreateSiliconInstanceWithMockInstances) {
    auto descriptor = std::make_shared<MetalliumObjectDescriptor>();
    ContextId context_id = MetalContext::create_instance(descriptor);
    EXPECT_EQ(context_id, SILICON_CONTEXT_ID);

    auto descriptor_wh = std::make_shared<MetalliumObjectDescriptor>(
        experimental::get_mock_cluster_desc_for_config(tt::ARCH::WORMHOLE_B0, 1).value());
    ContextId context_id_wh = MetalContext::create_instance(descriptor_wh);

    auto descriptor_bh = std::make_shared<MetalliumObjectDescriptor>(
        experimental::get_mock_cluster_desc_for_config(tt::ARCH::BLACKHOLE, 1).value());
    ContextId context_id_bh = MetalContext::create_instance(descriptor_bh);

    ASSERT_EQ(MetalContext::instance(context_id_wh).get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
    ASSERT_EQ(MetalContext::instance(context_id_bh).get_cluster().arch(), tt::ARCH::BLACKHOLE);
}

TEST_F(MetalContextTest, DestroyInstanceExplicit) {
    auto descriptor_wh = std::make_shared<MetalliumObjectDescriptor>(
        experimental::get_mock_cluster_desc_for_config(tt::ARCH::WORMHOLE_B0, 1).value());
    ContextId context_id_wh = MetalContext::create_instance(descriptor_wh);
    ASSERT_EQ(MetalContext::instance(context_id_wh).get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
    MetalContext::destroy_instance(context_id_wh);
    EXPECT_THROW(MetalContext::instance(context_id_wh), std::runtime_error);
}

TEST_F(MetalContextTest, CreateImplicitAfterDestroy) {
    auto descriptor = std::make_shared<MetalliumObjectDescriptor>();
    ContextId context_id = MetalContext::create_instance(descriptor);
    MetalContext::destroy_instance(context_id);
    auto& context = MetalContext::instance(context_id);
    EXPECT_EQ(context.rtoptions().get_mock_enabled(), false);
}

}  // namespace tt::tt_metal
