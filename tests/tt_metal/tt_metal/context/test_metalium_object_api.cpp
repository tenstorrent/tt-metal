// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "impl/context/context_descriptor.hpp"
#include "impl/context/metalium_env.hpp"
#include "impl/device/mock_device_util.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal {

TEST(MetaliumEnv, Physical) {
    auto descriptor = std::make_shared<MetaliumEnvDescriptor>();
    auto metallium_object = MetaliumEnv();
    metallium_object.initialize(descriptor);
    EXPECT_TRUE(metallium_object.is_initialized());
}

TEST(MetaliumEnv, Mock) {
    // Multiple mocks can be created without a hang
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value();
    auto descriptor = std::make_shared<MetaliumEnvDescriptor>(mock_path);
    auto metallium_object_1 = MetaliumEnv();
    metallium_object_1.initialize(descriptor);
    auto metallium_object_2 = MetaliumEnv();
    metallium_object_2.initialize(descriptor);
    EXPECT_TRUE(metallium_object_1.is_initialized());
    EXPECT_TRUE(metallium_object_2.is_initialized());
    EXPECT_EQ(metallium_object_1.get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
    EXPECT_EQ(metallium_object_2.get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
}

TEST(MetaliumEnv, OnePhysicalMultipleMock) {
    // Check a physical can be created alongside multiple mocks without a hang
    auto descriptor = std::make_shared<MetaliumEnvDescriptor>();

    auto mock_path_1 = experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2).value();
    auto mock_descriptor_1 = std::make_shared<MetaliumEnvDescriptor>(mock_path_1);

    auto mock_path_2 = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2).value();
    auto mock_descriptor_2 = std::make_shared<MetaliumEnvDescriptor>(mock_path_2);

    auto metallium_object_1 = MetaliumEnv();
    metallium_object_1.initialize(descriptor);
    auto metallium_object_2 = MetaliumEnv();
    metallium_object_2.initialize(mock_descriptor_1);
    auto metallium_object_3 = MetaliumEnv();
    metallium_object_3.initialize(mock_descriptor_2);
    EXPECT_TRUE(metallium_object_1.is_initialized());
    EXPECT_TRUE(metallium_object_2.is_initialized());
    EXPECT_TRUE(metallium_object_3.is_initialized());

    EXPECT_EQ(metallium_object_2.get_cluster().arch(), tt::ARCH::BLACKHOLE);
    EXPECT_EQ(metallium_object_3.get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
}

TEST(MetaliumEnv, Destroy) {
    auto descriptor = std::make_shared<MetaliumEnvDescriptor>();
    auto metallium_object = MetaliumEnv();
    metallium_object.initialize(descriptor);
    metallium_object.destroy();
    EXPECT_FALSE(metallium_object.is_initialized());
}

TEST(MetaliumEnv, DestroyMultiple) {
    auto descriptor = std::make_shared<MetaliumEnvDescriptor>();
    auto metallium_object = MetaliumEnv();
    metallium_object.initialize(descriptor);
    metallium_object.destroy();
    metallium_object.destroy();
    metallium_object.destroy();
    EXPECT_FALSE(metallium_object.is_initialized());
}

TEST(MetaliumEnv, InitMultipleTimesThrows) {
    auto descriptor = std::make_shared<MetaliumEnvDescriptor>();
    auto metallium_object = MetaliumEnv();
    metallium_object.initialize(descriptor);
    EXPECT_THROW(metallium_object.initialize(descriptor), std::runtime_error);
}

}  // namespace tt::tt_metal
