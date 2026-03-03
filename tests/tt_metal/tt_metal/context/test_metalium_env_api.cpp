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
    auto env = MetaliumEnv();
    EXPECT_TRUE(env.is_initialized());
}

TEST(MetaliumEnv, Mock) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value();
    auto env_1 = MetaliumEnv(MetaliumEnvDescriptor(mock_path));
    auto env_2 = MetaliumEnv(MetaliumEnvDescriptor(mock_path));
    EXPECT_TRUE(env_1.is_initialized());
    EXPECT_TRUE(env_2.is_initialized());
    EXPECT_EQ(env_1.get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
    EXPECT_EQ(env_2.get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
}

TEST(MetaliumEnv, OnePhysicalMultipleMock) {
    auto env_physical = MetaliumEnv();

    auto mock_path_1 = experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2).value();
    auto env_mock_1 = MetaliumEnv(MetaliumEnvDescriptor(mock_path_1));

    auto mock_path_2 = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2).value();
    auto env_mock_2 = MetaliumEnv(MetaliumEnvDescriptor(mock_path_2));

    EXPECT_TRUE(env_physical.is_initialized());
    EXPECT_TRUE(env_mock_1.is_initialized());
    EXPECT_TRUE(env_mock_2.is_initialized());

    EXPECT_EQ(env_mock_1.get_cluster().arch(), tt::ARCH::BLACKHOLE);
    EXPECT_EQ(env_mock_2.get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
}

TEST(MetaliumEnv, Destroy) {
    auto env = MetaliumEnv();
    env.destroy();
    EXPECT_FALSE(env.is_initialized());
}

TEST(MetaliumEnv, DestroyMultiple) {
    auto env = MetaliumEnv();
    env.destroy();
    env.destroy();
    env.destroy();
    EXPECT_FALSE(env.is_initialized());
}

}  // namespace tt::tt_metal
