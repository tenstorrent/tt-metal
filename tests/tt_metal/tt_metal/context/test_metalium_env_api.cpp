// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/experimental/context/context_descriptor.hpp>
#include <tt-metalium/experimental/context/metalium_env.hpp>
#include <tt-metalium/experimental/hal.hpp>

#include "impl/device/mock_device_util.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "impl/context/metalium_env_accessor.hpp"

namespace tt::tt_metal {

TEST(MetaliumEnv, Physical) {
    auto env = MetaliumEnv();
    EXPECT_TRUE(env.is_initialized());
}

TEST(MetaliumEnv, Mock) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value();
    auto env_1 = MetaliumEnvAccessor(std::make_unique<MetaliumEnv>(MetaliumEnvDescriptor(mock_path)));
    auto env_2 = MetaliumEnvAccessor(std::make_unique<MetaliumEnv>(MetaliumEnvDescriptor(mock_path)));
    EXPECT_TRUE(env_1.get_metalium_env().is_initialized());
    EXPECT_TRUE(env_2.get_metalium_env().is_initialized());
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

    EXPECT_EQ(MetaliumEnvAccessor(env_mock_1).get_cluster().arch(), tt::ARCH::BLACKHOLE);
    EXPECT_EQ(MetaliumEnvAccessor(env_mock_2).get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
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

TEST(MetaliumEnv, HalFunctions) {
    MetaliumEnv env;
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_arch(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_arch_name(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_l1_size(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_dram_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_l1_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_pcie_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_base(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_size(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_max_worker_l1_unreserved_size(env));
}

TEST(MetaliumEnv, HalFunctionsMock) {
    auto env_settings = MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2));
    MetaliumEnv env(env_settings);
    EXPECT_EQ(tt::tt_metal::experimental::hal::get_arch(env), tt::ARCH::BLACKHOLE);
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_l1_size(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_dram_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_l1_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_pcie_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_base(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_size(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_max_worker_l1_unreserved_size(env));
}

}  // namespace tt::tt_metal
