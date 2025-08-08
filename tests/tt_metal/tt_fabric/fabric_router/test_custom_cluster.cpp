// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <typeinfo>

#include <umd/device/cluster.h>
#include <umd/device/tt_cluster_descriptor.h>
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {

namespace mock_cluster_tests {

class ClusterFixture : public ::testing::Test {
protected:
    std::unique_ptr<tt_ClusterDescriptor> cluster_desc;

    void SetUp() override { printf("ClusterFixture SetUp\n"); }

    void TearDown() override { printf("ClusterFixture TearDown\n"); }
};

TEST_F(ClusterFixture, TestCustomCluster) {
    std::unique_ptr<tt::umd::Cluster> device_driver;

    std::unique_ptr<tt_ClusterDescriptor> cluster_desc =
        tt_ClusterDescriptor::create_from_yaml("../t3k_cluster_desc.yaml");

    device_driver = std::make_unique<tt::umd::Cluster>(
        tt::umd::ClusterOptions{.chip_type = tt::umd::ChipType::MOCK, .cluster_descriptor = cluster_desc.get()});

    printf("You should see this\n");
}

}  // namespace mock_cluster_tests
}  // namespace tt::tt_fabric
