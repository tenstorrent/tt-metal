// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <typeinfo>

#include "umd/device/cluster.h"
#include "umd/device/tt_cluster_descriptor.h"
#include "llrt/rtoptions.hpp"
#include <tt-metalium/control_plane.hpp>
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
    std::unique_ptr<tt_ClusterDescriptor> cluster_desc =
        tt_ClusterDescriptor::create_from_yaml("./t3k_cluster_desc.yaml");

   tt::llrt::RunTimeOptions rtoptions;
   tt::tt_metal::Hal hal(tt::ARCH::WORMHOLE_B0, true);

   auto cluster = std::make_unique<tt::Cluster>(cluster_desc.get(), rtoptions, hal);
   tt::tt_metal::MetalContext::set_default_cluster(std::move(cluster));

    tt::tt_metal::MetalContext::instance();

    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/custom_2x2_mesh_graph_descriptor.yaml";

    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(mesh_graph_desc_path.string());
}

}  // namespace mock_cluster_tests
}  // namespace tt::tt_fabric
