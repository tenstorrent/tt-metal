// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression: mock mesh + fabric (descriptor, post-hoc set_fabric_config, legacy configure_mock_mode).

#include <gtest/gtest.h>

#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/shape.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"

#include <ttnn/graph/graph_query_op_constraints.hpp>
#include <ttnn/operations/ccl/all_gather/all_gather.hpp>
#include <ttnn/tensor/layout/page_config.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>
#include <ttnn/tensor/tensor_ops.hpp>
#include <ttnn/tensor/tensor_spec.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/types.hpp>

#include <limits>
#include <optional>

namespace tt::tt_metal {
namespace {

ttnn::TensorSpec MakeInputSpec() {
    return ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 7 * 32}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
}

ttnn::graph::ConstraintQueryResponse RunAllGatherConstraintQuery(distributed::MeshDevice* device) {
    auto sharded_topology = TensorTopology::create_sharded_tensor_topology(device->shape(), /*shard_dim=*/0);
    ttnn::graph::DistributedTensorSpec dist_input{MakeInputSpec(), sharded_topology};

    return ttnn::graph::query_op_constraints(
        [](auto&&... args) { return ttnn::all_gather(std::forward<decltype(args)>(args)...); },
        device,
        dist_input,
        /*dim=*/3,
        /*cluster_axis=*/std::optional<uint32_t>(1),
        /*subdevice_id=*/std::optional<SubDeviceId>{},
        /*memory_config=*/std::optional<MemoryConfig>{},
        /*optional_output_tensor=*/std::optional<::ttnn::Tensor>{},
        /*num_links=*/std::optional<uint32_t>(1),
        /*topology=*/std::optional<tt_fabric::Topology>(tt_fabric::Topology::Linear));
}

TEST(MetalEnvMockCCL, FabricInDescriptor_CreatesMeshAndQuerySucceeds) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2);
    ASSERT_TRUE(mock_path.has_value());

    FabricConfigDescriptor fabric_desc{};
    fabric_desc.fabric_config = tt_fabric::FabricConfig::FABRIC_1D;
    fabric_desc.reliability_mode = tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE;
    fabric_desc.num_routing_planes = std::numeric_limits<uint8_t>::max();
    MetalEnv env{MetalEnvDescriptor{*mock_path, fabric_desc}};

    auto device = env.create_mesh_device(distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
    ASSERT_NE(device, nullptr);
    auto response = RunAllGatherConstraintQuery(device.get());
    EXPECT_EQ(response.status, ttnn::graph::ExecutionStatus::Success)
        << "query failed: " << response.error_message.value_or("(no message)");
}

TEST(MetalEnvMockCCL, FabricAfterDeviceCreation_QuerySucceeds) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2);
    ASSERT_TRUE(mock_path.has_value());

    MetalEnv env{MetalEnvDescriptor{*mock_path}};
    auto device = env.create_mesh_device(distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
    ASSERT_NE(device, nullptr);

    MetalEnvAccessor{env}.impl().set_fabric_config(
        tt_fabric::FabricConfig::FABRIC_1D, tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);

    auto response = RunAllGatherConstraintQuery(device.get());
    EXPECT_EQ(response.status, ttnn::graph::ExecutionStatus::Success)
        << "query failed: " << response.error_message.value_or("(no message)");
}

TEST(MetalEnvMockCCL, LegacyConfigureMockMode_QueryPasses) {
    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, /*num_chips=*/2);

    // Fabric must be configured before opening devices; set_fabric_config rejects
    // non-DISABLED changes while devices are still open.
    tt_fabric::SetFabricConfig(
        tt_fabric::FabricConfig::FABRIC_1D, tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    MetalContext::instance().initialize_fabric_config();

    ttnn::graph::ConstraintQueryResponse response;
    {
        auto device = distributed::MeshDevice::create(distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
        ASSERT_NE(device, nullptr);
        response = RunAllGatherConstraintQuery(device.get());
    }

    tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);
    experimental::disable_mock_mode();

    EXPECT_EQ(response.status, ttnn::graph::ExecutionStatus::Success)
        << "query failed: " << response.error_message.value_or("(no message)");
}

}  // namespace
}  // namespace tt::tt_metal
