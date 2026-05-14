// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// Repro: CCL graph-trace constraint query on a multi-chip mock MeshDevice
//        throws std::out_of_range ("map::at") inside metal when devices are
//        created via MetalEnv (instead of the legacy configure_mock_mode path).
// =============================================================================

#include <gtest/gtest.h>

#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/mock_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/shape.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"

#include <ttnn/graph/graph_query_op_constraints.hpp>
#include <ttnn/operations/ccl/all_gather/all_gather.hpp>
#include <ttnn/tensor/layout/page_config.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>
#include <ttnn/tensor/tensor_ops.hpp>  // create_device_tensor used by query_op_constraints
#include <ttnn/tensor/tensor_spec.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/types.hpp>

#include <limits>
#include <optional>

namespace tt::tt_metal {
namespace {

// Same shape/layout the existing graph-query CCL test uses
// (g_interleave_4_2_160_244_tiled in test_graph_query_op_constraints.cpp).
ttnn::TensorSpec MakeInputSpec() {
    return ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 7 * 32}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
}

// Runs the same all_gather query the existing
// DistributedTensorOpIfTest.AllGatherWithShardedTopology test runs.
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

// -----------------------------------------------------------------------------
// TEST 1: MetalEnv with FABRIC_1D in the construction-time descriptor.
//         Crashes during `env.create_mesh_device` in fabric_builder_context.
// -----------------------------------------------------------------------------
TEST(MetalEnvMockCCL, FabricInDescriptor_CrashesDuringDeviceCreation) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2);
    ASSERT_TRUE(mock_path.has_value());

    FabricConfigDescriptor fabric_desc{};
    fabric_desc.fabric_config = tt_fabric::FabricConfig::FABRIC_1D;
    fabric_desc.reliability_mode = tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE;
    fabric_desc.num_routing_planes = std::numeric_limits<uint8_t>::max();
    MetalEnv env{MetalEnvDescriptor{*mock_path, fabric_desc}};

    // Expected: this throws inside MeshDevice creation, at
    //   tt_cluster.cpp:1187    Expected non-zero num_routing_planes ...
    // or
    //   fabric_builder_context.cpp:173   querying num initialized routers
    //   for an unknown device 0
    // (depending on which assertion fires first).
    auto device = env.create_mesh_device(distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
    auto response = RunAllGatherConstraintQuery(device.get());
    EXPECT_EQ(response.status, ttnn::graph::ExecutionStatus::Success)
        << "query failed: " << response.error_message.value_or("(no message)");
}

// -----------------------------------------------------------------------------
// TEST 2: MetalEnv with fabric DISABLED at construction, enabled AFTER device
//         creation via MetalEnvAccessor.  Gets past topology mapping but
//         `query_op_constraints` then throws std::out_of_range ("map::at").
// -----------------------------------------------------------------------------
TEST(MetalEnvMockCCL, FabricAfterDeviceCreation_QueryThrowsMapAt) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2);
    ASSERT_TRUE(mock_path.has_value());

    // Step 1: env with fabric DISABLED.
    MetalEnv env{MetalEnvDescriptor{*mock_path}};

    // Step 2: create the MeshDevice.
    auto device = env.create_mesh_device(distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
    ASSERT_NE(device, nullptr);

    // Step 3: enable fabric AFTER device exists, mirroring the legacy lifecycle.
    //         nullopt num_routing_planes resolves to max() inside
    //         MetalEnvImpl::set_fabric_config via value_or — matches what
    //         `tt::tt_fabric::SetFabricConfig(...)` does today.
    MetalEnvAccessor{env}.impl().set_fabric_config(
        tt_fabric::FabricConfig::FABRIC_1D, tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);

    // Step 4: graph-trace constraint query.  This is where it currently
    // throws std::out_of_range ("map::at") inside the fabric / control-plane
    // router-table lookup.
    auto response = RunAllGatherConstraintQuery(device.get());
    EXPECT_EQ(response.status, ttnn::graph::ExecutionStatus::Success)
        << "query failed: " << response.error_message.value_or("(no message)");
}

// -----------------------------------------------------------------------------
// TEST 3: Baseline using the legacy configure_mock_mode +
//         MetalContext::instance().initialize_fabric_config() path.
//         This PASSES today and is the path tt-mlir's OpModel currently uses.
// -----------------------------------------------------------------------------
TEST(MetalEnvMockCCL, LegacyConfigureMockMode_QueryPasses) {
    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, /*num_chips=*/2);

    ttnn::graph::ConstraintQueryResponse response;
    {
        auto device = distributed::MeshDevice::create(distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
        ASSERT_NE(device, nullptr);

        tt_fabric::SetFabricConfig(
            tt_fabric::FabricConfig::FABRIC_1D, tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
        MetalContext::instance().initialize_fabric_config();

        response = RunAllGatherConstraintQuery(device.get());

        // Tear fabric down while the device + mock cluster are still alive.
        tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);
    }  // ~MeshDevice runs here, before disable_mock_mode() clears mock cluster state.

    experimental::disable_mock_mode();

    EXPECT_EQ(response.status, ttnn::graph::ExecutionStatus::Success)
        << "query failed: " << response.error_message.value_or("(no message)");
}

}  // namespace
}  // namespace tt::tt_metal
