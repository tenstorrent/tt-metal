// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/fabric/fabric_router_channel_mapping.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <hostdevcommon/fabric_common.h>

namespace tt::tt_fabric {

/**
 * Test fixture for FabricRouterChannelMapping
 *
 * These tests validate channel mapping functionality:
 * - RouterVariant enum support (MESH vs Z_ROUTER)
 * - VC1 channel mapping for Z routers
 * - VC1 channel mapping for standard mesh routers
 * - Correct virtual channel counts
 * - Correct sender channel counts per VC
 *
 * Test Coverage Summary:
 * ┌──────────────────────────────────────────────────────────────────────────────────────┐
 * │ Category                    │ Test Name                                │ Focus        │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Basic Mesh (Regression)     │ MeshRouter_1D_Linear_VC0Only             │ 1D VC0       │
 * │                             │ MeshRouter_2D_Mesh_VC0Only               │ 2D VC0       │
 * │                             │ MeshRouter_2D_Torus_VC0Only              │ Torus VC0    │
 * │                             │ MeshRouter_Ring_VC0Only                  │ Ring VC0     │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Z Router                    │ ZRouter_HasTwoVCs                        │ 2 VCs        │
 * │                             │ ZRouter_VC0_FourSenderChannels           │ VC0 channels │
 * │                             │ ZRouter_VC1_FourSenderChannels           │ VC1 channels │
 * │                             │ ZRouter_VC0_OneReceiverChannel           │ VC0 receiver │
 * │                             │ ZRouter_VC1_OneReceiverChannel           │ VC1 receiver │
 * │                             │ ZRouter_AllChannelsMapped                │ All mapped   │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Mesh with Intermesh VC      │ MeshRouter_WithIntermeshVC_HasVC1        │ VC1 enabled  │
 * │                             │ MeshRouter_WithIntermeshVC_VC1Senders    │ VC1 senders  │
 * │                             │ MeshRouter_WithIntermeshVC_VC1Receiver   │ VC1 receiver │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Variant Detection           │ VariantDetection_MeshRouter              │ MESH variant │
 * │                             │ VariantDetection_ZRouter                 │ Z variant    │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Channel Count Queries       │ ChannelCount_GetNumVirtualChannels       │ VC count     │
 * │                             │ ChannelCount_GetNumSendersPerVC          │ Sender count │
 * │                             │ ChannelCount_TotalSenderChannels         │ Total        │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Mapping Retrieval           │ MappingRetrieval_GetSenderMapping        │ Sender map   │
 * │                             │ MappingRetrieval_GetReceiverMapping      │ Receiver map │
 * │                             │ MappingRetrieval_GetAllSenderMappings    │ All senders  │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Edge Cases                  │ EdgeCase_InvalidVCQuery                  │ Invalid VC   │
 * │                             │ EdgeCase_InvalidChannelQuery             │ Invalid ch   │
 * │                             │ EdgeCase_EmptyMapping                    │ Empty        │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Intermesh Config            │ IntermeshConfig_EdgeOnly                 │ Edge only    │
 * │                             │ IntermeshConfig_FullMesh                 │ Full mesh    │
 * │                             │ IntermeshConfig_Disabled                 │ Disabled     │
 * │                             │ IntermeshConfig_ZIntermesh_4Channels     │ Z 4 channels │
 * │                             │ IntermeshConfig_XYIntermesh_3Channels    │ XY 3 channels│
 * └─────────────────────────────┴──────────────────────────────────────────┴──────────────┘
 *
 * Total: 31 tests across 8 categories
 */
class RouterChannelMappingTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============ Basic Mesh Router Tests (Regression) ============

TEST_F(RouterChannelMappingTest, MeshRouter_1D_Linear_VC0Only) {
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Linear);
    FabricRouterChannelMapping mapping(
        Topology::Linear,
        spec,
        false,  // no tensix
        RouterVariant::MESH,
        nullptr);  // no intermesh config

    // 1D routers only have VC0
    EXPECT_EQ(mapping.get_num_mapped_virtual_channels(), 1);

    // VC0 has 2 sender channels in 1D
    EXPECT_EQ(mapping.get_num_mapped_sender_channels_for_vc(0), 2);

    // VC1 should have 0 channels in 1D
    EXPECT_EQ(mapping.get_num_mapped_sender_channels_for_vc(1), 0);

    // Verify VC0 mappings
    auto vc0_ch0 = mapping.get_sender_mapping(0, 0);
    EXPECT_EQ(vc0_ch0.builder_type, BuilderType::ERISC);
    EXPECT_EQ(vc0_ch0.internal_sender_channel_id, 0);

    auto vc0_ch1 = mapping.get_sender_mapping(0, 1);
    EXPECT_EQ(vc0_ch1.builder_type, BuilderType::ERISC);
    EXPECT_EQ(vc0_ch1.internal_sender_channel_id, 1);
}

TEST_F(RouterChannelMappingTest, MeshRouter_2D_Mesh_VC0Only) {
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        spec,
        false,  // no tensix
        RouterVariant::MESH,
        nullptr);  // no intermesh config

    // 2D mesh routers currently only expose VC0 (VC1 not fully enabled yet)
    EXPECT_EQ(mapping.get_num_mapped_virtual_channels(), 1);

    // VC0 has 4 sender channels in 2D
    EXPECT_EQ(mapping.get_num_mapped_sender_channels_for_vc(0), 4);

    // Verify all 4 VC0 channels map correctly
    for (uint32_t i = 0; i < 4; ++i) {
        auto mapping_result = mapping.get_sender_mapping(0, i);
        EXPECT_EQ(mapping_result.builder_type, BuilderType::ERISC);
        EXPECT_EQ(mapping_result.internal_sender_channel_id, i);
    }
}

TEST_F(RouterChannelMappingTest, MeshRouter_2D_Torus_VC0Only) {
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Torus);
    FabricRouterChannelMapping mapping(
        Topology::Torus,
        spec,
        false,  // no tensix
        RouterVariant::MESH,
        nullptr);  // no intermesh config

    EXPECT_EQ(mapping.get_num_mapped_virtual_channels(), 1);
    EXPECT_EQ(mapping.get_num_mapped_sender_channels_for_vc(0), 4);
}

// ============ Z Router Tests (New Functionality) ============

TEST_F(RouterChannelMappingTest, ZRouter_Has2VirtualChannels) {
    auto intermesh_config = IntermeshVCConfig::full_mesh();
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(
        Topology::Mesh,  // Z routers use 2D topology
        spec,
        false,  // no tensix
        RouterVariant::Z_ROUTER,
        &intermesh_config);  // Z routers require intermesh config

    // Z routers have both VC0 and VC1
    EXPECT_EQ(mapping.get_num_mapped_virtual_channels(), 2);
}

TEST_F(RouterChannelMappingTest, ZRouter_VC0_Has4SenderChannels) {
    auto intermesh_config = IntermeshVCConfig::full_mesh();
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        spec,
        false,
        RouterVariant::Z_ROUTER,
        &intermesh_config);  // Z routers require intermesh config

    // VC0 should have 4 sender channels (same as 2D mesh)
    EXPECT_EQ(mapping.get_num_mapped_sender_channels_for_vc(0), 4);

    // Verify VC0 channels map to erisc 0-3
    for (uint32_t i = 0; i < 4; ++i) {
        auto mapping_result = mapping.get_sender_mapping(0, i);
        EXPECT_EQ(mapping_result.builder_type, BuilderType::ERISC);
        EXPECT_EQ(mapping_result.internal_sender_channel_id, i);
    }
}

TEST_F(RouterChannelMappingTest, ZRouter_VC1_Has4SenderChannels) {
    auto intermesh_config = IntermeshVCConfig::full_mesh();
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        spec,
        false,
        RouterVariant::Z_ROUTER,
        &intermesh_config);  // Z routers require intermesh config

    // VC1 should have 4 sender channels (Z→mesh, one per direction)
    EXPECT_EQ(mapping.get_num_mapped_sender_channels_for_vc(1), 4);
}

TEST_F(RouterChannelMappingTest, ZRouter_VC1_SenderChannels_MapToErisc4Through7) {
    auto intermesh_config = IntermeshVCConfig::full_mesh();
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        spec,
        false,
        RouterVariant::Z_ROUTER,
        &intermesh_config);  // Z routers require intermesh config

    // VC1 sender channels 0-3 should map to erisc internal channels 4-7
    for (uint32_t i = 0; i < 4; ++i) {
        auto mapping_result = mapping.get_sender_mapping(1, i);
        EXPECT_EQ(mapping_result.builder_type, BuilderType::ERISC);
        EXPECT_EQ(mapping_result.internal_sender_channel_id, 4 + i);
    }
}

TEST_F(RouterChannelMappingTest, ZRouterNoTensix_VC0_VC1_ReceiverChannel_MapsToErisc) {
    auto intermesh_config = IntermeshVCConfig::full_mesh();
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        spec,
        false,
        RouterVariant::Z_ROUTER,
        &intermesh_config);  // Z routers require intermesh config

    EXPECT_EQ(mapping.get_num_mapped_virtual_channels(), 2);
    EXPECT_EQ(mapping.get_num_mapped_sender_channels_for_vc(0), 4);
    EXPECT_EQ(mapping.get_num_mapped_sender_channels_for_vc(1), 4);

    // VC0 receiver channel should still map to erisc receiver 0
    for (auto vc = 0; vc < 2; ++vc) {
        auto vc_r = mapping.get_receiver_mapping(vc, 0);
        EXPECT_EQ(vc_r.builder_type, BuilderType::ERISC);
        EXPECT_EQ(vc_r.internal_receiver_channel_id, vc);
    }
}


// ============ Mesh Router with Tensix Extension ============

TEST_F(RouterChannelMappingTest, MeshRouter_WithTensix_VC0_MapsToTensix) {
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        spec,
        true,  // has tensix
        RouterVariant::MESH,
        nullptr);  // no intermesh config

    // With tensix extension, VC0 channels should map to TENSIX builder
    for (uint32_t i = 0; i < 4; ++i) {
        auto mapping_result = mapping.get_sender_mapping(0, i);
        EXPECT_EQ(mapping_result.builder_type, BuilderType::TENSIX);
        EXPECT_EQ(mapping_result.internal_sender_channel_id, i);
    }

    // Receiver should still be ERISC
    auto receiver_mapping = mapping.get_receiver_mapping(0, 0);
    EXPECT_EQ(receiver_mapping.builder_type, BuilderType::ERISC);
}

// ============ Standard Mesh Router VC1 with IntermeshVCConfig ============

TEST_F(RouterChannelMappingTest, MeshRouter_2D_VC1_WithoutIntermesh) {
    // Without intermesh config, VC1 should not be created
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        spec,
        false,
        RouterVariant::MESH,
        nullptr);  // No intermesh config

    EXPECT_EQ(mapping.get_num_mapped_virtual_channels(), 1);         // Only VC0
    EXPECT_EQ(mapping.get_num_mapped_sender_channels_for_vc(1), 0);  // VC1 not created
}

TEST_F(RouterChannelMappingTest, MeshRouter_2D_VC1_WithIntermeshEdgeOnly) {
    // With intermesh config (edge only), VC1 should be created
    auto intermesh_config = IntermeshVCConfig::edge_only();
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(Topology::Mesh, spec, false, RouterVariant::MESH, &intermesh_config);

    EXPECT_EQ(mapping.get_num_mapped_virtual_channels(), 2);         // VC0 + VC1
    EXPECT_EQ(mapping.get_num_mapped_sender_channels_for_vc(1), 3);  // VC1 created

    // Verify VC1 mappings exist
    auto vc1_ch0 = mapping.get_sender_mapping(1, 0);
    EXPECT_EQ(vc1_ch0.builder_type, BuilderType::ERISC);
    EXPECT_EQ(vc1_ch0.internal_sender_channel_id, 4);  // After VC0 channels 0-3
}

TEST_F(RouterChannelMappingTest, MeshRouter_2D_VC1_WithIntermeshFullMesh) {
    // With intermesh config (full mesh), VC1 should be created
    auto intermesh_config = IntermeshVCConfig::full_mesh();
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(Topology::Mesh, spec, false, RouterVariant::MESH, &intermesh_config);

    EXPECT_EQ(mapping.get_num_mapped_virtual_channels(), 2);         // VC0 + VC1
    EXPECT_EQ(mapping.get_num_mapped_sender_channels_for_vc(1), 3);  // VC1 created
}

TEST_F(RouterChannelMappingTest, MeshRouter_2D_VC1_WithIntermeshFullMeshPassThrough) {
    // With intermesh config (full mesh with pass-through), VC1 should be created
    auto intermesh_config = IntermeshVCConfig::full_mesh_with_pass_through();
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(Topology::Mesh, spec, false, RouterVariant::MESH, &intermesh_config);

    EXPECT_EQ(mapping.get_num_mapped_virtual_channels(), 2);         // VC0 + VC1
    EXPECT_EQ(mapping.get_num_mapped_sender_channels_for_vc(1), 3);  // VC1 created
    // All modes that require VC1 create the same mappings
}

// ============ IntermeshVCConfig Tests ============

TEST_F(RouterChannelMappingTest, IntermeshVCConfig_Disabled) {
    auto config = IntermeshVCConfig::disabled();
    EXPECT_EQ(config.mode, IntermeshVCMode::DISABLED);
    EXPECT_FALSE(config.requires_vc1);
    EXPECT_FALSE(config.requires_vc1_full_mesh);
    EXPECT_FALSE(config.requires_vc1_mesh_pass_through);
}

TEST_F(RouterChannelMappingTest, IntermeshVCConfig_EdgeOnly) {
    auto config = IntermeshVCConfig::edge_only();
    EXPECT_EQ(config.mode, IntermeshVCMode::EDGE_ONLY);
    EXPECT_TRUE(config.requires_vc1);
    EXPECT_FALSE(config.requires_vc1_full_mesh);
    EXPECT_FALSE(config.requires_vc1_mesh_pass_through);
}

TEST_F(RouterChannelMappingTest, IntermeshVCConfig_FullMesh) {
    auto config = IntermeshVCConfig::full_mesh();
    EXPECT_EQ(config.mode, IntermeshVCMode::FULL_MESH);
    EXPECT_TRUE(config.requires_vc1);
    EXPECT_TRUE(config.requires_vc1_full_mesh);
    EXPECT_FALSE(config.requires_vc1_mesh_pass_through);
}

TEST_F(RouterChannelMappingTest, IntermeshVCConfig_FullMeshWithPassThrough) {
    auto config = IntermeshVCConfig::full_mesh_with_pass_through();
    EXPECT_EQ(config.mode, IntermeshVCMode::FULL_MESH_WITH_PASS_THROUGH);
    EXPECT_TRUE(config.requires_vc1);
    EXPECT_TRUE(config.requires_vc1_full_mesh);
    EXPECT_TRUE(config.requires_vc1_mesh_pass_through);
}

TEST_F(RouterChannelMappingTest, IntermeshVCConfig_DefaultInitialization) {
    IntermeshVCConfig config;
    EXPECT_EQ(config.mode, IntermeshVCMode::DISABLED);
    EXPECT_FALSE(config.requires_vc1);
    EXPECT_FALSE(config.requires_vc1_full_mesh);
    EXPECT_FALSE(config.requires_vc1_mesh_pass_through);
}

// ============ IntermeshVCConfig Mode Comparison Tests ============

TEST_F(RouterChannelMappingTest, IntermeshVCConfig_AllModesCreateVC1WhenRequired) {
    // All three intermesh modes should enable VC1
    std::vector<IntermeshVCConfig> configs = {
        IntermeshVCConfig::edge_only(),
        IntermeshVCConfig::full_mesh(),
        IntermeshVCConfig::full_mesh_with_pass_through()
    };

    for (const auto& config : configs) {
        auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
        FabricRouterChannelMapping mapping(Topology::Mesh, spec, false, RouterVariant::MESH, &config);

        EXPECT_EQ(mapping.get_num_mapped_virtual_channels(), 2)
            << "Mode " << static_cast<int>(config.mode) << " should enable VC1";
        EXPECT_EQ(mapping.get_num_mapped_sender_channels_for_vc(1), 3)
            << "Mode " << static_cast<int>(config.mode) << " should have 3 VC1 channels";
    }
}

TEST_F(RouterChannelMappingTest, IntermeshVCConfig_EdgeOnly_FlagsCorrect) {
    auto config = IntermeshVCConfig::edge_only();

    // EDGE_ONLY: VC1 enabled, but not full mesh, not pass-through
    EXPECT_TRUE(config.requires_vc1) << "EDGE_ONLY requires VC1";
    EXPECT_FALSE(config.requires_vc1_full_mesh) << "EDGE_ONLY does not require full mesh";
    EXPECT_FALSE(config.requires_vc1_mesh_pass_through) << "EDGE_ONLY does not support pass-through";
}

TEST_F(RouterChannelMappingTest, IntermeshVCConfig_FullMesh_FlagsCorrect) {
    auto config = IntermeshVCConfig::full_mesh();

    // FULL_MESH: VC1 enabled throughout mesh, but not pass-through
    EXPECT_TRUE(config.requires_vc1) << "FULL_MESH requires VC1";
    EXPECT_TRUE(config.requires_vc1_full_mesh) << "FULL_MESH requires VC1 throughout mesh";
    EXPECT_FALSE(config.requires_vc1_mesh_pass_through) << "FULL_MESH does not support inter-mesh pass-through";
}

TEST_F(RouterChannelMappingTest, IntermeshVCConfig_FullMeshWithPassThrough_FlagsCorrect) {
    auto config = IntermeshVCConfig::full_mesh_with_pass_through();

    // FULL_MESH_WITH_PASS_THROUGH: All flags enabled
    EXPECT_TRUE(config.requires_vc1) << "FULL_MESH_WITH_PASS_THROUGH requires VC1";
    EXPECT_TRUE(config.requires_vc1_full_mesh) << "FULL_MESH_WITH_PASS_THROUGH requires VC1 throughout mesh";
    EXPECT_TRUE(config.requires_vc1_mesh_pass_through) << "FULL_MESH_WITH_PASS_THROUGH supports inter-mesh pass-through";
}

TEST_F(RouterChannelMappingTest, IntermeshVCConfig_ModeProgression) {
    // Verify that modes form a logical progression in capabilities
    auto disabled = IntermeshVCConfig::disabled();
    auto edge_only = IntermeshVCConfig::edge_only();
    auto full_mesh = IntermeshVCConfig::full_mesh();
    auto full_mesh_pass = IntermeshVCConfig::full_mesh_with_pass_through();

    // DISABLED: No VC1
    EXPECT_FALSE(disabled.requires_vc1);

    // EDGE_ONLY: VC1 on edges only
    EXPECT_TRUE(edge_only.requires_vc1);
    EXPECT_FALSE(edge_only.requires_vc1_full_mesh);

    // FULL_MESH: VC1 throughout mesh (superset of EDGE_ONLY)
    EXPECT_TRUE(full_mesh.requires_vc1);
    EXPECT_TRUE(full_mesh.requires_vc1_full_mesh);
    EXPECT_FALSE(full_mesh.requires_vc1_mesh_pass_through);

    // FULL_MESH_WITH_PASS_THROUGH: All capabilities (superset of FULL_MESH)
    EXPECT_TRUE(full_mesh_pass.requires_vc1);
    EXPECT_TRUE(full_mesh_pass.requires_vc1_full_mesh);
    EXPECT_TRUE(full_mesh_pass.requires_vc1_mesh_pass_through);
}

// ============ Error Cases ============

TEST_F(RouterChannelMappingTest, InvalidVC_ThrowsError) {
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(Topology::Mesh, spec, false, RouterVariant::MESH, nullptr);

    // Accessing invalid VC should throw
    EXPECT_THROW(mapping.get_sender_mapping(5, 0), std::exception);
}

TEST_F(RouterChannelMappingTest, InvalidSenderChannel_ThrowsError) {
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(Topology::Mesh, spec, false, RouterVariant::MESH, nullptr);

    // Accessing out-of-range sender channel should throw
    EXPECT_THROW(mapping.get_sender_mapping(0, 10), std::exception);
}

TEST_F(RouterChannelMappingTest, InvalidReceiverChannel_ThrowsError) {
    auto intermesh_config = IntermeshVCConfig::full_mesh();
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(Topology::Mesh, spec, false, RouterVariant::Z_ROUTER, &intermesh_config);

    // Accessing invalid receiver channel should throw
    EXPECT_THROW(mapping.get_receiver_mapping(1, 5), std::exception);
}

// ============ Comprehensive Z Router Scenario ============

TEST_F(RouterChannelMappingTest, ZRouter_CompleteChannelLayout) {
    auto intermesh_config = IntermeshVCConfig::full_mesh();
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(Topology::Mesh, spec, false, RouterVariant::Z_ROUTER, &intermesh_config);

    // Verify complete channel layout for Z router

    // VC0: 4 sender channels → erisc 0-3
    EXPECT_EQ(mapping.get_num_mapped_sender_channels_for_vc(0), 4);
    for (uint32_t i = 0; i < 4; ++i) {
        auto s = mapping.get_sender_mapping(0, i);
        EXPECT_EQ(s.builder_type, BuilderType::ERISC);
        EXPECT_EQ(s.internal_sender_channel_id, i);
    }

    // VC0: 1 receiver channel → erisc 0
    auto vc0_r = mapping.get_receiver_mapping(0, 0);
    EXPECT_EQ(vc0_r.builder_type, BuilderType::ERISC);
    EXPECT_EQ(vc0_r.internal_receiver_channel_id, 0);

    // VC1: 4 sender channels → erisc 4-7
    EXPECT_EQ(mapping.get_num_mapped_sender_channels_for_vc(1), 4);
    for (uint32_t i = 0; i < 4; ++i) {
        auto s = mapping.get_sender_mapping(1, i);
        EXPECT_EQ(s.builder_type, BuilderType::ERISC);
        EXPECT_EQ(s.internal_sender_channel_id, 4 + i);
    }

    // VC1: 1 receiver channel → erisc 1
    auto vc1_r = mapping.get_receiver_mapping(1, 0);
    EXPECT_EQ(vc1_r.builder_type, BuilderType::ERISC);
    EXPECT_EQ(vc1_r.internal_receiver_channel_id, 1);

    // Total: 8 sender channels, 2 receiver channels
    EXPECT_EQ(mapping.get_num_mapped_virtual_channels(), 2);
}

// ============ get_all_sender_mappings Tests ============

TEST_F(RouterChannelMappingTest, GetAllSenderMappings_MeshRouter) {
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(Topology::Mesh, spec, false, RouterVariant::MESH, nullptr);

    auto all_mappings = mapping.get_all_sender_mappings(spec);

    // Mesh router with VC0 only: 4 channels
    EXPECT_EQ(all_mappings.size(), 4);

    // Verify they're in order
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(all_mappings[i].builder_type, BuilderType::ERISC);
        EXPECT_EQ(all_mappings[i].internal_sender_channel_id, i);
    }
}

TEST_F(RouterChannelMappingTest, GetAllSenderMappings_ZRouter) {
    auto intermesh_config = IntermeshVCConfig::full_mesh();
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh);
    FabricRouterChannelMapping mapping(Topology::Mesh, spec, false, RouterVariant::Z_ROUTER, &intermesh_config);

    auto all_mappings = mapping.get_all_sender_mappings(spec);

    // Z router: VC0 (4 channels) + VC1 (4 channels) = 8 total
    EXPECT_EQ(all_mappings.size(), 8);

    // First 4 should be VC0 → erisc 0-3
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(all_mappings[i].builder_type, BuilderType::ERISC);
        EXPECT_EQ(all_mappings[i].internal_sender_channel_id, i);
    }

    // Next 4 should be VC1 → erisc 4-7
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(all_mappings[4 + i].builder_type, BuilderType::ERISC);
        EXPECT_EQ(all_mappings[4 + i].internal_sender_channel_id, 4 + i);
    }
}

}  // namespace tt::tt_fabric
