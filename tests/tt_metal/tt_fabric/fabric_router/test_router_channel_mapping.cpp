// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/fabric/fabric_router_channel_mapping.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <hostdevcommon/fabric_common.h>

namespace tt::tt_fabric {

/**
 * Test fixture for FabricRouterChannelMapping
 * 
 * These tests validate Phase 1 functionality:
 * - RouterVariant enum support (MESH vs Z_ROUTER)
 * - VC1 channel mapping for Z routers
 * - VC1 channel mapping for standard mesh routers
 * - Correct virtual channel counts
 * - Correct sender channel counts per VC
 */
class RouterChannelMappingTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============ Basic Mesh Router Tests (Regression) ============

TEST_F(RouterChannelMappingTest, MeshRouter_1D_Linear_VC0Only) {
    FabricRouterChannelMapping mapping(
        Topology::Linear,
        eth_chan_directions::EAST,
        false,  // no tensix
        RouterVariant::MESH);

    // 1D routers only have VC0
    EXPECT_EQ(mapping.get_num_virtual_channels(), 1);
    
    // VC0 has 2 sender channels in 1D
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(0), 2);
    
    // VC1 should have 0 channels in 1D
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(1), 0);
    
    // Verify VC0 mappings
    auto vc0_ch0 = mapping.get_sender_mapping(0, 0);
    EXPECT_EQ(vc0_ch0.builder_type, BuilderType::ERISC);
    EXPECT_EQ(vc0_ch0.internal_sender_channel_id, 0);
    
    auto vc0_ch1 = mapping.get_sender_mapping(0, 1);
    EXPECT_EQ(vc0_ch1.builder_type, BuilderType::ERISC);
    EXPECT_EQ(vc0_ch1.internal_sender_channel_id, 1);
}

TEST_F(RouterChannelMappingTest, MeshRouter_2D_Mesh_VC0Only) {
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        eth_chan_directions::NORTH,
        false,  // no tensix
        RouterVariant::MESH);

    // 2D mesh routers currently only expose VC0 (VC1 not fully enabled yet)
    EXPECT_EQ(mapping.get_num_virtual_channels(), 1);
    
    // VC0 has 4 sender channels in 2D
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(0), 4);
    
    // Verify all 4 VC0 channels map correctly
    for (uint32_t i = 0; i < 4; ++i) {
        auto mapping_result = mapping.get_sender_mapping(0, i);
        EXPECT_EQ(mapping_result.builder_type, BuilderType::ERISC);
        EXPECT_EQ(mapping_result.internal_sender_channel_id, i);
    }
}

TEST_F(RouterChannelMappingTest, MeshRouter_2D_Torus_VC0Only) {
    FabricRouterChannelMapping mapping(
        Topology::Torus,
        eth_chan_directions::SOUTH,
        false,  // no tensix
        RouterVariant::MESH);

    EXPECT_EQ(mapping.get_num_virtual_channels(), 1);
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(0), 4);
}

// ============ Z Router Tests (New Functionality) ============

TEST_F(RouterChannelMappingTest, ZRouter_Has2VirtualChannels) {
    FabricRouterChannelMapping mapping(
        Topology::Mesh,  // Z routers use 2D topology
        eth_chan_directions::EAST,
        false,  // no tensix
        RouterVariant::Z_ROUTER);

    // Z routers have both VC0 and VC1
    EXPECT_EQ(mapping.get_num_virtual_channels(), 2);
}

TEST_F(RouterChannelMappingTest, ZRouter_VC0_Has4SenderChannels) {
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        eth_chan_directions::EAST,
        false,
        RouterVariant::Z_ROUTER);

    // VC0 should have 4 sender channels (same as 2D mesh)
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(0), 4);
    
    // Verify VC0 channels map to erisc 0-3
    for (uint32_t i = 0; i < 4; ++i) {
        auto mapping_result = mapping.get_sender_mapping(0, i);
        EXPECT_EQ(mapping_result.builder_type, BuilderType::ERISC);
        EXPECT_EQ(mapping_result.internal_sender_channel_id, i);
    }
}

TEST_F(RouterChannelMappingTest, ZRouter_VC1_Has4SenderChannels) {
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        eth_chan_directions::EAST,
        false,
        RouterVariant::Z_ROUTER);

    // VC1 should have 4 sender channels (Z→mesh, one per direction)
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(1), 4);
}

TEST_F(RouterChannelMappingTest, ZRouter_VC1_SenderChannels_MapToErisc4Through7) {
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        eth_chan_directions::EAST,
        false,
        RouterVariant::Z_ROUTER);

    // VC1 sender channels 0-3 should map to erisc internal channels 4-7
    for (uint32_t i = 0; i < 4; ++i) {
        auto mapping_result = mapping.get_sender_mapping(1, i);
        EXPECT_EQ(mapping_result.builder_type, BuilderType::ERISC);
        EXPECT_EQ(mapping_result.internal_sender_channel_id, 4 + i);
    }
}

TEST_F(RouterChannelMappingTest, ZRouter_VC1_ReceiverChannel_MapsToErisc1) {
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        eth_chan_directions::EAST,
        false,
        RouterVariant::Z_ROUTER);

    // VC1 receiver channel 0 should map to erisc internal receiver channel 1
    auto mapping_result = mapping.get_receiver_mapping(1, 0);
    EXPECT_EQ(mapping_result.builder_type, BuilderType::ERISC);
    EXPECT_EQ(mapping_result.internal_receiver_channel_id, 1);
}

TEST_F(RouterChannelMappingTest, ZRouter_VC0_ReceiverChannel_MapsToErisc0) {
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        eth_chan_directions::EAST,
        false,
        RouterVariant::Z_ROUTER);

    // VC0 receiver channel should still map to erisc receiver 0
    auto mapping_result = mapping.get_receiver_mapping(0, 0);
    EXPECT_EQ(mapping_result.builder_type, BuilderType::ERISC);
    EXPECT_EQ(mapping_result.internal_receiver_channel_id, 0);
}

// ============ Z Router with Different Directions ============

TEST_F(RouterChannelMappingTest, ZRouter_AllDirections_HaveSameChannelLayout) {
    std::vector<eth_chan_directions> directions = {
        eth_chan_directions::NORTH,
        eth_chan_directions::EAST,
        eth_chan_directions::SOUTH,
        eth_chan_directions::WEST
    };

    for (auto dir : directions) {
        FabricRouterChannelMapping mapping(
            Topology::Mesh,
            dir,
            false,
            RouterVariant::Z_ROUTER);

        // All Z routers have same channel layout regardless of direction
        EXPECT_EQ(mapping.get_num_virtual_channels(), 2);
        EXPECT_EQ(mapping.get_num_sender_channels_for_vc(0), 4);
        EXPECT_EQ(mapping.get_num_sender_channels_for_vc(1), 4);
    }
}

// ============ Mesh Router with Tensix Extension ============

TEST_F(RouterChannelMappingTest, MeshRouter_WithTensix_VC0_MapsToTensix) {
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        eth_chan_directions::EAST,
        true,  // has tensix
        RouterVariant::MESH);

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

// ============ Standard Mesh Router VC1 (Future Support) ============

TEST_F(RouterChannelMappingTest, MeshRouter_2D_VC1_NotYetEnabled) {
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        eth_chan_directions::EAST,
        false,
        RouterVariant::MESH);

    // Standard mesh routers don't expose VC1 yet
    EXPECT_EQ(mapping.get_num_virtual_channels(), 1);
    
    // VC1 should report 0 channels (not enabled)
    // Note: When VC1 is enabled for mesh routers, this test will need updating
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(1), 3);  // Would be 3 when enabled
}

// ============ Error Cases ============

TEST_F(RouterChannelMappingTest, InvalidVC_ThrowsError) {
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        eth_chan_directions::EAST,
        false,
        RouterVariant::MESH);

    // Accessing invalid VC should throw
    EXPECT_THROW(mapping.get_sender_mapping(5, 0), std::exception);
}

TEST_F(RouterChannelMappingTest, InvalidSenderChannel_ThrowsError) {
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        eth_chan_directions::EAST,
        false,
        RouterVariant::MESH);

    // Accessing out-of-range sender channel should throw
    EXPECT_THROW(mapping.get_sender_mapping(0, 10), std::exception);
}

TEST_F(RouterChannelMappingTest, InvalidReceiverChannel_ThrowsError) {
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        eth_chan_directions::EAST,
        false,
        RouterVariant::Z_ROUTER);

    // Accessing invalid receiver channel should throw
    EXPECT_THROW(mapping.get_receiver_mapping(1, 5), std::exception);
}

// ============ Comprehensive Z Router Scenario ============

TEST_F(RouterChannelMappingTest, ZRouter_CompleteChannelLayout) {
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        eth_chan_directions::EAST,
        false,
        RouterVariant::Z_ROUTER);

    // Verify complete channel layout for Z router
    
    // VC0: 4 sender channels → erisc 0-3
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(0), 4);
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
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(1), 4);
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
    EXPECT_EQ(mapping.get_num_virtual_channels(), 2);
}

// ============ get_all_sender_mappings Tests ============

TEST_F(RouterChannelMappingTest, GetAllSenderMappings_MeshRouter) {
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        eth_chan_directions::EAST,
        false,
        RouterVariant::MESH);

    auto all_mappings = mapping.get_all_sender_mappings();
    
    // Mesh router with VC0 only: 4 channels
    EXPECT_EQ(all_mappings.size(), 4);
    
    // Verify they're in order
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(all_mappings[i].builder_type, BuilderType::ERISC);
        EXPECT_EQ(all_mappings[i].internal_sender_channel_id, i);
    }
}

TEST_F(RouterChannelMappingTest, GetAllSenderMappings_ZRouter) {
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        eth_chan_directions::EAST,
        false,
        RouterVariant::Z_ROUTER);

    auto all_mappings = mapping.get_all_sender_mappings();
    
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

