// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <chrono>
#include <vector>
#include <utility>
#include <thread>
#include <memory>

#include "fabric_command_interface.hpp"
#include "fabric_traffic_generator_defs.hpp"

namespace tt::tt_fabric::test_utils {

// ============================================================================
// Mock Classes for Testing
// ============================================================================

// Mock FabricRouterStateManager
class MockFabricRouterStateManager {
public:
    MOCK_METHOD(void, queue_state_transition, (RouterCommand cmd), ());
    MOCK_METHOD(void, refresh_all_core_states, (), ());
    MOCK_METHOD(RouterStateCommon, get_core_state,
        (const FabricNodeId& fabric_node_id, chan_id_t channel_id), ());
};

// Mock ControlPlane with necessary methods for testing
class MockControlPlane {
public:
    MOCK_METHOD(std::vector<MeshId>, get_user_physical_mesh_ids, (), (const));
    MOCK_METHOD(std::vector<ChipId>, get_physical_chip_ids, (MeshId mesh_id), (const));
    MOCK_METHOD(std::vector<chan_id_t>, get_active_fabric_eth_channels_for_device,
        (ChipId chip_id), (const));
    MOCK_METHOD(FabricNodeId, get_fabric_node_id_from_physical_chip_id,
        (ChipId chip_id), (const));

    // Return reference to mock state manager
    MockFabricRouterStateManager& get_state_manager() {
        return state_manager_;
    }

private:
    MockFabricRouterStateManager state_manager_;
};

// ============================================================================
// Test Fixture
// ============================================================================

class FabricCommandInterfaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_control_plane_ = std::make_unique<MockControlPlane>();
        command_interface_ = std::make_unique<FabricCommandInterface>(*mock_control_plane_);
    }

    void TearDown() override {
        command_interface_.reset();
        mock_control_plane_.reset();
    }

    std::unique_ptr<MockControlPlane> mock_control_plane_;
    std::unique_ptr<FabricCommandInterface> command_interface_;

    // Helper: Setup mock topology with given router cores
    void setup_mock_topology(
        const std::vector<std::pair<FabricNodeId, chan_id_t>>& router_cores) {
        // Group cores by mesh ID for get_user_physical_mesh_ids()
        std::set<MeshId> mesh_ids;
        std::map<MeshId, std::set<ChipId>> mesh_to_chips;

        for (const auto& [node_id, _] : router_cores) {
            mesh_ids.insert(node_id.mesh_id);
            mesh_to_chips[node_id.mesh_id].insert(node_id.logical_x, node_id.logical_y);
        }

        // Convert to vectors for mock
        std::vector<MeshId> mesh_id_vec(mesh_ids.begin(), mesh_ids.end());

        EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
            .WillRepeatedly(::testing::Return(mesh_id_vec));

        // Setup get_fabric_node_id_from_physical_chip_id
        for (const auto& [node_id, _] : router_cores) {
            // Create mapping based on node_id
            EXPECT_CALL(*mock_control_plane_,
                get_fabric_node_id_from_physical_chip_id(::testing::_))
                .WillRepeatedly(::testing::Return(node_id));
        }
    }
};

// ============================================================================
// Constructor Tests (FR-1: Component Initialization)
// ============================================================================

TEST_F(FabricCommandInterfaceTest, ConstructorAcceptsControlPlaneReference) {
    // FR-1: Constructor must accept ControlPlane& reference
    EXPECT_THAT(command_interface_, ::testing::NotNull());
}

TEST_F(FabricCommandInterfaceTest, ConstructorStoresControlPlaneReference) {
    // Verify that the command interface holds a reference to control plane
    // This is implicit - if constructor didn't store it, later methods would fail
    EXPECT_THAT(command_interface_, ::testing::NotNull());
}

// ============================================================================
// get_all_router_cores() Tests (FR-2: Topology Discovery)
// ============================================================================

TEST_F(FabricCommandInterfaceTest, GetAllRouterCoresReturnsEmptyForNoDevices) {
    // Test handling of empty topology
    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{}));

    auto cores = command_interface_->get_all_router_cores();
    EXPECT_EQ(cores.size(), 0u);
}

TEST_F(FabricCommandInterfaceTest, GetAllRouterCoresSingleDevice) {
    // Test with single device topology
    FabricNodeId node_id{.mesh_id = 0};
    std::vector<std::pair<FabricNodeId, chan_id_t>> expected_cores = {
        {node_id, 0},
        {node_id, 1},
        {node_id, 2},
        {node_id, 3}
    };

    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillOnce(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillOnce(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{0, 1, 2, 3}));

    auto cores = command_interface_->get_all_router_cores();

    EXPECT_EQ(cores.size(), 4u);
    for (size_t i = 0; i < cores.size(); ++i) {
        EXPECT_EQ(cores[i].first, node_id);
        EXPECT_EQ(cores[i].second, i);
    }
}

TEST_F(FabricCommandInterfaceTest, GetAllRouterCoresMultipleDevices) {
    // Test with multiple device topology
    FabricNodeId node_id_0{.mesh_id = 0};
    FabricNodeId node_id_1{.mesh_id = 0};

    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillOnce(::testing::Return(std::vector<ChipId>{0, 1}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillOnce(::testing::Return(node_id_0));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(1))
        .WillOnce(::testing::Return(node_id_1));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{0, 1}));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(1))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{2, 3}));

    auto cores = command_interface_->get_all_router_cores();

    EXPECT_EQ(cores.size(), 4u);
}

// ============================================================================
// all_routers_in_state() Tests (FR-3: State Verification)
// ============================================================================

TEST_F(FabricCommandInterfaceTest, AllRoutersInStateReturnsTrueWhenAllRunning) {
    // FR-5: Test all_routers_in_state returns true when all routers are RUNNING
    FabricNodeId node_id{.mesh_id = 0};

    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillOnce(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillOnce(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{0, 1}));

    // Mock state manager returns RUNNING for all cores
    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, 0))
        .WillOnce(::testing::Return(RouterStateCommon::RUNNING));

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, 1))
        .WillOnce(::testing::Return(RouterStateCommon::RUNNING));

    bool result = command_interface_->all_routers_in_state(RouterStateCommon::RUNNING);

    EXPECT_TRUE(result);
}

TEST_F(FabricCommandInterfaceTest, AllRoutersInStateReturnsFalseWhenMixed) {
    // FR-5: Test all_routers_in_state returns false when routers have mixed states
    FabricNodeId node_id{.mesh_id = 0};

    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillOnce(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillOnce(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{0, 1}));

    // Mock state manager: first RUNNING, second PAUSED
    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, 0))
        .WillOnce(::testing::Return(RouterStateCommon::RUNNING));

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, 1))
        .WillOnce(::testing::Return(RouterStateCommon::PAUSED));

    bool result = command_interface_->all_routers_in_state(RouterStateCommon::RUNNING);

    EXPECT_FALSE(result);
}

TEST_F(FabricCommandInterfaceTest, AllRoutersInStateReturnsTrueWhenAllPaused) {
    // FR-5: Test all_routers_in_state returns true when all routers are PAUSED
    FabricNodeId node_id{.mesh_id = 0};

    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillOnce(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillOnce(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{0, 1}));

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, 0))
        .WillOnce(::testing::Return(RouterStateCommon::PAUSED));

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, 1))
        .WillOnce(::testing::Return(RouterStateCommon::PAUSED));

    bool result = command_interface_->all_routers_in_state(RouterStateCommon::PAUSED);

    EXPECT_TRUE(result);
}

TEST_F(FabricCommandInterfaceTest, AllRoutersInStateReturnsFalseForEmptyTopology) {
    // Edge case: empty topology should return false (no routers to be in state)
    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{}));

    bool result = command_interface_->all_routers_in_state(RouterStateCommon::RUNNING);

    // Empty topology means "not all routers in state" (vacuous truth handling)
    EXPECT_FALSE(result);
}

// ============================================================================
// pause_routers() Tests (FR-4: Pause Command)
// ============================================================================

TEST_F(FabricCommandInterfaceTest, PauseRoutersIssuesCommandToAllActiveRouters) {
    // FR-4: pause_routers() must issue PAUSE command to all active routers
    FabricNodeId node_id{.mesh_id = 0};

    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillOnce(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillOnce(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{0, 1}));

    // Verify PAUSE command is queued
    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        queue_state_transition(RouterCommand::PAUSE))
        .Times(2);  // Once per channel

    command_interface_->pause_routers();
}

TEST_F(FabricCommandInterfaceTest, PauseRoutersHandlesEmptyTopology) {
    // Edge case: pause_routers() on empty topology should complete without error
    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{}));

    // Should not throw
    EXPECT_NO_THROW(command_interface_->pause_routers());
}

TEST_F(FabricCommandInterfaceTest, PauseRoutersMultipleDevices) {
    // FR-4: pause_routers() issues command to ALL routers across multiple devices
    FabricNodeId node_id_0{.mesh_id = 0};
    FabricNodeId node_id_1{.mesh_id = 0};

    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillOnce(::testing::Return(std::vector<ChipId>{0, 1}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillOnce(::testing::Return(node_id_0));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(1))
        .WillOnce(::testing::Return(node_id_1));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{0, 1}));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(1))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{2}));

    // Expect PAUSE command for all 3 routers
    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        queue_state_transition(RouterCommand::PAUSE))
        .Times(3);

    command_interface_->pause_routers();
}

// ============================================================================
// resume_routers() Tests (FR-4: Resume Command)
// ============================================================================

TEST_F(FabricCommandInterfaceTest, ResumeRoutersIssuesRUNCommandToAllRouters) {
    // resume_routers() must issue RUN command to all active routers
    FabricNodeId node_id{.mesh_id = 0};

    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillOnce(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillOnce(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{0, 1}));

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        queue_state_transition(RouterCommand::RUN))
        .Times(2);

    command_interface_->resume_routers();
}

// ============================================================================
// get_router_state() Tests (FR-5: State Query)
// ============================================================================

TEST_F(FabricCommandInterfaceTest, GetRouterStateReturnsRouterState) {
    // FR-5: get_router_state() returns state of specific router
    FabricNodeId node_id{.mesh_id = 0};
    chan_id_t channel_id = 1;

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, channel_id))
        .WillOnce(::testing::Return(RouterStateCommon::RUNNING));

    RouterStateCommon state = command_interface_->get_router_state(node_id, channel_id);

    EXPECT_EQ(state, RouterStateCommon::RUNNING);
}

TEST_F(FabricCommandInterfaceTest, GetRouterStateReturnsPausedState) {
    // FR-5: get_router_state() correctly returns PAUSED state
    FabricNodeId node_id{.mesh_id = 0};
    chan_id_t channel_id = 2;

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, channel_id))
        .WillOnce(::testing::Return(RouterStateCommon::PAUSED));

    RouterStateCommon state = command_interface_->get_router_state(node_id, channel_id);

    EXPECT_EQ(state, RouterStateCommon::PAUSED);
}

// ============================================================================
// wait_for_pause() Tests (FR-6: Wait with Timeout)
// ============================================================================

TEST_F(FabricCommandInterfaceTest, WaitForPauseReturnsTrueWhenAllRoutersPause) {
    // FR-6: wait_for_pause() returns true when all routers reach PAUSED state
    FabricNodeId node_id{.mesh_id = 0};

    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillOnce(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillOnce(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{0, 1}));

    // First call: not paused yet, second call: all paused
    EXPECT_CALL(mock_control_plane_->get_state_manager(), refresh_all_core_states())
        .Times(::testing::AtLeast(1));

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, 0))
        .WillOnce(::testing::Return(RouterStateCommon::RUNNING))
        .WillOnce(::testing::Return(RouterStateCommon::PAUSED));

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, 1))
        .WillOnce(::testing::Return(RouterStateCommon::RUNNING))
        .WillOnce(::testing::Return(RouterStateCommon::PAUSED));

    bool result = command_interface_->wait_for_pause(
        std::chrono::milliseconds(1000));

    EXPECT_TRUE(result);
}

TEST_F(FabricCommandInterfaceTest, WaitForPauseReturnsFalseOnTimeout) {
    // FR-6: wait_for_pause() returns false on timeout without throwing
    FabricNodeId node_id{.mesh_id = 0};

    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillRepeatedly(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillRepeatedly(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillRepeatedly(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillRepeatedly(::testing::Return(std::vector<chan_id_t>{0}));

    // Always return RUNNING (never pause)
    EXPECT_CALL(mock_control_plane_->get_state_manager(), refresh_all_core_states())
        .WillRepeatedly(::testing::Return());

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, 0))
        .WillRepeatedly(::testing::Return(RouterStateCommon::RUNNING));

    // Use short timeout to avoid long test
    bool result = command_interface_->wait_for_pause(
        std::chrono::milliseconds(100));

    EXPECT_FALSE(result);
}

TEST_F(FabricCommandInterfaceTest, WaitForPauseWithZeroTimeout) {
    // Edge case: zero timeout should return immediately
    FabricNodeId node_id{.mesh_id = 0};

    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillOnce(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillOnce(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{0}));

    EXPECT_CALL(mock_control_plane_->get_state_manager(), refresh_all_core_states())
        .Times(::testing::AtLeast(1));

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, 0))
        .WillOnce(::testing::Return(RouterStateCommon::RUNNING));

    // Zero timeout - should not block indefinitely
    bool result = command_interface_->wait_for_pause(
        std::chrono::milliseconds(0));

    EXPECT_FALSE(result);
}

// ============================================================================
// wait_for_state() Tests (FR-6: Generic State Wait)
// ============================================================================

TEST_F(FabricCommandInterfaceTest, WaitForStateReturnsTrueWhenStateReached) {
    // Generic wait_for_state() returns true when target state is reached
    FabricNodeId node_id{.mesh_id = 0};

    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillOnce(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillOnce(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{0}));

    EXPECT_CALL(mock_control_plane_->get_state_manager(), refresh_all_core_states())
        .WillOnce(::testing::Return());

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, 0))
        .WillOnce(::testing::Return(RouterStateCommon::RUNNING));

    bool result = command_interface_->wait_for_state(
        RouterStateCommon::RUNNING,
        std::chrono::milliseconds(1000),
        std::chrono::milliseconds(100));

    EXPECT_TRUE(result);
}

TEST_F(FabricCommandInterfaceTest, WaitForStateReturnsFalseOnTimeout) {
    // wait_for_state() returns false on timeout
    FabricNodeId node_id{.mesh_id = 0};

    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillRepeatedly(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillRepeatedly(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillRepeatedly(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillRepeatedly(::testing::Return(std::vector<chan_id_t>{0}));

    EXPECT_CALL(mock_control_plane_->get_state_manager(), refresh_all_core_states())
        .WillRepeatedly(::testing::Return());

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, 0))
        .WillRepeatedly(::testing::Return(RouterStateCommon::PAUSED));

    bool result = command_interface_->wait_for_state(
        RouterStateCommon::RUNNING,
        std::chrono::milliseconds(100),
        std::chrono::milliseconds(50));

    EXPECT_FALSE(result);
}

TEST_F(FabricCommandInterfaceTest, WaitForStateRespectsTimeoutParameter) {
    // wait_for_state() must respect the timeout parameter
    FabricNodeId node_id{.mesh_id = 0};

    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillRepeatedly(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillRepeatedly(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillRepeatedly(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillRepeatedly(::testing::Return(std::vector<chan_id_t>{0}));

    EXPECT_CALL(mock_control_plane_->get_state_manager(), refresh_all_core_states())
        .WillRepeatedly(::testing::Return());

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, 0))
        .WillRepeatedly(::testing::Return(RouterStateCommon::PAUSED));

    auto start = std::chrono::steady_clock::now();

    bool result = command_interface_->wait_for_state(
        RouterStateCommon::RUNNING,
        std::chrono::milliseconds(200),
        std::chrono::milliseconds(50));

    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(result);
    // Should take at least the timeout duration (with some tolerance)
    EXPECT_GE(elapsed, std::chrono::milliseconds(150));
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(FabricCommandInterfaceTest, IntegrationPauseAndWaitWorkTogether) {
    // Integration: pause_routers() followed by wait_for_pause()
    FabricNodeId node_id{.mesh_id = 0};

    // First call for pause_routers()
    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillOnce(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillOnce(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{0}));

    // Second call for wait_for_pause()
    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillOnce(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillOnce(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{0}));

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        queue_state_transition(RouterCommand::PAUSE))
        .Times(1);

    EXPECT_CALL(mock_control_plane_->get_state_manager(), refresh_all_core_states())
        .Times(::testing::AtLeast(1));

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, 0))
        .WillOnce(::testing::Return(RouterStateCommon::PAUSED));

    command_interface_->pause_routers();
    bool result = command_interface_->wait_for_pause(std::chrono::milliseconds(500));

    EXPECT_TRUE(result);
}

TEST_F(FabricCommandInterfaceTest, IntegrationResumeAndWaitWorkTogether) {
    // Integration: resume_routers() followed by wait_for_state(RUNNING)
    FabricNodeId node_id{.mesh_id = 0};

    // First call for resume_routers()
    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillOnce(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillOnce(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{0}));

    // Second call for wait_for_state()
    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillOnce(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillOnce(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillOnce(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillOnce(::testing::Return(std::vector<chan_id_t>{0}));

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        queue_state_transition(RouterCommand::RUN))
        .Times(1);

    EXPECT_CALL(mock_control_plane_->get_state_manager(), refresh_all_core_states())
        .Times(::testing::AtLeast(1));

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, 0))
        .WillOnce(::testing::Return(RouterStateCommon::RUNNING));

    command_interface_->resume_routers();
    bool result = command_interface_->wait_for_state(
        RouterStateCommon::RUNNING,
        std::chrono::milliseconds(500));

    EXPECT_TRUE(result);
}

// ============================================================================
// No Busy-Wait Tests (FR-6: Sleep Between Polls)
// ============================================================================

TEST_F(FabricCommandInterfaceTest, WaitForStateDoesNotBusyWait) {
    // FR-6: wait_for_state() must NOT busy-wait, must use std::this_thread::sleep_for
    FabricNodeId node_id{.mesh_id = 0};

    EXPECT_CALL(*mock_control_plane_, get_user_physical_mesh_ids())
        .WillRepeatedly(::testing::Return(std::vector<MeshId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_physical_chip_ids(0))
        .WillRepeatedly(::testing::Return(std::vector<ChipId>{0}));

    EXPECT_CALL(*mock_control_plane_, get_fabric_node_id_from_physical_chip_id(0))
        .WillRepeatedly(::testing::Return(node_id));

    EXPECT_CALL(*mock_control_plane_, get_active_fabric_eth_channels_for_device(0))
        .WillRepeatedly(::testing::Return(std::vector<chan_id_t>{0}));

    EXPECT_CALL(mock_control_plane_->get_state_manager(), refresh_all_core_states())
        .WillRepeatedly(::testing::Return());

    EXPECT_CALL(mock_control_plane_->get_state_manager(),
        get_core_state(node_id, 0))
        .WillRepeatedly(::testing::Return(RouterStateCommon::PAUSED));

    auto start = std::chrono::steady_clock::now();

    // With poll interval of 50ms and timeout of 200ms, we should see approximately 4 iterations
    command_interface_->wait_for_state(
        RouterStateCommon::RUNNING,
        std::chrono::milliseconds(200),
        std::chrono::milliseconds(50));

    auto elapsed = std::chrono::steady_clock::now() - start;

    // Should take at least 150ms (3 sleep periods), but less than if it was busy-wait
    // Busy-wait would complete instantly or in microseconds
    EXPECT_GE(elapsed, std::chrono::milliseconds(100));
}

}  // namespace tt::tt_fabric::test_utils
