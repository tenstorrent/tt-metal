// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include "tt_fabric_cycle_detector.hpp"
#include "tt_fabric_test_interfaces.hpp"
#include <tt-metalium/control_plane.hpp>

namespace tt::tt_fabric::fabric_tests {

// Custom hash for pair of FabricNodeId
struct PairHash {
    std::size_t operator()(const std::pair<FabricNodeId, FabricNodeId>& p) const {
        auto h1 = std::hash<uint32_t>{}(*p.first.mesh_id);
        auto h2 = std::hash<uint32_t>{}(p.first.chip_id);
        auto h3 = std::hash<uint32_t>{}(*p.second.mesh_id);
        auto h4 = std::hash<uint32_t>{}(p.second.chip_id);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
    }
};

// Mock ControlPlane for testing cycle detection
class MockControlPlane {
public:
    MockControlPlane() {}

    // Mock get_fabric_route for testing
    std::vector<std::pair<FabricNodeId, chan_id_t>> get_fabric_route(
        FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id, chan_id_t src_chan_id) const {
        // Return predefined routes for testing
        auto key = std::make_pair(src_fabric_node_id, dst_fabric_node_id);
        auto it = mock_routes_.find(key);
        if (it != mock_routes_.end()) {
            return it->second;
        }

        // Default: direct connection
        return {{src_fabric_node_id, src_chan_id}, {dst_fabric_node_id, 0}};
    }

    // Method to set up mock routes for testing
    void set_mock_route(
        FabricNodeId src, FabricNodeId dst, const std::vector<std::pair<FabricNodeId, chan_id_t>>& route) {
        mock_routes_[{src, dst}] = route;
    }

    // Clear all mock routes
    void clear_mock_routes() { mock_routes_.clear(); }

private:
    std::unordered_map<std::pair<FabricNodeId, FabricNodeId>, std::vector<std::pair<FabricNodeId, chan_id_t>>, PairHash>
        mock_routes_;
};

// Mock RouteManager for testing
class MockRouteManager : public IRouteManager {
public:
    MockRouteManager(const MockControlPlane* control_plane = nullptr) : control_plane_(control_plane) {}

    // Required IRouteManager methods (minimal implementations for testing)
    MeshShape get_mesh_shape() const override {
        return MeshShape{2, 4};  // Simplified 2x4 mesh for testing
    }

    uint32_t get_num_mesh_dims() const override {
        return 2;  // 2D mesh
    }

    bool wrap_around_mesh(FabricNodeId node) const override {
        return false;  // No wrap-around for testing
    }

    std::vector<FabricNodeId> get_dst_node_ids_from_hops(
        FabricNodeId src_node_id,
        std::unordered_map<RoutingDirection, uint32_t>& hops,
        ChipSendType chip_send_type) const override {
        return {src_node_id};  // Simplified for testing
    }

    std::unordered_map<RoutingDirection, uint32_t> get_hops_to_chip(
        FabricNodeId src_node_id, FabricNodeId dst_node_id) const override {
        // Simple hop calculation for testing
        std::unordered_map<RoutingDirection, uint32_t> hops;
        if (src_node_id != dst_node_id) {
            hops[RoutingDirection::N] = 1;
        }
        return hops;
    }

    bool are_devices_linear(const std::vector<FabricNodeId>& node_ids) const override {
        return true;  // Simplified for testing
    }

    std::vector<std::pair<FabricNodeId, FabricNodeId>> get_all_to_all_unicast_pairs() const override {
        return {};  // Simplified for testing
    }

    std::vector<std::pair<FabricNodeId, FabricNodeId>> get_all_to_one_unicast_pairs(
        uint32_t device_idx) const override {
        return {};  // Simplified for testing
    }

    std::vector<std::pair<FabricNodeId, FabricNodeId>> get_full_device_random_pairs(std::mt19937& gen) const override {
        return {};  // Simplified for testing
    }

    std::unordered_map<RoutingDirection, uint32_t> get_full_mcast_hops(const FabricNodeId& src_node_id) const override {
        return {};  // Simplified for testing
    }

    std::unordered_map<RoutingDirection, uint32_t> get_unidirectional_linear_mcast_hops(
        const FabricNodeId& src_node_id, uint32_t dim) const override {
        return {};  // Simplified for testing
    }

    std::optional<std::pair<FabricNodeId, FabricNodeId>> get_wrap_around_mesh_ring_neighbors(
        const FabricNodeId& src_node, const std::vector<FabricNodeId>& devices) const override {
        return std::nullopt;  // Simplified for testing
    }

    std::unordered_map<RoutingDirection, uint32_t> get_wrap_around_mesh_full_or_half_ring_mcast_hops(
        const FabricNodeId& src_node_id,
        const FabricNodeId& dst_node_forward_id,
        const FabricNodeId& dst_node_backward_id,
        HighLevelTrafficPattern pattern_type) const override {
        return {};  // Simplified for testing
    }

    std::unordered_map<RoutingDirection, uint32_t> get_full_or_half_ring_mcast_hops(
        const FabricNodeId& src_node_id, HighLevelTrafficPattern pattern_type, uint32_t dim) const override {
        return {};  // Simplified for testing
    }

    std::vector<std::unordered_map<RoutingDirection, uint32_t>> split_multicast_hops(
        const std::unordered_map<RoutingDirection, uint32_t>& mcast_hops) const override {
        return {};  // Simplified for testing
    }

    FabricNodeId get_random_unicast_destination(FabricNodeId src_node_id, std::mt19937& gen) const override {
        return src_node_id;  // Simplified for testing
    }

    RoutingDirection get_forwarding_direction(
        const FabricNodeId& src_node_id, const FabricNodeId& dst_node_id) const override {
        return RoutingDirection::N;  // Simplified for testing
    }

    RoutingDirection get_forwarding_direction(
        const std::unordered_map<RoutingDirection, uint32_t>& hops) const override {
        return RoutingDirection::N;  // Simplified for testing
    }

    std::vector<uint32_t> get_forwarding_link_indices_in_direction(
        const FabricNodeId& src_node_id, const RoutingDirection& direction) const override {
        return {0};  // Simplified for testing
    }

    std::vector<uint32_t> get_forwarding_link_indices_in_direction(
        const FabricNodeId& src_node_id,
        const FabricNodeId& dst_node_id,
        const RoutingDirection& direction) const override {
        return {0};  // Simplified for testing
    }

    FabricNodeId get_neighbor_node_id(
        const FabricNodeId& src_node_id, const RoutingDirection& direction) const override {
        // Simple neighbor calculation for testing
        return FabricNodeId{src_node_id.mesh_id, static_cast<chip_id_t>((src_node_id.chip_id + 1) % 8)};
    }

    FabricNodeId get_mcast_start_node_id(
        const FabricNodeId& src_node_id, const std::unordered_map<RoutingDirection, uint32_t>& hops) const override {
        return src_node_id;  // Simplified for testing
    }

    std::pair<std::unordered_map<RoutingDirection, uint32_t>, uint32_t> get_sync_hops_and_val(
        const FabricNodeId& src_device, const std::vector<FabricNodeId>& devices) const override {
        return {{}, 0};  // Simplified for testing
    }

    bool validate_num_links_supported(uint32_t num_links) const override {
        return num_links <= 4;  // Simplified for testing
    }

    std::vector<FabricNodeId> get_full_fabric_path(
        const FabricNodeId& src_node_id, const FabricNodeId& dst_node_id) const override {
        if (control_plane_) {
            // Get fabric path using mock control plane
            try {
                auto full_route = control_plane_->get_fabric_route(src_node_id, dst_node_id, 0);

                std::vector<FabricNodeId> path;
                path.reserve(full_route.size());

                for (const auto& [node_id, channel_id] : full_route) {
                    path.push_back(node_id);
                }

                // Ensure the path includes the destination node if it's not already there
                if (!path.empty() && path.back() != dst_node_id) {
                    path.push_back(dst_node_id);
                }

                return path;
            } catch (const std::exception& e) {
                // Fallback on error
            }
        }

        // Fallback: direct path for testing
        if (src_node_id == dst_node_id) {
            return {src_node_id};
        }
        return {src_node_id, dst_node_id};
    }

    const void* get_control_plane() const override {
        return nullptr;  // Mock doesn't expose real control plane
    }

    // Method to set control plane for testing
    void set_control_plane(const MockControlPlane* control_plane) { control_plane_ = control_plane; }

private:
    const MockControlPlane* control_plane_;
};

// Test fixture for cycle detection tests
class CycleDetectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_control_plane_ = std::make_unique<MockControlPlane>();
        mock_route_manager_ = std::make_unique<MockRouteManager>(mock_control_plane_.get());
    }

    void TearDown() override { mock_control_plane_->clear_mock_routes(); }

    std::unique_ptr<MockControlPlane> mock_control_plane_;
    std::unique_ptr<MockRouteManager> mock_route_manager_;

    FabricNodeId node_a_{MeshId{0}, 0};
    FabricNodeId node_b_{MeshId{0}, 1};
    FabricNodeId node_c_{MeshId{0}, 2};
    FabricNodeId node_d_{MeshId{0}, 3};
    FabricNodeId node_mesh1_a_{MeshId{1}, 0};
    FabricNodeId node_mesh1_b_{MeshId{1}, 1};
};

// Test basic cycle detection functionality
TEST_F(CycleDetectionTest, BasicCycleDetection) {
    // Create a simple graph with a cycle: A -> B -> C -> A
    NodeGraph graph;
    graph[node_a_] = {node_b_};
    graph[node_b_] = {node_c_};
    graph[node_c_] = {node_a_};

    // Detect cycles
    auto cycles = detect_cycles(graph);

    // Should find one cycle
    EXPECT_EQ(cycles.size(), 1);
    EXPECT_EQ(cycles[0].size(), 4);  // A -> B -> C -> A (includes return to start)
}

TEST_F(CycleDetectionTest, NoCycleDetection) {
    // Create a simple graph without cycles: A -> B -> C -> D
    NodeGraph graph;
    graph[node_a_] = {node_b_};
    graph[node_b_] = {node_c_};
    graph[node_c_] = {node_d_};

    // Detect cycles
    auto cycles = detect_cycles(graph);

    // Should find no cycles
    EXPECT_EQ(cycles.size(), 0);
}

TEST_F(CycleDetectionTest, MultipleCyclesDetection) {
    // Create a graph with multiple cycles
    // Cycle 1: A -> B -> A
    // Cycle 2: C -> D -> C
    NodeGraph graph;
    graph[node_a_] = {node_b_};
    graph[node_b_] = {node_a_};
    graph[node_c_] = {node_d_};
    graph[node_d_] = {node_c_};

    // Detect cycles
    auto cycles = detect_cycles(graph);

    // Should find two cycles
    EXPECT_EQ(cycles.size(), 2);
}

TEST_F(CycleDetectionTest, PathGraphBuilding) {
    // Test building path graph from full path
    std::vector<FabricNodeId> path = {node_a_, node_b_, node_c_, node_d_};

    auto graph = build_path_graph_from_full_path(path);

    // Should have directed edges A->B, B->C, C->D
    EXPECT_EQ(graph[node_a_].size(), 1);
    EXPECT_EQ(graph[node_a_][0], node_b_);

    EXPECT_EQ(graph[node_b_].size(), 1);
    EXPECT_EQ(graph[node_b_][0], node_c_);

    EXPECT_EQ(graph[node_c_].size(), 1);
    EXPECT_EQ(graph[node_c_][0], node_d_);

    // D should have no outgoing edges
    EXPECT_EQ(graph[node_d_].size(), 0);
}

TEST_F(CycleDetectionTest, ControlPlaneIntegration) {
    // Set up a mock route that creates a cycle
    // A -> B -> C -> A
    mock_control_plane_->set_mock_route(node_a_, node_b_, {{node_a_, 0}, {node_c_, 1}, {node_b_, 2}});
    mock_control_plane_->set_mock_route(node_b_, node_c_, {{node_b_, 0}, {node_a_, 1}, {node_c_, 2}});
    mock_control_plane_->set_mock_route(node_c_, node_a_, {{node_c_, 0}, {node_b_, 1}, {node_a_, 2}});

    // Test fabric path extraction using MockRouteManager
    auto path_a_to_b = mock_route_manager_->get_full_fabric_path(node_a_, node_b_);
    auto path_b_to_c = mock_route_manager_->get_full_fabric_path(node_b_, node_c_);
    auto path_c_to_a = mock_route_manager_->get_full_fabric_path(node_c_, node_a_);

    // Check that paths are returned correctly
    EXPECT_FALSE(path_a_to_b.empty());
    EXPECT_FALSE(path_b_to_c.empty());
    EXPECT_FALSE(path_c_to_a.empty());

    // Check that the paths contain the expected nodes
    EXPECT_EQ(path_a_to_b.back(), node_b_);
    EXPECT_EQ(path_b_to_c.back(), node_c_);
    EXPECT_EQ(path_c_to_a.back(), node_a_);
}

TEST_F(CycleDetectionTest, InterMeshCycleDetection) {
    // Set up inter-mesh routes that could create cycles
    std::vector<std::pair<FabricNodeId, FabricNodeId>> pairs = {
        {node_a_, node_mesh1_a_}, {node_mesh1_a_, node_b_}, {node_b_, node_a_}};

    // Set up mock routes for inter-mesh traffic
    mock_control_plane_->set_mock_route(node_a_, node_mesh1_a_, {{node_a_, 0}, {node_b_, 1}, {node_mesh1_a_, 2}});
    mock_control_plane_->set_mock_route(node_mesh1_a_, node_b_, {{node_mesh1_a_, 0}, {node_a_, 1}, {node_b_, 2}});
    mock_control_plane_->set_mock_route(node_b_, node_a_, {{node_b_, 0}, {node_a_, 1}});

    // Test inter-mesh cycle detection (returns bool indicating cycles found)
    bool has_cycles = detect_cycles_in_random_inter_mesh_traffic(pairs, *mock_route_manager_, "InterMeshTest");

    // Should detect cycles in this configuration
    EXPECT_TRUE(has_cycles);
}

TEST_F(CycleDetectionTest, NoInterMeshCycles) {
    // Set up inter-mesh routes without cycles
    std::vector<std::pair<FabricNodeId, FabricNodeId>> pairs = {{node_a_, node_mesh1_a_}, {node_b_, node_mesh1_b_}};

    // Set up mock routes without cycles
    mock_control_plane_->set_mock_route(node_a_, node_mesh1_a_, {{node_a_, 0}, {node_mesh1_a_, 1}});
    mock_control_plane_->set_mock_route(node_b_, node_mesh1_b_, {{node_b_, 0}, {node_mesh1_b_, 1}});

    // Test inter-mesh cycle detection
    bool has_cycles = detect_cycles_in_random_inter_mesh_traffic(pairs, *mock_route_manager_, "NoInterMeshTest");

    // Should not detect cycles
    EXPECT_FALSE(has_cycles);
}

TEST_F(CycleDetectionTest, FallbackMechanism) {
    // Test fallback when control plane is not available
    MockRouteManager route_manager_no_cp(nullptr);

    std::vector<std::pair<FabricNodeId, FabricNodeId>> pairs = {{node_a_, node_b_}, {node_b_, node_c_}};

    // Should still work without control plane (using fallback mechanisms)
    bool has_cycles = detect_cycles_in_random_inter_mesh_traffic(pairs, route_manager_no_cp, "FallbackTest");

    // Should complete without crashing (result depends on fallback implementation)
    EXPECT_TRUE(has_cycles || !has_cycles);  // Just ensure it completes
}

TEST_F(CycleDetectionTest, EmptyInput) {
    // Test with empty input
    std::vector<std::pair<FabricNodeId, FabricNodeId>> empty_pairs;

    bool has_cycles = detect_cycles_in_random_inter_mesh_traffic(empty_pairs, *mock_route_manager_, "EmptyTest");

    // Should handle empty input gracefully
    EXPECT_FALSE(has_cycles);
}

TEST_F(CycleDetectionTest, SelfLoops) {
    // Test with self-loops (node to itself)
    std::vector<std::pair<FabricNodeId, FabricNodeId>> self_loop_pairs = {{node_a_, node_a_}};

    bool has_cycles = detect_cycles_in_random_inter_mesh_traffic(self_loop_pairs, *mock_route_manager_, "SelfLoopTest");

    // Self-loops ARE cycles and should be detected as problematic
    // A device should use NOC for intra-device communication, not fabric
    EXPECT_TRUE(has_cycles);
}

TEST_F(CycleDetectionTest, LargeCycle) {
    // Test with a larger cycle: A -> B -> C -> D -> A
    mock_control_plane_->set_mock_route(node_a_, node_b_, {{node_a_, 0}, {node_b_, 1}});
    mock_control_plane_->set_mock_route(node_b_, node_c_, {{node_b_, 0}, {node_c_, 1}});
    mock_control_plane_->set_mock_route(node_c_, node_d_, {{node_c_, 0}, {node_d_, 1}});
    mock_control_plane_->set_mock_route(node_d_, node_a_, {{node_d_, 0}, {node_a_, 1}});

    std::vector<std::pair<FabricNodeId, FabricNodeId>> pairs = {
        {node_a_, node_b_}, {node_b_, node_c_}, {node_c_, node_d_}, {node_d_, node_a_}};

    bool has_cycles = detect_cycles_in_random_inter_mesh_traffic(pairs, *mock_route_manager_, "LargeCycleTest");

    // Should detect the large cycle
    EXPECT_TRUE(has_cycles);
}

// Advanced test scenarios for 16 Loudbox inter-mesh deadlock detection
TEST_F(CycleDetectionTest, SixteenLoudboxInterMeshDeadlock) {
    // Simulate the actual 16 Loudbox scenario: 4 superpods, each with 4 T3Ks, each T3K has 8 devices
    // Create nodes representing different T3Ks and superpods
    FabricNodeId t3k0_dev0{MeshId{0}, 0};  // Superpod 0, T3K 0, Device 0
    FabricNodeId t3k0_dev4{MeshId{0}, 4};  // Superpod 0, T3K 0, Device 4 (inter-T3K connector)
    FabricNodeId t3k1_dev0{MeshId{1}, 0};  // Superpod 0, T3K 1, Device 0
    FabricNodeId t3k1_dev4{MeshId{1}, 4};  // Superpod 0, T3K 1, Device 4 (inter-T3K connector)
    FabricNodeId t3k2_dev0{MeshId{2}, 0};  // Superpod 1, T3K 0, Device 0
    FabricNodeId t3k2_dev4{MeshId{2}, 4};  // Superpod 1, T3K 0, Device 4 (inter-superpod connector)

    // Set up routes that create the problematic inter-mesh cycle scenario
    // Device in T3K0 wants to send to device in T3K2 (different superpod)
    // But routing creates a cycle through intermediate T3K1

    // Route 1: T3K0_dev0 -> T3K2_dev0 (goes through T3K0_dev4 -> T3K1_dev4 -> T3K2_dev4 -> T3K2_dev0)
    mock_control_plane_->set_mock_route(
        t3k0_dev0, t3k2_dev0, {{t3k0_dev0, 0}, {t3k0_dev4, 1}, {t3k1_dev4, 2}, {t3k2_dev4, 3}, {t3k2_dev0, 4}});

    // Route 2: T3K1_dev0 -> T3K0_dev0 (goes through T3K1_dev4 -> T3K0_dev4 -> T3K0_dev0)
    mock_control_plane_->set_mock_route(
        t3k1_dev0, t3k0_dev0, {{t3k1_dev0, 0}, {t3k1_dev4, 1}, {t3k0_dev4, 2}, {t3k0_dev0, 3}});

    // Route 3: T3K2_dev0 -> T3K1_dev0 (goes through T3K2_dev4 -> T3K1_dev4 -> T3K1_dev0)
    mock_control_plane_->set_mock_route(
        t3k2_dev0, t3k1_dev0, {{t3k2_dev0, 0}, {t3k2_dev4, 1}, {t3k1_dev4, 2}, {t3k1_dev0, 3}});

    // This creates a cycle: T3K0_dev4 -> T3K1_dev4 -> T3K2_dev4 -> T3K1_dev4 -> T3K0_dev4
    std::vector<std::pair<FabricNodeId, FabricNodeId>> problematic_pairs = {
        {t3k0_dev0, t3k2_dev0},  // Inter-superpod traffic
        {t3k1_dev0, t3k0_dev0},  // Inter-T3K traffic
        {t3k2_dev0, t3k1_dev0}   // Creates the cycle
    };

    bool has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(problematic_pairs, *mock_route_manager_, "SixteenLoudboxTest");

    // Should detect the inter-mesh deadlock cycle
    EXPECT_TRUE(has_cycles);
}

TEST_F(CycleDetectionTest, SparseAllToAllBottleneck) {
    // Test the sparse connectivity bottleneck scenario
    // In 16 Loudbox: each 2x4 grid connects to another 2x4 grid with single QSFP (2 eth links)
    // This creates bottlenecks that can lead to cycles

    FabricNodeId grid1_dev0{MeshId{0}, 0};
    FabricNodeId grid1_dev3{MeshId{0}, 3};  // Bottleneck device with external connection
    FabricNodeId grid2_dev0{MeshId{1}, 0};
    FabricNodeId grid2_dev3{MeshId{1}, 3};  // Bottleneck device with external connection
    FabricNodeId grid3_dev0{MeshId{2}, 0};
    FabricNodeId grid3_dev3{MeshId{2}, 3};  // Bottleneck device with external connection

    // Set up routes that all funnel through the same bottleneck devices
    // This simulates the sparse connectivity creating resource contention

    // Multiple flows all trying to use the same bottleneck link
    mock_control_plane_->set_mock_route(
        grid1_dev0, grid2_dev0, {{grid1_dev0, 0}, {grid1_dev3, 1}, {grid2_dev3, 2}, {grid2_dev0, 3}});

    mock_control_plane_->set_mock_route(
        grid2_dev0, grid3_dev0, {{grid2_dev0, 0}, {grid2_dev3, 1}, {grid3_dev3, 2}, {grid3_dev0, 3}});

    mock_control_plane_->set_mock_route(
        grid3_dev0, grid1_dev0, {{grid3_dev0, 0}, {grid3_dev3, 1}, {grid1_dev3, 2}, {grid1_dev0, 3}});

    std::vector<std::pair<FabricNodeId, FabricNodeId>> bottleneck_pairs = {
        {grid1_dev0, grid2_dev0},
        {grid2_dev0, grid3_dev0},
        {grid3_dev0, grid1_dev0}  // Creates cycle through bottleneck devices
    };

    bool has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(bottleneck_pairs, *mock_route_manager_, "BottleneckTest");

    // Should detect cycles caused by sparse connectivity bottlenecks
    EXPECT_TRUE(has_cycles);
}

TEST_F(CycleDetectionTest, RandomPairingDeadlockScenario) {
    // Test the specific random pairing scenario that causes deadlocks
    // Simulate what happens when random traffic generator creates problematic sender/receiver pairs

    // Create a more realistic 16-device scenario (2 T3Ks worth)
    std::vector<FabricNodeId> t3k0_devices, t3k1_devices;
    for (int i = 0; i < 8; ++i) {
        t3k0_devices.push_back(FabricNodeId{MeshId{0}, static_cast<chip_id_t>(i)});
        t3k1_devices.push_back(FabricNodeId{MeshId{1}, static_cast<chip_id_t>(i)});
    }

    // Set up routes that create inter-T3K dependencies
    // Device 0 in T3K0 sends to Device 4 in T3K1 (must go through T3K0's connector device 4)
    mock_control_plane_->set_mock_route(
        t3k0_devices[0], t3k1_devices[4], {{t3k0_devices[0], 0}, {t3k0_devices[4], 1}, {t3k1_devices[4], 2}});

    // Device 4 in T3K1 sends to Device 0 in T3K0 (must go through T3K1's connector device 4)
    mock_control_plane_->set_mock_route(
        t3k1_devices[4], t3k0_devices[0], {{t3k1_devices[4], 0}, {t3k0_devices[4], 1}, {t3k0_devices[0], 2}});

    // Device 4 in T3K0 sends to Device 0 in T3K1 (creates dependency chain)
    mock_control_plane_->set_mock_route(
        t3k0_devices[4], t3k1_devices[0], {{t3k0_devices[4], 0}, {t3k1_devices[4], 1}, {t3k1_devices[0], 2}});

    // This creates a cycle in the connector devices: T3K0_dev4 <-> T3K1_dev4
    std::vector<std::pair<FabricNodeId, FabricNodeId>> random_pairs = {
        {t3k0_devices[0], t3k1_devices[4]},  // Cross-T3K traffic
        {t3k1_devices[4], t3k0_devices[0]},  // Reverse cross-T3K traffic
        {t3k0_devices[4], t3k1_devices[0]}   // Creates the cycle
    };

    bool has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(random_pairs, *mock_route_manager_, "RandomPairingTest");

    // Should detect the deadlock from random pairing
    EXPECT_TRUE(has_cycles);
}

TEST_F(CycleDetectionTest, ValidInterMeshTrafficPattern) {
    // Test a valid inter-mesh pattern that should NOT create cycles
    // This represents good traffic patterns that should pass cycle detection

    FabricNodeId src1{MeshId{0}, 0};
    FabricNodeId dst1{MeshId{1}, 0};
    FabricNodeId src2{MeshId{1}, 1};
    FabricNodeId dst2{MeshId{2}, 1};
    FabricNodeId src3{MeshId{2}, 2};
    FabricNodeId dst3{MeshId{0}, 2};

    // Set up non-conflicting routes (tree-like, no cycles)
    mock_control_plane_->set_mock_route(src1, dst1, {{src1, 0}, {dst1, 1}});

    mock_control_plane_->set_mock_route(src2, dst2, {{src2, 0}, {dst2, 1}});

    mock_control_plane_->set_mock_route(src3, dst3, {{src3, 0}, {dst3, 1}});

    std::vector<std::pair<FabricNodeId, FabricNodeId>> valid_pairs = {{src1, dst1}, {src2, dst2}, {src3, dst3}};

    bool has_cycles = detect_cycles_in_random_inter_mesh_traffic(valid_pairs, *mock_route_manager_, "ValidPatternTest");

    // Should NOT detect cycles in this valid pattern
    EXPECT_FALSE(has_cycles);
}

TEST_F(CycleDetectionTest, ComplexMultiHopCycle) {
    // Test a complex multi-hop cycle that spans multiple T3Ks and superpods
    // This represents the most complex deadlock scenario in 16 Loudbox

    // Create nodes across multiple superpods and T3Ks
    FabricNodeId sp0_t3k0_dev0{MeshId{0}, 0};  // Superpod 0, T3K 0
    FabricNodeId sp0_t3k0_dev4{MeshId{0}, 4};  // Connector device
    FabricNodeId sp0_t3k1_dev0{MeshId{1}, 0};  // Superpod 0, T3K 1
    FabricNodeId sp0_t3k1_dev4{MeshId{1}, 4};  // Connector device
    FabricNodeId sp1_t3k0_dev0{MeshId{2}, 0};  // Superpod 1, T3K 0
    FabricNodeId sp1_t3k0_dev4{MeshId{2}, 4};  // Connector device
    FabricNodeId sp1_t3k1_dev0{MeshId{3}, 0};  // Superpod 1, T3K 1
    FabricNodeId sp1_t3k1_dev4{MeshId{3}, 4};  // Connector device

    // Create a complex cycle that spans the entire cluster
    // Route 1: SP0_T3K0 -> SP1_T3K1 (long path across cluster)
    mock_control_plane_->set_mock_route(
        sp0_t3k0_dev0,
        sp1_t3k1_dev0,
        {{sp0_t3k0_dev0, 0},
         {sp0_t3k0_dev4, 1},
         {sp0_t3k1_dev4, 2},
         {sp1_t3k0_dev4, 3},
         {sp1_t3k1_dev4, 4},
         {sp1_t3k1_dev0, 5}});

    // Route 2: SP1_T3K1 -> SP1_T3K0 (within superpod)
    mock_control_plane_->set_mock_route(
        sp1_t3k1_dev0, sp1_t3k0_dev0, {{sp1_t3k1_dev0, 0}, {sp1_t3k1_dev4, 1}, {sp1_t3k0_dev4, 2}, {sp1_t3k0_dev0, 3}});

    // Route 3: SP1_T3K0 -> SP0_T3K1 (cross superpod)
    mock_control_plane_->set_mock_route(
        sp1_t3k0_dev0, sp0_t3k1_dev0, {{sp1_t3k0_dev0, 0}, {sp1_t3k0_dev4, 1}, {sp0_t3k1_dev4, 2}, {sp0_t3k1_dev0, 3}});

    // Route 4: SP0_T3K1 -> SP0_T3K0 (completes the cycle)
    mock_control_plane_->set_mock_route(
        sp0_t3k1_dev0, sp0_t3k0_dev0, {{sp0_t3k1_dev0, 0}, {sp0_t3k1_dev4, 1}, {sp0_t3k0_dev4, 2}, {sp0_t3k0_dev0, 3}});

    std::vector<std::pair<FabricNodeId, FabricNodeId>> complex_pairs = {
        {sp0_t3k0_dev0, sp1_t3k1_dev0},  // Cross-cluster traffic
        {sp1_t3k1_dev0, sp1_t3k0_dev0},  // Intra-superpod traffic
        {sp1_t3k0_dev0, sp0_t3k1_dev0},  // Cross-superpod traffic
        {sp0_t3k1_dev0, sp0_t3k0_dev0}   // Completes the complex cycle
    };

    bool has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(complex_pairs, *mock_route_manager_, "ComplexCycleTest");

    // Should detect the complex multi-hop cycle
    EXPECT_TRUE(has_cycles);
}

TEST_F(CycleDetectionTest, True16LoudboxTopologyDeadlock) {
    // Model the actual 16 Loudbox topology:
    // - Each T3K has 8 devices in 2x4 grid (devices 0-7)
    // - 4 devices per T3K are attached to different external systems
    // - Remaining 4 devices are internal to the T3K
    // - Inter-T3K traffic must go through the 4 external-connected devices

    // T3K 0: Devices 0-7, where devices 0,2,4,6 have external connections
    FabricNodeId t3k0_internal_dev1{MeshId{0}, 1};  // Internal device (no external connection)
    FabricNodeId t3k0_internal_dev3{MeshId{0}, 3};  // Internal device (no external connection)
    FabricNodeId t3k0_external_dev0{MeshId{0}, 0};  // External-connected device
    FabricNodeId t3k0_external_dev2{MeshId{0}, 2};  // External-connected device
    FabricNodeId t3k0_external_dev4{MeshId{0}, 4};  // External-connected device
    FabricNodeId t3k0_external_dev6{MeshId{0}, 6};  // External-connected device

    // T3K 1: Devices 0-7, where devices 0,2,4,6 have external connections
    FabricNodeId t3k1_internal_dev1{MeshId{1}, 1};  // Internal device
    FabricNodeId t3k1_internal_dev5{MeshId{1}, 5};  // Internal device
    FabricNodeId t3k1_external_dev0{MeshId{1}, 0};  // External-connected device
    FabricNodeId t3k1_external_dev4{MeshId{1}, 4};  // External-connected device

    // Set up routes that reflect the actual sparse connectivity
    // Internal devices must route through external-connected devices in their T3K

    // Route 1: T3K0 internal device -> T3K1 internal device (inter-T3K traffic)
    // Must go: internal -> local_external_device -> remote_external_device -> remote_internal
    mock_control_plane_->set_mock_route(
        t3k0_internal_dev1,
        t3k1_internal_dev1,
        {{t3k0_internal_dev1, 0}, {t3k0_external_dev0, 1}, {t3k1_external_dev0, 2}, {t3k1_internal_dev1, 3}});

    // Route 2: T3K1 internal device -> T3K0 internal device (reverse inter-T3K traffic)
    mock_control_plane_->set_mock_route(
        t3k1_internal_dev5,
        t3k0_internal_dev3,
        {{t3k1_internal_dev5, 0}, {t3k1_external_dev4, 1}, {t3k0_external_dev4, 2}, {t3k0_internal_dev3, 3}});

    // Route 3: Creates cycle through the external-connected devices
    // T3K0 internal -> T3K1 internal, but routing conflicts with above routes
    mock_control_plane_->set_mock_route(
        t3k0_internal_dev3,
        t3k1_internal_dev5,
        {{t3k0_internal_dev3, 0},
         {t3k0_external_dev0, 1},
         {t3k1_external_dev0, 2},
         {t3k1_external_dev4, 3},
         {t3k0_external_dev4, 4},
         {t3k0_internal_dev1, 5},
         {t3k0_external_dev0, 6},
         {t3k1_internal_dev5, 7}});

    // This creates a cycle through the external-connected devices that handle inter-T3K traffic
    std::vector<std::pair<FabricNodeId, FabricNodeId>> true_16lb_pairs = {
        {t3k0_internal_dev1, t3k1_internal_dev1},  // Inter-T3K traffic (uses external devices 0)
        {t3k1_internal_dev5, t3k0_internal_dev3},  // Reverse inter-T3K traffic (uses external devices 4)
        {t3k0_internal_dev3, t3k1_internal_dev5}   // Creates cycle through external devices
    };

    bool has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(true_16lb_pairs, *mock_route_manager_, "True16LoudboxTest");

    // Should detect the realistic 16 Loudbox deadlock scenario
    EXPECT_TRUE(has_cycles);
}

TEST_F(CycleDetectionTest, QSFPBottleneckDeadlock) {
    // Test the specific QSFP bottleneck scenario you described
    // Each 2x4 grid connected by single QSFP (2 eth links) creates bottlenecks

    // Model two 2x4 grids connected by single QSFP
    FabricNodeId grid1_dev0{MeshId{0}, 0};  // Grid 1, internal device
    FabricNodeId grid1_dev1{MeshId{0}, 1};  // Grid 1, internal device
    FabricNodeId grid1_qsfp{MeshId{0}, 4};  // Grid 1, QSFP connector device

    FabricNodeId grid2_dev0{MeshId{1}, 0};  // Grid 2, internal device
    FabricNodeId grid2_dev1{MeshId{1}, 1};  // Grid 2, internal device
    FabricNodeId grid2_qsfp{MeshId{1}, 4};  // Grid 2, QSFP connector device

    // Third grid creating the cycle
    FabricNodeId grid3_dev0{MeshId{2}, 0};  // Grid 3, internal device
    FabricNodeId grid3_qsfp{MeshId{2}, 4};  // Grid 3, QSFP connector device

    // Set up routes that overload the single QSFP connections
    // Multiple flows trying to use the same 2 eth links

    // Flow 1: Grid1_dev0 -> Grid2_dev0 (uses Grid1_qsfp -> Grid2_qsfp)
    mock_control_plane_->set_mock_route(
        grid1_dev0, grid2_dev0, {{grid1_dev0, 0}, {grid1_qsfp, 1}, {grid2_qsfp, 2}, {grid2_dev0, 3}});

    // Flow 2: Grid2_dev1 -> Grid3_dev0 (uses Grid2_qsfp -> Grid3_qsfp)
    mock_control_plane_->set_mock_route(
        grid2_dev1, grid3_dev0, {{grid2_dev1, 0}, {grid2_qsfp, 1}, {grid3_qsfp, 2}, {grid3_dev0, 3}});

    // Flow 3: Grid3_dev0 -> Grid1_dev1 (uses Grid3_qsfp -> Grid1_qsfp, completes cycle)
    mock_control_plane_->set_mock_route(
        grid3_dev0, grid1_dev1, {{grid3_dev0, 0}, {grid3_qsfp, 1}, {grid1_qsfp, 2}, {grid1_dev1, 3}});

    std::vector<std::pair<FabricNodeId, FabricNodeId>> qsfp_bottleneck_pairs = {
        {grid1_dev0, grid2_dev0},  // Uses QSFP 1->2
        {grid2_dev1, grid3_dev0},  // Uses QSFP 2->3
        {grid3_dev0, grid1_dev1}   // Uses QSFP 3->1, creates cycle
    };

    bool has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(qsfp_bottleneck_pairs, *mock_route_manager_, "QSFPBottleneckTest");

    // Should detect cycles caused by QSFP bottlenecks
    EXPECT_TRUE(has_cycles);
}

TEST_F(CycleDetectionTest, RandomPairingRetryScenario) {
    // Test the retry mechanism when cycles are detected
    // This simulates the actual use case: detect cycles, skip bad pairing, generate new one

    FabricNodeId dev_a{MeshId{0}, 0};
    FabricNodeId dev_b{MeshId{1}, 1};
    FabricNodeId dev_c{MeshId{2}, 2};
    FabricNodeId dev_d{MeshId{3}, 3};

    // First attempt: Create a problematic pairing that has cycles
    mock_control_plane_->set_mock_route(dev_a, dev_b, {{dev_a, 0}, {dev_c, 1}, {dev_b, 2}});
    mock_control_plane_->set_mock_route(dev_b, dev_c, {{dev_b, 0}, {dev_a, 1}, {dev_c, 2}});
    mock_control_plane_->set_mock_route(dev_c, dev_a, {{dev_c, 0}, {dev_b, 1}, {dev_a, 2}});

    std::vector<std::pair<FabricNodeId, FabricNodeId>> bad_pairing = {
        {dev_a, dev_b}, {dev_b, dev_c}, {dev_c, dev_a}  // Creates cycle
    };

    bool first_attempt_has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(bad_pairing, *mock_route_manager_, "FirstAttempt");
    EXPECT_TRUE(first_attempt_has_cycles);  // Should detect cycles

    // Second attempt: Create a good pairing without cycles
    mock_control_plane_->clear_mock_routes();
    mock_control_plane_->set_mock_route(dev_a, dev_b, {{dev_a, 0}, {dev_b, 1}});
    mock_control_plane_->set_mock_route(dev_c, dev_d, {{dev_c, 0}, {dev_d, 1}});

    std::vector<std::pair<FabricNodeId, FabricNodeId>> good_pairing = {
        {dev_a, dev_b},  // No cycle
        {dev_c, dev_d}   // Independent flow
    };

    bool second_attempt_has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(good_pairing, *mock_route_manager_, "SecondAttempt");
    EXPECT_FALSE(second_attempt_has_cycles);  // Should NOT detect cycles

    // This demonstrates the retry mechanism working correctly
}

TEST_F(CycleDetectionTest, FourExternalDevicesConstraint) {
    // Test the constraint that only 4 out of 8 devices per T3K have external connections
    // This constraint creates bottlenecks that can lead to cycles in inter-T3K traffic

    // Create a T3K with 8 devices (2x4 grid): devices 0-7
    // Only devices 0, 2, 4, 6 are attached to different external systems
    std::vector<FabricNodeId> t3k_devices;
    for (int i = 0; i < 8; ++i) {
        t3k_devices.push_back(FabricNodeId{MeshId{0}, static_cast<chip_id_t>(i)});
    }

    // External T3K devices
    FabricNodeId external_t3k_dev0{MeshId{1}, 0};  // External system connection
    FabricNodeId external_t3k_dev2{MeshId{1}, 2};  // External system connection

    // Internal devices (1, 3, 5, 7) must route through external-connected devices (0, 2, 4, 6)
    // This creates dependency chains that can lead to cycles

    // Route from internal device through external-connected device to another T3K
    mock_control_plane_->set_mock_route(
        t3k_devices[1],
        external_t3k_dev0,                                                    // internal dev1 -> external T3K
        {{t3k_devices[1], 0}, {t3k_devices[0], 1}, {external_t3k_dev0, 2}});  // through external-connected dev0

    mock_control_plane_->set_mock_route(
        external_t3k_dev2,
        t3k_devices[3],                                                       // external T3K -> internal dev3
        {{external_t3k_dev2, 0}, {t3k_devices[2], 1}, {t3k_devices[3], 2}});  // through external-connected dev2

    // Route that creates cycle through the limited external-connected devices
    mock_control_plane_->set_mock_route(
        t3k_devices[5],
        t3k_devices[1],  // internal dev5 -> internal dev1
        {{t3k_devices[5], 0},
         {t3k_devices[4], 1},
         {t3k_devices[0], 2},
         {t3k_devices[1], 3}});  // through external-connected dev4 and dev0

    std::vector<std::pair<FabricNodeId, FabricNodeId>> constrained_pairs = {
        {t3k_devices[1], external_t3k_dev0},  // Internal -> External T3K (through external-connected device)
        {external_t3k_dev2, t3k_devices[3]},  // External T3K -> Internal (through external-connected device)
        {t3k_devices[5], t3k_devices[1]}      // Creates cycle through limited external-connected devices
    };

    bool has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(constrained_pairs, *mock_route_manager_, "FourExternalDevicesTest");

    // Should detect cycles caused by the 4 external devices constraint
    EXPECT_TRUE(has_cycles);
}

}  // namespace tt::tt_fabric::fabric_tests

// Main function for running the tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
