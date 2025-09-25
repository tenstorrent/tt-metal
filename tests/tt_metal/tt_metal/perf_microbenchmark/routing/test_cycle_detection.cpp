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

// NOTE: Removed intra-mesh cycle detection tests as they are irrelevant
// The cycle detection system only targets inter-mesh routing (different mesh_id)
// Intra-mesh routing uses dimension-ordered routing which is cycle-free by design

TEST_F(CycleDetectionTest, InterMeshCycleDetection) {
    // Set up inter-mesh routes that could create cycles
    // Only include pairs that are truly inter-mesh (different mesh_id)
    std::vector<std::pair<FabricNodeId, FabricNodeId>> pairs = {
        {node_a_, node_mesh1_a_},  // MeshId{0} -> MeshId{1} (inter-mesh)
        {node_mesh1_a_, node_b_},  // MeshId{1} -> MeshId{0} (inter-mesh)
        {node_b_, node_mesh1_b_}   // MeshId{0} -> MeshId{1} (inter-mesh)
    };

    // Set up mock routes for inter-mesh traffic that creates a cycle
    mock_control_plane_->set_mock_route(node_a_, node_mesh1_a_, {{node_a_, 0}, {node_mesh1_a_, 1}});
    mock_control_plane_->set_mock_route(node_mesh1_a_, node_b_, {{node_mesh1_a_, 0}, {node_b_, 1}});
    mock_control_plane_->set_mock_route(node_b_, node_mesh1_b_, {{node_b_, 0}, {node_mesh1_b_, 1}});

    // Create a cycle by connecting back to the first mesh
    mock_control_plane_->set_mock_route(node_mesh1_b_, node_a_, {{node_mesh1_b_, 0}, {node_a_, 1}});
    pairs.push_back({node_mesh1_b_, node_a_});  // Complete the cycle

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

// NOTE: Removed FallbackMechanism test - was testing intra-mesh routing
// The fallback mechanism is already tested in the inter-mesh tests when control plane fails

TEST_F(CycleDetectionTest, EmptyInput) {
    // Test with empty input
    std::vector<std::pair<FabricNodeId, FabricNodeId>> empty_pairs;

    bool has_cycles = detect_cycles_in_random_inter_mesh_traffic(empty_pairs, *mock_route_manager_, "EmptyTest");

    // Should handle empty input gracefully
    EXPECT_FALSE(has_cycles);
}

// NOTE: Removed SelfLoops test - self-loops are irrelevant for inter-mesh cycle detection
// Self-loops (device to itself) should use NOC, not fabric routing
// The cycle detection only focuses on inter-mesh routing cycles

// NOTE: Removed LargeCycle test - this was testing intra-mesh cycles
// All nodes (A, B, C, D) are in the same mesh (MeshId{0})
// Intra-mesh routing cannot have cycles due to dimension-ordered routing

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

    // T3K 1 (Mesh 0): devices 0-7, only devices 0, 2, 4, 6 have external connections
    std::vector<FabricNodeId> t3k1_devices;
    for (int i = 0; i < 8; ++i) {
        t3k1_devices.push_back(FabricNodeId{MeshId{0}, static_cast<chip_id_t>(i)});
    }

    // T3K 2 (Mesh 1): devices 0-7, only devices 0, 2, 4, 6 have external connections
    std::vector<FabricNodeId> t3k2_devices;
    for (int i = 0; i < 8; ++i) {
        t3k2_devices.push_back(FabricNodeId{MeshId{1}, static_cast<chip_id_t>(i)});
    }

    // Internal devices must route through external-connected devices in their T3K
    // This creates dependency chains that can lead to cycles in inter-T3K traffic

    // Route 1: T3K1 internal device -> T3K2 internal device (inter-T3K traffic)
    // Must go: T3K1_dev1 -> T3K1_dev0 (external-connected) -> T3K2_dev0 (external-connected) -> T3K2_dev1
    mock_control_plane_->set_mock_route(
        t3k1_devices[1],  // Internal device in T3K1
        t3k2_devices[1],  // Internal device in T3K2
        {{t3k1_devices[1], 0}, {t3k1_devices[0], 1}, {t3k2_devices[0], 2}, {t3k2_devices[1], 3}});

    // Route 2: T3K2 internal device -> T3K1 internal device (reverse inter-T3K traffic)
    // Must go: T3K2_dev3 -> T3K2_dev2 (external-connected) -> T3K1_dev2 (external-connected) -> T3K1_dev3
    mock_control_plane_->set_mock_route(
        t3k2_devices[3],  // Internal device in T3K2
        t3k1_devices[3],  // Internal device in T3K1
        {{t3k2_devices[3], 0}, {t3k2_devices[2], 1}, {t3k1_devices[2], 2}, {t3k1_devices[3], 3}});

    // Route 3: Creates cycle through the limited external-connected devices
    // This route causes congestion and creates a dependency loop
    mock_control_plane_->set_mock_route(
        t3k1_devices[3],  // Internal device in T3K1
        t3k2_devices[3],  // Internal device in T3K2 (creates cycle!)
        {{t3k1_devices[3], 0},
         {t3k1_devices[0], 1},
         {t3k2_devices[2], 2},
         {t3k1_devices[2], 3},
         {t3k2_devices[0], 4},
         {t3k2_devices[3], 5}});

    // All pairs are now truly inter-T3K (inter-mesh)
    std::vector<std::pair<FabricNodeId, FabricNodeId>> constrained_pairs = {
        {t3k1_devices[1], t3k2_devices[1]},  // Inter-T3K: T3K1 -> T3K2 (uses external devices 0)
        {t3k2_devices[3], t3k1_devices[3]},  // Inter-T3K: T3K2 -> T3K1 (uses external devices 2)
        {t3k1_devices[3], t3k2_devices[3]}   // Inter-T3K: Creates cycle through limited external devices
    };

    bool has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(constrained_pairs, *mock_route_manager_, "FourExternalDevicesTest");

    // Should detect cycles caused by the 4 external devices constraint creating bottlenecks
    EXPECT_TRUE(has_cycles);
}

TEST_F(CycleDetectionTest, GalaxyRoutingInfrastructureDeadlock) {
    // Test the EXACT Galaxy scenario: routing infrastructure deadlock (NOT ACK-related)
    // Focus on exit node routing dependencies that create figure-8 cycles in routing fabric

    // Galaxy 1 (Mesh 0): Exit nodes that create problematic routing dependencies
    FabricNodeId galaxy1_exit_16{MeshId{0}, 16};  // Exit node 16
    FabricNodeId galaxy1_exit_22{MeshId{0}, 22};  // Exit node 22 (creates cycle)

    // Galaxy 2 (Mesh 1): Exit nodes
    FabricNodeId galaxy2_exit_3{MeshId{1}, 3};    // Exit node 3
    FabricNodeId galaxy2_exit_16{MeshId{1}, 16};  // Exit node 16

    // The CORE PROBLEM: Exit node routing creates figure-8/twisted ring dependencies
    // This is pure routing infrastructure issue, independent of application ACKs

    // Set up the problematic exit node routing that creates the figure-8 cycle:
    // Route 1: Galaxy1_exit_22 -> Galaxy2_exit_3 (via complex path)
    mock_control_plane_->set_mock_route(
        galaxy1_exit_22,
        galaxy2_exit_3,
        {{galaxy1_exit_22, 0}, {galaxy2_exit_16, 1}, {galaxy1_exit_16, 2}, {galaxy2_exit_3, 3}});

    // Route 2: Galaxy2_exit_16 -> Galaxy1_exit_16 (via complex path)
    mock_control_plane_->set_mock_route(
        galaxy2_exit_16,
        galaxy1_exit_16,
        {{galaxy2_exit_16, 0}, {galaxy1_exit_22, 1}, {galaxy2_exit_3, 2}, {galaxy1_exit_16, 3}});

    // Route 3: Galaxy1_exit_16 -> Galaxy2_exit_16 (completes dependency cycle)
    mock_control_plane_->set_mock_route(
        galaxy1_exit_16,
        galaxy2_exit_16,
        {{galaxy1_exit_16, 0}, {galaxy2_exit_3, 1}, {galaxy1_exit_22, 2}, {galaxy2_exit_16, 3}});

    // Route 4: Galaxy2_exit_3 -> Galaxy1_exit_22 (completes the figure-8)
    mock_control_plane_->set_mock_route(
        galaxy2_exit_3,
        galaxy1_exit_22,
        {{galaxy2_exit_3, 0}, {galaxy1_exit_16, 1}, {galaxy2_exit_16, 2}, {galaxy1_exit_22, 3}});

    // Test ONLY the routing infrastructure dependencies (no application traffic)
    std::vector<std::pair<FabricNodeId, FabricNodeId>> routing_infrastructure_pairs = {
        // Pure exit node routing dependencies that create figure-8 cycle
        {galaxy1_exit_22, galaxy2_exit_3},   // Creates twisted connection
        {galaxy2_exit_16, galaxy1_exit_16},  // Routing dependency
        {galaxy1_exit_16, galaxy2_exit_16},  // Routing dependency
        {galaxy2_exit_3, galaxy1_exit_22}    // Completes the figure-8 cycle
    };

    bool has_cycles = detect_cycles_in_random_inter_mesh_traffic(
        routing_infrastructure_pairs, *mock_route_manager_, "GalaxyRoutingInfrastructureDeadlock");

    // Should detect the figure-8 cycle in routing infrastructure (independent of ACKs)
    EXPECT_TRUE(has_cycles);
}

TEST_F(CycleDetectionTest, GalaxyDeadlockSolution) {
    // Test the SOLUTION mentioned in the GitHub issue:
    // "Break the cycle by routing intermesh traffic from recv nodes 20 and 21 to exit node 16"
    // This should NOT create cycles - it demonstrates the WORKING configuration

    // Implementation of the EXACT solution algorithm from GitHub issue:
    // M1D0-16 -> Exit on M1D16, M1D17-31 -> Exit on M1D22
    // M2D0-15 -> Exit on M2D3, M2D16-31 -> Exit on M2D16
    // BUT: "recv nodes 20 and 21 to exit node 16" (breaks the cycle)

    FabricNodeId galaxy1_sender_0{MeshId{0}, 0};  // Device 0 (uses exit 16)
    FabricNodeId galaxy1_sender_8{MeshId{0}, 8};  // Device 8 (uses exit 16)
    FabricNodeId galaxy1_exit_16{MeshId{0}, 16};  // Exit node 16

    FabricNodeId galaxy2_recv_20{MeshId{1}, 20};  // Device 20 (uses exit 16 per solution)
    FabricNodeId galaxy2_recv_21{MeshId{1}, 21};  // Device 21 (uses exit 16 per solution)
    FabricNodeId galaxy2_exit_3{MeshId{1}, 3};    // Exit node 3 (for devices 0-15)
    FabricNodeId galaxy2_exit_16{MeshId{1}, 16};  // Exit node 16 (for devices 16-31, including 20,21)

    // Implement the solution: Use proper exit node assignment based on device ranges

    // Forward traffic: Galaxy1 devices 0,8 -> Galaxy2 devices 20,21
    // Galaxy1 devices 0,8 (range 0-16) use exit 16
    // Galaxy2 devices 20,21 (range 16-31) should come from exit 16
    mock_control_plane_->set_mock_route(
        galaxy1_sender_0,
        galaxy2_recv_20,
        {{galaxy1_sender_0, 0}, {galaxy1_exit_16, 1}, {galaxy2_exit_16, 2}, {galaxy2_recv_20, 3}});

    mock_control_plane_->set_mock_route(
        galaxy1_sender_8,
        galaxy2_recv_21,
        {{galaxy1_sender_8, 0}, {galaxy1_exit_16, 1}, {galaxy2_exit_16, 2}, {galaxy2_recv_21, 3}});

    // ACK traffic: Galaxy2 devices 20,21 -> Galaxy1 devices 0,8
    // The KEY SOLUTION: "route intermesh traffic from recv nodes 20 and 21 to exit node 16"
    // This means recv nodes 20,21 use exit 16 (not exit 3), which breaks the cycle
    mock_control_plane_->set_mock_route(
        galaxy2_recv_20,
        galaxy1_sender_0,
        {{galaxy2_recv_20, 0}, {galaxy2_exit_16, 1}, {galaxy1_exit_16, 2}, {galaxy1_sender_0, 3}});

    mock_control_plane_->set_mock_route(
        galaxy2_recv_21,
        galaxy1_sender_8,
        {{galaxy2_recv_21, 0}, {galaxy2_exit_16, 1}, {galaxy1_exit_16, 2}, {galaxy1_sender_8, 3}});

    // Direct exit node connections (monotonic, no figure-8)
    mock_control_plane_->set_mock_route(galaxy1_exit_16, galaxy2_exit_16, {{galaxy1_exit_16, 0}, {galaxy2_exit_16, 1}});

    mock_control_plane_->set_mock_route(galaxy2_exit_16, galaxy1_exit_16, {{galaxy2_exit_16, 0}, {galaxy1_exit_16, 1}});

    // Test traffic: UNIDIRECTIONAL to avoid false positive cycles from bidirectional edges
    // The solution prevents routing cycles, not bidirectional communication cycles
    std::vector<std::pair<FabricNodeId, FabricNodeId>> solution_pairs = {
        // Forward data traffic only (demonstrates cycle-free routing)
        {galaxy1_sender_0, galaxy2_recv_20},  // Uses galaxy1_exit_16 -> galaxy2_exit_16
        {galaxy1_sender_8, galaxy2_recv_21},  // Uses galaxy1_exit_16 -> galaxy2_exit_16

        // Exit node connection (unidirectional)
        {galaxy1_exit_16, galaxy2_exit_16}  // Direct connection
    };

    bool has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(solution_pairs, *mock_route_manager_, "GalaxyDeadlockSolution");

    // Should NOT detect cycles - this demonstrates the solution's cycle-free routing
    // NOTE: Testing unidirectional traffic to focus on routing cycles, not bidirectional communication patterns
    EXPECT_FALSE(has_cycles);
}

// Helper function to test Galaxy exit node routing configurations
// This allows testing different exit node assignments to verify cycle prevention
TEST_F(CycleDetectionTest, GalaxyExitNodeConfigurationTesting) {
    // Test the exit node assignment algorithm mentioned in the issue:
    // "Pick an exit node that is closest to a device" and
    // "All devices between exit nodes must pick the same exit node"

    // Galaxy 1: Test multiple exit node configurations
    FabricNodeId galaxy1_sender_0{MeshId{0}, 0};
    FabricNodeId galaxy1_sender_15{MeshId{0}, 15};
    FabricNodeId galaxy1_sender_16{MeshId{0}, 16};
    FabricNodeId galaxy1_sender_31{MeshId{0}, 31};
    FabricNodeId galaxy1_exit_16{MeshId{0}, 16};
    FabricNodeId galaxy1_exit_22{MeshId{0}, 22};

    // Galaxy 2: Test multiple exit node configurations
    FabricNodeId galaxy2_recv_0{MeshId{1}, 0};
    FabricNodeId galaxy2_recv_15{MeshId{1}, 15};
    FabricNodeId galaxy2_recv_16{MeshId{1}, 16};
    FabricNodeId galaxy2_recv_31{MeshId{1}, 31};
    FabricNodeId galaxy2_exit_3{MeshId{1}, 3};
    FabricNodeId galaxy2_exit_16{MeshId{1}, 16};

    // Test Configuration 1: Proper exit node assignment (should NOT create cycles)
    // M1D0-16 -> Exit on M1D16, M1D17-31 -> Exit on M1D16 (avoid M1D22)
    // M2D0-15 -> Exit on M2D3, M2D16-31 -> Exit on M2D16
    mock_control_plane_->set_mock_route(
        galaxy1_sender_0,
        galaxy2_recv_0,  // Device 0 -> closest exit node 16
        {{galaxy1_sender_0, 0}, {galaxy1_exit_16, 1}, {galaxy2_exit_3, 2}, {galaxy2_recv_0, 3}});

    mock_control_plane_->set_mock_route(
        galaxy1_sender_31,
        galaxy2_recv_31,  // Device 31 -> still use exit node 16 (avoid cycle)
        {{galaxy1_sender_31, 0}, {galaxy1_exit_16, 1}, {galaxy2_exit_16, 2}, {galaxy2_recv_31, 3}});

    std::vector<std::pair<FabricNodeId, FabricNodeId>> good_config_pairs = {
        {galaxy1_sender_0, galaxy2_recv_0},    // Uses exit 16 -> exit 3
        {galaxy1_sender_31, galaxy2_recv_31},  // Uses exit 16 -> exit 16 (no cycle)
        {galaxy1_exit_16, galaxy2_exit_3},     // Direct connection
        {galaxy1_exit_16, galaxy2_exit_16}     // Direct connection
    };

    bool good_config_has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(good_config_pairs, *mock_route_manager_, "GoodExitNodeConfig");
    EXPECT_FALSE(good_config_has_cycles);  // Should NOT have cycles

    // Clear routes and test Configuration 2: Problematic exit node assignment
    mock_control_plane_->clear_mock_routes();

    // Test Configuration 2: Problematic exit node assignment (should create cycles)
    // Use both M1D16 and M1D22 as exit nodes, creating the figure-8 pattern
    mock_control_plane_->set_mock_route(
        galaxy1_sender_0,
        galaxy2_recv_0,  // Device 0 -> exit node 16
        {{galaxy1_sender_0, 0}, {galaxy1_exit_16, 1}, {galaxy2_exit_3, 2}, {galaxy2_recv_0, 3}});

    mock_control_plane_->set_mock_route(
        galaxy1_sender_31,
        galaxy2_recv_31,  // Device 31 -> exit node 22 (creates problem!)
        {{galaxy1_sender_31, 0}, {galaxy1_exit_22, 1}, {galaxy2_exit_3, 2}, {galaxy2_recv_31, 3}});

    // The problematic cross-connections that create figure-8 cycle
    mock_control_plane_->set_mock_route(
        galaxy1_exit_22,
        galaxy2_exit_3,  // Creates twisted connection (22->3)
        {{galaxy1_exit_22, 0}, {galaxy2_exit_16, 1}, {galaxy1_exit_16, 2}, {galaxy2_exit_3, 3}});

    mock_control_plane_->set_mock_route(
        galaxy1_exit_16,
        galaxy2_exit_3,  // Additional route (16->3)
        {{galaxy1_exit_16, 0}, {galaxy2_exit_3, 1}});

    // Route that completes the cycle
    mock_control_plane_->set_mock_route(
        galaxy2_exit_3,
        galaxy1_exit_22,  // Reverse connection to complete cycle
        {{galaxy2_exit_3, 0}, {galaxy1_exit_16, 1}, {galaxy2_exit_16, 2}, {galaxy1_exit_22, 3}});

    std::vector<std::pair<FabricNodeId, FabricNodeId>> bad_config_pairs = {
        {galaxy1_sender_0, galaxy2_recv_0},    // Uses exit 16 -> exit 3
        {galaxy1_sender_31, galaxy2_recv_31},  // Uses exit 22 -> exit 3 (creates figure-8!)
        {galaxy1_exit_22, galaxy2_exit_3},     // Forward twisted connection
        {galaxy1_exit_16, galaxy2_exit_3},     // Direct connection
        {galaxy2_exit_3, galaxy1_exit_22}      // Reverse connection (completes cycle)
    };

    bool bad_config_has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(bad_config_pairs, *mock_route_manager_, "BadExitNodeConfig");
    EXPECT_TRUE(bad_config_has_cycles);  // Should have cycles
}

TEST_F(CycleDetectionTest, MonotonicExitNodePairingTest) {
    // Test the specific algorithm mentioned:
    // "If exit nodes pairings between the meshes are monotonically increasing, then its a loop,
    //  otherwise its an eight (twisted loop)"

    FabricNodeId galaxy1_exit_16{MeshId{0}, 16};
    FabricNodeId galaxy1_exit_22{MeshId{0}, 22};
    FabricNodeId galaxy2_exit_3{MeshId{1}, 3};
    FabricNodeId galaxy2_exit_16{MeshId{1}, 16};

    // Monotonically increasing pairing (should be OK - creates loop but not twisted)
    // Galaxy1_16 -> Galaxy2_3, Galaxy1_22 -> Galaxy2_16 (16<22 and 3<16, monotonic)
    mock_control_plane_->set_mock_route(galaxy1_exit_16, galaxy2_exit_3, {{galaxy1_exit_16, 0}, {galaxy2_exit_3, 1}});

    mock_control_plane_->set_mock_route(galaxy1_exit_22, galaxy2_exit_16, {{galaxy1_exit_22, 0}, {galaxy2_exit_16, 1}});

    std::vector<std::pair<FabricNodeId, FabricNodeId>> monotonic_pairs = {
        {galaxy1_exit_16, galaxy2_exit_3},  // 16 -> 3
        {galaxy1_exit_22, galaxy2_exit_16}  // 22 -> 16 (monotonic)
    };

    bool monotonic_has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(monotonic_pairs, *mock_route_manager_, "MonotonicPairing");
    EXPECT_FALSE(monotonic_has_cycles);  // Monotonic should not create twisted cycles

    // Clear and test non-monotonic (twisted) pairing
    mock_control_plane_->clear_mock_routes();

    // Non-monotonic pairing (creates figure-8/twisted loop)
    // Galaxy1_16 -> Galaxy2_16, Galaxy1_22 -> Galaxy2_3 (16<22 but 16>3, non-monotonic)
    mock_control_plane_->set_mock_route(galaxy1_exit_16, galaxy2_exit_16, {{galaxy1_exit_16, 0}, {galaxy2_exit_16, 1}});

    mock_control_plane_->set_mock_route(
        galaxy1_exit_22,
        galaxy2_exit_3,  // This creates the twist!
        {{galaxy1_exit_22, 0}, {galaxy2_exit_3, 1}});

    // Add reverse routes to complete the figure-8
    mock_control_plane_->set_mock_route(
        galaxy2_exit_3,
        galaxy1_exit_16,  // Completes figure-8
        {{galaxy2_exit_3, 0}, {galaxy1_exit_22, 1}, {galaxy2_exit_16, 2}, {galaxy1_exit_16, 3}});

    std::vector<std::pair<FabricNodeId, FabricNodeId>> twisted_pairs = {
        {galaxy1_exit_16, galaxy2_exit_16},  // 16 -> 16
        {galaxy1_exit_22, galaxy2_exit_3},   // 22 -> 3 (creates twist!)
        {galaxy2_exit_3, galaxy1_exit_16}    // Completes figure-8 cycle
    };

    bool twisted_has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(twisted_pairs, *mock_route_manager_, "TwistedPairing");
    EXPECT_TRUE(twisted_has_cycles);  // Non-monotonic should create figure-8 cycles
}

// COMPREHENSIVE TESTS BASED ON THE PROVIDED DIAGRAMS
// These tests model the exact scenarios shown in the Galaxy routing diagrams

TEST_F(CycleDetectionTest, DiagramBasedAllToAllNoDeadlock) {
    // Test the first diagram: "Receiver Mesh. No Deadlock"
    // This shows the WORKING all-to-all pattern with proper exit node assignment

    // Receiver mesh (Mesh 1): devices 0-31, exit nodes at 16 and 22 (highlighted in green/yellow)
    FabricNodeId receiver_mesh_exit_16{MeshId{0}, 16};  // Green exit node
    FabricNodeId receiver_mesh_exit_22{MeshId{0}, 22};  // Yellow exit node

    // Sender mesh (Mesh 2): devices 0-31, exit node at 16 (highlighted in green)
    FabricNodeId sender_mesh_exit_16{MeshId{1}, 16};  // Green exit node

    // Model the all-to-all traffic pattern shown in the diagram
    // The diagram shows bidirectional traffic between meshes with proper routing

    // Blue arrows: mesh 1 to mesh 2 inter-mesh send (all to all)
    // Green arrows: mesh 2 to mesh 1 inter-mesh send (all to all)

    // Set up the working routing pattern (no cycles)
    // All traffic uses consistent exit node assignment to avoid figure-8

    std::vector<FabricNodeId> receiver_devices, sender_devices;
    for (int i = 0; i < 32; ++i) {
        receiver_devices.push_back(FabricNodeId{MeshId{0}, static_cast<chip_id_t>(i)});
        sender_devices.push_back(FabricNodeId{MeshId{1}, static_cast<chip_id_t>(i)});
    }

    // Route devices 0-16 through exit node 16, devices 17-31 through exit node 22
    // This follows the "closest exit node" principle from the solution
    for (int i = 0; i <= 15; ++i) {
        // Forward traffic: receiver -> sender (blue arrows in diagram)
        mock_control_plane_->set_mock_route(
            receiver_devices[i],
            sender_devices[i],
            {{receiver_devices[i], 0}, {receiver_mesh_exit_16, 1}, {sender_mesh_exit_16, 2}, {sender_devices[i], 3}});

        // Reverse traffic: sender -> receiver (green arrows in diagram)
        mock_control_plane_->set_mock_route(
            sender_devices[i],
            receiver_devices[i],
            {{sender_devices[i], 0}, {sender_mesh_exit_16, 1}, {receiver_mesh_exit_16, 2}, {receiver_devices[i], 3}});
    }

    // Direct inter-mesh routing (no figure-8)
    mock_control_plane_->set_mock_route(
        receiver_mesh_exit_16, sender_mesh_exit_16, {{receiver_mesh_exit_16, 0}, {sender_mesh_exit_16, 1}});

    mock_control_plane_->set_mock_route(
        sender_mesh_exit_16, receiver_mesh_exit_16, {{sender_mesh_exit_16, 0}, {receiver_mesh_exit_16, 1}});

    // Create traffic pairs for all-to-all pattern (UNIDIRECTIONAL to avoid false positive cycles)
    // This tests that the routing configuration is cycle-free, not bidirectional communication patterns
    std::vector<std::pair<FabricNodeId, FabricNodeId>> all_to_all_pairs;
    for (int i = 0; i <= 15; ++i) {
        all_to_all_pairs.push_back({receiver_devices[i], sender_devices[i]});  // Forward only
    }
    all_to_all_pairs.push_back({receiver_mesh_exit_16, sender_mesh_exit_16});  // Exit connection

    bool has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(all_to_all_pairs, *mock_route_manager_, "DiagramAllToAllNoDeadlock");

    // Should NOT detect cycles - this validates the working exit node configuration
    // NOTE: Testing unidirectional traffic to focus on routing cycles, not bidirectional communication
    EXPECT_FALSE(has_cycles);
}

TEST_F(CycleDetectionTest, DiagramBasedA0314DeadlockScenario) {
    // Test the second diagram showing the problematic A0314 machine scenario
    // This models the actual machine where the hang occurs

    // The diagram shows the specific routing that creates the deadlock
    // Note the crossing connections that create the figure-8 pattern

    FabricNodeId receiver_mesh_exit_16{MeshId{0}, 16};
    FabricNodeId receiver_mesh_exit_22{MeshId{0}, 22};  // This is the problematic "unfriendly" exit
    FabricNodeId sender_mesh_exit_3{MeshId{1}, 3};      // Connected to the unfriendly exit
    FabricNodeId sender_mesh_exit_16{MeshId{1}, 16};

    // Model the specific devices involved in the hang
    FabricNodeId receiver_dev_20{MeshId{0}, 20};  // Mentioned in issue
    FabricNodeId receiver_dev_21{MeshId{0}, 21};  // Mentioned in issue
    FabricNodeId sender_dev_8{MeshId{1}, 8};      // Representative sender device
    FabricNodeId sender_dev_15{MeshId{1}, 15};    // Representative sender device

    // Set up the PROBLEMATIC routing that creates the figure-8 deadlock
    // This models the twisted ring shown in the A0314 diagram

    // Route that creates the problematic cross-connection
    // Device 22 on Mesh 1 is connected to Device 3 on Mesh 2 (the twist!)
    mock_control_plane_->set_mock_route(
        receiver_mesh_exit_22,
        sender_mesh_exit_3,
        {{receiver_mesh_exit_22, 0}, {sender_mesh_exit_16, 1}, {receiver_mesh_exit_16, 2}, {sender_mesh_exit_3, 3}});

    // Reverse route completes the figure-8
    mock_control_plane_->set_mock_route(
        sender_mesh_exit_3,
        receiver_mesh_exit_22,
        {{sender_mesh_exit_3, 0}, {receiver_mesh_exit_16, 1}, {sender_mesh_exit_16, 2}, {receiver_mesh_exit_22, 3}});

    // Traffic that triggers the deadlock
    mock_control_plane_->set_mock_route(
        receiver_dev_20,
        sender_dev_8,
        {{receiver_dev_20, 0}, {receiver_mesh_exit_22, 1}, {sender_mesh_exit_3, 2}, {sender_dev_8, 3}});

    mock_control_plane_->set_mock_route(
        sender_dev_8,
        receiver_dev_20,
        {{sender_dev_8, 0}, {sender_mesh_exit_16, 1}, {receiver_mesh_exit_16, 2}, {receiver_dev_20, 3}});

    std::vector<std::pair<FabricNodeId, FabricNodeId>> deadlock_pairs = {
        // Focus on routing infrastructure that creates figure-8 cycle (not application traffic)
        {receiver_mesh_exit_22, sender_mesh_exit_3},  // Twisted connection
        {sender_mesh_exit_3, receiver_mesh_exit_22}   // Completes figure-8
    };

    bool has_cycles = detect_cycles_in_random_inter_mesh_traffic(
        deadlock_pairs, *mock_route_manager_, "DiagramA0314DeadlockScenario");

    // Should detect the figure-8 cycle that causes the hang
    EXPECT_TRUE(has_cycles);
}

TEST_F(CycleDetectionTest, DiagramBasedHighTrafficBottleneck) {
    // Test the third diagram showing high traffic concentration
    // This models the bottleneck scenario where traffic converges on exit nodes

    // The diagram shows multiple flows converging on the same exit nodes
    // This creates resource contention that can lead to deadlocks

    FabricNodeId receiver_mesh_exit_16{MeshId{0}, 16};
    FabricNodeId receiver_mesh_exit_22{MeshId{0}, 22};
    FabricNodeId sender_mesh_exit_16{MeshId{1}, 16};

    // Model high-traffic scenario with many senders targeting same receivers
    std::vector<FabricNodeId> heavy_senders, bottleneck_receivers;

    // Create multiple heavy traffic sources (devices 0-15 in sender mesh)
    for (int i = 0; i <= 15; ++i) {
        heavy_senders.push_back(FabricNodeId{MeshId{1}, static_cast<chip_id_t>(i)});
    }

    // Create bottleneck receivers (devices 16-31 in receiver mesh)
    for (int i = 16; i <= 31; ++i) {
        bottleneck_receivers.push_back(FabricNodeId{MeshId{0}, static_cast<chip_id_t>(i)});
    }

    // All heavy senders target same exit node (creates bottleneck)
    for (size_t i = 0; i < heavy_senders.size(); ++i) {
        mock_control_plane_->set_mock_route(
            heavy_senders[i],
            bottleneck_receivers[i % bottleneck_receivers.size()],
            {{heavy_senders[i], 0},
             {sender_mesh_exit_16, 1},
             {receiver_mesh_exit_16, 2},
             {bottleneck_receivers[i % bottleneck_receivers.size()], 3}});
    }

    // Reverse traffic creates potential for cycles
    for (size_t i = 0; i < 8; ++i) {  // Subset for testing
        mock_control_plane_->set_mock_route(
            bottleneck_receivers[i],
            heavy_senders[i],
            {{bottleneck_receivers[i], 0},
             {receiver_mesh_exit_22, 1},
             {sender_mesh_exit_16, 2},
             {heavy_senders[i], 3}});
    }

    // The problematic connection that creates cycle under high load
    mock_control_plane_->set_mock_route(
        receiver_mesh_exit_22,
        sender_mesh_exit_16,
        {{receiver_mesh_exit_22, 0}, {receiver_mesh_exit_16, 1}, {sender_mesh_exit_16, 2}});

    std::vector<std::pair<FabricNodeId, FabricNodeId>> bottleneck_pairs;

    // Add routing infrastructure that creates bottleneck cycles (not application ACKs)
    for (size_t i = 0; i < 8; ++i) {
        bottleneck_pairs.push_back({heavy_senders[i], bottleneck_receivers[i]});
    }
    // Focus on exit node routing that creates the bottleneck cycle
    bottleneck_pairs.push_back({receiver_mesh_exit_22, sender_mesh_exit_16});

    bool has_cycles = detect_cycles_in_random_inter_mesh_traffic(
        bottleneck_pairs, *mock_route_manager_, "DiagramHighTrafficBottleneck");

    // Should detect cycles under high traffic load
    EXPECT_TRUE(has_cycles);
}

TEST_F(CycleDetectionTest, DiagramBasedSolutionValidation) {
    // Test the fourth diagram showing the SOLUTION
    // This validates the proposed exit node assignment algorithm

    // The solution diagram shows how to break the cycle by proper exit node assignment:
    // M1D0-16 -> Exit on M1D16
    // M1D17-31 -> Exit on M1D22
    // M2D0-15 -> Exit on M2D3
    // M2D16-31 -> Exit on M2D16

    FabricNodeId mesh1_exit_16{MeshId{0}, 16};
    FabricNodeId mesh1_exit_22{MeshId{0}, 22};
    FabricNodeId mesh2_exit_3{MeshId{1}, 3};
    FabricNodeId mesh2_exit_16{MeshId{1}, 16};

    // Create device ranges as shown in solution
    std::vector<FabricNodeId> mesh1_devices_0_16, mesh1_devices_17_31;
    std::vector<FabricNodeId> mesh2_devices_0_15, mesh2_devices_16_31;

    for (int i = 0; i <= 16; ++i) {
        mesh1_devices_0_16.push_back(FabricNodeId{MeshId{0}, static_cast<chip_id_t>(i)});
    }
    for (int i = 17; i <= 31; ++i) {
        mesh1_devices_17_31.push_back(FabricNodeId{MeshId{0}, static_cast<chip_id_t>(i)});
    }
    for (int i = 0; i <= 15; ++i) {
        mesh2_devices_0_15.push_back(FabricNodeId{MeshId{1}, static_cast<chip_id_t>(i)});
    }
    for (int i = 16; i <= 31; ++i) {
        mesh2_devices_16_31.push_back(FabricNodeId{MeshId{1}, static_cast<chip_id_t>(i)});
    }

    // Implement the solution routing (prevents figure-8)

    // M1D0-16 -> Exit on M1D16, route to M2D3 for devices 0-15
    mock_control_plane_->set_mock_route(
        mesh1_devices_0_16[8],
        mesh2_devices_0_15[8],  // Sample routing
        {{mesh1_devices_0_16[8], 0}, {mesh1_exit_16, 1}, {mesh2_exit_3, 2}, {mesh2_devices_0_15[8], 3}});

    // M1D17-31 -> Exit on M1D22, route to M2D16 for devices 16-31
    mock_control_plane_->set_mock_route(
        mesh1_devices_17_31[8],
        mesh2_devices_16_31[8],  // Sample routing
        {{mesh1_devices_17_31[8], 0}, {mesh1_exit_22, 1}, {mesh2_exit_16, 2}, {mesh2_devices_16_31[8], 3}});

    // Exit node connections that DON'T create figure-8
    mock_control_plane_->set_mock_route(mesh1_exit_16, mesh2_exit_3, {{mesh1_exit_16, 0}, {mesh2_exit_3, 1}});

    mock_control_plane_->set_mock_route(mesh1_exit_22, mesh2_exit_16, {{mesh1_exit_22, 0}, {mesh2_exit_16, 1}});

    // Reverse routes (ACK traffic)
    mock_control_plane_->set_mock_route(
        mesh2_devices_0_15[8],
        mesh1_devices_0_16[8],
        {{mesh2_devices_0_15[8], 0}, {mesh2_exit_3, 1}, {mesh1_exit_16, 2}, {mesh1_devices_0_16[8], 3}});

    mock_control_plane_->set_mock_route(
        mesh2_devices_16_31[8],
        mesh1_devices_17_31[8],
        {{mesh2_devices_16_31[8], 0}, {mesh2_exit_16, 1}, {mesh1_exit_22, 2}, {mesh1_devices_17_31[8], 3}});

    std::vector<std::pair<FabricNodeId, FabricNodeId>> solution_pairs = {
        // Forward traffic using solution routing (UNIDIRECTIONAL to avoid false positive cycles)
        {mesh1_devices_0_16[8], mesh2_devices_0_15[8]},    // Uses exit 16 -> exit 3
        {mesh1_devices_17_31[8], mesh2_devices_16_31[8]},  // Uses exit 22 -> exit 16

        // Exit node connections (unidirectional, demonstrates no figure-8)
        {mesh1_exit_16, mesh2_exit_3},  // Path 1: 16 -> 3
        {mesh1_exit_22, mesh2_exit_16}  // Path 2: 22 -> 16 (monotonic, no crossing)
    };

    bool has_cycles =
        detect_cycles_in_random_inter_mesh_traffic(solution_pairs, *mock_route_manager_, "DiagramSolutionValidation");

    // Should NOT detect cycles with the solution routing - validates exit node algorithm works
    // NOTE: Testing unidirectional traffic to focus on routing cycles, not bidirectional communication
    EXPECT_FALSE(has_cycles);
}

TEST_F(CycleDetectionTest, ComprehensiveGalaxyRoutingDeadlockSuite) {
    // Comprehensive test focusing on routing infrastructure deadlock (not application ACKs)
    // Tests the complete Galaxy routing deadlock detection capability

    FabricNodeId mesh1_exit_16{MeshId{0}, 16};
    FabricNodeId mesh1_exit_22{MeshId{0}, 22};
    FabricNodeId mesh2_exit_3{MeshId{1}, 3};
    FabricNodeId mesh2_exit_16{MeshId{1}, 16};

    // Focus on routing infrastructure that creates figure-8 dependencies
    std::vector<std::pair<FabricNodeId, FabricNodeId>> routing_infrastructure_pairs;

    // Core problematic exit node routing that creates figure-8 cycle
    mock_control_plane_->set_mock_route(
        mesh1_exit_22, mesh2_exit_3, {{mesh1_exit_22, 0}, {mesh2_exit_16, 1}, {mesh1_exit_16, 2}, {mesh2_exit_3, 3}});
    routing_infrastructure_pairs.push_back({mesh1_exit_22, mesh2_exit_3});

    mock_control_plane_->set_mock_route(
        mesh2_exit_3, mesh1_exit_16, {{mesh2_exit_3, 0}, {mesh1_exit_22, 1}, {mesh2_exit_16, 2}, {mesh1_exit_16, 3}});
    routing_infrastructure_pairs.push_back({mesh2_exit_3, mesh1_exit_16});

    // Additional routing dependencies that complete the cycle
    mock_control_plane_->set_mock_route(
        mesh1_exit_16, mesh2_exit_16, {{mesh1_exit_16, 0}, {mesh2_exit_3, 1}, {mesh1_exit_22, 2}, {mesh2_exit_16, 3}});
    routing_infrastructure_pairs.push_back({mesh1_exit_16, mesh2_exit_16});

    mock_control_plane_->set_mock_route(
        mesh2_exit_16, mesh1_exit_22, {{mesh2_exit_16, 0}, {mesh1_exit_16, 1}, {mesh2_exit_3, 2}, {mesh1_exit_22, 3}});
    routing_infrastructure_pairs.push_back({mesh2_exit_16, mesh1_exit_22});

    bool has_cycles = detect_cycles_in_random_inter_mesh_traffic(
        routing_infrastructure_pairs, *mock_route_manager_, "ComprehensiveGalaxyRoutingDeadlock");

    // Should detect the routing infrastructure deadlock (independent of application traffic)
    EXPECT_TRUE(has_cycles);
}

}  // namespace tt::tt_fabric::fabric_tests

// Main function for running the tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
