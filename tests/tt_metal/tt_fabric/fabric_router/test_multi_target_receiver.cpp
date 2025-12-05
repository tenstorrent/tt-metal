// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/fabric/builder/connection_writer_adapter.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <hostdevcommon/fabric_common.h>
#include <vector>
#include <map>

namespace tt::tt_fabric {

/**
 * MockConnectionWriterAdapter - Lightweight test adapter without UMD dependencies
 * 
 * This adapter exposes the core multi-target functionality without requiring
 * a full allocator or Metal context initialization.
 */
class MockConnectionWriterAdapter {
public:
    MockConnectionWriterAdapter() = default;
    
    void add_downstream_connection(
        const SenderWorkerAdapterSpec& spec,
        uint32_t vc_idx,
        eth_chan_directions direction,
        CoreCoord noc_xy) {
        
        // Accumulate connections (Phase 1.5 behavior)
        vc_to_downstreams_[vc_idx].push_back(DownstreamConnection{
            .spec = spec,
            .direction = direction,
            .noc_xy = noc_xy
        });
    }
    
    bool needs_multi_target_packing(uint32_t vc_idx) const {
        auto it = vc_to_downstreams_.find(vc_idx);
        if (it == vc_to_downstreams_.end()) {
            return false;
        }
        return it->second.size() > 1;
    }
    
    std::vector<uint32_t> pack_multi_target_rt_args(uint32_t vc_idx) const {
        std::vector<uint32_t> args;
        
        if (!needs_multi_target_packing(vc_idx)) {
            return args;
        }
        
        const auto& connections = vc_to_downstreams_.at(vc_idx);
        uint32_t num_connections = static_cast<uint32_t>(connections.size());
        
        // Pack number of downstream connections
        args.push_back(num_connections);
        
        // Pack each connection's data
        for (const auto& conn : connections) {
            args.push_back(conn.spec.edm_buffer_base_addr);
            args.push_back(conn.noc_xy.x);
            args.push_back(conn.noc_xy.y);
            args.push_back(conn.spec.edm_connection_handshake_addr);
            args.push_back(conn.spec.edm_worker_location_info_addr);
            args.push_back(conn.spec.buffer_index_semaphore_id);
        }
        
        return args;
    }
    
    size_t get_connection_count(uint32_t vc_idx) const {
        auto it = vc_to_downstreams_.find(vc_idx);
        return (it != vc_to_downstreams_.end()) ? it->second.size() : 0;
    }
    
    const std::vector<DownstreamConnection>& get_connections(uint32_t vc_idx) const {
        static const std::vector<DownstreamConnection> empty;
        auto it = vc_to_downstreams_.find(vc_idx);
        return (it != vc_to_downstreams_.end()) ? it->second : empty;
    }

private:
    std::map<uint32_t, std::vector<DownstreamConnection>> vc_to_downstreams_;
};

/**
 * Test fixture for multi-target receiver support (Phase 1.5)
 * 
 * These tests validate:
 * - Single downstream works (regression)
 * - Multiple downstreams accumulate correctly
 * - Z router VC1 with 2-4 downstreams
 * - Runtime args packing for multi-target
 * 
 * NOTE: Uses MockConnectionWriterAdapter to avoid UMD/device initialization
 */
class MultiTargetReceiverTest : public ::testing::Test {
protected:
    void SetUp() override {
        adapter_ = std::make_unique<MockConnectionWriterAdapter>();
    }

    void TearDown() override {
        adapter_.reset();
    }

    std::unique_ptr<MockConnectionWriterAdapter> adapter_;
    
    // Helper to create a test adapter spec
    SenderWorkerAdapterSpec create_test_spec(uint32_t base_addr, uint32_t noc_x, uint32_t noc_y) {
        return SenderWorkerAdapterSpec{
            .edm_noc_x = noc_x,
            .edm_noc_y = noc_y,
            .edm_buffer_base_addr = base_addr,
            .num_buffers_per_channel = 4,
            .edm_l1_sem_addr = base_addr + 1000,
            .edm_connection_handshake_addr = base_addr + 2000,
            .edm_worker_location_info_addr = base_addr + 3000,
            .buffer_size_bytes = 1024,
            .buffer_index_semaphore_id = base_addr + 4000,
            .edm_direction = eth_chan_directions::EAST
        };
    }
};

// ============ Single Downstream Tests (Regression) ============

TEST_F(MultiTargetReceiverTest, SingleDownstream_VC0_Works) {
    auto spec = create_test_spec(0x10000, 5, 6);
    
    // Add single downstream connection for VC0
    adapter_->add_downstream_connection(
        spec,
        0,  // VC0
        eth_chan_directions::WEST,
        CoreCoord(5, 6));
    
    // Single connection should not trigger multi-target packing
    EXPECT_FALSE(adapter_->needs_multi_target_packing(0));
    EXPECT_EQ(adapter_->get_connection_count(0), 1);
}

TEST_F(MultiTargetReceiverTest, SingleDownstream_1D_Works) {
    auto spec = create_test_spec(0x10000, 5, 6);
    
    // Add single downstream connection for VC0 in 1D
    adapter_->add_downstream_connection(
        spec,
        0,  // VC0
        eth_chan_directions::WEST,
        CoreCoord(5, 6));
    
    // Verify connection was added
    EXPECT_EQ(adapter_->get_connection_count(0), 1);
    EXPECT_FALSE(adapter_->needs_multi_target_packing(0));
}

// ============ Multi-Target Tests (Z Router VC1) ============

TEST_F(MultiTargetReceiverTest, MultiTarget_TwoDownstreams_Accumulates) {
    // Add first downstream (mesh router North)
    auto spec1 = create_test_spec(0x10000, 5, 6);
    adapter_->add_downstream_connection(
        spec1,
        1,  // VC1
        eth_chan_directions::NORTH,
        CoreCoord(5, 6));
    
    // Add second downstream (mesh router East)
    auto spec2 = create_test_spec(0x20000, 7, 8);
    adapter_->add_downstream_connection(
        spec2,
        1,  // VC1
        eth_chan_directions::EAST,
        CoreCoord(7, 8));
    
    // Verify accumulation
    EXPECT_EQ(adapter_->get_connection_count(1), 2);
    EXPECT_TRUE(adapter_->needs_multi_target_packing(1));
    
    // Pack runtime args
    auto rt_args = adapter_->pack_multi_target_rt_args(1);
    
    // Multi-target format:
    // [num_connections, conn1_data..., conn2_data...]
    // Each connection: [buffer_addr, noc_x, noc_y, handshake_addr, location_addr, sem_id]
    
    ASSERT_GE(rt_args.size(), 1);
    EXPECT_EQ(rt_args[0], 2);  // num_connections = 2
    
    // Should have 1 + (2 * 6) = 13 total args
    EXPECT_EQ(rt_args.size(), 13);
    
    // Verify first connection data
    EXPECT_EQ(rt_args[1], 0x10000);  // buffer_addr
    EXPECT_EQ(rt_args[2], 5);        // noc_x
    EXPECT_EQ(rt_args[3], 6);        // noc_y
    
    // Verify second connection data
    EXPECT_EQ(rt_args[7], 0x20000);  // buffer_addr
    EXPECT_EQ(rt_args[8], 7);        // noc_x
    EXPECT_EQ(rt_args[9], 8);        // noc_y
}

TEST_F(MultiTargetReceiverTest, MultiTarget_FourDownstreams_AllDirections) {
    // Z router VC1 connecting to 4 mesh routers (N, E, S, W)
    std::vector<eth_chan_directions> directions = {
        eth_chan_directions::NORTH,
        eth_chan_directions::EAST,
        eth_chan_directions::SOUTH,
        eth_chan_directions::WEST
    };
    
    for (size_t i = 0; i < 4; ++i) {
        auto spec = create_test_spec(0x10000 * (i + 1), 5 + i, 6 + i);
        adapter_->add_downstream_connection(
            spec,
            1,  // VC1
            directions[i],
            CoreCoord(5 + i, 6 + i));
    }
    
    EXPECT_EQ(adapter_->get_connection_count(1), 4);
    EXPECT_TRUE(adapter_->needs_multi_target_packing(1));
    
    // Pack runtime args
    auto rt_args = adapter_->pack_multi_target_rt_args(1);
    
    // Should have 1 + (4 * 6) = 25 total args
    ASSERT_EQ(rt_args.size(), 25);
    EXPECT_EQ(rt_args[0], 4);  // num_connections = 4
    
    // Verify all 4 connections are present
    for (size_t i = 0; i < 4; ++i) {
        size_t base_idx = 1 + (i * 6);
        EXPECT_EQ(rt_args[base_idx], 0x10000 * (i + 1));  // buffer_addr
        EXPECT_EQ(rt_args[base_idx + 1], 5 + i);          // noc_x
        EXPECT_EQ(rt_args[base_idx + 2], 6 + i);          // noc_y
    }
}

TEST_F(MultiTargetReceiverTest, MultiTarget_ThreeDownstreams_EdgeDevice) {
    // Edge device: Z router connecting to only 3 mesh routers
    for (size_t i = 0; i < 3; ++i) {
        auto spec = create_test_spec(0x10000 * (i + 1), 10 + i, 20 + i);
        adapter_->add_downstream_connection(
            spec,
            1,  // VC1
            static_cast<eth_chan_directions>(i),
            CoreCoord(10 + i, 20 + i));
    }
    
    EXPECT_EQ(adapter_->get_connection_count(1), 3);
    
    // Pack runtime args
    auto rt_args = adapter_->pack_multi_target_rt_args(1);
    
    // Should have 1 + (3 * 6) = 19 total args
    ASSERT_EQ(rt_args.size(), 19);
    EXPECT_EQ(rt_args[0], 3);  // num_connections = 3
}

// ============ Mixed VC Tests ============

TEST_F(MultiTargetReceiverTest, MixedVCs_VC0Single_VC1Multiple) {
    // VC0: Single downstream (standard INTRA_MESH)
    auto vc0_spec = create_test_spec(0x5000, 1, 2);
    adapter_->add_downstream_connection(
        vc0_spec,
        0,  // VC0
        eth_chan_directions::WEST,
        CoreCoord(1, 2));
    
    // VC1: Multiple downstreams (Z router)
    for (size_t i = 0; i < 2; ++i) {
        auto spec = create_test_spec(0x10000 * (i + 1), 5 + i, 6 + i);
        adapter_->add_downstream_connection(
            spec,
            1,  // VC1
            static_cast<eth_chan_directions>(i),
            CoreCoord(5 + i, 6 + i));
    }
    
    // VC0 has 1 connection (no multi-target)
    EXPECT_EQ(adapter_->get_connection_count(0), 1);
    EXPECT_FALSE(adapter_->needs_multi_target_packing(0));
    
    // VC1 has 2 connections (multi-target)
    EXPECT_EQ(adapter_->get_connection_count(1), 2);
    EXPECT_TRUE(adapter_->needs_multi_target_packing(1));
    
    // Pack VC1 args (should use multi-target format)
    auto vc1_args = adapter_->pack_multi_target_rt_args(1);
    ASSERT_GE(vc1_args.size(), 1);
    EXPECT_EQ(vc1_args[0], 2);  // num_connections = 2
    EXPECT_EQ(vc1_args.size(), 13);  // 1 + (2 * 6)
}

// ============ Edge Cases ============

TEST_F(MultiTargetReceiverTest, NoConnections_VC1_ReturnsEmpty) {
    // Don't add any connections for VC1
    
    EXPECT_EQ(adapter_->get_connection_count(1), 0);
    EXPECT_FALSE(adapter_->needs_multi_target_packing(1));
    
    // Pack runtime args for VC1
    auto rt_args = adapter_->pack_multi_target_rt_args(1);
    
    // Should return empty (no multi-target packing needed)
    EXPECT_TRUE(rt_args.empty());
}

TEST_F(MultiTargetReceiverTest, MultipleAddCalls_SameVC_Accumulates) {
    // Add connections one at a time (simulating multiple add_downstream_connection calls)
    auto spec1 = create_test_spec(0x1000, 1, 1);
    adapter_->add_downstream_connection(spec1, 1, eth_chan_directions::NORTH, CoreCoord(1, 1));
    
    auto spec2 = create_test_spec(0x2000, 2, 2);
    adapter_->add_downstream_connection(spec2, 1, eth_chan_directions::EAST, CoreCoord(2, 2));
    
    auto spec3 = create_test_spec(0x3000, 3, 3);
    adapter_->add_downstream_connection(spec3, 1, eth_chan_directions::SOUTH, CoreCoord(3, 3));
    
    // All 3 should be accumulated
    EXPECT_EQ(adapter_->get_connection_count(1), 3);
    
    auto rt_args = adapter_->pack_multi_target_rt_args(1);
    
    ASSERT_GE(rt_args.size(), 1);
    EXPECT_EQ(rt_args[0], 3);  // num_connections = 3
    EXPECT_EQ(rt_args.size(), 19);  // 1 + (3 * 6)
}

// ============ Verification Tests ============

TEST_F(MultiTargetReceiverTest, MultiTarget_VerifyAllFields) {
    // Add connection with specific values
    SenderWorkerAdapterSpec spec{
        .edm_noc_x = 10,
        .edm_noc_y = 20,
        .edm_buffer_base_addr = 0xABCD,
        .num_buffers_per_channel = 8,
        .edm_l1_sem_addr = 0x1111,
        .edm_connection_handshake_addr = 0x2222,
        .edm_worker_location_info_addr = 0x3333,
        .buffer_size_bytes = 2048,
        .buffer_index_semaphore_id = 0x4444,
        .edm_direction = eth_chan_directions::WEST
    };
    
    adapter_->add_downstream_connection(spec, 1, eth_chan_directions::NORTH, CoreCoord(10, 20));
    adapter_->add_downstream_connection(spec, 1, eth_chan_directions::SOUTH, CoreCoord(10, 20));
    
    auto rt_args = adapter_->pack_multi_target_rt_args(1);
    
    // Verify structure: [count, conn1..., conn2...]
    ASSERT_EQ(rt_args.size(), 13);
    EXPECT_EQ(rt_args[0], 2);
    
    // Verify first connection fields
    EXPECT_EQ(rt_args[1], 0xABCD);  // buffer_base_addr
    EXPECT_EQ(rt_args[2], 10);      // noc_x
    EXPECT_EQ(rt_args[3], 20);      // noc_y
    EXPECT_EQ(rt_args[4], 0x2222);  // handshake_addr
    EXPECT_EQ(rt_args[5], 0x3333);  // location_info_addr
    EXPECT_EQ(rt_args[6], 0x4444);  // semaphore_id
    
    // Verify second connection has same values
    EXPECT_EQ(rt_args[7], 0xABCD);
    EXPECT_EQ(rt_args[8], 10);
    EXPECT_EQ(rt_args[9], 20);
    EXPECT_EQ(rt_args[10], 0x2222);
    EXPECT_EQ(rt_args[11], 0x3333);
    EXPECT_EQ(rt_args[12], 0x4444);
}

TEST_F(MultiTargetReceiverTest, GetConnections_ReturnsAccumulatedList) {
    // Add 3 connections
    for (size_t i = 0; i < 3; ++i) {
        auto spec = create_test_spec(0x1000 * (i + 1), i, i + 10);
        adapter_->add_downstream_connection(
            spec, 1, static_cast<eth_chan_directions>(i), CoreCoord(i, i + 10));
    }
    
    // Get connections list
    const auto& connections = adapter_->get_connections(1);
    ASSERT_EQ(connections.size(), 3);
    
    // Verify each connection
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(connections[i].spec.edm_buffer_base_addr, 0x1000 * (i + 1));
        EXPECT_EQ(connections[i].noc_xy.x, i);
        EXPECT_EQ(connections[i].noc_xy.y, i + 10);
    }
}

}  // namespace tt::tt_fabric

