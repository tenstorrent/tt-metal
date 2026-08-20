// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the offline FSD -> PSD builder (physical_descriptor_builder).
// Ported/adapted from tenstorrent/tt-fabric-manager tests/unit/topology_mapper_test.cpp (FSD cases).

#include <cstdint>
#include <map>
#include <string>

#include <gtest/gtest.h>

#include <tt-metalium/experimental/fabric/physical_descriptor_builder.hpp>
#include "protobuf/factory_system_descriptor.pb.h"
#include "protobuf/physical_system_descriptor.pb.h"

namespace tt::scaleout_tools {
namespace {

using FSD = ::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor;

// Two hosts (hostA, hostB), one tray each, board type N300. One intra-host link on hostA (asic0<->asic1) and
// one cross-host link (hostA asic1 <-> hostB asic0).
FSD make_two_host_fsd() {
    FSD fsd;

    auto* h0 = fsd.add_hosts();
    h0->set_hostname("hostA");
    h0->set_motherboard("moboA");
    auto* h1 = fsd.add_hosts();
    h1->set_hostname("hostB");
    h1->set_motherboard("moboB");

    auto* bt = fsd.mutable_board_types();
    for (uint32_t host_id = 0; host_id < 2; ++host_id) {
        auto* loc = bt->add_board_locations();
        loc->set_host_id(host_id);
        loc->set_tray_id(0);
        loc->set_board_type("N300");
    }

    auto add_conn = [&](uint32_t ha, uint32_t aa, uint32_t ca, uint32_t hb, uint32_t ab, uint32_t cb) {
        auto* c = fsd.mutable_eth_connections()->add_connection();
        auto* a = c->mutable_endpoint_a();
        a->set_host_id(ha);
        a->set_tray_id(0);
        a->set_asic_location(aa);
        a->set_chan_id(ca);
        auto* b = c->mutable_endpoint_b();
        b->set_host_id(hb);
        b->set_tray_id(0);
        b->set_asic_location(ab);
        b->set_chan_id(cb);
    };
    add_conn(0, 0, 0, 0, 1, 0);  // intra-host on hostA
    add_conn(0, 1, 1, 1, 0, 1);  // cross-host hostA<->hostB
    return fsd;
}

TEST(PhysicalDescriptorBuilder, EnumeratesAsicsAndRanks) {
    auto psd = build_physical_descriptor(make_two_host_fsd());

    // 3 ASICs from the union of endpoints: (0,0,0), (0,0,1), (1,0,0).
    EXPECT_EQ(psd.asic_descriptors_size(), 3);

    // host_to_rank: hostA -> 0, hostB -> 1.
    std::map<std::string, uint32_t> rank;
    for (const auto& r : psd.host_to_rank()) {
        rank[r.host_name()] = r.rank();
    }
    EXPECT_EQ(rank["hostA"], 0u);
    EXPECT_EQ(rank["hostB"], 1u);

    // Every ASIC descriptor carries a non-zero synthesized unique_id and a board type.
    for (const auto& m : psd.asic_descriptors()) {
        EXPECT_NE(m.asic_id(), 0u);
        EXPECT_EQ(m.asic_descriptor().unique_id(), m.asic_id());
        EXPECT_FALSE(m.asic_descriptor().host_name().empty());
    }
}

TEST(PhysicalDescriptorBuilder, EdgesAreBidirectionalAndLocalFlagged) {
    auto psd = build_physical_descriptor(make_two_host_fsd());
    ASSERT_TRUE(psd.has_system_graph());

    // Count edges and how many are marked local, per host, across the asic_connectivity_graph.
    int total_eth = 0, local_eth = 0;
    for (const auto& host : psd.system_graph().asic_connectivity_graph()) {
        for (const auto& asic : host.asic_topologies()) {
            for (const auto& edge : asic.topology().asic_connections()) {
                for (const auto& eth : edge.eth_connections()) {
                    ++total_eth;
                    if (eth.is_local()) {
                        ++local_eth;
                    }
                }
            }
        }
    }
    // Each connection is emitted in both directions => 2 connections * 2 directions = 4 directed eth entries.
    EXPECT_EQ(total_eth, 4);
    // The intra-host connection (both directions) is local; the cross-host one is not.
    EXPECT_EQ(local_eth, 2);

    // The cross-host link populates the exit_node_connection_table for both hosts.
    std::map<std::string, int> exit_counts;
    for (const auto& table : psd.exit_node_connection_table()) {
        exit_counts[table.host_name()] = table.exit_connections_size();
    }
    EXPECT_EQ(exit_counts["hostA"], 1);
    EXPECT_EQ(exit_counts["hostB"], 1);
}

TEST(PhysicalDescriptorBuilder, ConnectedComponentsSplit) {
    // Connected (cross-host link present) => one PSD.
    EXPECT_EQ(build_physical_descriptors(make_two_host_fsd()).size(), 1u);

    // Remove the cross-host link => two independent single-host components => two PSDs.
    FSD fsd = make_two_host_fsd();
    fsd.mutable_eth_connections()->mutable_connection()->RemoveLast();  // drop the cross-host connection
    // hostB now has no eth_connections; it still forms its own group.
    auto psds = build_physical_descriptors(fsd);
    EXPECT_EQ(psds.size(), 2u);
}

TEST(PhysicalDescriptorBuilder, ThrowsOnOutOfRangeHostId) {
    FSD fsd = make_two_host_fsd();
    // Point an endpoint at a non-existent host_id.
    fsd.mutable_eth_connections()->mutable_connection(0)->mutable_endpoint_b()->set_host_id(99);
    EXPECT_THROW(build_physical_descriptor(fsd), std::runtime_error);
}

TEST(PhysicalDescriptorBuilder, AcceptsUbbWormholeBoardTypeAlias) {
    // UBB_WORMHOLE is a compile-time alias of UBB (same enum value). It must be accepted (and its lowercase
    // form) for backward compatibility with Fabric Manager FSDs, mapping to the same value as "UBB".
    auto first_asic_board_type = [](const std::string& board_type_str) {
        FSD fsd = make_two_host_fsd();
        for (auto& loc : *fsd.mutable_board_types()->mutable_board_locations()) {
            loc.set_board_type(board_type_str);
        }
        auto psd = build_physical_descriptor(fsd);
        return psd.asic_descriptors(0).asic_descriptor().board_type();
    };

    uint32_t ubb = 0;
    EXPECT_NO_THROW(ubb = first_asic_board_type("UBB"));
    EXPECT_EQ(first_asic_board_type("UBB_WORMHOLE"), ubb);
    EXPECT_EQ(first_asic_board_type("ubb_wormhole"), ubb);
}

TEST(PhysicalDescriptorBuilder, LoadThrowsOnMissingFile) {
    EXPECT_THROW(load_factory_descriptor("/nonexistent/path/to/fsd.textproto"), std::runtime_error);
}

}  // namespace
}  // namespace tt::scaleout_tools
