// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the offline FSD -> PSD builder (physical_descriptor_builder).
// Ported/adapted from tenstorrent/tt-fabric-manager tests/unit/topology_mapper_test.cpp (FSD cases).

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <google/protobuf/text_format.h>

#include <tt-metalium/experimental/fabric/physical_descriptor_builder.hpp>
#include "protobuf/factory_system_descriptor.pb.h"
#include "protobuf/physical_system_descriptor.pb.h"

namespace tt::tt_metal::experimental::tt_fabric {
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

TEST(PhysicalDescriptorBuilder, ThrowsOnDuplicateHostname) {
    // The PSD keys ranks / motherboards / graph entries by hostname, so duplicate hostnames would silently
    // collapse hosts together. The builder must reject them.
    FSD fsd = make_two_host_fsd();
    fsd.mutable_hosts(1)->set_hostname(fsd.hosts(0).hostname());  // make hostB a duplicate of hostA
    EXPECT_THROW(build_physical_descriptor(fsd), std::runtime_error);
}

TEST(PhysicalDescriptorBuilder, LoadThrowsOnMissingFile) {
    EXPECT_THROW(load_factory_descriptor("/nonexistent/path/to/fsd.textproto"), std::runtime_error);
}

// Exercises the packaged consumer path: parse an FSD textproto file -> proto conversion -> C++
// PhysicalSystemDescriptor.
TEST(PhysicalDescriptorBuilder, BuildFromFileReturnsCppDescriptor) {
    const FSD fsd = make_two_host_fsd();
    std::string text;
    ASSERT_TRUE(google::protobuf::TextFormat::PrintToString(fsd, &text));
    const auto path = std::filesystem::temp_directory_path() / "pdb_build_from_file_test.textproto";
    {
        std::ofstream out(path);
        out << text;
    }

    auto psd = build_physical_descriptor_from_file(path.string());
    EXPECT_EQ(psd.get_all_hostnames().size(), 2u);     // hostA + hostB, both wired via the cross-host link
    EXPECT_EQ(psd.get_asic_descriptors().size(), 3u);  // (0,0,0), (0,0,1), (1,0,0)
    EXPECT_EQ(psd.get_rank_for_hostname("hostA"), 0u);
    EXPECT_EQ(psd.get_rank_for_hostname("hostB"), 1u);

    std::filesystem::remove(path);
}

TEST(PhysicalDescriptorBuilder, FilterFactoryDescriptorRestrictsHosts) {
    // Carve hostA out of the two-host FSD. Its intra-host link survives; the cross-host link (to hostB) is dropped.
    const FSD filtered = filter_factory_descriptor(make_two_host_fsd(), {"hostA"});
    ASSERT_EQ(filtered.hosts_size(), 1);
    EXPECT_EQ(filtered.hosts(0).hostname(), "hostA");

    auto psd = build_physical_descriptor(filtered);
    EXPECT_EQ(psd.host_to_rank_size(), 1);
    EXPECT_EQ(psd.asic_descriptors_size(), 2);  // hostA's (0,0,0) and (0,0,1); hostB's ASIC is gone

    // Empty filter is a no-op (full copy).
    EXPECT_EQ(filter_factory_descriptor(make_two_host_fsd(), {}).hosts_size(), 2);
}

TEST(PhysicalDescriptorBuilder, FilterFactoryDescriptorThrowsOnUnknownHost) {
    EXPECT_THROW(filter_factory_descriptor(make_two_host_fsd(), {"does-not-exist"}), std::runtime_error);
}

TEST(PhysicalDescriptorBuilder, ThrowsOnLocalLinkWithInvalidHostId) {
    // A connection whose endpoints share an out-of-range host_id (99 -> 99) must be reported, not silently
    // treated as a same-host link and dropped, when partitioning for build_physical_descriptors().
    FSD fsd = make_two_host_fsd();
    auto* c = fsd.mutable_eth_connections()->add_connection();
    c->mutable_endpoint_a()->set_host_id(99);
    c->mutable_endpoint_b()->set_host_id(99);
    EXPECT_THROW(build_physical_descriptors(fsd), std::runtime_error);
}

TEST(PhysicalDescriptorBuilder, SingleHostEmitsHostConnectivityEntry) {
    // A single-host descriptor (no inter-host links) must still emit a host_connectivity_graph entry, matching
    // the runtime discovery path, so PhysicalSystemDescriptor::get_host_neighbors() returns an empty list
    // instead of failing its contains() check.
    FSD fsd;
    auto* h = fsd.add_hosts();
    h->set_hostname("solo");
    h->set_motherboard("mobo");
    auto* loc = fsd.mutable_board_types()->add_board_locations();
    loc->set_host_id(0);
    loc->set_tray_id(0);
    loc->set_board_type("N300");
    auto* c = fsd.mutable_eth_connections()->add_connection();  // one intra-host link so the host has ASICs
    c->mutable_endpoint_a()->set_host_id(0);
    c->mutable_endpoint_a()->set_tray_id(0);
    c->mutable_endpoint_a()->set_asic_location(0);
    c->mutable_endpoint_b()->set_host_id(0);
    c->mutable_endpoint_b()->set_tray_id(0);
    c->mutable_endpoint_b()->set_asic_location(1);

    const auto psd_proto = build_physical_descriptor(fsd);
    ASSERT_TRUE(psd_proto.has_system_graph());
    bool found = false;
    for (const auto& hc : psd_proto.system_graph().host_connectivity_graph()) {
        if (hc.src_host_name() == "solo") {
            found = true;
            EXPECT_EQ(hc.host_connections_size(), 0);
        }
    }
    EXPECT_TRUE(found) << "no host_connectivity_graph entry for the single host";

    // The C++ wrapper returns an empty neighbor list rather than asserting.
    ::tt::tt_metal::PhysicalSystemDescriptor psd(psd_proto);
    EXPECT_TRUE(psd.get_host_neighbors("solo").empty());
}

// Path to a small real FSD already checked into the repo (4-host quietbox, ~12 KB) so we don't add a large fixture.
std::string quietbox_fsd_path() {
    const char* home = std::getenv("TT_METAL_HOME");
    return std::string(home != nullptr ? home : ".") +
           "/tests/scale_out/4x_bh_quietbox/factory_system_descriptors/"
           "factory_system_descriptor_4x_bh_quietbox.textproto";
}

// Integration: build a PSD from a real (small, in-repo) FSD end-to-end and sanity-check hosts, ASIC nodes,
// host<->rank, and connectivity.
TEST(PhysicalDescriptorBuilder, IntegrationBuildFromQuietboxFsd) {
    ASSERT_NE(std::getenv("TT_METAL_HOME"), nullptr) << "TT_METAL_HOME must be set";
    const std::string fsd_path = quietbox_fsd_path();
    ASSERT_TRUE(std::filesystem::exists(fsd_path)) << "fixture missing: " << fsd_path;

    // FSD -> PSD via the proto-free entry point (returns the C++ PhysicalSystemDescriptor).
    auto psd = build_physical_descriptor_from_file(fsd_path);

    const auto hostnames = psd.get_all_hostnames();
    EXPECT_EQ(hostnames.size(), 4u);  // 4-host quietbox

    const auto& asics = psd.get_asic_descriptors();
    EXPECT_GT(asics.size(), 0u);

    std::cout << "[quietbox] hosts=" << hostnames.size() << ", total ASIC nodes=" << asics.size() << "\n";
    for (const auto& host : hostnames) {
        std::cout << "[quietbox]   " << host << "  asics=" << psd.get_asics_connected_to_host(host).size() << "\n";
    }

    // Host <-> rank is a bijection, and every host owns at least one ASIC.
    std::set<uint32_t> ranks;
    for (const auto& host : hostnames) {
        const uint32_t rank = psd.get_rank_for_hostname(host);
        EXPECT_EQ(psd.get_hostname_for_rank(rank), host);
        ranks.insert(rank);
        EXPECT_GT(psd.get_asics_connected_to_host(host).size(), 0u) << "host: " << host;
    }
    EXPECT_EQ(ranks.size(), hostnames.size());

    // Connectivity: every ASIC is wired into the topology.
    size_t asics_with_neighbors = 0;
    for (const auto& [asic_id, _] : asics) {
        if (!psd.get_asic_neighbors(asic_id).empty()) {
            ++asics_with_neighbors;
        }
    }
    std::cout << "[quietbox] ASICs with >=1 neighbor: " << asics_with_neighbors << "/" << asics.size() << "\n";
    EXPECT_EQ(asics_with_neighbors, asics.size());
}

// Host filter carves a subset of hosts out of an FSD before building.
TEST(PhysicalDescriptorBuilder, IntegrationHostFilterRestrictsToSubset) {
    ASSERT_NE(std::getenv("TT_METAL_HOME"), nullptr) << "TT_METAL_HOME must be set";
    const std::string fsd_path = quietbox_fsd_path();
    ASSERT_TRUE(std::filesystem::exists(fsd_path)) << "fixture missing: " << fsd_path;

    // No filter == whole FSD.
    EXPECT_EQ(build_physical_descriptor_from_file(fsd_path).get_all_hostnames().size(), 4u);

    // Filter to a strict subset: result hosts are a non-empty subset of the requested set.
    const std::vector<std::string> subset = {"sjc1-tt-qb-01", "sjc1-tt-qb-02"};
    auto psd = build_physical_descriptor_from_file(fsd_path, subset);
    const auto hostnames = psd.get_all_hostnames();
    EXPECT_GT(hostnames.size(), 0u);
    EXPECT_LE(hostnames.size(), subset.size());
    const std::set<std::string> allowed(subset.begin(), subset.end());
    for (const auto& host : hostnames) {
        EXPECT_TRUE(allowed.contains(host)) << "unexpected host after filter: " << host;
    }
}

}  // namespace
}  // namespace tt::tt_metal::experimental::tt_fabric
