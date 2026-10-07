// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The ControlPlane's link-health surface with no factory system descriptor configured, which is the state
// every caller is in today.
//
// This walks the whole surface deliberately. Each accessor forwards to a `LinkHealth` that does not exist
// without a factory descriptor, so a forwarder missing its null check is a segfault in the default
// configuration rather than an edge case, and only calling all of them finds it.

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <string>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/link_health.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>

#include "fabric_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

const ControlPlane& open_control_plane(FabricReliabilityMode mode) {
    auto& context = tt::tt_metal::MetalContext::instance();
    // set_default_fabric_topology rebuilds the control plane in STRICT. Disable first so that rebuild
    // does not run, then bring fabric up in the mode this test asked for.
    context.set_fabric_config(tt::tt_fabric::FabricConfig::DISABLED);
    context.set_default_fabric_topology();
    context.set_fabric_config(tt::tt_fabric::FabricConfig::FABRIC_2D, mode);
    context.initialize_fabric_config();
    return context.get_control_plane();
}

const ControlPlane& control_plane_without_factory_descriptor() {
    // The quad-host mesh graph is RELAXED, which cannot run under STRICT system health.
    return open_control_plane(tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
}

TEST_F(ControlPlaneFixture, NoFactoryDescriptorReportsNone) {
    const auto& control_plane = control_plane_without_factory_descriptor();

    EXPECT_FALSE(control_plane.has_factory_descriptor());
    EXPECT_EQ(control_plane.get_link_health(), nullptr);
    EXPECT_FALSE(control_plane.fsd_rerouting_active());
}

// Healthy, not unknown. With no descriptor saying what should be cabled, there is no expectation for a
// cable to be missing from, so callers gating on this must not start failing when the feature ships.
TEST_F(ControlPlaneFixture, NoFactoryDescriptorReportsEveryLinkHealthy) {
    const auto& control_plane = control_plane_without_factory_descriptor();

    for (const auto& mesh_id : control_plane.get_mesh_graph().get_mesh_ids()) {
        for (const auto& [_, chip_id] : control_plane.get_mesh_graph().get_chip_ids(mesh_id)) {
            const FabricNodeId fabric_node_id(mesh_id, chip_id);
            for (chan_id_t chan = 0; chan < 16; ++chan) {
                EXPECT_TRUE(control_plane.is_link_healthy(fabric_node_id, chan))
                    << fabric_node_id << " chan " << static_cast<int>(chan);
            }
        }
    }
}

TEST_F(ControlPlaneFixture, NoFactoryDescriptorReportsNoDownedLinks) {
    const auto& control_plane = control_plane_without_factory_descriptor();

    EXPECT_TRUE(control_plane.get_downed_links().empty());
    EXPECT_TRUE(control_plane.get_locally_unhealthy_links().empty());
}

// The empty vectors must be the same object every call, since they are returned by reference and a
// per-call temporary would leave callers holding a dangling reference.
TEST_F(ControlPlaneFixture, NoFactoryDescriptorEmptyResultsAreStable) {
    const auto& control_plane = control_plane_without_factory_descriptor();

    EXPECT_EQ(&control_plane.get_downed_links(), &control_plane.get_downed_links());
    EXPECT_EQ(&control_plane.get_locally_unhealthy_links(), &control_plane.get_locally_unhealthy_links());
}

TEST_F(ControlPlaneFixture, RefreshWithoutAFactoryDescriptorIsANoOp) {
    tt::tt_metal::MetalContext::instance().set_default_fabric_topology();
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    control_plane.refresh_connectivity_diff();
    control_plane.refresh_connectivity_diff();

    EXPECT_FALSE(control_plane.has_factory_descriptor());
    EXPECT_TRUE(control_plane.get_downed_links().empty());
}

// The factory-descriptor path. Gated on the environment because the descriptor path is read from
// RTOptions when MetalContext is first built, so it cannot be set from inside a test; the driver runs this
// with TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH and a mock cluster descriptor for one of its hosts.
class FactoryDescriptorControlPlaneFixture : public ControlPlaneFixture {
protected:
    void SetUp() override {
        if (getenv("TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH") == nullptr) {
            GTEST_SKIP() << "needs TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH";
        }
        ControlPlaneFixture::SetUp();
    }
};

// A descriptor that agrees with the hardware produces no downed links. This is the case that has to be
// silent: if ingesting a matching descriptor reported holes, the feature would be unusable on a healthy
// machine.
TEST_F(FactoryDescriptorControlPlaneFixture, AMatchingDescriptorReportsNoDownedLinks) {
    const auto& control_plane = control_plane_without_factory_descriptor();

    ASSERT_TRUE(control_plane.has_factory_descriptor());
    ASSERT_NE(control_plane.get_link_health(), nullptr);

    EXPECT_FALSE(control_plane.fsd_rerouting_active());
    EXPECT_TRUE(control_plane.get_downed_links().empty());
    EXPECT_TRUE(control_plane.get_locally_unhealthy_links().empty());

    // The comparison actually ran, rather than finding nothing to compare.
    EXPECT_GT(control_plane.get_link_health()->fsd_expected_count(), 0u);
}

// The mesh is placed on the factory topology, so every node still resolves to a real chip. Solving on a
// descriptor that carries no UMD ids and forgetting to resolve them would leave these all zero.
TEST_F(FactoryDescriptorControlPlaneFixture, EveryMappedNodeResolvesToARealChip) {
    const auto& control_plane = control_plane_without_factory_descriptor();
    ASSERT_TRUE(control_plane.has_factory_descriptor());

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto local_chips = cluster.user_exposed_chip_ids();
    ASSERT_FALSE(local_chips.empty());

    std::size_t resolved = 0;
    for (const auto& mesh_id : control_plane.get_mesh_graph().get_mesh_ids()) {
        for (const auto& [_, chip_id] : control_plane.get_mesh_graph().get_chip_ids(mesh_id)) {
            const FabricNodeId fabric_node_id(mesh_id, chip_id);
            const auto chip = control_plane.try_get_physical_chip_id_from_fabric_node_id(fabric_node_id);
            if (!chip.has_value()) {
                continue;
            }
            EXPECT_TRUE(local_chips.contains(*chip)) << fabric_node_id << " resolved to unknown chip " << *chip;
            ++resolved;
        }
    }
    EXPECT_EQ(resolved, local_chips.size());
}

// Refreshing against an unchanged live descriptor must not invent holes.
TEST_F(FactoryDescriptorControlPlaneFixture, RefreshAgainstUnchangedLiveIsStable) {
    tt::tt_metal::MetalContext::instance().set_default_fabric_topology();
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D, tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    ASSERT_TRUE(control_plane.has_factory_descriptor());

    const auto before = control_plane.get_link_health()->fsd_expected_count();
    control_plane.refresh_connectivity_diff();

    EXPECT_EQ(control_plane.get_link_health()->fsd_expected_count(), before);
    EXPECT_TRUE(control_plane.get_downed_links().empty());
    EXPECT_FALSE(control_plane.fsd_rerouting_active());
}

ChipId physical_chip_id(const ControlPlane& control_plane, tt::tt_metal::AsicID asic) {
    return control_plane.get_physical_system_descriptor().get_umd_unique_id(asic);
}

bool has_cable(
    const ControlPlane& control_plane,
    const std::vector<LinkInfo>& links,
    ChipId src_chip,
    chan_id_t src_chan,
    ChipId dst_chip,
    chan_id_t dst_chan) {
    for (const auto& link : links) {
        if (physical_chip_id(control_plane, link.src_asic) == src_chip && link.src_chan == src_chan &&
            physical_chip_id(control_plane, link.dst_asic) == dst_chip && link.dst_chan == dst_chan) {
            return true;
        }
    }
    return false;
}

const LinkInfo* find_cable(
    const ControlPlane& control_plane,
    const std::vector<LinkInfo>& links,
    ChipId src_chip,
    chan_id_t src_chan,
    ChipId dst_chip,
    chan_id_t dst_chan) {
    for (const auto& link : links) {
        if (physical_chip_id(control_plane, link.src_asic) == src_chip && link.src_chan == src_chan &&
            physical_chip_id(control_plane, link.dst_asic) == dst_chip && link.dst_chan == dst_chan) {
            return &link;
        }
    }
    return nullptr;
}

std::size_t intermesh_request(const MeshGraph& mesh_graph, MeshId src, MeshId dst) {
    const auto& table = mesh_graph.get_requested_intermesh_connections();
    auto lookup = [&](MeshId from, MeshId to) {
        const auto by_src = table.find(*from);
        if (by_src == table.end()) {
            return std::size_t{0};
        }
        const auto by_dst = by_src->second.find(*to);
        return by_dst == by_src->second.end() ? std::size_t{0} : by_dst->second;
    };
    return std::max(lookup(src, dst), lookup(dst, src));
}

// Channels pairing actually bound from one mesh toward the other. A rank only sees its own mesh.
std::size_t assigned_toward(const ControlPlane& control_plane, MeshId local, MeshId peer) {
    std::size_t count = 0;
    for (ChipId chip = 0; chip < 32; ++chip) {
        const FabricNodeId node{local, chip};
        if (!control_plane.try_get_physical_chip_id_from_fabric_node_id(node).has_value()) {
            continue;
        }
        for (const auto chan : control_plane.get_intermesh_facing_eth_chans(node)) {
            const auto peer_end = control_plane.try_get_connected_mesh_chip_chan_ids(node, chan);
            if (peer_end.has_value() && peer_end->first.mesh_id == peer) {
                ++count;
            }
        }
    }
    return count;
}

void expect_distinct_intermesh(const ControlPlane& control_plane, bool relaxed) {
    const auto& mesh_graph = control_plane.get_mesh_graph();
    EXPECT_TRUE(mesh_graph.is_inter_mesh_policy_specified());
    EXPECT_EQ(mesh_graph.is_inter_mesh_policy_relaxed(), relaxed);
    // The mesh graph stores each connection in both directions, so four boundaries are eight entries.
    std::set<std::size_t> counts;
    std::set<std::pair<std::uint32_t, std::uint32_t>> pairs;
    for (const auto& [src_mesh, dsts] : mesh_graph.get_requested_intermesh_connections()) {
        for (const auto& [dst_mesh, count] : dsts) {
            const auto lo = std::min(src_mesh, dst_mesh);
            const auto hi = std::max(src_mesh, dst_mesh);
            pairs.insert({lo, hi});
            counts.insert(count);
        }
    }
    EXPECT_EQ(pairs.size(), 4u);
    EXPECT_EQ(counts.size(), 4u);
}

std::size_t count_direction(const std::vector<LinkInfo>& links, const LinkInfo& sample) {
    std::size_t count = 0;
    for (const auto& link : links) {
        if (link.src_node == sample.src_node && link.src_direction == sample.src_direction) {
            ++count;
        }
    }
    return count;
}

// Each missing cable is checked in both directions, present in one set and absent from the other.
// Relaxed keeps the holes and finishes init. STRICT on the same pod allows an unused hole when the
// factory descriptor and the live descriptor both cover the requested count.
TEST_F(FactoryDescriptorControlPlaneFixture, ACableTheMeshGraphDoesNotUseIsUnused) {
    const auto& control_plane =
        open_control_plane(tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    ASSERT_TRUE(control_plane.has_factory_descriptor());
    const auto* link_health = control_plane.get_link_health();
    ASSERT_NE(link_health, nullptr);

    const auto& downed = link_health->get_downed_links();
    const auto& unused = link_health->get_unused_downed_links();

    // Subtorus connection. Chip 0 chan 0 -- chip 8 chan 0 is unused by the mesh graph.
    EXPECT_TRUE(has_cable(control_plane, unused, 0, 0, 8, 0));
    EXPECT_TRUE(has_cable(control_plane, unused, 8, 0, 0, 0));
    EXPECT_FALSE(has_cable(control_plane, downed, 0, 0, 8, 0));
    EXPECT_FALSE(has_cable(control_plane, downed, 8, 0, 0, 0));

    // Torus connection. Mesh graph 2, factory 4, live 3. The mesh graph is already covered, so
    // nothing is downed and the one missing factory cable is unused. Routing planes stay at 2.
    const auto* torus = find_cable(control_plane, unused, 15, 4, 23, 4);
    ASSERT_NE(torus, nullptr);
    EXPECT_TRUE(has_cable(control_plane, unused, 23, 4, 15, 4));
    EXPECT_FALSE(has_cable(control_plane, downed, 15, 4, 23, 4));
    EXPECT_FALSE(has_cable(control_plane, downed, 23, 4, 15, 4));
    EXPECT_EQ(count_direction(unused, *torus), 1u);
    EXPECT_EQ(count_direction(downed, *torus), 0u);

    // Mesh connection. Mesh graph 2, factory 2, live 1. The missing cable is still needed, so it is
    // downed (1 per direction) and unused stays empty on that direction. Routing planes stay at 2.
    const auto* mesh = find_cable(control_plane, downed, 0, 6, 4, 0);
    ASSERT_NE(mesh, nullptr);
    EXPECT_TRUE(has_cable(control_plane, downed, 4, 0, 0, 6));
    EXPECT_FALSE(has_cable(control_plane, unused, 0, 6, 4, 0));
    EXPECT_FALSE(has_cable(control_plane, unused, 4, 0, 0, 6));
    EXPECT_EQ(count_direction(downed, *mesh), 1u);
    EXPECT_EQ(count_direction(unused, *mesh), 0u);

    // Plane counts are filled for the meshes this rank owns. Chip ids repeat on every host, so a
    // local-chip check would fire on a rank that never computed these planes.
    const auto local_meshes = control_plane.get_local_mesh_id_bindings();
    const auto owns = [&](const FabricNodeId& node) {
        return std::find(local_meshes.begin(), local_meshes.end(), node.mesh_id) != local_meshes.end();
    };
    if (owns(torus->src_node)) {
        EXPECT_EQ(control_plane.get_num_usable_routing_planes(torus->src_node, torus->src_direction), 2u);
    }
    if (owns(mesh->src_node)) {
        EXPECT_EQ(control_plane.get_num_usable_routing_planes(mesh->src_node, mesh->src_direction), 2u);
    }

    // Intermesh, all RELAXED, a different count on each boundary. Mesh 2 to mesh 3 is missing
    // chip 5 chan 9 -- chip 29 chan 9 (16 factory cables, 15 live). This graph asks for 4, which
    // the live cables already cover, so that cable is unused and 4 channels are registered.
    expect_distinct_intermesh(control_plane, true);
    const auto* inter = find_cable(control_plane, unused, 5, 9, 29, 9);
    ASSERT_NE(inter, nullptr);
    const auto* inter_back = find_cable(control_plane, unused, 29, 9, 5, 9);
    ASSERT_NE(inter_back, nullptr);
    EXPECT_FALSE(has_cable(control_plane, downed, 5, 9, 29, 9));
    EXPECT_FALSE(has_cable(control_plane, downed, 29, 9, 5, 9));
    std::size_t intermesh_downed = 0;
    std::size_t intermesh_unused = 0;
    for (const auto& link : downed) {
        intermesh_downed += link.is_intermesh();
    }
    for (const auto& link : unused) {
        intermesh_unused += link.is_intermesh();
    }
    EXPECT_EQ(intermesh_downed, 0u);
    EXPECT_EQ(intermesh_unused, 2u);
    EXPECT_EQ(intermesh_request(control_plane.get_mesh_graph(), inter->src_mesh(), inter->dst_mesh()), 4u);
    if (owns(inter->src_node)) {
        EXPECT_EQ(assigned_toward(control_plane, inter->src_mesh(), inter->dst_mesh()), 4u);
    }
    if (owns(inter_back->src_node)) {
        EXPECT_EQ(assigned_toward(control_plane, inter_back->src_mesh(), inter_back->dst_mesh()), 4u);
    }
}

// Mesh graph 4, factory 2, live 1, on the mesh connection. Planes drop from 4 to the factory count
// of 2. The two channels the factory descriptor never had are that downgrade, not downed links.
// The one factory cable missing from the live descriptor stays downed. Nothing is unused.
TEST_F(FactoryDescriptorControlPlaneFixture, MeshGraphCountAboveTheFactoryDropsTheRoutingPlane) {
    const auto& control_plane =
        open_control_plane(tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    ASSERT_TRUE(control_plane.has_factory_descriptor());
    const auto* link_health = control_plane.get_link_health();
    ASSERT_NE(link_health, nullptr);

    const auto& downed = link_health->get_downed_links();
    const auto& unused = link_health->get_unused_downed_links();

    const auto* mesh = find_cable(control_plane, downed, 0, 6, 4, 0);
    ASSERT_NE(mesh, nullptr);
    const auto* mesh_back = find_cable(control_plane, downed, 4, 0, 0, 6);
    ASSERT_NE(mesh_back, nullptr);
    EXPECT_FALSE(has_cable(control_plane, unused, 0, 6, 4, 0));
    EXPECT_FALSE(has_cable(control_plane, unused, 4, 0, 0, 6));
    EXPECT_EQ(count_direction(downed, *mesh), 1u);
    EXPECT_EQ(count_direction(downed, *mesh_back), 1u);
    EXPECT_EQ(count_direction(unused, *mesh), 0u);
    EXPECT_EQ(count_direction(unused, *mesh_back), 0u);

    const auto local_meshes = control_plane.get_local_mesh_id_bindings();
    const auto owns = [&](const FabricNodeId& node) {
        return std::find(local_meshes.begin(), local_meshes.end(), node.mesh_id) != local_meshes.end();
    };
    if (owns(mesh->src_node)) {
        EXPECT_EQ(control_plane.get_num_usable_routing_planes(mesh->src_node, mesh->src_direction), 2u);
    }
    if (owns(mesh_back->src_node)) {
        EXPECT_EQ(control_plane.get_num_usable_routing_planes(mesh_back->src_node, mesh_back->src_direction), 2u);
    }

    // Intermesh, all RELAXED, a different count on each boundary. This graph asks for 16 on the
    // boundary missing chip 5 chan 9 -- chip 29 chan 9 (16 factory cables, 15 live). That one
    // cable stays downed, and 15 live cables are registered.
    expect_distinct_intermesh(control_plane, true);
    const auto* inter = find_cable(control_plane, downed, 5, 9, 29, 9);
    ASSERT_NE(inter, nullptr);
    const auto* inter_back = find_cable(control_plane, downed, 29, 9, 5, 9);
    ASSERT_NE(inter_back, nullptr);
    EXPECT_FALSE(has_cable(control_plane, unused, 5, 9, 29, 9));
    EXPECT_FALSE(has_cable(control_plane, unused, 29, 9, 5, 9));
    std::size_t intermesh_downed = 0;
    std::size_t intermesh_unused = 0;
    for (const auto& link : downed) {
        intermesh_downed += link.is_intermesh();
    }
    for (const auto& link : unused) {
        intermesh_unused += link.is_intermesh();
    }
    EXPECT_EQ(intermesh_downed, 2u);
    EXPECT_EQ(intermesh_unused, 0u);
    EXPECT_EQ(intermesh_request(control_plane.get_mesh_graph(), inter->src_mesh(), inter->dst_mesh()), 16u);
    if (owns(inter->src_node)) {
        EXPECT_EQ(assigned_toward(control_plane, inter->src_mesh(), inter->dst_mesh()), 15u);
    }
    if (owns(inter_back->src_node)) {
        EXPECT_EQ(assigned_toward(control_plane, inter_back->src_mesh(), inter_back->dst_mesh()), 15u);
    }
}

// STRICT intermesh is fulfilled when both the factory descriptor and the live descriptor cover the
// requested count. The missing cable on mesh 2 to mesh 3 is then unused, not a downed link the graph
// uses. Intra-mesh stays relaxed, so the mesh connection can still be downed. System health is relaxed,
// so this is the intermesh policy and not the system-health check.
TEST_F(FactoryDescriptorControlPlaneFixture, StrictIntermeshAllowsAnUnusedHoleWhenBothDescriptorsCoverTheCount) {
    const auto& control_plane =
        open_control_plane(tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    ASSERT_TRUE(control_plane.has_factory_descriptor());
    const auto* link_health = control_plane.get_link_health();
    ASSERT_NE(link_health, nullptr);
    const auto& mesh_graph = control_plane.get_mesh_graph();
    expect_distinct_intermesh(control_plane, false);
    for (const auto mesh_id : mesh_graph.get_mesh_ids()) {
        EXPECT_TRUE(mesh_graph.is_intra_mesh_policy_relaxed(mesh_id));
    }

    const auto& downed = link_health->get_downed_links();
    const auto& unused = link_health->get_unused_downed_links();
    const auto* inter = find_cable(control_plane, unused, 5, 9, 29, 9);
    ASSERT_NE(inter, nullptr);
    const auto* inter_back = find_cable(control_plane, unused, 29, 9, 5, 9);
    ASSERT_NE(inter_back, nullptr);
    EXPECT_FALSE(has_cable(control_plane, downed, 5, 9, 29, 9));
    EXPECT_FALSE(has_cable(control_plane, downed, 29, 9, 5, 9));

    std::size_t intermesh_downed = 0;
    std::size_t intermesh_unused = 0;
    for (const auto& link : downed) {
        intermesh_downed += link.is_intermesh();
    }
    for (const auto& link : unused) {
        intermesh_unused += link.is_intermesh();
    }
    EXPECT_EQ(intermesh_downed, 0u);
    EXPECT_EQ(intermesh_unused, 2u);
    EXPECT_EQ(intermesh_request(mesh_graph, inter->src_mesh(), inter->dst_mesh()), 4u);

    const auto local_meshes = control_plane.get_local_mesh_id_bindings();
    auto owns = [&](const FabricNodeId& node) {
        return std::find(local_meshes.begin(), local_meshes.end(), node.mesh_id) != local_meshes.end();
    };
    if (owns(inter->src_node)) {
        EXPECT_EQ(assigned_toward(control_plane, inter->src_mesh(), inter->dst_mesh()), 4u);
    }
    if (owns(inter_back->src_node)) {
        EXPECT_EQ(assigned_toward(control_plane, inter_back->src_mesh(), inter_back->dst_mesh()), 4u);
    }

    // Intra-mesh is relaxed on every mesh. The mesh connection is still a used downed link, and that
    // is allowed. A STRICT mesh would reject this same cable.
    const auto* mesh = find_cable(control_plane, downed, 0, 6, 4, 0);
    ASSERT_NE(mesh, nullptr);
    if (owns(mesh->src_node)) {
        EXPECT_EQ(control_plane.get_num_usable_routing_planes(mesh->src_node, mesh->src_direction), 2u);
    }
}

// This pod's mesh graph is RELAXED. STRICT system health is not allowed with that, even though a STRICT
// mesh graph may still run under relaxed system health.
TEST_F(FactoryDescriptorControlPlaneFixture, StrictSystemHealthRejectsARelaxedMeshGraph) {
    try {
        open_control_plane(tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
        FAIL() << "STRICT system health must reject a RELAXED mesh graph";
    } catch (const std::exception& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("RELAXED mesh graph"), std::string::npos) << what;
        EXPECT_NE(what.find("STRICT system health"), std::string::npos) << what;
    }
}

// Driven against a live cluster and a factory descriptor that name no host in common. The host filter
// rejects that during ingest, and every rank then fails together, before the mapper runs.
TEST_F(FactoryDescriptorControlPlaneFixture, AnIncompatibleDescriptorFailsAtIngest) {
    try {
        control_plane_without_factory_descriptor();
        FAIL() << "a descriptor that shares no host with the live cluster must fail during ingest";
    } catch (const std::exception& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("Factory System Descriptor host filter"), std::string::npos) << what;
        EXPECT_NE(what.find("local_ok_min=0"), std::string::npos) << what;
    }
}

}  // namespace
}  // namespace tt::tt_fabric::fabric_router_tests
