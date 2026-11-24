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
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/link_health.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt_stl/cleanup.hpp>
#include <tt_stl/span.hpp>

#include "fabric_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "tests/tt_metal/tt_fabric/common/utils.hpp"
#include "tt_metal/fabric/link_retrainer.hpp"
#include "tt_metal/fabric/physical_system_discovery.hpp"

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
    EXPECT_TRUE(control_plane.get_unused_downed_links().empty());
    EXPECT_TRUE(control_plane.get_locally_unhealthy_links().empty());
}

// The empty vectors must be the same object every call, since they are returned by reference and a
// per-call temporary would leave callers holding a dangling reference.
TEST_F(ControlPlaneFixture, NoFactoryDescriptorEmptyResultsAreStable) {
    const auto& control_plane = control_plane_without_factory_descriptor();

    EXPECT_EQ(&control_plane.get_downed_links(), &control_plane.get_downed_links());
    EXPECT_EQ(&control_plane.get_unused_downed_links(), &control_plane.get_unused_downed_links());
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
    const std::vector<experimental::LinkInfo>& links,
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

const experimental::LinkInfo* find_cable(
    const ControlPlane& control_plane,
    const std::vector<experimental::LinkInfo>& links,
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

std::size_t count_direction(const std::vector<experimental::LinkInfo>& links, const experimental::LinkInfo& sample) {
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

// A RELAXED connection the factory never cabled (#56762): the diagonal mesh 0 <-> mesh 2 of the ring-cabled pod
// is logical-only, nothing is filed for it, and the missing_links captures' four missing cables are filed as before.
TEST_F(FactoryDescriptorControlPlaneFixture, RelaxedConnectionWithNoFactoryCableIsLogicalOnly) {
    const auto& control_plane =
        open_control_plane(tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    ASSERT_TRUE(control_plane.has_factory_descriptor());
    const auto* link_health = control_plane.get_link_health();
    ASSERT_NE(link_health, nullptr);
    const auto& mesh_graph = control_plane.get_mesh_graph();
    EXPECT_TRUE(mesh_graph.is_inter_mesh_policy_relaxed());

    const MeshId diagonal_a{0};
    const MeshId diagonal_b{2};
    ASSERT_EQ(intermesh_request(mesh_graph, diagonal_a, diagonal_b), 4u);
    const auto on_diagonal = [&](MeshId a, MeshId b) {
        return (a == diagonal_a && b == diagonal_b) || (a == diagonal_b && b == diagonal_a);
    };

    const auto& downed = link_health->get_downed_links();
    const auto& unused = link_health->get_unused_downed_links();

    // Nothing is filed for the diagonal.
    for (const auto& link : downed) {
        EXPECT_FALSE(link.is_intermesh() && on_diagonal(link.src_mesh(), link.dst_mesh()))
            << "downed " << link.src_node << " chan " << static_cast<int>(link.src_chan);
    }
    for (const auto& link : unused) {
        EXPECT_FALSE(link.is_intermesh() && on_diagonal(link.src_mesh(), link.dst_mesh()))
            << "unused " << link.src_node << " chan " << static_cast<int>(link.src_chan);
    }
    EXPECT_TRUE(link_health->get_downed_intermesh_links(diagonal_a, diagonal_b).empty());
    EXPECT_TRUE(link_health->get_downed_intermesh_links(diagonal_b, diagonal_a).empty());
    EXPECT_TRUE(link_health->get_exit_nodes_with_downed_links(diagonal_a, diagonal_b).empty());

    // The captures' four missing factory cables are the only records, one per direction each.
    EXPECT_EQ(downed.size() + unused.size(), 8u);
    // Subtorus, not routed by the mesh graph.
    EXPECT_TRUE(has_cable(control_plane, unused, 0, 0, 8, 0));
    // Torus: mesh graph 2, factory 4, live 3.
    EXPECT_TRUE(has_cable(control_plane, unused, 15, 4, 23, 4));
    // Mesh: mesh graph 2, factory 2, live 1.
    EXPECT_TRUE(has_cable(control_plane, downed, 0, 6, 4, 0));
    // Intermesh: 16 factory, 15 live; downed only when the count on its boundary needs all 16.
    const auto* inter = find_cable(control_plane, downed, 5, 9, 29, 9);
    const bool inter_downed = inter != nullptr;
    if (!inter_downed) {
        inter = find_cable(control_plane, unused, 5, 9, 29, 9);
    }
    ASSERT_NE(inter, nullptr);
    EXPECT_FALSE(on_diagonal(inter->src_mesh(), inter->dst_mesh()));
    const std::size_t inter_request = intermesh_request(mesh_graph, inter->src_mesh(), inter->dst_mesh());
    EXPECT_EQ(inter_downed, inter_request > 15) << "request " << inter_request;
    EXPECT_TRUE(has_cable(control_plane, inter_downed ? downed : unused, 29, 9, 5, 9));
    EXPECT_EQ(control_plane.fsd_rerouting_active(), !downed.empty());

    // Pairing: nothing across the diagonal, min(request, live) on every ring boundary (a rank sees its own mesh).
    const auto holds_missing_cable = [&](MeshId a, MeshId b) {
        return (a == inter->src_mesh() && b == inter->dst_mesh()) || (a == inter->dst_mesh() && b == inter->src_mesh());
    };
    for (const auto local : control_plane.get_local_mesh_id_bindings()) {
        for (const auto peer : mesh_graph.get_mesh_ids()) {
            const std::size_t request = peer == local ? 0 : intermesh_request(mesh_graph, local, peer);
            if (request == 0) {
                continue;
            }
            std::size_t expected = 0;
            if (!on_diagonal(local, peer)) {
                expected = std::min<std::size_t>(request, holds_missing_cable(local, peer) ? 15 : 16);
            }
            EXPECT_EQ(assigned_toward(control_plane, local, peer), expected) << local << " -> " << peer;
        }
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

// Stands in for the cluster's retrainer, which a mock cannot run. The retrain does nothing, so rediscovery
// finds the same cables: what a cable that will not come back looks like.
class NoOpLinkRetrainer final : public LinkRetrainer {
public:
    explicit NoOpLinkRetrainer(bool can_retrain) : can_retrain_(can_retrain) {}

    bool can_retrain() const override { return can_retrain_; }

    std::vector<EthLinkEndpoint> retrain(const std::vector<EthLinkEndpoint>& local_endpoints) override {
        endpoints_per_round.push_back(local_endpoints.size());
        return {};
    }

    tt::tt_metal::PhysicalSystemDescriptor rediscover() override {
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        return tt::tt_metal::run_physical_system_discovery(
            *cluster.get_cluster_desc(),
            tt::tt_metal::distributed::multihost::DistributedContext::get_current_world(),
            cluster.get_target_device_type());
    }

    std::vector<std::size_t> endpoints_per_round;

private:
    bool can_retrain_;
};

std::unique_ptr<ControlPlane> control_plane_with_retrainer(LinkRetrainer& retrainer, uint32_t attempts) {
    auto& context = tt::tt_metal::MetalContext::instance();
    auto& rtoptions = context.rtoptions();
    rtoptions.set_fsd_link_retrain_attempts(attempts);
    auto reset = ttsl::make_cleanup([&rtoptions] { rtoptions.set_fsd_link_retrain_attempts(0); });
    return std::make_unique<ControlPlane>(
        context.get_cluster(),
        rtoptions,
        context.hal(),
        *tt::tt_metal::distributed::multihost::DistributedContext::get_current_world(),
        rtoptions.get_custom_fabric_mesh_graph_desc_path(),
        make_control_plane_config(
            tt::tt_fabric::FabricConfig::FABRIC_2D,
            tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE),
        &retrainer);
}

std::uint64_t sum_over_ranks(std::uint64_t local) {
    std::array<std::uint64_t, 1> mine{local};
    std::array<std::uint64_t, 1> total{};
    using tt::tt_metal::distributed::multihost::ReduceOp;
    tt::tt_metal::distributed::multihost::DistributedContext::get_current_world()->all_reduce(
        ttsl::Span<std::uint64_t>(mine), ttsl::Span<std::uint64_t>(total), ReduceOp::SUM);
    return total[0];
}

// The missing-links pod. Its four missing factory cables are retrained from both ends, each end by the
// rank that discovered it, in every round, because on a mock nothing comes back.
TEST_F(FactoryDescriptorControlPlaneFixture, RetrainSpendsEveryRoundOnCablesThatDoNotComeBack) {
    NoOpLinkRetrainer retrainer(/*can_retrain=*/true);
    const auto control_plane = control_plane_with_retrainer(retrainer, 2);

    // The round count is agreed across ranks, so every rank takes part in the same number of sums.
    ASSERT_EQ(retrainer.endpoints_per_round.size(), 2u);
    for (const auto endpoints : retrainer.endpoints_per_round) {
        EXPECT_EQ(sum_over_ranks(endpoints), 8u);
    }

    // Retraining in vain leaves the same records as not retraining at all.
    const auto* link_health = control_plane->get_link_health();
    ASSERT_NE(link_health, nullptr);
    EXPECT_EQ(link_health->get_downed_links().size() + link_health->get_unused_downed_links().size(), 8u);

    if (*tt::tt_metal::distributed::multihost::DistributedContext::get_current_world()->rank() != 0) {
        return;
    }
    const auto report = std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_logs_dir()) /
                        "generated" / "fabric" / "unretrainable_channels.yaml";
    std::ifstream in(report);
    ASSERT_TRUE(in.is_open()) << report;
    const std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    std::filesystem::remove(report);
    EXPECT_NE(contents.find("total_retrain_iterations: 2\n"), std::string::npos) << contents;
    std::size_t entries = 0;
    for (auto at = contents.find("  - host: "); at != std::string::npos; at = contents.find("  - host: ", at + 1)) {
        ++entries;
    }
    EXPECT_EQ(entries, 8u) << contents;
}

// One rank that cannot retrain stops all of them: half a collective retrain would hang the other half.
TEST_F(FactoryDescriptorControlPlaneFixture, RetrainNeedsEveryRankToBeAble) {
    const bool rank_zero = *tt::tt_metal::distributed::multihost::DistributedContext::get_current_world()->rank() == 0;
    NoOpLinkRetrainer retrainer(/*can_retrain=*/!rank_zero);
    const auto control_plane = control_plane_with_retrainer(retrainer, 2);

    EXPECT_TRUE(retrainer.endpoints_per_round.empty());
    ASSERT_NE(control_plane->get_link_health(), nullptr);
}

TEST_F(FactoryDescriptorControlPlaneFixture, RetrainIsOffByDefault) {
    NoOpLinkRetrainer retrainer(/*can_retrain=*/true);
    const auto control_plane = control_plane_with_retrainer(retrainer, 0);

    EXPECT_TRUE(retrainer.endpoints_per_round.empty());
    ASSERT_NE(control_plane->get_link_health(), nullptr);
}

}  // namespace
}  // namespace tt::tt_fabric::fabric_router_tests
