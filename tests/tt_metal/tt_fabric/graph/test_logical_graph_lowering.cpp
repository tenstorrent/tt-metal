// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Logical and physical lowering from real MGDs (MeshGraph only: no devices, no mock cluster).
// Expectations follow the stock builder's wiring (builder/router_wiring_rules.cpp) and channel
// indexing (builder/fabric_builder_helpers.hpp).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>

#include <tt-metalium/cluster.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/graph/channel_graph_lowering.hpp"
#include "tt_metal/fabric/graph/fabric_graph_inputs.hpp"
#include "tt_metal/fabric/graph/fabric_wiring_rules.hpp"
#include "tt_metal/fabric/graph/logical_graph_lowering.hpp"
#include "tt_metal/fabric/graph/logical_graph_validation.hpp"
#include "tt_metal/fabric/graph/physical_graph_lowering.hpp"

namespace tt::tt_fabric::graph {
namespace {

std::filesystem::path metal_path(const std::string& rel) {
    const char* home = std::getenv("TT_METAL_HOME");
    EXPECT_NE(home, nullptr) << "TT_METAL_HOME must point at the tt-metal checkout";
    return std::filesystem::path(home ? home : ".") / rel;
}

std::filesystem::path metal_mgd(const std::string& name) {
    return metal_path("tt_metal/fabric/mesh_graph_descriptors/" + name);
}

const PortDirection kE{RoutingDirection::E};
const PortDirection kS{RoutingDirection::S};

// Test fixture only: dimension-order routing (dim 0 before dim 1, the shorter way round a wrapped
// axis), the rule RoutingTableGenerator applies. Product code takes routes from the control plane.
class DimensionOrderOracle : public RouteOracle {
public:
    DimensionOrderOracle(const MeshGraph& mesh_graph, const std::vector<Port>& ports) : mesh_graph_(mesh_graph) {
        for (const auto& p : ports) {
            directions_[p.node].insert(p.direction);
        }
    }

    std::optional<PortDirection> next_hop(const FabricNodeId& at, const FabricNodeId& dst) const override {
        const auto a = mesh_graph_.chip_to_coordinate(at.mesh_id, at.chip_id);
        const auto d = mesh_graph_.chip_to_coordinate(dst.mesh_id, dst.chip_id);
        const auto shape = mesh_graph_.get_mesh_shape(at.mesh_id);
        for (uint32_t axis = 0; axis < 2; ++axis) {
            if (a[axis] == d[axis]) {
                continue;
            }
            const PortDirection up = axis == 0 ? kS : kE;  // increasing coordinate
            const PortDirection down{get_opposite_direction(*up)};
            const int n = static_cast<int>(shape[axis]);
            const int fwd = (static_cast<int>(d[axis]) - static_cast<int>(a[axis]) + n) % n;
            const auto edge = mesh_graph_.coordinate_to_chip(
                at.mesh_id, axis == 0 ? MeshCoordinate(0, a[1]) : MeshCoordinate(a[0], 0));
            const bool wraps = directions_.at(FabricNodeId(at.mesh_id, edge)).contains(down);
            if (!wraps) {
                return d[axis] > a[axis] ? up : down;
            }
            return fwd <= n - fwd ? up : down;
        }
        return std::nullopt;
    }

private:
    const MeshGraph& mesh_graph_;
    std::map<FabricNodeId, std::set<PortDirection>> directions_;
};

struct Lowered {
    std::vector<Port> ports;
    LogicalGraph graph;
    std::shared_ptr<WiringRules> rules;
};

Lowered lower_mesh_graph(const MeshGraph& mesh_graph, const WiringOptions& options, bool narrow = false) {
    auto ports = ports_from_mesh_graph(mesh_graph, options.express_axis);
    std::shared_ptr<WiringRules> rules = stock_wiring_rules(ports, options);
    if (narrow) {
        rules = narrowed_to_routes(
            stock_wiring_rules(ports, options), transits_from_routes(ports, DimensionOrderOracle(mesh_graph, ports)));
    }
    auto graph = lower_to_logical(ports, *rules);
    validate(graph, *rules, ValidationLimits{});
    return Lowered{std::move(ports), std::move(graph), std::move(rules)};
}

Lowered lower(
    const std::filesystem::path& mgd,
    const WiringOptions& options,
    bool narrow = false,
    tt::tt_metal::ClusterType cluster = tt::tt_metal::ClusterType::BLACKHOLE_GALAXY) {
    MeshGraph mesh_graph(cluster, mgd.string());
    return lower_mesh_graph(mesh_graph, options, narrow);
}

const WiringOptions k2D{.topology = Topology::Mesh};
const WiringOptions k1D{.topology = Topology::Linear};
const WiringOptions kInterMesh{.topology = Topology::Mesh, .vc_config = IntermeshVCConfig::full_mesh()};

IntermeshVCConfig vc2_only() {
    auto config = IntermeshVCConfig::disabled();
    config.requires_vc2 = true;
    return config;
}

uint32_t planes(const std::vector<Port>& ports) {
    uint32_t n = 0;
    for (const auto& p : ports) {
        n = std::max(n, p.plane + 1);
    }
    return n;
}

std::set<std::string> forwarding(const LogicalGraph& g, const std::string& router, uint32_t vc = 0) {
    std::set<std::string> names;
    for (ChannelIndex c : g.channels_of(*g.find_router(router), ChannelKind::Sender)) {
        if (!g.channels[c].injection && g.channels[c].vc == vc) {
            names.insert(channel_name(g.channels[c]));
        }
    }
    return names;
}

using Names = std::set<std::string>;

// ---------------------------------------------------------------- logical, stock wiring

TEST(FabricGraphLowering, Galaxy8x4Mesh2DStockWiring) {
    const auto [ports, g, rules] = lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D);
    const uint32_t p = planes(ports);
    EXPECT_EQ(g.routers.size(), (8 * 3 + 7 * 4) * 2 * p);  // 8*3 + 7*4 links, two ends, per plane
    EXPECT_EQ(g.vcs.size(), 1u);                           // single mesh: no VC1
    EXPECT_TRUE(g.rings.empty());
    // Stock 2D wiring is producer-blind: every other cardinal on the chip feeds this router.
    EXPECT_EQ(forwarding(g, "m0.c0.E.p0"), (Names{"S"}));  // corner chip 0: ports E, S
    EXPECT_EQ(forwarding(g, "m0.c0.S.p0"), (Names{"E"}));
    EXPECT_EQ(forwarding(g, "m0.c5.E.p0"), (Names{"N", "S", "W"}));  // chip 5 = (1, 1)
    EXPECT_EQ(forwarding(g, "m0.c5.N.p0"), (Names{"E", "S", "W"}));
}

TEST(FabricGraphLowering, Galaxy8x4TorusXY) {
    const auto [ports, g, rules] = lower(metal_mgd("single_bh_galaxy_torus_xy_graph_descriptor.textproto"), k2D);
    const uint32_t p = planes(ports);
    EXPECT_EQ(g.routers.size(), 32u * 4 * p);  // every chip has N, E, S and W ports
    EXPECT_EQ(g.rings.size(), (8u + 4u) * p);  // a ring per row (X) and per column (Y), per plane
    for (const auto& ring : g.rings) {
        const auto direction = g.routers[ring.members.front()].port.direction;
        EXPECT_EQ(ring.members.size(), direction == kE ? 4u : 8u) << ring.id;
    }
}

TEST(FabricGraphLowering, Galaxy8x4Linear1D) {
    const auto [ports, g, rules] = lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k1D);
    for (const auto& e : g.edges) {
        if (e.transport == Transport::Noc) {
            EXPECT_EQ(
                g.routers[g.channels[e.to].router].port.direction,
                PortDirection{get_opposite_direction(*g.routers[g.channels[e.from].router].port.direction)});
        }
    }
    EXPECT_EQ(forwarding(g, "m0.c5.E.p0"), (Names{"W"}));
    EXPECT_EQ(forwarding(g, "m0.c5.N.p0"), (Names{"S"}));
    EXPECT_TRUE(forwarding(g, "m0.c0.E.p0").empty());  // corner: no W to go straight through
}

// 1D on a wrap-around mesh: a corner chip turns between its two sides, like stock's pairing; others go straight
TEST(FabricGraphLowering, OneDimensionalFoldsCorners) {
    for (const auto topology : {Topology::Linear, Topology::Ring}) {
        const auto [ports, g, rules] = lower(
            metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"),
            WiringOptions{.topology = topology, .wrap_around_meshes = {MeshId{0}}});
        EXPECT_EQ(forwarding(g, "m0.c0.E.p0"), (Names{"S"}));
        EXPECT_EQ(forwarding(g, "m0.c0.S.p0"), (Names{"E"}));
        EXPECT_EQ(forwarding(g, "m0.c5.E.p0"), (Names{"W"}));
        EXPECT_TRUE(forwarding(g, "m0.c1.E.p0").count("W"));  // edge chip: 3 sides, straight through
    }
}

// QuietBox 2 (two P300s, 2x2): every chip is a corner, so 1D wires a ring through all four
TEST(FabricGraphLowering, QuietBox1DFoldsEveryChip) {
    const auto [ports, g, rules] = lower(
        metal_mgd("p300_x2_mesh_graph_descriptor.textproto"),
        WiringOptions{.topology = Topology::Linear, .wrap_around_meshes = {MeshId{0}}},
        /*narrow=*/false,
        tt::tt_metal::ClusterType::P300_X2);
    for (const auto& r : g.routers) {
        EXPECT_EQ(forwarding(g, r.id).size(), 1u) << r.id;
    }
    EXPECT_EQ(forwarding(g, "m0.c0.E.p0"), (Names{"S"}));
    EXPECT_EQ(forwarding(g, "m0.c3.N.p0"), (Names{"W"}));
}

TEST(FabricGraphLowering, Galaxy2x16TorusXY) {
    const auto [ports, g, rules] = lower(metal_path("tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_2x16_torus_xy_mesh_graph_descriptor.textproto"), k2D);
    const uint32_t p = planes(ports);
    // X: a 16-chip ring per row. Y: 2 rows is not a genuine torus axis, so no Y ring.
    EXPECT_EQ(g.rings.size(), 2u * p);
    for (const auto& ring : g.rings) {
        EXPECT_EQ(g.routers[ring.members.front()].port.direction, kE);
        EXPECT_EQ(ring.members.size(), 16u);
    }
    EXPECT_FALSE(g.find_router("m0.c3.N.p0").has_value());
    EXPECT_TRUE(g.find_router("m0.c3.S.p0").has_value());
}

TEST(FabricGraphLowering, IsDeterministic) {
    const auto a = lower(metal_mgd("single_bh_galaxy_torus_xy_graph_descriptor.textproto"), k2D).graph;
    const auto b = lower(metal_mgd("single_bh_galaxy_torus_xy_graph_descriptor.textproto"), k2D).graph;
    ASSERT_EQ(a.routers.size(), b.routers.size());
    ASSERT_EQ(a.channels.size(), b.channels.size());
    ASSERT_EQ(a.edges.size(), b.edges.size());
    for (size_t i = 0; i < a.routers.size(); ++i) {
        EXPECT_EQ(a.routers[i].id, b.routers[i].id);
    }
    for (size_t i = 0; i < a.edges.size(); ++i) {
        EXPECT_EQ(a.edges[i].from, b.edges[i].from);
        EXPECT_EQ(a.edges[i].to, b.edges[i].to);
    }
}

// ---------------------------------------------------------------- cleanup pass

TEST(FabricGraphNarrowing, DimensionOrderDropsUnusedTransits) {
    const auto [ports, g, rules] =
        lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D, /*narrow=*/true);
    // N/S before E/W: E/W arrivals never turn into N/S.
    EXPECT_EQ(forwarding(g, "m0.c5.E.p0"), (Names{"N", "S", "W"}));
    EXPECT_EQ(forwarding(g, "m0.c5.N.p0"), (Names{"S"}));
    EXPECT_TRUE(forwarding(g, "m0.c0.S.p0").empty());
}

// ---------------------------------------------------------------- validation

TEST(FabricGraphValidation, RejectsUnwiredTransit) {
    auto [ports, g, rules] = lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D, /*narrow=*/true);
    // Chip 5's N router fed from its E router's receiver: an E -> N transit no route takes.
    const auto e5 = *g.find_router("m0.c5.E.p0");
    const auto n5 = *g.find_router("m0.c5.N.p0");
    const auto fwd = g.channels_of(n5, ChannelKind::Sender).back();
    g.edges.push_back(LogicalEdge{.from = *g.receiver(e5, 0), .to = fwd, .transport = Transport::Noc});
    const auto problems = find_problems(g, *rules, ValidationLimits{});
    ASSERT_EQ(problems.size(), 1u);
    EXPECT_NE(problems.front().find("don't wire this"), std::string::npos) << problems.front();
}

TEST(FabricGraphValidation, ChannelLimits) {
    const auto [ports, g, rules] = lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D);
    // Interior router: worker + 3 producers = 4 VC0 senders (stock 2D shape).
    EXPECT_TRUE(find_problems(g, *rules, ValidationLimits{.max_sender_channels_per_vc = 4}).empty());
    EXPECT_FALSE(find_problems(g, *rules, ValidationLimits{.max_sender_channels_per_vc = 3}).empty());
}

// ---------------------------------------------------------------- inter-mesh

// Two 2x4 meshes stacked: mesh 0's south edge (chips 4..7) to mesh 1's north edge (chips 0..3),
// two ports per chip. `dir` is the port direction both ends use: S/N (borrowed) or Z.
MeshGraph dual_2x4(bool z) {
    MeshGraph mesh_graph(
        tt::tt_metal::ClusterType::BLACKHOLE_GALAXY,
        metal_path("tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_dual_2x4_intermesh.textproto").string());
    AnnotatedIntermeshConnections connections;
    for (uint32_t k = 0; k < 8; ++k) {
        // Z ports are numbered per chip from chip 0 (2 per chip): mesh 0 chips 4..7 = Z 8..15.
        const port_id_t src = z ? port_id_t{RoutingDirection::Z, 8 + k} : port_id_t{RoutingDirection::S, k};
        const port_id_t dst = z ? port_id_t{RoutingDirection::Z, k} : port_id_t{RoutingDirection::N, k};
        connections.emplace_back(std::pair{0u, src}, std::pair{1u, dst}, k);
        connections.emplace_back(std::pair{1u, dst}, std::pair{0u, src}, k);
    }
    mesh_graph.load_intermesh_connections(connections);
    return mesh_graph;
}

TEST(FabricGraphInterMesh, BorrowedDirection) {
    const auto mesh_graph = dual_2x4(/*z=*/false);
    const auto [ports, g, rules] = lower_mesh_graph(mesh_graph, kInterMesh);
    ASSERT_EQ(g.vcs.size(), 2u);  // VC1 appears fabric-wide with inter-mesh links
    size_t inter = 0;
    for (const auto& r : g.routers) {
        if (r.port.capability == EdgeCapability::INTERMESH) {
            ++inter;
            EXPECT_NE(g.routers[*r.peer].port.node.mesh_id, r.port.node.mesh_id) << r.id;
        }
    }
    EXPECT_EQ(inter, 2u * 4 * planes(ports));  // 4 chips per side, both meshes
    // Mesh 0 chip 4 = (1, 0): N and E intra-mesh, S inter-mesh (borrowed). Wired like a cardinal.
    EXPECT_EQ(forwarding(g, "m0.c4.S.p0", 0), (Names{"E", "N"}));
    EXPECT_EQ(forwarding(g, "m0.c4.S.p0", 1), (Names{"E", "N"}));
    EXPECT_EQ(forwarding(g, "m0.c4.N.p0", 1), (Names{"E", "S"}));
}

TEST(FabricGraphInterMesh, ZBoundary) {
    const auto mesh_graph = dual_2x4(/*z=*/true);
    const auto [ports, g, rules] = lower_mesh_graph(mesh_graph, kInterMesh);
    ASSERT_EQ(g.vcs.size(), 2u);
    // Exit on VC0: cardinals feed the Z router. Entry on VC1: the Z router feeds the cardinals.
    EXPECT_EQ(forwarding(g, "m0.c5.Z.p0", 0), (Names{"E", "N", "W"}));  // chip 5 = (1, 1): N, E, W
    EXPECT_TRUE(forwarding(g, "m0.c5.Z.p0", 1).empty());
    EXPECT_EQ(forwarding(g, "m0.c5.E.p0", 1), (Names{"N", "W", "Z"}));
    EXPECT_EQ(forwarding(g, "m0.c1.E.p0", 1), (Names{"S", "W"}));  // chip 1 has no Z port
}

// ---------------------------------------------------------------- express and VC2

// 8x4 with express chords along Y between rows 2 and 5 (start 2, step 4, LINE).
TEST(FabricGraphLowering, ExpressChordsAlongY) {
    const auto [ports, g, rules] = lower(
        metal_path(
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors/express_links_8x4_mesh_graph_descriptor.textproto"),
        WiringOptions{.topology = Topology::Torus, .express_axis = {{MeshId{0}, ExpressAxis::Y}}});
    // Chip 9 = (2, 1) has N, E, S, W and a chord. E/W traffic can't turn back into Y (N, S, chord).
    EXPECT_EQ(forwarding(g, "m0.c9.Z.p0"), (Names{"N", "S"}));
    EXPECT_EQ(forwarding(g, "m0.c9.E.p0"), (Names{"N", "S", "W", "Z"}));
    EXPECT_EQ(forwarding(g, "m0.c9.N.p0"), (Names{"S", "Z"}));
    EXPECT_FALSE(g.find_router("m0.c13.Z.p0").has_value());  // row 3: no chord
}

TEST(FabricGraphLowering, Vc2IsOneHop) {
    const auto [ports, g, rules] = lower(
        metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"),
        WiringOptions{.topology = Topology::Mesh, .vc_config = vc2_only()});
    ASSERT_EQ(g.vcs.size(), 2u);
    EXPECT_EQ(g.vcs.back().id, 2u);
    for (RouterIndex r = 0; r < g.routers.size(); ++r) {
        EXPECT_TRUE(g.receiver(r, 2).has_value()) << g.routers[r].id;
        EXPECT_TRUE(forwarding(g, g.routers[r].id, 2).empty()) << g.routers[r].id;
    }
}

// ---------------------------------------------------------------- channel graph

uint32_t index_of(const ChannelGraph& cg, const std::string& router, uint32_t vc, const std::string& name) {
    const auto& g = cg.logical;
    for (ChannelIndex c : g.routers[*g.find_router(router)].channels) {
        if (g.channels[c].vc == vc && channel_name(g.channels[c]) == name) {
            return cg.index_in_vc[c];
        }
    }
    ADD_FAILURE() << router << " has no vc" << vc << " channel " << name;
    return ~0u;
}

// widest router per VC
std::vector<uint32_t> max_senders(const ChannelGraph& cg) {
    std::vector<uint32_t> max(cg.sender_counts.front().size(), 0);
    for (const auto& counts : cg.sender_counts) {
        for (size_t vc = 0; vc < counts.size(); ++vc) {
            max[vc] = std::max(max[vc], counts[vc]);
        }
    }
    return max;
}

TEST(FabricGraphChannels, StockIndexing) {
    const auto cg = lower_to_channels(lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D).graph, true);
    // E-facing: [worker, W, N, S]; N-facing: [worker, E, W, S] (compact rank in E, W, N, S order).
    EXPECT_EQ(index_of(cg, "m0.c5.E.p0", 0, "worker"), 0u);
    EXPECT_EQ(index_of(cg, "m0.c5.E.p0", 0, "W"), 1u);
    EXPECT_EQ(index_of(cg, "m0.c5.E.p0", 0, "N"), 2u);
    EXPECT_EQ(index_of(cg, "m0.c5.E.p0", 0, "S"), 3u);
    EXPECT_EQ(index_of(cg, "m0.c5.N.p0", 0, "E"), 1u);
    EXPECT_EQ(index_of(cg, "m0.c5.N.p0", 0, "W"), 2u);
    EXPECT_EQ(index_of(cg, "m0.c5.N.p0", 0, "S"), 3u);
    EXPECT_EQ(cg.sender_counts[*cg.logical.find_router("m0.c5.E.p0")][0], 4u);  // interior: all 4
    EXPECT_EQ(cg.sender_counts[*cg.logical.find_router("m0.c0.E.p0")][0], 4u);  // corner: S lands at 3
    EXPECT_EQ(max_senders(cg), (std::vector<uint32_t>{4}));
}

TEST(FabricGraphChannels, OneDimensionalUsesSlotOne) {
    const auto cg = lower_to_channels(lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k1D).graph, false);
    EXPECT_EQ(index_of(cg, "m0.c5.N.p0", 0, "S"), 1u);
    EXPECT_EQ(index_of(cg, "m0.c5.E.p0", 0, "W"), 1u);
    EXPECT_EQ(max_senders(cg), (std::vector<uint32_t>{2}));
}

TEST(FabricGraphChannels, ZBoundaryMaxima) {
    const auto cg = lower_to_channels(lower_mesh_graph(dual_2x4(/*z=*/true), kInterMesh).graph, true);
    // Stock with a Z boundary: [5, 4] senders. Mesh 1's Z chips (its top row) have an S port, so
    // their Z router uses worker + E, W, S = index 4; VC1 "from Z" sits at index 3 on cardinals.
    EXPECT_EQ(max_senders(cg), (std::vector<uint32_t>{5, 4}));
    EXPECT_EQ(index_of(cg, "m0.c5.E.p0", 1, "Z"), 3u);
}

TEST(FabricGraphChannels, ExpressAndVc2Maxima) {
    auto g = lower(
                 metal_path("tests/tt_metal/tt_fabric/custom_mesh_descriptors/"
                            "express_links_8x4_mesh_graph_descriptor.textproto"),
                 WiringOptions{
                     .topology = Topology::Torus,
                     .vc_config = vc2_only(),
                     .express_axis = {{MeshId{0}, ExpressAxis::Y}}})
                 .graph;
    const auto cg = lower_to_channels(std::move(g), true);
    // Stock express family: 5 VC0 senders (a chord feeding E/W lands at index 4); VC2: one worker.
    EXPECT_EQ(max_senders(cg), (std::vector<uint32_t>{5, 0, 1}));
    EXPECT_EQ(index_of(cg, "m0.c9.E.p0", 0, "Z"), 4u);
    EXPECT_EQ(index_of(cg, "m0.c9.E.p0", 2, "worker"), 0u);
}

TEST(FabricGraphChannels, LegacyStockSenderCounts) {
    auto cg = lower_to_channels(lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D).graph, true);
    EXPECT_EQ(cg.sender_counts[*cg.logical.find_router("m0.c0.S.p0")][0], 2u);  // corner: worker + E
    legacy_stock_sender_counts(cg, k2D);
    for (const auto& counts : cg.sender_counts) {
        EXPECT_EQ(counts, (std::vector<uint32_t>{4}));  // stock: every 2D router is 4 wide
    }

    auto zb = lower_to_channels(lower_mesh_graph(dual_2x4(/*z=*/true), kInterMesh).graph, true);
    legacy_stock_sender_counts(zb, kInterMesh);
    const auto counts = [&](const std::string& id) { return zb.sender_counts[*zb.logical.find_router(id)]; };
    EXPECT_EQ(counts("m0.c5.Z.p0"), (std::vector<uint32_t>{5, 4}));  // boundary, nothing on its VC1
    EXPECT_EQ(counts("m0.c5.E.p0"), (std::vector<uint32_t>{4, 4}));  // boundary chip: + from Z on VC1
    EXPECT_EQ(counts("m0.c1.E.p0"), (std::vector<uint32_t>{4, 3}));  // no Z port
}

// ---------------------------------------------------------------- physical

constexpr uint32_t kRegionStart = 0x20000;
constexpr uint32_t kSlotBytes = 4416;

// Stand-in for the stock allocator: the router's senders by VC, then a receiver per VC, from the region start.
ChannelAllocator fake_allocator(std::vector<std::vector<uint32_t>> sender_slots, std::vector<uint32_t> receiver_slots) {
    return [=](const std::array<size_t, builder_config::MAX_NUM_VCS>& counts) {
        RouterChannelLayout layout;
        uint32_t addr = kRegionStart;
        for (size_t vc = 0; vc < sender_slots.size(); ++vc) {
            TT_FATAL(counts[vc] <= sender_slots[vc].size(), "fake allocator: too many vc{} senders", vc);
            for (size_t i = 0; i < counts[vc]; ++i) {
                layout.senders[vc].push_back({addr, sender_slots[vc][i]});
                addr += sender_slots[vc][i] * kSlotBytes;
            }
        }
        for (size_t vc = 0; vc < receiver_slots.size(); ++vc) {
            layout.receivers[vc].push_back({addr, receiver_slots[vc]});
            addr += receiver_slots[vc] * kSlotBytes;
        }
        return layout;
    };
}

TEST(FabricGraphPhysical, AddressesComeFromTheAllocator) {
    auto cg = lower_to_channels(lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D).graph, true);
    const std::vector<uint32_t> slots{18, 16, 16, 16, 16};
    const auto pg = lower_to_physical(std::move(cg), fake_allocator({slots}, {16}));
    for (ChannelIndex c = 0; c < pg.logical.channels.size(); ++c) {
        const auto& ch = pg.logical.channels[c];
        const auto r = ch.router;
        uint32_t sender_bytes = 0;
        for (uint32_t i = 0; i < pg.sender_counts[r][0]; ++i) {
            sender_bytes += slots[i] * kSlotBytes;
        }
        if (ch.kind == ChannelKind::Receiver) {
            EXPECT_EQ(pg.addresses[c].base, kRegionStart + sender_bytes) << pg.logical.routers[r].id;
            EXPECT_EQ(pg.addresses[c].slots, 16u);
        } else if (pg.index_in_vc[c] == 0) {
            EXPECT_EQ(pg.addresses[c].base, kRegionStart);
            EXPECT_EQ(pg.addresses[c].slots, 18u);
        }
    }
}

TEST(FabricGraphPhysical, RejectsTooSmallAllocation) {
    auto cg = lower_to_channels(lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D).graph, true);
    // stock 2D needs 4 VC0 senders
    EXPECT_ANY_THROW(lower_to_physical(std::move(cg), fake_allocator({{18, 16, 16}}, {16})));
}

size_t pairings_on(const LogicalGraph& g, const std::vector<LinkPairing>& links, const std::string& chip_prefix) {
    return std::count_if(links.begin(), links.end(), [&](const LinkPairing& l) {
        return g.routers[l.a].id.starts_with(chip_prefix);
    });
}

TEST(FabricGraphPhysical, LegacyLinkPairings) {
    const auto mgd = metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto");
    const auto two_d = lower(mgd, k2D);
    const auto links_2d = legacy_link_pairings(two_d.graph, Topology::Mesh, {});
    const uint32_t p = planes(two_d.ports);
    EXPECT_EQ(pairings_on(two_d.graph, links_2d, "m0.c5."), 6u * p);  // N, E, S, W: every pair
    EXPECT_EQ(pairings_on(two_d.graph, links_2d, "m0.c0."), 1u * p);  // corner: E, S
    for (const auto& l : links_2d) {
        EXPECT_EQ(l.noc_vc, FabricEriscDatamoverConfig::DEFAULT_NOC_VC + (l.link % FabricEriscDatamoverConfig::NUM_EDM_NOC_VCS));
    }

    const auto one_d = lower(mgd, k1D);
    EXPECT_EQ(pairings_on(one_d.graph, legacy_link_pairings(one_d.graph, Topology::Linear, {}), "m0.c0."), 0u);
    EXPECT_EQ(pairings_on(one_d.graph, legacy_link_pairings(one_d.graph, Topology::Linear, {MeshId{0}}), "m0.c0."), p);
    EXPECT_EQ(pairings_on(one_d.graph, legacy_link_pairings(one_d.graph, Topology::Linear, {}), "m0.c5."), 2u * p);
    EXPECT_TRUE(legacy_link_pairings(one_d.graph, Topology::NeighborExchange, {}).empty());
}

}  // namespace
}  // namespace tt::tt_fabric::graph
