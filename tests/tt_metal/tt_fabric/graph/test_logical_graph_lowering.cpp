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
#include <functional>
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

// packet comes in on `ingress` and leaves on `egress`
struct Transit {
    PortDirection ingress;
    PortDirection egress;
    auto operator<=>(const Transit&) const = default;
};

// Next direction a packet at `at` takes toward `dst`. On device: ControlPlane::get_forwarding_direction.
class RouteOracle {
public:
    virtual ~RouteOracle() = default;
    virtual std::optional<PortDirection> next_hop(const FabricNodeId& at, const FabricNodeId& dst) const = 0;
};

// `rules` minus the VC0 transits between cardinals no route takes.
class NarrowedRules : public WiringRules {
public:
    NarrowedRules(std::unique_ptr<WiringRules> base, std::set<Transit> used) :
        base_(std::move(base)), used_(std::move(used)) {}

    std::vector<Vc> vcs() const override { return base_->vcs(); }
    bool carries(const Port& port, uint32_t vc) const override { return base_->carries(port, vc); }
    bool has_worker(const Port& port, uint32_t vc) const override { return base_->has_worker(port, vc); }
    bool folds_corners(MeshId mesh) const override { return base_->folds_corners(mesh); }
    bool wires(const Port& in, const Port& out, uint32_t vc) const override {
        const bool cardinals =
            in.capability == EdgeCapability::INTRAMESH_CARDINAL && out.capability == EdgeCapability::INTRAMESH_CARDINAL;
        return base_->wires(in, out, vc) &&
               (vc != 0 || !cardinals || used_.contains(Transit{.ingress = in.direction, .egress = out.direction}));
    }

private:
    std::unique_ptr<WiringRules> base_;
    std::set<Transit> used_;
};

std::set<Transit> transits_from_routes(const std::vector<Port>& ports, const RouteOracle& oracle) {
    std::map<std::pair<FabricNodeId, PortDirection>, FabricNodeId> link;
    std::map<MeshId, std::set<FabricNodeId>> nodes;
    for (const auto& p : ports) {
        if (p.capability != EdgeCapability::INTRAMESH_CARDINAL) {
            continue;
        }
        nodes[p.node.mesh_id].insert(p.node);
        if (p.plane == 0) {
            link.emplace(std::pair{p.node, p.direction}, p.peer);
        }
    }

    std::set<Transit> transits;
    for (const auto& [mesh, members] : nodes) {
        for (const auto& src : members) {
            for (const auto& dst : members) {
                FabricNodeId at = src;
                std::optional<PortDirection> arrived;
                for (size_t hops = 0; at != dst; ++hops) {
                    TT_FATAL(hops <= members.size(), "fabric graph: route {} -> {} does not converge", src, dst);
                    const auto leave = oracle.next_hop(at, dst);
                    TT_FATAL(leave.has_value(), "fabric graph: no next hop at {} toward {}", at, dst);
                    if (arrived) {
                        transits.insert(Transit{.ingress = *arrived, .egress = *leave});
                    }
                    const auto it = link.find({at, *leave});
                    TT_FATAL(it != link.end(), "fabric graph: route leaves {} by a direction with no port", at);
                    at = it->second;
                    arrived = PortDirection{get_opposite_direction(**leave)};
                }
            }
        }
    }
    return transits;
}

std::unique_ptr<WiringRules> narrowed_to_routes(std::unique_ptr<WiringRules> rules, std::set<Transit> used) {
    return std::make_unique<NarrowedRules>(std::move(rules), std::move(used));
}

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
    // it also feeds a sender that belongs to another ingress direction
    const auto problems = find_problems(g, *rules, ValidationLimits{});
    ASSERT_EQ(problems.size(), 2u);
    EXPECT_TRUE(std::any_of(problems.begin(), problems.end(), [](const std::string& p) {
        return p.find("don't wire this") != std::string::npos;
    }));
}

TEST(FabricGraphValidation, ChannelLimits) {
    const auto [ports, g, rules] = lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D);
    // Interior router: worker + 3 producers = 4 VC0 senders (stock 2D shape).
    EXPECT_TRUE(find_problems(g, *rules, ValidationLimits{.max_sender_channels_per_vc = 4}).empty());
    EXPECT_FALSE(find_problems(g, *rules, ValidationLimits{.max_sender_channels_per_vc = 3}).empty());
}

// Each case breaks a valid graph one way: validation names the problem and validate() throws.
TEST(FabricGraphValidation, RejectsEachMalformation) {
    // chip 5 is interior, so its routers have a worker and forwarding senders
    const auto sender = [](const LogicalGraph& g, const std::string& router, bool injection) {
        for (ChannelIndex c : g.channels_of(*g.find_router(router), ChannelKind::Sender)) {
            if (g.channels[c].injection == injection) {
                return c;
            }
        }
        return ChannelIndex{~0u};
    };
    const auto rx = [](const LogicalGraph& g, const std::string& router) {
        return *g.receiver(*g.find_router(router), 0);
    };
    const auto eth_edge = [](LogicalGraph& g, ChannelIndex c) -> LogicalEdge& {
        return *std::find_if(g.edges.begin(), g.edges.end(), [c](const LogicalEdge& e) {
            return e.from == c && e.transport == Transport::Eth;
        });
    };

    struct Case {
        std::string expected;
        std::function<void(LogicalGraph&)> mutate;
        ValidationLimits limits{};
    };
    const std::vector<Case> cases = {
        {"does not reach",
         [&](LogicalGraph& g) { eth_edge(g, sender(g, "m0.c5.N.p0", true)).to = rx(g, "m0.c5.E.p0"); }},
        {"edge, found 2",
         [&](LogicalGraph& g) { g.edges.push_back(eth_edge(g, sender(g, "m0.c5.N.p0", true))); }},
        {"no inbound noc edge",
         [&](LogicalGraph& g) {
             const auto fwd = sender(g, "m0.c5.N.p0", false);
             std::erase_if(
                 g.edges, [fwd](const LogicalEdge& e) { return e.to == fwd && e.transport == Transport::Noc; });
         }},
        {"must run receiver -> sender",
         [&](LogicalGraph& g) {
             g.edges.push_back({sender(g, "m0.c5.E.p0", true), sender(g, "m0.c5.N.p0", false), Transport::Noc});
         }},
        {"forwarding sender for its ingress direction",
         [&](LogicalGraph& g) {
             g.edges.push_back({rx(g, "m0.c5.W.p0"), sender(g, "m0.c5.E.p0", true), Transport::Noc});
         }},
        {"forwarding sender for its ingress direction",
         [&](LogicalGraph& g) {
             const auto fwd = sender(g, "m0.c5.E.p0", false);
             g.channels[fwd].from = g.routers[g.channels[fwd].router].port.direction;
         }},
        {"U-turn",
         [&](LogicalGraph& g) {
             g.edges.push_back({rx(g, "m0.c5.N.p0"), sender(g, "m0.c5.N.p0", false), Transport::Noc});
         }},
        {"leaves its chip",
         [&](LogicalGraph& g) {
             g.edges.push_back({rx(g, "m0.c6.E.p0"), sender(g, "m0.c5.N.p0", false), Transport::Noc});
         }},
        {"changes VC", [&](LogicalGraph& g) { g.channels[rx(g, "m0.c5.E.p0")].vc = 1; }},
        {"receiver channels, limit 1",
         [&](LogicalGraph& g) {
             const auto n5 = *g.find_router("m0.c5.N.p0");
             g.channels.push_back({.router = n5, .vc = 0, .kind = ChannelKind::Receiver});
             g.routers[n5].channels.push_back(static_cast<ChannelIndex>(g.channels.size() - 1));
         },
         {.max_receiver_channels_per_vc = 1}},
    };

    for (const auto& [expected, mutate, limits] : cases) {
        auto [ports, g, rules] = lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D);
        ASSERT_TRUE(find_problems(g, *rules, limits).empty()) << expected;
        mutate(g);
        const auto problems = find_problems(g, *rules, limits);
        EXPECT_TRUE(std::any_of(
            problems.begin(),
            problems.end(),
            [&](const std::string& p) { return p.find(expected) != std::string::npos; }))
            << "expected \"" << expected << "\", got " << problems.size() << " problem(s)"
            << (problems.empty() ? std::string{} : ", first: " + problems.front());
        EXPECT_ANY_THROW(validate(g, *rules, limits)) << expected;
    }
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
    const auto cg = lower_to_channels(lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D).graph, Topology::Mesh);
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
    const auto cg = lower_to_channels(lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k1D).graph, Topology::Linear);
    EXPECT_EQ(index_of(cg, "m0.c5.N.p0", 0, "S"), 1u);
    EXPECT_EQ(index_of(cg, "m0.c5.E.p0", 0, "W"), 1u);
    EXPECT_EQ(max_senders(cg), (std::vector<uint32_t>{2}));
}

TEST(FabricGraphChannels, ZBoundaryMaxima) {
    const auto cg = lower_to_channels(lower_mesh_graph(dual_2x4(/*z=*/true), kInterMesh).graph, Topology::Mesh);
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
    const auto cg = lower_to_channels(std::move(g), Topology::Torus);
    // Stock express family: 5 VC0 senders (a chord feeding E/W lands at index 4); VC2: one worker.
    EXPECT_EQ(max_senders(cg), (std::vector<uint32_t>{5, 0, 1}));
    EXPECT_EQ(index_of(cg, "m0.c9.E.p0", 0, "Z"), 4u);
    EXPECT_EQ(index_of(cg, "m0.c9.E.p0", 2, "worker"), 0u);
}

TEST(FabricGraphChannels, LegacyStockSenderCounts) {
    auto cg = lower_to_channels(lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D).graph, Topology::Mesh);
    EXPECT_EQ(cg.sender_counts[*cg.logical.find_router("m0.c0.S.p0")][0], 2u);  // corner: worker + E
    legacy_stock_sender_counts(cg, k2D);
    for (const auto& counts : cg.sender_counts) {
        EXPECT_EQ(counts, (std::vector<uint32_t>{4}));  // stock: every 2D router is 4 wide
    }

    auto zb = lower_to_channels(lower_mesh_graph(dual_2x4(/*z=*/true), kInterMesh).graph, Topology::Mesh);
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
ChannelAllocator fake_allocator(
    const std::vector<std::vector<uint32_t>>& sender_slots, const std::vector<uint32_t>& receiver_slots) {
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
    auto cg = lower_to_channels(lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D).graph, Topology::Mesh);
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
    auto cg = lower_to_channels(lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D).graph, Topology::Mesh);
    // stock 2D needs 4 VC0 senders
    EXPECT_ANY_THROW(lower_to_physical(std::move(cg), fake_allocator({{18, 16, 16}}, {16})));
}

TEST(FabricGraphPhysical, LaysOutOnlyLocalChips) {
    auto cg = lower_to_channels(lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D).graph, Topology::Mesh);
    // another host's router wider than this host's allocator allows (an express mesh's extra VC0 sender)
    const std::vector<uint32_t> slots{18, 16, 16, 16, 16};
    cg.sender_counts[*cg.logical.find_router("m0.c1.E.p0")][0] = slots.size() + 1;
    const auto allocate = fake_allocator({slots}, {16});
    const auto pg = lower_to_physical(std::move(cg), allocate, [](const FabricNodeId& n) { return n.chip_id == 0; });
    for (ChannelIndex c = 0; c < pg.logical.channels.size(); ++c) {
        const auto r = pg.logical.channels[c].router;
        const bool local = pg.logical.routers[r].port.node.chip_id == 0;
        EXPECT_EQ(pg.addresses[c].slots != 0, local) << pg.logical.routers[r].id;
        EXPECT_EQ(pg.layouts[r].receivers[0].empty(), !local) << pg.logical.routers[r].id;
    }
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


// ---------------------------------------------------------------- router inputs

const ChannelIndex kNone = ~0u;

ChannelIndex channel_of(const ChannelGraph& cg, const std::string& router, uint32_t vc, const std::string& name) {
    const auto& g = cg.logical;
    for (ChannelIndex c : g.routers[*g.find_router(router)].channels) {
        if (g.channels[c].vc == vc && channel_name(g.channels[c]) == name) {
            return c;
        }
    }
    return kNone;
}

const LegacySlot* legacy_slot(const ChannelGraph& cg, const std::string& router, uint32_t vc, uint32_t index) {
    const auto r = *cg.logical.find_router(router);
    for (const auto& slot : cg.legacy_slots) {
        if (slot.router == r && slot.vc == vc && slot.index == index) {
            return &slot;
        }
    }
    return nullptr;
}

const RingQueries kNoRingQueries = [](const FabricNodeId&) { return ProtectedRingQueries{}; };

// 2x16 torus: X wraps, Y (2 rows) doesn't. A row-0 E router has no N producer; stock keeps its slot.
TEST(FabricGraphRouterInputs, TorusRingEntries) {
    const WiringOptions torus{.topology = Topology::Torus};
    auto cg = lower_to_channels(
        lower(metal_path("tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_2x16_torus_xy_mesh_graph_descriptor.textproto"), torus).graph,
        Topology::Torus);
    legacy_stock_sender_counts(cg, torus);
    mark_ring_entries(cg, torus, kNoRingQueries);
    EXPECT_TRUE(cg.enters_ring[channel_of(cg, "m0.c3.E.p0", 0, "worker")]);
    EXPECT_FALSE(cg.enters_ring[channel_of(cg, "m0.c3.E.p0", 0, "W")]);  // straight along the ring
    EXPECT_TRUE(cg.enters_ring[channel_of(cg, "m0.c3.E.p0", 0, "S")]);   // turns onto it
    EXPECT_EQ(channel_of(cg, "m0.c3.E.p0", 0, "N"), kNone);
    const auto* n_slot = legacy_slot(cg, "m0.c3.E.p0", 0, 2);  // E-facing: [worker, W, N, S]
    ASSERT_NE(n_slot, nullptr);
    EXPECT_TRUE(n_slot->enters_ring);  // flagged as stock flags it
    EXPECT_EQ(legacy_slot(cg, "m0.c3.E.p0", 0, 1), nullptr);
}

TEST(FabricGraphRouterInputs, MeshHasNoRingEntries) {
    auto cg = lower_to_channels(lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D).graph, Topology::Mesh);
    legacy_stock_sender_counts(cg, k2D);
    mark_ring_entries(cg, k2D, kNoRingQueries);
    EXPECT_TRUE(std::none_of(cg.enters_ring.begin(), cg.enters_ring.end(), [](bool b) { return b; }));
    // corner chip 0: S router has worker + E; stock is 4 wide, so slots 2 and 3 are legacy
    EXPECT_NE(legacy_slot(cg, "m0.c0.S.p0", 0, 2), nullptr);
    EXPECT_NE(legacy_slot(cg, "m0.c0.S.p0", 0, 3), nullptr);
    EXPECT_EQ(legacy_slot(cg, "m0.c0.S.p0", 0, 0), nullptr);
}

// fake live eth channels: direction d, plane p on chip c is channel 4 * d + p
const LiveEthChannels kFakeLive = [](const FabricNodeId&, RoutingDirection d) {
    std::vector<chan_id_t> chans;
    chans.reserve(4);
    for (uint32_t p = 0; p < 4; ++p) {
        chans.push_back(static_cast<chan_id_t>((4 * static_cast<uint32_t>(d)) + p));
    }
    return chans;
};

TEST(FabricGraphRouterInputs, EthChannelBinding) {
    const auto [ports, g, rules] = lower(metal_mgd("single_bh_galaxy_mesh_graph_descriptor.textproto"), k2D);
    const auto chans = bind_eth_channels(g, kFakeLive);
    ASSERT_EQ(chans.size(), g.routers.size());
    for (RouterIndex r = 0; r < g.routers.size(); ++r) {
        const auto& p = g.routers[r].port;
        EXPECT_EQ(chans[r], (4 * static_cast<uint32_t>(*p.direction)) + p.plane) << g.routers[r].id;
    }
    // a router with no live channel for its plane
    EXPECT_ANY_THROW(bind_eth_channels(g, [](const FabricNodeId&, RoutingDirection) { return std::vector<chan_id_t>{}; }));

    const auto masters = legacy_handshake_masters(g, chans);
    EXPECT_EQ(masters.size(), 32u);
    for (const auto& [node, r] : masters) {
        EXPECT_EQ(g.routers[r].port.node, node);
    }
}

// The layout a router hands its link peer is the peer router's, across an intermesh boundary too.
TEST(FabricGraphRouterInputs, PeerLayoutAcrossIntermesh) {
    auto cg = lower_to_channels(lower_mesh_graph(dual_2x4(/*z=*/true), kInterMesh).graph, Topology::Mesh);
    legacy_stock_sender_counts(cg, kInterMesh);
    const auto pg = lower_to_physical(std::move(cg), fake_allocator({{18, 16, 16, 16, 16}, {16, 16, 16, 16}}, {16, 12}));
    const auto& g = pg.logical;
    const auto z = *g.find_router("m0.c5.Z.p0");
    const auto peer = *g.routers[z].peer;
    EXPECT_EQ(g.routers[peer].port.node.mesh_id, MeshId{1});
    EXPECT_EQ(pg.layouts[peer].senders[0].size(), pg.sender_counts[peer][0]);
    for (uint32_t vc = 0; vc < 2; ++vc) {
        const auto rx = *g.receiver(peer, vc);
        ASSERT_EQ(pg.layouts[peer].receivers[vc].size(), 1u);
        EXPECT_EQ(pg.layouts[peer].receivers[vc][0].base_address, pg.addresses[rx].base);
        EXPECT_EQ(pg.layouts[peer].receivers[vc][0].num_slots, pg.addresses[rx].slots);
    }
    EXPECT_EQ(pg.layouts[z].receivers[1][0].num_slots, 12u);
}

// ---------------------------------------------------------------- inter-domain perimeter

UnusedLink unused(uint32_t chip, chan_id_t chan, uint64_t peer, bool in_domain = false, bool trained = true) {
    return UnusedLink{
        .node = FabricNodeId(MeshId{0}, chip),
        .chan = chan,
        .peer_asic = peer,
        .peer_in_domain = in_domain,
        .trained = trained};
}

std::vector<PerimeterPort> perimeter(
    const std::string& mgd,
    tt::tt_metal::ClusterType cluster,
    const std::vector<UnusedLink>& links,
    Topology topology = Topology::Mesh) {
    MeshGraph mesh_graph(cluster, metal_mgd(mgd).string());
    return perimeter_ports(links, ports_from_mesh_graph(mesh_graph, {}), mesh_graph, topology);
}

const auto kP150 = tt::tt_metal::ClusterType::P150;
const auto kGalaxy = tt::tt_metal::ClusterType::BLACKHOLE_GALAXY;

// bringup: one chip per domain, two links to the other domain's chip
TEST(FabricGraphPerimeter, OneChipDomain) {
    const auto p = perimeter("p150_mesh_graph_descriptor.textproto", kP150, {unused(0, 9, 7), unused(0, 8, 7)});
    ASSERT_EQ(p.size(), 2u);
    for (uint32_t i = 0; i < 2; ++i) {
        EXPECT_EQ(p[i].direction, PortDirection{RoutingDirection::N});
        EXPECT_EQ(p[i].plane, i);
        EXPECT_EQ(p[i].eth_chan, 8 + i);
        EXPECT_TRUE(p[i].link_trained);
    }
}

// like intermesh: a side per peer chip, N, E, S, W then Z; past that the link is dropped
TEST(FabricGraphPerimeter, SidePerPeer) {
    std::vector<UnusedLink> links;
    for (chan_id_t c = 1; c <= 6; ++c) {
        links.push_back(unused(0, c, 100 + c));
    }
    const auto p = perimeter("p150_mesh_graph_descriptor.textproto", kP150, links);
    ASSERT_EQ(p.size(), 5u);
    const RoutingDirection order[] = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W, RoutingDirection::Z};
    for (uint32_t i = 0; i < 5; ++i) {
        EXPECT_EQ(p[i].direction, PortDirection{order[i]});
        EXPECT_EQ(p[i].eth_chan, i + 1);
        EXPECT_EQ(p[i].plane, 0u);
    }
}

// a link back into the domain (a torus wrap under a mesh MGD) is not perimeter; sides the MGD uses are taken
TEST(FabricGraphPerimeter, FoldBackExcluded) {
    const auto p = perimeter(
        "single_bh_galaxy_mesh_graph_descriptor.textproto",
        kGalaxy,
        {unused(0, 0, 31, /*in_domain=*/true),
         unused(0, 1, 200),
         unused(3, 2, 28, /*in_domain=*/true),
         unused(5, 3, 300)});
    ASSERT_EQ(p.size(), 2u);
    EXPECT_EQ(p[0].node, FabricNodeId(MeshId{0}, 0));
    EXPECT_EQ(p[0].direction, PortDirection{RoutingDirection::N});  // corner chip 0 uses E and S
    EXPECT_EQ(p[1].node, FabricNodeId(MeshId{0}, 5));
    EXPECT_EQ(p[1].direction, PortDirection{RoutingDirection::Z});  // interior chip: only Z is free
}

// a dead link still makes a port, with the failed link train recorded
TEST(FabricGraphPerimeter, DeadLinkRecorded) {
    const auto p = perimeter(
        "p150_mesh_graph_descriptor.textproto",
        kP150,
        {unused(0, 8, 7, /*in_domain=*/false, /*trained=*/false), unused(0, 9, 7)});
    ASSERT_EQ(p.size(), 2u);
    EXPECT_FALSE(p[0].link_trained);
    EXPECT_TRUE(p[1].link_trained);
}

// an FSD cable that never trained isn't in UMD's list; it comes back untrained so the side keeps the FSD count
TEST(FabricGraphPerimeter, UntrainedFsdLinkKeepsTheCount) {
    const auto links = with_missing_links(
        {unused(0, 9, 7), unused(0, 11, 7)}, {unused(0, 10, 7, /*in_domain=*/false, /*trained=*/false)});
    const auto p = perimeter("p150_mesh_graph_descriptor.textproto", kP150, links);
    ASSERT_EQ(p.size(), 3u);
    for (uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(p[i].direction, PortDirection{RoutingDirection::N});
        EXPECT_EQ(p[i].plane, i);
        EXPECT_EQ(p[i].eth_chan, 9 + i);
        EXPECT_EQ(p[i].link_trained, i != 1);
    }
}

// a peer chip whose cables are all down still takes its side, in channel order
TEST(FabricGraphPerimeter, AllDownPeerTakesItsSide) {
    const auto links =
        with_missing_links({unused(0, 5, 8)}, {unused(0, 2, 7, false, false), unused(0, 3, 7, false, false)});
    const auto p = perimeter("p150_mesh_graph_descriptor.textproto", kP150, links);
    ASSERT_EQ(p.size(), 3u);
    EXPECT_EQ(p[0].direction, PortDirection{RoutingDirection::N});
    EXPECT_EQ(p[1].direction, PortDirection{RoutingDirection::N});
    EXPECT_FALSE(p[0].link_trained);
    EXPECT_FALSE(p[1].link_trained);
    EXPECT_EQ(p[2].direction, PortDirection{RoutingDirection::E});
    EXPECT_EQ(p[2].eth_chan, 5);
    EXPECT_TRUE(p[2].link_trained);
}

// a chan UMD lists (miswired against the FSD) stays as listed; a missing link back into the domain is dropped
TEST(FabricGraphPerimeter, MissingLinksMerge) {
    const auto links = with_missing_links(
        {unused(0, 4, 7)}, {unused(0, 4, 9, false, false), unused(0, 6, 31, /*in_domain=*/true, false)});
    ASSERT_EQ(links.size(), 2u);
    EXPECT_EQ(links[0].peer_asic, 7u);
    EXPECT_TRUE(links[0].trained);
    const auto p = perimeter("p150_mesh_graph_descriptor.textproto", kP150, links);
    ASSERT_EQ(p.size(), 1u);
    EXPECT_TRUE(p[0].link_trained);
}

// no FSD: nothing missing, the list is unchanged
TEST(FabricGraphPerimeter, NoFsdNoChange) {
    const std::vector<UnusedLink> live = {unused(0, 9, 7), unused(0, 8, 7)};
    const auto links = with_missing_links(live, {});
    ASSERT_EQ(links.size(), 2u);
    EXPECT_EQ(links[0].chan, 8);
    EXPECT_EQ(links[1].chan, 9);
}

TEST(FabricGraphPerimeter, TwoDimensionalOnly) {
    for (const auto topology : {Topology::NeighborExchange, Topology::Linear, Topology::Ring}) {
        EXPECT_ANY_THROW(perimeter("p150_mesh_graph_descriptor.textproto", kP150, {}, topology));
    }
    EXPECT_NO_THROW(perimeter("p150_mesh_graph_descriptor.textproto", kP150, {}, Topology::Torus));
}

TEST(FabricGraphPerimeter, NoLinksNoPorts) {
    EXPECT_TRUE(perimeter("single_bh_galaxy_mesh_graph_descriptor.textproto", kGalaxy, {}).empty());
}

// ---------------------------------------------------------------- inter-domain lowering

const auto kP300 = tt::tt_metal::ClusterType::P300;

// a P300's 1x2 mesh with each chip's north links leaving the domain (QuietBox 2 split into two domains)
Lowered lower_with_perimeter(const std::vector<UnusedLink>& links) {
    MeshGraph mesh_graph(kP300, metal_mgd("p300_mesh_graph_descriptor.textproto").string());
    auto ports = ports_from_mesh_graph(mesh_graph, {});
    const auto open = ports_from_perimeter(perimeter_ports(links, ports, mesh_graph, Topology::Mesh));
    ports.insert(ports.end(), open.begin(), open.end());
    auto vc_config = vc2_only();
    vc_config.vc2_forwards = true;  // as inter-domain sets it
    const WiringOptions options{.topology = Topology::Mesh, .vc_config = vc_config};
    std::shared_ptr<WiringRules> rules = stock_wiring_rules(ports, options);
    auto graph = lower_to_logical(ports, *rules);
    validate(graph, *rules, ValidationLimits{});
    return Lowered{std::move(ports), std::move(graph), std::move(rules)};
}

const std::vector<UnusedLink> kQuietBoxSplit = {unused(0, 8, 7), unused(0, 9, 7), unused(1, 8, 9), unused(1, 9, 9)};

// a perimeter router: dynamic peer, every sender on a dynamic edge, a VC2 receiver, every other local router feeding
// its VC0, and nothing fed from its VC0 receiver
TEST(FabricGraphInterDomain, PerimeterRouterShape) {
    const auto [ports, g, rules] = lower_with_perimeter(kQuietBoxSplit);
    for (uint32_t chip = 0; chip < 2; ++chip) {
        for (uint32_t plane = 0; plane < 2; ++plane) {
            const std::string id = "m0.c" + std::to_string(chip) + ".N.p" + std::to_string(plane);
            const auto r = g.find_router(id);
            ASSERT_TRUE(r.has_value()) << id;
            const auto& router = g.routers[*r];
            EXPECT_FALSE(router.peer.has_value());
            EXPECT_TRUE(g.receiver(*r, 2).has_value()) << id;
            for (const auto& e : g.edges) {
                if (g.channels[e.from].router == *r && e.transport != Transport::Noc) {
                    EXPECT_EQ(e.transport, Transport::Dynamic) << id;
                }
            }
            const std::string side = chip == 0 ? "E" : "W";
            const std::string mesh_router = "m0.c" + std::to_string(chip) + "." + side + ".p" + std::to_string(plane);
            // leaving on VC0: the in-domain router turns into it, even off the X axis; nothing comes in on VC0
            EXPECT_EQ(forwarding(g, id), (Names{side}));
            EXPECT_TRUE(forwarding(g, mesh_router).empty());
            // coming in on VC2: it lands and feeds the in-domain router; nothing leaves on VC2
            EXPECT_EQ(forwarding(g, mesh_router, 2), (Names{"N"}));
            EXPECT_TRUE(forwarding(g, id, 2).empty());
        }
    }
    // the mesh link still links
    EXPECT_TRUE(g.routers[*g.find_router("m0.c0.E.p0")].peer.has_value());
}

// a dead perimeter link is recorded but gets no router
TEST(FabricGraphInterDomain, DeadLinkNoRouter) {
    const auto [ports, g, rules] =
        lower_with_perimeter({unused(0, 8, 7), unused(0, 9, 7, /*in_domain=*/false, /*trained=*/false)});
    EXPECT_TRUE(g.find_router("m0.c0.N.p0").has_value());
    EXPECT_FALSE(g.find_router("m0.c0.N.p1").has_value());
}

// dynamic edges belong to perimeter routers only
TEST(FabricGraphInterDomain, ValidationRejectsDynamicEdgeOffThePerimeter) {
    auto [ports, g, rules] = lower_with_perimeter(kQuietBoxSplit);
    EXPECT_TRUE(find_problems(g, *rules, ValidationLimits{}).empty());
    const auto mesh_router = *g.find_router("m0.c0.E.p0");
    for (auto& e : g.edges) {
        if (e.transport == Transport::Eth && g.channels[e.from].router == mesh_router) {
            e.transport = Transport::Dynamic;
            break;
        }
    }
    EXPECT_FALSE(find_problems(g, *rules, ValidationLimits{}).empty());
}

// no perimeter ports: the graph is the stock one
TEST(FabricGraphInterDomain, EmptyPerimeterIsANoOp) {
    MeshGraph mesh_graph(kP300, metal_mgd("p300_mesh_graph_descriptor.textproto").string());
    auto vc_config = vc2_only();
    vc_config.vc2_forwards = true;
    const auto with = lower_with_perimeter({});
    const auto without =
        lower_mesh_graph(mesh_graph, WiringOptions{.topology = Topology::Mesh, .vc_config = vc_config});
    ASSERT_EQ(with.graph.routers.size(), without.graph.routers.size());
    EXPECT_EQ(with.graph.channels.size(), without.graph.channels.size());
    EXPECT_EQ(with.graph.edges.size(), without.graph.edges.size());
}

// the channel graph gives a perimeter router stock-indexed slots: worker 0, forwarding by arrival side
TEST(FabricGraphInterDomain, ChannelIndexing) {
    const auto cg = lower_to_channels(lower_with_perimeter(kQuietBoxSplit).graph, Topology::Mesh);
    EXPECT_EQ(index_of(cg, "m0.c0.N.p0", 0, "worker"), 0u);
    EXPECT_NE(index_of(cg, "m0.c0.N.p0", 0, "E"), 0u);
    EXPECT_EQ(index_of(cg, "m0.c0.E.p0", 2, "worker"), 0u);  // VC2 forwards after its worker, as VC0 does
    EXPECT_EQ(index_of(cg, "m0.c0.E.p0", 2, "N"), 2u);       // 1 + N's compact index on an E-facing router (W, N, S)
}

}  // namespace
}  // namespace tt::tt_fabric::graph
