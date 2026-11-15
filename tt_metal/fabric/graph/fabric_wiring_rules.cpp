// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_wiring_rules.hpp"

#include <algorithm>
#include <utility>

#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/builder/router_wiring_rules.hpp"

namespace tt::tt_fabric::graph {

namespace {

bool is_z_boundary(const Port& p) { return is_z_boundary_router(*p.direction, p.capability); }

class StockWiringRules : public WiringRules {
public:
    StockWiringRules(const std::vector<Port>& ports, const WiringOptions& options) : options_(options) {
        for (const auto& p : ports) {
            chips_[p.node].at(*p.direction) = p.capability;
        }
        two_d_ = is_2D_topology(options.topology);
    }

    std::vector<Vc> vcs() const override {
        std::vector<Vc> vcs{{.id = 0, .name = "vc0"}};
        if (two_d_ && options_.vc_config.requires_vc1) {
            vcs.push_back({.id = 1, .name = "vc1_intermesh"});
        }
        if (two_d_ && options_.vc_config.requires_vc2) {
            vcs.push_back({.id = 2, .name = "vc2"});
        }
        return vcs;
    }

    // stock skips the VC2 receiver on the inter-mesh Z router, so it gets no VC2 at all here
    bool carries(const Port& port, uint32_t vc) const override { return vc != 2 || !is_z_boundary(port); }

    bool has_worker(const Port&, uint32_t vc) const override { return vc != 1; }

    bool wires(const Port& in, const Port& out, uint32_t vc) const override {
        if (in.direction == out.direction || vc == 2 || options_.topology == Topology::NeighborExchange) {
            return false;
        }
        const auto turns = turn_set_for_router(
            options_.topology, *in.direction, chips_.at(in.node), axis_of(in.node.mesh_id), &options_.vc_config);
        return std::any_of(turns[vc].begin(), turns[vc].end(), [&](const ConnectionTarget& t) {
            return t.target_vc == vc && t.target_direction == *out.direction;
        });
    }

    bool folds_corners(MeshId mesh) const override {
        return !two_d_ && options_.topology != Topology::NeighborExchange &&
               options_.wrap_around_meshes.contains(mesh);
    }

private:
    ExpressAxis axis_of(MeshId mesh) const {
        const auto it = options_.express_axis.find(mesh);
        return it == options_.express_axis.end() ? ExpressAxis::NONE : it->second;
    }

    WiringOptions options_;
    std::map<FabricNodeId, PerDirectionCapabilities> chips_;
    bool two_d_ = false;
};

class NarrowedRules : public WiringRules {
public:
    NarrowedRules(std::unique_ptr<WiringRules> base, std::set<Transit> used) :
        base_(std::move(base)), used_(std::move(used)) {}

    std::vector<Vc> vcs() const override { return base_->vcs(); }
    bool carries(const Port& port, uint32_t vc) const override { return base_->carries(port, vc); }
    bool has_worker(const Port& port, uint32_t vc) const override { return base_->has_worker(port, vc); }
    bool folds_corners(MeshId mesh) const override { return base_->folds_corners(mesh); }
    bool wires(const Port& in, const Port& out, uint32_t vc) const override {
        const bool cardinals = in.capability == EdgeCapability::INTRAMESH_CARDINAL &&
                               out.capability == EdgeCapability::INTRAMESH_CARDINAL;
        return base_->wires(in, out, vc) &&
               (vc != 0 || !cardinals || used_.contains(Transit{.ingress = in.direction, .egress = out.direction}));
    }

private:
    std::unique_ptr<WiringRules> base_;
    std::set<Transit> used_;
};

}  // namespace

std::unique_ptr<WiringRules> stock_wiring_rules(const std::vector<Port>& ports, const WiringOptions& options) {
    return std::make_unique<StockWiringRules>(ports, options);
}

std::unique_ptr<WiringRules> narrowed_to_routes(std::unique_ptr<WiringRules> rules, std::set<Transit> used) {
    return std::make_unique<NarrowedRules>(std::move(rules), std::move(used));
}

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

}  // namespace tt::tt_fabric::graph
