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
    StockWiringRules(const std::vector<Port>& ports, const WiringOptions& options) :
        options_(options), is_2d_(is_2D_topology(options.topology)) {
        for (const auto& p : ports) {
            chips_[p.node].at(*p.direction) = stock_capability(p.capability);
        }
    }

    std::vector<Vc> vcs() const override {
        std::vector<Vc> vcs{{.id = 0, .name = "vc0"}};
        if (is_2d_ && options_.vc_config.requires_vc1) {
            vcs.push_back({.id = 1, .name = "vc1_intermesh"});
        }
        if (is_2d_ && options_.vc_config.requires_vc2) {
            vcs.push_back({.id = 2, .name = "vc2"});
        }
        return vcs;
    }

    // stock skips the VC2 receiver on the inter-mesh Z router, so it gets no VC2 at all here
    bool carries(const Port& port, uint32_t vc) const override { return vc != 2 || !is_z_boundary(port); }

    bool has_worker(const Port&, uint32_t vc) const override { return vc != 1; }

    bool wires(const Port& in, const Port& out, uint32_t vc) const override {
        const bool vc2_forwards = vc == 2 && options_.vc_config.vc2_forwards;
        if (in.direction == out.direction || (vc == 2 && !vc2_forwards) ||
            options_.topology == Topology::NeighborExchange) {
            return false;
        }
        // Inter-domain, like an intermesh boundary: traffic leaves on VC0 (every other router feeds a perimeter
        // router's VC0) and comes in on VC2 (a perimeter router's VC2 lands and feeds any direction). Nothing comes in
        // on VC0, nothing leaves on VC2, and domains don't route through.
        const bool from_perimeter = in.capability == EdgeCapability::INTERDOMAIN;
        const bool to_perimeter = out.capability == EdgeCapability::INTERDOMAIN;
        if (from_perimeter || to_perimeter) {
            return vc != 1 && from_perimeter != to_perimeter && (vc == 0 ? to_perimeter : from_perimeter);
        }
        // in the domain VC2 forwards as VC0 does
        if (vc2_forwards) {
            vc = 0;
        }
        const auto turns = turn_set_for_router(
            options_.topology, *in.direction, chips_.at(in.node), axis_of(in.node.mesh_id), &options_.vc_config);
        return std::any_of(turns[vc].begin(), turns[vc].end(), [&](const ConnectionTarget& t) {
            return t.target_vc == vc && t.target_direction == *out.direction;
        });
    }

    bool folds_corners(MeshId mesh) const override {
        return !is_2d_ && options_.topology != Topology::NeighborExchange &&
               options_.wrap_around_meshes.contains(mesh);
    }

private:
    ExpressAxis axis_of(MeshId mesh) const {
        const auto it = options_.express_axis.find(mesh);
        return it == options_.express_axis.end() ? ExpressAxis::NONE : it->second;
    }

    WiringOptions options_;
    std::map<FabricNodeId, PerDirectionCapabilities> chips_;
    bool is_2d_ = false;
};

}  // namespace

std::unique_ptr<WiringRules> stock_wiring_rules(const std::vector<Port>& ports, const WiringOptions& options) {
    return std::make_unique<StockWiringRules>(ports, options);
}

}  // namespace tt::tt_fabric::graph
