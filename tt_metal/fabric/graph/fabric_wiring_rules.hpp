// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "fabric_graph_types.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"  // IntermeshVCConfig

namespace tt::tt_fabric::graph {

// What a fabric topology instantiates and wires. Lowering asks it about every router and every
// pair of routers on the same chip and plane.
class WiringRules {
public:
    virtual ~WiringRules() = default;
    virtual std::vector<Vc> vcs() const = 0;
    virtual bool carries(const Port& port, uint32_t vc) const = 0;
    virtual bool has_worker(const Port& port, uint32_t vc) const = 0;
    virtual bool has_receiver(const Port&, uint32_t) const { return true; }
    // does the receiver on `ingress` feed the router on `egress`?
    virtual bool wires(const Port& ingress, const Port& egress, uint32_t vc) const = 0;
    // 1D on a wrap-around mesh: a chip with two non-opposite directions turns between them
    virtual bool folds_corners(MeshId mesh) const = 0;
};

struct WiringOptions {
    Topology topology = Topology::Mesh;
    IntermeshVCConfig vc_config;
    std::map<MeshId, ExpressAxis> express_axis;
    std::set<MeshId> wrap_around_meshes;
    // where VC2 forwards: each port on a domain edge, by its direction in the domain
    std::map<std::pair<FabricNodeId, RoutingDirection>, RoutingDirection> vc2_directions;
};

inline EdgeCapability stock_capability(EdgeCapability c) {
    return c == EdgeCapability::INTERDOMAIN || c == EdgeCapability::INTRADOMAIN ? EdgeCapability::INTRAMESH_CARDINAL
                                                                                : c;
}

// Same routers, VCs and turns as the stock builder: the turn sets come from turn_set_for_router.
std::unique_ptr<WiringRules> stock_wiring_rules(const std::vector<Port>& ports, const WiringOptions& options);

}  // namespace tt::tt_fabric::graph
