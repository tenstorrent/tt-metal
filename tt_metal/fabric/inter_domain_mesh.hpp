// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

namespace tt::tt_fabric {

// A physical link: its eth channels at a and at b.
struct DomainLink {
    FabricNodeId a{MeshId{0}, 0};
    chan_id_t a_chan = 0;
    FabricNodeId b{MeshId{0}, 0};
    chan_id_t b_chan = 0;
};

// VC2's view of a domain: one 2D mesh over all of its chips, whatever meshes the workload splits them into.
struct InterDomainMesh {
    std::shared_ptr<const MeshGraph> mesh_graph;  // mesh 0, every axis a line
    std::map<FabricNodeId, ChipId> domain_chip;
    std::vector<DomainLink> vc2_links;  // the domain's edges the workload has no link on: VC2's alone

    MeshShape shape() const { return mesh_graph->get_mesh_shape(MeshId{0}); }
    // the domain direction of the link a -> b, if it is a domain edge
    std::optional<RoutingDirection> direction(const FabricNodeId& a, const FabricNodeId& b) const;
    // a landing maps the workload's (mesh, chip) onto the domain's chip
    bool translates() const;
};

// The squarest grid over the workload's chips: its one mesh if that is as square, else the first grid, squarest first,
// that the workload's links embed, or failing those its physical links (`physical`: every link between two of its
// chips, the workload's or not; `perimeter`: each chip's peers outside the domain, whose links keep a direction each).
InterDomainMesh make_inter_domain_mesh(
    const MeshGraph& workload,
    const std::vector<DomainLink>& physical = {},
    const std::map<FabricNodeId, uint32_t>& perimeter = {});

// The squarest grid over these chips that these links embed.
InterDomainMesh embed_inter_domain_mesh(const std::map<FabricNodeId, std::set<FabricNodeId>>& links);

}  // namespace tt::tt_fabric
