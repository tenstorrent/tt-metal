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
};

// One mesh: the workload's own. Several: the squarest grid over all their chips that their physical links embed
// (`physical`: every link between two of them, the workload's or not).
InterDomainMesh make_inter_domain_mesh(const MeshGraph& workload, const std::vector<DomainLink>& physical = {});

// The squarest grid over these chips that these links embed.
InterDomainMesh embed_inter_domain_mesh(const std::map<FabricNodeId, std::set<FabricNodeId>>& links);

}  // namespace tt::tt_fabric
