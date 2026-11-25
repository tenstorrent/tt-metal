// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "inter_domain_mesh.hpp"

#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric {

namespace {

std::shared_ptr<const MeshGraph> line_mesh_graph(uint32_t rows, uint32_t cols) {
    const auto text = fmt::format(
        R"(mesh_descriptors {{
  name: "D" arch: BLACKHOLE
  device_topology {{ dims: [ {}, {} ] dim_types: [ LINE, LINE ] }}
  host_topology {{ dims: [ 1, 1 ] }}
  channels {{ count: 1 policy: RELAXED }}
}}
top_level_instance {{ mesh {{ mesh_descriptor: "D" mesh_id: 0 }} }}
)",
        rows,
        cols);
    return std::make_shared<const MeshGraph>(MeshGraphDescriptor(text));
}

// the workload's links, between any two of its chips
std::map<FabricNodeId, std::set<FabricNodeId>> workload_links(const MeshGraph& workload) {
    std::map<FabricNodeId, std::set<FabricNodeId>> links;
    const auto& intra = workload.get_intra_mesh_connectivity();
    const auto& inter = workload.get_inter_mesh_connectivity();
    for (MeshId mesh : workload.get_mesh_ids()) {
        for (ChipId chip = 0; chip < intra.at(*mesh).size(); ++chip) {
            const FabricNodeId node(mesh, chip);
            links[node];
            for (const auto& [peer, _] : intra.at(*mesh).at(chip)) {
                if (peer != chip) {
                    links[node].insert(FabricNodeId(mesh, peer));
                }
            }
            if (*mesh < inter.size() && chip < inter.at(*mesh).size()) {
                for (const auto& [peer_mesh, edge] : inter.at(*mesh).at(chip)) {
                    for (ChipId peer : edge.connected_chip_ids) {
                        links[node].insert(FabricNodeId(peer_mesh, peer));
                        links[FabricNodeId(peer_mesh, peer)].insert(node);
                    }
                }
            }
        }
    }
    return links;
}

std::vector<uint32_t> squarest_first(uint32_t chips) {
    std::vector<uint32_t> rows;
    for (uint32_t r = 1; r * r <= chips; ++r) {
        if (chips % r == 0) {
            rows.push_back(r);
        }
    }
    std::reverse(rows.begin(), rows.end());
    return rows;
}

// A rows x (chips / rows) grid over these chips on these links: the first of up to `tries` embeddings `accept` takes.
std::optional<InterDomainMesh> embed(
    const std::map<FabricNodeId, std::set<FabricNodeId>>& links,
    uint32_t rows,
    size_t tries,
    const std::function<bool(const InterDomainMesh&)>& accept) {
    // by index: the solver needs default-constructible ids
    std::vector<FabricNodeId> nodes;
    std::map<FabricNodeId, uint32_t> index;
    for (const auto& [node, _] : links) {
        index.emplace(node, static_cast<uint32_t>(nodes.size()));
        nodes.push_back(node);
    }
    AdjacencyGraph<uint32_t>::AdjacencyMap adjacency;
    for (const auto& [node, peers] : links) {
        auto& out = adjacency[index.at(node)];
        for (const auto& peer : peers) {
            out.push_back(index.at(peer));
        }
    }
    const AdjacencyGraph<uint32_t> graph(std::move(adjacency));
    const auto chips = static_cast<uint32_t>(nodes.size());
    auto grid = line_mesh_graph(rows, chips / rows);
    const auto target = build_adjacency_graph_logical(*grid).at(MeshId{0});
    for (const auto& result : solve_topology_mapping_n(
             target,
             graph,
             MappingConstraints<FabricNodeId, uint32_t>{},
             tries,
             ConnectionValidationMode::RELAXED,
             /*quiet_mode=*/true)) {
        InterDomainMesh domain{.mesh_graph = grid};
        for (const auto& [d, node] : result.target_to_global) {
            domain.domain_chip.emplace(nodes[node], d.chip_id);
        }
        if (accept(domain)) {
            log_info(
                tt::LogFabric,
                "inter-domain: VC2 spans the domain's {} chips as one {}x{} mesh",
                chips,
                rows,
                chips / rows);
            return domain;
        }
    }
    return std::nullopt;
}

}  // namespace

std::optional<RoutingDirection> InterDomainMesh::direction(const FabricNodeId& a, const FabricNodeId& b) const {
    const auto ia = domain_chip.find(a);
    const auto ib = domain_chip.find(b);
    if (ia == domain_chip.end() || ib == domain_chip.end()) {
        return std::nullopt;
    }
    const auto& edges = mesh_graph->get_intra_mesh_connectivity()[0][ia->second];
    const auto it = edges.find(ib->second);
    return it == edges.end() ? std::nullopt : std::optional(it->second.port_direction);
}

InterDomainMesh make_inter_domain_mesh(const MeshGraph& workload, const std::vector<DomainLink>& physical) {
    const auto meshes = workload.get_mesh_ids();
    if (meshes.size() == 1) {
        const auto shape = workload.get_mesh_shape(meshes.front());
        InterDomainMesh domain{.mesh_graph = line_mesh_graph(shape[0], shape[1])};
        for (ChipId chip = 0; chip < shape.mesh_size(); ++chip) {
            domain.domain_chip[FabricNodeId(meshes.front(), chip)] = chip;
        }
        return domain;
    }

    const auto workload_link = workload_links(workload);
    // a chip has four directions for its routers (Z aside): a physical link is a candidate only where both have one
    // free
    auto links = workload_link;
    for (const auto& l : physical) {
        if (workload_link.at(l.a).size() < 4 && workload_link.at(l.b).size() < 4) {
            links[l.a].insert(l.b);
            links[l.b].insert(l.a);
        }
    }
    std::vector<DomainLink> vc2_links;
    auto fits = [&](const InterDomainMesh& domain) {
        vc2_links.clear();
        std::map<FabricNodeId, std::set<FabricNodeId>> extra;
        for (const auto& l : physical) {
            if (domain.direction(l.a, l.b) && !workload_link.at(l.a).contains(l.b)) {
                vc2_links.push_back(l);
                extra[l.a].insert(l.b);
                extra[l.b].insert(l.a);
            }
        }
        return std::all_of(extra.begin(), extra.end(), [&](const auto& e) {
            return workload_link.at(e.first).size() + e.second.size() <= 4;
        });
    };
    // each shape, squarest first: on the workload's links if they embed it, else on every link
    for (uint32_t rows : squarest_first(static_cast<uint32_t>(links.size()))) {
        auto domain = embed(workload_link, rows, 1, [](const InterDomainMesh&) { return true; });
        if (!domain && !physical.empty()) {
            domain = embed(links, rows, 16, fits);
        }
        if (domain) {
            fits(*domain);
            domain->vc2_links = vc2_links;
            return std::move(*domain);
        }
    }
    TT_THROW("inter-domain: the links embed no grid over the domain's {} chips", links.size());
}

InterDomainMesh embed_inter_domain_mesh(const std::map<FabricNodeId, std::set<FabricNodeId>>& links) {
    for (uint32_t rows : squarest_first(static_cast<uint32_t>(links.size()))) {
        if (auto domain = embed(links, rows, 1, [](const InterDomainMesh&) { return true; })) {
            return std::move(*domain);
        }
    }
    TT_THROW("inter-domain: the links embed no grid over the domain's {} chips", links.size());
}

}  // namespace tt::tt_fabric
