// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <set>
#include <string>

#include <fmt/format.h>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>

#include "tt_metal/fabric/inter_domain_mesh.hpp"

namespace tt::tt_fabric {
namespace {

// `meshes` stacked rows x cols meshes, mesh m's last row cabled to mesh m+1's first
MeshGraph stacked(uint32_t meshes, uint32_t rows, uint32_t cols) {
    std::string text = fmt::format(
        R"(mesh_descriptors {{
  name: "M" arch: BLACKHOLE
  device_topology {{ dims: [ {}, {} ] }}
  host_topology {{ dims: [ 1, 1 ] }}
  channels {{ count: 2 policy: RELAXED }}
}}
graph_descriptors {{
  name: "G" type: "FABRIC"
)",
        rows,
        cols);
    for (uint32_t m = 0; m < meshes; ++m) {
        text += fmt::format("  instances {{ mesh {{ mesh_descriptor: \"M\" mesh_id: {} }} }}\n", m);
    }
    for (uint32_t m = 0; m + 1 < meshes; ++m) {
        for (uint32_t c = 0; c < cols; ++c) {
            text += fmt::format(
                "  connections {{ nodes {{ mesh {{ mesh_descriptor: \"M\" mesh_id: {} device_id: {} }} }} "
                "nodes {{ mesh {{ mesh_descriptor: \"M\" mesh_id: {} device_id: {} }} }} channels {{ count: 1 }} }}\n",
                m,
                (rows - 1) * cols + c,
                m + 1,
                c);
        }
    }
    text += "}\ntop_level_instance { graph { graph_descriptor: \"G\" graph_id: 0 } }\n";
    return MeshGraph(MeshGraphDescriptor(text));
}

using Links = std::map<FabricNodeId, std::set<FabricNodeId>>;

// what the control plane resolves for stacked(): each mesh a grid, mesh m's last row cabled to mesh m+1's first
Links stacked_links(uint32_t meshes, uint32_t rows, uint32_t cols) {
    Links links;
    auto link = [&](const FabricNodeId& a, const FabricNodeId& b) {
        links[a].insert(b);
        links[b].insert(a);
    };
    for (uint32_t m = 0; m < meshes; ++m) {
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                const FabricNodeId node(MeshId{m}, r * cols + c);
                links[node];
                if (c + 1 < cols) {
                    link(node, FabricNodeId(MeshId{m}, r * cols + c + 1));
                }
                if (r + 1 < rows) {
                    link(node, FabricNodeId(MeshId{m}, (r + 1) * cols + c));
                } else if (m + 1 < meshes) {
                    link(node, FabricNodeId(MeshId{m + 1}, c));
                }
            }
        }
    }
    return links;
}

// every chip once, and every domain edge one of the links
void expect_embedded(const InterDomainMesh& domain, const Links& links) {
    std::set<ChipId> seen;
    std::map<ChipId, FabricNodeId> node_of;
    for (const auto& [node, chip] : domain.domain_chip) {
        EXPECT_TRUE(seen.insert(chip).second) << "domain chip " << chip << " twice";
        node_of.emplace(chip, node);
    }
    EXPECT_EQ(seen.size(), domain.shape().mesh_size());
    EXPECT_EQ(seen.size(), links.size());
    const auto& edges = domain.mesh_graph->get_intra_mesh_connectivity()[0];
    for (ChipId a = 0; a < edges.size(); ++a) {
        for (const auto& [b, _] : edges[a]) {
            const auto& na = node_of.at(a);
            const auto& nb = node_of.at(b);
            EXPECT_TRUE(links.at(na).contains(nb)) << "domain edge " << a << "-" << b << " is no link";
            EXPECT_TRUE(domain.direction(na, nb).has_value());
        }
    }
}

TEST(InterDomainMesh, OneMeshIsTheWorkloads) {
    const auto workload = stacked(1, 4, 4);
    const auto domain = make_inter_domain_mesh(workload);
    EXPECT_EQ(domain.shape(), MeshShape(4, 4));
    EXPECT_FALSE(domain.translates());
    for (const auto& [node, chip] : domain.domain_chip) {
        EXPECT_EQ(node.chip_id, chip);
    }
    EXPECT_EQ(domain.direction(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 1)), RoutingDirection::E);
    EXPECT_EQ(domain.direction(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 4)), RoutingDirection::S);
    EXPECT_FALSE(domain.direction(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 2)).has_value());
}

// a 1x16 mesh's chip i at its place in a 4x4 grid, snaking
std::pair<ChipId, ChipId> snake_at(ChipId i) { return {i / 4, (i / 4) % 2 == 0 ? i % 4 : 3 - i % 4}; }

std::vector<DomainLink> snake_links() {
    const auto at = snake_at;
    std::vector<DomainLink> physical;
    for (ChipId a = 0; a < 16; ++a) {
        for (ChipId b = a + 1; b < 16; ++b) {
            const auto [ya, xa] = at(a);
            const auto [yb, xb] = at(b);
            if (std::abs(static_cast<int>(ya) - static_cast<int>(yb)) +
                    std::abs(static_cast<int>(xa) - static_cast<int>(xb)) ==
                1) {
                physical.push_back(DomainLink{.a = FabricNodeId(MeshId{0}, a), .b = FabricNodeId(MeshId{0}, b)});
            }
        }
    }
    return physical;
}

TEST(InterDomainMesh, OneLineMeshTakesTheSquareOnItsPhysicalLinks) {
    const auto workload = stacked(1, 1, 16);
    const auto physical = snake_links();
    const auto domain = make_inter_domain_mesh(workload, physical);
    EXPECT_EQ(domain.shape(), MeshShape(4, 4));
    EXPECT_TRUE(domain.translates());
    EXPECT_EQ(domain.vc2_links.size(), physical.size() - 15);
    for (const auto& l : domain.vc2_links) {
        EXPECT_TRUE(domain.direction(l.a, l.b).has_value());
    }
}

TEST(InterDomainMesh, ThePerimeterKeepsItsDirections) {
    // the 4x4's inner chips each have a cable out of the domain: the square would leave it no direction
    const auto workload = stacked(1, 1, 16);
    const auto physical = snake_links();
    std::map<FabricNodeId, uint32_t> perimeter;
    for (ChipId i = 0; i < 16; ++i) {
        const auto [y, x] = snake_at(i);
        if (y > 0 && y < 3 && x > 0 && x < 3) {
            perimeter[FabricNodeId(MeshId{0}, i)] = 1;
        }
    }
    const auto domain = make_inter_domain_mesh(workload, physical, perimeter);
    EXPECT_NE(domain.shape(), MeshShape(4, 4));
    std::map<FabricNodeId, uint32_t> used = perimeter;
    for (ChipId i = 0; i < 16; ++i) {
        used[FabricNodeId(MeshId{0}, i)] += (i > 0) + (i < 15);
    }
    for (const auto& l : domain.vc2_links) {
        EXPECT_LE(++used[l.a], 4u);
        EXPECT_LE(++used[l.b], 4u);
    }
}

TEST(InterDomainMesh, TwoMeshesOfTwoByFourAreOneFourByFour) {
    const auto links = stacked_links(2, 2, 4);
    const auto domain = embed_inter_domain_mesh(links);
    EXPECT_EQ(domain.shape(), MeshShape(4, 4));
    expect_embedded(domain, links);
}

TEST(InterDomainMesh, TwoSingleChipMeshesAreOneByTwo) {
    const auto links = stacked_links(2, 1, 1);
    const auto domain = embed_inter_domain_mesh(links);
    EXPECT_EQ(domain.shape(), MeshShape(1, 2));
    expect_embedded(domain, links);
}

TEST(InterDomainMesh, QuadOfFourMeshesIsItsThirtyTwoByFour) {
    const auto links = stacked_links(4, 8, 4);
    const auto domain = embed_inter_domain_mesh(links);
    EXPECT_EQ(domain.shape().mesh_size(), 128u);
    expect_embedded(domain, links);
}

}  // namespace
}  // namespace tt::tt_fabric
