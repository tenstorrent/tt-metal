// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

namespace tt::tt_fabric {
namespace {

// Only used as the descriptor label echoed back in the error message; no file is read.
constexpr std::string_view kMeshGraphDescPath = "mesh_graph_descriptors/example_two_meshes.textproto";

MeshFabricConfigObservation make_observation(
    uint32_t mesh_id, uint32_t rank, uint32_t mesh_host_rank, FabricConfig fabric_config) {
    return MeshFabricConfigObservation{
        .mesh_id = MeshId{mesh_id}, .rank = rank, .mesh_host_rank = mesh_host_rank, .fabric_config = fabric_config};
}

// Two single-chip meshes cabled to each other.
InterMeshConnectivity make_two_connected_meshes() {
    RouterEdge edge_to_mesh1{.port_direction = RoutingDirection::E, .connected_chip_ids = {0}, .weight = 1};
    RouterEdge edge_to_mesh0{.port_direction = RoutingDirection::W, .connected_chip_ids = {0}, .weight = 1};
    return InterMeshConnectivity{{{{MeshId{1}, edge_to_mesh1}}}, {{{MeshId{0}, edge_to_mesh0}}}};
}

// Two single-chip meshes with no inter-mesh cabling.
InterMeshConnectivity make_two_disconnected_meshes() { return InterMeshConnectivity{{{}}, {{}}}; }

TEST(FabricConfigConsistency, MatchingConfigsAcrossRanksOfOneMeshIsAccepted) {
    const std::vector<MeshFabricConfigObservation> observations = {
        make_observation(0, 0, 0, FabricConfig::FABRIC_2D), make_observation(0, 1, 1, FabricConfig::FABRIC_2D)};
    EXPECT_NO_THROW(validate_fabric_config_consistency(
        observations, make_two_disconnected_meshes(), std::string(kMeshGraphDescPath)));
}

TEST(FabricConfigConsistency, MismatchedConfigsAcrossRanksOfOneMeshIsRejected) {
    const std::vector<MeshFabricConfigObservation> observations = {
        make_observation(0, 0, 0, FabricConfig::FABRIC_2D), make_observation(0, 1, 1, FabricConfig::FABRIC_2D_TORUS_Y)};
    try {
        validate_fabric_config_consistency(
            observations, make_two_disconnected_meshes(), std::string(kMeshGraphDescPath));
        FAIL() << "Expected mismatched FabricConfig across ranks of one mesh to be rejected";
    } catch (const std::exception& e) {
        const std::string message = e.what();
        // The error must be actionable: MGD path, mesh/rank identity, both observed configs and the ticket
        // tracking heterogeneous per-mesh support.
        EXPECT_NE(message.find(kMeshGraphDescPath), std::string::npos);
        EXPECT_NE(message.find("mesh_id=0"), std::string::npos);
        EXPECT_NE(message.find("mesh_host_rank=1"), std::string::npos);
        EXPECT_NE(message.find("rank=1"), std::string::npos);
        EXPECT_NE(message.find("FABRIC_2D_TORUS_Y"), std::string::npos);
        EXPECT_NE(message.find("56561"), std::string::npos);
    }
}

TEST(FabricConfigConsistency, MismatchedConfigsAcrossConnectedMeshesIsRejected) {
    const std::vector<MeshFabricConfigObservation> observations = {
        make_observation(0, 0, 0, FabricConfig::FABRIC_2D), make_observation(1, 1, 0, FabricConfig::FABRIC_2D_TORUS_Y)};
    try {
        validate_fabric_config_consistency(observations, make_two_connected_meshes(), std::string(kMeshGraphDescPath));
        FAIL() << "Expected mismatched FabricConfig across connected meshes to be rejected";
    } catch (const std::exception& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find(kMeshGraphDescPath), std::string::npos);
        EXPECT_NE(message.find("mesh_id=1"), std::string::npos);
        EXPECT_NE(message.find("FABRIC_2D_TORUS_Y"), std::string::npos);
        EXPECT_NE(message.find("56561"), std::string::npos);
    }
}

TEST(FabricConfigConsistency, MismatchedConfigsAcrossDisconnectedMeshesIsAccepted) {
    // Without inter-mesh links the two meshes never exchange fabric packets, so they are independent.
    const std::vector<MeshFabricConfigObservation> observations = {
        make_observation(0, 0, 0, FabricConfig::FABRIC_2D), make_observation(1, 1, 0, FabricConfig::FABRIC_2D_TORUS_Y)};
    EXPECT_NO_THROW(validate_fabric_config_consistency(
        observations, make_two_disconnected_meshes(), std::string(kMeshGraphDescPath)));
}

TEST(FabricConfigConsistency, UnsetMeshHostRankIsReportedAsUnset) {
    const std::vector<MeshFabricConfigObservation> observations = {
        make_observation(0, 0, *MESH_HOST_RANK_UNSET, FabricConfig::FABRIC_2D),
        make_observation(0, 1, *MESH_HOST_RANK_UNSET, FabricConfig::FABRIC_1D)};
    try {
        validate_fabric_config_consistency(
            observations, make_two_disconnected_meshes(), std::string(kMeshGraphDescPath));
        FAIL() << "Expected mismatched FabricConfig across ranks of one mesh to be rejected";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("mesh_host_rank=UNSET"), std::string::npos);
    }
}

}  // namespace
}  // namespace tt::tt_fabric
