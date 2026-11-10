// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_paths.hpp"

#include <vector>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"

namespace tt::tt_fabric::manifest {
namespace {

// `path`'s parts, between its slashes.
std::vector<std::string_view> split_path(std::string_view path) {
    std::vector<std::string_view> parts;
    for (size_t slash = path.find('/'); slash != std::string_view::npos; slash = path.find('/')) {
        parts.push_back(path.substr(0, slash));
        path.remove_prefix(slash + 1);
    }
    parts.push_back(path);
    return parts;
}

// The router a path's mesh, chip and router keys name.
RouterRef parse_router_keys(std::string_view mesh, std::string_view chip, std::string_view router) {
    return {.node = FabricNodeId(parse_mesh_key(mesh), parse_chip_key(chip)), .key = parse_router_key(router)};
}

// `router` as a sibling of the router `identity` on `node`, which `path` must name.
SiblingRouterRef check_sibling(
    const RouterRef& router, std::string_view path, FabricNodeId node, const RouterIdentity& identity) {
    TT_FATAL(
        router.node == node && router.key.routing_plane == identity.key.routing_plane,
        "Fabric manifest: {} is not a sibling of {}",
        path,
        router_path({.node = node, .key = identity.key}));
    return {.direction = router.key.direction};
}

}  // namespace

std::string router_path(const RouterRef& router) {
    return fmt::format(
        "{}/{}/{}", mesh_key(router.node.mesh_id), chip_key(router.node.chip_id), router_key(router.key));
}

std::string chip_path(FabricNodeId node) {
    return fmt::format("{}/{}", mesh_key(node.mesh_id), chip_key(node.chip_id));
}

std::string chip_node_pointer(FabricNodeId node) {
    return fmt::format("/meshes/{}/chips/{}", mesh_key(node.mesh_id), chip_key(node.chip_id));
}

std::string router_node_pointer(const RouterRef& router) {
    return fmt::format("{}/routers/{}", chip_node_pointer(router.node), router_key(router.key));
}

RouterRef parse_router_path(std::string_view path) {
    const auto parts = split_path(path);
    TT_FATAL(parts.size() == 3, "Fabric manifest: {} is not a router path, such as M0/C7/E0", path);
    return parse_router_keys(parts[0], parts[1], parts[2]);
}

std::string sibling_router_path(FabricNodeId node, const RouterIdentity& identity, const SiblingRouterRef& sibling) {
    return router_path(
        {.node = node, .key = {.direction = sibling.direction, .routing_plane = identity.key.routing_plane}});
}

SiblingRouterRef parse_sibling_router_path(std::string_view path, FabricNodeId node, const RouterIdentity& identity) {
    return check_sibling(parse_router_path(path), path, node, identity);
}

std::string downstream_channel_path(
    FabricNodeId node, const RouterIdentity& identity, uint32_t vc, const DownstreamChannel& channel) {
    return fmt::format(
        "{}/channels/senders/vc{}/ch{}",
        sibling_router_path(node, identity, channel.target),
        vc,
        channel.landing_channel);
}

DownstreamChannel parse_downstream_channel_path(
    std::string_view path, FabricNodeId node, const RouterIdentity& identity, uint32_t vc) {
    const auto parts = split_path(path);
    TT_FATAL(
        parts.size() == 7 && parts[3] == "channels" && parts[4] == "senders",
        "Fabric manifest: {} is not a sender channel path, such as M0/C7/E0/channels/senders/vc0/ch1",
        path);
    TT_FATAL(
        parse_key_number(parts[5], "vc") == vc,
        "Fabric manifest: {} lands on another VC than its edge's, VC{}",
        path,
        vc);
    const RouterRef router = parse_router_keys(parts[0], parts[1], parts[2]);
    return {
        .target = check_sibling(router, path, node, identity),
        .landing_channel = parse_key_number(parts[6], "ch"),
    };
}

}  // namespace tt::tt_fabric::manifest
