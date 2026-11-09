// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

// One part of the manifest refers to a router elsewhere by its path, e.g. "M0/C7/E0": its mesh, chip and router keys
// (fabric_manifest_names.hpp). The writer writes each path below and the reader parses it back, so each pair must stay
// inverses.
namespace tt::tt_fabric::manifest {

std::string router_path(const RouterRef& router);
RouterRef parse_router_path(std::string_view path);

// A sibling is on the same chip and routing plane as the router `identity` on `node` that refers to it.
std::string sibling_router_path(FabricNodeId node, const RouterIdentity& identity, const SiblingRouterRef& sibling);
// Fails unless `path` is a sibling's: on `node`, and on `identity`'s routing plane.
SiblingRouterRef parse_sibling_router_path(std::string_view path, FabricNodeId node, const RouterIdentity& identity);

// The sender channel a downstream edge on `vc` lands on: the sibling's path, then channels/senders/vc<v>/ch<c>.
std::string downstream_channel_path(
    FabricNodeId node, const RouterIdentity& identity, uint32_t vc, const DownstreamChannel& channel);
// Fails unless `path` is a sender channel on `vc` of a sibling of the router `identity` on `node`.
DownstreamChannel parse_downstream_channel_path(
    std::string_view path, FabricNodeId node, const RouterIdentity& identity, uint32_t vc);

}  // namespace tt::tt_fabric::manifest
