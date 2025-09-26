// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <tuple>
#include <string>
#include <utility>
#include "tt_metal/api/tt-metalium/mesh_graph.hpp"

namespace tt::tt_fabric {

/**
 * Serializes AnnotatedIntermeshConnections to Protobuf
 * bytes
 */
std::vector<uint8_t> serialize_intermesh_connections_to_bytes(const AnnotatedIntermeshConnections& connections);

/**
 * Deserializes Protobuf bytes to AnnotatedIntermeshConnections
 */
AnnotatedIntermeshConnections deserialize_intermesh_connections_from_bytes(const std::vector<uint8_t>& data);

}  // namespace tt::tt_fabric
