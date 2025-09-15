// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include "tt_metal/distributed/mesh_socket_utils.hpp"

namespace tt::tt_metal::distributed {
// Utility functions to serialize and deserialize SocketPeerDescriptor to/from bytes.
std::vector<uint8_t> serialize_to_bytes(const SocketPeerDescriptor& socket_md);
SocketPeerDescriptor deserialize_from_bytes(const std::vector<uint8_t>& data);

}  // namespace tt::tt_metal::distributed
