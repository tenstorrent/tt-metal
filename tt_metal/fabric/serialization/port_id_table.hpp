// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <tt-metalium/mesh_graph.hpp>

namespace tt::tt_fabric {

std::vector<uint8_t> serialize_to_bytes(const PortIdTable& port_id_table);
PortIdTable deserialize_port_id_table_from_bytes(const std::vector<uint8_t>& data);

}  // namespace tt::tt_fabric
