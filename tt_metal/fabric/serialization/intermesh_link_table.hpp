// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include <tt_metal/api/tt-metalium/multi_mesh_types.hpp>

namespace tt::tt_fabric {

std::vector<uint8_t> serialize_to_bytes(const IntermeshLinkTable& intermesh_link_table);
IntermeshLinkTable deserialize_from_bytes(const std::vector<uint8_t>& data);

}  // namespace tt::tt_fabric
