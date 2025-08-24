// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/multi_mesh_types.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"

namespace tt::tt_fabric {

std::vector<uint8_t> serialize_to_bytes(const IntermeshLinkTable& intermesh_link_table);
IntermeshLinkTable deserialize_from_bytes(const std::vector<uint8_t>& data);

std::vector<uint8_t> serialize_physical_descriptor_to_bytes(
    const tt_metal::PhysicalSystemDescriptor& physical_descriptor);
tt_metal::PhysicalSystemDescriptor deserialize_physical_descriptor_from_bytes(const std::vector<uint8_t>& data);

}  // namespace tt::tt_fabric
