// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <tt-metalium/control_plane.hpp>

namespace tt::tt_fabric {

std::vector<uint8_t> serialize_to_bytes(const PortDescriptorTable& port_id_table);
PortDescriptorTable deserialize_port_descriptors_from_bytes(const std::vector<uint8_t>& data);

}  // namespace tt::tt_fabric
