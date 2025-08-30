// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include "tt_metal/fabric/serialization/physical_desc.hpp"
#include "tt_metal/fabric/serialization/physical_system_descriptor_serialization.hpp"

namespace tt::tt_metal {

std::vector<uint8_t> serialize_physical_descriptor_to_bytes(
    const tt_metal::PhysicalSystemDescriptor& physical_descriptor) {
    // Use the byte vector API directly
    return serialization::serialize_physical_system_descriptor_to_bytes(physical_descriptor);
}

tt_metal::PhysicalSystemDescriptor deserialize_physical_descriptor_from_bytes(const std::vector<uint8_t>& data) {
    // Use the byte vector API directly
    auto descriptor = serialization::deserialize_physical_system_descriptor_from_bytes(data);
    // Move the unique_ptr content to a stack object
    return std::move(*descriptor);
}

}  // namespace tt::tt_metal
