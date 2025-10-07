// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <memory>
#include <optional>
#include <vector>

namespace tt::tt_metal {

class PhysicalSystemDescriptor;
// Emit PhysicalSystemDescriptor to a text proto file
void emit_physical_system_descriptor_to_text_proto(
    const PhysicalSystemDescriptor& descriptor, const std::optional<std::string>& file_path);
// Serialize PhysicalSystemDescriptor to protobuf binary format (byte vector)
std::vector<uint8_t> serialize_physical_system_descriptor_to_bytes(const PhysicalSystemDescriptor& descriptor);

// Deserialize from protobuf binary format to PhysicalSystemDescriptor (byte vector)
PhysicalSystemDescriptor deserialize_physical_system_descriptor_from_bytes(const std::vector<uint8_t>& data);

}  // namespace tt::tt_metal
