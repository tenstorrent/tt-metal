// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <memory>
#include <vector>

namespace tt::tt_metal {

class PhysicalSystemDescriptor;

namespace serialization {

// Emit PhysicalSystemDescriptor to a text proto file
void emit_physical_system_descriptor_to_text_proto(
    const PhysicalSystemDescriptor& descriptor, const std::string& file_path);

// Serialize PhysicalSystemDescriptor to protobuf binary format
std::string serialize_physical_system_descriptor(const PhysicalSystemDescriptor& descriptor);

// Serialize PhysicalSystemDescriptor to protobuf binary format (byte vector)
std::vector<uint8_t> serialize_physical_system_descriptor_to_bytes(const PhysicalSystemDescriptor& descriptor);

// Deserialize from protobuf binary format to PhysicalSystemDescriptor
std::unique_ptr<PhysicalSystemDescriptor> deserialize_physical_system_descriptor(const std::string& serialized_data);

// Deserialize from protobuf binary format to PhysicalSystemDescriptor (byte vector)
std::unique_ptr<PhysicalSystemDescriptor> deserialize_physical_system_descriptor_from_bytes(
    const std::vector<uint8_t>& data);

// Load PhysicalSystemDescriptor from a text proto file
std::unique_ptr<PhysicalSystemDescriptor> load_physical_system_descriptor_from_text_proto(const std::string& file_path);

}  // namespace serialization
}  // namespace tt::tt_metal
