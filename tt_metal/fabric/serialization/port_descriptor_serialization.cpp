// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <stdexcept>
#include "tt_metal/fabric/serialization/port_descriptor_serialization.hpp"
#include "protobuf/port_descriptor_table.pb.h"

namespace tt::tt_fabric {

std::vector<uint8_t> serialize_to_bytes(const PortDescriptorTable& port_id_table) {
    tt::fabric::proto::PortDescriptorTable proto_table;

    // Iterate through the outer map (source_mesh_id -> inner_map)
    for (const auto& [source_mesh_id, inner_map] : port_id_table) {
        // Create MeshPortDescriptorMap for this source mesh
        auto* mesh_map = proto_table.add_mesh_maps();

        // Set source mesh ID
        mesh_map->mutable_source_mesh_id()->set_value(*source_mesh_id);

        // Iterate through the inner map (dest_mesh_id -> vector<PortDescriptor>)
        for (const auto& [dest_mesh_id, port_descriptors] : inner_map) {
            // Create PortDescriptorEntry
            auto* entry = mesh_map->add_entries();

            // Set destination mesh ID
            entry->mutable_dest_mesh_id()->set_value(*dest_mesh_id);

            // Add port descriptors
            for (const auto& port_desc : port_descriptors) {
                auto* proto_port_desc = entry->add_port_descriptors();
                proto_port_desc->set_port_direction(static_cast<uint32_t>(port_desc.port_id.first));
                proto_port_desc->set_port_channel(port_desc.port_id.second);
                proto_port_desc->set_connection_hash(port_desc.connection_hash);
            }
        }
    }

    // Serialize to bytes
    size_t size = proto_table.ByteSizeLong();
    std::vector<uint8_t> result(size);

    if (!proto_table.SerializeToArray(result.data(), size)) {
        throw std::runtime_error("Failed to serialize PortDescriptorTable to protobuf binary format");
    }

    return result;
}

PortDescriptorTable deserialize_port_descriptors_from_bytes(const std::vector<uint8_t>& data) {
    PortDescriptorTable result;

    // Parse the protobuf
    tt::fabric::proto::PortDescriptorTable proto_table;
    if (!proto_table.ParseFromArray(data.data(), data.size())) {
        throw std::runtime_error("Failed to parse PortDescriptorTable from protobuf binary format");
    }

    // Extract mesh maps
    for (const auto& mesh_map : proto_table.mesh_maps()) {
        MeshId source_mesh_id{mesh_map.source_mesh_id().value()};

        // Create inner map for this source mesh
        std::unordered_map<MeshId, std::vector<PortDescriptor>> inner_map;

        // Extract entries
        for (const auto& entry : mesh_map.entries()) {
            MeshId dest_mesh_id{entry.dest_mesh_id().value()};

            // Extract port descriptors
            std::vector<PortDescriptor> port_descriptors;
            for (const auto& proto_port_desc : entry.port_descriptors()) {
                PortDescriptor port_desc;
                port_desc.port_id = {
                    static_cast<RoutingDirection>(proto_port_desc.port_direction()), proto_port_desc.port_channel()};
                port_desc.connection_hash = proto_port_desc.connection_hash();
                port_descriptors.push_back(port_desc);
            }

            inner_map[dest_mesh_id] = std::move(port_descriptors);
        }

        result[source_mesh_id] = std::move(inner_map);
    }

    return result;
}

}  // namespace tt::tt_fabric
