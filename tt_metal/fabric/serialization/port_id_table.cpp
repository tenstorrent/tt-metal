// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include "tt_metal/fabric/serialization/port_id_table.hpp"
#include "flatbuffers/port_id_table_generated.h"

namespace tt::tt_fabric {

std::vector<uint8_t> serialize_to_bytes(const PortIdTable& port_id_table) {
    flatbuffers::FlatBufferBuilder builder;

    // Create vector of MeshPortIdMap objects
    std::vector<flatbuffers::Offset<tt::tt_fabric::flatbuffer::MeshPortIdMap>> mesh_maps;
    mesh_maps.reserve(port_id_table.size());

    // Iterate through the outer map (source_mesh_id -> inner_map)
    for (const auto& [source_mesh_id, inner_map] : port_id_table) {
        // Create MeshId for source
        auto source_mesh = tt::tt_fabric::flatbuffer::CreateMeshId(builder, *source_mesh_id);

        // Create vector of PortIdEntry objects for this source mesh
        std::vector<flatbuffers::Offset<tt::tt_fabric::flatbuffer::PortIdEntry>> entries;
        entries.reserve(inner_map.size());

        // Iterate through the inner map (dest_mesh_id -> vector<PortIdentifier>)
        for (const auto& [dest_mesh_id, port_identifiers] : inner_map) {
            // Create MeshId for destination
            auto dest_mesh = tt::tt_fabric::flatbuffer::CreateMeshId(builder, *dest_mesh_id);

            // Create MeshIdPair
            auto mesh_pair = tt::tt_fabric::flatbuffer::CreateMeshIdPair(builder, source_mesh, dest_mesh);

            // Create vector of PortIdentifier objects
            std::vector<flatbuffers::Offset<tt::tt_fabric::flatbuffer::PortIdentifier>> port_ids;
            port_ids.reserve(port_identifiers.size());

            for (const auto& port_identifier : port_identifiers) {
                auto port_tag_str = builder.CreateString(port_identifier.port_tag);
                auto port_id = tt::tt_fabric::flatbuffer::CreatePortIdentifier(
                    builder, port_tag_str, port_identifier.connection_hash);
                port_ids.push_back(port_id);
            }

            // Create vector of port identifiers
            auto port_ids_vector = builder.CreateVector(port_ids);

            // Create PortIdEntry with vector of port identifiers
            auto entry = tt::tt_fabric::flatbuffer::CreatePortIdEntry(builder, mesh_pair, port_ids_vector);
            entries.push_back(entry);
        }

        // Create vector of entries
        auto entries_vector = builder.CreateVector(entries);

        // Create MeshPortIdMap
        auto mesh_map = tt::tt_fabric::flatbuffer::CreateMeshPortIdMap(builder, source_mesh, entries_vector);
        mesh_maps.push_back(mesh_map);
    }

    // Create vector of mesh maps
    auto mesh_maps_vector = builder.CreateVector(mesh_maps);

    // Create the root PortIdTable
    auto serialized_port_id_table = tt::tt_fabric::flatbuffer::CreatePortIdTable(builder, mesh_maps_vector);

    // Finish the buffer
    builder.Finish(serialized_port_id_table);

    // Return the serialized data
    return std::vector<uint8_t>(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}

PortIdTable deserialize_port_id_table_from_bytes(const std::vector<uint8_t>& data) {
    PortIdTable result;

    // Verify the buffer
    auto verifier = flatbuffers::Verifier(data.data(), data.size());
    if (!tt::tt_fabric::flatbuffer::VerifyPortIdTableBuffer(verifier)) {
        throw std::runtime_error("Invalid FlatBuffer data for PortIdTable");
    }

    // Get the root PortIdTable
    auto port_id_table = tt::tt_fabric::flatbuffer::GetPortIdTable(data.data());

    // Extract mesh maps
    if (port_id_table->mesh_maps()) {
        for (const auto* mesh_map : *port_id_table->mesh_maps()) {
            if (!mesh_map->source_mesh_id()) {
                continue;  // Skip if source mesh ID is missing
            }

            MeshId source_mesh_id{mesh_map->source_mesh_id()->value()};

            // Create the inner map for this source mesh
            std::unordered_map<MeshId, std::vector<PortIdentifier>> inner_map;

            // Extract entries
            if (mesh_map->entries()) {
                for (const auto* entry : *mesh_map->entries()) {
                    if (!entry->mesh_pair() || !entry->mesh_pair()->dest_mesh_id() || !entry->port_identifiers()) {
                        continue;  // Skip incomplete entries
                    }

                    MeshId dest_mesh_id{entry->mesh_pair()->dest_mesh_id()->value()};

                    // Extract vector of PortIdentifiers
                    std::vector<PortIdentifier> port_identifiers;
                    for (const auto* port_id : *entry->port_identifiers()) {
                        PortIdentifier port_identifier;
                        if (port_id->port_tag()) {
                            port_identifier.port_tag = port_id->port_tag()->str();
                        }
                        port_identifier.connection_hash = port_id->connection_hash();
                        port_identifiers.push_back(port_identifier);
                    }

                    inner_map[dest_mesh_id] = port_identifiers;
                }
            }

            result[source_mesh_id] = inner_map;
        }
    }

    return result;
}

}  // namespace tt::tt_fabric
