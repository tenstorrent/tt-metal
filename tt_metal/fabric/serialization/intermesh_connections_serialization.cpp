// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <flatbuffers/flatbuffers.h>
#include "tt_metal/fabric/serialization/intermesh_connections_serialization.hpp"
#include "flatbuffers/intermesh_connection_table_generated.h"

namespace tt::tt_fabric {

std::vector<uint8_t> serialize_intermesh_connections_to_bytes(const AnnotatedIntermeshConnections& connections) {
    flatbuffers::FlatBufferBuilder builder;

    // Create vector of ConnectionPair objects
    std::vector<flatbuffers::Offset<tt::tt_fabric::flatbuffer::ConnectionPair>> connection_pairs;
    connection_pairs.reserve(connections.size());

    for (const auto& [first_pair, second_pair] : connections) {
        // Create first endpoint
        auto first_endpoint = tt::tt_fabric::flatbuffer::CreateConnectionEndpoint(
            builder,
            first_pair.first,
            static_cast<uint32_t>(first_pair.second.first),  // port_direction (RoutingDirection)
            first_pair.second.second                         // port_channel
        );

        // Create second endpoint
        auto second_endpoint = tt::tt_fabric::flatbuffer::CreateConnectionEndpoint(
            builder,
            second_pair.first,
            static_cast<uint32_t>(second_pair.second.first),  // port_direction (RoutingDirection)
            second_pair.second.second                         // port_channel
        );

        // Create connection pair
        auto connection_pair =
            tt::tt_fabric::flatbuffer::CreateConnectionPair(builder, first_endpoint, second_endpoint);

        connection_pairs.push_back(connection_pair);
    }

    // Create vector of connection pairs
    auto connections_vector = builder.CreateVector(connection_pairs);

    // Create the root ConnectionsTable
    auto table = tt::tt_fabric::flatbuffer::CreateConnectionsTable(builder, connections_vector);

    // Finish the buffer
    builder.Finish(table);

    // Return the serialized data
    return std::vector<uint8_t>(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}

AnnotatedIntermeshConnections deserialize_intermesh_connections_from_bytes(const std::vector<uint8_t>& data) {
    AnnotatedIntermeshConnections result;

    auto verifier = flatbuffers::Verifier(data.data(), data.size());
    if (!tt::tt_fabric::flatbuffer::VerifyConnectionsTableBuffer(verifier)) {
        throw std::runtime_error("Invalid FlatBuffer data for ConnectionsTable");
    }

    auto table = tt::tt_fabric::flatbuffer::GetConnectionsTable(data.data());

    // Extract connection pairs
    if (table->connections()) {
        for (const auto* connection_pair : *table->connections()) {
            std::pair<uint32_t, port_id_t> first_endpoint;
            std::pair<uint32_t, port_id_t> second_endpoint;

            if (connection_pair->first()) {
                first_endpoint.first = connection_pair->first()->id();
                first_endpoint.second = {
                    static_cast<RoutingDirection>(connection_pair->first()->port_direction()),
                    connection_pair->first()->port_channel()};
            }

            if (connection_pair->second()) {
                second_endpoint.first = connection_pair->second()->id();
                second_endpoint.second = {
                    static_cast<RoutingDirection>(connection_pair->second()->port_direction()),
                    connection_pair->second()->port_channel()};
            }

            result.emplace_back(first_endpoint, second_endpoint);
        }
    }

    return result;
}

}  // namespace tt::tt_fabric
