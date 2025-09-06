// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <flatbuffers/flatbuffers.h>
#include "tt_metal/fabric/serialization/connections_table.hpp"
#include "flatbuffers/connections_table_generated.h"

namespace tt::tt_fabric {

std::vector<uint8_t> serialize_connections_table_to_bytes(
    const std::vector<std::tuple<std::pair<uint32_t, std::string>, std::pair<uint32_t, std::string>>>& connections) {
    flatbuffers::FlatBufferBuilder builder;

    // Create vector of ConnectionPair objects
    std::vector<flatbuffers::Offset<tt::tt_fabric::flatbuffer::ConnectionPair>> connection_pairs;
    connection_pairs.reserve(connections.size());

    for (const auto& [first_pair, second_pair] : connections) {
        // Create first endpoint
        auto first_name_fb = builder.CreateString(first_pair.second);
        auto first_endpoint =
            tt::tt_fabric::flatbuffer::CreateConnectionEndpoint(builder, first_pair.first, first_name_fb);

        // Create second endpoint
        auto second_name_fb = builder.CreateString(second_pair.second);
        auto second_endpoint =
            tt::tt_fabric::flatbuffer::CreateConnectionEndpoint(builder, second_pair.first, second_name_fb);

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

std::vector<std::tuple<std::pair<uint32_t, std::string>, std::pair<uint32_t, std::string>>>
deserialize_connections_table_from_bytes(const std::vector<uint8_t>& data) {
    std::vector<std::tuple<std::pair<uint32_t, std::string>, std::pair<uint32_t, std::string>>> result;

    auto verifier = flatbuffers::Verifier(data.data(), data.size());
    if (!tt::tt_fabric::flatbuffer::VerifyConnectionsTableBuffer(verifier)) {
        throw std::runtime_error("Invalid FlatBuffer data for ConnectionsTable");
    }

    auto table = tt::tt_fabric::flatbuffer::GetConnectionsTable(data.data());

    // Extract connection pairs
    if (table->connections()) {
        for (const auto* connection_pair : *table->connections()) {
            std::pair<uint32_t, std::string> first_endpoint;
            std::pair<uint32_t, std::string> second_endpoint;

            if (connection_pair->first()) {
                first_endpoint.first = connection_pair->first()->id();
                if (connection_pair->first()->name()) {
                    first_endpoint.second = connection_pair->first()->name()->str();
                }
            }

            if (connection_pair->second()) {
                second_endpoint.first = connection_pair->second()->id();
                if (connection_pair->second()->name()) {
                    second_endpoint.second = connection_pair->second()->name()->str();
                }
            }

            result.emplace_back(first_endpoint, second_endpoint);
        }
    }

    return result;
}

}  // namespace tt::tt_fabric
