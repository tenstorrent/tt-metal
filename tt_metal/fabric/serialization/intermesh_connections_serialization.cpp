// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <stdexcept>
#include "tt_metal/fabric/serialization/intermesh_connections_serialization.hpp"
#include "protobuf/intermesh_connection_table.pb.h"

namespace tt::tt_fabric {

std::vector<uint8_t> serialize_intermesh_connections_to_bytes(const AnnotatedIntermeshConnections& connections) {
    tt::fabric::proto::ConnectionsTable proto_table;

    // Create connection pairs
    for (const auto& [first_pair, second_pair] : connections) {
        auto* connection_pair = proto_table.add_connections();

        // Set first endpoint
        auto* first_endpoint = connection_pair->mutable_first();
        first_endpoint->set_id(first_pair.first);
        first_endpoint->set_port_direction(static_cast<uint32_t>(first_pair.second.first));  // RoutingDirection
        first_endpoint->set_port_channel(first_pair.second.second);                          // Channel

        // Set second endpoint
        auto* second_endpoint = connection_pair->mutable_second();
        second_endpoint->set_id(second_pair.first);
        second_endpoint->set_port_direction(static_cast<uint32_t>(second_pair.second.first));  // RoutingDirection
        second_endpoint->set_port_channel(second_pair.second.second);                          // Channel
    }

    // Serialize to bytes
    size_t size = proto_table.ByteSizeLong();
    std::vector<uint8_t> result(size);

    if (!proto_table.SerializeToArray(result.data(), size)) {
        throw std::runtime_error("Failed to serialize ConnectionsTable to protobuf binary format");
    }

    return result;
}

AnnotatedIntermeshConnections deserialize_intermesh_connections_from_bytes(const std::vector<uint8_t>& data) {
    AnnotatedIntermeshConnections result;

    // Parse the protobuf
    tt::fabric::proto::ConnectionsTable proto_table;
    if (!proto_table.ParseFromArray(data.data(), data.size())) {
        throw std::runtime_error("Failed to parse ConnectionsTable from protobuf binary format");
    }

    // Extract connection pairs
    for (const auto& connection_pair : proto_table.connections()) {
        std::pair<uint32_t, port_id_t> first_endpoint;
        std::pair<uint32_t, port_id_t> second_endpoint;

        // Extract first endpoint
        if (connection_pair.has_first()) {
            first_endpoint.first = connection_pair.first().id();
            first_endpoint.second = {
                static_cast<RoutingDirection>(connection_pair.first().port_direction()),
                connection_pair.first().port_channel()};
        }

        // Extract second endpoint
        if (connection_pair.has_second()) {
            second_endpoint.first = connection_pair.second().id();
            second_endpoint.second = {
                static_cast<RoutingDirection>(connection_pair.second().port_direction()),
                connection_pair.second().port_channel()};
        }

        result.emplace_back(first_endpoint, second_endpoint);
    }

    return result;
}

}  // namespace tt::tt_fabric
