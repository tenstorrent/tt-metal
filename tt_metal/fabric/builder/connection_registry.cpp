// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "connection_registry.hpp"
#include <algorithm>

namespace tt::tt_fabric {

void ConnectionRegistry::record_connection(const RouterConnectionRecord& record) {
    connections_.push_back(record);
}

const std::vector<RouterConnectionRecord>& ConnectionRegistry::get_all_connections() const {
    return connections_;
}

std::vector<RouterConnectionRecord> ConnectionRegistry::get_connections_from_source(
    FabricNodeId source_node,
    RoutingDirection source_direction) const {
    std::vector<RouterConnectionRecord> result;
    std::copy_if(
        connections_.begin(),
        connections_.end(),
        std::back_inserter(result),
        [&](const RouterConnectionRecord& record) {
            return record.source_node == source_node && record.source_direction == source_direction;
        });
    return result;
}

std::vector<RouterConnectionRecord> ConnectionRegistry::get_connections_to_dest(
    FabricNodeId dest_node,
    RoutingDirection dest_direction) const {
    std::vector<RouterConnectionRecord> result;
    std::copy_if(
        connections_.begin(),
        connections_.end(),
        std::back_inserter(result),
        [&](const RouterConnectionRecord& record) {
            return record.dest_node == dest_node && record.dest_direction == dest_direction;
        });
    return result;
}

std::vector<RouterConnectionRecord> ConnectionRegistry::get_connections_by_type(ConnectionType type) const {
    std::vector<RouterConnectionRecord> result;
    std::copy_if(
        connections_.begin(),
        connections_.end(),
        std::back_inserter(result),
        [&](const RouterConnectionRecord& record) {
            return record.connection_type == type;
        });
    return result;
}

std::vector<RouterConnectionRecord> ConnectionRegistry::get_connections_by_source_node(FabricNodeId source_node) const {
    std::vector<RouterConnectionRecord> result;
    std::copy_if(
        connections_.begin(),
        connections_.end(),
        std::back_inserter(result),
        [&](const RouterConnectionRecord& record) {
            return record.source_node == source_node;
        });
    return result;
}

std::vector<RouterConnectionRecord> ConnectionRegistry::get_connections_by_dest_node(FabricNodeId dest_node) const {
    std::vector<RouterConnectionRecord> result;
    std::copy_if(
        connections_.begin(),
        connections_.end(),
        std::back_inserter(result),
        [&](const RouterConnectionRecord& record) {
            return record.dest_node == dest_node;
        });
    return result;
}

void ConnectionRegistry::clear() {
    connections_.clear();
}

size_t ConnectionRegistry::size() const {
    return connections_.size();
}

}  // namespace tt::tt_fabric
