// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "physical_system_descriptor_serialization.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "protobuf/physical_system_descriptor.pb.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <fstream>
#include <sstream>

namespace tt::tt_metal {

namespace {

// Helper function to convert BoardType enum to protobuf value
uint32_t board_type_to_proto(BoardType board_type) { return static_cast<uint32_t>(board_type); }

// Helper function to convert protobuf value to BoardType enum
BoardType proto_to_board_type(uint32_t proto_value) { return static_cast<BoardType>(proto_value); }

// Convert EthConnection to protobuf
void eth_connection_to_proto(const EthConnection& eth_conn, tt::fabric::proto::EthConnection* proto_conn) {
    proto_conn->set_src_chan(eth_conn.src_chan);
    proto_conn->set_dst_chan(eth_conn.dst_chan);
    proto_conn->set_is_local(eth_conn.is_local);
}

// Convert protobuf to EthConnection
EthConnection proto_to_eth_connection(const tt::fabric::proto::EthConnection& proto_conn) {
    EthConnection eth_conn;
    eth_conn.src_chan = proto_conn.src_chan();
    eth_conn.dst_chan = proto_conn.dst_chan();
    eth_conn.is_local = proto_conn.is_local();
    return eth_conn;
}

// Convert ExitNodeConnection to protobuf
void exit_node_connection_to_proto(
    const ExitNodeConnection& exit_conn, tt::fabric::proto::ExitNodeConnection* proto_conn) {
    proto_conn->set_src_exit_node(*exit_conn.src_exit_node);
    proto_conn->set_dst_exit_node(*exit_conn.dst_exit_node);
    eth_connection_to_proto(exit_conn.eth_conn, proto_conn->mutable_eth_conn());
}

// Convert protobuf to ExitNodeConnection
ExitNodeConnection proto_to_exit_node_connection(const tt::fabric::proto::ExitNodeConnection& proto_conn) {
    ExitNodeConnection exit_conn;
    exit_conn.src_exit_node = AsicID{proto_conn.src_exit_node()};
    exit_conn.dst_exit_node = AsicID{proto_conn.dst_exit_node()};
    exit_conn.eth_conn = proto_to_eth_connection(proto_conn.eth_conn());
    return exit_conn;
}

// Convert AsicTopology to protobuf
void asic_topology_to_proto(const AsicTopology& topology, tt::fabric::proto::HostAsicConnectivity* host_asic_conn) {
    for (const auto& [asic_id, connections] : topology) {
        auto* asic_graph = host_asic_conn->add_asic_topologies();
        asic_graph->set_asic_id(*asic_id);

        auto* asic_connections = asic_graph->mutable_topology();
        for (const auto& [dst_asic_id, eth_connections] : connections) {
            auto* edge = asic_connections->add_asic_connections();
            edge->set_dst_asic_id(*dst_asic_id);

            for (const auto& eth_conn : eth_connections) {
                eth_connection_to_proto(eth_conn, edge->add_eth_connections());
            }
        }
    }
}

// Convert protobuf to AsicTopology
AsicTopology proto_to_asic_topology(const tt::fabric::proto::HostAsicConnectivity& host_asic_conn) {
    AsicTopology topology;

    for (const auto& asic_graph : host_asic_conn.asic_topologies()) {
        AsicID asic_id{asic_graph.asic_id()};
        std::vector<AsicConnectionEdge>& connections = topology[asic_id];

        for (const auto& edge : asic_graph.topology().asic_connections()) {
            AsicID dst_asic_id{edge.dst_asic_id()};
            std::vector<EthConnection> eth_connections;

            for (const auto& proto_eth_conn : edge.eth_connections()) {
                eth_connections.push_back(proto_to_eth_connection(proto_eth_conn));
            }

            connections.emplace_back(dst_asic_id, std::move(eth_connections));
        }
    }

    return topology;
}

// Convert HostTopology to protobuf
void host_topology_to_proto(const HostTopology& topology, tt::fabric::proto::PhysicalConnectivityGraph* graph) {
    for (const auto& [src_host, connections] : topology) {
        auto* host_conn = graph->add_host_connectivity_graph();
        host_conn->set_src_host_name(src_host);

        for (const auto& [dst_host, exit_connections] : connections) {
            auto* edge = host_conn->add_host_connections();
            edge->set_dst_host_name(dst_host);

            for (const auto& exit_conn : exit_connections) {
                exit_node_connection_to_proto(exit_conn, edge->add_exit_node_connections());
            }
        }
    }
}

// Convert protobuf to HostTopology
HostTopology proto_to_host_topology(const tt::fabric::proto::PhysicalConnectivityGraph& graph) {
    HostTopology topology;

    for (const auto& host_conn : graph.host_connectivity_graph()) {
        const std::string& src_host = host_conn.src_host_name();
        std::vector<HostConnectionEdge>& connections = topology[src_host];

        for (const auto& edge : host_conn.host_connections()) {
            std::string dst_host = edge.dst_host_name();
            std::vector<ExitNodeConnection> exit_connections;

            for (const auto& proto_exit_conn : edge.exit_node_connections()) {
                exit_connections.push_back(proto_to_exit_node_connection(proto_exit_conn));
            }

            connections.emplace_back(dst_host, std::move(exit_connections));
        }
    }

    return topology;
}

// Convert PhysicalSystemDescriptor to protobuf
void physical_system_descriptor_to_proto(
    const PhysicalSystemDescriptor& descriptor, tt::fabric::proto::PhysicalSystemDescriptor* proto_desc) {
    // Convert system graph
    auto* proto_graph = proto_desc->mutable_system_graph();

    // Convert ASIC connectivity graph
    for (const auto& [host_name, asic_topology] : descriptor.get_system_graph().asic_connectivity_graph) {
        auto* host_asic_conn = proto_graph->add_asic_connectivity_graph();
        host_asic_conn->set_host_name(host_name);
        asic_topology_to_proto(asic_topology, host_asic_conn);
    }

    // Convert host connectivity graph
    host_topology_to_proto(descriptor.get_system_graph().host_connectivity_graph, proto_graph);

    // Convert ASIC descriptors
    for (const auto& [asic_id, asic_desc] : descriptor.get_asic_descriptors()) {
        auto* proto_asic_map = proto_desc->add_asic_descriptors();
        proto_asic_map->set_asic_id(*asic_id);

        auto* proto_asic_desc = proto_asic_map->mutable_asic_descriptor();
        proto_asic_desc->set_tray_id(*asic_desc.tray_id);
        proto_asic_desc->set_asic_location(*asic_desc.asic_location);
        proto_asic_desc->set_board_type(board_type_to_proto(asic_desc.board_type));
        proto_asic_desc->set_unique_id(*asic_desc.unique_id);
        proto_asic_desc->set_host_name(asic_desc.host_name);
    }

    // Convert host to mobo name map
    for (const auto& [host_name, mobo_name] : descriptor.get_host_mobo_name_map()) {
        auto* proto_map = proto_desc->add_host_to_mobo_name();
        proto_map->set_host_name(host_name);
        proto_map->set_mobo_name(mobo_name);
    }

    // Convert host to rank map
    for (const auto& [host_name, rank] : descriptor.get_host_to_rank_map()) {
        auto* proto_map = proto_desc->add_host_to_rank();
        proto_map->set_host_name(host_name);
        proto_map->set_rank(rank);
    }

    // Convert exit node connection table
    for (const auto& [host_name, exit_connections] : descriptor.get_exit_node_connection_table()) {
        auto* proto_table = proto_desc->add_exit_node_connection_table();
        proto_table->set_host_name(host_name);

        for (const auto& exit_conn : exit_connections) {
            exit_node_connection_to_proto(exit_conn, proto_table->add_exit_connections());
        }
    }
}

// Convert protobuf to PhysicalSystemDescriptor
std::unique_ptr<PhysicalSystemDescriptor> proto_to_physical_system_descriptor(
    const tt::fabric::proto::PhysicalSystemDescriptor& proto_desc) {
    auto descriptor = std::make_unique<PhysicalSystemDescriptor>(false);  // Don't run discovery

    // Convert system graph
    auto& system_graph = descriptor->get_system_graph();

    // Convert ASIC connectivity graph
    for (const auto& host_asic_conn : proto_desc.system_graph().asic_connectivity_graph()) {
        const std::string& host_name = host_asic_conn.host_name();
        system_graph.asic_connectivity_graph[host_name] = proto_to_asic_topology(host_asic_conn);
    }

    // Convert host connectivity graph
    system_graph.host_connectivity_graph = proto_to_host_topology(proto_desc.system_graph());

    // Convert ASIC descriptors
    auto& asic_descriptors = descriptor->get_asic_descriptors();
    for (const auto& proto_asic_map : proto_desc.asic_descriptors()) {
        AsicID asic_id{proto_asic_map.asic_id()};
        ASICDescriptor asic_desc;

        const auto& proto_asic_desc = proto_asic_map.asic_descriptor();
        asic_desc.tray_id = TrayID{proto_asic_desc.tray_id()};
        asic_desc.asic_location = ASICLocation{proto_asic_desc.asic_location()};
        asic_desc.board_type = proto_to_board_type(proto_asic_desc.board_type());
        asic_desc.unique_id = AsicID{proto_asic_desc.unique_id()};
        asic_desc.host_name = proto_asic_desc.host_name();

        asic_descriptors[asic_id] = asic_desc;
    }

    // Convert host to mobo name map
    auto& host_to_mobo_name = descriptor->get_host_mobo_name_map();
    for (const auto& proto_map : proto_desc.host_to_mobo_name()) {
        host_to_mobo_name[proto_map.host_name()] = proto_map.mobo_name();
    }

    // Convert host to rank map
    auto& host_to_rank = descriptor->get_host_to_rank_map();
    for (const auto& proto_map : proto_desc.host_to_rank()) {
        host_to_rank[proto_map.host_name()] = proto_map.rank();
    }

    // Convert exit node connection table
    auto& exit_node_connection_table = descriptor->get_exit_node_connection_table();
    for (const auto& proto_table : proto_desc.exit_node_connection_table()) {
        std::vector<ExitNodeConnection> exit_connections;

        for (const auto& proto_exit_conn : proto_table.exit_connections()) {
            exit_connections.push_back(proto_to_exit_node_connection(proto_exit_conn));
        }

        exit_node_connection_table[proto_table.host_name()] = std::move(exit_connections);
    }

    return descriptor;
}

}  // namespace

void emit_physical_system_descriptor_to_text_proto(
    const PhysicalSystemDescriptor& descriptor, const std::optional<std::string>& file_path) {
    tt::fabric::proto::PhysicalSystemDescriptor proto_desc;
    physical_system_descriptor_to_proto(descriptor, &proto_desc);

    std::string text_proto;
    google::protobuf::TextFormat::PrintToString(proto_desc, &text_proto);
    if (file_path.has_value()) {
        std::ofstream file(file_path.value());
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + file_path.value());
        }

        file << text_proto;
        file.close();
    } else {
        std::cout << text_proto << std::endl;
    }
}

std::vector<uint8_t> serialize_physical_system_descriptor_to_bytes(const PhysicalSystemDescriptor& descriptor) {
    tt::fabric::proto::PhysicalSystemDescriptor proto_desc;
    physical_system_descriptor_to_proto(descriptor, &proto_desc);

    // Get the serialized size and allocate vector
    size_t size = proto_desc.ByteSizeLong();
    std::vector<uint8_t> result(size);

    // Serialize directly to the vector
    if (!proto_desc.SerializeToArray(result.data(), size)) {
        throw std::runtime_error("Failed to serialize PhysicalSystemDescriptor to protobuf binary format");
    }

    return result;
}

PhysicalSystemDescriptor deserialize_physical_system_descriptor_from_bytes(const std::vector<uint8_t>& data) {
    tt::fabric::proto::PhysicalSystemDescriptor proto_desc;
    if (!proto_desc.ParseFromArray(data.data(), data.size())) {
        throw std::runtime_error("Failed to parse PhysicalSystemDescriptor from protobuf binary format");
    }

    return std::move(*proto_to_physical_system_descriptor(proto_desc));
}

}  // namespace tt::tt_metal
