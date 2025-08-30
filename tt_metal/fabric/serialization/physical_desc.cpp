// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include "tt_metal/fabric/serialization/physical_desc.hpp"
#include "physical_desc_generated.h"

namespace tt::tt_metal {

std::vector<uint8_t> serialize_physical_descriptor_to_bytes(
    const tt_metal::PhysicalSystemDescriptor& physical_descriptor) {
    flatbuffers::FlatBufferBuilder builder;

    // Helper lambda to create EthConnection
    auto create_eth_connection = [&builder](const tt_metal::EthConnection& conn) {
        return tt::tt_metal::flatbuffer::CreateEthConnection(builder, conn.src_chan, conn.dst_chan, conn.is_local);
    };

    // Helper lambda to create ExitNodeConnection
    auto create_exit_node_connection = [&builder, &create_eth_connection](const tt_metal::ExitNodeConnection& conn) {
        auto eth_conn = create_eth_connection(conn.eth_conn);
        return tt::tt_metal::flatbuffer::CreateExitNodeConnection(
            builder, *(conn.src_exit_node), *(conn.dst_exit_node), eth_conn);
    };

    // Serialize ASIC descriptors
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::AsicDescriptorMap>> asic_descriptor_maps;
    for (const auto& [asic_id, descriptor] : physical_descriptor.get_asic_descriptors()) {
        auto asic_desc = tt::tt_metal::flatbuffer::CreateAsicDescriptor(
            builder,
            *(descriptor.tray_id),
            *(descriptor.n_id),
            descriptor.board_type,
            *(descriptor.unique_id),
            builder.CreateString(descriptor.host_name));
        auto asic_desc_map = tt::tt_metal::flatbuffer::CreateAsicDescriptorMap(builder, *asic_id, asic_desc);
        asic_descriptor_maps.push_back(asic_desc_map);
    }

    // Serialize host to motherboard mapping
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::HostToMoboMap>> host_to_mobo_maps;
    for (const auto& [host_name, mobo_name] : physical_descriptor.get_host_mobo_name_map()) {
        auto host_str = builder.CreateString(host_name);
        auto mobo_str = builder.CreateString(mobo_name);
        auto host_mobo_map = tt::tt_metal::flatbuffer::CreateHostToMoboMap(builder, host_str, mobo_str);
        host_to_mobo_maps.push_back(host_mobo_map);
    }

    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::HostToRankMap>> host_to_rank_maps;
    for (const auto& [host_name, rank] : physical_descriptor.get_host_to_rank_map()) {
        auto host_str = builder.CreateString(host_name);
        auto host_to_rank_map = tt::tt_metal::flatbuffer::CreateHostToRankMap(builder, host_str, rank);
        host_to_rank_maps.push_back(host_to_rank_map);
    }

    // Serialize exit node connection table
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::ExitNodeConnectionTable>> exit_node_tables;
    for (const auto& [host_name, connections] : physical_descriptor.get_exit_node_connection_table()) {
        std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::ExitNodeConnection>> exit_connections;
        for (const auto& conn : connections) {
            exit_connections.push_back(create_exit_node_connection(conn));
        }
        auto host_str = builder.CreateString(host_name);
        auto connections_vec = builder.CreateVector(exit_connections);
        auto exit_table = tt::tt_metal::flatbuffer::CreateExitNodeConnectionTable(builder, host_str, connections_vec);
        exit_node_tables.push_back(exit_table);
    }

    // Serialize host deployment descriptors
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::HostDeploymentDescriptor>> host_deployment_descriptors;
    for (const auto& [host_name, descriptor] : physical_descriptor.get_host_deployment_descriptors()) {
        auto host_str = builder.CreateString(host_name);
        auto hall_str = builder.CreateString(*descriptor.hall);
        auto aisle_str = builder.CreateString(*descriptor.aisle);
        auto host_deployment_desc = tt::tt_metal::flatbuffer::CreateHostDeploymentDescriptor(
            builder, host_str, hall_str, aisle_str, *descriptor.rack, *descriptor.shelf_u);
        host_deployment_descriptors.push_back(host_deployment_desc);
    }

    // Serialize ASIC connectivity graph
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::HostAsicConnectivity>> host_asic_connectivities;
    for (const auto& [host_name, asic_topology] : physical_descriptor.get_system_graph().asic_connectivity_graph) {
        std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::AsicGraph>> asic_graphs;

        for (const auto& [asic_id, connection_edges] : asic_topology) {
            std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::AsicConnectionEdge>> asic_edges;

            for (const auto& edge : connection_edges) {
                std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::EthConnection>> eth_connections;
                eth_connections.reserve(edge.second.size());
                for (const auto& eth_conn : edge.second) {
                    eth_connections.push_back(create_eth_connection(eth_conn));
                }
                auto eth_conn_vec = builder.CreateVector(eth_connections);
                auto asic_edge =
                    tt::tt_metal::flatbuffer::CreateAsicConnectionEdge(builder, *(edge.first), eth_conn_vec);
                asic_edges.push_back(asic_edge);
            }

            auto asic_edges_vec = builder.CreateVector(asic_edges);
            auto asic_connections = tt::tt_metal::flatbuffer::CreateAsicConnnections(builder, asic_edges_vec);
            auto asic_graph = tt::tt_metal::flatbuffer::CreateAsicGraph(builder, *asic_id, asic_connections);
            asic_graphs.push_back(asic_graph);
        }

        auto host_str = builder.CreateString(host_name);
        auto asic_graphs_vec = builder.CreateVector(asic_graphs);
        auto host_asic_conn = tt::tt_metal::flatbuffer::CreateHostAsicConnectivity(builder, host_str, asic_graphs_vec);
        host_asic_connectivities.push_back(host_asic_conn);
    }

    // Serialize host connectivity graph
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::HostConnections>> host_connections_vec;
    for (const auto& [host_name, connection_edges] : physical_descriptor.get_system_graph().host_connectivity_graph) {
        std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::HostConnectionEdge>> host_edges;

        for (const auto& edge : connection_edges) {
            std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::ExitNodeConnection>> exit_node_connections;
            exit_node_connections.reserve(edge.second.size());

            for (const auto& exit_conn : edge.second) {
                exit_node_connections.push_back(create_exit_node_connection(exit_conn));
            }
            auto dst_host_str = builder.CreateString(edge.first);
            auto exit_conn_vec = builder.CreateVector(exit_node_connections);
            auto host_edge = tt::tt_metal::flatbuffer::CreateHostConnectionEdge(builder, dst_host_str, exit_conn_vec);
            host_edges.push_back(host_edge);
        }

        auto src_host_str = builder.CreateString(host_name);
        auto host_edges_vec = builder.CreateVector(host_edges);
        auto host_connections = tt::tt_metal::flatbuffer::CreateHostConnections(builder, src_host_str, host_edges_vec);
        host_connections_vec.push_back(host_connections);
    }

    // Create PhysicalConnectivityGraph
    auto host_asic_conn_vec = builder.CreateVector(host_asic_connectivities);
    auto host_conn_vec = builder.CreateVector(host_connections_vec);
    auto connectivity_graph =
        tt::tt_metal::flatbuffer::CreatePhysicalConnectivityGraph(builder, host_asic_conn_vec, host_conn_vec);

    // Create final PhysicalSystemDescriptor
    auto asic_desc_vec = builder.CreateVector(asic_descriptor_maps);
    auto host_mobo_vec = builder.CreateVector(host_to_mobo_maps);
    auto host_rank_vec = builder.CreateVector(host_to_rank_maps);
    auto exit_table_vec = builder.CreateVector(exit_node_tables);

    auto physical_system_desc = tt::tt_metal::flatbuffer::CreatePhysicalSystemDescriptor(
        builder, connectivity_graph, asic_desc_vec, host_mobo_vec, host_rank_vec, exit_table_vec);

    builder.Finish(physical_system_desc);

    // Return the serialized data
    uint8_t* buf = builder.GetBufferPointer();
    int size = builder.GetSize();
    return std::vector<uint8_t>(buf, buf + size);
}

tt_metal::PhysicalSystemDescriptor deserialize_physical_descriptor_from_bytes(const std::vector<uint8_t>& data) {
    // Verify buffer
    flatbuffers::Verifier verifier(data.data(), data.size());
    if (!tt::tt_metal::flatbuffer::VerifyPhysicalSystemDescriptorBuffer(verifier)) {
        throw std::runtime_error("Invalid FlatBuffer data");
    }

    auto fb_desc = tt::tt_metal::flatbuffer::GetPhysicalSystemDescriptor(data.data());
    tt_metal::PhysicalSystemDescriptor result(false);

    // Deserialize ASIC descriptors
    if (fb_desc->asic_descriptors()) {
        for (auto fb_asic_desc : *fb_desc->asic_descriptors()) {
            tt_metal::ASICDescriptor desc;
            desc.tray_id = TrayID{fb_asic_desc->descriptor()->tray_id()};
            desc.n_id = NID{fb_asic_desc->descriptor()->n_id()};
            desc.board_type = static_cast<BoardType>(fb_asic_desc->descriptor()->board_type());
            desc.unique_id = AsicID{fb_asic_desc->descriptor()->unique_id()};
            desc.host_name = fb_asic_desc->descriptor()->host_name()->str();
            result.get_asic_descriptors()[AsicID{fb_asic_desc->asic_id()}] = desc;
        }
    }

    // Deserialize host to motherboard mapping
    if (fb_desc->host_to_mobo_name()) {
        for (auto fb_host_mobo : *fb_desc->host_to_mobo_name()) {
            result.get_host_mobo_name_map()[fb_host_mobo->host_name()->str()] = fb_host_mobo->mobo_name()->str();
        }
    }
    if (fb_desc->host_to_rank()) {
        for (auto fb_host_rank : *fb_desc->host_to_rank()) {
            result.get_host_to_rank_map()[fb_host_rank->host_name()->str()] = fb_host_rank->rank();
        }
    }
    // Deserialize exit node connection table
    if (fb_desc->exit_node_connection_table()) {
        for (auto fb_exit_table : *fb_desc->exit_node_connection_table()) {
            std::vector<tt_metal::ExitNodeConnection> connections;
            if (fb_exit_table->exit_connections()) {
                for (auto fb_exit_conn : *fb_exit_table->exit_connections()) {
                    tt_metal::ExitNodeConnection conn;
                    conn.src_exit_node = AsicID{fb_exit_conn->src_exit_node()};
                    conn.dst_exit_node = AsicID{fb_exit_conn->dst_exit_node()};
                    conn.eth_conn.src_chan = fb_exit_conn->eth_conn()->src_chan();
                    conn.eth_conn.dst_chan = fb_exit_conn->eth_conn()->dst_chan();
                    connections.push_back(conn);
                }
            }
            result.get_exit_node_connection_table()[fb_exit_table->host_name()->str()] = connections;
        }
    }
    // Deserialize system graph
    if (fb_desc->system_graph()) {
        // Deserialize ASIC connectivity graph
        if (fb_desc->system_graph()->asic_connectivity_graph()) {
            for (auto fb_host_asic : *fb_desc->system_graph()->asic_connectivity_graph()) {
                tt_metal::AsicTopology asic_topology;

                if (fb_host_asic->asic_topologies()) {
                    for (auto fb_asic_graph : *fb_host_asic->asic_topologies()) {
                        std::vector<tt_metal::AsicConnectionEdge> connection_edges;

                        if (fb_asic_graph->topology() && fb_asic_graph->topology()->asic_connections()) {
                            for (auto fb_asic_edge : *fb_asic_graph->topology()->asic_connections()) {
                                std::vector<tt_metal::EthConnection> eth_connections;

                                if (fb_asic_edge->eth_connections()) {
                                    for (auto fb_eth_conn : *fb_asic_edge->eth_connections()) {
                                        tt_metal::EthConnection eth_conn;
                                        eth_conn.src_chan = fb_eth_conn->src_chan();
                                        eth_conn.dst_chan = fb_eth_conn->dst_chan();
                                        eth_conn.is_local = fb_eth_conn->is_local();
                                        eth_connections.push_back(eth_conn);
                                    }
                                }

                                connection_edges.emplace_back(fb_asic_edge->dst_asic_id(), eth_connections);
                            }
                        }

                        asic_topology[AsicID{fb_asic_graph->asic_id()}] = connection_edges;
                    }
                }

                result.get_system_graph().asic_connectivity_graph[fb_host_asic->host_name()->str()] = asic_topology;
            }
        }
        // Deserialize host connectivity graph
        if (fb_desc->system_graph()->host_connectivity_graph()) {
            for (auto fb_host_conn : *fb_desc->system_graph()->host_connectivity_graph()) {
                std::vector<tt_metal::HostConnectionEdge> host_edges;

                if (fb_host_conn->host_connections()) {
                    for (auto fb_host_edge : *fb_host_conn->host_connections()) {
                        std::vector<tt_metal::ExitNodeConnection> exit_node_connections;

                        if (fb_host_edge->exit_node_connections()) {
                            for (auto fb_exit_conn : *fb_host_edge->exit_node_connections()) {
                                tt_metal::ExitNodeConnection exit_conn;
                                exit_conn.src_exit_node = AsicID{fb_exit_conn->src_exit_node()};
                                exit_conn.dst_exit_node = AsicID{fb_exit_conn->dst_exit_node()};
                                exit_conn.eth_conn.src_chan = fb_exit_conn->eth_conn()->src_chan();
                                exit_conn.eth_conn.dst_chan = fb_exit_conn->eth_conn()->dst_chan();
                                exit_node_connections.push_back(exit_conn);
                            }
                        }

                        host_edges.emplace_back(fb_host_edge->dst_host_name()->str(), exit_node_connections);
                    }
                }

                result.get_system_graph().host_connectivity_graph[fb_host_conn->src_host_name()->str()] = host_edges;
            }
        }
    }
    // Deserialize host deployment descriptors
    if (fb_desc->host_deployment_descriptors()) {
        for (auto fb_host_desc : *fb_desc->host_deployment_descriptors()) {
            HostDeploymentDescriptor desc{
                .hall = HallID{fb_host_desc->hall()->str()},
                .aisle = AisleID{fb_host_desc->aisle()->str()},
                .rack = RackID{fb_host_desc->rack()},
                .shelf_u = UID{fb_host_desc->shelf_u()}};
            result.get_host_deployment_descriptors()[fb_host_desc->host_name()->str()] = desc;
        }
    }
    return result;
}

}  // namespace tt::tt_metal
