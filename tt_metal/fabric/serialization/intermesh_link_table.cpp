// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include "tt_metal/fabric/serialization/intermesh_link_table.hpp"
#include "intermesh_link_table_generated.h"
#include "system_descriptor_generated.h"

namespace tt::tt_fabric {

std::vector<uint8_t> serialize_to_bytes(const IntermeshLinkTable& intermesh_link_table) {
    flatbuffers::FlatBufferBuilder builder;

    auto mesh_id = tt::tt_fabric::flatbuffer::CreateMeshId(builder, *(intermesh_link_table.local_mesh_id));
    auto host_rank_id =
        tt::tt_fabric::flatbuffer::CreateMeshHostRankId(builder, *(intermesh_link_table.local_host_rank_id));

    // Create vector of EthernetLink objects (flatbuffers dont support std::unordered_map directly)
    std::vector<flatbuffers::Offset<tt::tt_fabric::flatbuffer::EthernetLink>> ethernet_links;
    ethernet_links.reserve(intermesh_link_table.intermesh_links.size());

    for (const auto& [local_chan, remote_chan] : intermesh_link_table.intermesh_links) {
        auto local_eth_chan =
            tt::tt_fabric::flatbuffer::CreateEthChanDescriptor(builder, local_chan.board_id, local_chan.chan_id);

        auto remote_eth_chan =
            tt::tt_fabric::flatbuffer::CreateEthChanDescriptor(builder, remote_chan.board_id, remote_chan.chan_id);

        auto ethernet_link = tt::tt_fabric::flatbuffer::CreateEthernetLink(builder, local_eth_chan, remote_eth_chan);

        ethernet_links.push_back(ethernet_link);
    }

    // Create vector of ethernet links
    auto ethernet_links_vector = builder.CreateVector(ethernet_links);

    // Create the root IntermeshLinkTable
    auto serialized_intermesh_link_table =
        tt::tt_fabric::flatbuffer::CreateIntermeshLinkTable(builder, mesh_id, host_rank_id, ethernet_links_vector);

    // Finish the buffer
    builder.Finish(serialized_intermesh_link_table);

    // Return the serialized data
    return std::vector<uint8_t>(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}

IntermeshLinkTable deserialize_from_bytes(const std::vector<uint8_t>& data) {
    IntermeshLinkTable result;

    auto verifier = flatbuffers::Verifier(data.data(), data.size());
    if (!tt::tt_fabric::flatbuffer::VerifyIntermeshLinkTableBuffer(verifier)) {
        throw std::runtime_error("Invalid FlatBuffer data");
    }

    auto intermesh_link_table = tt::tt_fabric::flatbuffer::GetIntermeshLinkTable(data.data());

    if (intermesh_link_table->local_mesh_id()) {
        result.local_mesh_id = MeshId{intermesh_link_table->local_mesh_id()->value()};
    }

    if (intermesh_link_table->local_host_rank_id()) {
        result.local_host_rank_id = MeshHostRankId{intermesh_link_table->local_host_rank_id()->value()};
    }

    // Extract intermesh links into unordered_map
    if (intermesh_link_table->intermesh_links()) {
        for (const auto* ethernet_link : *intermesh_link_table->intermesh_links()) {
            EthChanDescriptor local_chan;
            EthChanDescriptor remote_chan;

            if (ethernet_link->local_chan()) {
                local_chan.board_id = ethernet_link->local_chan()->board_id();
                local_chan.chan_id = ethernet_link->local_chan()->chan_id();
            }

            if (ethernet_link->remote_chan()) {
                remote_chan.board_id = ethernet_link->remote_chan()->board_id();
                remote_chan.chan_id = ethernet_link->remote_chan()->chan_id();
            }

            result.intermesh_links[local_chan] = remote_chan;
        }
    }

    return result;
}

std::vector<uint8_t> serialize_system_descriptor_to_bytes(const tt::tt_fabric::SystemDescriptor& system_descriptor) {
    flatbuffers::FlatBufferBuilder builder;

    // Helper lambda to create FlatBuffer EthChanDescriptor
    auto create_eth_chan_descriptor = [&](const tt::tt_fabric::EthChanDescriptor& chan_desc) {
        return tt::tt_fabric::flatbuffer::CreateEthChanDescriptor(builder, chan_desc.board_id, chan_desc.chan_id);
    };

    // Helper lambda to create FlatBuffer ASICDescriptor
    auto create_asic_descriptor = [&](const tt::tt_fabric::ASICDescriptor& asic_desc) {
        return tt::tt_fabric::flatbuffer::CreateASICDescriptor(
            builder, asic_desc.unique_id, asic_desc.tray_id, asic_desc.n_id, asic_desc.board_type);
    };

    // Create HostASICGroup vector
    std::vector<flatbuffers::Offset<tt::tt_fabric::flatbuffer::HostASICGroup>> host_asic_groups;
    for (const auto& [host_name, asic_descriptors] : system_descriptor.asic_ids) {
        // Create ASIC descriptors vector for this host
        std::vector<flatbuffers::Offset<tt::tt_fabric::flatbuffer::ASICDescriptor>> asics;
        for (const auto& asic_desc : asic_descriptors) {
            asics.push_back(create_asic_descriptor(asic_desc));
        }

        auto host_name_fb = builder.CreateString(host_name);
        auto asics_vector = builder.CreateVector(asics);
        auto host_asic_group = CreateHostASICGroup(builder, host_name_fb, asics_vector);
        host_asic_groups.push_back(host_asic_group);
    }

    // Create EthConnectivityDescriptor vector
    std::vector<flatbuffers::Offset<tt::tt_fabric::flatbuffer::EthConnectivityDescriptor>> eth_connectivity_descs;
    for (const auto& eth_conn_desc : system_descriptor.eth_connectivity_descs) {
        // Create local ethernet connections
        std::vector<flatbuffers::Offset<tt::tt_fabric::flatbuffer::EthernetLink>> local_eth_connections;
        for (const auto& [local_chan, remote_chan] : eth_conn_desc.local_eth_connections) {
            auto local_chan_fb = create_eth_chan_descriptor(local_chan);
            auto remote_chan_fb = create_eth_chan_descriptor(remote_chan);
            auto ethernet_link = CreateEthernetLink(builder, local_chan_fb, remote_chan_fb);
            local_eth_connections.push_back(ethernet_link);
        }

        // Create remote ethernet connections
        std::vector<flatbuffers::Offset<tt::tt_fabric::flatbuffer::RemoteEthConnection>> remote_eth_connections;
        for (const auto& [local_chan, remote_info] : eth_conn_desc.remote_eth_connections) {
            auto local_chan_fb = create_eth_chan_descriptor(local_chan);
            auto remote_host_fb = builder.CreateString(remote_info.first);
            auto remote_chan_fb = create_eth_chan_descriptor(remote_info.second);
            auto remote_eth_conn = CreateRemoteEthConnection(builder, local_chan_fb, remote_host_fb, remote_chan_fb);
            remote_eth_connections.push_back(remote_eth_conn);
        }

        auto host_name_fb = builder.CreateString(eth_conn_desc.host_name);
        auto local_connections_vector = builder.CreateVector(local_eth_connections);
        auto remote_connections_vector = builder.CreateVector(remote_eth_connections);

        auto eth_connectivity_desc =
            CreateEthConnectivityDescriptor(builder, host_name_fb, local_connections_vector, remote_connections_vector);
        eth_connectivity_descs.push_back(eth_connectivity_desc);
    }

    // Create the root SystemDescriptor
    auto host_asic_groups_vector = builder.CreateVector(host_asic_groups);
    auto eth_connectivity_descs_vector = builder.CreateVector(eth_connectivity_descs);
    auto system_descriptor_fb = CreateSystemDescriptor(builder, host_asic_groups_vector, eth_connectivity_descs_vector);

    builder.Finish(system_descriptor_fb);

    // Return the serialized data
    return std::vector<uint8_t>(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}

// Deserialization function
tt::tt_fabric::SystemDescriptor deserialize_system_descriptor_from_bytes(const std::vector<uint8_t>& data) {
    tt::tt_fabric::SystemDescriptor result;

    auto verifier = flatbuffers::Verifier(data.data(), data.size());
    if (!tt::tt_fabric::flatbuffer::VerifySystemDescriptorBuffer(verifier)) {
        throw std::runtime_error("Invalid FlatBuffer data");
    }

    // Get the root SystemDescriptor from the buffer
    const tt::tt_fabric::flatbuffer::SystemDescriptor* system_desc_fb =
        tt::tt_fabric::flatbuffer::GetSystemDescriptor(data.data());
    // Helper lambda to convert FlatBuffer EthChanDescriptor to C++ struct
    auto convert_eth_chan_descriptor = [](const tt::tt_fabric::flatbuffer::EthChanDescriptor* chan_desc_fb) {
        tt::tt_fabric::EthChanDescriptor chan_desc;
        chan_desc.board_id = chan_desc_fb->board_id();
        chan_desc.chan_id = chan_desc_fb->chan_id();
        return chan_desc;
    };

    // Helper lambda to convert FlatBuffer ASICDescriptor to C++ struct
    auto convert_asic_descriptor = [](const tt::tt_fabric::flatbuffer::ASICDescriptor* asic_desc_fb) {
        tt::tt_fabric::ASICDescriptor asic_desc;
        asic_desc.unique_id = asic_desc_fb->unique_id();
        asic_desc.tray_id = asic_desc_fb->tray_id();
        asic_desc.n_id = asic_desc_fb->n_id();
        asic_desc.board_type = static_cast<BoardType>(asic_desc_fb->board_type());
        return asic_desc;
    };

    // Convert host ASIC groups
    if (system_desc_fb->host_asic_groups()) {
        for (const auto* host_asic_group_fb : *system_desc_fb->host_asic_groups()) {
            std::string host_name = host_asic_group_fb->host_name()->str();
            std::vector<tt::tt_fabric::ASICDescriptor> asic_descriptors;

            if (host_asic_group_fb->asics()) {
                for (const auto* asic_desc_fb : *host_asic_group_fb->asics()) {
                    asic_descriptors.push_back(convert_asic_descriptor(asic_desc_fb));
                }
            }

            result.asic_ids[host_name] = std::move(asic_descriptors);
        }
    }

    // Convert ethernet connectivity descriptors
    if (system_desc_fb->eth_connectivity_descs()) {
        for (const auto* eth_conn_desc_fb : *system_desc_fb->eth_connectivity_descs()) {
            tt::tt_fabric::EthConnectivityDescriptor eth_conn_desc;
            eth_conn_desc.host_name = eth_conn_desc_fb->host_name()->str();

            // Convert local ethernet connections
            if (eth_conn_desc_fb->local_eth_connections()) {
                for (const auto* ethernet_link_fb : *eth_conn_desc_fb->local_eth_connections()) {
                    auto local_chan = convert_eth_chan_descriptor(ethernet_link_fb->local_chan());
                    auto remote_chan = convert_eth_chan_descriptor(ethernet_link_fb->remote_chan());
                    eth_conn_desc.local_eth_connections[local_chan] = remote_chan;
                }
            }

            // Convert remote ethernet connections
            if (eth_conn_desc_fb->remote_eth_connections()) {
                for (const auto* remote_eth_conn_fb : *eth_conn_desc_fb->remote_eth_connections()) {
                    auto local_chan = convert_eth_chan_descriptor(remote_eth_conn_fb->local_chan());
                    std::string remote_host = remote_eth_conn_fb->remote_host()->str();
                    auto remote_chan = convert_eth_chan_descriptor(remote_eth_conn_fb->remote_chan());
                    eth_conn_desc.remote_eth_connections[local_chan] = std::make_pair(remote_host, remote_chan);
                }
            }

            result.eth_connectivity_descs.push_back(std::move(eth_conn_desc));
        }
    }

    return result;
}

}  // namespace tt::tt_fabric
