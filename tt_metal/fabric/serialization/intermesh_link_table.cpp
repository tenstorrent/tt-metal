// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include "tt_metal/fabric/serialization/intermesh_link_table.hpp"
#include "intermesh_link_table_generated.h"

namespace tt::tt_fabric {

std::vector<uint8_t> serialize_to_bytes(const IntermeshLinkTable& intermesh_link_table) {
    flatbuffers::FlatBufferBuilder builder;

    auto mesh_id = tt::tt_fabric::flatbuffer::CreateMeshId(builder, *(intermesh_link_table.local_mesh_id));
    auto host_rank_id =
        tt::tt_fabric::flatbuffer::CreateHostRankId(builder, *(intermesh_link_table.local_host_rank_id));

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
        result.local_host_rank_id = HostRankId{intermesh_link_table->local_host_rank_id()->value()};
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

}  // namespace tt::tt_fabric
