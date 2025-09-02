// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <flatbuffers/flatbuffers.h>
#include "tt_metal/fabric/serialization/logical_port_to_eth_chan.hpp"
#include "logical_eth_chan_table_generated.h"

namespace tt::tt_fabric {

std::vector<uint8_t> serialize_logical_port_to_eth_chan_to_bytes(
    const std::map<std::string, EthChanDescriptor>& logical_port_to_eth_chan) {
    flatbuffers::FlatBufferBuilder builder;

    // Create vector of LogicalPortEntry objects
    std::vector<flatbuffers::Offset<tt::tt_fabric::flatbuffer::LogicalPortEntry>> entries;
    entries.reserve(logical_port_to_eth_chan.size());

    for (const auto& [port_name, eth_chan] : logical_port_to_eth_chan) {
        // Create the port name string
        auto port_name_fb = builder.CreateString(port_name);

        // Create the EthChanDescriptor
        auto eth_channel_fb =
            tt::tt_fabric::flatbuffer::CreateEthChanDescriptor(builder, eth_chan.board_id, eth_chan.chan_id);

        // Create the LogicalPortEntry
        auto entry = tt::tt_fabric::flatbuffer::CreateLogicalPortEntry(builder, port_name_fb, eth_channel_fb);

        entries.push_back(entry);
    }

    // Create vector of entries
    auto entries_vector = builder.CreateVector(entries);

    // Create the root LogicalPortToEthChanTable
    auto table = tt::tt_fabric::flatbuffer::CreateLogicalPortToEthChanTable(builder, entries_vector);

    // Finish the buffer
    builder.Finish(table);

    // Return the serialized data
    return std::vector<uint8_t>(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}

std::map<std::string, EthChanDescriptor> deserialize_logical_port_to_eth_chan_from_bytes(
    const std::vector<uint8_t>& data) {
    std::map<std::string, EthChanDescriptor> result;

    auto verifier = flatbuffers::Verifier(data.data(), data.size());
    if (!tt::tt_fabric::flatbuffer::VerifyLogicalPortToEthChanTableBuffer(verifier)) {
        throw std::runtime_error("Invalid FlatBuffer data for LogicalPortToEthChanTable");
    }

    auto table = tt::tt_fabric::flatbuffer::GetLogicalPortToEthChanTable(data.data());

    // Extract entries into map
    if (table->entries()) {
        for (const auto* entry : *table->entries()) {
            std::string port_name;
            EthChanDescriptor eth_chan;

            if (entry->port_name()) {
                port_name = entry->port_name()->str();
            }

            if (entry->eth_channel()) {
                eth_chan.board_id = entry->eth_channel()->board_id();
                eth_chan.chan_id = entry->eth_channel()->chan_id();
            }

            result[port_name] = eth_chan;
        }
    }

    return result;
}

}  // namespace tt::tt_fabric
