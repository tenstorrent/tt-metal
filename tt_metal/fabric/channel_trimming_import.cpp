// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "channel_trimming_import.hpp"

#include <limits>
#include <stdexcept>
#include <string>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <yaml-cpp/yaml.h>

#include "tt_metal/fabric/channel_trimming_io.hpp"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"  // NocSendType

namespace tt::tt_fabric {

namespace {

// Map NocSendType string name back to its enum value.
// Returns -1 if the string is not recognized.
int parse_noc_send_type(const std::string& name) {
    if (name == "NOC_UNICAST_WRITE") {
        return NocSendType::NOC_UNICAST_WRITE;
    }
    if (name == "NOC_UNICAST_INLINE_WRITE") {
        return NocSendType::NOC_UNICAST_INLINE_WRITE;
    }
    if (name == "NOC_UNICAST_ATOMIC_INC") {
        return NocSendType::NOC_UNICAST_ATOMIC_INC;
    }
    if (name == "NOC_FUSED_UNICAST_ATOMIC_INC") {
        return NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC;
    }
    if (name == "NOC_UNICAST_SCATTER_WRITE") {
        return NocSendType::NOC_UNICAST_SCATTER_WRITE;
    }
    if (name == "NOC_MULTICAST_WRITE") {
        return NocSendType::NOC_MULTICAST_WRITE;
    }
    if (name == "NOC_MULTICAST_ATOMIC_INC") {
        return NocSendType::NOC_MULTICAST_ATOMIC_INC;
    }
    if (name == "NOC_UNICAST_READ") {
        return NocSendType::NOC_UNICAST_READ;
    }
    return -1;
}

// Import sender_channels sequence into the entry's per-channel arrays and forwarded-to bitfields.
void import_sender_channels(const YAML::Node& chan_data, ChannelTrimmingOverrides& entry) {
    if (!chan_data["sender_channels"]) {
        return;
    }
    for (const auto& sc : chan_data["sender_channels"]) {
        size_t ch = sc["id"].as<size_t>();
        if (ch >= entry.sender_channel_min_packet_size_seen_bytes_by_vc.size()) {
            continue;
        }
        if (sc["min_packet_size_bytes"]) {
            entry.sender_channel_min_packet_size_seen_bytes_by_vc[ch] = sc["min_packet_size_bytes"].as<uint16_t>();
        }
        if (sc["max_packet_size_bytes"]) {
            entry.sender_channel_max_packet_size_seen_bytes_by_vc[ch] = sc["max_packet_size_bytes"].as<uint16_t>();
        }
        if (sc["forwarded_from_receiver_vcs"]) {
            for (const auto& vc_node : sc["forwarded_from_receiver_vcs"]) {
                size_t vc = vc_node.as<size_t>();
                if (vc < entry.sender_channel_forwarded_to_bitfield_by_vc.size()) {
                    entry.sender_channel_forwarded_to_bitfield_by_vc[vc] |= (1u << ch);
                }
            }
        }
    }
}

// Import sender_channel_forwarded_to_by_vc sequence into per-vc forwarding bitfields.
void import_sender_forwarded_to_bitfields(const YAML::Node& chan_data, ChannelTrimmingOverrides& entry) {
    if (!chan_data["sender_channel_forwarded_to_by_vc"]) {
        return;
    }
    for (const auto& vc_entry : chan_data["sender_channel_forwarded_to_by_vc"]) {
        size_t vc = vc_entry["vc"].as<size_t>();
        if (vc >= entry.sender_channel_forwarded_to_bitfield_by_vc.size()) {
            continue;
        }
        if (vc_entry["bitfield"]) {
            entry.sender_channel_forwarded_to_bitfield_by_vc[vc] =
                parse_hex_bitfield(vc_entry["bitfield"].as<std::string>());
        }
    }
}

// Import noc_send_types_by_vc sequence into the entry's per-vc noc send type bitfields.
void import_noc_send_types(const YAML::Node& chan_data, ChannelTrimmingOverrides& entry) {
    if (!chan_data["noc_send_types_by_vc"]) {
        return;
    }
    for (const auto& vc_entry : chan_data["noc_send_types_by_vc"]) {
        size_t vc = vc_entry["vc"].as<size_t>();
        if (vc >= entry.used_noc_send_type_by_vc_bitfield.size()) {
            continue;
        }
        if (vc_entry["types"]) {
            for (const auto& type_node : vc_entry["types"]) {
                int type_val = parse_noc_send_type(type_node.as<std::string>());
                if (type_val >= 0) {
                    entry.used_noc_send_type_by_vc_bitfield[vc] |= (1u << type_val);
                }
            }
        }
    }
}

}  // namespace

ChannelTrimmingOverrideMap load_channel_trimming_overrides(const std::string& yaml_path) {
    log_info(tt::LogFabric, "Loading channel trimming profile from: {}", yaml_path);

    YAML::Node root = YAML::LoadFile(yaml_path);
    TT_FATAL(
        root["channel_trimming_capture"],
        "Trimming profile YAML missing 'channel_trimming_capture' root key: {}",
        yaml_path);

    ChannelTrimmingOverrideMap overrides;
    const auto& capture = root["channel_trimming_capture"];

    auto chip_keys = collect_map_keys(capture);
    for (const auto& chip_key : chip_keys) {
        ChipId chip_id = parse_chip_key(chip_key);
        const auto& channels = capture[chip_key];

        auto chan_keys = collect_map_keys(channels);
        for (const auto& chan_key : chan_keys) {
            chan_id_t eth_chan = parse_eth_channel_key(chan_key);
            const auto& chan_data = channels[chan_key];

            ChannelTrimmingOverrides entry;
            // Match device-side initialization: min packet size starts at max sentinel
            entry.sender_channel_min_packet_size_seen_bytes_by_vc.fill(std::numeric_limits<uint16_t>::max());

            if (chan_data["sender_channel_used_bitfield"]) {
                entry.sender_channel_used_bitfield_by_vc =
                    parse_hex_bitfield(chan_data["sender_channel_used_bitfield"].as<std::string>());
            }

            if (chan_data["receiver_channel_data_forwarded_bitfield"]) {
                entry.receiver_channel_data_forwarded_bitfield_by_vc =
                    parse_hex_bitfield(chan_data["receiver_channel_data_forwarded_bitfield"].as<std::string>());
            }

            import_sender_channels(chan_data, entry);
            import_sender_forwarded_to_bitfields(chan_data, entry);
            import_noc_send_types(chan_data, entry);

            uint64_t key = make_override_key(chip_id, eth_chan);
            overrides[key] = entry;

            log_debug(
                tt::LogFabric,
                "  chip={} eth_chan={}: sender_used=0x{:04X} rx_fwd=0x{:04X}",
                chip_id,
                eth_chan,
                entry.sender_channel_used_bitfield_by_vc,
                entry.receiver_channel_data_forwarded_bitfield_by_vc);
        }
    }

    log_info(tt::LogFabric, "Loaded {} channel trimming overrides from profile", overrides.size());
    return overrides;
}

}  // namespace tt::tt_fabric
