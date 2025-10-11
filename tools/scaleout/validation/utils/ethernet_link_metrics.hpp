// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt_stl/reflection.hpp>

struct TrafficParams {
    uint32_t packet_size_bytes = 0;
    uint32_t data_size = 0;
};

struct LinkStatus {
    tt::tt_metal::EthernetMetrics metrics;
    TrafficParams traffic_params;
    uint32_t num_mismatched_words = 0;
};

struct EthChannelIdentifier {
    std::string host;
    tt::tt_metal::AsicID asic_id;
    tt::tt_metal::TrayID tray_id;
    tt::tt_metal::ASICLocation asic_location;
    uint8_t channel = 0;
};

inline bool operator==(const EthChannelIdentifier& lhs, const EthChannelIdentifier& rhs) {
    return lhs.host == rhs.host && lhs.asic_id == rhs.asic_id && lhs.tray_id == rhs.tray_id &&
           lhs.asic_location == rhs.asic_location && lhs.channel == rhs.channel;
}

namespace std {
template <>
struct hash<EthChannelIdentifier> {
    size_t operator()(const EthChannelIdentifier& identifier) const {
        return ttsl::hash::hash_objects_with_default_seed(
            identifier.host, identifier.asic_id, identifier.tray_id, identifier.asic_location, identifier.channel);
    }
};
}  // namespace std

struct EthernetLinkMetrics {
    EthChannelIdentifier channel_identifier;
    LinkStatus link_status;
};
