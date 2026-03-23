// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdint>
#include "tools/scaleout/validation/utils/ethernet_link_metrics.hpp"

namespace tt::scaleout::validation {

std::vector<uint8_t> serialize_link_metrics_to_bytes(const std::vector<::EthernetLinkMetrics>& link_metrics);
std::vector<::EthernetLinkMetrics> deserialize_link_metrics_from_bytes(const std::vector<uint8_t>& data);

std::vector<uint8_t> serialize_eth_chan_identifiers_to_bytes(const std::vector<::EthChannelIdentifier>& exit_nodes);
std::vector<::EthChannelIdentifier> deserialize_eth_chan_identifiers_from_bytes(const std::vector<uint8_t>& data);

}  // namespace tt::scaleout::validation
