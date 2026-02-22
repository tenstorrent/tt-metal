// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include <hostdevcommon/fabric_common.h>                   // chan_id_t

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_trimming_types.hpp"

namespace tt::tt_fabric {

// Reuse the capture data structure for import overrides so export and import are consistent.
using ChannelTrimmingOverrides =
    FabricDatapathUsageL1Results<true, builder_config::MAX_NUM_VCS, builder_config::num_max_sender_channels>;

// Key: pack(chip_id, eth_channel_id) → overrides
using ChannelTrimmingOverrideMap = std::unordered_map<uint64_t, ChannelTrimmingOverrides>;

inline uint64_t make_override_key(ChipId chip_id, chan_id_t eth_chan) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(chip_id)) << 32) | eth_chan;
}

// Parse a previously exported channel trimming capture YAML and return per-router overrides.
ChannelTrimmingOverrideMap load_channel_trimming_overrides(const std::string& yaml_path);

}  // namespace tt::tt_fabric
