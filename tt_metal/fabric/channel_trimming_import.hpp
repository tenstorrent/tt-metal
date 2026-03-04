// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

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

// Per-VC override specification for channel trimming.
// Sender and receiver overrides are independent — specifying only sender overrides
// for a VC will not affect receiver bits, and vice versa.
// When a field has a value, it REPLACES the capture's decision for that VC+direction.
struct ChannelTrimmingVcOverride {
    std::optional<bool> force_enable_all_sender_channels;               // true = enable all senders on this VC
    std::optional<std::vector<size_t>> force_enable_sender_channels;    // VC-relative indices
    std::optional<bool> force_enable_all_receiver_channels;             // true = enable all receivers on this VC
    std::optional<std::vector<size_t>> force_enable_receiver_channels;  // VC-relative indices

    bool has_sender_override() const {
        return force_enable_all_sender_channels.has_value() || force_enable_sender_channels.has_value();
    }
    bool has_receiver_override() const {
        return force_enable_all_receiver_channels.has_value() || force_enable_receiver_channels.has_value();
    }
    bool has_override() const { return has_sender_override() || has_receiver_override(); }
};

// Global overrides that apply across all routers, keyed by VC.
struct ChannelTrimmingGlobalOverrides {
    std::array<ChannelTrimmingVcOverride, builder_config::MAX_NUM_VCS> per_vc = {};

    bool has_any_override() const {
        for (const auto& vc_override : per_vc) {
            if (vc_override.has_override()) {
                return true;
            }
        }
        return false;
    }
};

// Parse a previously exported channel trimming capture YAML and return per-router overrides.
ChannelTrimmingOverrideMap load_channel_trimming_overrides(const std::string& yaml_path);

// Parse a channel trimming global override YAML and return global overrides.
ChannelTrimmingGlobalOverrides load_channel_trimming_global_overrides(const std::string& yaml_path);

// Apply global overrides to a per-router trimming entry using replacement semantics.
// Sender and receiver overrides are applied independently per VC.
// sender_channels_per_vc and receiver_channels_per_vc provide the topology-known channel counts.
void apply_global_overrides_to_entry(
    ChannelTrimmingOverrides& entry,
    const ChannelTrimmingGlobalOverrides& global_overrides,
    const std::array<std::size_t, builder_config::MAX_NUM_VCS>& sender_channels_per_vc,
    const std::array<std::size_t, builder_config::MAX_NUM_VCS>& receiver_channels_per_vc);

}  // namespace tt::tt_fabric
