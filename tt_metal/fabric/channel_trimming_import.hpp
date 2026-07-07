// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

struct Vc0TrimFastPathInfo {
    bool terminal_or_source_only = false;
    bool worker_only_nonforwarding = false;
    bool terminal_only_nonforwarding = false;
    bool enable_terminal_speedy_rx = false;
};

// Key: pack(chip_id, eth_channel_id) → overrides
using ChannelTrimmingOverrideMap = std::unordered_map<uint64_t, ChannelTrimmingOverrides>;

inline uint64_t make_override_key(ChipId chip_id, chan_id_t eth_chan) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(chip_id)) << 32) | eth_chan;
}

// True only when imported capture YAML contained an explicit row for this router.
// Override-only mode may synthesize a fully-enabled baseline later, but that
// synthetic state must not be treated as trustworthy forwarding metadata.
inline bool has_real_channel_trimming_capture_entry(
    const std::optional<ChannelTrimmingOverrideMap>& capture_overrides, ChipId chip_id, chan_id_t eth_chan) {
    if (!capture_overrides.has_value()) {
        return false;
    }
    return capture_overrides->find(make_override_key(chip_id, eth_chan)) != capture_overrides->end();
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

// Read the optional top-level `preserve_vc0_forwarding` flag from a capture profile YAML.
// When true, VC0 must be kept topology-complete (all VC0 sender/receiver channels serviced
// and the VC0 fast-path collapse suppressed) regardless of what a single capture observed.
// This is required for workloads with data-dependent, multi-hop all-to-all routing (e.g.
// DeepSeek prefill MoE dispatch/combine) whose VC0 forwarding paths one capture run cannot
// fully cover. Defaults to false when the key is absent.
bool load_channel_trimming_preserve_vc0_forwarding(const std::string& yaml_path);

// Parse a channel trimming global override YAML and return global overrides.
ChannelTrimmingGlobalOverrides load_channel_trimming_global_overrides(const std::string& yaml_path);

// Derive trusted trim-aware VC0 fast-path metadata for a single router after overrides have been resolved.
// Returns nullopt when VC0 forwarding capture cannot be trusted for fast-path inference.
std::optional<Vc0TrimFastPathInfo> try_derive_vc0_trim_fast_path_info(
    const ChannelTrimmingOverrides& entry,
    std::size_t actual_sender_channels_vc0,
    const ChannelTrimmingGlobalOverrides& global_overrides);

// Apply global overrides to a per-router trimming entry using replacement semantics.
// Sender and receiver overrides are applied independently per VC.
// sender_channels_per_vc and receiver_channels_per_vc provide the topology-known channel counts.
void apply_global_overrides_to_entry(
    ChannelTrimmingOverrides& entry,
    const ChannelTrimmingGlobalOverrides& global_overrides,
    const std::array<std::size_t, builder_config::MAX_NUM_VCS>& sender_channels_per_vc,
    const std::array<std::size_t, builder_config::MAX_NUM_VCS>& receiver_channels_per_vc);

}  // namespace tt::tt_fabric
