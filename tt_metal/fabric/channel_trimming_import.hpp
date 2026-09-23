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
    std::optional<uint16_t> local_sender_max_packet_size_bytes;
    std::optional<uint16_t> peer_sender_max_packet_size_bytes;
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

// Parse a channel trimming global override YAML and return global overrides.
ChannelTrimmingGlobalOverrides load_channel_trimming_global_overrides(const std::string& yaml_path);

// Derive trusted trim-aware VC0 fast-path metadata for a single router after overrides have been resolved.
// Returns nullopt when VC0 forwarding capture cannot be trusted for fast-path inference.
std::optional<Vc0TrimFastPathInfo> try_derive_vc0_trim_fast_path_info(
    const ChannelTrimmingOverrides& entry,
    std::size_t actual_sender_channels_vc0,
    const ChannelTrimmingGlobalOverrides& global_overrides);

// Whether VC0 can use the speedy path after trimming has been resolved.
bool vc0_speedy_path_enabled(
    std::size_t actual_sender_channels_vc0, bool deadlock_avoidance_enabled, const Vc0TrimFastPathInfo& info);

// Propagate sender packet-size metadata across a physical link. Terminal speedy RX
// remains a 2D-only optimization and additionally requires a matching worker-only peer.
void apply_vc0_trim_fast_path_peer_info(
    Vc0TrimFastPathInfo& local_info, const Vc0TrimFastPathInfo& peer_info, bool allow_terminal_speedy_rx);

// Keep the existing packet-count amortization for normal-sized packets, but cap
// the number of packets batched so that a batch does not exceed the same byte
// budget when a trimmed speedy path carries larger packets. Missing packet-size
// metadata (including an observed size of zero) uses the conservative single-packet cadence.
uint32_t limit_credit_amortization_frequency_by_packet_size(
    uint32_t default_frequency,
    uint32_t reference_packet_size_bytes,
    std::optional<uint16_t> max_packet_size_bytes,
    bool log_missing_packet_size_metadata = true);

// Apply global overrides to a per-router trimming entry using replacement semantics.
// Sender and receiver overrides are applied independently per VC.
// sender_channels_per_vc and receiver_channels_per_vc provide the topology-known channel counts.
void apply_global_overrides_to_entry(
    ChannelTrimmingOverrides& entry,
    const ChannelTrimmingGlobalOverrides& global_overrides,
    const std::array<std::size_t, builder_config::MAX_NUM_VCS>& sender_channels_per_vc,
    const std::array<std::size_t, builder_config::MAX_NUM_VCS>& receiver_channels_per_vc);

}  // namespace tt::tt_fabric
