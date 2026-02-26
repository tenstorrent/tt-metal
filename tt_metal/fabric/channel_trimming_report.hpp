// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include <hostdevcommon/fabric_common.h>                   // chan_id_t
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace tt::tt_fabric {

struct RouterTrimmingStats {
    ChipId chip_id;
    chan_id_t eth_chan;
    uint32_t total_sender_channels;
    uint32_t used_sender_channels;
    uint32_t total_receiver_channels;
    uint32_t used_receiver_channels;

    uint32_t sender_channels_removed() const { return total_sender_channels - used_sender_channels; }
    uint32_t receiver_channels_removed() const { return total_receiver_channels - used_receiver_channels; }
    uint32_t total_channels_removed() const { return sender_channels_removed() + receiver_channels_removed(); }
};

// Parse a channel trimming capture YAML and log a summary report.
// Topology and has_vc1 are used to infer expected channel counts per router direction.
// Summary stats are logged at INFO; histograms at DEBUG.
void generate_and_log_channel_trimming_report(const std::string& yaml_path, Topology topology, bool has_vc1);

}  // namespace tt::tt_fabric
