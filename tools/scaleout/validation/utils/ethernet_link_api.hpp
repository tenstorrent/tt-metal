// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/core_coord.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include "tt_metal/llrt/hal.hpp"

namespace tt::scaleout_tools {

using tt::ChipId;
using tt::CoordSystem;
using tt::CoreType;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::FWMailboxMsg;
using tt::tt_metal::PhysicalSystemDescriptor;

struct ResetLink {
    ChipId chip_id;
    uint32_t channel;
    std::string log_message;
};

// ============================================================================
// Blackhole-specific helpers (write to mailbox)
// ============================================================================

// Bring Blackhole ethernet ports DOWN and leave them down (no reinit). The links stay down until
// up_links_bh() / reset_links_bh() is called or the chip is reset (e.g. tt-smi -r).
void down_links_bh(const std::vector<ResetLink>& links_to_reset);

// Bring previously-downed Blackhole ethernet ports back UP by reinitializing MAC/PCS.
void up_links_bh(const std::vector<ResetLink>& links_to_reset);

// UNSAFE: Bring every local Blackhole ethernet link DOWN by writing the port-down message directly
// through a private UMD cluster, WITHOUT acquiring UMD's CHIP_IN_USE lock. This is a deliberate
// fault-injection tool: it lets links be brought down while another process (e.g. a running fabric
// test) already holds the chip. Because two processes then poke the same chip concurrently this is
// inherently racy and must only be used for exercising link recovery, never in normal operation.
// Unlike down_links_bh(), this constructs its own cluster (no MetalContext), enumerates links from
// the SoC descriptor, and only fires the port-down message -- it does not wait for completion or
// reinitialize anything.
void down_links_bh_unsafe();

// UNSAFE: Like down_links_bh_unsafe(), but brings down only ONE endpoint of each physical ethernet link
// instead of both ends, leaving the partner end UP. This is the stimulus for exercising link recovery:
// the FW link-recovery retrain needs a live peer on the far end to complete the training handshake.
// Downing both ends (down_links_bh_unsafe()) makes recovery's retrain time out at signal-detect because
// nothing answers on the wire. Endpoints are enumerated from the cluster's ethernet connection map, which
// is only populated for trained links, so this must be run while links are still up. Same racy,
// CHIP_IN_USE-bypassing caveats as down_links_bh_unsafe(); recovery testing only.
void down_links_bh_single_ended_unsafe();

// ============================================================================
// Consolidated helpers (should be arch agnostic)
// ============================================================================

void send_reset_msg_to_links(const std::vector<ResetLink>& links_to_reset);

// Dump all ethernet peer connections as a JSON object to stdout.
// Uses the same private UMD cluster as the unsafe down path (no CHIP_IN_USE lock needed).
// Keys are "chip/channel"; each entry includes local NOC0 coords and remote chip/channel/NOC0 coords.
void dump_eth_peers_json();

}  // namespace tt::scaleout_tools
