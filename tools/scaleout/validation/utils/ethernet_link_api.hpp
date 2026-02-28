// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tt_metal/impl/context/metal_context.hpp"

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
// Blackhole-specific defines (write to mailbox)
// ============================================================================

struct BHEthMsg {
    FWMailboxMsg msg_type;
    std::vector<uint32_t> msg_args;
    std::string log_message;
};

// ============================================================================
// Consolidated helpers (should be arch agnostic)
// ============================================================================

void send_reset_msg_to_links(const std::vector<ResetLink>& links_to_reset);

// ============================================================================
// Wormhole-specific helpers (write to specific L1 addresses)
// ============================================================================

void reset_links_wh(const std::vector<ResetLink>& links_to_reset);

// ============================================================================
// Blackhole-specific helpers (write to mailbox)
// ============================================================================

bool eth_mailbox_ready(ChipId chip_id, uint32_t channel, bool wait_for_ready);

void send_eth_msg(
    ChipId chip_id,
    uint32_t channel,
    FWMailboxMsg msg_type,
    std::vector<uint32_t> args,
    bool wait_for_ready,
    bool wait_for_done);

void send_eth_msg_to_links(const std::vector<ResetLink>& links, BHEthMsg eth_msg);

void reset_links_bh(const std::vector<ResetLink>& links_to_reset);

}  // namespace tt::scaleout_tools
