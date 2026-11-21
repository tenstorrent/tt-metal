// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host+device compatible header: the layout of the channel progress log, the L1 block a fabric router writes when
// TT_METAL_FABRIC_ENABLE_CHANNEL_PROGRESS_LOG is set. The router copies each channel's progress every time it
// changes.
//
// Every field is its own aligned 32-bit word, written with one store.
// The block has one size for every router, set by the most channels and downstreams any router has. A narrower
// router leaves the records it doesn't use alone. The channel and VC counts are template arguments: the host passes
// builder_config's maximums, and the kernel passes the same maximums as it receives them in its compile-time args
// (MAX_NUM_SENDER_CHANNELS, MAX_NUM_RECEIVER_CHANNELS, MAX_NUM_VCS), so both see one layout.
// 
// Types here partially mirror some of the types in router_sender.hpp and router_receiver.hpp, but are redefined here as
// we only want to log these fields.

#include <array>
#include <cstddef>
#include <cstdint>

#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

// One sender channel's progress.
struct SenderChannelProgressRecord {
    // The slot the channel sends next.
    uint32_t next_slot_addr;
    // Packets sent over the link.
    uint32_t sent;
    // Packets the remote receiver has sent completions for.
    uint32_t completed;
    // Packets the remote receiver has acked (first-level-ack VCs only).
    uint32_t first_level_acked;
};

// One receiver channel's ReceiverChannelProgress (router_receiver.hpp).
struct ReceiverChannelProgressRecord {
    // The slot of the oldest packet that arrived and hasn't been sent on.
    uint32_t next_index;
    // Slots sent on but not yet credited back to the remote sender.
    uint32_t uncredited;
    // Slots acked but not yet sent on (first-level-ack VCs only).
    uint32_t packets_ready;
};

template <size_t NUM_SENDER_CHANNELS, size_t NUM_RECEIVER_CHANNELS, size_t NUM_VCS>
struct ChannelProgressLog {
    // A router forwards to every direction but its own.
    static constexpr size_t max_downstreams_per_forwarding_vc = eth_chan_directions::COUNT - 1;

    // By the router's own sender channel index (the kernel's CH), not the fabric position.
    std::array<SenderChannelProgressRecord, NUM_SENDER_CHANNELS> sender_channels;
    // By receiver channel ID.
    std::array<ReceiverChannelProgressRecord, NUM_RECEIVER_CHANNELS> receiver_channels;
    // Each sender VC's RemoteReceiverChannelProgress word: the free slots it believes the remote (downstream) receiver
    // channel has (low 16 bits) and the next slot it writes there, as address / 16 (high 16 bits).
    std::array<uint32_t, NUM_VCS> sender_pov_receiver_free_slots_per_vc;
    // Each receiver's downstream's DownstreamSenderChannelProgress::next_slot_addr: the slot in that router's sender
    // channel this router's receiver forwards into next. By forwarding VC, then downstream entry in the host's
    // compact order.
    std::array<std::array<uint32_t, max_downstreams_per_forwarding_vc>, num_forwarding_vcs>
        receiver_pov_sender_next_slot_addr_by_vc;
    // The UDM relay's next slot, the same way.
    uint32_t relay_next_slot_addr;
};

// Catch and drift between host and device layouts.
static_assert(sizeof(SenderChannelProgressRecord) == 16);
static_assert(sizeof(ReceiverChannelProgressRecord) == 12);

}  // namespace tt::tt_fabric
