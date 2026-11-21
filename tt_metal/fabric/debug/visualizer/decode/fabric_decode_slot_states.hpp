// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Slot states: each channel's slots as the router sees them, from its channel progress log (the channel_progress_log
// field, a RouterChannelProgressLog) and the channel's count register. Only on a router built with the log on.
//
// A sender channel, from its next slot: written by its producer but not sent yet, then free, then sent and still
// held until the remote receiver credits it, ending just before the next slot. A receiver channel, from its next slot:
// arrived and not sent on yet, then free, then sent on but not credited back to the remote sender yet.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_findings.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_stream_registers.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_values.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_fabric::decode {

// One slot's state. The decoded file writes it by lower_enum_name, and lists every value in
// analysis.vocabulary.slot_state.
enum class SlotState : uint8_t {
    FREE,
    // Sender: the producer wrote a packet the router hasn't sent over Ethernet yet.
    WRITTEN_NOT_SENT,
    // Sender: sent over Ethernet, held until the remote receiver credits it.
    SENT_AWAITING_CREDIT,
    // Receiver: a packet arrived that the router hasn't sent on yet (forwarded or delivered).
    ARRIVED_NOT_SENT_ON,
    // Receiver: sent on, not yet credited back to the remote sender.
    SENT_ON_AWAITING_CREDIT,
};

// Why a channel's slot states aren't known, on a router with the log on. Written by lower_enum_name.
enum class SlotsUnknownReason : uint8_t {
    // Capture has no full read of the channel's ERISC's edm_status.
    STATUS_UNREAD,
    // The channel's ERISC hasn't reported EDMStatus::STARTED, so the log holds nothing from this run yet.
    NOT_STARTED,
    // Capture has no full read of the log.
    LOG_UNREAD,
    // Capture has no full read of the channel's count register.
    REGISTER_UNREAD,
    // The count register's count is above the channel's slots.
    COUNT_OUT_OF_RANGE,
    // The log's next slot isn't a slot of the channel's ring buffer.
    NEXT_SLOT_OUTSIDE_RING,
};

// Where a channel's slots stand, around its ring from the next slot.
struct SlotRing {
    // The slot the router works on next: a sender's next slot to send, a receiver's oldest packet not sent on.
    uint32_t next_slot = 0;
    // The slots from next_slot on, by state: a sender's written, not sent; a receiver's arrived, not sent on.
    uint32_t pending = 0;
    // The slots just before next_slot, by state: sent and awaiting credit, on either side. At most the channel's
    // slots; the record it comes from has the count as logged.
    uint32_t held = 0;
    // The slots counted both pending and held, where pending + held comes to more than the channel's slots. Neither
    // count is cut: the overlap is what the log and the register say. On a router that's moving, the two were read
    // apart; on a stopped one, the router's own counts disagree.
    uint32_t overlap = 0;
    // Each slot's states: FREE alone, one of the two, or both pending and held on an overlapping slot.
    std::vector<std::vector<SlotState>> states;
};

// A sender channel's progress as logged and read, and its slots.
struct SenderSlots {
    // Null when the states are known.
    std::optional<SlotsUnknownReason> unknown;
    // The log's record, by the router's flat sender index.
    uint32_t next_slot_addr = 0;
    uint32_t sent = 0;
    uint32_t completed = 0;
    uint32_t first_level_acked = 0;
    // Whether the channel's slots are credited at first-level ack, not at completion.
    bool credited_at_first_level_ack = false;
    // The free-slots register's count.
    std::optional<uint32_t> free_slots;
    SlotRing ring;
};

// A receiver channel's progress as logged and read, and its slots.
struct ReceiverSlots {
    std::optional<SlotsUnknownReason> unknown;
    // The log's record, by the router's flat receiver index.
    uint32_t next_index = 0;
    uint32_t uncredited = 0;
    uint32_t packets_ready = 0;
    // Whether the VC takes packets at first-level ack: the register then counts arrived, not acked, and packets_ready
    // the acked ones not sent on.
    bool first_level_ack = false;
    // The packets-received register's count.
    std::optional<uint32_t> pkts_sent;
    SlotRing ring;
};

// A router's slot states, indexed [vc][channel] as the manifest's channels are. Null where the channel has no slots
// or no ERISC runs it.
struct RouterSlots {
    std::vector<std::vector<std::optional<SenderSlots>>> senders;
    std::vector<std::vector<std::optional<ReceiverSlots>>> receivers;
};

// The slot states of every channel of the router `values` reads, with `count_reads` its count registers' full reads.
// Null when the router was built with the log off.
std::optional<RouterSlots> find_router_slots(const RouterValues& values, const StreamCountReads& count_reads);

// A finding for each channel of the router at `router_path` whose log's next slot isn't in its ring, or whose counts
// came to more than its slots.
std::vector<Finding> find_slot_findings(std::string_view router_path, const RouterSlots& slots);

// The states around a ring of `depth` slots: `pending` slots from `next_slot` on in `pending_state`, and `held` slots
// before it in `held_state`, each at most `depth`. A slot in both ranges has both states.
SlotRing make_slot_ring(
    uint32_t depth,
    uint32_t next_slot,
    uint32_t pending,
    uint32_t held,
    SlotState pending_state,
    SlotState held_state);

// A channel's analysis.slots: null, or why its states aren't known, or its progress, counts and states.
nlohmann::ordered_json make_sender_slots_json(const std::optional<SenderSlots>& slots);
nlohmann::ordered_json make_receiver_slots_json(const std::optional<ReceiverSlots>& slots);

}  // namespace tt::tt_fabric::decode
