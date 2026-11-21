// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_slot_states.hpp"

#include <algorithm>
#include <array>
#include <string>

#include <fmt/format.h>

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"

namespace tt::tt_fabric::decode {

namespace {

using json = nlohmann::ordered_json;

// Whether the router has set up its channels by the time it reports `status`: it writes every record of the log
// before it reports STARTED, and every status after STARTED comes later.
bool has_logged_start(EDMStatus status) {
    switch (status) {
        case EDMStatus::STARTED:
        case EDMStatus::REMOTE_HANDSHAKE_COMPLETE:
        case EDMStatus::LOCAL_HANDSHAKE_COMPLETE:
        case EDMStatus::READY_FOR_TRAFFIC:
        case EDMStatus::TERMINATED: return true;
        default: return false;
    }
}

// What the slot states of channels run by one ERISC are read against: its status and its copy of the log.
struct EriscLog {
    std::optional<SlotsUnknownReason> unknown;
    const RouterChannelProgressLog* log = nullptr;
};

// ERISC `erisc`'s status and log, or null when the router has no log on that ERISC (built with the log off).
std::optional<EriscLog> find_erisc_log(const RouterValues& values, uint32_t erisc) {
    const std::string log_path = fmt::format("fields/erisc{}/channel_progress_log", erisc);
    const RegionRead<RouterChannelProgressLog>* log_read =
        values.find_read<RouterChannelProgressLog>(log_path, std::nullopt);
    if (log_read == nullptr && values.find_unread(log_path) == nullptr) {
        return std::nullopt;
    }
    const RegionRead<EDMStatus>* status_read =
        values.find_read<EDMStatus>(fmt::format("fields/erisc{}/edm_status", erisc), std::nullopt);
    if (status_read == nullptr) {
        return EriscLog{.unknown = SlotsUnknownReason::STATUS_UNREAD};
    }
    if (!has_logged_start(status_read->elements.at(0))) {
        return EriscLog{.unknown = SlotsUnknownReason::NOT_STARTED};
    }
    if (log_read == nullptr) {
        return EriscLog{.unknown = SlotsUnknownReason::LOG_UNREAD};
    }
    return EriscLog{.log = &log_read->elements.at(0)};
}

// The index of the slot at `address` in `ring_buffer`, or null when no slot starts there.
std::optional<uint32_t> find_slot_index(const manifest::content::L1& ring_buffer, uint32_t address) {
    if (address < ring_buffer.address || ring_buffer.element_size == 0) {
        return std::nullopt;
    }
    const uint32_t offset = address - ring_buffer.address;
    if (offset % ring_buffer.element_size != 0 || offset / ring_buffer.element_size >= ring_buffer.num_elements) {
        return std::nullopt;
    }
    return offset / ring_buffer.element_size;
}

// The router's flat index of channel `channel` on `vc`, as the kernel indexes its channels and the log its records:
// its index on the VC, after every lower VC's channels.
uint32_t find_flat_index(
    const std::array<uint32_t, builder_config::MAX_NUM_VCS>& channels_per_vc, size_t vc, size_t channel) {
    uint32_t flat = 0;
    for (size_t lower = 0; lower < vc; ++lower) {
        flat += channels_per_vc[lower];
    }
    return flat + static_cast<uint32_t>(channel);
}

SenderSlots find_sender_slots(
    const manifest::SenderChannel& sender,
    uint32_t erisc,
    uint32_t flat_index,
    const EriscLog& erisc_log,
    const StreamCountReads& count_reads) {
    SenderSlots slots;
    if (erisc_log.unknown.has_value()) {
        slots.unknown = erisc_log.unknown;
        return slots;
    }
    const SenderChannelProgressRecord& record = erisc_log.log->sender_channels.at(flat_index);
    slots.next_slot_addr = record.next_slot_addr;
    slots.sent = record.sent;
    slots.completed = record.completed;
    slots.first_level_acked = record.first_level_acked;
    slots.credited_at_first_level_ack =
        erisc < sender.credits_by_erisc.size() && sender.credits_by_erisc[erisc].acked.has_value();

    const manifest::content::L1& ring_buffer = *sender.ring_buffer;
    const StreamCountRead* free_slots = find_field_count_read(sender.fields_by_erisc, "free_slots", count_reads);
    if (free_slots == nullptr) {
        slots.unknown = SlotsUnknownReason::REGISTER_UNREAD;
        return slots;
    }
    slots.free_slots = free_slots->count;
    const std::optional<uint32_t> unsent = find_slots_not_counted(free_slots->count, ring_buffer.num_elements);
    if (!unsent.has_value()) {
        slots.unknown = SlotsUnknownReason::COUNT_OUT_OF_RANGE;
        return slots;
    }
    const std::optional<uint32_t> next_slot = find_slot_index(ring_buffer, record.next_slot_addr);
    if (!next_slot.has_value()) {
        slots.unknown = SlotsUnknownReason::NEXT_SLOT_OUTSIDE_RING;
        return slots;
    }
    // The counters wrap at 2^32, and the router subtracts them as unsigned, as here.
    const uint32_t credited = slots.credited_at_first_level_ack ? record.first_level_acked : record.completed;
    slots.ring = make_slot_ring(
        ring_buffer.num_elements,
        *next_slot,
        *unsent,
        record.sent - credited,
        SlotState::WRITTEN_NOT_SENT,
        SlotState::SENT_AWAITING_CREDIT);
    return slots;
}

ReceiverSlots find_receiver_slots(
    const manifest::Router& router,
    const manifest::ReceiverChannel& receiver,
    size_t vc,
    uint32_t erisc,
    uint32_t flat_index,
    const EriscLog& erisc_log,
    const StreamCountReads& count_reads) {
    ReceiverSlots slots;
    if (erisc_log.unknown.has_value()) {
        slots.unknown = erisc_log.unknown;
        return slots;
    }
    const ReceiverChannelProgressRecord& record = erisc_log.log->receiver_channels.at(flat_index);
    slots.next_index = record.next_index;
    slots.uncredited = record.uncredited;
    slots.packets_ready = record.packets_ready;
    slots.first_level_ack = find_first_level_ack(router, vc, erisc).value_or(false);

    const uint32_t depth = receiver.ring_buffer->num_elements;
    const StreamCountRead* pkts_sent = find_field_count_read(receiver.fields_by_erisc, "pkts_sent", count_reads);
    if (pkts_sent == nullptr) {
        slots.unknown = SlotsUnknownReason::REGISTER_UNREAD;
        return slots;
    }
    slots.pkts_sent = pkts_sent->count;
    if (!is_stream_count_in_range(pkts_sent->count, depth)) {
        slots.unknown = SlotsUnknownReason::COUNT_OUT_OF_RANGE;
        return slots;
    }
    if (record.next_index >= depth) {
        slots.unknown = SlotsUnknownReason::NEXT_SLOT_OUTSIDE_RING;
        return slots;
    }
    // On a first-level-ack VC the ack takes the packet off the register, and packets_ready counts it until it's sent
    // on. Elsewhere packets_ready is only the router's copy of the register.
    const uint32_t arrived = pkts_sent->count + (slots.first_level_ack ? record.packets_ready : 0);
    slots.ring = make_slot_ring(
        depth,
        record.next_index,
        arrived,
        record.uncredited,
        SlotState::ARRIVED_NOT_SENT_ON,
        SlotState::SENT_ON_AWAITING_CREDIT);
    return slots;
}

json make_ring_json(const SlotRing& ring) {
    json states = json::array();
    for (const std::vector<SlotState>& slot : ring.states) {
        json slot_states = json::array();
        for (const SlotState state : slot) {
            slot_states.push_back(manifest::lower_enum_name(state));
        }
        states.push_back(std::move(slot_states));
    }
    json out;
    out["next_slot"] = ring.next_slot;
    out["pending"] = ring.pending;
    out["held"] = ring.held;
    out["overlap"] = ring.overlap;
    out["states"] = std::move(states);
    return out;
}

json make_optional_count_json(std::optional<uint32_t> count) {
    return count.has_value() ? json(*count) : json(nullptr);
}

// Whether the channel's record was read from the log: it was unless the log or the status that says it's from this
// run is missing.
bool is_record_known(const std::optional<SlotsUnknownReason>& unknown) {
    return !unknown.has_value() ||
           (*unknown != SlotsUnknownReason::STATUS_UNREAD && *unknown != SlotsUnknownReason::NOT_STARTED &&
            *unknown != SlotsUnknownReason::LOG_UNREAD);
}

json make_unknown_reason_json(const std::optional<SlotsUnknownReason>& unknown) {
    return unknown.has_value() ? json(manifest::lower_enum_name(*unknown)) : json(nullptr);
}

// `value` when the record is known, else null.
json make_record_json(bool record_known, uint32_t value) { return record_known ? json(value) : json(nullptr); }

}  // namespace

SlotRing make_slot_ring(
    uint32_t depth,
    uint32_t next_slot,
    uint32_t pending,
    uint32_t held,
    SlotState pending_state,
    SlotState held_state) {
    SlotRing ring{.next_slot = next_slot, .states = std::vector<std::vector<SlotState>>(depth)};
    ring.pending = std::min(pending, depth);
    ring.held = std::min(held, depth);
    ring.overlap = ring.pending + ring.held > depth ? ring.pending + ring.held - depth : 0;
    for (uint32_t i = 0; i < ring.pending; ++i) {
        ring.states[(next_slot + i) % depth].push_back(pending_state);
    }
    for (uint32_t i = 1; i <= ring.held; ++i) {
        ring.states[(next_slot + depth - i) % depth].push_back(held_state);
    }
    for (std::vector<SlotState>& slot : ring.states) {
        if (slot.empty()) {
            slot.push_back(SlotState::FREE);
        }
    }
    return ring;
}

std::vector<Finding> find_slot_findings(std::string_view router_path, const RouterSlots& slots) {
    std::vector<Finding> findings;
    const auto add_findings = [&](const std::string& channel_path,
                                  const std::optional<SlotsUnknownReason>& unknown,
                                  const SlotRing& ring,
                                  uint32_t logged_next_slot) {
        if (unknown == SlotsUnknownReason::NEXT_SLOT_OUTSIDE_RING) {
            findings.push_back(Finding{
                .code = FindingCode::PROGRESS_LOG_NEXT_SLOT_OUTSIDE_RING,
                .paths = {channel_path},
                .message = fmt::format(
                    "The log's next slot, {:#x}, isn't a slot of the channel's ring buffer", logged_next_slot)});
        } else if (!unknown.has_value() && ring.overlap > 0) {
            findings.push_back(Finding{
                .code = FindingCode::SLOT_COUNTS_OVER_DEPTH,
                .paths = {channel_path},
                .message = fmt::format(
                    "{} pending and {} held slots come to more than the channel's {}: {} slots are counted as both",
                    ring.pending,
                    ring.held,
                    ring.states.size(),
                    ring.overlap)});
        }
    };
    for (size_t vc = 0; vc < slots.senders.size(); ++vc) {
        for (size_t channel = 0; channel < slots.senders[vc].size(); ++channel) {
            if (const std::optional<SenderSlots>& sender = slots.senders[vc][channel]) {
                add_findings(
                    fmt::format("{}/channels/senders/vc{}/ch{}", router_path, vc, channel),
                    sender->unknown,
                    sender->ring,
                    sender->next_slot_addr);
            }
        }
    }
    for (size_t vc = 0; vc < slots.receivers.size(); ++vc) {
        for (size_t channel = 0; channel < slots.receivers[vc].size(); ++channel) {
            if (const std::optional<ReceiverSlots>& receiver = slots.receivers[vc][channel]) {
                add_findings(
                    fmt::format("{}/channels/receivers/vc{}/ch{}", router_path, vc, channel),
                    receiver->unknown,
                    receiver->ring,
                    receiver->next_index);
            }
        }
    }
    return findings;
}

std::optional<RouterSlots> find_router_slots(const RouterValues& values, const StreamCountReads& count_reads) {
    const manifest::Router& router = *values.get_snapshot().get_router().router;
    std::vector<std::optional<EriscLog>> logs_by_erisc;
    bool has_log = false;
    for (uint32_t erisc = 0; erisc < router.fields_by_erisc.size(); ++erisc) {
        logs_by_erisc.push_back(find_erisc_log(values, erisc));
        has_log = has_log || logs_by_erisc.back().has_value();
    }
    if (!has_log) {
        return std::nullopt;
    }
    // The ERISC that runs a channel's step, and its log. Null when no ERISC runs it, or that ERISC has no log.
    const auto find_channel_log = [&](const std::vector<uint32_t>& serviced_by) -> const EriscLog* {
        if (serviced_by.empty() || serviced_by.front() >= logs_by_erisc.size() ||
            !logs_by_erisc[serviced_by.front()].has_value()) {
            return nullptr;
        }
        return &*logs_by_erisc[serviced_by.front()];
    };

    RouterSlots slots;
    const auto& senders = router.channels.senders;
    slots.senders.resize(senders.size());
    for (size_t vc = 0; vc < senders.size(); ++vc) {
        slots.senders[vc].resize(senders[vc].size());
        for (size_t channel = 0; channel < senders[vc].size(); ++channel) {
            const manifest::SenderChannel& sender = senders[vc][channel];
            const EriscLog* erisc_log = find_channel_log(sender.serviced_by);
            if (!sender.ring_buffer.has_value() || erisc_log == nullptr) {
                continue;
            }
            slots.senders[vc][channel] = find_sender_slots(
                sender,
                sender.serviced_by.front(),
                find_flat_index(router.shape.senders_per_vc, vc, channel),
                *erisc_log,
                count_reads);
        }
    }
    const auto& receivers = router.channels.receivers;
    slots.receivers.resize(receivers.size());
    for (size_t vc = 0; vc < receivers.size(); ++vc) {
        slots.receivers[vc].resize(receivers[vc].size());
        for (size_t channel = 0; channel < receivers[vc].size(); ++channel) {
            const manifest::ReceiverChannel& receiver = receivers[vc][channel];
            const EriscLog* erisc_log = find_channel_log(receiver.serviced_by);
            if (!receiver.ring_buffer.has_value() || erisc_log == nullptr) {
                continue;
            }
            slots.receivers[vc][channel] = find_receiver_slots(
                router,
                receiver,
                vc,
                receiver.serviced_by.front(),
                find_flat_index(router.shape.receivers_per_vc, vc, channel),
                *erisc_log,
                count_reads);
        }
    }
    return slots;
}

json make_sender_slots_json(const std::optional<SenderSlots>& slots) {
    if (!slots.has_value()) {
        return json(nullptr);
    }
    const bool record_known = is_record_known(slots->unknown);
    json out;
    out["unknown"] = make_unknown_reason_json(slots->unknown);
    out["next_slot_addr"] = make_record_json(record_known, slots->next_slot_addr);
    out["sent"] = make_record_json(record_known, slots->sent);
    out["completed"] = make_record_json(record_known, slots->completed);
    out["first_level_acked"] = make_record_json(record_known, slots->first_level_acked);
    out["credited_at"] = slots->credited_at_first_level_ack ? "first_level_ack" : "completion";
    out["free_slots"] = make_optional_count_json(slots->free_slots);
    out["ring"] = slots->unknown.has_value() ? json(nullptr) : make_ring_json(slots->ring);
    return out;
}

json make_receiver_slots_json(const std::optional<ReceiverSlots>& slots) {
    if (!slots.has_value()) {
        return json(nullptr);
    }
    const bool record_known = is_record_known(slots->unknown);
    json out;
    out["unknown"] = make_unknown_reason_json(slots->unknown);
    out["next_index"] = make_record_json(record_known, slots->next_index);
    out["uncredited"] = make_record_json(record_known, slots->uncredited);
    out["packets_ready"] = make_record_json(record_known, slots->packets_ready);
    out["taken_at"] = slots->first_level_ack ? "first_level_ack" : "retire";
    out["pkts_sent"] = make_optional_count_json(slots->pkts_sent);
    out["ring"] = slots->unknown.has_value() ? json(nullptr) : make_ring_json(slots->ring);
    return out;
}

}  // namespace tt::tt_fabric::decode
