// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The channel progress log (TT_METAL_FABRIC_ENABLE_CHANNEL_PROGRESS_LOG): with it on, the router copies each channel's
// progress into a block of its L1 (RouterChannelProgressLog, at CHANNEL_PROGRESS_LOG_L1_ADDRESS) every time it changes,
// so the fabric debug visualizer can read every slot's state. The progress itself stays in registers and on the stack,
// where the router keeps it; the block is a copy the router never reads back.
//
// One function per kind of change, called where the change happens. Each one stores each changed field with one
// volatile store to a compile-time address. With the log off, each one is empty, so the call sites carry no code.

#include <cstddef>
#include <cstdint>

#include "router_common.hpp"

namespace fabric_router::progress_log {

namespace detail {

template <uint32_t ADDR>
FORCE_INLINE void store_word(uint32_t value) {
    static_assert(ADDR % sizeof(uint32_t) == 0, "each field is an aligned word");
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ADDR) = value;
}

// Where sender channel CH's record is stored.
template <size_t CH>
constexpr uint32_t sender_record_addr = CHANNEL_PROGRESS_LOG_L1_ADDRESS +
                                        offsetof(RouterChannelProgressLog, sender_channels) +
                                        CH * sizeof(tt::tt_fabric::SenderChannelProgressRecord);

// Sender channel CH's FIELD.
template <size_t CH, uint32_t FIELD_OFFSET>
FORCE_INLINE void store_sender_field(uint32_t value) {
    static_assert(CH < MAX_NUM_SENDER_CHANNELS);
    store_word<sender_record_addr<CH> + FIELD_OFFSET>(value);
}

// Where VC's sender's view of its remote (downstream) receiver channel is stored.
template <size_t VC>
constexpr uint32_t remote_receiver_view_addr =
    CHANNEL_PROGRESS_LOG_L1_ADDRESS + offsetof(RouterChannelProgressLog, sender_pov_receiver_free_slots_per_vc) +
    VC * sizeof(uint32_t);

// Where receiver channel RX's record is stored.
template <size_t RX>
constexpr uint32_t receiver_record_addr = CHANNEL_PROGRESS_LOG_L1_ADDRESS +
                                          offsetof(RouterChannelProgressLog, receiver_channels) +
                                          RX * sizeof(tt::tt_fabric::ReceiverChannelProgressRecord);

// Receiver channel RX's FIELD.
template <size_t RX, uint32_t FIELD_OFFSET>
FORCE_INLINE void store_receiver_field(uint32_t value) {
    static_assert(RX < MAX_NUM_RECEIVER_CHANNELS);
    store_word<receiver_record_addr<RX> + FIELD_OFFSET>(value);
}

// Where forwarding VC's receiver's view of downstream ENTRY's next slot is stored.
template <size_t VC, size_t ENTRY>
constexpr uint32_t downstream_next_slot_addr_addr =
    CHANNEL_PROGRESS_LOG_L1_ADDRESS + offsetof(RouterChannelProgressLog, receiver_pov_sender_next_slot_addr_by_vc) +
    (VC * RouterChannelProgressLog::max_downstreams_per_forwarding_vc + ENTRY) * sizeof(uint32_t);

// Where the receiver's view of the UDM relay's next slot is stored.
constexpr uint32_t relay_next_slot_addr_addr =
    CHANNEL_PROGRESS_LOG_L1_ADDRESS + offsetof(RouterChannelProgressLog, relay_next_slot_addr);

}  // namespace detail

// Sender channel CH's whole record: at init, before the router reports EDMStatus::STARTED.
template <size_t CH>
FORCE_INLINE void log_sender_record(
    uint32_t next_slot_addr, uint32_t sent, uint32_t completed, uint32_t first_level_acked) {
    if constexpr (ENABLE_CHANNEL_PROGRESS_LOG) {
        using Record = tt::tt_fabric::SenderChannelProgressRecord;
        detail::store_sender_field<CH, offsetof(Record, next_slot_addr)>(next_slot_addr);
        detail::store_sender_field<CH, offsetof(Record, sent)>(sent);
        detail::store_sender_field<CH, offsetof(Record, completed)>(completed);
        detail::store_sender_field<CH, offsetof(Record, first_level_acked)>(first_level_acked);
    }
}

// Sender channel CH sent a packet: its next slot and its sent count.
template <size_t CH>
FORCE_INLINE void log_sent(uint32_t next_slot_addr, uint32_t sent) {
    if constexpr (ENABLE_CHANNEL_PROGRESS_LOG) {
        using Record = tt::tt_fabric::SenderChannelProgressRecord;
        detail::store_sender_field<CH, offsetof(Record, next_slot_addr)>(next_slot_addr);
        detail::store_sender_field<CH, offsetof(Record, sent)>(sent);
    }
}

// Sender channel CH took completions: its completed count.
template <size_t CH>
FORCE_INLINE void log_completed(uint32_t completed) {
    if constexpr (ENABLE_CHANNEL_PROGRESS_LOG) {
        using Record = tt::tt_fabric::SenderChannelProgressRecord;
        detail::store_sender_field<CH, offsetof(Record, completed)>(completed);
    }
}

// Sender channel CH took first-level acks: its acked count (first-level-ack VCs only).
template <size_t CH>
FORCE_INLINE void log_first_level_acked(uint32_t first_level_acked) {
    if constexpr (ENABLE_CHANNEL_PROGRESS_LOG) {
        using Record = tt::tt_fabric::SenderChannelProgressRecord;
        detail::store_sender_field<CH, offsetof(Record, first_level_acked)>(first_level_acked);
    }
}

// VC's sender's view of its remote (downstream) receiver channel changed: the RemoteReceiverChannelProgress word.
template <size_t VC>
FORCE_INLINE void log_remote_receiver(uint32_t remote_progress_word) {
    if constexpr (ENABLE_CHANNEL_PROGRESS_LOG) {
        static_assert(VC < MAX_NUM_VCS);
        detail::store_word<detail::remote_receiver_view_addr<VC>>(remote_progress_word);
    }
}

// Receiver channel RX's whole record: at init, before the router reports EDMStatus::STARTED.
template <size_t RX>
FORCE_INLINE void log_receiver_record(uint32_t next_index, uint32_t uncredited, uint32_t packets_ready) {
    if constexpr (ENABLE_CHANNEL_PROGRESS_LOG) {
        using Record = tt::tt_fabric::ReceiverChannelProgressRecord;
        detail::store_receiver_field<RX, offsetof(Record, next_index)>(next_index);
        detail::store_receiver_field<RX, offsetof(Record, uncredited)>(uncredited);
        detail::store_receiver_field<RX, offsetof(Record, packets_ready)>(packets_ready);
    }
}

// Receiver channel RX retired its oldest packet (its NoC commands are issued): its next packet and uncredited count.
template <size_t RX>
FORCE_INLINE void log_retired(uint32_t next_index, uint32_t uncredited) {
    if constexpr (ENABLE_CHANNEL_PROGRESS_LOG) {
        using Record = tt::tt_fabric::ReceiverChannelProgressRecord;
        detail::store_receiver_field<RX, offsetof(Record, next_index)>(next_index);
        detail::store_receiver_field<RX, offsetof(Record, uncredited)>(uncredited);
    }
}

// Receiver channel RX credited slots back to the remote sender: its uncredited count.
template <size_t RX>
FORCE_INLINE void log_uncredited(uint32_t uncredited) {
    if constexpr (ENABLE_CHANNEL_PROGRESS_LOG) {
        using Record = tt::tt_fabric::ReceiverChannelProgressRecord;
        detail::store_receiver_field<RX, offsetof(Record, uncredited)>(uncredited);
    }
}

// Receiver channel RX's acked, not yet retired count changed (first-level-ack VCs only: elsewhere it is a copy of the
// packets-received register, which capture reads).
template <size_t RX>
FORCE_INLINE void log_packets_ready(uint32_t packets_ready) {
    if constexpr (ENABLE_CHANNEL_PROGRESS_LOG) {
        using Record = tt::tt_fabric::ReceiverChannelProgressRecord;
        detail::store_receiver_field<RX, offsetof(Record, packets_ready)>(packets_ready);
    }
}

// Forwarding VC's downstream ENTRY (the host's compact order) has a new next slot: after a forward into it, or at init.
template <size_t VC, size_t ENTRY>
FORCE_INLINE void log_downstream_next_slot(uint32_t next_slot_addr) {
    if constexpr (ENABLE_CHANNEL_PROGRESS_LOG) {
        static_assert(VC < tt::tt_fabric::num_forwarding_vcs);
        static_assert(ENTRY < RouterChannelProgressLog::max_downstreams_per_forwarding_vc);
        detail::store_word<detail::downstream_next_slot_addr_addr<VC, ENTRY>>(next_slot_addr);
    }
}

// The UDM relay has a new next slot: after a forward into it, or at init.
FORCE_INLINE void log_relay_next_slot(uint32_t next_slot_addr) {
    if constexpr (ENABLE_CHANNEL_PROGRESS_LOG) {
        detail::store_word<detail::relay_next_slot_addr_addr>(next_slot_addr);
    }
}

}  // namespace fabric_router::progress_log
