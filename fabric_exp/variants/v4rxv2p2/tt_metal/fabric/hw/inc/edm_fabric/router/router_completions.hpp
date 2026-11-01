// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Completion credits: a receiver slot goes back to the far sender channel its packet came from once every NoC command
// that reads the slot has left the NIU (the far sender may then overwrite it).
//
// The credit frame. The receiver's credits to the far sender are absolute counts per sender channel in this router's L1,
// completions and (first-level-ack VCs) acks, copied to the far sender's L1 as one Ethernet frame; the far sender reads
// both kinds from it. Acks and completions only raise the counts (add_ack, release_completions); the loop sends one frame
// per pass when the counts changed and the TX queue is free (router_receiver.hpp, send_credit_frame), so a pass's acks
// and completions share a frame.
//
// Groups. A receiver ring is credited in groups of slots: a quarter of the ring, at most 4 slots, a power of 2 (rings
// under 8 slots, and first-level-ack VCs, one slot at a time). Each group costs the far sender ERISC one credit return to
// its own upstream instead of one per slot, which saves on steady streams. Groups are aligned (group g is slots g * T to
// g * T + T - 1), so a group never wraps around the ring.
//
// Proof. Every NoC command that reads a slot (the local writes, the forwards) is tagged with its group's transaction ID
// (GroupIds), and the NIU counts the outgoing writes of each ID over every command buffer. So one status read proves a
// whole group free, whatever its slots' commands were. Where a channel has fewer IDs than groups (rings and tori: one
// slot per group, 8 IDs; VC1 builds, 4 IDs per channel: rings over 16 slots; meshes and 1D, 16 IDs: over 64 slots), two
// groups share an ID, and a group's proof also waits for the other's writes: conservative, never wrong.
//
// When a group goes out (release_completions):
//   due    while packets keep arriving: once more than a group waits (the slot just issued is never in it); on a pass
//          without a packet: a full group at once, a partial one after `release_after_idle_passes` such passes. So a
//          channel holds back at most a group minus one of its slots, and only until its next packet or a few idle
//          passes (holds bounded by the ring, LEDGER E1-28: VC1 -12 to -83, no bursty losses);
//   proof  the oldest group's ID (the due slots: its slots from the oldest uncredited on, up to its end). A group of
//          several slots waits for it (bounded) rather than leaving it to the next pass: under load the next pass
//          handles the next packet first, which would delay the credit by a packet (LEDGER E1-24, E1-25: means -1.6 to
//          -4.8). A single slot is checked once (waiting cost ~3 cycles/packet on rings and tori, RESULTS §98 v4sc2);
//   credit one count of n to the slots' sender channel when they all came from one (the usual case), else one per slot.

#include "router_common.hpp"

namespace fabric_router {

// The slots per group of a ring of N slots.
template <uint32_t N>
constexpr uint32_t completion_group_size = N < 8 ? 1 : (N < 16 ? 2 : 4);

// The transaction IDs of a receiver channel's ring: every slot of a group carries the group's ID, one of the channel's
// NUM_IDS IDs from FIRST.
template <uint32_t T, uint32_t FIRST, uint32_t NUM_IDS>
struct GroupIds {
    static_assert(T == 1 || T == 2 || T == 4);
    static_assert((NUM_IDS & (NUM_IDS - 1)) == 0);
    static FORCE_INLINE uint8_t of(uint32_t index) { return static_cast<uint8_t>(FIRST + ((index / T) & (NUM_IDS - 1))); }
};

// The sender channel each slot's packet came from, one byte per slot: byte k of word w is slot 4w + k. A group (of 1, 2
// or 4 aligned slots) lies in one word.
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
static_assert(MAX_NUM_SENDER_CHANNELS <= 0x100);
template <uint32_t N>
struct SlotRecords {
    static constexpr uint32_t num_words = (N + 3) / 4;
    std::array<uint32_t, num_words> words;

    FORCE_INLINE void set(uint32_t index, uint32_t source) {
        reinterpret_cast<uint8_t*>(words.data())[index] = static_cast<uint8_t>(source);
    }

    // The source of slot `index`.
    FORCE_INLINE uint32_t of(uint32_t index) const { return reinterpret_cast<const uint8_t*>(words.data())[index]; }

    // The sources of slots `first` .. `first + n - 1`, all in one group, slot `first` in the low byte.
    FORCE_INLINE uint32_t of_run(uint32_t first, uint32_t n) const {
        const uint32_t run = words[first / 4] >> (8 * (first % 4));
        return n == 4 ? run : run & ((1u << (8 * n)) - 1);
    }
};

// Slot `base + n` of a ring of N slots (n < N).
template <uint32_t N>
FORCE_INLINE uint32_t ring_add(uint32_t base, uint32_t n) {
    const uint32_t i = base + n;
    if constexpr ((N & (N - 1)) == 0) {
        return i & (N - 1);
    } else {
        return i >= N ? i - N : i;
    }
}

// Has every write tagged `trid` left this ERISC's NIU? (Local writes and forwards are on one NoC, so one register.)
static_assert(tt::tt_fabric::local_chip_noc_equals_downstream_noc);
FORCE_INLINE bool writes_left(uint32_t trid) {
    return ncrisc_noc_nonposted_write_with_transaction_id_sent(tt::tt_fabric::edm_to_local_chip_noc, trid);
}

constexpr uint32_t release_after_idle_passes = 4;
constexpr uint32_t completion_wait_checks = 32;

// The completion counters of one receiver channel (loop locals).
struct CompletionCounts {
    uint32_t uncompleted = 0;  // slots retired (NoC commands issued) but not yet credited
    uint32_t idle_passes = 0;  // groups of several slots: passes without a packet while a partial group waits
};

// A slot was retired; `index` is its index in the ring, `source` its packet's sender channel.
template <uint32_t N>
FORCE_INLINE void slot_retired(CompletionCounts& counts, SlotRecords<N>& records, uint32_t index, uint32_t source) {
    records.set(index, source);
    counts.uncompleted++;
}

// One more ack to sender channel `src` in the credit counts (counter credits only; stream-register credits, which send
// each credit on its own, are not brought up).
template <typename CreditSender>
FORCE_INLINE void add_ack(CreditSender& credits, uint8_t src) {
    static_assert(std::is_same_v<CreditSender, ReceiverChannelCounterBasedResponseCreditSender>);
    const uint32_t acks = credits.ack_counters[src] + 1;
    credits.ack_counters[src] = acks;
    credits.ack_counters_base_ptr[src] = acks;
}

// Whether the writes tagged `trid` have left, read up to CHECKS times.
template <uint32_t CHECKS>
FORCE_INLINE bool writes_leave(uint32_t trid) {
    for (uint32_t k = 1; k < CHECKS; k++) {
        if (writes_left(trid)) {
            return true;
        }
    }
    return writes_left(trid);
}

// The oldest uncredited slot of a ring of N slots: `waiting` slots before the next one to retire.
template <uint32_t N>
FORCE_INLINE uint32_t oldest_uncredited(uint32_t next_index, uint32_t waiting) {
    return ring_add<N>(next_index, N - waiting);
}

// Due: whether the oldest uncredited slots go back this pass, and how many (`n`; see the top of this file). The slot
// this pass retired is never among them: its commands were only just issued.
template <uint32_t T, uint32_t N>
FORCE_INLINE bool due(CompletionCounts& counts, uint32_t next_index, bool progress, uint32_t& n) {
    const uint32_t waiting = counts.uncompleted;
    if constexpr (T == 1) {
        // Groups of one slot: one as soon as a slot other than this pass's waits (no partial group, so no idle count).
        n = 1;
        return waiting > static_cast<uint32_t>(progress);
    } else {
        // While packets keep arriving, once more than a group waits; on a pass without a packet, a full group at once
        // and a partial one after release_after_idle_passes such passes.
        bool due;
        if (progress) {
            counts.idle_passes = 0;
            due = waiting > T;
        } else {
            due = waiting >= T || (waiting != 0 && ++counts.idle_passes >= release_after_idle_passes);
        }
        if (!due) {
            return false;
        }
        // The oldest uncredited slot and the rest of its group that has been retired.
        const uint32_t first = oldest_uncredited<N>(next_index, waiting);
        const uint32_t group_end = (first | (T - 1)) + 1;
        const uint32_t to_end = (group_end < N ? group_end : N) - first;
        n = waiting < to_end ? waiting : to_end;
        return true;
    }
}

// Credit: n slots from `first` (one group's) to their sender channels: a single slot's source alone; a group's as one
// count when its slots all came from one sender channel (the usual case), else one per slot.
template <uint32_t T, uint32_t N, typename CreditSender>
FORCE_INLINE void credit_slots(const SlotRecords<N>& records, uint32_t first, uint32_t n, CreditSender& credits) {
    if constexpr (T == 1) {
        credits.accumulate_completion_credit(static_cast<uint8_t>(records.of(first)), 1);
    } else {
        const uint32_t sources = records.of_run(first, n);
        const uint32_t source = sources & 0xFF;
        const uint32_t each_slot = n == 4 ? 0x01010101u : 0x01010101u & ((1u << (8 * n)) - 1);
        if (sources == source * each_slot) {
            credits.accumulate_completion_credit(static_cast<uint8_t>(source), n);
        } else {
            for (uint32_t k = 0; k < n; k++) {
                credits.accumulate_completion_credit(static_cast<uint8_t>((sources >> (8 * k)) & 0xFF), 1);
            }
        }
    }
}

// Credits the oldest uncredited slots when they are due and their writes have left (see the top of this file); returns
// whether it did. The ring has N slots in groups of T, tagged by IDS; `next_index` is the index of the next slot to
// retire; `progress` says whether this pass handled a packet.
template <uint32_t T, uint32_t N, typename Ids, typename CreditSender>
FORCE_INLINE bool release_completions(
    CompletionCounts& counts, const SlotRecords<N>& records, uint32_t next_index, bool progress, CreditSender& credits) {
    static_assert(T >= 1 && T <= 4 && T <= N);
    const uint32_t waiting = counts.uncompleted;
    uint32_t n;
    if (!due<T, N>(counts, next_index, progress, n)) {
        return false;
    }
    // Proof: a single slot is checked once; a group waits for it, bounded.
    const uint32_t first = oldest_uncredited<N>(next_index, waiting);
    if (!writes_leave<T == 1 ? 1 : completion_wait_checks>(Ids::of(first))) {
        return false;
    }
    if constexpr (T > 1) {
        counts.idle_passes = 0;
    }
    credit_slots<T>(records, first, n, credits);
    counts.uncompleted = waiting - n;
    return true;
}

}  // namespace fabric_router
