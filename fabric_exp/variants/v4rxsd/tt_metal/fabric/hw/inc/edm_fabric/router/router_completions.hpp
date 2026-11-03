// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Completion credits: a receiver slot goes back to the far sender channel its packet came from once every NoC command
// that reads the slot has left the NIU (the far sender may then overwrite it).
//
// - Credit frame: the credits are absolute counts per sender channel in this router's L1 (completions, and acks on
//   first-level-ack VCs), copied to the far sender's L1 as one Ethernet frame. add_ack and release_completions only
//   raise the counts; the receiver loop sends one frame per pass when they changed (send_credit_frame).
// - Groups: a ring is credited in aligned groups of slots (completion_group_size), so the far sender returns one credit
//   to its own upstream per group instead of one per slot. An aligned group never wraps around the ring.
// - Proof: every NoC command that reads a slot carries its group's transaction ID (GroupIds), and the NIU counts each
//   ID's outgoing writes over every command buffer, so one status read proves a whole group free. Where a channel has
//   fewer IDs than groups (rings and tori: one slot per group, 8 IDs; VC1 builds, 4 IDs per channel: rings over 16
//   slots; meshes and 1D, 16 IDs: over 64 slots), two groups share an ID and a group's proof also waits for the
//   other's writes: conservative, never wrong.
//
// When a group goes out (release_completions):
//   due     while packets keep arriving, once more than a group waits (the slot retired this pass is never in it); on
//           a pass without a packet, a full group at once, a partial one after release_after_idle_passes such passes;
//   proof   the oldest group's transaction ID: a single slot is checked once, a group waits for it (bounded);
//   credit  one count of n to the slots' sender channel when they all came from one (usual), else one per slot.
//
// Why (measured; keep unless re-measured):
//   - a channel holds back at most a group minus one of its slots, and only until its next packet or a few idle passes
//     (LEDGER E1-28: VC1 -12 to -83, no bursty losses);
//   - a group waits for its proof rather than leaving it to the next pass, which under load handles the next packet
//     first and delays the credit by a packet (LEDGER E1-24, E1-25);
//   - a single slot is checked once: waiting costs ~3 cycles/packet on rings and tori (RESULTS §98 v4sc2).

#include "router_common.hpp"

namespace fabric_router {

// ---------------------------------------------------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------------------------------------------------

// Passes without a packet after which a partial group goes out.
constexpr uint32_t release_after_idle_passes = 4;
// Status reads a group's proof waits for, at most; then its credit is left to a later pass.
constexpr uint32_t completion_wait_checks = 32;

// ---------------------------------------------------------------------------------------------------------------------
// Ring arithmetic
// ---------------------------------------------------------------------------------------------------------------------

// Ring index `index + n` in a ring of NUM_SLOTS slots (n < NUM_SLOTS).
template <uint32_t NUM_SLOTS>
FORCE_INLINE uint32_t ring_add(uint32_t index, uint32_t n) {
    const uint32_t sum = index + n;
    if constexpr (is_power_of_2(NUM_SLOTS)) {
        return sum & (NUM_SLOTS - 1);
    } else {
        return sum >= NUM_SLOTS ? sum - NUM_SLOTS : sum;
    }
}

// The slots per completion group of a ring of NUM_SLOTS slots: a quarter of the ring, at most 4, a power of 2.
template <uint32_t NUM_SLOTS>
constexpr uint32_t completion_group_size = NUM_SLOTS < 8 ? 1 : (NUM_SLOTS < 16 ? 2 : 4);

// The group of slot `index`.
template <uint32_t GROUP_SIZE>
FORCE_INLINE uint32_t group_of(uint32_t index) {
    return index / GROUP_SIZE;
}

// The index just past the group of slot `index` (its last slot + 1; groups are aligned).
template <uint32_t GROUP_SIZE>
FORCE_INLINE uint32_t group_end(uint32_t index) {
    return (index | (GROUP_SIZE - 1)) + 1;
}

// The oldest uncredited slot: `waiting` slots before the next one to retire.
template <uint32_t NUM_SLOTS>
FORCE_INLINE uint32_t oldest_uncredited(uint32_t next_index, uint32_t waiting) {
    return ring_add<NUM_SLOTS>(next_index, NUM_SLOTS - waiting);
}

// ---------------------------------------------------------------------------------------------------------------------
// Transaction IDs
// ---------------------------------------------------------------------------------------------------------------------

// The transaction IDs of a receiver ring: every slot of a group carries its group's ID, one of NUM_IDS IDs from
// FIRST_ID (groups past NUM_IDS reuse them in order).
template <uint32_t GROUP_SIZE, uint32_t FIRST_ID, uint32_t NUM_IDS>
struct GroupIds {
    static_assert(GROUP_SIZE == 1 || GROUP_SIZE == 2 || GROUP_SIZE == 4);
    static_assert(is_power_of_2(NUM_IDS));
    static FORCE_INLINE uint8_t of(uint32_t index) {
        return static_cast<uint8_t>(FIRST_ID + group_of<GROUP_SIZE>(index) % NUM_IDS);
    }
};

// Whether every write tagged `trid` has left this ERISC's NIU (local writes and forwards share one NoC: one register).
static_assert(tt::tt_fabric::local_chip_noc_equals_downstream_noc);
FORCE_INLINE bool writes_left(uint32_t trid) {
    return ncrisc_noc_nonposted_write_with_transaction_id_sent(tt::tt_fabric::edm_to_local_chip_noc, trid);
}

// Whether the writes tagged `trid` have left, reading the status up to MAX_CHECKS times.
template <uint32_t MAX_CHECKS>
FORCE_INLINE bool writes_left_within(uint32_t trid) {
    for (uint32_t check = 1; check < MAX_CHECKS; check++) {
        if (writes_left(trid)) {
            return true;
        }
    }
    return writes_left(trid);
}

// ---------------------------------------------------------------------------------------------------------------------
// Slot records: each slot's sender channel
// ---------------------------------------------------------------------------------------------------------------------

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
static_assert(MAX_NUM_SENDER_CHANNELS <= 0x100);  // a sender channel fits in a byte

// The low n bytes of `word` (n <= 4), the others 0.
FORCE_INLINE uint32_t low_bytes(uint32_t word, uint32_t n) {
    return n == 4 ? word : word & ((1u << (8 * n)) - 1);
}

// The sender channel each slot's packet came from, one byte per slot: byte k of word w is slot 4w + k, so an aligned
// group (1, 2 or 4 slots) lies in one word.
template <uint32_t NUM_SLOTS>
struct SlotRecords {
    static constexpr uint32_t slots_per_word = 4;
    static constexpr uint32_t num_words = (NUM_SLOTS + slots_per_word - 1) / slots_per_word;
    std::array<uint32_t, num_words> words;

    FORCE_INLINE void record(uint32_t index, uint32_t source) { bytes()[index] = static_cast<uint8_t>(source); }

    // Slot `index`'s sender channel.
    FORCE_INLINE uint32_t source_of(uint32_t index) const { return bytes()[index]; }

    // The sender channels of slots `first` .. `first + n - 1`, all in one group: slot `first + k`'s in byte k.
    FORCE_INLINE uint32_t group_sources(uint32_t first, uint32_t n) const {
        const uint32_t word = words[first / slots_per_word];
        return low_bytes(word >> (8 * (first % slots_per_word)), n);
    }

private:
    FORCE_INLINE uint8_t* bytes() { return reinterpret_cast<uint8_t*>(words.data()); }
    FORCE_INLINE const uint8_t* bytes() const { return reinterpret_cast<const uint8_t*>(words.data()); }
};

// ---------------------------------------------------------------------------------------------------------------------
// Counts
// ---------------------------------------------------------------------------------------------------------------------

// The completion counters of one receiver channel (loop locals).
struct CompletionCounts {
    uint32_t uncompleted = 0;  // slots retired (NoC commands issued) but not yet credited
    uint32_t idle_passes = 0;  // groups of several slots: passes without a packet while a partial group waits
};

// Slot `index` was retired; its packet came from sender channel `source`.
template <uint32_t NUM_SLOTS>
FORCE_INLINE void slot_retired(
    CompletionCounts& counts, SlotRecords<NUM_SLOTS>& records, uint32_t index, uint32_t source) {
    records.record(index, source);
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

// ---------------------------------------------------------------------------------------------------------------------
// Release: due, proof, credit
// ---------------------------------------------------------------------------------------------------------------------

// The number of oldest uncredited slots due to go back this pass; 0 when none is (see the top of this file).
template <uint32_t GROUP_SIZE, uint32_t NUM_SLOTS>
FORCE_INLINE uint32_t slots_due(CompletionCounts& counts, uint32_t next_index, bool progress) {
    const uint32_t waiting = counts.uncompleted;
    if constexpr (GROUP_SIZE == 1) {
        // One slot as soon as a slot other than this pass's waits (no partial group, so no idle count).
        return waiting > static_cast<uint32_t>(progress) ? 1 : 0;
    } else {
        bool is_due;
        if (progress) {
            counts.idle_passes = 0;
            is_due = waiting > GROUP_SIZE;
        } else {
            is_due = waiting >= GROUP_SIZE || (waiting != 0 && ++counts.idle_passes >= release_after_idle_passes);
        }
        if (!is_due) {
            return 0;
        }
        // The oldest uncredited slot and the rest of its group that has been retired.
        const uint32_t first = oldest_uncredited<NUM_SLOTS>(next_index, waiting);
        const uint32_t left_in_group = std::min(group_end<GROUP_SIZE>(first), NUM_SLOTS) - first;
        return std::min(waiting, left_in_group);
    }
}

// Credit: `count` slots from `first` (one group's) to their sender channels: as one count when they all came from one
// (the usual case), else one per slot.
template <uint32_t GROUP_SIZE, uint32_t NUM_SLOTS, typename CreditSender>
FORCE_INLINE void credit_slots(
    const SlotRecords<NUM_SLOTS>& records, uint32_t first, uint32_t count, CreditSender& credits) {
    if constexpr (GROUP_SIZE == 1) {
        credits.accumulate_completion_credit(static_cast<uint8_t>(records.source_of(first)), 1);
    } else {
        // One sender channel when every byte of `sources` equals the first. The expressions stay inline: wrapped in
        // helpers, or with the test in a named bool, they change the -Os code (production VC1 builds).
        const uint32_t sources = records.group_sources(first, count);  // slot first + k's sender channel in byte k
        const uint32_t first_source = sources & 0xFF;
        const uint32_t one_per_slot = count == 4 ? 0x01010101u : 0x01010101u & ((1u << (8 * count)) - 1);
        if (sources == first_source * one_per_slot) {
            credits.accumulate_completion_credit(static_cast<uint8_t>(first_source), count);
        } else {
            for (uint32_t k = 0; k < count; k++) {
                credits.accumulate_completion_credit(static_cast<uint8_t>((sources >> (8 * k)) & 0xFF), 1);
            }
        }
    }
}

// Credits the oldest uncredited slots when they are due and their writes have left; returns whether it did. The ring
// has NUM_SLOTS slots in groups of GROUP_SIZE, tagged by Ids; `next_index` is the next slot to retire; `progress` says
// whether this pass handled a packet.
template <uint32_t GROUP_SIZE, uint32_t NUM_SLOTS, typename Ids, typename CreditSender>
FORCE_INLINE bool release_completions(
    CompletionCounts& counts,
    const SlotRecords<NUM_SLOTS>& records,
    uint32_t next_index,
    bool progress,
    CreditSender& credits) {
    static_assert(GROUP_SIZE >= 1 && GROUP_SIZE <= 4 && GROUP_SIZE <= NUM_SLOTS);
    const uint32_t waiting = counts.uncompleted;
    const uint32_t count = slots_due<GROUP_SIZE, NUM_SLOTS>(counts, next_index, progress);
    if (count == 0) {
        return false;
    }
    // Proof: a single slot is checked once; a group waits for it, bounded.
    constexpr uint32_t proof_checks = GROUP_SIZE == 1 ? 1 : completion_wait_checks;
    const uint32_t first = oldest_uncredited<NUM_SLOTS>(next_index, waiting);
    if (!writes_left_within<proof_checks>(Ids::of(first))) {
        return false;
    }
    if constexpr (GROUP_SIZE > 1) {
        counts.idle_passes = 0;
    }
    credit_slots<GROUP_SIZE>(records, first, count, credits);
    counts.uncompleted = waiting - count;
    return true;
}

}  // namespace fabric_router
