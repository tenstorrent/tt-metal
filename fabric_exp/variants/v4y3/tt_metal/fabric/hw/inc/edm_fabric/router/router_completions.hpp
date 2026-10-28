// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Completion credits: a receiver slot goes back to the far sender channel its packet came from once every NoC command
// that reads the slot has left the NIU (the far sender may then overwrite it).
//
// The credit frame. The receiver's credits to the far sender are absolute counts per sender channel in this router's L1,
// completions and (first-level-ack VCs) acks, copied to the far sender's L1 as one Ethernet frame; the far sender reads
// both kinds from it. Acks and completions only raise the counts (add_ack, credit_completion); the loop sends one frame
// per pass when the counts changed and the TX queue is free (send_credit_frame), so a pass's acks and completions share a
// frame.
//
// Batching. Completions go back in batches: each batch costs the far sender ERISC a credit return to its own upstream,
// so batching saves that on all but one packet of a batch on steady streams. A batch is a
// quarter of the ring, at most 4 slots; rings under 8 slots credit every slot on its own. When a batch goes out:
//   - while packets keep arriving: once more than a batch waits (the newest slot is never in it);
//   - on a pass without a packet: a full batch at once, a partial one after `release_after_idle_passes` such passes.
// So a channel holds back at most a batch minus one of its slots, and only until its next packet or a few idle passes
// (holds bounded by the ring, LEDGER E1-28: VC1 -12 to -83, no bursty losses).
//
// Proving a batch's slots free takes one NoC status check in the usual case, not one per slot. It rests on two facts:
//   - two command buffers read slots: the local-write buffer (local delivery) and the forwarding data buffer (forwards),
//     and each slot's record says which of them read it;
//   - the commands of one command buffer leave the NIU in issue order, so when the newest slot a buffer read has no
//     outgoing write, neither has any older slot on that buffer.
// So a batch needs the status of its newest slot on each buffer it used: one check, or two (as one group) when the batch
// mixes local writes and forwards; none when no slot was read (atomics, inline writes). A send type that read a slot on
// a third command buffer would break this; every send type's record bits are set in router_receiver.hpp.
//
// A full batch whose newest write is still leaving waits for it (bounded), then is credited, rather than being left to
// the next pass: under load the next pass handles the next packet first, which would delay the credit by a packet
// (LEDGER E1-24, E1-25: means -1.6 to -4.8). A single slot does not wait (writes_have_left).

#include "router_common.hpp"

namespace fabric_router {

// A slot's record, one byte: the sender channel its packet came from (for the credit) and the command buffers whose
// NoC commands read the slot's data.
struct SlotRecord {
    static constexpr uint32_t SOURCE_MASK = 0x3F;
    static constexpr uint32_t READ_LOCAL = 0x40;  // read by a local write (local-write command buffer)
    static constexpr uint32_t READ_FORWARD = 0x80;  // read by a forward (forwarding data command buffer)
    static constexpr uint32_t READS = READ_LOCAL | READ_FORWARD;
};
static_assert(MAX_NUM_SENDER_CHANNELS <= SlotRecord::SOURCE_MASK + 1);

// The records of a ring of N slots, kept as words: byte k of word w is slot 4w + k, so a batch of four is one word load
// (two when it straddles a word).
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
template <uint32_t N>
struct SlotRecords {
    static constexpr uint32_t num_words = (N + 3) / 4;
    std::array<uint32_t, num_words> words;

    FORCE_INLINE void set(uint32_t index, uint32_t record) {
        reinterpret_cast<uint8_t*>(words.data())[index] = static_cast<uint8_t>(record);
    }
    FORCE_INLINE uint32_t get(uint32_t index) const { return reinterpret_cast<const uint8_t*>(words.data())[index]; }

    // The records of slots base .. base + 3 in ring order, slot `base` in the low byte.
    FORCE_INLINE uint32_t four_from(uint32_t base) const {
        if constexpr (N % 4 == 0) {
            const uint32_t w = base >> 2;
            const uint32_t shift = (base & 3) * 8;
            uint32_t four = words[w];
            if (shift != 0) {
                const uint32_t next = w + 1 == num_words ? 0 : w + 1;
                four = (four >> shift) | (words[next] << (32 - shift));
            }
            return four;
        } else {
            uint32_t four = 0;
            uint32_t index = base;
            for (uint32_t k = 0; k < 4; k++) {
                four |= get(index) << (8 * k);
                index = index + 1 == N ? 0 : index + 1;
            }
            return four;
        }
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

// Batch-record helpers: a batch's records are one word, its oldest slot in the low byte.
//   in_each_byte(field): `field` in every slot's byte;
//   newest_slot_in(bits): the batch position (0 = oldest) of the newest slot whose byte in `bits` is not zero.
constexpr uint32_t in_each_byte(uint32_t field) { return field * 0x01010101u; }
FORCE_INLINE uint32_t newest_slot_in(uint32_t bits) { return (31 - static_cast<uint32_t>(__builtin_clz(bits))) >> 3; }

// The transaction ID of the slot `n` places after slot `base` in a ring of N slots.
template <uint32_t N, typename TridTracker>
FORCE_INLINE uint32_t trid_after(const TridTracker& trids, uint32_t base, uint32_t n) {
    return trids.get_buffer_slot_trid(tt::tt_fabric::BufferIndex{static_cast<uint8_t>(ring_add<N>(base, n))});
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
    uint32_t idle_passes = 0;  // passes without a packet while a partial batch waits
};

template <uint32_t N>
constexpr uint32_t completion_batch_size = N < 8 ? 1 : (N / 4 < 4 ? N / 4 : 4);


// A slot was retired with `record`; `index` is its index in the ring.
template <uint32_t N>
FORCE_INLINE void slot_retired(CompletionCounts& counts, SlotRecords<N>& records, uint32_t index, uint32_t record) {
    records.set(index, record);
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

// Sends the credit frame, all the counts, unless the TX queue is busy (the next pass sends it then). Returns whether it
// was sent. `txq_was_free`: the queue was seen free earlier in this pass; only this ERISC sends on it, so it still is.
template <typename CreditSender>
FORCE_INLINE bool send_credit_frame(CreditSender& credits, bool txq_was_free) {
    if constexpr (ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
        }
    } else if (!txq_was_free && internal_::eth_txq_is_busy(receiver_txq_id)) {
        return false;
    }
    credits.flush_completion_credits();  // the whole credit block: completions and acks
    return true;
}

// The next completion credit: the oldest `count` uncredited slots from `base`, and which slots' write-sent status proves
// them free.
//   proof   the batch position of the newest slot a command buffer read: its status covers that buffer's older slots
//   other   when the batch used both buffers and the other buffer's newest slot is older, that slot (else `proof`)
struct Completion {
    uint32_t base;
    uint32_t count;  // 0: no credit due
    uint32_t recs;   // the slots' records, the oldest in the low byte
    uint32_t proof;
    uint32_t other;
    bool read;  // some slot was read by a command buffer
};

// Which credit is due, in batches of T: while packets keep coming (`progress`: this pass handled one), once more than a
// batch waits, so the newest slot stays out; on a pass without a packet, a full batch, or a partial one after
// `release_after_idle_passes` such passes.
template <uint32_t T, uint32_t N>
FORCE_INLINE Completion next_completion(
    CompletionCounts& counts, const SlotRecords<N>& records, uint32_t next_index, bool progress) {
    static_assert(T >= 1 && T <= 4 && T <= N);
    const uint32_t waiting = counts.uncompleted;
    uint32_t count = 0;
    if (progress) {
        counts.idle_passes = 0;
        count = waiting > T ? T : 0;
    } else if (waiting >= T) {
        count = T;
    } else if (waiting != 0 && ++counts.idle_passes >= release_after_idle_passes) {
        count = waiting;
    }
    Completion c{ring_add<N>(next_index, N - waiting), count, 0, 0, 0, false};
    if (count == 0) {
        return c;
    }
    if constexpr (T == 1) {
        c.recs = records.get(c.base);
        c.read = (c.recs & SlotRecord::READS) != 0;
    } else {
        c.recs = records.four_from(c.base);
        if (count < 4) {
            c.recs &= (1u << (8 * count)) - 1;
        }
        const uint32_t local_reads = c.recs & in_each_byte(SlotRecord::READ_LOCAL);
        const uint32_t forward_reads = c.recs & in_each_byte(SlotRecord::READ_FORWARD);
        c.read = (local_reads | forward_reads) != 0;
        if (c.read) {
            c.proof = newest_slot_in(local_reads | forward_reads);
            c.other = c.proof;
            if (local_reads != 0 && forward_reads != 0) {
                const uint32_t newest_local = newest_slot_in(local_reads);
                const uint32_t newest_forward = newest_slot_in(forward_reads);
                c.other = newest_local < newest_forward ? newest_local : newest_forward;
            }
        }
    }
    return c;
}

// Whether every NoC command that read `c`'s slots has left the NIU. A single slot is checked once: a slot still being
// read is credited by a later pass (waiting for it cost ~3 cycles/packet on rings, RESULTS §98 v4sc2). A full batch's
// newest write usually leaves just after the first check (LEDGER E1-24), so a batch waits for it, bounded, rather than
// going a pass later (E1-25).
template <uint32_t T, uint32_t N, typename TridTracker>
FORCE_INLINE bool writes_have_left(const Completion& c, const TridTracker& trids) {
    const uint32_t proof = trid_after<N>(trids, c.base, c.proof);
    if constexpr (T == 1) {
        return writes_left(proof);
    } else {
        uint32_t checks = 0;
        while (!writes_left(proof)) {
            if (++checks == completion_wait_checks) {
                return false;
            }
        }
        return c.other == c.proof || writes_left(trid_after<N>(trids, c.base, c.other));
    }
}

// Credits `c` to the counts when one is due and its writes have left. Returns whether it did.
template <uint32_t T, uint32_t N, typename TridTracker, typename CreditSender>
FORCE_INLINE bool credit_completion(
    CompletionCounts& counts, const Completion& c, TridTracker& trids, CreditSender& credits) {
    if (c.count == 0 || (c.read && !writes_have_left<T, N>(c, trids))) {
        return false;
    }
    if constexpr (T == 1) {
        credits.accumulate_completion_credit(static_cast<uint8_t>(c.recs & SlotRecord::SOURCE_MASK), 1);
        if constexpr (!(TridTracker::BOTH_PARAMS_ARE_EQUAL || TridTracker::BOTH_PARAMS_ARE_POW2)) {
            trids.clear_trid_at_buffer_slot(tt::tt_fabric::BufferIndex{static_cast<uint8_t>(c.base)});
        }
        counts.uncompleted--;
    } else {
        // Slots from one sender channel (the usual case) are one credit of `count`; otherwise one credit per slot.
        const uint32_t changes = c.recs ^ (c.recs >> 8);
        const uint32_t pairs = (1u << (8 * (c.count - 1))) - 1;
        if ((changes & pairs & in_each_byte(SlotRecord::SOURCE_MASK)) == 0) {
            credits.accumulate_completion_credit(static_cast<uint8_t>(c.recs & SlotRecord::SOURCE_MASK), c.count);
        } else {
            for (uint32_t k = 0; k < c.count; k++) {
                credits.accumulate_completion_credit(
                    static_cast<uint8_t>((c.recs >> (8 * k)) & SlotRecord::SOURCE_MASK), 1);
            }
        }
        if constexpr (!(TridTracker::BOTH_PARAMS_ARE_EQUAL || TridTracker::BOTH_PARAMS_ARE_POW2)) {
            for (uint32_t k = 0; k < c.count; k++) {
                trids.clear_trid_at_buffer_slot(
                    tt::tt_fabric::BufferIndex{static_cast<uint8_t>(ring_add<N>(c.base, k))});
            }
        }
        counts.uncompleted -= c.count;
    }
    counts.idle_passes = 0;
    return true;
}

}  // namespace fabric_router
