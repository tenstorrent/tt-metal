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
// (LEDGER E1-24, E1-25: means -1.6 to -4.8). A single slot does not wait.

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

// Credits a batch of T slots to the counts when one is due and provably free; returns whether it did. `next_index` is
// the index of the next slot to retire; `progress` says whether this pass handled a packet.
template <uint32_t T, uint32_t N, typename TridTracker, typename CreditSender>
FORCE_INLINE bool release_completions(
    CompletionCounts& counts,
    const SlotRecords<N>& records,
    uint32_t next_index,
    bool progress,
    TridTracker& trids,
    CreditSender& credits) {
    static_assert(T >= 1 && T <= 4 && T <= N);
    const uint32_t waiting = counts.uncompleted;
    bool due;
    if constexpr (T == 1) {
        // Experiment (v4eg): a single slot is due as soon as one waits, the slot just written too, as main credits: on
        // a saturated ring (bubble flow control) that slot is the free slot the ring moves on (Galaxy job 158477).
        due = waiting != 0;
    } else if (progress) {
        counts.idle_passes = 0;
        due = waiting > T;
    } else {
        due = waiting >= T || (waiting != 0 && ++counts.idle_passes >= release_after_idle_passes);
    }
    if (!due) {
        return false;
    }
    if constexpr (T == 1) {
        // One slot (first-level-ack VCs): its writes, if it was read, then its credit.
        const uint32_t index = ring_add<N>(next_index, N - waiting);
        const uint32_t record = records.get(index);
        // Checked once: a slot still being read is credited by a later pass (waiting for it here cost ~3 cycles/packet
        // on rings and tori, RESULTS §98 v4sc2).
        if ((record & SlotRecord::READS) != 0 && !writes_left(trid_after<N>(trids, index, 0))) {
            return false;
        }
        counts.idle_passes = 0;
        credits.accumulate_completion_credit(static_cast<uint8_t>(record & SlotRecord::SOURCE_MASK), 1);
        if constexpr (!(TridTracker::BOTH_PARAMS_ARE_EQUAL || TridTracker::BOTH_PARAMS_ARE_POW2)) {
            trids.clear_trid_at_buffer_slot(tt::tt_fabric::BufferIndex{static_cast<uint8_t>(index)});
        }
        counts.uncompleted = waiting - 1;
        return true;
    }
    const uint32_t batch = waiting > T ? T : waiting;
    const uint32_t base = ring_add<N>(next_index, N - waiting);  // the oldest uncredited slot

    uint32_t recs = records.four_from(base);
    if (batch < 4) {
        recs &= (1u << (8 * batch)) - 1;
    }
    // Byte k: where slot k's record differs from slot k + 1's, over the T - 1 pairs of a full batch.
    constexpr uint32_t pairs = (1u << (8 * (T - 1))) - 1;
    const uint32_t changes = recs ^ (recs >> 8);
    const bool full = batch == T;
    const bool one_sender = full && (changes & pairs & in_each_byte(SlotRecord::SOURCE_MASK)) == 0;
    const bool reads_alike = full && (changes & pairs & in_each_byte(SlotRecord::READS)) == 0;

    if (reads_alike) {
        // Every slot reads its data the same way: the newest slot proves them all (or no slot is read).
        if (((recs >> (8 * (T - 1))) & SlotRecord::READS) != 0) {
            const uint32_t newest = trid_after<N>(trids, base, T - 1);
            uint32_t checks = 0;
            while (!writes_left(newest)) {
                if (++checks == completion_wait_checks) {
                    return false;  // leave the batch to a later pass
                }
            }
        }
    } else {
        const uint32_t local_reads = recs & in_each_byte(SlotRecord::READ_LOCAL);
        const uint32_t forward_reads = recs & in_each_byte(SlotRecord::READ_FORWARD);
        if ((local_reads | forward_reads) != 0) {
            // The newest slot read on any buffer, and the other buffer's newest slot when that one is older.
            const uint32_t newest = newest_slot_in(local_reads | forward_reads);
            const uint32_t other = local_reads == 0 || forward_reads == 0      ? 0u
                                   : newest_slot_in(local_reads) == newest ? forward_reads
                                                                           : local_reads;
            if (other == 0 || newest_slot_in(other) == newest) {
                if (!writes_left(trid_after<N>(trids, base, newest))) {
                    return false;
                }
            } else {
                const uint32_t newest_trid = trid_after<N>(trids, base, newest);
                const uint32_t older_trid = trid_after<N>(trids, base, newest_slot_in(other));
                std::atomic_signal_fence(std::memory_order_seq_cst);
                const bool newest_left = writes_left(newest_trid);
                const bool older_left = writes_left(older_trid);
                std::atomic_signal_fence(std::memory_order_seq_cst);
                if (!(newest_left && older_left)) {
                    return false;
                }
            }
        }
    }
    counts.idle_passes = 0;
    // A full batch from one sender channel (the usual case) is one credit of T; any other batch is one credit per slot.
    // The counts end the same either way.
    if (one_sender) {
        credits.accumulate_completion_credit(static_cast<uint8_t>(recs & SlotRecord::SOURCE_MASK), T);
    } else {
        for (uint32_t k = 0; k < batch; k++) {
            credits.accumulate_completion_credit(static_cast<uint8_t>((recs >> (8 * k)) & SlotRecord::SOURCE_MASK), 1);
        }
    }
    if constexpr (!(TridTracker::BOTH_PARAMS_ARE_EQUAL || TridTracker::BOTH_PARAMS_ARE_POW2)) {
        for (uint32_t k = 0; k < batch; k++) {
            trids.clear_trid_at_buffer_slot(tt::tt_fabric::BufferIndex{static_cast<uint8_t>(ring_add<N>(base, k))});
        }
    }
    counts.uncompleted = waiting - batch;
    return true;
}

}  // namespace fabric_router
