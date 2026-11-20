// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Hand-overs of a fused layer program on the 8x4 mesh (fused/layer_common/transport.py): the persistent words every
// core has, bounded waits on them, and the two ways a core passes a payload on (to this chip's sender core, or
// straight into a landing on this chip).
//
// The words are 16 uint32 per core in a host-zeroed L1 buffer at the same address on every core of every chip
// (HY4L_WORDS, a define). They are not program semaphores (those are rewritten on every core at every launch, which
// costs launch time, and are lost between programs): a peer chip may count into one before this chip's program has
// started. A count is taken off by an atomic of the waiting core on itself, never cleared, so a count that arrives
// early is kept.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#ifndef HY4L_WAIT_LIMIT
#define HY4L_WAIT_LIMIT 30000000u  // polls of a bounded wait: about 10 ns each on Blackhole
#endif
#ifndef HY4L_TIMING
#define HY4L_TIMING 0
#endif

namespace hy4l {

constexpr uint32_t W_X = 0;       // hub core: the layer input x is in the hub's buffer (counted by the root's sender)
constexpr uint32_t W_S1 = 1;      // reduce core: partial outputs in its stage-1 landing (row stage)
constexpr uint32_t W_S2 = 2;      // reduce core of the root chip: row sums in its stage-2 landing (column stage)
constexpr uint32_t W_GATHER = 3;  // root chip's first site core: site cores whose shard of x is in the sender's staging
constexpr uint32_t W_READY = 4;   // sender core: its ready word, two bits per op (collective_sender.cpp)
constexpr uint32_t W_STATUS = 5;  // any core: the last wait that ran out here (code << 24 | what it saw)
constexpr uint32_t W_RUNS = 6;    // hub core: launches
constexpr uint32_t W_FED = 7;     // hub core: partial outputs handed on
constexpr uint32_t W_STAMP = 8;   // .. 11: wall-clock stamps of this core (HY4L_TIMING), low 32 bits; a fifth in 15
// 12 .. 14 belong to the sender: an op that was never handed over is reported at its counter's address + 48
// (collective.hpp CTRL_STATUS), which for the counters W_X, W_S1, W_S2 is word 12, 13, 14 of the landing core.

constexpr uint32_t ST_X = 0xC1u << 24;       // the hub saw no x
constexpr uint32_t ST_OUT = 0xC2u << 24;     // the hub's compute did not complete
constexpr uint32_t ST_S1 = 0xC3u << 24;      // a reduce core's stage 1 did not fill
constexpr uint32_t ST_S2 = 0xC4u << 24;      // a reduce core's stage 2 did not fill
constexpr uint32_t ST_GATHER = 0xC5u << 24;  // the root's site cores did not all stage their shard

#ifdef HY4L_WORDS
FORCE_INLINE volatile tt_l1_ptr uint32_t* word(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(HY4L_WORDS + 4 * i);
}
constexpr uint32_t word_addr(uint32_t i) { return HY4L_WORDS + 4 * i; }

// Poll until this core's word i reaches `need`. False when the bound ran out (the status word then says what was
// seen); the kernel carries on with what it has, so the program always ends.
inline bool wait_word(uint32_t i, uint32_t need, uint32_t code) {
    volatile tt_l1_ptr uint32_t* p = word(i);
    uint32_t seen = 0;
    for (uint32_t polls = 0; polls < HY4L_WAIT_LIMIT; ++polls) {
        invalidate_l1_cache();
        seen = p[0];
        if (seen >= need) {
            return true;
        }
    }
    word(W_STATUS)[0] = code | (seen & 0xFFFFFF);
    return false;
}

// Take `n` off this core's word i (an atomic on itself: a peer may be counting the next round into it).
FORCE_INLINE void take_word(uint32_t i, uint32_t n) {
    noc_semaphore_inc(get_noc_addr(word_addr(i)), 0u - n);
    noc_async_atomic_barrier();
}

FORCE_INLINE void stamp([[maybe_unused]] uint32_t n) {
#if HY4L_TIMING
    word(n < 4 ? W_STAMP + n : 15)[0] = get_timestamp_32b();
#endif
}

// Poll until the word at `address` of this core's L1 (another piece's hand-over word) reaches `need`. False when
// the bound ran out, with the code and what was seen in this core's status word.
inline bool wait_address(uint32_t address, uint32_t need, uint32_t code) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    uint32_t seen = 0;
    for (uint32_t polls = 0; polls < HY4L_WAIT_LIMIT; ++polls) {
        invalidate_l1_cache();
        seen = p[0];
        if (seen >= need) {
            return true;
        }
    }
    word(W_STATUS)[0] = code | (seen & 0xFFFFFF);
    return false;
}
#endif  // HY4L_WORDS

// Where a hub's partial output goes once it is complete, as runtime arguments from index `arg` on:
//   mode        0: to this chip's sender core (one op per part), 1: into this chip's reduce cores' stage-1 landing
//   parts, part_bytes
//   mode 0:     sender core NoC x | y << 16, staging address, ready word address, ready bits of the ops (one add)
//   mode 1:     this chip's slot in the stage-1 landing (address on a reduce core), the counter's address,
//               then per part the reduce core's NoC x | y << 16
// `source`: the partial output in this core's L1, `parts * part_bytes`.
inline void feed_reduce(size_t arg, uint32_t source) {
    const uint32_t mode = get_arg_val<uint32_t>(arg++);
    const uint32_t parts = get_arg_val<uint32_t>(arg++);
    const uint32_t part_bytes = get_arg_val<uint32_t>(arg++);
    if (mode == 0) {
        const uint32_t sender = get_arg_val<uint32_t>(arg++);
        const uint32_t staging = get_arg_val<uint32_t>(arg++);
        const uint32_t ready = get_arg_val<uint32_t>(arg++);
        const uint32_t bits = get_arg_val<uint32_t>(arg++);
        const uint32_t sx = sender & 0xFFFF;
        const uint32_t sy = sender >> 16;
        for (uint32_t k = 0; k < parts; ++k) {
            noc_async_write(source + k * part_bytes, get_noc_addr(sx, sy, staging + k * part_bytes), part_bytes);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(sx, sy, ready), bits);
        noc_async_atomic_barrier();
    } else {
        const uint32_t slot = get_arg_val<uint32_t>(arg++);
        const uint32_t counter = get_arg_val<uint32_t>(arg++);
        for (uint32_t k = 0; k < parts; ++k) {
            const uint32_t core = get_arg_val<uint32_t>(arg + k);
            noc_async_write(source + k * part_bytes, get_noc_addr(core & 0xFFFF, core >> 16, slot), part_bytes);
        }
        noc_async_write_barrier();
        for (uint32_t k = 0; k < parts; ++k) {
            const uint32_t core = get_arg_val<uint32_t>(arg + k);
            noc_semaphore_inc(get_noc_addr(core & 0xFFFF, core >> 16, counter), 1);
        }
        noc_async_atomic_barrier();
    }
}

}  // namespace hy4l
