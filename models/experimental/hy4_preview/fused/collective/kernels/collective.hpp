// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The landing-core side of the in-program collective (fused/collective/collective.py). The data-movement kernel of
// a landing core includes this file and calls, per use of a site:
//
//     auto use = hy4_collective::read_use(arg);       // 10 per-core runtime args, see Use below
//     hy4_collective::hand_over(use, payload_l1_addr);  // payload -> this chip's sender core -> every ring member
//     uint32_t parity;
//     bool ok = hy4_collective::wait_landed(use, parity);  // every member's payload is in this core's landing
//     ... consume use.landing + parity * use.parity_stride: `members` slots of `slot_stride` bytes in ring order,
//         or push them to the landing CB of that parity: cb_push_back(cb_landing[parity], members * tiles) ...
//     hy4_collective::finish(use);                    // after the landing has been consumed, never before
//
// A site is a set of persistent L1 objects at one address on its landing cores of every chip:
//   ctrl     uint32 words 16 bytes apart: arrival counter of parity 0, of parity 1, the round, the status.
//   landing  2 parities x `members` slots; slot i holds the payload (this core's part of it) of ring member i.
// Chips start a program at different times. A member that has finished round r may send round r + 1 while a slower
// member still reads round r, so rounds alternate between the two parities: a member cannot reach round r + 2
// before every member has sent round r + 1, which a member does only after finish() of round r. The rule that
// keeps this true: on one site, finish() of a use comes before hand_over() of the next use.
// Counters are consumed by subtraction, never cleared: a peer may count the next round before this one is consumed.
//
// Waits are bounded when HY4_COLLECTIVE_WAIT_LIMIT is not 0 (polls; 10 ns each measured on Blackhole): a wait that
// runs out writes a code into the status word and returns false; the kernel is expected to finish so the program
// terminates, and the host reads the status (Site.read_ctrl / Site.check_idle in collective.py).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#ifndef HY4_COLLECTIVE_WAIT_LIMIT
#define HY4_COLLECTIVE_WAIT_LIMIT 0
#endif
// 1: the kernels leave wall-clock stamps (core clock cycles, low word) in the control block, for a breakdown of
// where a collective's time goes (Site.read_stamps in collective.py). Off, nothing is measured or written.
#ifndef HY4_COLLECTIVE_TIMING
#define HY4_COLLECTIVE_TIMING 0
#endif

namespace hy4_collective {

constexpr uint32_t CTRL_COUNTER = 0;  // + 16 * parity
constexpr uint32_t CTRL_ROUND = 32;
constexpr uint32_t CTRL_STATUS = 48;
constexpr uint32_t CTRL_STAMPS = 64;  // + 16 * stamp

// Stamps of the landing core's data-movement kernel, then of the sender (written when it has closed).
enum Stamp : uint32_t {
    STAMP_START = 0,      // kernel entered (the kernel stamps this itself)
    STAMP_HAND_OVER = 1,  // hand_over() entered
    STAMP_HANDED = 2,     // hand_over() done
    STAMP_LANDED = 3,     // wait_landed() saw every member
    STAMP_FINISH = 4,     // finish(): the landing has been consumed
    STAMP_SENDER_START = 5,
    STAMP_SENDER_OPEN = 6,    // fabric connections open
    STAMP_SENDER_READY = 7,   // first op seen ready
    STAMP_SENDER_SENT = 8,    // last op sent and its own slot counted
    STAMP_SENDER_CLOSED = 9,  // fabric connections closed
};

// Status codes (top byte); the low 24 bits hold what the wait last saw.
constexpr uint32_t STATUS_LANDING_TIMEOUT = 0xA1u << 24;
constexpr uint32_t STATUS_READY_TIMEOUT = 0xA2u << 24;

constexpr uint32_t USE_ARGS = 10;

struct Use {
    uint32_t ctrl;           // this core: the site's control words
    uint32_t landing;        // this core: slot 0 of parity 0
    uint32_t parity_stride;  // bytes from parity 0 to parity 1
    uint32_t slot_stride;    // bytes from slot i to slot i + 1
    uint32_t members;        // ring size: arrivals per round, this chip's own copy included
    uint32_t bytes;          // payload bytes handed over
    uint32_t staging;        // sender core: where the payload is handed over
    uint32_t sender_xy;      // sender core NoC x | y << 16
    uint32_t ready_id;       // program semaphore id of the sender's ready word
    uint32_t op;             // this use's index in the sender's table
};

FORCE_INLINE Use read_use(size_t& arg) {
    Use use;
    use.ctrl = get_arg_val<uint32_t>(arg++);
    use.landing = get_arg_val<uint32_t>(arg++);
    use.parity_stride = get_arg_val<uint32_t>(arg++);
    use.slot_stride = get_arg_val<uint32_t>(arg++);
    use.members = get_arg_val<uint32_t>(arg++);
    use.bytes = get_arg_val<uint32_t>(arg++);
    use.staging = get_arg_val<uint32_t>(arg++);
    use.sender_xy = get_arg_val<uint32_t>(arg++);
    use.ready_id = get_arg_val<uint32_t>(arg++);
    use.op = get_arg_val<uint32_t>(arg++);
    return use;
}

FORCE_INLINE volatile tt_l1_ptr uint32_t* word(uint32_t address) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
}

FORCE_INLINE uint32_t now() {
#if HY4_COLLECTIVE_TIMING
    return get_timestamp_32b();
#else
    return 0;
#endif
}

// Leave a stamp in the control block at `ctrl` on this core.
FORCE_INLINE void stamp([[maybe_unused]] uint32_t ctrl, [[maybe_unused]] uint32_t which) {
#if HY4_COLLECTIVE_TIMING
    word(ctrl + CTRL_STAMPS + 16 * which)[0] = get_timestamp_32b();
#endif
}

// Poll until the word at `address` reaches `value`. False when the bound ran out; `seen` is the last value read.
inline bool wait_min(uint32_t address, uint32_t value, uint32_t& seen) {
    volatile tt_l1_ptr uint32_t* p = word(address);
#if HY4_COLLECTIVE_WAIT_LIMIT != 0
    for (uint32_t polls = 0; polls < HY4_COLLECTIVE_WAIT_LIMIT; ++polls) {
        invalidate_l1_cache();
        seen = p[0];
        if (seen >= value) {
            return true;
        }
    }
    return false;
#else
    do {
        invalidate_l1_cache();
        seen = p[0];
    } while (seen < value);
    return true;
#endif
}

// The parity of the round this core is in on the site.
FORCE_INLINE uint32_t parity(const Use& use) {
    invalidate_l1_cache();
    return word(use.ctrl + CTRL_ROUND)[0] & 1u;
}

// Give `use.bytes` at local L1 address `source` to the sender core. The sender copies it into this chip's own
// slot and sends it to every other member; the source may be reused once this returns.
inline void hand_over(const Use& use, uint32_t source) {
    WAYPOINT("MKHO");
    stamp(use.ctrl, STAMP_HAND_OVER);
    const uint32_t sx = use.sender_xy & 0xFFFF;
    const uint32_t sy = use.sender_xy >> 16;
    noc_async_write(source, get_noc_addr(sx, sy, use.staging), use.bytes);
    noc_async_write_barrier();
    // Two bits per op in the ready word: bit 2 op + parity. The ready word is a program semaphore, at the same
    // address on every worker core of the program.
    noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(use.ready_id)), 1u << (2 * use.op + parity(use)));
    noc_async_atomic_barrier();
    stamp(use.ctrl, STAMP_HANDED);
    WAYPOINT("MKHD");
}

// Wait until every member's payload of this round is in the landing; take the arrivals off the counter.
// `round_parity` selects the landing: use.landing + round_parity * use.parity_stride.
inline bool wait_landed(const Use& use, uint32_t& round_parity) {
    WAYPOINT("MKLW");
    round_parity = parity(use);
    const uint32_t counter = use.ctrl + CTRL_COUNTER + 16 * round_parity;
    uint32_t seen = 0;
    if (!wait_min(counter, use.members, seen)) {
        word(use.ctrl + CTRL_STATUS)[0] = STATUS_LANDING_TIMEOUT | (round_parity << 20) | (seen & 0xFFFFF);
        return false;
    }
    stamp(use.ctrl, STAMP_LANDED);
    noc_semaphore_inc(get_noc_addr(counter), 0u - use.members);
    noc_async_atomic_barrier();
    WAYPOINT("MKLD");
    return true;
}

// The landing of this round has been consumed: the next round uses the other parity.
FORCE_INLINE void finish(const Use& use) {
    stamp(use.ctrl, STAMP_FINISH);
    volatile tt_l1_ptr uint32_t* round = word(use.ctrl + CTRL_ROUND);
    invalidate_l1_cache();
    round[0] = round[0] + 1;
}

}  // namespace hy4_collective
