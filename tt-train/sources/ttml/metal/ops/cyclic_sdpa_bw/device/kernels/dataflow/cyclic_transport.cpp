// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The transport of the cyclic SDPA backward pass, carrying a checksum instead
// of tensors.
//
// The gradient arithmetic is not here and not yet anywhere: this kernel moves
// a row packet through the schedule and accumulates a checksum into it, so
// that a packet delivered to the wrong slot, consumed twice, or consumed
// before its producer finished shows up as a wrong integer rather than as
// numerical drift in a gradient. Compute arrives in a later step and slots
// into the same structure.
//
// Modes, selected at compile time, in the order the paper builds them:
//
//   TRANSPORT_DRAM      every timestep loads the row packet from DRAM and
//                       writes it back, ordered by a chip-wide barrier.
//                       This is Algorithm 2's row traffic.
//   TRANSPORT_RELAY     the packet stays in L1 for the length of an active
//                       streak and is forwarded to the next consumer over the
//                       NoC; DRAM is touched only at a streak start (load) and
//                       a streak end (spill). Algorithm 3, barrier variant.
//
// The relay implements the paper's communication contract:
//
//   1. two receive slots per core, the packet for destination timestep u in
//      slot u mod 2;
//   2. timestep-specific credits, as one monotone counter per (producer,
//      receiver) pair living on the producer: before its n-th forward to r, a
//      core waits for r's counter to reach n. Equivalent to the paper's
//      tagged permissions because, for a fixed pair, grants and forwards are
//      the same timesteps in the same order. The initial permissions for
//      destination timesteps 0 and 1 are granted by the receivers at startup,
//      which avoids having to initialise per-core counters to different
//      values and needs no synchronisation of its own, since increments are
//      atomic and order-independent;
//   3. payload before readiness: write the packet, complete the write, then
//      set the receiver's readiness word for that exact timestep to u + 1.
//      The consumer waits for its slot's word to reach t + 1, so a stale tag
//      from the slot's previous use cannot satisfy it;
//   4. release only after every use: the credit for destination t + 2 is
//      granted after this timestep's forward or spill has completed.
//
// Row packet, kPacketWords uint32 in DRAM, one page per row block:
//   0 row id (immutable; a packet in the wrong place is detectable)
//   1 number of consumers that have updated it
//   2 checksum, accumulating row_term(i, j) over visits
//
// Column state, one page per column block, same shape:
//   0 column id   1 number of visits   2 checksum over column_term(i, j)
//
// The column pages are poisoned by the host, because the column-state rules
// say a first visit initialises the gradients locally and must not read them
// from DRAM. A kernel that reads them anyway accumulates poison.
//
// Two compile-time knobs exist so the tests can show they are load-bearing,
// in the spirit of the simulator's fault injection:
//
//   FAULT_NO_BARRIER  drop the chip-wide barrier entirely.
//   FAULT_STALE_TAG_OK          wait for any positive readiness tag instead of
//                               this timestep's, so a tag left over from the
//                               slot's previous use satisfies the wait. This is
//                               the "generic binary flag" the contract rejects.
//   FAULT_NO_CREDIT_WAIT        forward without waiting for the receiver's
//                               permission for that destination timestep.
//   FAULT_READINESS_BEFORE_PAYLOAD  publish readiness without completing the
//                               payload write first.
//   SKEW_ITERS        spin this many iterations at the top of each timestep on
//                     odd-numbered cores, to drive cores apart. Harmless with
//                     the barrier in place, which is itself worth asserting.
//   RMW_SPIN_ITERS    spin between reading a row packet and writing it back,
//                     widening the read-modify-write window.
//   RMW_SPIN_CORE     restrict that widening to one core (0 = every core).
//
// The last one needs explaining, because the obvious fault design does not
// work. This checksum accumulates commutatively, so a read-modify-write that
// happens out of timestep order still lands the right total: the only way to
// lose a term is for two cores to overlap inside the window, one's write
// landing on a page the other already read. Skewing cores apart makes that
// *less* likely, not more, because the windows are short and move away from
// each other. Widening the window is what exposes it -- and that is precisely
// the property the barrier provides, since the schedule already guarantees
// distinct rows within a timestep. What the barrier adds is that no two cores
// are ever in different timesteps that share a row at the same moment.
//
// Widening the window on every core does not expose it either: they all slow
// down together and stay in step, so the two consumers of a row remain a
// whole timestep apart in time. The exposure needs one core holding a packet
// while the others run ahead, which is RMW_SPIN_CORE.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/parity_snake.hpp"

namespace {

constexpr uint32_t kPacketWords = 8;
constexpr uint32_t kPacketBytes = kPacketWords * sizeof(uint32_t);

// Distinct per pair, and distinct between the two accumulators, so a swapped
// i and j or a row term added to a column does not cancel out.
constexpr uint32_t row_term(uint32_t i, uint32_t j) {
    return i * 4096u + j;
}
constexpr uint32_t column_term(uint32_t i, uint32_t j) {
    return j * 4096u + i;
}

using Words = volatile tt_l1_ptr uint32_t*;

#ifndef SKEW_ITERS
#define SKEW_ITERS 0
#endif

#ifndef RMW_SPIN_ITERS
#define RMW_SPIN_ITERS 0
#endif

#ifndef RMW_SPIN_CORE
#define RMW_SPIN_CORE 0
#endif

// A portable busy wait. riscv_wait() lives in an internal arch header, and
// this only has to be long, not precise.
inline void spin(uint32_t iterations) {
    volatile uint32_t sink = 0;
    for (uint32_t n = 0; n < iterations; ++n) {
        sink = sink + 1u;
    }
}

}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t my_core = get_arg_val<uint32_t>(arg++);
    const uint32_t packet_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t column_addr = get_arg_val<uint32_t>(arg++);
    // Barrier: the coordinator holds the arrival counter and publishes the
    // release value to every core, its own copy included.
    const uint32_t coord_noc_x = get_arg_val<uint32_t>(arg++);
    const uint32_t coord_noc_y = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_x_start = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_y_start = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_x_end = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_y_end = get_arg_val<uint32_t>(arg++);
    const uint32_t is_coordinator = get_arg_val<uint32_t>(arg++);
    const uint32_t stats_addr = get_arg_val<uint32_t>(arg++);
    // Snake neighbours: the only cores this one ever exchanges packets with.
    const uint32_t prev_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t prev_noc_x = get_arg_val<uint32_t>(arg++);
    const uint32_t prev_noc_y = get_arg_val<uint32_t>(arg++);
    const uint32_t next_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t next_noc_x = get_arg_val<uint32_t>(arg++);
    const uint32_t next_noc_y = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t kCores = get_compile_time_arg_val(0);
    constexpr uint32_t arrive_sem_id = get_compile_time_arg_val(1);
    constexpr uint32_t release_sem_id = get_compile_time_arg_val(2);
    // Readiness, one per slot, and credits, one per possible receiver.
    constexpr uint32_t ready_sem_id[2] = {get_compile_time_arg_val(3), get_compile_time_arg_val(4)};
    constexpr uint32_t credit_prev_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t credit_next_sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t credit_self_sem_id = get_compile_time_arg_val(7);
    constexpr auto packet_args = TensorAccessorArgs<8>();
    constexpr auto column_args = TensorAccessorArgs<packet_args.next_compile_time_args_offset()>();
    constexpr auto stats_args = TensorAccessorArgs<column_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_slot0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_slot1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_column = tt::CBIndex::c_2;
    constexpr uint32_t cb_scratch = tt::CBIndex::c_3;
    constexpr uint32_t cb_stats = tt::CBIndex::c_4;

    using ttml::metal::ops::cyclic_sdpa_bw::CyclicSchedule;
    constexpr CyclicSchedule sched(kCores);
    constexpr uint32_t kTimesteps = 2u * kCores + 1u;

    const uint32_t slot_addr[2] = {get_write_ptr(cb_slot0), get_write_ptr(cb_slot1)};
    const uint32_t column_l1 = get_write_ptr(cb_column);
    const uint32_t scratch_l1 = get_write_ptr(cb_scratch);
    Words column = reinterpret_cast<Words>(column_l1);
    Words scratch = reinterpret_cast<Words>(scratch_l1);

    volatile tt_l1_ptr uint32_t* arrive_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(arrive_sem_id));
    volatile tt_l1_ptr uint32_t* release_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(release_sem_id));
    const uint64_t arrive_noc_addr = get_noc_addr(coord_noc_x, coord_noc_y, get_semaphore(arrive_sem_id));
    const uint64_t release_mcast_addr = get_noc_multicast_addr(
        mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, get_semaphore(release_sem_id));

    const auto packets = TensorAccessor(packet_args, packet_addr, kPacketBytes);
    const auto columns = TensorAccessor(column_args, column_addr, kPacketBytes);
    const auto stats = TensorAccessor(stats_args, stats_addr, kPacketBytes);

    // Per-core counters, checked against the paper's traffic lemmas.
    uint32_t n_row_loads = 0;
    uint32_t n_row_spills = 0;
    uint32_t n_forwards_sent = 0;
    uint32_t n_forwards_received = 0;
    uint32_t n_self_forwards = 0;
    uint32_t n_column_changes = 0;

#ifdef TRANSPORT_RELAY
    using ttml::metal::ops::cyclic_sdpa_bw::kNoCore;
    using ttml::metal::ops::cyclic_sdpa_bw::snake_neighbors;

    const auto neighbors = snake_neighbors(kCores, my_core);

    volatile tt_l1_ptr uint32_t* ready_sem[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sem_id[0])),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sem_id[1]))};

    // credit_*[me] counts the slot releases of one particular receiver. Which
    // receiver depends only on the snake, so three words cover every core.
    volatile tt_l1_ptr uint32_t* credit_from_prev =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_prev_sem_id));
    volatile tt_l1_ptr uint32_t* credit_from_next =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_next_sem_id));
    volatile tt_l1_ptr uint32_t* credit_from_self =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_self_sem_id));

    // Forwards already sent to each possible receiver, so the n-th forward can
    // wait for the n-th grant.
    uint32_t sent_to_prev = 0;
    uint32_t sent_to_next = 0;
    uint32_t sent_to_self = 0;

    const auto noc_xy_of = [&](uint32_t core, uint32_t& x, uint32_t& y) {
        if (core == neighbors.prev) {
            x = prev_noc_x;
            y = prev_noc_y;
        } else {
            x = next_noc_x;
            y = next_noc_y;
        }
    };

    // A slot release by this core grants permission to whoever produces for
    // that slot next. Seen from the producer p, this core is p's snake
    // successor when p is this core's predecessor, and vice versa.
    const auto grant_credit_to = [&](uint32_t p) {
        if (p == my_core) {
            noc_semaphore_inc(get_noc_addr(get_semaphore(credit_self_sem_id)), 1u);
            return;
        }
        uint32_t x = 0;
        uint32_t y = 0;
        noc_xy_of(p, x, y);
        const uint32_t sem_id = (p == neighbors.prev) ? credit_next_sem_id : credit_prev_sem_id;
        noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem_id)), 1u);
    };

    // Wait for the receiver's permission for this core's next forward to it.
    const auto await_credit = [&](uint32_t r) {
        if (r == my_core) {
            noc_semaphore_wait_min(credit_from_self, ++sent_to_self);
        } else if (r == neighbors.prev) {
            noc_semaphore_wait_min(credit_from_prev, ++sent_to_prev);
        } else {
            noc_semaphore_wait_min(credit_from_next, ++sent_to_next);
        }
    };

    // Initial permissions, for destination timesteps 0 and 1 only. A slot
    // whose producer is this core itself needs none: it loads from DRAM.
    for (uint32_t u = 0; u < 2u && u < kTimesteps; ++u) {
        const auto initial = sched.producer(my_core, u);
        if (initial.internal) {
            grant_credit_to(initial.core);
        }
    }
    noc_async_atomic_barrier();
#endif

    // Column residency state. resident == 0 means nothing is resident yet;
    // visited_* record whether each owned column has been seen, which decides
    // initialise-locally against reload-from-DRAM.
    uint32_t resident = 0;
    const auto owned = sched.owned_columns(my_core);
    bool visited_first = false;
    bool visited_second = false;

    for (uint32_t t = 0; t < kTimesteps; ++t) {
        if constexpr (SKEW_ITERS > 0) {
            if ((my_core % 2u) == 1u) {
                spin(SKEW_ITERS);
            }
        }
        const auto pair = sched.pair(my_core, t);

        // ---- column state (main.tex, "Column-state management")
        if (resident != pair.j) {
            if (resident != 0) {
                // Write the old column back and complete the write before its
                // storage is reused.
                noc_async_write_page(resident - 1u, columns, column_l1);
                noc_async_write_barrier();
                ++n_column_changes;
            }
            const bool seen = (pair.j == owned.first) ? visited_first : visited_second;
            if (seen) {
                noc_async_read_page(pair.j - 1u, columns, column_l1);
                noc_async_read_barrier();
            } else {
                // First visit: initialise locally. Reading DRAM here would
                // pick up the host's poison.
                column[0] = pair.j;
                column[1] = 0u;
                column[2] = 0u;
                if (pair.j == owned.first) {
                    visited_first = true;
                } else {
                    visited_second = true;
                }
            }
            resident = pair.j;
        }

        // ---- row packet
        const uint32_t slot = slot_addr[t % 2u];
        Words packet = reinterpret_cast<Words>(slot);
#ifdef TRANSPORT_RELAY
        const auto producer = sched.producer(my_core, t);
        if (producer.internal) {
            // Inside a streak: the packet arrives over the NoC, or as a local
            // copy for the one self-transition. The tag is specific to this
            // destination timestep, so a stale tag cannot satisfy the wait.
#ifdef FAULT_STALE_TAG_OK
            // Any positive tag will do -- including the one this slot still
            // carries from its use at t - 2, which is why the contract
            // insists the tag identify the destination timestep.
            noc_semaphore_wait_min(ready_sem[t % 2u], 1u);
#else
            noc_semaphore_wait_min(ready_sem[t % 2u], t + 1u);
#endif
            if (producer.core == my_core) {
                ++n_self_forwards;
            } else {
                ++n_forwards_received;
            }
        } else {
            // A streak start: this core loads the packet itself. Its own slot
            // was released at t - 2 and its loop is sequential, so the
            // permission is local and implicit.
            noc_async_read_page(pair.i - 1u, packets, slot);
            noc_async_read_barrier();
            ++n_row_loads;
        }
#else
        noc_async_read_page(pair.i - 1u, packets, slot);
        noc_async_read_barrier();
        ++n_row_loads;
#endif

        // ---- the update this stands in for: dQ_i += ...
        const uint32_t updates = packet[1];
        const uint32_t checksum = packet[2];
        if constexpr (RMW_SPIN_ITERS > 0) {
            // Stand-in for the gradient computation, which in the real kernel
            // sits between the packet arriving and the updated packet leaving.
            if (RMW_SPIN_CORE == 0u || my_core == RMW_SPIN_CORE) {
                spin(RMW_SPIN_ITERS);
            }
        }
        packet[1] = updates + 1u;
        packet[2] = checksum + row_term(pair.i, pair.j);

#ifndef TRANSPORT_RELAY
        // The write must be visible before this core arrives, so that the
        // consumer of row i at a later timestep sees it.
        noc_async_write_page(pair.i - 1u, packets, slot);
        noc_async_write_barrier();
#endif

        // ---- barrier arrive
#ifndef FAULT_NO_BARRIER
        noc_semaphore_inc(arrive_noc_addr, 1u);
#endif

        // ---- the column-side update: dK_j, dV_j += ...
        column[1] = column[1] + 1u;
        column[2] = column[2] + column_term(pair.i, pair.j);

        // ---- release timestep t once every core has arrived
#ifndef FAULT_NO_BARRIER
        if (is_coordinator != 0u) {
            noc_semaphore_wait_min(arrive_sem, kCores * (t + 1u));
            *scratch = t + 1u;
            noc_semaphore_set_multicast_loopback_src(scratch_l1, release_mcast_addr, kCores);
            noc_async_write_barrier();
        }

        // ---- barrier wait
        noc_semaphore_wait_min(release_sem, t + 1u);
#endif

#ifdef TRANSPORT_RELAY
        // ---- forward to the next consumer, or spill at a streak end
        const uint32_t receiver = sched.next_consumer(pair.i, t);
        if (receiver != kNoCore) {
            const uint32_t next_slot = slot_addr[(t + 1u) % 2u];
#ifndef FAULT_NO_CREDIT_WAIT
            await_credit(receiver);
#endif
            if (receiver == my_core) {
                // The self-transition: a local copy between the two slots,
                // with the same lifetimes as a remote forward.
                Words dst = reinterpret_cast<Words>(next_slot);
                for (uint32_t w = 0; w < kPacketWords; ++w) {
                    dst[w] = packet[w];
                }
                noc_semaphore_set(ready_sem[(t + 1u) % 2u], t + 2u);
            } else {
                uint32_t x = 0;
                uint32_t y = 0;
                noc_xy_of(receiver, x, y);
                // Payload first, and completed, before readiness is published.
                noc_async_write(slot, get_noc_addr(x, y, next_slot), kPacketBytes);
#ifndef FAULT_READINESS_BEFORE_PAYLOAD
                noc_async_write_barrier();
#endif
                *scratch = t + 2u;
                noc_semaphore_set_remote(
                    scratch_l1, get_noc_addr(x, y, get_semaphore(ready_sem_id[(t + 1u) % 2u])));
                // The readiness word is a 4-byte write out of scratch; it must
                // complete before that word is reused.
                noc_async_write_barrier();
                ++n_forwards_sent;
            }
        } else {
            // Streak end: spill the packet and complete the write, so the
            // reload at the next streak start observes it.
            noc_async_write_page(pair.i - 1u, packets, slot);
            noc_async_write_barrier();
            ++n_row_spills;
        }

        // ---- release this slot for destination timestep t + 2
        const uint32_t released_for = t + 2u;
        if (released_for < kTimesteps) {
            const auto next_producer = sched.producer(my_core, released_for);
            if (next_producer.internal) {
                grant_credit_to(next_producer.core);
            }
            // else: this core loads it from DRAM; the permission is its own.
        }
#endif
    }

    if (resident != 0u) {
        noc_async_write_page(resident - 1u, columns, column_l1);
        noc_async_write_barrier();
    }

    Words stats_page = reinterpret_cast<Words>(get_write_ptr(cb_stats));
    stats_page[0] = n_row_loads;
    stats_page[1] = n_row_spills;
    stats_page[2] = n_forwards_sent;
    stats_page[3] = n_forwards_received;
    stats_page[4] = n_self_forwards;
    stats_page[5] = n_column_changes;
    stats_page[6] = kTimesteps;
    stats_page[7] = my_core;
    noc_async_write_page(my_core - 1u, stats, get_write_ptr(cb_stats));
    noc_async_write_barrier();
}
