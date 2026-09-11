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
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"

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

    constexpr uint32_t kCores = get_compile_time_arg_val(0);
    constexpr uint32_t arrive_sem_id = get_compile_time_arg_val(1);
    constexpr uint32_t release_sem_id = get_compile_time_arg_val(2);
    constexpr auto packet_args = TensorAccessorArgs<3>();
    constexpr auto column_args = TensorAccessorArgs<packet_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_slot0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_slot1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_column = tt::CBIndex::c_2;
    constexpr uint32_t cb_scratch = tt::CBIndex::c_3;

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
        noc_async_read_page(pair.i - 1u, packets, slot);
        noc_async_read_barrier();

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

        // The write must be visible before this core arrives, so that the
        // consumer of row i at a later timestep sees it.
        noc_async_write_page(pair.i - 1u, packets, slot);
        noc_async_write_barrier();

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
    }

    if (resident != 0u) {
        noc_async_write_page(resident - 1u, columns, column_l1);
        noc_async_write_barrier();
    }
}
