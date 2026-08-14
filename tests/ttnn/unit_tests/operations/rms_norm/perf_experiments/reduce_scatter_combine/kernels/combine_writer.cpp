// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BENCH — rms_norm's cross-core combine, writer (NoC1) half: the
// contributor side of the gather.
//
//   V_FLAT_ROOT  : all B stat tiles of this core go to the ROOT, page r*s + slice.
//                  One progress increment (unicast atomic) to the root.
//   V_SCATTER_*  : stat tile r goes to the core that OWNS row r (owner = r/OWN_ROWS),
//                  page (r % OWN_ROWS)*s + slice on THAT core.  The transaction
//                  COUNT is unchanged (B single-tile writes); only the destination
//                  set changes, from 1 core to num_owners cores.
//
// The progress signal is the one place the scatter could have cost s atomics per
// contributor instead of 1.  It does not: one EXCLUDE-source multicast atomic
// covers the whole row-group and one loopback atomic covers this core, so the
// signal stays O(1) in s.  (A local `up()` would be a non-atomic read-modify-write
// racing the inbound increments — hence the loopback.)  Non-owners in the box get
// an increment nobody waits on, which is harmless.
//
// The destination address is this core's OWN cb_gathered write pointer: every CB in
// the program is declared on one common core set, so the L1 map is identical on
// every participating core.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

using namespace dataflow_kernel_lib;

constexpr uint32_t cb_sq_partials = 2;
constexpr uint32_t cb_gathered = 4;

constexpr uint32_t V_FLAT_ROOT = 0;

constexpr uint32_t MAX_OWNERS = 64;

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NSLICE = get_compile_time_arg_val(1);
    constexpr uint32_t OWN_ROWS = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_OWNERS = get_compile_time_arg_val(3);
    constexpr uint32_t VARIANT = get_compile_time_arg_val(4);
    constexpr uint32_t STAT_BYTES = get_compile_time_arg_val(5);
    constexpr uint32_t GATHER_SEM = get_compile_time_arg_val(6);
    // How a contributor tells the owners "my partials for this block have landed".
    //   0 = ONE exclude-source multicast atomic over the row-group + one loopback
    //       atomic for self: O(1) transactions in s.
    //   1 = one unicast atomic per owner: O(num_owners) transactions, but no
    //       multicast path reservation.
    // Measured both: the multicast atomic is the cheaper signal on a 1-D row-group
    // (s <= 8, one grid row) and the more expensive one on a 2-D row-group.
    constexpr uint32_t SIGNAL_UNICAST = get_compile_time_arg_val(7);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t slice_index = get_arg_val<uint32_t>(1);
    const uint32_t root_x = get_arg_val<uint32_t>(2);
    const uint32_t root_y = get_arg_val<uint32_t>(3);
    const uint32_t rect_xlo = get_arg_val<uint32_t>(4);
    const uint32_t rect_ylo = get_arg_val<uint32_t>(5);
    const uint32_t rect_xhi = get_arg_val<uint32_t>(6);
    const uint32_t rect_yhi = get_arg_val<uint32_t>(7);
    const uint32_t dests_excl = get_arg_val<uint32_t>(8);

    uint32_t owner_x[MAX_OWNERS];
    uint32_t owner_y[MAX_OWNERS];
    if constexpr (VARIANT != V_FLAT_ROOT) {
        for (uint32_t o = 0; o < NUM_OWNERS; ++o) {
            owner_x[o] = get_arg_val<uint32_t>(9 + 2 * o);
            owner_y[o] = get_arg_val<uint32_t>(10 + 2 * o);
        }
    }

    Noc noc;
    Semaphore<> gather_progress(GATHER_SEM);
    const McastRect<> rect(rect_xlo, rect_ylo, rect_xhi, rect_yhi);
    const auto& rb = rect.bounds();

    // Captured before any push/pop: the CB base, identical on every core of the box.
    const uint32_t gather_base = get_write_ptr(cb_gathered);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        {
            MaybeDeviceZoneScope("wr_stat_wait");
            cb_wait_front(cb_sq_partials, B);
        }
        const uint32_t src = get_read_ptr(cb_sq_partials);
        {
            MaybeDeviceZoneScope("wr_gather_issue");
            if constexpr (VARIANT == V_FLAT_ROOT) {
                for (uint32_t r = 0; r < B; ++r) {
                    const uint32_t page = r * NSLICE + slice_index;
                    noc_async_write(
                        src + r * STAT_BYTES,
                        get_noc_addr(root_x, root_y, gather_base + page * STAT_BYTES),
                        STAT_BYTES);
                }
            } else {
                for (uint32_t r = 0; r < B; ++r) {
                    const uint32_t owner = r / OWN_ROWS;
                    const uint32_t page = (r % OWN_ROWS) * NSLICE + slice_index;
                    noc_async_write(
                        src + r * STAT_BYTES,
                        get_noc_addr(owner_x[owner], owner_y[owner], gather_base + page * STAT_BYTES),
                        STAT_BYTES);
                }
            }
        }
        {
            MaybeDeviceZoneScope("wr_gather_barrier");
            noc_async_write_barrier();
        }
        {
            MaybeDeviceZoneScope("wr_gather_signal");
            if constexpr (VARIANT == V_FLAT_ROOT || NUM_OWNERS == 1) {
                gather_progress.up(noc, root_x, root_y, 1);
            } else if constexpr (SIGNAL_UNICAST) {
                for (uint32_t o = 0; o < NUM_OWNERS; ++o) {
                    gather_progress.up(noc, owner_x[o], owner_y[o], 1);
                }
            } else {
                // No atomic barrier: the baseline's single unicast `up()` is not
                // barriered either, so barriering these would charge the scatter for
                // synchronization the baseline never pays.  The receiver's wait_min is
                // the arrival proof, and the preceding write barrier is what orders the
                // DATA in front of the signal.
                gather_progress.inc_multicast(noc, rb.sx, rb.sy, rb.ex, rb.ey, 1, dests_excl);
                gather_progress.up(noc, my_x[noc_index], my_y[noc_index], 1);
            }
        }
        cb_pop_front(cb_sq_partials, B);
    }
}
