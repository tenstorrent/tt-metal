// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BENCH — rms_norm's cross-core combine, CONTRIBUTOR half of the gather.
//
// The one knob under test is the LANDING-PAGE MAP, which decides how many NoC
// transactions the same bytes take:
//
//   PAGING_ROWMAJOR (the shipped op)
//       page(r) = (r % OWN_ROWS) * NSLICE + slice_index
//       The OWN_ROWS tiles this core sends to ONE owner land at page stride
//       NSLICE, so they cannot be merged: B separate STAT_BYTES writes.
//
//   PAGING_COALESCED (the candidate)
//       page(r) = slice_index * OWN_ROWS + (r % OWN_ROWS)
//       A contributor's OWN_ROWS tiles for one owner are now CONTIGUOUS both at
//       the source (cb_sq_partials is row-major in r) and at the destination, so
//       they merge into ONE write of OWN_ROWS * STAT_BYTES: NUM_OWNERS writes
//       instead of B.  Identical bytes, identical destinations, identical
//       arithmetic downstream — only the transaction count changes.
//       At NUM_OWNERS == 1 this degenerates to ONE B-tile-wide write per
//       contributor (the "push it all the way" form).
//
// GATHER_CHUNK: 0 = one barrier after the whole burst (the shipped op).  N > 0 =
// a noc_async_write_barrier every N transactions, the writer-side analogue of the
// op's DM_CHUNK_TILES store batching, measured here because the gather is
// issue-bound and a mid-burst barrier trades issue overlap for less in-flight
// pressure.
//
// The destination address is derived from THIS core's own cb_gathered write
// pointer, which is valid because every CB in this program is declared on one
// common core set, so the L1 map is identical on every participating core.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

constexpr uint32_t cb_sq_partials = 2;
constexpr uint32_t cb_gathered = 4;

constexpr uint32_t PAGING_COALESCED = 1;
constexpr uint32_t MAX_OWNERS = 64;

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NSLICE = get_compile_time_arg_val(1);
    constexpr uint32_t OWN_ROWS = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_OWNERS = get_compile_time_arg_val(3);
    constexpr uint32_t STAT_BYTES = get_compile_time_arg_val(4);
    constexpr uint32_t GATHER_SEM = get_compile_time_arg_val(5);
    constexpr uint32_t PAGING = get_compile_time_arg_val(6);
    constexpr uint32_t GATHER_CHUNK = get_compile_time_arg_val(7);
    // Deliberately SPLIT every transaction into SPLIT equal pieces: the opposite
    // direction from coalescing, same bytes and destinations.  This is the control
    // that tells us whether this stage costs per TRANSACTION or per BYTE — a flat
    // issue time across 0.5x / 1x / 2x the transaction count means the coalescing
    // lever has nothing to pull on.
    constexpr uint32_t SPLIT = get_compile_time_arg_val(8);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t slice_index = get_arg_val<uint32_t>(1);
    const uint32_t owner_xy_base = 2;

    uint32_t owner_x[MAX_OWNERS];
    uint32_t owner_y[MAX_OWNERS];
    for (uint32_t o = 0; o < NUM_OWNERS; ++o) {
        owner_x[o] = get_arg_val<uint32_t>(owner_xy_base + 2 * o);
        owner_y[o] = get_arg_val<uint32_t>(owner_xy_base + 2 * o + 1);
    }

    Noc noc;
    Semaphore<> gather_progress(GATHER_SEM);

    // Captured before any push/pop: the CB base, identical on every core of the group.
    const uint32_t gather_base = get_write_ptr(cb_gathered);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        {
            MaybeDeviceZoneScope("wr_stat_wait");
            cb_wait_front(cb_sq_partials, B);
        }
        const uint32_t src = get_read_ptr(cb_sq_partials);
        {
            MaybeDeviceZoneScope("wr_gather_issue");
            uint32_t pending = 0;
            // `emit` is the one place a transaction is issued, so SPLIT applies to
            // every paging uniformly.
            auto emit = [&pending](uint32_t s_addr, uint64_t d_addr, uint32_t bytes) {
                constexpr uint32_t N = SPLIT < 1 ? 1 : SPLIT;
                const uint32_t piece = bytes / N;
                for (uint32_t i = 0; i < N; ++i) {
                    noc_async_write(s_addr + i * piece, d_addr + i * piece, piece);
                    if constexpr (GATHER_CHUNK > 0) {
                        if (++pending == GATHER_CHUNK) {
                            noc_async_write_barrier();
                            pending = 0;
                        }
                    }
                }
            };
            if constexpr (PAGING == PAGING_COALESCED) {
                // NUM_OWNERS wide writes: owner o takes rows [o*OWN_ROWS, (o+1)*OWN_ROWS)
                // from a contiguous source run into a contiguous destination run.
                for (uint32_t o = 0; o < NUM_OWNERS; ++o) {
                    emit(
                        src + o * OWN_ROWS * STAT_BYTES,
                        get_noc_addr(owner_x[o], owner_y[o], gather_base + slice_index * OWN_ROWS * STAT_BYTES),
                        OWN_ROWS * STAT_BYTES);
                }
            } else {
                // B single-tile writes: the shipped op.
                for (uint32_t r = 0; r < B; ++r) {
                    const uint32_t owner = r / OWN_ROWS;
                    const uint32_t page = (r % OWN_ROWS) * NSLICE + slice_index;
                    emit(
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
            // One unicast atomic per owner — the Perf-1 winner (a multicast atomic
            // serializes its path reservation against every other row-group's).
            // Identical in every variant here; not under test.
            MaybeDeviceZoneScope("wr_gather_signal");
            for (uint32_t o = 0; o < NUM_OWNERS; ++o) {
                gather_progress.up(noc, owner_x[o], owner_y[o], 1);
            }
        }
        cb_pop_front(cb_sq_partials, B);
    }
}
