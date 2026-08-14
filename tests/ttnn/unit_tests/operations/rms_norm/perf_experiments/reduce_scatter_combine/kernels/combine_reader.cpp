// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BENCH — rms_norm's cross-core combine, reader (NoC0) half.
//
// Three topologies for the SAME collective ("every core of a row-group needs the
// finalized 1/rms of every row of the block"):
//
//   V_FLAT_ROOT     (baseline == the shipped op)
//       root gathers s*B stat tiles, reduces ALL B rows, mcasts B tiles.
//   V_SCATTER_ROOT
//       `num_owners` cores each gather s*OWN_ROWS and reduce their OWN rows, then
//       unicast their finalized rows to the root, which mcasts the assembled B
//       tiles.  Same broadcast machinery as the baseline (mcast_pipe SenderPipe /
//       ReceiverPipe), so a baseline-vs-this delta isolates "scatter the reduce".
//   V_SCATTER_MCAST
//       same scatter, but each owner BROADCASTS its own rows directly to the
//       whole row-group; every core waits for `num_owners` arrivals.
//
// RAW-LLK / helper bypass, and WHY (V_SCATTER_MCAST only):
//   `mcast_pipe`'s ReceiverPipe cannot express "wait for N senders in one round":
//   `receive()` waits a single Flag, or (DataReadySignal::Counter) `wait_min(++round_)`
//   — one increment per round.  An s-sender broadcast into DISJOINT page ranges of
//   one landing CB needs a receiver that waits a COUNT.  So the data mcast is a raw
//   `noc_async_write_multicast_loopback_src` and the readiness signal is a raw
//   counted semaphore (mcast atomic inc to the group + one loopback inc for self).
//   McastRect is still used, for its NoC-corner ordering.
//
// Deadlock argument for both candidates: a reducer's gather depends only on every
// core's LOCAL Sum(x^2) (the writers never wait on the combine), and the broadcast
// depends only on the reducers' gathers.  No core waits for a broadcast before
// sending, so the dependency graph stays acyclic — the same shape of argument that
// makes the flat root safe.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib;

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_sq_partials = 2;
constexpr uint32_t cb_gathered = 4;
constexpr uint32_t cb_stat_out = 5;
constexpr uint32_t cb_rms_recip = 6;
constexpr uint32_t cb_scaler = 7;
constexpr uint32_t cb_bcast_stage = 8;

constexpr uint32_t V_FLAT_ROOT = 0;
constexpr uint32_t V_SCATTER_ROOT = 1;
constexpr uint32_t V_SCATTER_MCAST = 2;
// V_SCATTER_UCAST: the same s-sender broadcast, but each owner UNICASTS its rows to
// every core of the group instead of multicasting them.  Measured because the s
// concurrent multicasts of V_SCATTER_MCAST were unexpectedly slow: each carries a
// path reservation over the SAME line, and at own_rows == 1 each moves only one
// tile, so the per-transaction cost is never amortized.  s unicasts pay s x the
// wire bytes but reserve nothing.
constexpr uint32_t V_SCATTER_UCAST = 3;
constexpr uint32_t MAX_SLICES = 64;

void kernel_main() {
    constexpr auto mc = McastArgs</*CT=*/0, /*RT=*/0>();
    constexpr uint32_t CT = mc.next_compile_time_args_offset();

    constexpr uint32_t S = get_compile_time_arg_val(CT + 0);
    constexpr uint32_t B = get_compile_time_arg_val(CT + 1);
    constexpr uint32_t NSLICE = get_compile_time_arg_val(CT + 2);
    constexpr uint32_t OWN_ROWS = get_compile_time_arg_val(CT + 3);
    constexpr uint32_t NUM_OWNERS = get_compile_time_arg_val(CT + 4);
    constexpr uint32_t VARIANT = get_compile_time_arg_val(CT + 5);
    constexpr uint32_t STAT_BYTES = get_compile_time_arg_val(CT + 6);
    constexpr uint32_t GATHER_SEM = get_compile_time_arg_val(CT + 7);
    constexpr uint32_t BCAST_SEM = get_compile_time_arg_val(CT + 8);
    constexpr uint32_t STAT_READY_SEM = get_compile_time_arg_val(CT + 9);
    constexpr uint32_t DRAIN_SEM = get_compile_time_arg_val(CT + 10);
    constexpr uint32_t IN_WAIT_TILES = get_compile_time_arg_val(CT + 11);
    // Prices the landing-buffer REUSE protocol the candidate would need in the real
    // op (a per-block group barrier so no owner's next broadcast overtakes a
    // consumer still reading the current one).  Off by default; the bench's landing
    // buffer is per-block-disjoint, so nothing needs it for correctness here.
    constexpr uint32_t PRICE_DRAIN_BARRIER = get_compile_time_arg_val(CT + 12);

    constexpr uint32_t BLOCK_TILES = B * S;

    constexpr uint32_t RT = mc.next_runtime_args_offset();
    const uint32_t num_blocks = get_arg_val<uint32_t>(RT + 0);
    const uint32_t is_root = get_arg_val<uint32_t>(RT + 1);
    const uint32_t is_owner = get_arg_val<uint32_t>(RT + 2);
    const uint32_t my_first_row = get_arg_val<uint32_t>(RT + 3);  // first block-row this core reduces
    const uint32_t rect_xlo = get_arg_val<uint32_t>(RT + 4);      // row-group bbox, VIRTUAL coords
    const uint32_t rect_ylo = get_arg_val<uint32_t>(RT + 5);
    const uint32_t rect_xhi = get_arg_val<uint32_t>(RT + 6);
    const uint32_t rect_yhi = get_arg_val<uint32_t>(RT + 7);
    const uint32_t dests_excl = get_arg_val<uint32_t>(RT + 8);  // s-1 (host-computed; never area())
    const uint32_t dests_incl = get_arg_val<uint32_t>(RT + 9);  // s
    const uint32_t root_x = get_arg_val<uint32_t>(RT + 10);
    const uint32_t root_y = get_arg_val<uint32_t>(RT + 11);

    uint32_t peer_x[MAX_SLICES];
    uint32_t peer_y[MAX_SLICES];
    if constexpr (VARIANT == V_SCATTER_UCAST) {
        for (uint32_t c = 0; c < NSLICE; ++c) {
            peer_x[c] = get_arg_val<uint32_t>(RT + 12 + 2 * c);
            peer_y[c] = get_arg_val<uint32_t>(RT + 13 + 2 * c);
        }
    }

    Noc noc;

    // =====================================================================
    // prologue: the reduce scaler, then publish the whole resident shard.
    // =====================================================================
    {
        MaybeDeviceZoneScope("rd_stat_consts");
        calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();
    }
    // cb_rms_recip IS the caller's resident output shard: the broadcast lands
    // straight into it, so the bench pays no copy for "the stat reached this core".
    // Addressed RAW from the base (block b, row r -> page b*B + r); the push/pop
    // cycle below carries only the handshake.
    const uint32_t recip_base = get_write_ptr(cb_rms_recip);
    const uint32_t bcast_stage_base = (VARIANT == V_SCATTER_ROOT) ? get_write_ptr(cb_bcast_stage) : 0;
    {
        MaybeDeviceZoneScope("rd_publish");
        cb_reserve_back(cb_in, IN_WAIT_TILES);
        cb_push_back(cb_in, IN_WAIT_TILES);
    }

    const McastRect<> rect(rect_xlo, rect_ylo, rect_xhi, rect_yhi);
    const auto& rb = rect.bounds();

    Semaphore<> gather_progress(GATHER_SEM);
    Semaphore<> bcast_progress(BCAST_SEM);
    Semaphore<> stat_ready(STAT_READY_SEM);
    Semaphore<> drain(DRAIN_SEM);

    if constexpr (VARIANT == V_SCATTER_MCAST || VARIANT == V_SCATTER_UCAST) {
        // ---------------- s-sender direct broadcast (raw) ----------------
        for (uint32_t block = 0; block < num_blocks; ++block) {
            if (block > 0) {
                cb_reserve_back(cb_in, BLOCK_TILES);
                cb_push_back(cb_in, BLOCK_TILES);
            }
            if (is_owner) {
                cb_reserve_back(cb_gathered, NSLICE * OWN_ROWS);
                {
                    MaybeDeviceZoneScope("rd_gather_wait");
                    gather_progress.wait_min((block + 1) * NSLICE);
                }
                cb_push_back(cb_gathered, NSLICE * OWN_ROWS);
                {
                    MaybeDeviceZoneScope("rd_bcast_wait_stat");
                    cb_wait_front(cb_stat_out, OWN_ROWS);
                }
                {
                    MaybeDeviceZoneScope("rd_bcast_send");
                    const uint32_t dst = recip_base + (block * B + my_first_row) * STAT_BYTES;
                    const uint32_t src = get_read_ptr(cb_stat_out);
                    if constexpr (VARIANT == V_SCATTER_UCAST) {
                        for (uint32_t c = 0; c < NSLICE; ++c) {
                            noc_async_write(src, get_noc_addr(peer_x[c], peer_y[c], dst), OWN_ROWS * STAT_BYTES);
                        }
                    } else {
                        noc_async_write_multicast_loopback_src(
                            src,
                            get_noc_multicast_addr(rb.sx, rb.sy, rb.ex, rb.ey, dst, noc_index),
                            OWN_ROWS * STAT_BYTES,
                            dests_incl);
                    }
                    noc_async_write_barrier();  // data before flag
                    // Counted readiness: everyone else through one mcast atomic, self
                    // through a loopback atomic (a local `up()` is a non-atomic RMW and
                    // would race the inbound mcast increments).  No atomic barrier here:
                    // the increments are fire-and-forget (each receiver's wait_min is the
                    // arrival proof) and the kernel-exit teardown drains them — and the
                    // baseline's unicast `up()` is not barriered either, so barriering
                    // these would charge the candidate for sync the baseline skips.
                    bcast_progress.inc_multicast(noc, rb.sx, rb.sy, rb.ex, rb.ey, 1, dests_excl);
                    bcast_progress.up(noc, my_x[noc_index], my_y[noc_index], 1);
                }
                cb_pop_front(cb_stat_out, OWN_ROWS);
            }
            cb_reserve_back(cb_rms_recip, B);
            {
                MaybeDeviceZoneScope("rd_bcast_recv");
                bcast_progress.wait_min((block + 1) * NUM_OWNERS);
            }
            cb_push_back(cb_rms_recip, B);
            if constexpr (PRICE_DRAIN_BARRIER) {
                // The reuse protocol, priced: one group barrier per block.
                MaybeDeviceZoneScope("rd_drain_barrier");
                drain.inc_multicast(noc, rb.sx, rb.sy, rb.ex, rb.ey, 1, dests_excl);
                drain.up(noc, my_x[noc_index], my_y[noc_index], 1);
                drain.wait_min((block + 1) * NSLICE);
            }
        }
    } else {
        // ---------------- flat root, and scatter-then-root ----------------
        auto sender_pipe = mc.sender(noc);
        auto receiver_pipe = mc.receiver(noc);
        for (uint32_t block = 0; block < num_blocks; ++block) {
            if (block > 0) {
                cb_reserve_back(cb_in, BLOCK_TILES);
                cb_push_back(cb_in, BLOCK_TILES);
            }
            if (is_owner) {
                cb_reserve_back(cb_gathered, NSLICE * OWN_ROWS);
                {
                    MaybeDeviceZoneScope("rd_gather_wait");
                    gather_progress.wait_min((block + 1) * NSLICE);
                }
                cb_push_back(cb_gathered, NSLICE * OWN_ROWS);
            }
            if constexpr (VARIANT == V_SCATTER_ROOT) {
                if (is_owner) {
                    {
                        MaybeDeviceZoneScope("rd_bcast_wait_stat");
                        cb_wait_front(cb_stat_out, OWN_ROWS);
                    }
                    {
                        // Funnel the finalized rows to the root's staging buffer.  The
                        // root is owner 0 and takes this path too (loopback write), so
                        // the code stays uniform and the count stays NUM_OWNERS.
                        MaybeDeviceZoneScope("rd_stat_to_root");
                        const uint32_t dst = bcast_stage_base + (block * B + my_first_row) * STAT_BYTES;
                        noc_async_write(
                            get_read_ptr(cb_stat_out), get_noc_addr(root_x, root_y, dst), OWN_ROWS * STAT_BYTES);
                        noc_async_write_barrier();
                        stat_ready.up(noc, root_x, root_y, 1);
                    }
                    cb_pop_front(cb_stat_out, OWN_ROWS);
                }
                if (is_root) {
                    {
                        MaybeDeviceZoneScope("rd_bcast_wait_stat_root");
                        stat_ready.wait_min((block + 1) * NUM_OWNERS);
                    }
                    cb_reserve_back(cb_rms_recip, B);
                    {
                        MaybeDeviceZoneScope("rd_bcast_send");
                        sender_pipe.send(
                            bcast_stage_base + block * B * STAT_BYTES,
                            recip_base + block * B * STAT_BYTES,
                            B * STAT_BYTES);
                    }
                    cb_push_back(cb_rms_recip, B);
                } else {
                    cb_reserve_back(cb_rms_recip, B);
                    {
                        MaybeDeviceZoneScope("rd_bcast_recv");
                        receiver_pipe.receive();
                    }
                    cb_push_back(cb_rms_recip, B);
                }
            } else {
                // V_FLAT_ROOT — the shipped op, verbatim in shape.
                if (is_root) {
                    {
                        MaybeDeviceZoneScope("rd_bcast_wait_stat");
                        cb_wait_front(cb_stat_out, B);
                    }
                    cb_reserve_back(cb_rms_recip, B);
                    {
                        MaybeDeviceZoneScope("rd_bcast_send");
                        sender_pipe.send(
                            get_read_ptr(cb_stat_out), recip_base + block * B * STAT_BYTES, B * STAT_BYTES);
                    }
                    cb_push_back(cb_rms_recip, B);
                    cb_pop_front(cb_stat_out, B);
                } else {
                    cb_reserve_back(cb_rms_recip, B);
                    {
                        MaybeDeviceZoneScope("rd_bcast_recv");
                        receiver_pipe.receive();
                    }
                    cb_push_back(cb_rms_recip, B);
                }
            }
        }
    }
}
