// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the decode tile's dense projection (tt/v41/decode_dense.py DecodeDense, ``V41_DECODE_DENSE=matvec``):
// ``y[b, 32, N] = x[b, 32, K] @ W[b, K, N]`` for one tile row of x (the decode tile, one live row) and a bfp8 weight.
// At one tile row the cost is the compute's K steps, ~0.35 us a tile matmul on a core (ttnn.linear's 1D multicast
// config and a one-column-a-core reader both sit at it, 45 us for wo_b's 160 columns of 64 K tiles on 130 cores), so
// the work is cut into units of one column and one K segment of L tiles (S segments a column), unit u = j * S + s on
// core u % GRID_N, every core ~the same number of tile matmuls; segment 0's core owns the column and sums the other
// segments' fp32 partial tiles into its own.
//
//   1. ``x``: X_SENDERS cores read a stripe of its B * KT tiles each into core 0's CB, which multicasts the whole
//      row block into the same CB of every core and multicasts the go.
//   2. per unit (j, s): the core streams W[b, s L + k, n] over its segment (BLK tiles at a time, two blocks in
//      flight by transaction id, a RING-block ring) for the compute's dot product. With the weight column-sharded
//      in DRAM (COL_CONTIG: a column's K tiles one contiguous shard in one bank) a block is one request.
//   3. the owners: once every partial of this core's owned units has arrived (the senders' writes into cb_recv and
//      their semaphore increments, kernels/dense_row_writer.cpp), the received tiles are pushed to the compute.
//
// The multicast is on this RISC's NoC (NoC 0). The exit drains every read, non-posted write
// and atomic.
//
// compile_time_args = [cb_x, cb_w, cb_recv, KT, NT, B, L, BLK, RING, GRID_N, X_SENDERS, X_TILE, W_TILE,
//                      COL_CONTIG, PROBE, S, TensorAccessorArgs(x)..., (w)...]
// runtime args = [x_addr, w_addr, is_core0, core0_nx, core0_ny, gx0, gy0, gx1, gy1, sem_xgo, sem_xready,
//                      sem_recv, x_sender, x_first, x_count, n_units, n_owned, (j, s) * n_units]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t is_core0 = get_arg_val<uint32_t>(2);
    const uint32_t core0_nx = get_arg_val<uint32_t>(3);
    const uint32_t core0_ny = get_arg_val<uint32_t>(4);
    const uint32_t gx0 = get_arg_val<uint32_t>(5);
    const uint32_t gy0 = get_arg_val<uint32_t>(6);
    const uint32_t gx1 = get_arg_val<uint32_t>(7);
    const uint32_t gy1 = get_arg_val<uint32_t>(8);
    const uint32_t sem_xgo_id = get_arg_val<uint32_t>(9);
    const uint32_t sem_xready_id = get_arg_val<uint32_t>(10);
    const uint32_t sem_recv_id = get_arg_val<uint32_t>(11);
    const uint32_t x_sender = get_arg_val<uint32_t>(12);
    const uint32_t x_first = get_arg_val<uint32_t>(13);
    const uint32_t x_count = get_arg_val<uint32_t>(14);
    const uint32_t n_units = get_arg_val<uint32_t>(15);
    const uint32_t n_owned = get_arg_val<uint32_t>(16);
    constexpr uint32_t UNITS = 17;

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(2);
    constexpr uint32_t KT = get_compile_time_arg_val(3);
    constexpr uint32_t NT = get_compile_time_arg_val(4);
    constexpr uint32_t B = get_compile_time_arg_val(5);
    constexpr uint32_t L = get_compile_time_arg_val(6);
    constexpr uint32_t BLK = get_compile_time_arg_val(7);
    constexpr uint32_t RING = get_compile_time_arg_val(8);
    constexpr uint32_t GRID_N = get_compile_time_arg_val(9);
    constexpr uint32_t X_SENDERS = get_compile_time_arg_val(10);
    constexpr uint32_t X_TILE = get_compile_time_arg_val(11);
    constexpr uint32_t W_TILE = get_compile_time_arg_val(12);
    constexpr uint32_t COL_CONTIG = get_compile_time_arg_val(13);  // 1: a column's K tiles contiguous in DRAM
    constexpr uint32_t PROBE = get_compile_time_arg_val(14);       // 1: no weight reads (a timing probe)
    constexpr uint32_t S = get_compile_time_arg_val(15);           // K segments a column
    constexpr auto x_args = TensorAccessorArgs<16>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    static_assert(L % BLK == 0 && KT % L == 0, "the blocks must tile the segment, the segments K");
    static_assert(RING >= 3, "two blocks in flight beside the one the compute reads");
    constexpr uint32_t NB = L / BLK;
    constexpr uint32_t XT = B * KT;

    const auto x = TensorAccessor(x_args, x_addr, X_TILE);
    const auto w = TensorAccessor(w_args, w_addr, W_TILE);
    DataflowBuffer xb(cb_x), wb(cb_w), recv(cb_recv);
    volatile tt_l1_ptr uint32_t* sem_xgo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_xgo_id));
    volatile tt_l1_ptr uint32_t* sem_xready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_xready_id));
    volatile tt_l1_ptr uint32_t* sem_recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_recv_id));
    const uint64_t core0_xready = get_noc_addr(core0_nx, core0_ny, get_semaphore(sem_xready_id));

    xb.reserve_back(XT);
    const uint32_t x_l1 = xb.get_write_ptr();
    const uint64_t core0_x = get_noc_addr(core0_nx, core0_ny, x_l1);
    const uint32_t w_l1 = wb.get_write_ptr();

    // the weight stream: block i (counted over the units) sits in ring slot i % RING and carries transaction id
    // 1 + i % 2, so two blocks are in flight while the compute consumes the one before them
    uint32_t gb = 0, gd = 0;  // blocks issued, blocks pushed
    auto issue = [&](uint32_t u, uint32_t b) {
        const uint32_t slot = gb % RING;
        noc_async_read_set_trid(1 + (gb & 1));
        const uint32_t j = get_arg_val<uint32_t>(UNITS + 2 * u), s = get_arg_val<uint32_t>(UNITS + 2 * u + 1);
        const uint32_t bb = j / NT, n = j % NT, k0 = s * L + b * BLK;
        if constexpr (COL_CONTIG) {
            if ((PROBE & 1) == 0) {
                noc_async_read(w.get_noc_addr((bb * KT + k0) * NT + n), w_l1 + slot * BLK * W_TILE, BLK * W_TILE);
            }
        } else {
            for (uint32_t t = 0; t < BLK && (PROBE & 1) == 0; ++t) {
                noc_async_read(w.get_noc_addr((bb * KT + k0 + t) * NT + n), w_l1 + (slot * BLK + t) * W_TILE, W_TILE);
            }
        }
        ++gb;
    };
    // the first unit's first blocks before the row: the DRAM is busy while the row is gathered
    if (n_units > 0) {
        wb.reserve_back(2 * BLK);
        issue(0, 0);
        if (NB > 1) {
            issue(0, 1);
        }
    }
    // 1. the row to every core, once (X_SENDERS stripes into core 0, one multicast from core 0: the senders each
    // multicasting their stripe to the grid was measured far slower: concurrent grid
    // multicasts serialize)
    if (x_sender) {
        noc_async_read_set_trid(3);
        for (uint32_t t = x_first; t < x_first + x_count; ++t) {
            noc_async_read(x.get_noc_addr(t), x_l1 + t * X_TILE, X_TILE);
        }
        noc_async_read_barrier_with_trid(3);
        noc_async_write(x_l1 + x_first * X_TILE, core0_x + x_first * X_TILE, x_count * X_TILE);
        noc_async_write_barrier();
        noc_semaphore_inc(core0_xready, 1);
    }
    if (is_core0) {
        noc_semaphore_wait_min(sem_xready, X_SENDERS);
        noc_async_write_multicast(x_l1, get_noc_multicast_addr(gx0, gy0, gx1, gy1, x_l1), XT * X_TILE, GRID_N - 1);
        noc_async_write_barrier();
        *sem_xgo = 1;
        noc_semaphore_set_multicast(
            get_semaphore(sem_xgo_id),
            get_noc_multicast_addr(gx0, gy0, gx1, gy1, get_semaphore(sem_xgo_id)),
            GRID_N - 1);
    }
    noc_semaphore_wait_min(sem_xgo, 1);
    xb.push_back(XT);
    // 2. the units' weights over their segments
    for (uint32_t u = 0; u < n_units; ++u) {
        if (u > 0) {
            wb.reserve_back(2 * BLK);
            issue(u, 0);
            if (NB > 1) {
                issue(u, 1);
            }
        }
        for (uint32_t b = 0; b < NB; ++b) {
            noc_async_read_barrier_with_trid(1 + (gd & 1));
            ++gd;
            wb.push_back(BLK);
            if (b + 2 < NB) {
                wb.reserve_back(2 * BLK);
                issue(u, b + 2);
            }
        }
    }
    // 3. the owned units' partials from the other segments' cores
    if (S > 1 && n_owned > 0) {
        const uint32_t n = n_owned * (S - 1);
        recv.reserve_back(n);
        noc_semaphore_wait_min(sem_recv, n);
        recv.push_back(n);
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
