// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// in1 (weight) reader / multicaster + output writer of matmul_add_norm, from minimal_matmul's
// dm_in1_sender_out.cpp (transposed grid: M split over x, N over y; in1 multicasts along x).
// Added per output block (one N block per core, so a core holds its whole N slice of the rows):
//   * the residual block -> CB 4 before the block's K loop, for the compute's h = x @ W + residual
//   * partial sums of squares: the compute packs one fp32 tile per tile-row (CB 9) with the row's
//     32 partial sums in row 0. Each core writes them into row `my index` of the gathered tile
//     (CB 10) on every core of the same rows (the y column), then increments their semaphores;
//     the compute column-sums a gathered tile into the 32 rows' mean squares.
//   * h (CB 16) and rmsnorm(h) * gamma (CB 17) written to DRAM during the next block's K loop.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "matmul_dataflow_common.hpp"

namespace {
constexpr uint32_t cb_in1_id = tt::CBIndex::c_1;
constexpr uint32_t cb_res_id = tt::CBIndex::c_4;
constexpr uint32_t cb_gamma_id = tt::CBIndex::c_5;
constexpr uint32_t cb_inv_w_id = tt::CBIndex::c_6;
constexpr uint32_t cb_part_id = tt::CBIndex::c_9;
constexpr uint32_t cb_gath_id = tt::CBIndex::c_10;
constexpr uint32_t cb_one_id = tt::CBIndex::c_12;
constexpr uint32_t cb_h_id = tt::CBIndex::c_16;
constexpr uint32_t cb_y_id = tt::CBIndex::c_17;
constexpr uint32_t FWD_TRID = 1;
constexpr uint32_t XCHG_TRID = 2;
constexpr uint32_t OUT_TRID = 3;
constexpr uint32_t FP32_TILE = 4096;
constexpr uint32_t FP32_FACE = 1024;
constexpr uint32_t FACE_ROW = 16 * sizeof(uint32_t);  // one row of one fp32 face

}  // namespace

void kernel_main() {
    Noc noc;
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t padded_M_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t padded_K_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t padded_N_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(9);
    constexpr uint32_t in1_tile_size = get_compile_time_arg_val(10);
    constexpr uint32_t h_tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t y_tile_size = get_compile_time_arg_val(12);
    constexpr uint32_t res_tile_size = get_compile_time_arg_val(13);
    Semaphore<> in1_sender_semaphore(get_compile_time_arg_val(14));
    Semaphore<> in1_receiver_semaphore(get_compile_time_arg_val(15));
    Semaphore<> in1_valid_semaphore(get_compile_time_arg_val(16));
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(17);
    constexpr uint32_t n_peers = get_compile_time_arg_val(18);
    const uint32_t xchg_sem_addr = get_semaphore(get_compile_time_arg_val(19));

    uint32_t argidx = 0;
    const uint32_t in1_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t res_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t inv_w_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t one_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t y_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    const uint32_t my_peer = get_arg_val<uint32_t>(argidx++);
    // IN1_MCAST: the x = 0 core multicasts each in1 block to the rest of its row (rectangle in the
    // order the NOC needs) instead of the block being relayed core to core.
    const uint32_t mcast_x0 = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_y0 = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_x1 = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_y1 = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_n = get_arg_val<uint32_t>(argidx++);
    const uint32_t row_sender_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t row_sender_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t peer_xy_idx = argidx;  // n_peers (x, y) NOC coordinates follow

    constexpr auto in1_args = TensorAccessorArgs<20>();
    constexpr auto res_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<res_args.next_compile_time_args_offset()>();
    constexpr auto inv_w_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr auto one_args = TensorAccessorArgs<inv_w_args.next_compile_time_args_offset()>();
    constexpr auto h_args = TensorAccessorArgs<one_args.next_compile_time_args_offset()>();
    constexpr auto y_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    const auto in1_reader = TensorAccessor(in1_args, in1_addr);
    const auto res_reader = TensorAccessor(res_args, res_addr);
    const auto gamma_reader = TensorAccessor(gamma_args, gamma_addr);
    const auto inv_w_reader = TensorAccessor(inv_w_args, inv_w_addr);
    const auto one_reader = TensorAccessor(one_args, one_addr);
    const auto h_writer = TensorAccessor(h_args, h_addr);
    const auto y_writer = TensorAccessor(y_args, y_addr);

    const TensorShape2D in1_shape(K_tiles, N_tiles, padded_K_tiles, padded_N_tiles);
    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t full_N_tiles_bytes = N_block_tiles * in1_tile_size;
    constexpr uint32_t gamma_tile_size = get_tile_size(cb_gamma_id);
    constexpr uint32_t scaler_tile_size = get_tile_size(cb_inv_w_id);

    CircularBuffer cb_in1(cb_in1_id);
    CircularBuffer cb_res(cb_res_id);
    CircularBuffer cb_gamma(cb_gamma_id);
    CircularBuffer cb_inv_w(cb_inv_w_id);
    CircularBuffer cb_one(cb_one_id);
    CircularBuffer cb_part(cb_part_id);
    CircularBuffer cb_gath(cb_gath_id);

    // This core's real N tiles (the last core along y may hold padding only past N_tiles) and the
    // end of its real rows: a block can run past both, and those tiles belong to other cores.
    const uint32_t n_real = (N_end_tile < N_tiles ? N_end_tile : N_tiles) - N_start_tile;
    const uint32_t m_limit = M_end_tile < M_tiles ? M_end_tile : M_tiles;

    // Resident constants: gamma for this core's columns, the 1/W and 1.0 scaler tiles.
    cb_gamma.reserve_back(N_block_tiles);
    cb_inv_w.reserve_back(1);
    cb_one.reserve_back(1);
    {
        uint32_t ptr = cb_gamma.get_write_ptr();
        for (uint32_t j = 0; j < n_real; ++j) {
            noc.async_read(
                gamma_reader, CoreLocalMem<uint32_t>(ptr), gamma_tile_size, {.page_id = N_start_tile + j}, {});
            ptr += gamma_tile_size;
        }
        noc.async_read(
            inv_w_reader, CoreLocalMem<uint32_t>(cb_inv_w.get_write_ptr()), scaler_tile_size, {.page_id = 0}, {});
        noc.async_read(
            one_reader, CoreLocalMem<uint32_t>(cb_one.get_write_ptr()), scaler_tile_size, {.page_id = 0}, {});
    }
    // The gathered tiles only ever get rows [0, n_peers) written; the rest must read as zero.
    const uint32_t gath_base = cb_gath.get_write_ptr();
    fill_zeros_async(noc, cb_gath, 2 * M_block_tiles * FP32_TILE);
    noc.write_zeros_l1_barrier();
    noc.async_read_barrier();
    cb_gamma.push_back(N_block_tiles);
    cb_inv_w.push_back(1);
    cb_one.push_back(1);

    volatile tt_l1_ptr uint32_t* xchg_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xchg_sem_addr);

    in1_valid_semaphore.set(VALID);

    volatile tt_l1_ptr uint32_t* in1_sender_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_compile_time_arg_val(14)));
    volatile tt_l1_ptr uint32_t* in1_receiver_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_compile_time_arg_val(15)));
    constexpr uint32_t out_tiles = M_block_tiles * N_block_tiles;
    CircularBuffer cb_h(cb_h_id), cb_y(cb_y_id);

    // Epilogue DRAM traffic (residual reads, h and y writes) is the bulk of this kernel's NOC work
    // and DRAM is close to saturated while it runs. Issued in blocking bursts it stalls this RISC,
    // which also relays the weights along its row, and the matmuls downstream starve. So it is
    // queued instead: `pump` issues one pending tile, and only when the NOC command buffer for it
    // is free; every wait in the weight relay polls it. The queue is drained (blocking) at the end
    // of each block, before the CBs it reads or fills are handed on.
    struct {
        uint32_t res_ptr, res_m, h_ptr, h_m, y_ptr, y_m;
        bool res, h, y;
        uint32_t cursor, end;
    } io = {};
    auto io_issue = [&](bool block) -> bool {
        while (io.cursor < io.end) {
            const uint32_t s = io.cursor % 3;
            const uint32_t t = io.cursor / 3;
            const uint32_t i = t / n_real;
            const uint32_t j = t - i * n_real;
            const uint32_t off = i * N_block_tiles + j;
            bool valid;
            uint32_t m;
            if (s == 0) {
                valid = io.res;
                m = io.res_m + i;
            } else if (s == 1) {
                valid = io.h;
                m = io.h_m + i;
            } else {
                valid = io.y;
                m = io.y_m + i;
            }
            if (!valid || m >= m_limit) {
                ++io.cursor;
                continue;
            }
            if (!block && !noc_cmd_buf_ready(noc_index, s == 0 ? read_cmd_buf : write_cmd_buf)) {
                return false;
            }
            const uint32_t page = m * N_tiles + N_start_tile + j;
#ifdef SPLIT_RESIDUAL
            // residual / h in the column-split layout: this core's slice of every row is contiguous
            const uint32_t split_page = (my_peer * M_tiles + m) * N_block_tiles + j;
#else
            const uint32_t split_page = page;
#endif
            if (s == 0) {
                noc.async_read(
                    res_reader,
                    CoreLocalMem<uint32_t>(io.res_ptr + off * res_tile_size),
                    res_tile_size,
                    {.page_id = split_page},
                    {});
            } else if (s == 1) {
#ifdef POSTED_OUT
                noc_async_write_one_packet_with_trid<true, true>(
                    io.h_ptr + off * h_tile_size, h_writer.get_noc_addr(split_page), h_tile_size, OUT_TRID);
#else
                noc.async_write(
                    CoreLocalMem<uint32_t>(io.h_ptr + off * h_tile_size),
                    h_writer,
                    h_tile_size,
                    {},
                    {.page_id = split_page});
#endif
            } else {
#ifdef POSTED_OUT
                noc_async_write_one_packet_with_trid<true, true>(
                    io.y_ptr + off * y_tile_size, y_writer.get_noc_addr(page), y_tile_size, OUT_TRID);
#else
                noc.async_write(
                    CoreLocalMem<uint32_t>(io.y_ptr + off * y_tile_size), y_writer, y_tile_size, {}, {.page_id = page});
#endif
            }
            ++io.cursor;
            if (!block) {
                return true;
            }
        }
        return false;
    };
    auto pump = [&]() { io_issue(false); };
    auto drain = [&]() { io_issue(true); };

    // Exchange the partial sums of squares of block `blk`. The compute leaves each tile-row's 32
    // partial sums in row 0 of its partial tile (two 16-value runs, faces 0 and 1); this core's
    // runs go straight into row `my_peer` of the matching gathered tile on every core of these
    // rows (itself included), into the half of the gathered CB for the block's parity. Rows past
    // n_peers stay zero, so the compute's column sum over a gathered tile is the full row statistic.
    auto exchange = [&](uint32_t blk) {
        while (!cb_part.pages_available_at_front(M_block_tiles)) {
            pump();
        }
        cb_part.wait_front(M_block_tiles);
        const uint32_t part = cb_part.get_read_ptr();
        const uint32_t gdst = gath_base + (blk & 1) * M_block_tiles * FP32_TILE + my_peer * FACE_ROW;
        noc_async_write_set_trid(XCHG_TRID);
        for (uint32_t p = 0; p < n_peers; ++p) {
            const uint32_t px = get_arg_val<uint32_t>(peer_xy_idx + 2 * p);
            const uint32_t py = get_arg_val<uint32_t>(peer_xy_idx + 2 * p + 1);
            for (uint32_t i = 0; i < M_block_tiles; ++i) {
                const uint32_t src = part + i * FP32_TILE;
                const uint32_t dst = gdst + i * FP32_TILE;
                noc_async_write(src, get_noc_addr(px, py, dst), FACE_ROW);
                noc_async_write(src + FP32_FACE, get_noc_addr(px, py, dst + FP32_FACE), FACE_ROW);
            }
        }
        noc_async_write_barrier_with_trid(XCHG_TRID);
        noc_async_write_set_trid(0);
        cb_part.pop_front(M_block_tiles);
        for (uint32_t p = 0; p < n_peers; ++p) {
            const uint32_t px = get_arg_val<uint32_t>(peer_xy_idx + 2 * p);
            const uint32_t py = get_arg_val<uint32_t>(peer_xy_idx + 2 * p + 1);
            noc_semaphore_inc(get_noc_addr(px, py, xchg_sem_addr), 1);
        }
        while (*xchg_sem < n_peers * (blk + 1)) {
            pump();
        }
        cb_gath.reserve_back(M_block_tiles);
        cb_gath.push_back(M_block_tiles);
    };

    // The compute runs block i's matmul, then block i - 1's second pass, then block i's first pass
    // (see compute_add_norm.cpp). So in block i's K loop, once the first two in1 K blocks are
    // pushed (all the in1 CB holds): block i - 1's partial sums exist and the exchange runs,
    // hidden behind block i's matmul; the residual CB is free (the first pass consumed it); and
    // block i - 1's h and block i - 2's y are ready. Those become the block's queued I/O.
    // One N block per core: the in0 kernel starts every M block at K block 0 (it only alternates
    // direction between N blocks of one M block), so the K order here is always forward.
    constexpr uint32_t xchg_k = K_num_blocks > 1 ? 1 : 0;
    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        const uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        const bool has_h = m_block_iter >= 1;
        const bool has_y = m_block_iter >= 2;
        const uint32_t n_tile = N_start_tile;
        const uint32_t n_tile_end = N_end_tile;
        const uint32_t current_N_tiles_bytes = (n_tile_end - n_tile) * in1_tile_size;

        for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
            while (!cb_in1.pages_reservable_at_back(in1_block_num_tiles)) {
                pump();
            }
            cb_in1.reserve_back(in1_block_num_tiles);
            uint32_t in1_start_address = cb_in1.get_write_ptr();
            if constexpr (is_injector_core) {
                read_in1_block_sync<K_block_tiles, N_block_tiles>(
                    in1_reader,
                    in1_shape,
                    cb_in1_id,
                    in1_tile_size,
                    k_block * K_block_tiles,
                    (k_block + 1) * K_block_tiles,
                    n_tile,
                    n_tile_end);
            } else {
                in1_receiver_semaphore.set(INVALID);
#ifdef IN1_MCAST
                in1_sender_semaphore.up(noc, row_sender_x, row_sender_y, 1);
#else
                in1_sender_semaphore.up(noc, in1_sender_noc_x, in1_sender_noc_y, 1);
#endif
                while (*in1_receiver_sem != VALID) {
                    pump();
                }
            }
            cb_in1.push_back(in1_block_num_tiles);
#ifdef IN1_MCAST
            if (is_injector_core && mcast_n > 0) {
                while (*in1_sender_sem != mcast_n) {
                    pump();
                }
                in1_sender_semaphore.set(0);
                noc_async_write_set_trid(FWD_TRID);
                noc.async_write_multicast(
                    CoreLocalMem<uint32_t>(in1_start_address),
                    MulticastEndpoint{},
                    in1_block_num_tiles * in1_tile_size,
                    mcast_n,
                    {},
                    {.noc_x_start = mcast_x0,
                     .noc_y_start = mcast_y0,
                     .noc_x_end = mcast_x1,
                     .noc_y_end = mcast_y1,
                     .addr = in1_start_address},
                    true);
                noc_async_write_flushed_with_trid(FWD_TRID);
                noc_async_write_set_trid(0);
                in1_valid_semaphore.set_multicast(noc, mcast_x0, mcast_y0, mcast_x1, mcast_y1, mcast_n);
            }
            if (false) {
#else
            if (!is_sink_core) {
#endif
                while (*in1_sender_sem != 1) {
                    pump();
                }
                in1_sender_semaphore.set(0);
                // Own transaction id: the flush must not also wait for queued epilogue writes.
                noc_async_write_set_trid(FWD_TRID);
                for (uint32_t i = 0; i < K_block_tiles; i++) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(in1_start_address),
                        UnicastEndpoint{},
                        current_N_tiles_bytes,
                        {},
                        {.noc_x = in1_dest_noc_x, .noc_y = in1_dest_noc_y, .addr = in1_start_address});
                    in1_start_address += full_N_tiles_bytes;
                }
                noc_async_write_flushed_with_trid(FWD_TRID);
                noc_async_write_set_trid(0);
                in1_valid_semaphore.relay_unicast(noc, in1_receiver_semaphore, in1_dest_noc_x, in1_dest_noc_y);
            }

            if (k_block == xchg_k) {
                if (has_h) {
                    exchange(m_block_iter - 1);
                }
                cb_res.reserve_back(out_tiles);
                io.res = true;
                io.res_ptr = cb_res.get_write_ptr();
                io.res_m = m_tile;
                io.h = has_h;
                io.y = has_y;
                if (has_h) {
                    cb_h.wait_front(out_tiles);
                    io.h_ptr = cb_h.get_read_ptr();
                    io.h_m = m_tile - M_block_tiles;
                }
                if (has_y) {
                    cb_y.wait_front(out_tiles);
                    io.y_ptr = cb_y.get_read_ptr();
                    io.y_m = m_tile - 2 * M_block_tiles;
                }
                io.cursor = 0;
                io.end = 3 * M_block_tiles * n_real;
            }
            pump();
        }
        drain();
        noc.async_read_barrier();
        cb_res.push_back(out_tiles);
        noc.async_writes_flushed();
        if (has_h) {
            cb_h.pop_front(out_tiles);
        }
        if (has_y) {
            cb_y.pop_front(out_tiles);
        }
        io.res = io.h = io.y = false;
    }
    // Tail: the last block's exchange and h, the last two blocks' y.
    const uint32_t last_m_tile = M_start_tile + (M_blocks_per_core - 1) * M_block_tiles;
    exchange(M_blocks_per_core - 1);
    cb_h.wait_front(out_tiles);
    io = {};
    io.h = true;
    io.h_ptr = cb_h.get_read_ptr();
    io.h_m = last_m_tile;
    io.end = 3 * M_block_tiles * n_real;
    drain();
    for (uint32_t b = (M_blocks_per_core >= 2 ? M_blocks_per_core - 2 : 0); b < M_blocks_per_core; ++b) {
        cb_y.wait_front(out_tiles);
        io = {};
        io.y = true;
        io.y_ptr = cb_y.get_read_ptr();
        io.y_m = M_start_tile + b * M_block_tiles;
        io.end = 3 * M_block_tiles * n_real;
        drain();
        noc.async_writes_flushed();
        cb_y.pop_front(out_tiles);
    }
    noc.async_writes_flushed();
    cb_h.pop_front(out_tiles);
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
