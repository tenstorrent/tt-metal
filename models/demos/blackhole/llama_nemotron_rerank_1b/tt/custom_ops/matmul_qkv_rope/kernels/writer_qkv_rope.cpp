// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QKV projection + RoPE: in1 (weight) reader / multicaster and output writer, from minimal_matmul's
// dm_in1_sender_out.cpp (transposed grid: M split over x, N over y). Two changes:
//   * at the start of each M block it reads the cos, sin, -sin tiles of the block's tile-rows
//     into CB 4 (row m's sequence tile is m % seq_tiles);
//   * output tiles go straight to the head layouts instead of a [M, N] tensor: QKV column tile n
//     of token tile m (batch b = m / seq_tiles, sequence tile s) lands in Q (n < q_tiles), K or V
//     as page ((b * heads + n / head_tiles) * seq_tiles + s) * head_tiles + n % head_tiles.
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
constexpr uint32_t cb_out_id = tt::CBIndex::c_2;
constexpr uint32_t cb_cs_id = tt::CBIndex::c_4;
constexpr uint32_t FWD_TRID = 1;
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
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(10);
    constexpr uint32_t in1_tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(12);
    Semaphore<> in1_sender_semaphore(get_compile_time_arg_val(13));
    Semaphore<> in1_receiver_semaphore(get_compile_time_arg_val(14));
    Semaphore<> in1_valid_semaphore(get_compile_time_arg_val(15));
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(16);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(17);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t q_heads = get_compile_time_arg_val(19);
    constexpr uint32_t kv_heads = get_compile_time_arg_val(20);
    constexpr uint32_t q_tiles = q_heads * head_tiles;
    constexpr uint32_t kv_tiles = kv_heads * head_tiles;

    uint32_t argidx = 0;
    const uint32_t in1_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t cs_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
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

    constexpr auto in1_args = TensorAccessorArgs<21>();
    constexpr auto cs_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    constexpr auto q_args = TensorAccessorArgs<cs_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    const auto in1_reader = TensorAccessor(in1_args, in1_addr);
    const auto cs_reader = TensorAccessor(cs_args, cs_addr);
    const auto q_writer = TensorAccessor(q_args, q_addr);
    const auto k_writer = TensorAccessor(k_args, k_addr);
    const auto v_writer = TensorAccessor(v_args, v_addr);

    const TensorShape2D in1_shape(K_tiles, N_tiles, padded_K_tiles, padded_N_tiles);
    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;
    constexpr uint32_t full_N_tiles_bytes = N_block_tiles * in1_tile_size;
    constexpr uint32_t cs_tile_size = get_tile_size(cb_cs_id);

    CircularBuffer cb_in1(cb_in1_id);
    CircularBuffer cb_out(cb_out_id);
    CircularBuffer cb_cs(cb_cs_id);

    // Output tiles of one block are queued and issued a tile at a time whenever the NOC write command
    // buffer is free; every wait in the weight relay polls the queue (pump), so writing outputs never
    // holds up the relay (the next cores in the chain wait on it). The relay's own writes carry
    // FWD_TRID so their flush does not wait for queued output writes.
    struct {
        uint32_t base, m_tile, m_tile_end, n_tile, n_tile_end, cursor;
        bool active;
    } io = {};
    auto io_issue = [&](bool block) {
        while (io.active && io.cursor < out_block_num_tiles) {
            const uint32_t i = io.cursor / N_block_tiles;
            const uint32_t j = io.cursor - i * N_block_tiles;
            const uint32_t m = io.m_tile + i;
            const uint32_t n = io.n_tile + j;
            if (m >= io.m_tile_end || m >= M_tiles || n >= io.n_tile_end || n >= N_tiles) {
                ++io.cursor;
                continue;
            }
            if (!block && !noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
                return;
            }
            const uint32_t b = m / seq_tiles;
            const uint32_t s = m - b * seq_tiles;
            const uint32_t src = io.base + io.cursor * out_tile_size;
            if (n < q_tiles) {
                const uint32_t page = ((b * q_heads + n / head_tiles) * seq_tiles + s) * head_tiles + n % head_tiles;
                noc.async_write(CoreLocalMem<uint32_t>(src), q_writer, out_tile_size, {}, {.page_id = page});
            } else if (n < q_tiles + kv_tiles) {
                const uint32_t c = n - q_tiles;
                const uint32_t page = ((b * kv_heads + c / head_tiles) * seq_tiles + s) * head_tiles + c % head_tiles;
                noc.async_write(CoreLocalMem<uint32_t>(src), k_writer, out_tile_size, {}, {.page_id = page});
            } else {
                const uint32_t c = n - q_tiles - kv_tiles;
                const uint32_t page = ((b * kv_heads + c / head_tiles) * seq_tiles + s) * head_tiles + c % head_tiles;
                noc.async_write(CoreLocalMem<uint32_t>(src), v_writer, out_tile_size, {}, {.page_id = page});
            }
            ++io.cursor;
            if (!block) {
                return;
            }
        }
    };
    auto pump = [&]() { io_issue(false); };
    // Queue a finished output block (waits for the compute to push it).
    auto start_block = [&](uint32_t m_tile, uint32_t m_tile_end, uint32_t n_tile, uint32_t n_tile_end) {
        cb_out.wait_front(out_block_num_tiles);
        io = {cb_out.get_read_ptr(), m_tile, m_tile_end, n_tile, n_tile_end, 0, true};
    };
    // Issue what is left of the queued block, wait for it to leave L1 and free it.
    auto finish_block = [&]() {
        if (io.active) {
            io_issue(true);
            noc.async_writes_flushed();
            cb_out.pop_front(out_block_num_tiles);
            io.active = false;
        }
    };
    volatile tt_l1_ptr uint32_t* in1_sender_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_compile_time_arg_val(13)));
    volatile tt_l1_ptr uint32_t* in1_receiver_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_compile_time_arg_val(14)));

    in1_valid_semaphore.set(VALID);

    bool k_forward = true;
    uint32_t defer_write_m_tile = 0;
    uint32_t defer_write_m_tile_end = 0;
    uint32_t defer_write_n_tile = 0;
    uint32_t defer_write_n_tile_end = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        const uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        const uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);

        // cos, sin, -sin for this block's tile-rows (CB 4 holds two blocks, so this rarely waits).
        cb_cs.reserve_back(3 * M_block_tiles);
        {
            uint32_t ptr = cb_cs.get_write_ptr();
            for (uint32_t i = 0; i < M_block_tiles; ++i) {
                const uint32_t m = m_tile + i;
                if (m < m_tile_end && m < M_tiles) {
                    const uint32_t s = m % seq_tiles;
                    for (uint32_t t = 0; t < 3; ++t) {
                        noc.async_read(
                            cs_reader,
                            CoreLocalMem<uint32_t>(ptr + t * cs_tile_size),
                            cs_tile_size,
                            {.page_id = 3 * s + t},
                            {});
                    }
                }
                ptr += 3 * cs_tile_size;
            }
            noc.async_read_barrier();
        }
        cb_cs.push_back(3 * M_block_tiles);

        k_forward = true;
        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            const uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            const uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            const uint32_t current_N_tiles_bytes = (n_tile_end - n_tile) * in1_tile_size;
            const bool is_last_block =
                (m_block_iter == M_blocks_per_core - 1) && (n_block_iter == N_blocks_per_core - 1);
            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (defer_write && k_block_iter == defer_write_k_block) {
                    start_block(defer_write_m_tile, defer_write_m_tile_end, defer_write_n_tile, defer_write_n_tile_end);
                }
                const uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
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
                    in1_sender_semaphore.up(noc, in1_sender_noc_x, in1_sender_noc_y, 1);
                    while (*in1_receiver_sem != VALID) {
                        pump();
                    }
                }
                cb_in1.push_back(in1_block_num_tiles);
                if (!is_sink_core) {
                    while (*in1_sender_sem != 1) {
                        pump();
                    }
                    in1_sender_semaphore.set(0);
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
                pump();
            }
            // The previous block's outputs must be out before the compute needs its CB space again.
            finish_block();
            k_forward = !k_forward;
            defer_write_m_tile = m_tile;
            defer_write_m_tile_end = m_tile_end;
            defer_write_n_tile = n_tile;
            defer_write_n_tile_end = n_tile_end;
            // As in minimal_matmul: write this block during the next block's K loop, except the last.
            defer_write = !is_last_block && !is_injector_core;
            if (!defer_write) {
                start_block(m_tile, m_tile_end, n_tile, n_tile_end);
                finish_block();
            }
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
