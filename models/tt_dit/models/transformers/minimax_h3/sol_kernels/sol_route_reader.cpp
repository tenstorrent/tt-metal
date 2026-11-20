// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn pre-pass, route step, reader kernel.
//
// Feeds the compute kernel. Per unit: the head's pooled K tiles (this shard's only with (head, shard) units), the
// head's pooled Q tiles, then per (Q-block tile, shard) the always-keep tiles and the valid-block tiles. Once per
// core, after the first unit's data: the ones tile and (with (head, shard) units) the shard selector tiles.
//
// (head, shard) units: a core only has its shard's K, so the head's cores exchange partial moments. Compute hands
// this kernel 8 reduced rows (cb_local_moments); they are written into the gather tiles (cb_gathered_moments) at this
// shard's slot on every core of the head, then semaphore 0 is bumped on each peer. When the semaphore reaches
// NUM_SHARDS - 1 every shard's rows have arrived. With Q_SHARED the shard-0 core reads the pooled Q and sends it to
// the other cores (semaphore 1).
//
// Compile-time args: 0..10 as sol_route_compute.cpp (names explained there), then TensorAccessorArgs of q_pooled,
// k_pooled, const_or, valid_blocks, ones, shard_select (host name shsel), then Q_SHARED.
// Common runtime args: 0..5 the buffer addresses in that order, 6.. the NoC coordinates of every logical core
// (x | y << 8). Semaphores: 0 moment exchange, 1 Q hand-off.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_k_pooled = 0, cb_q_pooled = 1, cb_ones = 2, cb_const_or = 6, cb_valid_blocks = 7,
                   cb_gathered_moments = 23, cb_local_moments = 24, cb_shard_select = 29;
constexpr uint32_t PARTIAL_MOMENT_TILES = 8;  // HEAD_DIM_TILES sum tiles + HEAD_DIM_TILES sum-of-squares tiles
constexpr uint32_t SHARD_SELECT_TILES = 8;

constexpr uint32_t TILE_BYTES = 2048;
}  // namespace

void kernel_main() {
    constexpr uint32_t NUM_SHARDS = get_compile_time_arg_val(1);
    constexpr uint32_t SHARD_BLOCK_TILES = get_compile_time_arg_val(2);
    constexpr uint32_t HEAD_DIM_TILES = get_compile_time_arg_val(3);
    constexpr uint32_t GRID_X = get_compile_time_arg_val(7);
    constexpr uint32_t GRID_Y = get_compile_time_arg_val(8);
    constexpr uint32_t NUM_UNITS = get_compile_time_arg_val(9);
    constexpr uint32_t UNITS_PER_SHARD = get_compile_time_arg_val(10);  // 1: (head, shard); 0: (head, Q-block tile)

    constexpr uint32_t ALL_BLOCK_TILES = NUM_SHARDS * SHARD_BLOCK_TILES;
    // Pooled-K block tiles a unit reads: its own shard's, or every shard's.
    constexpr uint32_t K_BLOCK_TILES_PER_UNIT = UNITS_PER_SHARD ? SHARD_BLOCK_TILES : ALL_BLOCK_TILES;
    constexpr uint32_t K_TILES_PER_UNIT = K_BLOCK_TILES_PER_UNIT * HEAD_DIM_TILES;
    constexpr uint32_t Q_TILES_PER_HEAD = SHARD_BLOCK_TILES * HEAD_DIM_TILES;
    constexpr auto q_pooled_args = TensorAccessorArgs<11>();
    constexpr auto k_pooled_args = TensorAccessorArgs<q_pooled_args.next_compile_time_args_offset()>();
    constexpr auto const_or_args = TensorAccessorArgs<k_pooled_args.next_compile_time_args_offset()>();
    constexpr auto valid_blocks_args = TensorAccessorArgs<const_or_args.next_compile_time_args_offset()>();
    constexpr auto ones_args = TensorAccessorArgs<valid_blocks_args.next_compile_time_args_offset()>();
    constexpr auto shard_select_args = TensorAccessorArgs<ones_args.next_compile_time_args_offset()>();
    // Q_SHARED: the shard-0 core reads the head's Q and sends it to the other cores (semaphore 1).
    constexpr uint32_t Q_SHARED = get_compile_time_arg_val(shard_select_args.next_compile_time_args_offset());

    // This core's units, [core_index * NUM_UNITS / NUM_CORES, (core_index + 1) * NUM_UNITS / NUM_CORES).
    constexpr uint32_t NUM_CORES = GRID_X * GRID_Y;
    static_assert(
        UNITS_PER_SHARD == 0 || NUM_UNITS <= NUM_CORES, "the partial-moment exchange needs one unit per core");
    static_assert(UNITS_PER_SHARD == 0 || NUM_SHARDS <= 8, "gather tiles hold 4 shards x 8 moment rows each");
    constexpr uint32_t GATHER_TILES = (NUM_SHARDS + 3) / 4;
    const uint32_t core_index = static_cast<uint32_t>(get_absolute_logical_y()) * GRID_X + get_absolute_logical_x();
    const uint32_t unit_start = core_index * NUM_UNITS / NUM_CORES;
    const uint32_t unit_count = (core_index + 1) * NUM_UNITS / NUM_CORES - unit_start;
    if (unit_count == 0) {
        return;
    }
    // Gather tiles for the exchanged moments, reserved once; the same address on every core.
    CircularBuffer gathered_moments_cb(cb_gathered_moments), local_moments_cb(cb_local_moments);
    uint32_t gather_base = 0;
    if constexpr (UNITS_PER_SHARD) {
        gathered_moments_cb.reserve_back(GATHER_TILES);
        gather_base = gathered_moments_cb.get_write_ptr();  // peers write their rows here
    }
    const auto q_pooled = TensorAccessor(q_pooled_args, get_common_arg_val<uint32_t>(0));
    const auto k_pooled = TensorAccessor(k_pooled_args, get_common_arg_val<uint32_t>(1));
    const auto const_or = TensorAccessor(const_or_args, get_common_arg_val<uint32_t>(2));
    const auto valid_blocks = TensorAccessor(valid_blocks_args, get_common_arg_val<uint32_t>(3));
    const auto ones = TensorAccessor(ones_args, get_common_arg_val<uint32_t>(4));
    const auto shard_select = TensorAccessor(shard_select_args, get_common_arg_val<uint32_t>(5));

    Noc noc;
    auto read_pages = [&](auto& accessor, uint32_t cb_id, uint32_t first_page, uint32_t count) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(count);
        for (uint32_t i = 0; i < count; ++i) {
            noc.async_read(accessor, cb, TILE_BYTES, {.page_id = first_page + i}, {.offset_bytes = i * TILE_BYTES});
        }
        noc.async_read_barrier();
        cb.push_back(count);
    };

    // The core that owns unit peer_unit.
    auto core_of_unit = [](uint32_t peer_unit) {
        uint32_t peer_core = 0;
        while (
            !((peer_core * NUM_UNITS / NUM_CORES <= peer_unit) &&
              (peer_unit < (peer_core + 1) * NUM_UNITS / NUM_CORES))) {
            ++peer_core;
        }
        return peer_core;
    };
    CircularBuffer k_pooled_cb(cb_k_pooled), q_pooled_cb(cb_q_pooled), const_or_cb(cb_const_or),
        valid_blocks_cb(cb_valid_blocks);
    for (uint32_t unit = unit_start; unit < unit_start + unit_count; ++unit) {
        // One of the two ranges is a single value; the loops are the same either way.
        const uint32_t head = UNITS_PER_SHARD ? unit / NUM_SHARDS : unit / SHARD_BLOCK_TILES;
        const uint32_t q_block_tile_begin = UNITS_PER_SHARD ? 0 : unit % SHARD_BLOCK_TILES;
        const uint32_t q_block_tile_end = UNITS_PER_SHARD ? SHARD_BLOCK_TILES : q_block_tile_begin + 1;
        const uint32_t shard_begin = UNITS_PER_SHARD ? unit % NUM_SHARDS : 0;
        const uint32_t shard_end = UNITS_PER_SHARD ? shard_begin + 1 : NUM_SHARDS;

        // Pooled K, column-major (tile k_block_tile * HEAD_DIM_TILES + dim_tile). Each column has its own transaction
        // id and is pushed as it lands; Q is read right behind it.
        q_pooled_cb.reserve_back(Q_TILES_PER_HEAD);
        const uint32_t q_addr = q_pooled_cb.get_write_ptr();  // the same address on every core
        const bool q_from_dram = !(Q_SHARED && UNITS_PER_SHARD) || shard_begin == 0;
        k_pooled_cb.reserve_back(K_TILES_PER_UNIT);
        {
            const uint32_t first_k_block_tile = UNITS_PER_SHARD ? shard_begin * SHARD_BLOCK_TILES : 0;
            for (uint32_t k_block_tile = 0; k_block_tile < K_BLOCK_TILES_PER_UNIT; ++k_block_tile) {
                for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
                    noc.async_read<NocOptions::TXN_ID>(
                        k_pooled,
                        k_pooled_cb,
                        TILE_BYTES,
                        {.page_id =
                             (head * ALL_BLOCK_TILES + first_k_block_tile + k_block_tile) * HEAD_DIM_TILES + dim_tile},
                        {.offset_bytes = (k_block_tile * HEAD_DIM_TILES + dim_tile) * TILE_BYTES},
                        {.trid = 1 + (k_block_tile % 7)});
                }
            }
        }
        if (q_from_dram) {
            for (uint32_t i = 0; i < Q_TILES_PER_HEAD; ++i) {
                noc.async_read<NocOptions::TXN_ID>(
                    q_pooled,
                    q_pooled_cb,
                    TILE_BYTES,
                    {.page_id = head * SHARD_BLOCK_TILES * HEAD_DIM_TILES + i},
                    {.offset_bytes = i * TILE_BYTES},
                    {.trid = 8});
            }
        }
        for (uint32_t k_block_tile = 0; k_block_tile < K_BLOCK_TILES_PER_UNIT; ++k_block_tile) {
            if (k_block_tile >= 7) {  // ids wrap every 7 columns
                noc.async_read_barrier();
            } else {
                noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1 + k_block_tile});
            }
            k_pooled_cb.push_back(HEAD_DIM_TILES);
        }
        if (q_from_dram) {
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 8});
            q_pooled_cb.push_back(Q_TILES_PER_HEAD);
            if constexpr (Q_SHARED && UNITS_PER_SHARD) {  // send the head's Q to the other cores; the exchange
                const UnicastEndpoint remote;             // below waits for the writes and bumps semaphore 1
                for (uint32_t peer_shard = 1; peer_shard < NUM_SHARDS; ++peer_shard) {
                    const uint32_t peer_noc_xy =
                        get_common_arg_val<uint32_t>(6 + core_of_unit(head * NUM_SHARDS + peer_shard));
                    noc.async_write(
                        CoreLocalMem<uint32_t>(q_addr),
                        remote,
                        Q_TILES_PER_HEAD * TILE_BYTES,
                        {},
                        {.noc_x = peer_noc_xy & 0xFF, .noc_y = (peer_noc_xy >> 8) & 0xFF, .addr = q_addr});
                }
            }
        }
        // Receivers push Q after the exchange.

        // Per-core constants, read once.
        if (unit == unit_start) {
            read_pages(ones, cb_ones, 0, 1);
            if constexpr (UNITS_PER_SHARD) {
                read_pages(shard_select, cb_shard_select, 0, SHARD_SELECT_TILES);
            }
        }

        // Partial-moment exchange with the head's other cores.
        if constexpr (UNITS_PER_SHARD) {
            const uint32_t shard = shard_begin;
            const uint32_t sem_addr = get_semaphore(0);
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
            const UnicastEndpoint remote;
            // This shard's 8 rows -> its slot in the gather tiles: two 256 B runs.
            const uint32_t gather_tile_addr = gather_base + (shard >> 2) * TILE_BYTES;  // gather tile of this shard
            const uint32_t left_run = gather_tile_addr + ((shard & 3) >> 1) * 1024 + (shard & 1) * 256;  // left halves
            const uint32_t right_run = left_run + 512;                                                   // right halves
            local_moments_cb.wait_front(PARTIAL_MOMENT_TILES);
            {
                const uint32_t local_moments_base = local_moments_cb.get_read_ptr();
                CoreLocalMem<volatile uint32_t> left(left_run), right(right_run);
                for (uint32_t moment = 0; moment < PARTIAL_MOMENT_TILES; ++moment) {
                    CoreLocalMem<volatile uint32_t> src(local_moments_base + moment * TILE_BYTES);
                    for (uint32_t i = 0; i < 8; ++i) {
                        left[moment * 8 + i] = src[i];
                        right[moment * 8 + i] = src[128 + i];
                    }
                }
            }
            local_moments_cb.pop_front(PARTIAL_MOMENT_TILES);
            uint32_t peer_noc_xy[NUM_SHARDS];
            for (uint32_t peer_shard = 0; peer_shard < NUM_SHARDS; ++peer_shard) {
                if (peer_shard == shard) {
                    peer_noc_xy[peer_shard] = 0;
                    continue;
                }
                peer_noc_xy[peer_shard] =
                    get_common_arg_val<uint32_t>(6 + core_of_unit(head * NUM_SHARDS + peer_shard));  // the peer's core
                const uint32_t peer_x = peer_noc_xy[peer_shard] & 0xFF, peer_y = (peer_noc_xy[peer_shard] >> 8) & 0xFF;
                for (uint32_t run : {left_run, right_run}) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(run), remote, 256, {}, {.noc_x = peer_x, .noc_y = peer_y, .addr = run});
                }
            }
            noc.async_write_barrier();  // all rows have landed before any semaphore moves
            for (uint32_t peer_shard = 0; peer_shard < NUM_SHARDS; ++peer_shard) {
                if (peer_shard != shard) {
                    const uint32_t peer_x = peer_noc_xy[peer_shard] & 0xFF;
                    const uint32_t peer_y = (peer_noc_xy[peer_shard] >> 8) & 0xFF;
                    noc_semaphore_inc(get_noc_addr(peer_x, peer_y, sem_addr), 1);
                    if constexpr (Q_SHARED) {
                        if (shard == 0) {
                            noc_semaphore_inc(get_noc_addr(peer_x, peer_y, get_semaphore(1)), 1);
                        }
                    }
                }
            }
            noc_async_atomic_barrier();
            noc_semaphore_wait(sem, NUM_SHARDS - 1);
            noc_semaphore_set(sem, 0);  // reset for the next launch
            gathered_moments_cb.push_back(GATHER_TILES);
            if constexpr (Q_SHARED) {
                if (!q_from_dram) {  // Q sent by the shard-0 core
                    volatile tt_l1_ptr uint32_t* q_sem =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
                    noc_semaphore_wait(q_sem, 1);
                    noc_semaphore_set(q_sem, 0);
                    q_pooled_cb.push_back(Q_TILES_PER_HEAD);
                }
            }
        }

        // The always-keep and valid-block tiles of (Q-block tile, shard), the same page in both tensors.
        for (uint32_t q_block_tile = q_block_tile_begin; q_block_tile < q_block_tile_end; ++q_block_tile) {
            for (uint32_t shard = shard_begin; shard < shard_end; ++shard) {
                const uint32_t first_page = q_block_tile * ALL_BLOCK_TILES + shard * SHARD_BLOCK_TILES;
                const_or_cb.reserve_back(SHARD_BLOCK_TILES);
                for (uint32_t k_block_tile = 0; k_block_tile < SHARD_BLOCK_TILES; ++k_block_tile) {
                    noc.async_read(
                        const_or,
                        const_or_cb,
                        TILE_BYTES,
                        {.page_id = first_page + k_block_tile},
                        {.offset_bytes = k_block_tile * TILE_BYTES});
                }
                valid_blocks_cb.reserve_back(SHARD_BLOCK_TILES);
                for (uint32_t k_block_tile = 0; k_block_tile < SHARD_BLOCK_TILES; ++k_block_tile) {
                    noc.async_read(
                        valid_blocks,
                        valid_blocks_cb,
                        TILE_BYTES,
                        {.page_id = first_page + k_block_tile},
                        {.offset_bytes = k_block_tile * TILE_BYTES});
                }
                noc.async_read_barrier();
                const_or_cb.push_back(SHARD_BLOCK_TILES);
                valid_blocks_cb.push_back(SHARD_BLOCK_TILES);
            }
        }
    }
    // Leave the NoC transaction id at 0 for the next kernel on this core.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
