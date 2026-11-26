// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// block_topk_route reader. First the pool phase (block_pool_reader.cpp), then the route. Once per core: the constant
// tiles (eye, SENTINEL, ones, the 32 row selectors). Per unit (head h, Q-block tile row it), in the order compute
// consumes them: the DT q_bar tiles of (h, it); the gathered pooled K of head h column by column (DT tiles per column
// tile); the transposed candidate bias per column tile; the row bias of it; the column bias per column tile; with
// COARSE, the pad row per column tile (row max), then per chunk of PC column tiles the pad row per column tile
// (exponent) and the gathered pooled V of head h column by column.
//
// Columns go shard by shard in hop order (this device's shard first, then by hop distance, so the local and near
// shards are scored while the far ones cross the fabric): column c is global column tile hop[c / NBPT] * NBPT +
// c % NBPT, for every column-indexed input alike.
//
// The sender cores of this program (block_pool_send.cpp) push this device's shard of kv_g to the peers while the peers'
// senders push theirs here. The reader on core (0, 0) is the gate. It waits until every pool writer has incremented its
// barrier semaphore and multicasts the count to every core: a reader starts on q_bar and this device's shard once its
// copy holds it. Then, for t = 1..S-1, it waits until the arrival slot of shard hop[t] reaches its expected count,
// takes that count off (an early increment for the next call stays), and multicasts t to program semaphore 0 of every
// core. A reader waits for the value t before its first column of shard hop[t].
//
// Compile-time args: 0 H, 1 NBPT, 2 NBGT, 3 DT, 4 KT, 5 MC, 6 NX, 7 NY, 8 UNITS, 9 COARSE, 10 SCALE, 11 RT, 12 PC,
// 13 NCORES (the cores that take units, the first NCORES in row-major order), then TensorAccessorArgs of q_bar, kv_g,
// consts, cand_t, col_bias, row_bias, pad_row, then the pool reader's.
// Common runtime args: 0..6 the seven buffer addresses in the same order, 7..10 noc x0, y0, x1, y1 of the grid, then
// S words each: hop (shards in release and column order, this device's first), the arrival slot address of each shard,
// its expected count; then the pool reader's. The barrier and release multicasts cover the full NX x NY grid.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "block_pool_reader.cpp"

namespace {
constexpr uint32_t cb_q = 0, cb_k = 1, cb_const = 2, cb_cand_t = 7, cb_colbias = 8, cb_rowbias = 9, cb_pad = 10,
                   cb_v = 11;
constexpr uint32_t BF16_TILE = 2048, FP32_TILE = 4096;
constexpr uint32_t N_CONST = 35;
}  // namespace

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t NBPT = get_compile_time_arg_val(1);
    constexpr uint32_t NBGT = get_compile_time_arg_val(2);
    constexpr uint32_t DT = get_compile_time_arg_val(3);
    constexpr uint32_t NX = get_compile_time_arg_val(6);
    constexpr uint32_t NY = get_compile_time_arg_val(7);
    constexpr uint32_t UNITS = get_compile_time_arg_val(8);
    constexpr bool COARSE = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t PC = get_compile_time_arg_val(12);
    constexpr uint32_t NCORES = get_compile_time_arg_val(13);
    constexpr auto q_args = TensorAccessorArgs<14>();
    constexpr auto kv_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto const_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    constexpr auto cand_args = TensorAccessorArgs<const_args.next_compile_time_args_offset()>();
    constexpr auto col_args = TensorAccessorArgs<cand_args.next_compile_time_args_offset()>();
    constexpr auto row_args = TensorAccessorArgs<col_args.next_compile_time_args_offset()>();
    constexpr auto pad_args = TensorAccessorArgs<row_args.next_compile_time_args_offset()>();

    constexpr uint32_t S = NBGT / NBPT;
    constexpr uint32_t HOP = 11, SLOT = HOP + S, EXPECTED = SLOT + S, POOL_ARGS = EXPECTED + S;
    block_pool::read<pad_args.next_compile_time_args_offset()>(POOL_ARGS);

    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    const uint32_t release_addr = get_semaphore(0);
    volatile tt_l1_ptr uint32_t* release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(release_addr);
    const uint32_t barrier_addr = get_semaphore(block_pool::BARRIER_SEM);
    volatile tt_l1_ptr uint32_t* barrier = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_addr);
    if (ci == 0) {
        const uint32_t x0 = get_common_arg_val<uint32_t>(7), y0 = get_common_arg_val<uint32_t>(8);
        const uint32_t x1 = get_common_arg_val<uint32_t>(9), y1 = get_common_arg_val<uint32_t>(10);
        // A NOC_1 multicast rectangle runs from the high corner to the low one.
        auto grid = [&](uint32_t addr) {
            return noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, addr)
                                  : get_noc_multicast_addr(x1, y1, x0, y0, addr);
        };
        noc_semaphore_wait_min(barrier, NCORES);
        noc_semaphore_set_multicast(barrier_addr, grid(barrier_addr), NX * NY - 1);
        noc_async_write_barrier();
        for (uint32_t t = 1; t < S; ++t) {
            const uint32_t s = get_common_arg_val<uint32_t>(HOP + t);
            const uint32_t slot_addr = get_common_arg_val<uint32_t>(SLOT + s);
            const uint32_t expected = get_common_arg_val<uint32_t>(EXPECTED + s);
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr), expected);
            noc_semaphore_inc(get_noc_addr(slot_addr), uint32_t{0} - expected);
            *release = t;
            noc_semaphore_set_multicast(release_addr, grid(release_addr), NX * NY - 1);
            // Each value lands before the next one is written, so no core sees the count go back.
            noc_async_write_barrier();
        }
        noc_async_atomic_barrier();
    }
    const uint32_t unit_start = ci * UNITS / NCORES;
    const uint32_t unit_count = (ci + 1) * UNITS / NCORES - unit_start;
    if (unit_count == 0) {
        return;
    }
    const auto q_bar = TensorAccessor(q_args, get_common_arg_val<uint32_t>(0));
    const auto kv_g = TensorAccessor(kv_args, get_common_arg_val<uint32_t>(1));
    const auto consts = TensorAccessor(const_args, get_common_arg_val<uint32_t>(2));
    const auto cand_t = TensorAccessor(cand_args, get_common_arg_val<uint32_t>(3));
    const auto col_bias = TensorAccessor(col_args, get_common_arg_val<uint32_t>(4));
    const auto row_bias = TensorAccessor(row_args, get_common_arg_val<uint32_t>(5));
    const auto pad_row = TensorAccessor(pad_args, get_common_arg_val<uint32_t>(6));

    Noc noc;
    auto read_pages = [&](const auto& acc, uint32_t cb_id, uint32_t page_bytes, uint32_t first, uint32_t count) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(count);
        for (uint32_t i = 0; i < count; ++i) {
            noc.async_read(acc, cb, page_bytes, {.page_id = first + i}, {.offset_bytes = i * page_bytes});
        }
        noc.async_read_barrier();
        cb.push_back(count);
    };
    auto column = [&](uint32_t c) { return get_common_arg_val<uint32_t>(HOP + c / NBPT) * NBPT + c % NBPT; };
    auto read_column_tiles = [&](const auto& acc, uint32_t cb_id) {
        for (uint32_t c = 0; c < NBGT; ++c) {
            read_pages(acc, cb_id, FP32_TILE, column(c), 1);
        }
    };

    read_pages(consts, cb_const, BF16_TILE, 0, N_CONST);
    noc_semaphore_wait_min(barrier, NCORES);

    uint32_t released = 0;
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t h = u / NBPT, it = u % NBPT;
        read_pages(q_bar, cb_q, BF16_TILE, (h * NBPT + it) * DT, DT);
        for (uint32_t c = 0; c < NBGT; ++c) {
            const uint32_t needed = c / NBPT;
            if (needed > released) {
                noc_semaphore_wait_min(release, needed);
                released = needed;
            }
            read_pages(kv_g, cb_k, BF16_TILE, (h * NBGT + column(c)) * DT, DT);
        }
        read_column_tiles(cand_t, cb_cand_t);
        read_pages(row_bias, cb_rowbias, FP32_TILE, it, 1);
        read_column_tiles(col_bias, cb_colbias);
        if constexpr (COARSE) {
            read_column_tiles(pad_row, cb_pad);
            for (uint32_t c0 = 0; c0 < NBGT; c0 += PC) {
                for (uint32_t c = c0; c < c0 + PC; ++c) {
                    read_pages(pad_row, cb_pad, FP32_TILE, column(c), 1);
                }
                for (uint32_t c = c0; c < c0 + PC; ++c) {
                    read_pages(kv_g, cb_v, BF16_TILE, ((H + h) * NBGT + column(c)) * DT, DT);
                }
            }
        }
    }
}
