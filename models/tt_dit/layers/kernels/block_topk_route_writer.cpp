// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// block_topk_route writer. First the pool phase (block_pool_writer.cpp), then one increment of the barrier semaphore on
// the gate core, then the route. Per unit (head h, Q-block tile row it): first the threshold tile, built from the KT
// sorted tiles compute hands over one at a time (cb_fin, transposed domain: row j of tile p = position 32 p + j, column
// r = Q block r) as thr[r, :] = position k - 1 of Q block r, with k - 1 read once from kth_index; then NBGT / MC
// row-major uint32 pages from compute, each the 32 mask rows of the unit over MC column tiles in the reader's column
// order, stored into mask rows h * NBP + 32 it + r of block_mask [1, H, NBP, NBG]: column c (32 words) at its global
// column tile order[c / NBPT] * NBPT + c % NBPT, columns that stay adjacent there written together; with COARSE, for
// each real block b = 32 it + r, its DT row-broadcast coarse tiles at tile rows 2b and 2b + 1 (when below RT), tile
// columns h * DT + n of coarse_out [1, 1, rows, H * D].
//
// Compile-time args: 0..13 as the compute kernel, then TensorAccessorArgs of block_mask, coarse_out and kth_index, then
// the pool writer's.
// Common runtime args: 0 block_mask address, 1 coarse_out address, 2 kth_index address, then the S shards in column
// order, noc x, y of the gate core (0, 0), then the pool writer's.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "block_pool_writer.cpp"

namespace {
constexpr uint32_t cb_mask = 16, cb_co = 17, cb_fin = 19, cb_thr_raw = 20, cb_kidx = 21;
constexpr uint32_t BF16_TILE = 2048, KTH_INDEX_BYTES = 64;

// Word index of (r, c) in a 32x32 fp32 tile (four 16x16 faces in row-major face order).
inline uint32_t tile_word(uint32_t r, uint32_t c) {
    const uint32_t face = ((r >> 4) << 1) | (c >> 4);
    return face * 256 + (r & 15) * 16 + (c & 15);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t NBPT = get_compile_time_arg_val(1);
    constexpr uint32_t NBGT = get_compile_time_arg_val(2);
    constexpr uint32_t DT = get_compile_time_arg_val(3);
    constexpr uint32_t KT = get_compile_time_arg_val(4);
    constexpr uint32_t MC = get_compile_time_arg_val(5);
    constexpr uint32_t NX = get_compile_time_arg_val(6);
    constexpr uint32_t UNITS = get_compile_time_arg_val(8);
    constexpr bool COARSE = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t RT = get_compile_time_arg_val(11);
    constexpr uint32_t NB = (RT + 1) / 2;
    constexpr uint32_t NCORES = get_compile_time_arg_val(13);
    constexpr auto mask_args = TensorAccessorArgs<14>();
    constexpr auto co_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto kidx_args = TensorAccessorArgs<co_args.next_compile_time_args_offset()>();
    constexpr uint32_t ROW_BYTES = MC * 32 * 4;
    constexpr uint32_t COL_BYTES = 32 * 4;
    constexpr uint32_t S = NBGT / NBPT;
    constexpr uint32_t GATE = 3 + S;

    block_pool::write<kidx_args.next_compile_time_args_offset()>(GATE + 2);
    noc_semaphore_inc(
        get_noc_addr(
            get_common_arg_val<uint32_t>(GATE),
            get_common_arg_val<uint32_t>(GATE + 1),
            get_semaphore(block_pool::BARRIER_SEM)),
        1);
    noc_async_atomic_barrier();

    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    const uint32_t unit_start = ci * UNITS / NCORES;
    const uint32_t unit_count = (ci + 1) * UNITS / NCORES - unit_start;
    if (unit_count == 0) {
        return;
    }
    const auto mask = TensorAccessor(mask_args, get_common_arg_val<uint32_t>(0));
    [[maybe_unused]] const auto coarse_out = TensorAccessor(co_args, get_common_arg_val<uint32_t>(1));
    const auto kth_index = TensorAccessor(kidx_args, get_common_arg_val<uint32_t>(2));

    auto column = [&](uint32_t c) { return get_common_arg_val<uint32_t>(3 + c / NBPT) * NBPT + c % NBPT; };

    Noc noc;
    CircularBuffer cm(cb_mask), cco(cb_co), cfin(cb_fin), cthr(cb_thr_raw), ckidx(cb_kidx);
    ckidx.reserve_back(1);
    noc.async_read(kth_index, ckidx, KTH_INDEX_BYTES, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const uint32_t kth = CoreLocalMem<volatile uint32_t>(ckidx.get_write_ptr())[0];
    const uint32_t kth_tile = kth / 32, kth_row = kth % 32;

    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t h = u / NBPT, it = u % NBPT;
        for (uint32_t p = 0; p < KT; ++p) {
            cfin.wait_front(1);
            if (p == kth_tile) {
                cthr.reserve_back(1);
                CoreLocalMem<volatile uint32_t> src(cfin.get_read_ptr());
                CoreLocalMem<volatile uint32_t> dst(cthr.get_write_ptr());
                for (uint32_t r = 0; r < 32; ++r) {
                    const uint32_t value = src[tile_word(kth_row, r)];
                    for (uint32_t c = 0; c < 32; ++c) {
                        dst[tile_word(r, c)] = value;
                    }
                }
                cthr.push_back(1);
            }
            cfin.pop_front(1);
        }
        for (uint32_t chunk = 0; chunk < NBGT / MC; ++chunk) {
            cm.wait_front(1);
            const uint32_t base = cm.get_read_ptr();
            for (uint32_t i = 0; i < MC;) {
                const uint32_t first = column(chunk * MC + i);
                uint32_t n = 1;
                while (i + n < MC && column(chunk * MC + i + n) == first + n) {
                    ++n;
                }
                for (uint32_t r = 0; r < 32; ++r) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(base + r * ROW_BYTES + i * COL_BYTES),
                        mask,
                        n * COL_BYTES,
                        {},
                        {.page_id = h * NBPT * 32 + it * 32 + r, .offset_bytes = first * COL_BYTES});
                }
                i += n;
            }
            noc.async_write_barrier();
            cm.pop_front(1);
        }
        if constexpr (COARSE) {
            for (uint32_t b = it * 32; b < it * 32 + 32 && b < NB; ++b) {
                cco.wait_front(DT);
                for (uint32_t tr = 2 * b; tr < 2 * b + 2 && tr < RT; ++tr) {
                    for (uint32_t n = 0; n < DT; ++n) {
                        noc.async_write(
                            cco,
                            coarse_out,
                            BF16_TILE,
                            {.offset_bytes = n * BF16_TILE},
                            {.page_id = (tr * H + h) * DT + n});
                    }
                }
                noc.async_write_barrier();
                cco.pop_front(DT);
            }
        }
    }
}
