// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// block_topk_route writer. Per unit (head h, Q-block tile row it): first the threshold tile, built from the KT sorted
// tiles compute hands over (cb_fin, transposed domain: row j of tile p = position 32 p + j, column r = Q block r) as
// thr[r, :] = position k - 1 of Q block r, with k - 1 read once from kth_index; then NBGT / MC row-major uint32 pages
// from compute, each the 32 mask rows of the unit over MC column tiles, stored as 32 partial rows (MC * 32 words at
// word offset chunk * MC * 32) of mask rows h * NBP + 32 it + r of block_mask [1, H, NBP, NBG]; with COARSE, for each
// real block b = 32 it + r, its DT row-broadcast coarse tiles at tile rows 2b and 2b + 1 (when below RT), tile
// columns h * DT + n of coarse_out [1, 1, rows, H * D].
//
// Compile-time args: 0..11 as the compute kernel, then TensorAccessorArgs of block_mask, coarse_out and kth_index.
// Common runtime args: 0 block_mask address, 1 coarse_out address, 2 kth_index address.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_mask = 16, cb_co = 17, cb_fin = 19, cb_thr_raw = 20, cb_kidx = 21;
constexpr uint32_t BF16_TILE = 2048, FP32_TILE = 4096, KTH_INDEX_BYTES = 64;

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
    constexpr uint32_t NY = get_compile_time_arg_val(7);
    constexpr uint32_t UNITS = get_compile_time_arg_val(8);
    constexpr bool COARSE = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t RT = get_compile_time_arg_val(11);
    constexpr uint32_t NB = (RT + 1) / 2;
    constexpr auto mask_args = TensorAccessorArgs<12>();
    constexpr auto co_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto kidx_args = TensorAccessorArgs<co_args.next_compile_time_args_offset()>();
    constexpr uint32_t ROW_BYTES = MC * 32 * 4;

    constexpr uint32_t NCORES = NX * NY;
    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    const uint32_t unit_start = ci * UNITS / NCORES;
    const uint32_t unit_count = (ci + 1) * UNITS / NCORES - unit_start;
    if (unit_count == 0) {
        return;
    }
    const auto mask = TensorAccessor(mask_args, get_common_arg_val<uint32_t>(0));
    [[maybe_unused]] const auto coarse_out = TensorAccessor(co_args, get_common_arg_val<uint32_t>(1));
    const auto kth_index = TensorAccessor(kidx_args, get_common_arg_val<uint32_t>(2));

    Noc noc;
    CircularBuffer cm(cb_mask), cco(cb_co), cfin(cb_fin), cthr(cb_thr_raw), ckidx(cb_kidx);
    ckidx.reserve_back(1);
    noc.async_read(kth_index, ckidx, KTH_INDEX_BYTES, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const uint32_t kth = CoreLocalMem<volatile uint32_t>(ckidx.get_write_ptr())[0];
    const uint32_t kth_tile = kth / 32, kth_row = kth % 32;

    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t h = u / NBPT, it = u % NBPT;
        cfin.wait_front(KT);
        cthr.reserve_back(1);
        {
            CoreLocalMem<volatile uint32_t> src(cfin.get_read_ptr() + kth_tile * FP32_TILE);
            CoreLocalMem<volatile uint32_t> dst(cthr.get_write_ptr());
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t value = src[tile_word(kth_row, r)];
                for (uint32_t c = 0; c < 32; ++c) {
                    dst[tile_word(r, c)] = value;
                }
            }
        }
        cthr.push_back(1);
        cfin.pop_front(KT);
        for (uint32_t chunk = 0; chunk < NBGT / MC; ++chunk) {
            cm.wait_front(1);
            const uint32_t base = cm.get_read_ptr();
            for (uint32_t r = 0; r < 32; ++r) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(base + r * ROW_BYTES),
                    mask,
                    ROW_BYTES,
                    {},
                    {.page_id = h * NBPT * 32 + it * 32 + r, .offset_bytes = chunk * ROW_BYTES});
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
