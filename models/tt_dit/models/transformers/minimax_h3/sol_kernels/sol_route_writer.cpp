// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn routing pre-pass, writer (BRISC). Unit = (head h, Q-chunk tile row it); outputs arrive per column shard.
//   chunk-form rows (MODE 0 mask, MODE 2 union row): the 0/1 chunk tiles of every shard are gathered into an L1 row
//     buffer [32 rows x NKCP words] (word s*NKC + c, padding words zero) and the 32 complete rows are written as whole
//     pages (NoC-aligned) to [1, H, NQCP, NKCP] at pages h*NQCP + 32*it + r.
//   gather-form rows (MODE 1/2 mask): per shard, the NBP words [sp*NBP, (sp+1)*NBP) of each row are written directly
//     (NBP*4 is a multiple of 128 bytes, so these partial-row writes are aligned).
//   bias: the expanded bias tiles, page (h, Q chunk i, global block tile sp*NBPT + j) of [1, H, NQCP*32, NBG];
//     batches always carry BIAS_BATCH pages of which the first nr are real.
//
// Compile-time args: 0..15 as the reader, then TensorAccessorArgs of mask, bias and union (dummy unless MODE 2).
// Common runtime args: 0 mask address, 1 bias address, 2 union address. Per-core: none.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_mask_out = 16, cb_bias_out = 18, cb_rows = 19, cb_union_out = 21, cb_ring = 22;
constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t RING_PAGES = 8;
constexpr uint32_t BIAS_BATCH = 8;
inline uint32_t min_u(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Element (r, c) of a 32x32 bf16 tile (face layout) is nonzero.
inline uint32_t tile_elem_nz(uint32_t tile_addr, uint32_t r, uint32_t c) {
    CoreLocalMem<volatile uint16_t> e(tile_addr);
    const uint32_t face = ((r >> 4) << 1) | (c >> 4);
    return e[face * 256 + (r & 15) * 16 + (c & 15)] != 0 ? 1u : 0u;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t NBPT = get_compile_time_arg_val(2);
    constexpr uint32_t NQCT = get_compile_time_arg_val(4);
    constexpr uint32_t NKCLT = get_compile_time_arg_val(5);
    constexpr uint32_t NQC = get_compile_time_arg_val(6);
    constexpr uint32_t NKC = get_compile_time_arg_val(7);
    constexpr uint32_t NKCP = get_compile_time_arg_val(8);
    constexpr uint32_t MODE = get_compile_time_arg_val(9);
    constexpr uint32_t EMIT_BIAS = get_compile_time_arg_val(10);
    constexpr uint32_t DEBUG = get_compile_time_arg_val(16);
    constexpr uint32_t NBGT = S * NBPT;
    constexpr uint32_t NBP = NBPT * 32;
    constexpr uint32_t NQCP = NQCT * 32;
    constexpr uint32_t MASK_T = MODE == 0 ? NKCLT : NBPT;  // per shard
    constexpr uint32_t ROW_BYTES = NKCP * 4;
    constexpr auto mk_args = TensorAccessorArgs<20>();
    constexpr auto bs_args = TensorAccessorArgs<mk_args.next_compile_time_args_offset()>();
    constexpr auto un_args = TensorAccessorArgs<bs_args.next_compile_time_args_offset()>();

    constexpr uint32_t NX = get_compile_time_arg_val(17);
    constexpr uint32_t NY = get_compile_time_arg_val(18);
    constexpr uint32_t UNITS = get_compile_time_arg_val(19);
    // Unit range of this core from its logical coordinates (no per-core runtime args): units [ci*U/N, (ci+1)*U/N).
    constexpr uint32_t NCORES = NX * NY;
    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    const uint32_t unit_start = ci * UNITS / NCORES;
    const uint32_t unit_count = (ci + 1) * UNITS / NCORES - unit_start;
    if (unit_count == 0) {
        return;
    }
    const auto mk = TensorAccessor(mk_args, get_common_arg_val<uint32_t>(0));
    const auto bs = TensorAccessor(bs_args, get_common_arg_val<uint32_t>(1));
    [[maybe_unused]] const auto un = TensorAccessor(un_args, get_common_arg_val<uint32_t>(2));

    Noc noc;
    CircularBuffer cbm(cb_mask_out), cbb(cb_bias_out), cbrows(cb_rows), cbu(cb_union_out), cbring(cb_ring);
    cbrows.reserve_back(32);
    const uint32_t rows_base = cbrows.get_write_ptr();  // 32 pages of ROW_BYTES
    cbring.reserve_back(RING_PAGES);
    const uint32_t ring_base = cbring.get_write_ptr();  // RING_PAGES pages of NBP*4 bytes
    uint32_t ring_used = 0;

    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t h = u / NQCT;
        const uint32_t it = u % NQCT;

        // Zero the row buffer (chunk-form outputs fill only the real chunk words).
        if constexpr (MODE != 1) {
            CoreLocalMem<volatile uint32_t> w(rows_base);
            for (uint32_t i = 0; i < 32 * NKCP; ++i) {
                w[i] = 0;
            }
        }

        if constexpr (DEBUG == 1) {  // 16 moment tiles of this unit -> bias pages [u*16, u*16+16)
            for (uint32_t b = 0; b < 2; ++b) {
                cbb.wait_front(BIAS_BATCH);
                for (uint32_t i = 0; i < BIAS_BATCH; ++i) {
                    noc.async_write(
                        cbb,
                        bs,
                        TILE_BYTES,
                        {.offset_bytes = i * TILE_BYTES},
                        {.page_id = u * 16 + b * BIAS_BATCH + i});
                }
                noc.async_write_barrier();
                cbb.pop_front(BIAS_BATCH);
            }
        }

        for (uint32_t sp = 0; sp < S; ++sp) {
            // ---- mask tiles of this shard
            cbm.wait_front(MASK_T);
            const uint32_t mbase = cbm.get_read_ptr();
            if constexpr (MODE == 0) {
                for (uint32_t r = 0; r < 32; ++r) {
                    CoreLocalMem<volatile uint32_t> w(rows_base + r * ROW_BYTES);
                    for (uint32_t c = 0; c < NKC; ++c) {
                        w[sp * NKC + c] = tile_elem_nz(mbase + (c >> 5) * TILE_BYTES, r, c & 31);
                    }
                }
            } else {
                for (uint32_t r = 0; r < 32; ++r) {
                    if (ring_used != 0 && (ring_used % RING_PAGES) == 0) {
                        noc.async_write_barrier();  // the ring wraps
                    }
                    const uint32_t saddr = ring_base + (ring_used % RING_PAGES) * (NBP * 4);
                    CoreLocalMem<volatile uint32_t> w(saddr);
                    for (uint32_t b = 0; b < NBP; ++b) {
                        w[b] = tile_elem_nz(mbase + (b >> 5) * TILE_BYTES, r, b & 31);
                    }
                    noc.async_write(
                        CoreLocalMem<uint32_t>(saddr),
                        mk,
                        NBP * 4,
                        {},
                        {.page_id = h * NQCP + it * 32 + r, .offset_bytes = sp * NBP * 4});
                    ++ring_used;
                }
            }
            cbm.pop_front(MASK_T);

            // ---- union-row tiles of this shard (MODE 2) into the row buffer
            if constexpr (MODE == 2) {
                cbu.wait_front(NKCLT);
                const uint32_t ubase = cbu.get_read_ptr();
                for (uint32_t r = 0; r < 32; ++r) {
                    CoreLocalMem<volatile uint32_t> w(rows_base + r * ROW_BYTES);
                    for (uint32_t c = 0; c < NKC; ++c) {
                        w[sp * NKC + c] = tile_elem_nz(ubase + (c >> 5) * TILE_BYTES, r, c & 31);
                    }
                }
                cbu.pop_front(NKCLT);
            }

            // ---- bias tiles of this shard
            if constexpr (EMIT_BIAS == 1 && DEBUG == 0) {
                for (uint32_t j = 0; j < NBPT; ++j) {
                    for (uint32_t r0 = 0; r0 < 32 && it * 32 + r0 < NQC; r0 += BIAS_BATCH) {
                        const uint32_t nr = min_u(BIAS_BATCH, min_u(32 - r0, NQC - (it * 32 + r0)));
                        cbb.wait_front(BIAS_BATCH);
                        for (uint32_t rr = 0; rr < nr; ++rr) {
                            const uint32_t i = it * 32 + r0 + rr;
                            noc.async_write(
                                cbb,
                                bs,
                                TILE_BYTES,
                                {.offset_bytes = rr * TILE_BYTES},
                                {.page_id = (h * NQCP + i) * NBGT + sp * NBPT + j});
                        }
                        noc.async_write_barrier();
                        cbb.pop_front(BIAS_BATCH);
                    }
                }
            }
        }

        // ---- complete chunk-form rows (whole, aligned pages)
        if constexpr (MODE != 1) {
            for (uint32_t r = 0; r < 32; ++r) {
                if constexpr (MODE == 0) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(rows_base + r * ROW_BYTES),
                        mk,
                        ROW_BYTES,
                        {},
                        {.page_id = h * NQCP + it * 32 + r});
                } else {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(rows_base + r * ROW_BYTES),
                        un,
                        ROW_BYTES,
                        {},
                        {.page_id = h * NQCP + it * 32 + r});
                }
            }
        }
        noc.async_write_barrier();
    }
}
