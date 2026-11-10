// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

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

inline uint32_t tile_elem_nz(uint32_t tile_addr, uint32_t r, uint32_t c) {
    CoreLocalMem<volatile uint16_t> e(tile_addr);
    const uint32_t face = ((r >> 4) << 1) | (c >> 4);
    return e[face * 256 + (r & 15) * 16 + (c & 15)] != 0 ? 1u : 0u;
}

inline void tile_row_to_words(uint32_t tiles_addr, uint32_t r, uint32_t ntiles, uint32_t out_addr) {
    CoreLocalMem<volatile uint32_t> w(out_addr);
    const uint32_t row_bytes = (((r >> 4) << 1) * 256 + (r & 15) * 16) * 2;
    uint32_t k = 0;
    for (uint32_t t = 0; t < ntiles; ++t) {
        for (uint32_t f = 0; f < 2; ++f) {
            CoreLocalMem<volatile uint32_t> e(tiles_addr + t * TILE_BYTES + row_bytes + f * 512);
            for (uint32_t i = 0; i < 8; ++i) {
                const uint32_t v = e[i];
                w[k++] = (v & 0xFFFFu) != 0 ? 1u : 0u;
                w[k++] = (v >> 16) != 0 ? 1u : 0u;
            }
        }
    }
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
    constexpr uint32_t NBGT = S * NBPT;
    constexpr uint32_t NBP = NBPT * 32;
    constexpr uint32_t NQCP = NQCT * 32;
    constexpr uint32_t MASK_T = MODE == 0 ? NKCLT : NBPT;
    constexpr uint32_t ROW_BYTES = NKCP * 4;
    constexpr auto mk_args = TensorAccessorArgs<20>();
    constexpr auto bs_args = TensorAccessorArgs<mk_args.next_compile_time_args_offset()>();
    constexpr auto un_args = TensorAccessorArgs<bs_args.next_compile_time_args_offset()>();
    constexpr uint32_t HEAD_WEIGHTS = get_compile_time_arg_val(un_args.next_compile_time_args_offset());
    constexpr uint32_t NB = get_compile_time_arg_val(un_args.next_compile_time_args_offset() + 1);
    constexpr auto hw_args = TensorAccessorArgs<un_args.next_compile_time_args_offset() + 2>();

    constexpr uint32_t NX = get_compile_time_arg_val(16);
    constexpr uint32_t NY = get_compile_time_arg_val(17);
    constexpr uint32_t UNITS = get_compile_time_arg_val(18);
    constexpr uint32_t UNITS_PER_SHARD = get_compile_time_arg_val(19);
    static_assert(UNITS_PER_SHARD == 0 || MODE == 1, "(head, shard) units emit only gather-form rows (MODE 1)");
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
    [[maybe_unused]] const auto hwt = TensorAccessor(hw_args, get_common_arg_val<uint32_t>(3));
    static_assert(
        HEAD_WEIGHTS == 0 || (UNITS_PER_SHARD == 1 && NQCT <= 8), "head weights: (head, shard) units, NQCT <= 8");

    Noc noc;
    CircularBuffer cbm(cb_mask_out), cbb(cb_bias_out), cbrows(cb_rows), cbu(cb_union_out), cbring(cb_ring);
    cbrows.reserve_back(32);
    const uint32_t rows_base = cbrows.get_write_ptr();
    cbring.reserve_back(RING_PAGES);
    const uint32_t ring_base = cbring.get_write_ptr();
    uint32_t ring_used = 0;

    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        [[maybe_unused]] uint32_t hw_count[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const uint32_t h = UNITS_PER_SHARD ? u / S : u / NQCT;
        const uint32_t it_lo = UNITS_PER_SHARD ? 0 : u % NQCT;
        const uint32_t it_hi = UNITS_PER_SHARD ? NQCT : it_lo + 1;
        const uint32_t sp_lo = UNITS_PER_SHARD ? u % S : 0;
        const uint32_t sp_hi = UNITS_PER_SHARD ? sp_lo + 1 : S;

        for (uint32_t it = it_lo; it < it_hi; ++it) {
            if constexpr (MODE != 1) {
                CoreLocalMem<volatile uint32_t> w(rows_base);
                for (uint32_t i = 0; i < 32 * NKCP; ++i) {
                    w[i] = 0;
                }
            }

            for (uint32_t sp = sp_lo; sp < sp_hi; ++sp) {
                if constexpr (UNITS_PER_SHARD) {
                    cbm.wait_front(1);
                    const uint32_t rbase = cbm.get_read_ptr();
                    for (uint32_t r = 0; r < 32; ++r) {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(rbase + r * NBP * 4),
                            mk,
                            NBP * 4,
                            {},
                            {.page_id = h * NQCP + it * 32 + r, .offset_bytes = sp * NBP * 4});
                    }
                    if constexpr (HEAD_WEIGHTS == 1) {
                        invalidate_l1_cache();
                        const uint32_t* page = reinterpret_cast<const uint32_t*>(rbase);
                        const uint32_t rows = NQC - it * 32 < 32 ? NQC - it * 32 : 32;
                        uint32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                        for (uint32_t r = 0; r < rows; ++r) {
                            const uint32_t* w = page + r * NBP;
                            uint32_t b = 0;
#pragma GCC unroll 2
                            for (; b + 4 <= NB; b += 4) {
                                c0 += w[b] != 0;
                                c1 += w[b + 1] != 0;
                                c2 += w[b + 2] != 0;
                                c3 += w[b + 3] != 0;
                            }
                            for (; b < NB; ++b) {
                                c0 += w[b] != 0;
                            }
                        }
                        hw_count[it] = c0 + c1 + c2 + c3;
                    }
                    noc.async_write_barrier();
                    cbm.pop_front(1);
                } else {
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
                                noc.async_write_barrier();
                            }
                            const uint32_t saddr = ring_base + (ring_used % RING_PAGES) * (NBP * 4);
                            tile_row_to_words(mbase, r, NBPT, saddr);
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
                }

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

                if constexpr (EMIT_BIAS == 1) {
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
        if constexpr (HEAD_WEIGHTS == 1) {
            noc.async_write_barrier();
            CoreLocalMem<volatile uint32_t> w(ring_base);
            for (uint32_t i = 0; i < 8; ++i) {
                w[i] = hw_count[i];
            }
            noc.async_write(CoreLocalMem<uint32_t>(ring_base), hwt, 32, {}, {.page_id = u});
            noc.async_write_barrier();
        }
    }
}
