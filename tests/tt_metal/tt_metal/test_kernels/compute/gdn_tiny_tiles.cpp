// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Spike A (GDN decode v2): the three tiny-tile matmul shapes the two-pass decode kernel rests on, next to their padded
// 32x32 forms, on one core, with wall-clock stamps from the MATH thread around REPS repetitions.
//
//   1  two-row read, tiny:    in0 = [q; k] as KT tiles of 2x32 (c_0), in1 = S as KT*VT tiles of 32x32 (c_1)
//                             -> VT tiles of 2x32 (c_16), each the sum over KT products
//   2  two-row read, padded:  the same with in0 padded to 32x32 tiles (rows 0-1 live) -> 32x32 outputs
//   3  rank-1 update, tiny:   seed = S tile copied into DEST (c_2), in0 = k^T as KT tiles of 32x16 (k in column 0),
//                             in1 = u as VT tiles of 16x32 (u in row 0) -> KT*VT tiles of 32x32 = S + k^T u
//   4  rank-1 update, padded: the same with in0 and in1 as 32x32 tiles
//   5  scalar q.k:            in0 = [q; k] 2x32 tiles, in1 = k padded into column 0 of 32x32 tiles -> one 2x32 tile
//                             whose column 0 holds (q.k, k.k)
//   6  copy seed:             S tiles copied into DEST and packed back (the exactness of the seed)
// Compile-time args: VARIANT, REPS, KT, VT. Runtime arg 0: L1 address of the stats buffer (two 32-bit stamps).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#if defined(TRISC_MATH)
#include "internal/tt-1xx/risc_common.h"
#endif

namespace {
constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
constexpr uint32_t REPS = get_compile_time_arg_val(1);
constexpr uint32_t KT = get_compile_time_arg_val(2);
constexpr uint32_t VT = get_compile_time_arg_val(3);
constexpr uint32_t GROUP = 4;  // fp32 tiles per DEST half

constexpr auto cb_a = tt::CBIndex::c_0;
constexpr auto cb_b = tt::CBIndex::c_1;
constexpr auto cb_s = tt::CBIndex::c_2;
constexpr auto cb_out = tt::CBIndex::c_16;

#ifdef TRISC_MATH
volatile tt_l1_ptr uint32_t* g_stats = nullptr;
#endif
inline void stamp(uint32_t slot) {
#ifdef TRISC_MATH
    g_stats[slot] = get_timestamp_32b();
#endif
}
}  // namespace

void kernel_main() {
#ifdef TRISC_MATH
    g_stats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
#endif
    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    CircularBuffer a(cb_a), b(cb_b), s(cb_s), out(cb_out);

    if constexpr (VARIANT == 1 || VARIANT == 2) {
        a.wait_front(KT);
        b.wait_front(KT * VT);
        matmul_init(cb_a, cb_b, 0);
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            out.reserve_back(VT);
            for (uint32_t j = 0; j < VT; j++) {
                tile_regs_acquire();
                for (uint32_t ki = 0; ki < KT; ki++) {
                    matmul_tiles(cb_a, cb_b, ki, ki * VT + j, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_out, j);
                tile_regs_release();
            }
            out.push_back(VT);
        }
        stamp(1);
    } else if constexpr (VARIANT == 3 || VARIANT == 4) {
        a.wait_front(KT);
        b.wait_front(VT);
        s.wait_front(KT * VT);
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            out.reserve_back(KT * VT);
            for (uint32_t i = 0; i < KT; i++) {
                for (uint32_t j0 = 0; j0 < VT; j0 += GROUP) {
                    const uint32_t nn = (VT - j0 < GROUP) ? (VT - j0) : GROUP;
                    tile_regs_acquire();
                    copy_init(cb_s);
                    for (uint32_t j = 0; j < nn; j++) {
                        copy_tile(cb_s, i * VT + j0 + j, j);
                    }
                    matmul_init(cb_a, cb_b, 0);
                    for (uint32_t j = 0; j < nn; j++) {
                        matmul_tiles(cb_a, cb_b, i, j0 + j, j);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t j = 0; j < nn; j++) {
                        pack_tile<true>(j, cb_out, i * VT + j0 + j);
                    }
                    tile_regs_release();
                }
            }
            out.push_back(KT * VT);
        }
        stamp(1);
    } else if constexpr (VARIANT == 7 || VARIANT == 8) {
        // rank-1 cost decomposition in groups of GROUP tiles: 7 = the matmul alone (no seed), 8 = the copy seed alone
        a.wait_front(KT);
        b.wait_front(VT);
        s.wait_front(KT * VT);
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            out.reserve_back(KT * VT);
            for (uint32_t i = 0; i < KT; i++) {
                for (uint32_t j0 = 0; j0 < VT; j0 += GROUP) {
                    const uint32_t nn = (VT - j0 < GROUP) ? (VT - j0) : GROUP;
                    tile_regs_acquire();
                    if constexpr (VARIANT == 8) {
                        copy_init(cb_s);
                        for (uint32_t j = 0; j < nn; j++) {
                            copy_tile(cb_s, i * VT + j0 + j, j);
                        }
                    } else {
                        matmul_init(cb_a, cb_b, 0);
                        for (uint32_t j = 0; j < nn; j++) {
                            matmul_tiles(cb_a, cb_b, i, j0 + j, j);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t j = 0; j < nn; j++) {
                        pack_tile<true>(j, cb_out, i * VT + j0 + j);
                    }
                    tile_regs_release();
                }
            }
            out.push_back(KT * VT);
        }
        stamp(1);
    } else if constexpr (VARIANT == 5) {
        a.wait_front(KT);
        b.wait_front(KT);
        matmul_init(cb_a, cb_b, 0);
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            out.reserve_back(1);
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < KT; ki++) {
                matmul_tiles(cb_a, cb_b, ki, ki, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_out, 0);
            tile_regs_release();
            out.push_back(1);
        }
        stamp(1);
    } else if constexpr (VARIANT == 6) {
        s.wait_front(KT * VT);
        copy_init(cb_s);
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            out.reserve_back(KT * VT);
            for (uint32_t t = 0; t < KT * VT; t++) {
                tile_regs_acquire();
                copy_tile(cb_s, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_out, t);
                tile_regs_release();
            }
            out.push_back(KT * VT);
        }
        stamp(1);
    }
}
