// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GDN decode (two-pass form), SFPU side: probes and the vector pass of fused_recurrent_gdn_sfpu.hpp on one core, with
// wall-clock stamps from the MATH thread around REPS repetitions. Inputs c_0..c_4 (all fp32, all UnpackToDestFp32),
// output c_16 (fp32; the host sizes it for every repetition).
//
//   1  DEST map probe:    c_0 = one tile; copied into slot 0, then every sfpi vector v of the slot is overwritten with
//                         1000 * v + vConstTileId (tile id = 2 * lane) and packed -> (row, col) -> (vector, lane)
//   2  sfpi op probe:     c_0 = one 16x32 tile copied into slots 0..3; slot 3 <- copy_dst(slot 0); move_row2_to_row0
//                         on slot 0, zero_rows_from_1 on slot 1, zero_rows_from_2 on slot 2; four output tiles
//   3  DEST persistence:  c_0 = tile A, c_1 = tile B (32x32); 11 acquires per repetition, see the test for the plan
//   4  vector pass:       c_0 = r16[VT], c_1 = v16[VT], c_2 = ecol32, c_3 = bcol16, c_4 = ccol32 ->
//                         per j the pair (u16[j], out16[j]), 2 * VT output tiles of 16x32
//                         MODE 0: 16-row SFPU helpers (faces 0 and 1 of the 16x32 tiles)
//                         MODE 1: 4-row helpers (rows 0..3, the one band the live rows 0 and 2 sit in)
//                         MODE 2: the full-tile (32-row) compute-API helpers, for the bit-equality check
//                         (5 copy_tiles per j in modes 0..2: r16, v16, ecol32, bcol16, then ccol32 once beta is dead)
//                         MODE 3: 4-row helpers with ONE constants tile in c_3 (ebc16: column 0 rows 0..3 = e, 4..7 =
//                         beta, 8..11 = c) read through mul_col_bcast_band0: 3 copy_tiles and one copy_init per j
//   5  exact exp:         c_0 = gcol32, c_1 = a 32x32 data tile; out[0] = exp(gcol32), out[1] = data (.) col(out[0])
//                         MODE 0: exp over the whole tile (VectorMode::RC), MODE 1: faces 0 and 2 only (VectorMode::C),
//                         MODE 2: column 0 only (exp_col0_exact)
//   6  build ebc16:       c_0 = ecol32, c_1 = bcol16, c_2 = ccol32 copied into slots 0, 1, 2; out[0] = build_ebc16 into
//                         the free slot 3, out[1] = build_ebc16 in place in slot 0 (two 16x32 tiles per repetition)
//   7  scalar x tile:     c_0 = a 32x32 scalar tile (e at [0, 0]), c_1 = four random 32x32 tiles -> four tiles = data *
//   e
//                         MODE 0: today's form, mul_col_bcast<32> against a copy of c_0 in slot 3 (3 + 1 tiles, two
//                         acquires) MODE 1: copy c_0 -> slot 0, load_scalar(0), then the four tiles in slots 0..3 in
//                         ONE acquire
//                                 with mul_loaded_scalar<32> (the scalar's slot is reused for data right away)
//                         MODE 2: as MODE 1 without the scalar copy, the load and the multiplies (copies + packs alone)
//                         MODE 3: as MODE 0 with mul_scalar_dst<32> (load + multiply per tile)
// Compile-time args: VARIANT, MODE, REPS, VT. Runtime arg 0: L1 address of the stats buffer (two 32-bit stamps).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#if defined(TRISC_MATH)
#include "internal/tt-1xx/risc_common.h"
#endif
#include "../../../../../ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/kernels/compute/fused_recurrent_gdn_sfpu.hpp"

namespace {
constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
constexpr uint32_t MODE = get_compile_time_arg_val(1);
constexpr uint32_t REPS = get_compile_time_arg_val(2);
constexpr uint32_t VT = get_compile_time_arg_val(3);

constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
constexpr uint32_t cb_in2 = tt::CBIndex::c_2;
constexpr uint32_t cb_in3 = tt::CBIndex::c_3;
constexpr uint32_t cb_in4 = tt::CBIndex::c_4;
constexpr uint32_t cb_out = tt::CBIndex::c_16;

#ifdef TRISC_MATH
volatile tt_l1_ptr uint32_t* g_stats = nullptr;

// Slot 0 <- the identity of its DEST geometry: vector v, lane l holds 1000 * v + 2 * l (vConstTileId = 2 * lane).
inline void probe_fill_slot0() {
    gdn_decode_sfpu::detail::run([] {
#pragma GCC unroll 0
        for (uint32_t v = 0; v < 32; v++) {
            // tile id as a float through the 2^23 trick: bits(2^23) + id reinterpreted is 2^23 + id exactly
            const sfpi::vFloat id = sfpi::as<sfpi::vFloat>(sfpi::vConstTileId + 0x4B000000) - 8388608.0f;
            sfpi::dst_reg[v] = id + static_cast<float>(1000 * v);
        }
    });
}
#endif

inline void stamp(uint32_t slot) {
#ifdef TRISC_MATH
    g_stats[slot] = get_timestamp_32b();
#endif
}

inline void copy_in(uint32_t cb, uint32_t tile, uint32_t slot) {
    copy_init(cb);
    copy_tile(cb, tile, slot);
}

inline void pack_out(uint32_t slot, uint32_t tile) { pack_tile<true>(slot, cb_out, tile); }

// One V tile of the vector pass, the SFPU part, slots: 0 = r16[j], 1 = v16[j], 2 = ecol32, 3 = bcol16 on entry.
// On return slot 1 = u (row 0; rows 1..15 zero), slot 0 = o (row 0; rows 1..15 zero); slots 2, 3 clobbered.
template <uint32_t ROWS>
inline void vector_tile_math_per_j() {
    using namespace gdn_decode_sfpu;
    mul_col_bcast<ROWS>(0, 2);  // row 0 = e * r_k, row 2 = e * r_q
    sub_dst<ROWS>(1, 0);        // row 0 = v - e * r_k
    mul_col_bcast<ROWS>(1, 3);  // row 0 = u
    zero_rows_from_1(1);        // u16: rows 1..15 exactly zero
    move_row2_to_row0(0);       // row 0 = e * r_q, rows 1..15 zero
    copy_in(cb_in4, 0, 3);      // c (beta is dead)
    copy_dst<ROWS>(2, 1);       // u again (e is dead); slot 1 stays u for the pack
    mul_col_bcast<ROWS>(2, 3);  // c * u
    add_dst<ROWS>(0, 2);        // o = e * r_q + c * u
}

// MODE 3: one constants tile, slots on entry: 0 = r16[j], 1 = v16[j], 2 = ebc16 (column 0: rows 0..3 = e, rows 4..7 =
// beta, rows 8..11 = c). Rows 0..3 helpers throughout; slot 3 is free scratch.
inline void vector_tile_math_ebc() {
    using namespace gdn_decode_sfpu;
    mul_col_bcast_band0(0, 2, 0);  // e * r (rows 0..3: row 0 = e r_k, row 2 = e r_q)
    sub_dst<4>(1, 0);              // v - e r_k
    mul_col_bcast_band0(1, 2, 1);  // beta * (v - e r_k) = u
    zero_rows_from_1(1);
    move_row2_to_row0(0);          // e r_q
    copy_dst<4>(3, 1);             // u
    mul_col_bcast_band0(3, 2, 2);  // c * u
    add_dst<4>(0, 3);              // o
}
}  // namespace

void kernel_main() {
#ifdef TRISC_MATH
    g_stats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
#endif
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
    CircularBuffer in0(cb_in0), in1(cb_in1), in2(cb_in2), in3(cb_in3), in4(cb_in4), out(cb_out);

    if constexpr (VARIANT == 1) {
        in0.wait_front(1);
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            out.reserve_back(1);
            tile_regs_acquire();
            copy_in(cb_in0, 0, 0);
#ifdef TRISC_MATH
            probe_fill_slot0();
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_out(0, 0);
            tile_regs_release();
            out.push_back(1);
        }
        stamp(1);
    } else if constexpr (VARIANT == 2) {
        in0.wait_front(1);
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            out.reserve_back(4);
            tile_regs_acquire();
            copy_init(cb_in0);
            copy_tile(cb_in0, 0, 0);
            copy_tile(cb_in0, 0, 1);
            copy_tile(cb_in0, 0, 2);
            gdn_decode_sfpu::copy_dst<16>(3, 0);
            gdn_decode_sfpu::move_row2_to_row0(0);
            gdn_decode_sfpu::zero_rows_from_1(1);
            gdn_decode_sfpu::zero_rows_from_2(2);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < 4; t++) {
                pack_out(t, t);
            }
            tile_regs_release();
            out.push_back(4);
        }
        stamp(1);
    } else if constexpr (VARIANT == 3) {
        // Acquires alternate DEST halves (half-sync): #0, #2, ... are one half, #1, #3, ... the other.
        //   #0, #1: A -> slot 3, B -> slot 0; pack slot 0                (slot 3 of both halves holds A)
        //   #2..#5: B -> slot 0; pack slot 0                             (four acquires that never touch slot 3)
        //   #6, #7: pack slot 3                                          (-> A if an un-packed slot persists)
        //   #8:     A -> slot 2; pack slot 2                             (a packed slot ...)
        //   #9:     B -> slot 0; pack slot 0
        //   #10:    pack slot 2                                          (... read again two acquires later -> A?)
        in0.wait_front(1);
        in1.wait_front(1);
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            out.reserve_back(11);
            for (uint32_t a = 0; a < 11; a++) {
                tile_regs_acquire();
                uint32_t pack_slot = 0;
                if (a < 2) {
                    copy_in(cb_in0, 0, 3);
                    copy_in(cb_in1, 0, 0);
                } else if (a < 6) {
                    copy_in(cb_in1, 0, 0);
                } else if (a < 8) {
                    pack_slot = 3;
                } else if (a == 8) {
                    copy_in(cb_in0, 0, 2);
                    pack_slot = 2;
                } else if (a == 9) {
                    copy_in(cb_in1, 0, 0);
                } else {
                    pack_slot = 2;
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_out(pack_slot, a);
                tile_regs_release();
            }
            out.push_back(11);
        }
        stamp(1);
    } else if constexpr (VARIANT == 4) {
        in0.wait_front(VT);
        in1.wait_front(VT);
        in2.wait_front(1);
        in3.wait_front(1);
        in4.wait_front(1);
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            out.reserve_back(2 * VT);
            for (uint32_t j = 0; j < VT; j++) {
                tile_regs_acquire();
                copy_init(cb_in0);  // 16x32 operands
                copy_tile(cb_in0, j, 0);
                copy_tile(cb_in1, j, 1);
                if constexpr (MODE == 3) {
                    copy_tile(cb_in3, 0, 2);  // ebc16: the one constants tile
                    vector_tile_math_ebc();
                } else {
                    copy_tile(cb_in3, 0, 3);
                    copy_init(cb_in2);  // 32x32 operands
                    copy_tile(cb_in2, 0, 2);
                    if constexpr (MODE == 2) {
                        vector_tile_math_per_j<32>();
                    } else if constexpr (MODE == 1) {
                        vector_tile_math_per_j<4>();
                    } else {
                        vector_tile_math_per_j<16>();
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_out(1, 2 * j);
                pack_out(0, 2 * j + 1);
                tile_regs_release();
            }
            out.push_back(2 * VT);
        }
        stamp(1);
    } else if constexpr (VARIANT == 5) {
        in0.wait_front(1);
        in1.wait_front(1);
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            out.reserve_back(2);
            tile_regs_acquire();
            copy_init(cb_in0);
            copy_tile(cb_in0, 0, 0);
            copy_tile(cb_in1, 0, 1);
            if constexpr (MODE == 2) {
                gdn_decode_sfpu::exp_col0_exact(0);
            } else if constexpr (MODE == 1) {
                gdn_decode_sfpu::exp_tile_exact<ckernel::VectorMode::C>(0);
            } else {
                gdn_decode_sfpu::exp_tile_exact<ckernel::VectorMode::RC>(0);
            }
            gdn_decode_sfpu::mul_col_bcast<32>(1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_out(0, 0);
            pack_out(1, 1);
            tile_regs_release();
            out.push_back(2);
        }
        stamp(1);
    } else if constexpr (VARIANT == 6) {
        in0.wait_front(1);
        in1.wait_front(1);
        in2.wait_front(1);
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            out.reserve_back(2);
            tile_regs_acquire();
            copy_in(cb_in0, 0, 0);                     // ecol32 (32x32)
            copy_in(cb_in1, 0, 1);                     // bcol16 (16x32)
            copy_in(cb_in2, 0, 2);                     // ccol32 (32x32)
            gdn_decode_sfpu::build_ebc16(0, 1, 2, 3);  // into the free slot 3 (ecol32 in slot 0 stays packable)
            gdn_decode_sfpu::build_ebc16(0, 1, 2);     // in place in slot 0
            tile_regs_commit();
            tile_regs_wait();
            pack_out(3, 0);
            pack_out(0, 1);
            tile_regs_release();
            out.push_back(2);
        }
        stamp(1);
    } else if constexpr (VARIANT == 7) {
        in0.wait_front(1);
        in1.wait_front(4);
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            out.reserve_back(4);
            if constexpr (MODE == 0 || MODE == 3) {
                for (uint32_t g = 0; g < 2; g++) {  // tiles 0..2 with the scalar in slot 3, then tile 3
                    const uint32_t n = (g == 0) ? 3 : 1;
                    tile_regs_acquire();
                    copy_init(cb_in0);
                    copy_tile(cb_in0, 0, 3);
                    for (uint32_t i = 0; i < n; i++) {
                        copy_tile(cb_in1, 3 * g + i, i);
                    }
                    for (uint32_t i = 0; i < n; i++) {
                        if constexpr (MODE == 0) {
                            gdn_decode_sfpu::mul_col_bcast<32>(i, 3);
                        } else {
                            gdn_decode_sfpu::mul_scalar_dst<32>(i, 3);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < n; i++) {
                        pack_out(i, 3 * g + i);
                    }
                    tile_regs_release();
                }
            } else {
                tile_regs_acquire();
                copy_init(cb_in0);
                if constexpr (MODE == 1) {
                    copy_tile(cb_in0, 0, 0);
                    gdn_decode_sfpu::load_scalar(0);  // slot 0 is reused for data right after
                }
                for (uint32_t i = 0; i < 4; i++) {
                    copy_tile(cb_in1, i, i);
                }
                if constexpr (MODE == 1) {
                    for (uint32_t i = 0; i < 4; i++) {
                        gdn_decode_sfpu::mul_loaded_scalar<32>(i);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < 4; i++) {
                    pack_out(i, i);
                }
                tile_regs_release();
            }
            out.push_back(4);
        }
        stamp(1);
    }
}
