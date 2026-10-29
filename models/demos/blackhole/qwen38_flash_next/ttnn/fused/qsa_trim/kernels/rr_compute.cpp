// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_trim reduce_w2n compute: row_reduce/kernels/compute.cpp's summation tree on the QSA output reduce, with the
// widen (bf16 -> fp32) and the narrow (fp32 -> bf16) folded in.  The partials arrive as bf16 tiles (CB_IN, CB_FS,
// CB_BS: the out-projection's bf16 values, exactly the values today's widen program turns into fp32), every sum is
// the same add_tiles with the same operand values, packed to fp32 (CB_F, CB_B, CB_G: today's fp32 CBs, so the same tf32
// unpack of the intermediate sums), and the FINAL add is followed in DST by the typecast op's SFPU conversion fp32 ->
// bf16 and packed bf16 into CB_OUT (today: pack fp32, then the Typecast program reads the fp32 tile at full precision).
// Compile-time args: 0 CB_IN, 1 CB_FS, 2 CB_BS, 3 CB_F, 4 CB_B, 5 CB_G, 6 CB_OUT, 7 OUT_TILES, 8 RING.  Runtime arg 0:
// L.

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "../../kernels/zones.h"

constexpr uint32_t CB_IN = get_compile_time_arg_val(0);
constexpr uint32_t CB_FS = get_compile_time_arg_val(1);
constexpr uint32_t CB_BS = get_compile_time_arg_val(2);
constexpr uint32_t CB_F = get_compile_time_arg_val(3);
constexpr uint32_t CB_B = get_compile_time_arg_val(4);
constexpr uint32_t CB_G = get_compile_time_arg_val(5);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(6);
constexpr uint32_t OUT_TILES = get_compile_time_arg_val(7);
constexpr uint32_t RING = get_compile_time_arg_val(8);

// cb_out <- front(a) + front(b) (a bf16 or fp32 tile each); pops both fronts after the pack.  `last`: the sum leaves
// as bf16 (the typecast op's conversion on the fp32 sum in DST)
template <bool last>
FORCE_INLINE void add_pop(uint32_t a, uint32_t b, uint32_t out) {
    cb_wait_front(a, 1);
    cb_wait_front(b, 1);
    cb_reserve_back(out, 1);
    reconfig_data_format(a, b);
    tile_regs_acquire();
    add_tiles(a, b, 0, 0, 0);
    if constexpr (last) {
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(out);
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
    cb_pop_front(a, 1);
    cb_pop_front(b, 1);
}

void kernel_main() {
    const uint32_t L = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_IN, CB_FS, CB_F);
    {
        FUSED_ZONE("fz_qt_rr_sum");
        for (uint32_t j = 0; j < OUT_TILES; ++j) {
            add_init(CB_IN, CB_FS, false);
            uint32_t own_sum = CB_FS;  // G = p_L + F (L >= 1), the own term itself for L == 0
            if (L >= 1) {
                uint32_t acc = CB_FS;  // F, seeded with p_0
                for (uint32_t k = 1; k < L; ++k) {
                    add_pop<false>(CB_IN, acc, CB_F);
                    acc = CB_F;
                }
                if (L == RING - 1) {
                    add_pop<true>(CB_IN, acc, CB_OUT);  // p_L + F, the final sum
                } else {
                    add_pop<false>(CB_IN, acc, CB_G);  // p_L + F
                }
                own_sum = CB_G;
            }
            if (L < RING - 1) {
                uint32_t acc = CB_BS;  // B, seeded with p_{R-1}
                for (uint32_t k = RING - 2; k > L; --k) {
                    add_pop<false>(CB_IN, acc, CB_B);
                    acc = CB_B;
                }
                add_pop<true>(own_sum, acc, CB_OUT);
            }
        }
    }
}
