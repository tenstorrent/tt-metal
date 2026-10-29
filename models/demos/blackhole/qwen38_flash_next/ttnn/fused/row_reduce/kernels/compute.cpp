// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// row_reduce compute: per output tile j the stock line reduce-scatter's summation tree of the logical rank L,
//   out = (p_L + F) + B,  F = p_{L-1} + (p_{L-2} + ... + (p_1 + p_0)),  B = p_{L+1} + (p_{L+2} + ... + (p_{R-2} +
//   p_{R-1})),
// one add_tiles per binary add and one pack (to the CB's own format: the input dtype) after every add, as
// line_reduction.cpp does (bf16: 16-bit dest, a bf16 rounding per add; fp32: HiFi4 + fp32 dest).  The tiles arrive in
// the order transport.cpp pushes them: CB_FS (p_0, or the own term for L == 0), CB_BS (p_{R-1}), CB_IN (p_1 .. p_{L-1},
// p_L, p_{R-2} .. p_{L+1}).  Every add reads the front tile of two CBs. Compile-time args: 0 CB_IN, 1 CB_FS, 2 CB_BS, 3
// CB_F, 4 CB_B, 5 CB_G, 6 CB_OUT, 7 OUT_TILES, 8 RING.  Runtime arg 0: L.

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
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

// cb_out <- front(a) + front(b); pops both fronts after the pack
FORCE_INLINE void add_pop(uint32_t a, uint32_t b, uint32_t out) {
    cb_wait_front(a, 1);
    cb_wait_front(b, 1);
    cb_reserve_back(out, 1);
    tile_regs_acquire();
    add_tiles(a, b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
    cb_pop_front(a, 1);
    cb_pop_front(b, 1);
}

void kernel_main() {
    const uint32_t L = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_IN, CB_FS, CB_OUT);
    add_init(CB_IN, CB_FS, false);
    {
        FUSED_ZONE("fz_rr_sum");
        for (uint32_t j = 0; j < OUT_TILES; ++j) {
            uint32_t own_sum = CB_FS;  // G = p_L + F (L >= 1), the own term itself for L == 0
            if (L >= 1) {
                uint32_t acc = CB_FS;  // F, seeded with p_0
                for (uint32_t k = 1; k < L; ++k) {
                    add_pop(CB_IN, acc, CB_F);
                    acc = CB_F;
                }
                add_pop(CB_IN, acc, L == RING - 1 ? CB_OUT : CB_G);  // p_L + F
                own_sum = CB_G;
            }
            if (L < RING - 1) {
                uint32_t acc = CB_BS;  // B, seeded with p_{R-1}
                for (uint32_t k = RING - 2; k > L; --k) {
                    add_pop(CB_IN, acc, CB_B);
                    acc = CB_B;
                }
                add_pop(own_sum, acc, CB_OUT);
            }
        }
    }
}
