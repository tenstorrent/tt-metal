// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One core's work for one row tile of an expert, all operands resident (TRISC, fp32 DEST, half sync), repeated PASSES
// times: today's 1 x 20 split against the K-split 10 x 2 split. Outputs are packed out of order into reserved CB space
// and never pushed (timing only).
//   RB_MODE 0 (1 x 20, today): gate/up 1 x 2 (one pair) over K 120 (CB 0 x bfp8, CB 1 W bfp4) + SwiGLU (PACK SFPU) ->
//     1 h tile (CB 17, bfp8); down 6 columns (2 x 1x3) over K 20 (CB 2 h bfp8, CB 3 W_down bfp4) -> CB 18 (bf16)
//   RB_MODE 1 (10 x 2): gate/up 5 x 1x4 over K 12 -> 20 partial tiles (CB 16, RB_PDT); fold of C contributions of
//     this core's (gate, up) pair (CB 4, RB_PDT) + SwiGLU -> 1 h tile; the same down
//   RB_MODE 2: RB_MODE 1 without the fold / SwiGLU (the matmuls alone)
//   RB_MODE 3: RB_MODE 0 without the SwiGLU
// Stamps at RT 0: [0, 1] UNPACK, [2, 3] MATH, [4, 5] PACK. CT: 0 PASSES, 1 C
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/expert_activation.hpp"
#include "glu_fused.hpp"

constexpr uint32_t PASSES = get_compile_time_arg_val(0);
constexpr uint32_t C = get_compile_time_arg_val(1);
constexpr uint32_t X = 0, W = 1, H = 2, WD = 3, P = 4, PO = 16, HO = 17, YO = 18;
constexpr uint32_t KD = 20, DC = 6, DB = 3;

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

FORCE_INLINE void swiglu_pack(uint32_t d) {
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    ws_glu::tile_pair<2>(d);
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
}

FORCE_INLINE void down() {
    reconfig_data_format(WD, H);
    pack_reconfig_data_format(YO);
    matmul_block_init(H, WD, false, DB, 1, KD);
    for (uint32_t b = 0; b < DC / DB; ++b) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < KD; ++k) {
            matmul_block(H, WD, k, k * DC + b * DB, 0, false, DB, 1, KD);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < DB; ++t) {
            pack_tile<true>(t, YO, b * DB + t);
        }
        tile_regs_release();
    }
}

template <uint32_t IN>
FORCE_INLINE void fold1(uint32_t d) {  // C contributions of tile 0 of each 2-tile contribution (gate 0, up 1)
    constexpr uint32_t t = (IN == 0) ? 0 : 1;
    uint32_t c;
    if constexpr (C & 1u) {
        copy_init(P);
        copy_tile(P, t, d);
        c = 1;
    } else {
        add_init(P, P, false);
        add_tiles(P, P, t, 2 + t, d);
        c = 2;
    }
    if constexpr (C > 2) {
        add_init(P, P, true);
        for (; c + 1 < C; c += 2) {
            add_tiles(P, P, c * 2 + t, (c + 1) * 2 + t, d);
        }
        add_init(P, P, false);
    }
}

void kernel_main() {
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    compute_kernel_hw_startup(X, W, PO);
    ws_glu::init<2>();
    cb_wait_front(X, RB_MODE == 0 || RB_MODE == 3 ? 120 : 12);
    cb_wait_front(W, 240);
    cb_wait_front(H, KD);
    cb_wait_front(WD, KD * DC);
    cb_wait_front(P, 2 * C);
    cb_reserve_back(PO, 20);
    cb_reserve_back(HO, 1);
    cb_reserve_back(YO, DC);
    UNPACK(st[0] = wall());
    MATH(st[2] = wall());
    PACK(st[4] = wall());
    for (uint32_t p = 0; p < PASSES; ++p) {
#if RB_MODE == 0 || RB_MODE == 3
        reconfig_data_format(W, X);
        pack_reconfig_data_format(HO);
        matmul_block_init(X, W, false, 2, 1, 120);
        tile_regs_acquire();
        for (uint32_t k = 0; k < 120; ++k) {
            matmul_block(X, W, k, k * 2, 0, false, 2, 1, 120);
        }
        tile_regs_commit();
        tile_regs_wait();
#if RB_MODE == 0
        swiglu_pack(0);
#endif
        pack_tile<true>(0, HO, 0);
        tile_regs_release();
#else
        // gate / up partials: 5 blocks of 1 x 4 over K 12
        reconfig_data_format(W, X);
        pack_reconfig_data_format(PO);
        matmul_block_init(X, W, false, 4, 1, 12);
        for (uint32_t b = 0; b < 5; ++b) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < 12; ++k) {
                matmul_block(X, W, k, k * 20 + b * 4, 0, false, 4, 1, 12);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < 4; ++t) {
                pack_tile<true>(t, PO, b * 4 + t);
            }
            tile_regs_release();
        }
#if RB_MODE == 1
        // the fold of this core's pair: gate sum -> DST 0, up sum -> DST 1, SwiGLU, h -> CB 17
        reconfig_data_format(P, P);
        pack_reconfig_data_format(HO);
        tile_regs_acquire();
        fold1<0>(0);
        fold1<1>(1);
        tile_regs_commit();
        tile_regs_wait();
        swiglu_pack(0);
        pack_tile<true>(0, HO, 0);
        tile_regs_release();
#endif
#endif
        down();
    }
    UNPACK(st[1] = wall());
    MATH(st[3] = wall());
    PACK(st[5] = wall());
}
