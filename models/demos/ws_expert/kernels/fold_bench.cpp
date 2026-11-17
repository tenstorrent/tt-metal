// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reduce-scatter fold microbench (TRISC, fp32 DEST, half sync: 4 DST tiles a half). CB 0 holds C gate contributions
// of NT tiles each (contribution c at tiles [c * NT, (c + 1) * NT)), CB 1 the same for up; both resident (bf16 or
// bfp8, the partial format under test). A window of 2 outputs: gate sums fold into DST 0 / 2, up sums into DST 1 / 3
// (moe_fused_swiglu's fold: a seed (copy if C odd, else one add of two contributions), then add_tiles with
// acc_to_dest, two contributions a call), then the lane's SwiGLU (ws_glu MODE 2, PACK thread) on each pair and the
// h tile packed to CB 16. PASSES passes over the same resident inputs.
// FB_MODE 0: fold + SwiGLU; 1: fold only (gate sums packed: the FPU / unpack cost alone); 2: copy only (C = 1 seed +
// SwiGLU: the floor).
// Stamps at RT 0: [0, 1] UNPACK start / end, [2, 3] MATH, [4, 5] PACK.
// CT: 0 C, 1 NT, 2 PASSES
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/expert_activation.hpp"
#include "glu_fused.hpp"

constexpr uint32_t C = get_compile_time_arg_val(0);
constexpr uint32_t NT = get_compile_time_arg_val(1);
constexpr uint32_t PASSES = get_compile_time_arg_val(2);
constexpr uint32_t G = 0, U = 1, OUT = 16;

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

// fold C contributions of tiles t0, t0 + 1 of CB `in` into DST d0, d0 + 2 (seed overwrites); one init a step for
// both tiles of the window (moe_fused_swiglu's order)
template <uint32_t IN>
FORCE_INLINE void fold(uint32_t t0, uint32_t d0) {
    uint32_t c;
    if constexpr (C & 1u) {
        copy_init(IN);
        copy_tile(IN, t0, d0);
        copy_tile(IN, t0 + 1, d0 + 2);
        c = 1;
    } else {
        add_init(IN, IN, false);
        add_tiles(IN, IN, t0, NT + t0, d0);
        add_tiles(IN, IN, t0 + 1, NT + t0 + 1, d0 + 2);
        c = 2;
    }
    if constexpr (C > 2) {
        add_init(IN, IN, true);
        for (; c + 1 < C; c += 2) {
            add_tiles(IN, IN, c * NT + t0, (c + 1) * NT + t0, d0);
            add_tiles(IN, IN, c * NT + t0 + 1, (c + 1) * NT + t0 + 1, d0 + 2);
        }
        add_init(IN, IN, false);
    }
}

void kernel_main() {
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    compute_kernel_hw_startup(G, U, OUT);
    ws_glu::init<2>();
    cb_wait_front(G, C * NT);
    cb_wait_front(U, C * NT);
    UNPACK(st[0] = wall());
    MATH(st[2] = wall());
    PACK(st[4] = wall());
    for (uint32_t p = 0; p < PASSES; ++p) {
        for (uint32_t t0 = 0; t0 < NT; t0 += 2) {
            tile_regs_acquire();
#if FB_MODE == 2
            copy_init(G);
            copy_tile(G, t0, 0);
            copy_tile(G, t0 + 1, 2);
            copy_init(U);
            copy_tile(U, t0, 1);
            copy_tile(U, t0 + 1, 3);
#else
            fold<G>(t0, 0);
            fold<U>(t0, 1);
#endif
            tile_regs_commit();
            cb_reserve_back(OUT, 2);
            tile_regs_wait();
#if FB_MODE != 1
            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
            ws_glu::tile_pair<2>(0);
            ws_glu::tile_pair<2>(2);
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
#endif
            pack_tile(0, OUT);
            pack_tile(2, OUT);
            tile_regs_release();
            cb_push_back(OUT, 2);
        }
    }
    UNPACK(st[1] = wall());
    MATH(st[3] = wall());
    PACK(st[5] = wall());
}
