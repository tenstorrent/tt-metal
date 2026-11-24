// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stage 1 single-core bench (TRISC) of GLM-5.3's gate/up sub-block, resident operands, LoFi, fp32 DEST half-sync.
//
// A sub-block is RT_SB row tiles (128 rows: 4) of x [RT_SB][KT] (CB 0, bfp8) times this core's weights (CB 1, bfp4,
// [KT][WC] tiles row-major by K). It runs in passes of PR row tiles (DEST holds 4 fp32 tiles: 2 rows x (gate, up)).
//   MODE 0 (today's flat expert): KT = H tiles (192), WC 2 (one pair: gate, up). Per pass: matmul over all KT, the
//     activation on the PACK thread's SFPU, PR h tiles packed to CB 16.
//   MODE 1 / 2 (K split in pairs): KT = H / 2 tiles (this core's half of K), WC 4: columns 0-1 the partner's pair
//     (gate, up), 2-3 this core's own pair. Per sub-block: the P passes (partner's pair over this half of K, the
//     fp32 DEST packed to CB 17 as the partner's partials), then the O passes (own pair): DEST first holds the
//     partner's partials for these rows (CB 2), the matmul accumulates this half of K on top, then the activation
//     and the h pack, exactly as MODE 0's passes.
//     MODE 1: the partials enter DEST through copy_block (CB 2 fp32, UnpackToDestFp32).
//     MODE 2: the partials enter DEST through a tile matmul against identity tiles (CB 2 partials [PR][2] as in0,
//     CB 3 = [[I, 0], [0, I]] as in1, kt 2): no copy init, no matmul re-init.
// Wall-clock stamps (cycles) at the runtime-arg L1 address: [0, 1] UNPACK start / end, [2, 3] MATH, [4, 5] PACK.
// CT: 0 MODE, 1 KT, 2 RT_SB, 3 PR, 4 NSB (sub-blocks a launch, operands reused), 5 WC
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/expert_activation.hpp"

constexpr uint32_t MODE = get_compile_time_arg_val(0);
constexpr uint32_t KT = get_compile_time_arg_val(1);
constexpr uint32_t RT_SB = get_compile_time_arg_val(2);
constexpr uint32_t PR = get_compile_time_arg_val(3);
constexpr uint32_t NSB = get_compile_time_arg_val(4);
constexpr uint32_t WC = get_compile_time_arg_val(5);
constexpr uint32_t PASSES = RT_SB / PR;

constexpr uint32_t x_cb = 0, w_cb = 1, pin_cb = 2, id_cb = 3, h_cb = 16, pout_cb = 17;
constexpr auto act = expert_activation::Kind::Silu;

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

// DEST[r][c] += x[rows row0 .. row0 + PR)[all KT] @ W[:, col0 .. col0 + 2)
FORCE_INLINE void mm_pass(uint32_t row0, uint32_t col0) {
    for (uint32_t k = 0; k < KT; ++k) {
        matmul_block(x_cb, w_cb, row0 * KT + k, k * WC + col0, 0, false, 2, PR, KT);
    }
}

FORCE_INLINE void pack_h(uint32_t row0) {
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    for (uint32_t r = 0; r < PR; ++r) {
        expert_activation::tile_pair<act>(2 * r);
    }
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    for (uint32_t r = 0; r < PR; ++r) {
        pack_tile<true>(2 * r, h_cb, row0 + r);
    }
}

void kernel_main() {
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    compute_kernel_hw_startup<SrcOrder::Reverse>(x_cb, w_cb, h_cb);
    matmul_block_init(x_cb, w_cb, false, 2, PR, KT);
    expert_activation::init<act>();
    cb_wait_front(x_cb, RT_SB * KT);
    cb_wait_front(w_cb, KT * WC);
    if constexpr (MODE != 0) {
        cb_wait_front(pin_cb, RT_SB * 2);
    }
    if constexpr (MODE == 2) {
        cb_wait_front(id_cb, 4);
    }
    UNPACK(stamps[0] = wall());
    MATH(stamps[2] = wall());
    PACK(stamps[4] = wall());
    for (uint32_t sb = 0; sb < NSB; ++sb) {
        if constexpr (MODE == 0) {
            cb_reserve_back(h_cb, RT_SB);
            for (uint32_t p = 0; p < PASSES; ++p) {
                tile_regs_acquire();
                mm_pass(p * PR, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_h(p * PR);
                tile_regs_release();
            }
            cb_push_back(h_cb, RT_SB);
        } else {
            // P passes: the partner's pair over this core's half of K -> the partner's partials (CB 17)
            cb_reserve_back(pout_cb, RT_SB * 2);
            pack_reconfig_data_format(h_cb, pout_cb);
            for (uint32_t p = 0; p < PASSES; ++p) {
                tile_regs_acquire();
                mm_pass(p * PR, 0);
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < PR * 2; ++t) {
                    pack_tile<true>(t, pout_cb, p * PR * 2 + t);
                }
                tile_regs_release();
            }
            pack_reconfig_data_format(pout_cb, h_cb);
            cb_push_back(pout_cb, RT_SB * 2);
            // O passes: the partner's partials into DEST, this half of K on top, the activation, h
            cb_reserve_back(h_cb, RT_SB);
            for (uint32_t p = 0; p < PASSES; ++p) {
                tile_regs_acquire();
                if constexpr (MODE == 1) {
                    reconfig_data_format_srca(w_cb, pin_cb);
                    copy_init(pin_cb);
                    copy_block(pin_cb, p * PR * 2, 0, PR * 2);
                    reconfig_data_format_srca(pin_cb, w_cb);
                    matmul_block_init(x_cb, w_cb, false, 2, PR, KT);
                } else {
                    reconfig_data_format<SrcOrder::Reverse>(pin_cb, id_cb);  // (srcB <- partials, srcA <- identity)
                    for (uint32_t k = 0; k < 2; ++k) {  // (DEST[r][c] += P[r][k] I[k][c])
                        matmul_block(pin_cb, id_cb, p * PR * 2 + k, k * 2, 0, false, 2, PR, 2);
                    }
                    reconfig_data_format<SrcOrder::Reverse>(x_cb, w_cb);
                }
                mm_pass(p * PR, 2);
                tile_regs_commit();
                tile_regs_wait();
                pack_h(p * PR);
                tile_regs_release();
            }
            cb_push_back(h_cb, RT_SB);
        }
    }
    UNPACK(stamps[1] = wall());
    MATH(stamps[3] = wall());
    PACK(stamps[5] = wall());
}
