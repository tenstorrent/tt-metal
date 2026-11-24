// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stage 2 gate/up compute (TRISC): kb_compute's sub-block with x streamed through the x ring (CB 0, XS blocks of
// [MT row tiles][KB K tiles], published by the receiver as they arrive) and the weights resident (CB 1, [KT][WC]).
// Passes of PR row tiles (fp32 DEST half-sync: 4 tiles), each over all NB blocks; the last pass of a sub-block frees
// the blocks as it goes (a block past the ring's end is addressed by a negative offset from the read pointer).
//   MODE 0: today's flat expert - one pair (WC 2), all of K; passes -> activation -> h (CB 16).
//   MODE 1: K split in pairs (WC 4: partner's pair, own pair; half of K): P passes -> the partner's partials (CB 17,
//     bf16); O passes: the partner's partials for this core's pair (CB 2) into DEST by an identity matmul (CB 3), this
//     half of K on top, activation -> h.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/expert_activation.hpp"
#include "kc_common.hpp"

using namespace kc;

constexpr uint32_t PR = 2;
constexpr uint32_t PASSES = MT / PR;
constexpr auto act = expert_activation::Kind::Silu;

uint32_t x_freed = 0;
uint32_t w_freed = 0;  // weight blocks freed (W_STREAM: the ring's read slot w_freed % WS)
uint32_t w_ready = 0;  // weight blocks of the current expert known landed

// tile offset of the current expert's weight block b from the weight ring's read pointer (resident: block b of K)
FORCE_INLINE uint32_t w_block(uint32_t b) {
    if constexpr (!W_STREAM) {
        return b * WBLK;
    } else {
        if (b + 1 > w_ready) {
            cb_wait_front(W_CB, (b + 1) * WBLK);
            w_ready = b + 1;
        }
        const uint32_t front = w_freed % WS;
        return static_cast<uint32_t>(static_cast<int32_t>((front + b) % WS) - static_cast<int32_t>(front)) * WBLK;
    }
}

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

// DEST[r][c] += x[rows row0 .. row0 + PR][this core's K] @ W[:, col0 .. col0 + 2); `last`: frees the x blocks
FORCE_INLINE void mm_pass(uint32_t row0, uint32_t col0, bool last) {
    for (uint32_t b = 0; b < NB; ++b) {
        uint32_t off = 0;
        if (last) {
            cb_wait_front(X_CB, X_BLK);
        } else {
            cb_wait_front(X_CB, (b + 1) * X_BLK);
            const uint32_t front = x_freed % XS;
            off = static_cast<uint32_t>(static_cast<int32_t>((front + b) % XS) - static_cast<int32_t>(front)) * X_BLK;
        }
        const uint32_t wo = w_block(b);
        for (uint32_t k = 0; k < KB; ++k) {
            matmul_block(X_CB, W_CB, off + row0 * KB + k, wo + k * WC + col0, 0, false, 2, PR, KB);
        }
        if (last) {
            cb_pop_front(X_CB, X_BLK);
            ++x_freed;
        }
    }
}

FORCE_INLINE void pack_h(uint32_t row0) {
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    for (uint32_t r = 0; r < PR; ++r) {
        expert_activation::tile_pair<act>(2 * r);
    }
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    for (uint32_t r = 0; r < PR; ++r) {
        pack_tile<true>(2 * r, H_CB, row0 + r);
    }
}

void kernel_main() {
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    compute_kernel_hw_startup<SrcOrder::Reverse>(X_CB, W_CB, H_CB);
    matmul_block_init(X_CB, W_CB, false, 2, PR, KB);
    expert_activation::init<act>();
    if constexpr (!W_STREAM) {
        cb_wait_front(W_CB, KT * WC);
    }
    if constexpr (MODE != 0) {
        cb_wait_front(ID_CB, 4);
    }
    MATH(stamps[0] = wall());
    for (uint32_t sb = 0; sb < NSB; ++sb) {
        if constexpr (MODE == 0) {
            cb_reserve_back(H_CB, MT);
            for (uint32_t p = 0; p < PASSES; ++p) {
                tile_regs_acquire();
                mm_pass(p * PR, 0, p + 1 == PASSES);
                tile_regs_commit();
                tile_regs_wait();
                pack_h(p * PR);
                tile_regs_release();
            }
            cb_push_back(H_CB, MT);
        } else {
            cb_reserve_back(POUT_CB, P_TILES);
            pack_reconfig_data_format(H_CB, POUT_CB);
            for (uint32_t p = 0; p < PASSES; ++p) {
                tile_regs_acquire();
                mm_pass(p * PR, 0, false);
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < PR * 2; ++t) {
                    pack_tile<true>(t, POUT_CB, p * PR * 2 + t);
                }
                tile_regs_release();
            }
            pack_reconfig_data_format(POUT_CB, H_CB);
            cb_push_back(POUT_CB, P_TILES);
            cb_wait_front(PIN_CB, P_TILES);
            cb_reserve_back(H_CB, MT);
            for (uint32_t p = 0; p < PASSES; ++p) {
                tile_regs_acquire();
                reconfig_data_format<SrcOrder::Reverse>(PIN_CB, ID_CB);
                for (uint32_t k = 0; k < 2; ++k) {  // DEST[r][c] = P[r][k] I[k][c]
                    matmul_block(PIN_CB, ID_CB, p * PR * 2 + k, k * 2, 0, false, 2, PR, 2);
                }
                reconfig_data_format<SrcOrder::Reverse>(X_CB, W_CB);
                mm_pass(p * PR, 2, p + 1 == PASSES);
                tile_regs_commit();
                tile_regs_wait();
                pack_h(p * PR);
                tile_regs_release();
            }
            cb_pop_front(PIN_CB, P_TILES);
            cb_push_back(H_CB, MT);
        }
        if constexpr (W_STREAM) {
            if (sb % SBE == SBE - 1) {  // the expert's last sub-block: its weight blocks go
                cb_pop_front(W_CB, NB * WBLK);
                w_freed += NB;
                w_ready = 0;
            }
        }
        if (sb == 0) {
            MATH(stamps[1] = wall());
        }
    }
    MATH(stamps[2] = wall());
}
