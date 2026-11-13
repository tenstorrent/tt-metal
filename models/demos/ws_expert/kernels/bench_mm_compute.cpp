// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Single-core matmul microbench (TRISC). in0 [NBR * RT rows][KT] and in1 [KT][NBC * CT] are resident in CB 0 / 1.
// Each pass computes every RT x CT output block over all KT (fp32 DEST when the host sets it) and packs it to CB 16:
//   ACT 0: all RT x CT tiles; ACT 1: gate/up pairs (in1 columns interleaved g, u): silu(g) * u on the PACK thread's
//   SFPU, RT x CT/2 h tiles.
// Wall-clock stamps (cycles) at the L1 address RT 0: [0, 1] UNPACK start / end, [2, 3] MATH, [4, 5] PACK.
// CT: 0 RT, 1 CT, 2 KT, 3 NBR, 4 NBC, 5 PASSES, 6 ACT
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/expert_activation.hpp"

constexpr uint32_t rt = get_compile_time_arg_val(0);
constexpr uint32_t ct = get_compile_time_arg_val(1);
constexpr uint32_t kt = get_compile_time_arg_val(2);
constexpr uint32_t nbr = get_compile_time_arg_val(3);
constexpr uint32_t nbc = get_compile_time_arg_val(4);
constexpr uint32_t passes = get_compile_time_arg_val(5);
constexpr uint32_t act = get_compile_time_arg_val(6);
constexpr uint32_t nc = nbc * ct;
constexpr uint32_t out_per_blk = act ? rt * ct / 2 : rt * ct;

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    matmul_block_init(0, 1, false, ct, rt, kt);
    if constexpr (act) {
        expert_activation::init<expert_activation::Kind::Silu>();
    }
    cb_wait_front(0, nbr * rt * kt);
    cb_wait_front(1, kt * nc);
    UNPACK(stamps[0] = wall());
    MATH(stamps[2] = wall());
    PACK(stamps[4] = wall());
    for (uint32_t p = 0; p < passes; ++p) {
        for (uint32_t br = 0; br < nbr; ++br) {
            for (uint32_t bc = 0; bc < nbc; ++bc) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < kt; ++k) {
                    matmul_block(0, 1, br * rt * kt + k, k * nc + bc * ct, 0, false, ct, rt, kt);
                }
                tile_regs_commit();
                cb_reserve_back(16, out_per_blk);
                tile_regs_wait();
                if constexpr (act) {
                    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                    for (uint32_t t = 0; t < rt * ct; t += 2) {
                        expert_activation::tile_pair<expert_activation::Kind::Silu>(t);
                    }
                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                    for (uint32_t t = 0; t < rt * ct; t += 2) {
                        pack_tile(t, 16);
                    }
                } else {
                    for (uint32_t t = 0; t < rt * ct; ++t) {
                        pack_tile(t, 16);
                    }
                }
                tile_regs_release();
                cb_push_back(16, out_per_blk);
            }
        }
    }
    UNPACK(stamps[1] = wall());
    MATH(stamps[3] = wall());
    PACK(stamps[5] = wall());
}
