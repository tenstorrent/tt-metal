// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert gate/up compute (TRISC). The core owns one pair group (NPAIR gate/up pairs, in1 columns interleaved
// g0 u0 g1 u1 ..) and a contiguous run of the expert's flattened (row, K-tile) work in its grid column (stream-K):
// pieces (row r, K [k0, k1)). Per piece, DST[1 x 2 NPAIR] (fp32) += x[r, k0:k1] @ Wgu[k0:k1, pair group]:
//   PRE  (k1 < KT: the row goes on in the next core): DST starts from that core's fp32 partial (CB 5, copied in)
//   SEND (k0 > 0: the row began in the previous core): DST is packed fp32 to CB 4 (sent to the previous core)
//   else silu(gate) * up on the PACK thread's SFPU -> NPAIR h tiles to CB 16.
// x: CB 0, the core's XT-tile store; this core's flattened work position t reads store tile t % XT.
// Wgu: CB 1, [KT][2 NPAIR] tiles, resident.
// RT: 0 stamps addr, 1 NPIECE, then per piece: k0, k1, flags (1 PRE, 2 SEND), xoff (work position of k0)
// CT: 0 KT, 1 NPAIR, 2 XT, 3 NEXP
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/expert_activation.hpp"

constexpr uint32_t x_cb = tt::CBIndex::c_0;
constexpr uint32_t w_cb = tt::CBIndex::c_1;
constexpr uint32_t pout_cb = tt::CBIndex::c_4;
constexpr uint32_t pin_cb = tt::CBIndex::c_5;
constexpr uint32_t h_cb = tt::CBIndex::c_16;

constexpr uint32_t kt = get_compile_time_arg_val(0);
constexpr uint32_t npair = get_compile_time_arg_val(1);
constexpr uint32_t xt = get_compile_time_arg_val(2);
constexpr uint32_t nexp = get_compile_time_arg_val(3);
constexpr uint32_t gw = 2 * npair;
static_assert(gw <= 4, "fp32 half-sync DST holds 4 tiles");

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    const uint32_t npiece = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(x_cb, w_cb, h_cb);
    matmul_block_init(x_cb, w_cb, false, gw, 1, kt);
    expert_activation::init<expert_activation::Kind::Silu>();
    cb_wait_front(x_cb, xt);
    cb_wait_front(w_cb, kt * gw);
    PACK(stamps[0] = wall());
    bool h_fmt = true;  // packer configured for h (else for the fp32 partial)
    for (uint32_t e = 0; e < nexp; ++e) {
        for (uint32_t p = 0; p < npiece; ++p) {
            const uint32_t k0 = get_arg_val<uint32_t>(2 + 4 * p);
            const uint32_t k1 = get_arg_val<uint32_t>(3 + 4 * p);
            const uint32_t flags = get_arg_val<uint32_t>(4 + 4 * p);
            const uint32_t xoff = get_arg_val<uint32_t>(5 + 4 * p);
            tile_regs_acquire();
            if (flags & 1) {  // start from the next core's partial (exact: unpacked straight to fp32 DEST)
                cb_wait_front(pin_cb, gw);
                reconfig_data_format_srca(w_cb, pin_cb);
                copy_init(pin_cb);
                for (uint32_t t = 0; t < gw; ++t) {
                    copy_tile(pin_cb, t, t);
                }
                cb_pop_front(pin_cb, gw);
                reconfig_data_format_srca(pin_cb, w_cb);
                matmul_block_init(x_cb, w_cb, false, gw, 1, kt);
            }
            uint32_t xi = xoff % xt;
            for (uint32_t k = k0; k < k1; ++k) {
                matmul_block(x_cb, w_cb, xi, k * gw, 0, false, gw, 1, kt);
                if (++xi == xt) {
                    xi = 0;
                }
            }
            tile_regs_commit();
            if (flags & 2) {  // partial for the previous core: fp32
                cb_reserve_back(pout_cb, gw);
                tile_regs_wait();
                if (h_fmt) {
                    pack_reconfig_data_format(h_cb, pout_cb);
                    h_fmt = false;
                }
                for (uint32_t t = 0; t < gw; ++t) {
                    pack_tile(t, pout_cb);
                }
                tile_regs_release();
                cb_push_back(pout_cb, gw);
            } else {
                cb_reserve_back(h_cb, npair);
                tile_regs_wait();
                if (!h_fmt) {
                    pack_reconfig_data_format(pout_cb, h_cb);
                    h_fmt = true;
                }
                PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                for (uint32_t t = 0; t < gw; t += 2) {
                    expert_activation::tile_pair<expert_activation::Kind::Silu>(t);
                }
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                for (uint32_t t = 0; t < gw; t += 2) {
                    pack_tile(t, h_cb);
                }
                tile_regs_release();
                cb_push_back(h_cb, npair);
            }
        }
        PACK(stamps[1 + e] = wall());
    }
}
