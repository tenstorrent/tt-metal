// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert column-cluster compute (TRISC). A cluster (one NoC column of GC cores) owns a stream-K share of the
// expert's (row, K-tile) work; core j of the cluster owns gate/up pairs [NPAIR j, NPAIR (j + 1)) and down columns
// [DCOLS j, DCOLS (j + 1)). Per expert, in order, for each gate/up piece (row r, K [k0, k1)):
//   DST[1 x 2 NPAIR] (fp32) += x[r, k0:k1] @ Wgu[k0:k1, pairs]   (PRE_L / PRE_R: DST starts from the left / right
//   neighbour's partial, CB 5 / CB 6)
//   SEND_L / SEND_R: fp32 partial -> CB 4 (to that neighbour); else silu(g) * u -> NPAIR h tiles -> CB 16 (multicast to the
//   cluster: row q of the cluster's h sequence), then down for row q - 1 (h ring CB 2, all KTD h tiles):
//   DST[1 x 4] = h[q - 1, :] @ Wd[:, 4 columns] per block -> CB 17. The last row's down runs after the last expert.
// The per-expert program (host-ordered, ws_cluster.plan) mixes gate/up pieces and DN ops; a DN op runs the down of the
// oldest h row this cluster produced and has not consumed yet (none: no-op); after the last expert every pending row
// is consumed.
// RT: 0 stamps addr, 1 NOP, then per op: k0, k1, flags (1 PRE_L, 2 PRE_R, 4 SEND_L, 8 SEND_R, 16 DN), xoff
// CT: 0 KT, 1 NPAIR, 2 XT, 3 NEXP, 4 KTD, 5 DCOLS
#ifndef DN_LAG
#define DN_LAG 0
#endif
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/expert_activation.hpp"
#include "glu_fused.hpp"
// WC_GLU: 0 stock (silu pass + multiply pass); 1.. fused one pass, glu_fused.hpp MODE = WC_GLU - 1
#ifndef WC_GLU
#define WC_GLU 0
#endif

constexpr uint32_t x_cb = tt::CBIndex::c_0;
constexpr uint32_t w_cb = tt::CBIndex::c_1;
constexpr uint32_t hr_cb = tt::CBIndex::c_2;
constexpr uint32_t wd_cb = tt::CBIndex::c_3;
constexpr uint32_t pout_cb = tt::CBIndex::c_4;
constexpr uint32_t pinl_cb = tt::CBIndex::c_5;
constexpr uint32_t pinr_cb = tt::CBIndex::c_6;
constexpr uint32_t h_cb = tt::CBIndex::c_16;
constexpr uint32_t y_cb = tt::CBIndex::c_17;

constexpr uint32_t kt = get_compile_time_arg_val(0);
constexpr uint32_t npair = get_compile_time_arg_val(1);
constexpr uint32_t xt = get_compile_time_arg_val(2);
constexpr uint32_t nexp = get_compile_time_arg_val(3);
constexpr uint32_t ktd = get_compile_time_arg_val(4);
constexpr uint32_t dcols = get_compile_time_arg_val(5);
constexpr uint32_t gw = 2 * npair;
constexpr uint32_t DB = 4;  // down block width (fp32 half-sync DST)
static_assert(gw == DB && dcols % DB == 0);
uint32_t t_gu = 0, t_dn = 0, t_pre = 0;

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

enum Mode : uint32_t { GU, DN };
Mode mode = GU;
bool h_fmt = true;  // packer set for bf16 h / y (else the fp32 partial)

// gate/up and down both run 1 x 4 blocks over bf16 in0 / bfp4 in1 tiles: the matmul configuration is the same, so
// switching costs nothing unless WC_SWITCH_INIT (the operands' CB ids are passed per call).
FORCE_INLINE void to_gu() {
#ifndef WC_SWITCH_INIT
    return;
#endif
    if (mode != GU) {
        matmul_block_init(x_cb, w_cb, false, gw, 1, kt);
        mode = GU;
    }
}
FORCE_INLINE void to_dn() {
#ifndef WC_SWITCH_INIT
    return;
#endif
    if (mode != DN) {
        matmul_block_init(hr_cb, wd_cb, false, DB, 1, ktd);
        mode = DN;
    }
}

// down for the h ring's front row: DCOLS / 4 blocks of [1 x 4] over all KTD
FORCE_INLINE void down_row() {
    const uint32_t a = wall();
    to_dn();
    cb_wait_front(hr_cb, ktd);
    cb_reserve_back(y_cb, dcols);
    for (uint32_t b = 0; b < dcols / DB; ++b) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < ktd; ++k) {
            matmul_block(hr_cb, wd_cb, k, k * dcols + b * DB, 0, false, DB, 1, ktd);
        }
        tile_regs_commit();
        tile_regs_wait();
        if (!h_fmt) {  // (after a partial: the packer is still set for fp32)
            pack_reconfig_data_format(pout_cb, y_cb);
            h_fmt = true;
        }
        for (uint32_t t = 0; t < DB; ++t) {
            pack_tile<true>(t, y_cb, b * DB + t);
        }
        tile_regs_release();
    }
    cb_pop_front(hr_cb, ktd);
    cb_push_back(y_cb, dcols);
    t_dn += wall() - a;
}

void kernel_main() {
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    const uint32_t npiece = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(x_cb, w_cb, h_cb);
    matmul_block_init(x_cb, w_cb, false, gw, 1, kt);
#if WC_GLU == 0
    expert_activation::init<expert_activation::Kind::Silu>();
#else
    ws_glu::init<WC_GLU - 1>();
#endif
    cb_wait_front(x_cb, xt);
    cb_wait_front(w_cb, kt * gw);
    cb_wait_front(wd_cb, ktd * dcols);
    UNPACK(stamps[0] = wall());
    uint32_t produced = 0, consumed = 0;
    for (uint32_t e = 0; e < nexp; ++e) {
        const uint32_t e_start = produced;  // (DN_LAG: DN ops take rows of earlier experts or >= DN_LAG productions old)
        if (e == 2) {
            t_gu = t_dn = t_pre = 0;
        }
        for (uint32_t p = 0; p < npiece; ++p) {
            const uint32_t k0 = get_arg_val<uint32_t>(2 + 4 * p);
            const uint32_t k1 = get_arg_val<uint32_t>(3 + 4 * p);
            const uint32_t flags = get_arg_val<uint32_t>(4 + 4 * p);
            const uint32_t xoff = get_arg_val<uint32_t>(5 + 4 * p);
            if (flags & 16) {
                if (consumed < produced && (!DN_LAG || consumed < e_start || consumed + DN_LAG < produced)) {
                    down_row();
                    ++consumed;
                }
                continue;
            }
            to_gu();
            uint32_t a = wall();
            tile_regs_acquire();
            if (flags & 3) {  // start from a neighbour's partial (exact: unpacked straight to fp32 DST)
                const uint32_t pin_cb = (flags & 1) ? pinl_cb : pinr_cb;
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
            const uint32_t b = wall();
            t_pre += b - a;
            uint32_t xi = xoff % xt;
            for (uint32_t k = k0; k < k1; ++k) {
                matmul_block(x_cb, w_cb, xi, k * gw, 0, false, gw, 1, kt);
                if (++xi == xt) {
                    xi = 0;
                }
            }
            tile_regs_commit();
            t_gu += wall() - b;
            if (flags & 12) {
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
#ifndef WC_NO_ACT
                PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                for (uint32_t t = 0; t < gw; t += 2) {
#if WC_GLU == 0
                    expert_activation::tile_pair<expert_activation::Kind::Silu>(t);
#else
                    ws_glu::tile_pair<WC_GLU - 1>(t);
#endif
                }
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
#endif
                for (uint32_t t = 0; t < gw; t += 2) {
                    pack_tile(t, h_cb);
                }
                tile_regs_release();
                cb_push_back(h_cb, npair);
                ++produced;
            }
        }
        PACK(stamps[1 + e] = wall());
    }
    while (consumed < produced) {
        down_row();
        ++consumed;
    }
    PACK(stamps[1 + nexp] = wall());
    MATH(stamps[40] = t_gu);
    MATH(stamps[41] = t_dn);
    MATH(stamps[42] = t_pre);
    UNPACK(stamps[43] = t_gu);
    UNPACK(stamps[44] = t_dn);
    UNPACK(stamps[45] = t_pre);
}
