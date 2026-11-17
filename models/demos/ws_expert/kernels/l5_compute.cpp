// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert lane design with 5 x 5 K-split groups compute (ws_lane5.py): g5_compute.cpp in the chip's compute order
// (order entry n: expert | weight slot s << 16, table L1 + 64, loaded by the BRISC), the weights read from slot s of
// the K-slot rings (use_slot), and every expert finished on its own (its remaining folds and downs drained before the
// next expert's weights are waited for) so that its slot frees without the next expert (the lane schedule's rule).
// Stamps: 256 + n the end of the n-th expert, 0 start, 63 end.
// Base: ws_expert 5 x 5 K-split groups compute (ws_g5.py; ks_compute.cpp generalized). Core (c, r), j = 5 c + r:
// gate/up partials of its column's 4 pairs over its K slice (KSL tiles; 2 blocks of 1 x 4 -> 8 partial tiles a row, CB
// 18); owners (rows 0-3, OWN) fold the 5 contributions of their pair (CB 4) in fp32 DEST, apply SwiGLU and produce the
// h tile (CB 16); every core runs the down of its DCOLS output columns for each h row (CB 2, DN_LAG rows behind). Row 4
// (OWN 0) has no fold: its rows count as folded FL rows after their gate/up (the down's lag rule). x rows q with q % 5
// == c are tilized here (the BRISC gathers them; CB 7 -> CB 8), up to TL rows ahead. RT: 0 stamps, 1 j, 2 g, 3 order
// length. CT: 0 KSL, 1 KTD, 2 DCOLS, 3 DB, 4 counts L1, 5 K (slots), 6 NW, 7 FL, 8 TL, 9 OWN, 10 order table L1
// WL_Y_RM: the down row tile is pack-untilized into CB 17 (32 rows x DCOLS * 32 columns, row after row) for the
// reader's row writes (kl_compute.cpp's scheme; DEST's untilize remap enabled once at the start, or with
// WL5_REMAP_ROW only around each down row: the gate/up, fold and tilize run without it).
#ifndef DN_LAG
#define DN_LAG 1
#endif
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/expert_activation.hpp"
#include "glu_fused.hpp"
#ifdef WL_Y_RM
#include "api/compute/pack_untilize.h"
#endif

constexpr uint32_t xr_cb = tt::CBIndex::c_0;
constexpr uint32_t w_cb = tt::CBIndex::c_1;
constexpr uint32_t hr_cb = tt::CBIndex::c_2;
constexpr uint32_t wd_cb = tt::CBIndex::c_3;
constexpr uint32_t land_cb = tt::CBIndex::c_4;
constexpr uint32_t xrm_cb = tt::CBIndex::c_7;
constexpr uint32_t xst_cb = tt::CBIndex::c_8;
constexpr uint32_t h_cb = tt::CBIndex::c_16;
constexpr uint32_t y_cb = tt::CBIndex::c_17;
constexpr uint32_t p_cb = tt::CBIndex::c_18;

constexpr uint32_t KSL = get_compile_time_arg_val(0);
constexpr uint32_t ktd = get_compile_time_arg_val(1);
constexpr uint32_t dcols = get_compile_time_arg_val(2);
constexpr uint32_t DB = get_compile_time_arg_val(3);
constexpr uint32_t counts_l1 = get_compile_time_arg_val(4);
constexpr uint32_t KSLOTS = get_compile_time_arg_val(5);
constexpr uint32_t NW = get_compile_time_arg_val(6);
constexpr uint32_t FL = get_compile_time_arg_val(7);
constexpr uint32_t TL = get_compile_time_arg_val(8);
constexpr bool OWN = get_compile_time_arg_val(9) != 0;
constexpr uint32_t order_l1 = get_compile_time_arg_val(10);
constexpr uint32_t NCON = 5;       // contributions to a pair (the column's rows)
constexpr uint32_t NCOLS = 5;      // columns: x rows are dealt q % 5
constexpr uint32_t GB = 4;         // gate/up block width (1 x 4)
constexpr uint32_t LT = 2 * NCON;  // landing tiles a row: 5 contributions x (gate, up)
constexpr uint32_t WGU = KSL * NW, WDN = ktd * dcols;
static_assert(dcols % DB == 0 && NW % GB == 0);

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

uint32_t t_til = 0, t_xw = 0, t_ww = 0, t_hw = 0, t_lw = 0;
uint32_t m_gu = 0, m_fold = 0, m_dn = 0, m_til = 0;
volatile uint32_t* counts = reinterpret_cast<volatile uint32_t*>(counts_l1);
uint32_t g_, nexp_g;
volatile uint32_t* order = reinterpret_cast<volatile uint32_t*>(order_l1);
FORCE_INLINE uint32_t rows_of(uint32_t n) { return (counts[16 + (order[n] & 0xFFFF)] + 31) / 32; }
// point a weight CB's read side at slot s (CB of K slots x `tiles` tiles; read pointers in 16-byte units)
FORCE_INLINE void use_slot(uint32_t cb, uint32_t s, uint32_t tiles) {
    UNPACK({
        auto& c = get_local_cb_interface(cb);
        c.fifo_rd_ptr = c.fifo_limit - c.fifo_size + s * tiles * c.fifo_page_size;
    });
}

uint32_t mst = 2;  // 0 gate/up configured, 1 down, 2 needs a full init
FORCE_INLINE void to_gu() {
    if (mst == 2) {
        reconfig_data_format(w_cb, xr_cb);
        matmul_block_init(xr_cb, w_cb, false, GB, 1, KSL);
    } else if (mst == 1) {
        reconfig_data_format_srcb(hr_cb, xr_cb);
    }
    mst = 0;
}
FORCE_INLINE void to_dn() {
    if (mst == 2) {
        reconfig_data_format(wd_cb, hr_cb);
        matmul_block_init(hr_cb, wd_cb, false, DB, 1, ktd);
    } else if (mst == 0) {
        reconfig_data_format_srcb(xr_cb, hr_cb);
    }
    mst = 1;
}

uint32_t consumed = 0, wd_n = 0, wd_end = 0;
FORCE_INLINE void pop_finished_down(uint32_t upto_n) {
    while (wd_n < upto_n && consumed >= wd_end) {
        use_slot(wd_cb, order[wd_n] >> 16, WDN);
        cb_wait_front(wd_cb, WDN);
        cb_pop_front(wd_cb, WDN);
        ++wd_n;
        if (wd_n < nexp_g) {
            wd_end += rows_of(wd_n);
        }
    }
}

FORCE_INLINE void down_row() {
    const uint32_t m0 = wall();
    {
        const uint32_t a = wall();
        cb_wait_front(hr_cb, ktd);
        t_hw += wall() - a;
    }
    {
        const uint32_t a = wall();
        use_slot(wd_cb, order[wd_n] >> 16, WDN);
        cb_wait_front(wd_cb, WDN);
        t_ww += wall() - a;
    }
    to_dn();
    pack_reconfig_data_format(p_cb, y_cb);
    cb_reserve_back(y_cb, dcols);
#if defined(WL_Y_RM) && defined(WL5_REMAP_ROW)
    // (the remap on for this row's down only: math writes DEST in the untilize order; off again after the commits)
    pack_untilize_dest_init<DB, dcols, false, TILE_C_DIM, false, true /*configure_remap*/>(y_cb);
#elif defined(WL_Y_RM)
    pack_untilize_dest_init<DB, dcols, false, TILE_C_DIM, false, false /*configure_remap*/>(y_cb);
#endif
    for (uint32_t b = 0; b < dcols / DB; ++b) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < ktd; ++k) {
            matmul_block(hr_cb, wd_cb, k, k * dcols + b * DB, 0, false, DB, 1, ktd);
        }
        tile_regs_commit();
        tile_regs_wait();
#ifdef WL_Y_RM
        pack_untilize_dest<DB, dcols>(y_cb, 1, b);
#else
        for (uint32_t t = 0; t < DB; ++t) {
            pack_tile<true>(t, y_cb, b * DB + t);
        }
#endif
        tile_regs_release();
    }
#ifdef WL_Y_RM
    pack_untilize_uninit(y_cb);
#endif
#if defined(WL_Y_RM) && defined(WL5_REMAP_ROW)
    MATH((llk_math_reconfig_remap(false)));
#endif
    pack_reconfig_data_format(y_cb, p_cb);
    cb_pop_front(hr_cb, ktd);
    cb_push_back(y_cb, dcols);
    ++consumed;
    MATH(m_dn += wall() - m0);
}

// fold the NCON contributions (landing tile 2 k + T: contributor k, T 0 gate / 1 up) into DST d0
template <uint32_t T>
FORCE_INLINE void foldw(uint32_t d0) {
    uint32_t c;
    if constexpr (NCON & 1u) {
        copy_init(land_cb);
        copy_tile(land_cb, T, d0);
        c = 1;
    } else {
        add_init(land_cb, land_cb, false);
        add_tiles(land_cb, land_cb, T, 2 + T, d0);
        c = 2;
    }
    add_init(land_cb, land_cb, true);
    for (; c + 1 < NCON + 1 && c < NCON; c += 2) {
        add_tiles(land_cb, land_cb, 2 * c + T, 2 * (c + 1) + T, d0);
    }
    add_init(land_cb, land_cb, false);
}
static_assert(NCON % 2 == 1, "fold pairs after the first copy");

uint32_t folded = 0;
FORCE_INLINE void fold_row() {
    if constexpr (!OWN) {
        ++folded;
        return;
    }
    const uint32_t m0 = wall();
    {
        const uint32_t a = wall();
        cb_wait_front(land_cb, LT);
        t_lw += wall() - a;
    }
    reconfig_data_format(land_cb, land_cb);
    mst = 2;
    tile_regs_acquire();
    foldw<0>(0);
    foldw<1>(1);
    tile_regs_commit();
    cb_reserve_back(h_cb, 1);
    tile_regs_wait();
    pack_reconfig_data_format(p_cb, h_cb);
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    ws_glu::tile_pair<2>(0);
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    pack_tile(0, h_cb);
    tile_regs_release();
    pack_reconfig_data_format(h_cb, p_cb);
    cb_push_back(h_cb, 1);
    cb_pop_front(land_cb, LT);
    ++folded;
    MATH(m_fold += wall() - m0);
}

void kernel_main() {
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    g_ = get_arg_val<uint32_t>(2);
    nexp_g = get_arg_val<uint32_t>(3);
    compute_kernel_hw_startup<SrcOrder::Reverse>(xr_cb, w_cb, p_cb);
#if defined(WL_Y_RM) && !defined(WL5_REMAP_ROW)
    MATH((llk_math_reconfig_remap(true)));  // (DEST's untilize remap on once: the down rows pack-untilize)
#endif
    ws_glu::init<2>();
    while (counts[0] == 0) {
    }
    UNPACK(stamps[0] = wall());
    uint32_t total_rows = 0;
    for (uint32_t n = 0; n < nexp_g; ++n) {
        total_rows += rows_of(n);
    }
    DataflowBuffer xrm_dfb(xrm_cb), xst_dfb(xst_cb);
    uint32_t til = 0, xq = 0;
    const uint32_t C = get_arg_val<uint32_t>(1) / NCON;
    auto own_before = [&](uint32_t n) { return (n + NCOLS - 1 - C) / NCOLS; };  // own rows q < n (q % 5 == C)
    // continuous look-ahead: every own row q <= need + TL is tilized (rows are dealt 1 in 5, so a batch-on-demand
    // window of TL rows often holds no own row and the row a peer needs next is tilized only when it is due)
    auto tilize_ahead = [&](uint32_t need) {
        uint32_t upto = need + 1 + TL;
        if (upto > total_rows) {
            upto = total_rows;
        }
        if (til >= upto || own_before(upto) == own_before(til)) {
            return;
        }
        const uint32_t a = wall();
        const uint32_t nb = own_before(upto) - own_before(til);
        if (nb) {
            compute_kernel_lib::tilize<KSL, xrm_cb, xst_cb>(xrm_dfb, xst_dfb, nb);
        }
        pack_reconfig_data_format(p_cb);
        mst = 2;
        t_til += wall() - a;
        til = upto;
        MATH(m_til += wall() - a);
    };
    auto down_step = [&](uint32_t n, uint32_t e_start) {
        if (consumed < folded && (consumed < e_start || consumed + DN_LAG < folded)) {
            pop_finished_down(n);
            down_row();
            pop_finished_down(n);
        }
    };
    wd_end = nexp_g ? rows_of(0) : 0;
    for (uint32_t n = 0; n < nexp_g; ++n) {
        if (n == 1) {
            t_til = t_xw = t_ww = t_hw = t_lw = 0;
            m_gu = m_fold = m_dn = m_til = 0;
        }
        const uint32_t R = rows_of(n);
        {
            const uint32_t a = wall();
            use_slot(w_cb, order[n] >> 16, WGU);
            cb_wait_front(w_cb, WGU);
            t_ww += wall() - a;
        }
        const uint32_t e_start = xq;
        for (uint32_t r = 0; r < R; ++r) {
            tilize_ahead(xq);
            const uint32_t mg = wall();
            to_gu();
            {
                const uint32_t a = wall();
                cb_wait_front(xr_cb, KSL);
                t_xw += wall() - a;
            }
            cb_reserve_back(p_cb, NW);
            for (uint32_t b = 0; b < NW / GB; ++b) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < KSL; ++k) {
                    matmul_block(xr_cb, w_cb, k, k * NW + b * GB, 0, false, GB, 1, KSL);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < GB; ++t) {
                    pack_tile<true>(t, p_cb, b * GB + t);
                }
                tile_regs_release();
            }
            cb_push_back(p_cb, NW);
            cb_pop_front(xr_cb, KSL);
            MATH(m_gu += wall() - mg);
            ++xq;
            if (folded + FL < xq) {
                fold_row();
            }
            down_step(n, e_start);
        }
        cb_pop_front(w_cb, WGU);
        // finish this expert on its own: its slot frees without the next expert's weights
        while (folded < xq) {
            fold_row();
        }
        while (consumed < xq) {
            pop_finished_down(n + 1);
            down_row();
            pop_finished_down(n + 1);
        }
        pop_finished_down(n + 1);
        PACK(stamps[256 + n] = wall());
    }
    while (folded < total_rows) {
        fold_row();
        down_step(nexp_g, 0);
    }
    while (consumed < folded) {
        pop_finished_down(nexp_g);
        down_row();
        pop_finished_down(nexp_g);
    }
    pop_finished_down(nexp_g);
    PACK(stamps[63] = wall());
    UNPACK(stamps[40] = t_til);
    UNPACK(stamps[41] = t_xw);
    UNPACK(stamps[42] = t_ww);
    UNPACK(stamps[43] = t_hw);
    UNPACK(stamps[44] = t_lw);
    MATH(stamps[45] = m_gu);
    MATH(stamps[46] = m_fold);
    MATH(stamps[47] = m_dn);
    MATH(stamps[48] = m_til);
}
