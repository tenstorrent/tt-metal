// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert lane design with the K-split gate/up (LCfg.ks, WL_KS), group compute (TRISC): ks_compute.cpp (the
// single-chip prototype: partials of the column's 10 pairs over K tiles [KSL r, +KSL), fold of the 10 contributions of
// pair j landed in CB 4, SwiGLU, h tile j, down as before) in this chip's compute order (order entry n: expert e |
// weight slot s << 16, at CT 9), weights from K slots out of order (use_slot), each expert finished on its own: at its
// end the remaining rows are folded and their downs drained, so its slot frees without the next expert.
// RT: 0 stamps, 1 j, 2 g, 3 order length. CT: 0 KSL, 1 KTD, 2 DCOLS, 3 DB, 4 counts L1, 5 G, 6 GC, 7 FL, 8 TL,
// 9 order table L1. Stamps: 256 + n the end of the n-th expert, 63 the end; waits 40 tilize, 41 x, 42 weights, 43 h,
// 44 landing; MATH time 45 gate/up, 46 fold, 47 down, 48 tilize.
// WL_Y_RM: the down row tile is pack-untilized into CB 17 (32 rows x DCOLS * 32 columns, row after row) for the
// reader's row writes (wl_compute's scheme): DEST's untilize remap is enabled once at the start (the tilize, gate/up,
// fold and SwiGLU work with it on); a down row only switches the packer's untilize init / uninit.
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
constexpr uint32_t G = get_compile_time_arg_val(5);
constexpr uint32_t GC = get_compile_time_arg_val(6);
constexpr uint32_t FL = get_compile_time_arg_val(7);
constexpr uint32_t TL = get_compile_time_arg_val(8);
constexpr uint32_t order_l1 = get_compile_time_arg_val(9);
constexpr uint32_t HALF = GC / 2;
constexpr uint32_t KH = KSL / 2;   // x tiles of a row this core tilizes
constexpr uint32_t NW = 2 * HALF;  // partial tiles a row (10 pairs: gate, up)
constexpr uint32_t GB = 4;         // gate/up block width (1 x 4)
constexpr uint32_t WGU = KSL * NW, WDN = ktd * dcols;
constexpr uint32_t LT = 2 * HALF;  // landing tiles a row: 10 contributions x (gate, up)
static_assert(dcols % DB == 0 && NW % GB == 0 && KSL % 2 == 0);

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

uint32_t t_til = 0, t_xw = 0, t_ww = 0, t_hw = 0, t_lw = 0;
uint32_t m_gu = 0, m_fold = 0, m_dn = 0, m_til = 0;  // MATH-thread time a phase
uint32_t p_pw = 0, p_hw = 0, p_yw = 0,
         m_acq = 0;  // PACK waits for partial / h / y space, MATH waits for DEST in gate/up
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

// unpacker / math state: 0 gate/up configured, 1 down configured (srcB h instead of x), 2 needs a full matmul init
// (after a fold's eltwise ops or a tilize). One full init a row (after the fold); gate/up <-> down only switch srcB.
uint32_t mst = 2;
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

// down consumption: rows in order; the down weights of expert wd_n are popped once its last row is consumed
uint32_t consumed = 0, wd_n = 0, wd_end = 0;
FORCE_INLINE void pop_finished_down(uint32_t upto_n) {
    while (wd_n < upto_n && consumed >= wd_end) {
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
#ifdef KS_OVL
    use_slot(wd_cb, order[wd_n] >> 16, WDN);  // (the row's expert: rows of the previous one may still be pending)
#endif
    {
        const uint32_t a = wall();
        cb_wait_front(hr_cb, ktd);
        t_hw += wall() - a;
    }
    {
        const uint32_t a = wall();
        cb_wait_front(wd_cb, WDN);
        t_ww += wall() - a;
    }
    to_dn();
    pack_reconfig_data_format(p_cb, y_cb);
    PACK(const uint32_t py = wall());
    cb_reserve_back(y_cb, dcols);
    PACK(p_yw += wall() - py);
#ifdef WL_Y_RM
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
    pack_reconfig_data_format(y_cb, p_cb);
    cb_pop_front(hr_cb, ktd);
    cb_push_back(y_cb, dcols);
    ++consumed;
    MATH(m_dn += wall() - m0);
}

// fold the 10 contributions (landing tile 2 k + t of row slot `row`: contributor k, t 0 gate / 1 up) into DST d0 /
// d0 + 2 for NR rows (one init a step for the window, moe_fused_swiglu's order)
template <uint32_t T, uint32_t NR>
FORCE_INLINE void foldw(uint32_t d0) {
    uint32_t c;
    if constexpr (HALF & 1u) {
        copy_init(land_cb);
        for (uint32_t i = 0; i < NR; ++i) {
            copy_tile(land_cb, i * LT + T, d0 + 2 * i);
        }
        c = 1;
    } else {
        add_init(land_cb, land_cb, false);
        for (uint32_t i = 0; i < NR; ++i) {
            add_tiles(land_cb, land_cb, i * LT + T, i * LT + 2 + T, d0 + 2 * i);
        }
        c = 2;
    }
    add_init(land_cb, land_cb, true);
    for (; c + 1 < HALF; c += 2) {
        for (uint32_t i = 0; i < NR; ++i) {
            add_tiles(land_cb, land_cb, i * LT + 2 * c + T, i * LT + 2 * (c + 1) + T, d0 + 2 * i);
        }
    }
    add_init(land_cb, land_cb, false);
}

uint32_t folded = 0;
template <uint32_t NR>  // rows folded together (1 or 2: 2 x (gate, up) fill the fp32 DST half)
FORCE_INLINE void fold_rows() {
    const uint32_t m0 = wall();
    {
        const uint32_t a = wall();
        cb_wait_front(land_cb, NR * LT);
        t_lw += wall() - a;
    }
    reconfig_data_format(land_cb, land_cb);
    mst = 2;
    tile_regs_acquire();
    if constexpr (HALF & 1u) {
        foldw<0, NR>(0);
        foldw<1, NR>(1);
    } else {
        // gate and up together: two inits a fold (not six)
        add_init(land_cb, land_cb, false);
        for (uint32_t i = 0; i < NR; ++i) {
            for (uint32_t t = 0; t < 2; ++t) {
                add_tiles(land_cb, land_cb, i * LT + t, i * LT + 2 + t, 2 * i + t);
            }
        }
        add_init(land_cb, land_cb, true);
        for (uint32_t c = 2; c + 1 < HALF; c += 2) {
            for (uint32_t i = 0; i < NR; ++i) {
                for (uint32_t t = 0; t < 2; ++t) {
                    add_tiles(land_cb, land_cb, i * LT + 2 * c + t, i * LT + 2 * (c + 1) + t, 2 * i + t);
                }
            }
        }
    }
    tile_regs_commit();
    PACK(const uint32_t ph = wall());
    cb_reserve_back(h_cb, NR);
    PACK(p_hw += wall() - ph);
    tile_regs_wait();
    pack_reconfig_data_format(p_cb, h_cb);  // (no-op unless h's format differs from the partials')
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    for (uint32_t i = 0; i < NR; ++i) {
        ws_glu::tile_pair<2>(2 * i);
    }
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    for (uint32_t i = 0; i < NR; ++i) {
        pack_tile(2 * i, h_cb);
    }
    tile_regs_release();
    pack_reconfig_data_format(h_cb, p_cb);
    cb_push_back(h_cb, NR);
    cb_pop_front(land_cb, NR * LT);
    folded += NR;
    MATH(m_fold += wall() - m0);
}
FORCE_INLINE void fold_row() { fold_rows<1>(); }

void kernel_main() {
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    g_ = get_arg_val<uint32_t>(2);
    nexp_g = get_arg_val<uint32_t>(3);
    compute_kernel_hw_startup<SrcOrder::Reverse>(xr_cb, w_cb, p_cb);
#ifdef WL_Y_RM
    MATH((llk_math_reconfig_remap(true)));
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
    // tilize this core's x (KS_XALT: whole rows q with q % 2 == C, KSL tiles; else half of every row, KH tiles) in
    // batches: when the row about to be consumed is not tilized yet, up to TL rows past it
    const uint32_t C = get_arg_val<uint32_t>(1) / HALF;
    auto tilize_ahead = [&](uint32_t need) {
        if (til > need || til >= total_rows) {
            return;
        }
        uint32_t upto = need + 1 + TL;
        if (upto > total_rows) {
            upto = total_rows;
        }
        const uint32_t a = wall();
#if defined(KS_XDIRECT)
        // tilize straight into this core's x ring slot r % KS_XDIRECT (compute never moves its own view of CB 0's write
        // pointer, so the out-of-order pack index is the slot offset; the slot's previous row r - NX is consumed:
        // r <= xq + TL < xq + NX), then one page of CB 8 tells the data movement the row is in place
        {
            constexpr uint32_t NXD = KS_XDIRECT;
            constexpr bool fast = compute_kernel_lib::can_use_fast_tilize<KSL, xrm_cb, xr_cb>();
            reconfig_data_format_srca(xrm_cb);
            pack_reconfig_data_format(xr_cb);
            if constexpr (fast) {
                fast_tilize_init(xrm_cb, KSL, xr_cb);
            } else {
                tilize_init(xrm_cb, KSL, xr_cb);
            }
            for (uint32_t r = til; r < upto; ++r) {
                if (r % 2 != C) {
                    continue;
                }
                cb_wait_front(xrm_cb, KSL);
                if constexpr (fast) {
                    fast_tilize_block(xrm_cb, KSL, xr_cb, 0, (r % NXD) * KSL);
                } else {
                    tilize_block(xrm_cb, KSL, xr_cb, 0, (r % NXD) * KSL);
                }
                cb_pop_front(xrm_cb, KSL);
                cb_reserve_back(xst_cb, 1);
                cb_push_back(xst_cb, 1);
            }
            if constexpr (fast) {
                fast_tilize_uninit(xrm_cb, xr_cb, KSL);
            } else {
                tilize_uninit(xrm_cb, xr_cb);
            }
        }
#elif defined(KS_XALT)
        const uint32_t nb = (upto + 1 - C) / 2 - (til + 1 - C) / 2;  // own rows in [til, upto)
        if (nb) {
            compute_kernel_lib::tilize<KSL, xrm_cb, xst_cb>(xrm_dfb, xst_dfb, nb);
        }
#else
        compute_kernel_lib::tilize<KH, xrm_cb, xst_cb>(xrm_dfb, xst_dfb, upto - til);
#endif
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
    [[maybe_unused]] uint32_t prev_start = 0;  // (KS_OVL: the previous expert's first row)
    for (uint32_t n = 0; n < nexp_g; ++n) {
        if (n == 1) {
            t_til = t_xw = t_ww = t_hw = t_lw = 0;
            m_gu = m_fold = m_dn = m_til = 0;
            p_pw = p_hw = p_yw = m_acq = 0;
        }
        const uint32_t R = rows_of(n);
#ifdef KS_OVL
        // the previous expert's last folds / downs overlap this one's first rows; drained first only if this expert
        // reuses its slot (the slot cannot hold this one before that one is done), and everything older always
        {
            const uint32_t upto = (n > 0 && (order[n] >> 16) == (order[n - 1] >> 16)) ? xq : prev_start;
            while (folded < upto) {
                fold_row();
            }
            while (consumed < upto) {
                pop_finished_down(n);
                down_row();
            }
            pop_finished_down(n);
        }
        prev_start = xq;
#else
        use_slot(wd_cb, order[n] >> 16, WDN);
#endif
        use_slot(w_cb, order[n] >> 16, WGU);
        {
            const uint32_t a = wall();
            cb_wait_front(w_cb, WGU);
            t_ww += wall() - a;
        }
        const uint32_t e_start = xq;
        for (uint32_t r = 0; r < R; ++r) {
            tilize_ahead(xq);
#ifdef KS_FOLD_FIRST
            if (folded + FL < xq) {
                fold_row();
            }
#endif
            const uint32_t mg = wall();
            to_gu();
            {
                const uint32_t a = wall();
                cb_wait_front(xr_cb, KSL);
                t_xw += wall() - a;
            }
            PACK(const uint32_t pp = wall());
            cb_reserve_back(p_cb, NW);
            PACK(p_pw += wall() - pp);
            for (uint32_t b = 0; b < NW / GB; ++b) {
                MATH(const uint32_t ma = wall());
                tile_regs_acquire();
                MATH(m_acq += wall() - ma);
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
#if defined(KS_FOLD_FIRST) || defined(KS_FOLD_LAST)
#elif defined(KS_FOLD2)
            if (folded + 1 + FL < xq) {
                fold_rows<2>();
            }
#else
            if (folded + FL < xq) {
                fold_row();
            }
#endif
            down_step(n, e_start);
#ifdef KS_FOLD_LAST
            if (folded + FL < xq) {  // (after the row's down: the SwiGLU on PACK overlaps the next row's gate/up)
                fold_row();
            }
#endif
        }
        cb_pop_front(w_cb, WGU);  // this expert's gate/up weights are done
#ifdef KS_OVL
        if (n + 1 < nexp_g) {
            if (n < 200) {
                PACK(stamps[256 + n] = wall());
            }
            continue;
        }
#endif
        // finish the expert on its own (its slot frees without the next expert): fold its remaining rows, drain downs
        pop_finished_down(n);
        while (folded < xq) {
            fold_row();
        }
        while (consumed < folded) {
            pop_finished_down(n);  // (KS_OVL: the drain can span the previous expert's rows too - its own down weights)
            down_row();
        }
        pop_finished_down(n + 1);
        if (n < 200) {
            PACK(stamps[256 + n] = wall());
        }
    }
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
    PACK(stamps[64] = p_pw);
    PACK(stamps[65] = p_hw);
    PACK(stamps[66] = p_yw);
    MATH(stamps[67] = m_acq);
}
