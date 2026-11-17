// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert head (ws_router.py), compute. Core c of a router group of GS cores; the group takes a contiguous range of
// row tiles, core c owns its group-local row tiles i = c, c + GS, .. Everything here is TRANSPOSED as in the MoE
// block's router (moe_head router_compute.cpp): a logits tile is 32 experts x 32 tokens, a token is a column.
// Per group row tile i, in order:
//   - tilize my pieces of row tile i + D (pieces p = c, c + GS, .. of its NP = KT / BK; D 2 when that is one piece a
//     row tile, else 1: the relay and the x ring then hold every piece tilized ahead): x's row-major pieces (CB 3)
//     -> BK tiles each (CB 2), which the BRISC relays to the group's hub, which multicasts them into the group's x
//     rings;
//   - the router: my EPC expert tiles' logits^T = W_r^T x^T over all KT K tiles (x ring CB 1, BK tiles a piece;
//     W_r^T resident, CB 0, K-major: the first row tile waits for it piece by piece), fp32 in DEST 0 .. EPC - 1; then, still in DEST, the top-K keys (the fp32 logit with its
//     low mantissa bits replaced by the expert id: the id tiles in DEST 2, 3) and the bitonic network over the pair
//     (moe_head's: K-runs, merged down to one sorted run of sort_run on top) -> tile 0, my 64 experts' best sort_run
//     keys descending on top, to CB 4 (the NCRISC sends it to the row tile's owner);
//   - (owned row tile i - LAG) the owner's merge: the group's GS run tiles (CB 5) through the same network (moe_head's
//     loop over score tiles) -> the token's top K keys on top of DEST 0; then DEST 1 = their logits (ids cleared),
//     exp -> CB 7 (keys) and CB 8 (exp of the picks' logits) for the NCRISC's weights and records.
// The keys compare sign-magnitude in the network (SFPSWAP), the floats' order: clearing low mantissa bits keeps the
// order of distinct logits of either sign (ties at the key's resolution, 2^-13 relative, go by id).
// WSR_DEBUG_LOGITS: the logits also to CB 9 (the NCRISC writes them to DRAM) and back through CB 10.
// RT: 0 group row tiles, 1 c, 2 stamps (8: start, 9: end; wall-clock cycles: 10 waiting for x blocks, 11 waiting
//     for an owned row tile's runs, 12 tilizing, 13 the router matmuls (x waits included), 14 the keys and the pair's
//     network, 15 the owner's merges (run waits included)). CT: 0 KT, 1 BK, 2 EPC, 3 GS, 4 K, 5 LAG, 6 expert tiles.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"
#include "api/compute/topk.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/exp.h"

constexpr uint32_t KT = get_compile_time_arg_val(0);
constexpr uint32_t BK = get_compile_time_arg_val(1);
constexpr uint32_t EPC = get_compile_time_arg_val(2);
constexpr uint32_t GS = get_compile_time_arg_val(3);
constexpr uint32_t top_k = get_compile_time_arg_val(4);
constexpr uint32_t LAG = get_compile_time_arg_val(5);
constexpr uint32_t NET = get_compile_time_arg_val(6);
constexpr uint32_t NB = KT / BK;
static_assert(EPC == 2 && KT % BK == 0);

constexpr uint32_t w_cb = tt::CBIndex::c_0, xr_cb = tt::CBIndex::c_1, tz_cb = tt::CBIndex::c_2;
constexpr uint32_t xs_cb = tt::CBIndex::c_3, run_cb = tt::CBIndex::c_4, ga_cb = tt::CBIndex::c_5;
constexpr uint32_t id_cb = tt::CBIndex::c_6, be_cb = tt::CBIndex::c_7, ee_cb = tt::CBIndex::c_8;
constexpr uint32_t lg_cb = tt::CBIndex::c_9, lr_cb = tt::CBIndex::c_10;

// the expert id bits of a key: as many low mantissa bits as the ids need
constexpr uint32_t id_mask = NET * 32 <= 256 ? 0xFF : NET * 32 <= 512 ? 0x1FF : 0x3FF;
static_assert(NET * 32 <= 1024);
// the network's run: top_k's power of two (at least 4); the rows take its first top_k
constexpr uint32_t sort_run = top_k <= 4 ? 4 : top_k <= 8 ? 8 : top_k <= 16 ? 16 : 32;
constexpr uint32_t log2_sort_run = sort_run == 8 ? 3 : sort_run == 16 ? 4 : sort_run == 32 ? 5 : 2;
constexpr uint32_t merge_rounds = 6 - log2_sort_run;  // (log2(64 / sort_run) + 1)
constexpr auto tie_order = ckernel::TopkTieOrder::Unset;

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

uint32_t t_xw = 0, t_gw = 0, t_tz = 0, t_mm = 0, t_ks = 0, t_om = 0;

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// a logits tile's top-K keys (moe_head router_common's router_topk_key): the fp32 logit with its low mantissa bits
// (id_mask) replaced by the expert id (the uint16 id tile two DEST tiles on; the high half of a uint16 in 32-bit DEST
// is junk)
template <uint32_t mask>
inline void router_topk_key() {
    TOPK_SFPENCC_ALL_LANES_ON();
    sfpi::vConstIntPrgm0 = mask;
    for (int i = 0; i < 32; i++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 128);
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TTI_SFPOR(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}
// a keys tile's logits: the id bits cleared
template <uint32_t mask>
inline void key_logit() {
    TOPK_SFPENCC_ALL_LANES_ON();
    sfpi::vConstIntPrgm0 = mask;
    for (int i = 0; i < 32; i++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPOR(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}
}  // namespace ckernel::sfpu
#endif

// my pieces of a row tile: each its 32 rows' BK K tiles (CB 3, row-major) -> BK tiles (CB 2)
void tilize_pieces(uint32_t pieces) {
    const uint32_t a = wall();
    reconfig_data_format_srca(xs_cb);
    pack_reconfig_data_format(tz_cb);
    tilize_init(xs_cb, BK, tz_cb);
    for (uint32_t b = 0; b < pieces; ++b) {
        cb_wait_front(xs_cb, BK);
        cb_reserve_back(tz_cb, BK);
        tilize_block(xs_cb, BK, tz_cb);
        cb_push_back(tz_cb, BK);
        cb_pop_front(xs_cb, BK);
    }
    tilize_uninit(xs_cb, tz_cb);
    t_tz += wall() - a;
}

// the pair in DEST 0, 1 (keys) -> its best sort_run keys, descending, on top of tile 0 (moe_head's first pair)
FORCE_INLINE void pair_run() {
    ckernel::topk_local_sort<false, DST_ACCUM_MODE, true, false, tie_order>(0, 0, log2_sort_run - 1);
    for (uint32_t merge = 0; merge + 1 < merge_rounds; ++merge) {
        ckernel::topk_merge<false, false, DST_ACCUM_MODE, true, false, tie_order>(0, merge, sort_run);
        ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, tie_order>(
            0, 0, merge, sort_run, log2_sort_run, false);
    }
    ckernel::topk_merge<false, false, DST_ACCUM_MODE, true, false, tie_order>(0, merge_rounds - 1, sort_run);
    ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, tie_order>(
        0, 0, merge_rounds - 1, sort_run, log2_sort_run, true);
}

// a group row tile: my EPC expert tiles' logits, their keys and best run -> CB 4
void router_row_tile(bool first) {
    const uint32_t a0 = wall();
    reconfig_data_format(xr_cb, w_cb);
    // (W_r^T K-major: an expert tile's K tile k at k EPC + e - the rows of a K step one tile apart)
    matmul_block_init(w_cb, xr_cb, true /*transpose: x^T*/, 1, EPC, 1);
    tile_regs_acquire();
    for (uint32_t b = 0; b < NB; ++b) {
        if (first) {
            cb_wait_front(w_cb, (b + 1) * BK * EPC);
        }
        const uint32_t a = wall();
        cb_wait_front(xr_cb, BK);
        t_xw += wall() - a;
        for (uint32_t k = 0; k < BK; ++k) {
            matmul_block(w_cb, xr_cb, (b * BK + k) * EPC, k, 0, true, 1, EPC, 1);
        }
        cb_pop_front(xr_cb, BK);
    }
#ifdef WSR_DEBUG_LOGITS
    tile_regs_commit();
    cb_reserve_back(lg_cb, EPC);
    cb_reserve_back(lr_cb, EPC);
    tile_regs_wait();
    pack_reconfig_data_format(lg_cb);
    for (uint32_t e = 0; e < EPC; ++e) {
        pack_tile(e, lg_cb);
        pack_tile(e, lr_cb);
    }
    tile_regs_release();
    cb_push_back(lg_cb, EPC);
    cb_push_back(lr_cb, EPC);
    cb_wait_front(lr_cb, EPC);
    tile_regs_acquire();
    reconfig_data_format_srca(lr_cb);
    copy_init(lr_cb);
    for (uint32_t e = 0; e < EPC; ++e) {
        copy_tile(lr_cb, e, e);
    }
    cb_pop_front(lr_cb, EPC);
#endif
    const uint32_t a1 = wall();
    t_mm += a1 - a0;
    // the keys: the id tiles into DEST 2, 3
    reconfig_data_format_srca(id_cb);
    copy_init(id_cb);
    copy_tile(id_cb, 0, 2);
    copy_tile(id_cb, 1, 3);
    copy_dest_values_init();  // (the SFPU's address modes for the keys)
    MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::router_topk_key<id_mask>, 0, VectorMode::RC_custom)));
    MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::router_topk_key<id_mask>, 1, VectorMode::RC_custom)));
    ckernel::topk_tile_init<true, false>();
    pair_run();
    tile_regs_commit();
    cb_reserve_back(run_cb, 1);
    tile_regs_wait();
    pack_reconfig_data_format(run_cb);
    pack_tile(0, run_cb);
    tile_regs_release();
    cb_push_back(run_cb, 1);
    t_ks += wall() - a1;
}

// an owned row tile: the group's GS run tiles (CB 5) -> the top K keys (CB 7) and exp of their logits (CB 8)
void owner_merge() {
    constexpr uint32_t expert_tiles = GS;
    const uint32_t a = wall();
    cb_wait_front(ga_cb, GS);
    t_gw += wall() - a;
    tile_regs_acquire();
    if constexpr (expert_tiles == 1) {  // (one run tile: the column's other half is -inf keys)
        fill_tile_init();
        fill_tile_int<DataFormat::Int32>(0, 0xFF800000u);
    }
    reconfig_data_format_srca(ga_cb);
    copy_init(ga_cb);
    ckernel::topk_tile_init<true, false>();
    for (uint32_t tile = 0; tile < expert_tiles; tile += 2) {
        const bool alone = tile + 1 == expert_tiles;
        const bool fold = tile > 0 && !alone;
        if (alone) {
            if constexpr (expert_tiles > 1) {
                copy_dest_values<DataFormat::Int32>(2, 0);
            }
            copy_tile(ga_cb, tile, 1);
        } else {
            copy_tile(ga_cb, tile, 0);
            copy_tile(ga_cb, tile + 1, 1);
        }
        ckernel::topk_local_sort<false, DST_ACCUM_MODE, true, false, tie_order>(0, 0, log2_sort_run - 1);
        for (uint32_t merge = 0; merge + 1 < merge_rounds; ++merge) {
            ckernel::topk_merge<false, false, DST_ACCUM_MODE, true, false, tie_order>(0, merge, sort_run);
            ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, tie_order>(
                0, 0, merge, sort_run, log2_sort_run, false);
        }
        if (!fold) {
            ckernel::topk_merge<false, false, DST_ACCUM_MODE, true, false, tie_order>(0, merge_rounds - 1, sort_run);
            ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, tie_order>(
                0, 0, merge_rounds - 1, sort_run, log2_sort_run, true);
        } else {
            ckernel::topk_merge<true, false, DST_ACCUM_MODE, true, false, tie_order>(0, merge_rounds - 1, sort_run);
            ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, tie_order>(
                0, 0, merge_rounds - 1, sort_run, log2_sort_run, false);
            copy_dest_values<DataFormat::Int32>(2, 0);
            ckernel::topk_merge<false, false, DST_ACCUM_MODE, true, false, tie_order>(0, merge_rounds - 1, sort_run);
            ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, tie_order>(
                0, 0, merge_rounds - 1, sort_run, log2_sort_run, true);
        }
        if (tile + 2 < expert_tiles) {
            copy_dest_values<DataFormat::Int32>(0, 2);
        }
    }
    // DEST 1: the picks' logits (ids cleared), then exp
    copy_dest_values<DataFormat::Int32>(0, 1);
    MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::key_logit<id_mask>, 1, VectorMode::RC_custom)));
    exp_tile_init();
    exp_tile(1);
    tile_regs_commit();
    cb_reserve_back(be_cb, 1);
    cb_reserve_back(ee_cb, 1);
    tile_regs_wait();
    pack_reconfig_data_format(be_cb);
    pack_tile(0, be_cb);
    pack_tile(1, ee_cb);
    tile_regs_release();
    cb_push_back(be_cb, 1);
    cb_push_back(ee_cb, 1);
    cb_pop_front(ga_cb, GS);
    t_om += wall() - a;
}

void kernel_main() {
    const uint32_t n = get_arg_val<uint32_t>(0), c = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(2));
    compute_kernel_hw_startup<SrcOrder::Reverse>(w_cb, xr_cb, run_cb);
    cb_wait_front(id_cb, EPC);
    UNPACK(stamps[8] = wall());
    auto own = [&](uint32_t i) { return i % GS == c; };
    const uint32_t ppc = c < NB ? (NB - c + GS - 1) / GS : 0;  // (my pieces of each row tile)
    constexpr uint32_t D = (NB + GS - 1) / GS == 1 ? 2 : 1;
    for (uint32_t i = 0; i < D && i < n; ++i) {
        tilize_pieces(ppc);
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (i + D < n) {
            tilize_pieces(ppc);
        }
        router_row_tile(i == 0);
        if (i >= LAG && own(i - LAG)) {
            owner_merge();
        }
    }
    for (uint32_t j = n > LAG ? n - LAG : 0; j < n; ++j) {
        if (own(j)) {
            owner_merge();
        }
    }
    PACK(stamps[9] = wall());
    UNPACK(stamps[10] = t_xw);
    UNPACK(stamps[11] = t_gw);
    UNPACK(stamps[12] = t_tz);
    UNPACK(stamps[13] = t_mm);
    MATH(stamps[14] = t_ks);
    MATH(stamps[15] = t_om);
}
