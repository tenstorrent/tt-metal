// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MLA megakernel attention phase, sequence-sharded DSA + split-K (kernels/mla/dsa_shard.hpp) — NCRISC (NoC1) of owner
// core (g, c) on chip `rank` of C. Four sweeps over the launch's T tokens:
//   A  local select: the keys of this core's LOCAL pools (< n_closed_local), the radix select rounds + the tie round as
//      att_dsa_dm1.cpp (the coordinator's tie message now carries (quota, record offset) per core); the selected
//      (key, local pool) pairs are kept per token; the group-0 cores write their records (key, core) at their offset
//      into token t's candidate chunk of x_ag (DRAM) and increment the all-gather lane's wait semaphore;
//   B  global filter: per token the coordinator's 10th message (T_g, k_rem) + per-core tie quota -> this core's final
//      selection = local keys > T_g, plus the first quota keys == T_g -> CB_ASEL page t (n_sel, tail rows on core 0 of
//      the open pool's OWNER chip, the tail's local pool, the ids);
//   C  pass 0 (the peer's heads): the group's partial tree as att_dm1.cpp; the root's final partial lands in the tree's
//      SEND CBs (the compute's last combine, as on a sending core) and the root writes those fp32 tiles (m, l; the 16
//      O tiles, otb bytes each) into the peer's block of x_rs, then increments the reduce-scatter lane's wait
//      semaphore;
//   D  pass 1 (this chip's heads): the tree as before; then the root waits for the reduce-scatter (RS_SEM: the lane's
//      waiter signals the owners) and lands the peer's partial of THIS chip's heads (y_rs) in the tree's RECEIVE CBs
//      (CB_ARX / CB_ARXO, after its children's messages: one more tree level for the compute), then writes o_lat.
// TP8 (MLA_DSA_SHARD_LINE = 4, kernels/mla/dsa_shard.hpp): C / D's root exchange runs in two rounds. Round 1 (axis 0)
// as above: pass 0's partial (the other row's head tile) -> the column peer; pass 1's root lands the column peer's
// partial of this row's tile and the compute's combine writes the COLUMN-COMBINED partial P1 into the send CBs; the
// root writes the 8-row slice of every column q of P1 into x_rs2 block q (its own column too) and increments the RS2
// lane's wait semaphore. E (round 2, after the scatter's waiter signalled the owners a second time): per token the root
// lands its own slice (x_rs2) and the 3 row peers' slices of its heads (y_rs2, column order) into rows 8 c .. of the
// receive CBs (the other rows keep the previous message: finite values), the compute takes / combines them and
// finalises, and the root writes rows 8 c .. 8 c + 7 of the o tiles to o_lat rows 0..7 (this chip's 8 heads). Rev 60
// (W, T > 4): token lanes (att_common.hpp LANES; c / C / LV lane-local, the coordinator = the lane's index core):
//   every sweep runs the lane's tokens t0, t0 + LANES, ..; the mailbox slots, the kept pairs and CB_ASEL pages are
//   indexed by the lane-local token j = t / LANES (< TL), the exchange blocks (x_ag / y_ag / x_rs / y_rs / x_rs2 /
//   y_rs2 / o_lat) by the launch token t; the lane root ships / lands its tokens' partials. o_T (MLA_ATT_OT = HR): the
//   root also writes the finalised rows of this chip's HR heads into W_uv's token-row in0 (att_dsa_dm1.cpp's rev-39
//   form; TP8: rows 8 lpos .. of the finalised tile).
// Compile-time args: att_dsa_dm1.cpp's (o_lat, cur_pos [, o_T]), then RS_SEM before the accessors, then the
//   TensorAccessorArgs of x_ag, x_rs, y_rs, y_ag [, TP8: x_rs2, y_rs2].
// Runtime args: att_dsa_dm1.cpp's (o_lat, g, c, T, 2 LV tree, 2 NCONS consumers, cur_pos, coordinator x, y, histogram,
//   mailbox [, the lane] [, the o_T address]), then rank, C, peer block, x_ag, x_rs, y_rs addresses, the AG lane core
//   (x, y) + its wait semaphore id, the RS lane core (x, y) + its wait semaphore id, the y_ag address (rev 59: the
//   merge helpers) [, TP8: x_rs2, y_rs2 addresses, the RS2 lane core (x, y) + its wait semaphore id].

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/dsa_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/dsa_shard.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/mla_common.hpp"

void kernel_main() {
    constexpr uint32_t CB_ARX = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t CB_ASEND = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t CB_AOUT = get_compile_time_arg_val(PH_CT0 + 2);
    constexpr uint32_t C = get_compile_time_arg_val(PH_CT0 + 3);
    constexpr uint32_t LV = get_compile_time_arg_val(PH_CT0 + 4);
    constexpr uint32_t HPT = get_compile_time_arg_val(PH_CT0 + 5);
    constexpr uint32_t SEM_RDY = get_compile_time_arg_val(PH_CT0 + 6);
    constexpr uint32_t SEM_DAT = get_compile_time_arg_val(PH_CT0 + 7);
    constexpr uint32_t SIGNAL_SEM = get_compile_time_arg_val(PH_CT0 + 8);
    constexpr uint32_t NCONS = get_compile_time_arg_val(PH_CT0 + 9);
    constexpr uint32_t CB_ARXO = get_compile_time_arg_val(PH_CT0 + 10);
    constexpr uint32_t CB_ASENDO = get_compile_time_arg_val(PH_CT0 + 11);
    constexpr uint32_t SEM_DATO = get_compile_time_arg_val(PH_CT0 + 12);
    constexpr uint32_t CB_ASC = get_compile_time_arg_val(PH_CT0 + 13);
    constexpr uint32_t CB_ASEL = get_compile_time_arg_val(PH_CT0 + 14);
    constexpr uint32_t CB_AKEY = get_compile_time_arg_val(PH_CT0 + 15);
    constexpr uint32_t SEL_SEM = get_compile_time_arg_val(PH_CT0 + 16);
    constexpr uint32_t H_SEM = get_compile_time_arg_val(PH_CT0 + 17);
    constexpr uint32_t RS_SEM = get_compile_time_arg_val(PH_CT0 + 18);
    constexpr auto o_args = TensorAccessorArgs<PH_CT0 + 19>();
    constexpr auto c_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
#if defined(MLA_ATT_OT)  // rev 60: o_T (W_uv's token-row in0) after cur_pos's args, as att_dsa_dm1.cpp
    constexpr auto ot_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto xag_args = TensorAccessorArgs<ot_args.next_compile_time_args_offset()>();
    constexpr uint32_t HR = MLA_ATT_OT;
    static_assert(HR >= 1 && HR <= 32 && (mlashard::LINE == 1 || HR == mlashard::PIECE_ROWS), "o_T: this chip's heads");
#else
    constexpr auto xag_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
#endif
    constexpr auto xrs_args = TensorAccessorArgs<xag_args.next_compile_time_args_offset()>();
    constexpr auto yrs_args = TensorAccessorArgs<xrs_args.next_compile_time_args_offset()>();
    constexpr auto yag_args = TensorAccessorArgs<yrs_args.next_compile_time_args_offset()>();  // rev 59: the helpers
#if defined(MLA_DSA_SHARD_LINE)
    constexpr auto xrs2_args = TensorAccessorArgs<yag_args.next_compile_time_args_offset()>();
    constexpr auto yrs2_args = TensorAccessorArgs<xrs2_args.next_compile_time_args_offset()>();
#endif
    constexpr uint32_t ST = mlaatt::STAT_TILES, DT = mlaatt::DT;
    static_assert(C <= mladsa::MAX_CHUNKS, "chunks");
    static_assert(mladsa::T_MAX * 4 <= mladsa::POS_BYTES, "cur_pos scratch");
    static_assert(mladsa::QUOTA_W == 2, "the shard tie message carries (quota, offset) per core");
    static_assert(HPT == 1, "split-K: one head group per chip (TP2: 32 heads; TP8: 8 heads, the 2-round exchange)");
    const auto olat = TensorAccessor(o_args, get_arg_val<uint32_t>(ph_rt0 + 0));
    const uint32_t g = get_arg_val<uint32_t>(ph_rt0 + 1);
    const uint32_t c = get_arg_val<uint32_t>(ph_rt0 + 2);
    const uint32_t t_rows = get_arg_val<uint32_t>(ph_rt0 + 3);
    const uint32_t lv0 = ph_rt0 + 4;
    const uint32_t d0 = lv0 + 2 * LV + 2 * NCONS;
    const auto cur = TensorAccessor(c_args, get_arg_val<uint32_t>(d0 + 0));
    const uint32_t cx = get_arg_val<uint32_t>(d0 + 1), cy = get_arg_val<uint32_t>(d0 + 2);
    const uint32_t hist_addr = get_arg_val<uint32_t>(d0 + 3);
    const uint32_t mbox = get_arg_val<uint32_t>(d0 + 4);
#if defined(MLA_ATT_LANES)  // rev 60: the token lane (its coordinator = the lane's index core)
    const uint32_t t0 = get_arg_val<uint32_t>(d0 + 5);
    constexpr uint32_t LANE_ARGS = 1;
#else
    const uint32_t t0 = 0;
    constexpr uint32_t LANE_ARGS = 0;
#endif
#if defined(MLA_ATT_OT)
    const auto ot = TensorAccessor(ot_args, get_arg_val<uint32_t>(d0 + 5 + LANE_ARGS));
    constexpr uint32_t OT_ARGS = 1;
#else
    constexpr uint32_t OT_ARGS = 0;
#endif
    const uint32_t sh = d0 + 5 + LANE_ARGS + OT_ARGS;  // the shard block
    const uint32_t rank = get_arg_val<uint32_t>(sh + 0);
    const uint32_t nchips = get_arg_val<uint32_t>(sh + 1);
    const uint32_t peer_blk = get_arg_val<uint32_t>(sh + 2);
    const auto xag = TensorAccessor(xag_args, get_arg_val<uint32_t>(sh + 3));
    const auto xrs = TensorAccessor(xrs_args, get_arg_val<uint32_t>(sh + 4));
    const auto yrs = TensorAccessor(yrs_args, get_arg_val<uint32_t>(sh + 5));
    const uint32_t ag_x = get_arg_val<uint32_t>(sh + 6), ag_y = get_arg_val<uint32_t>(sh + 7);
    const uint32_t ag_sem = get_arg_val<uint32_t>(sh + 8);
    const uint32_t rs_x = get_arg_val<uint32_t>(sh + 9), rs_y = get_arg_val<uint32_t>(sh + 10);
    const uint32_t rs_sem = get_arg_val<uint32_t>(sh + 11);
    const auto xyag = TensorAccessor(yag_args, get_arg_val<uint32_t>(sh + 12));  // y_ag (the merge helpers' lists)
    ASSERT(t_rows <= mlashard::T_MAX && nchips == mlashard::NCHIPS);
    ASSERT(mlashard::lane_tokens(t_rows, t0) <= mlashard::TL);
#if defined(MLA_DSA_SHARD_LINE)
    const auto xrs2 = TensorAccessor(xrs2_args, get_arg_val<uint32_t>(sh + 13));
    const auto yrs2 = TensorAccessor(yrs2_args, get_arg_val<uint32_t>(sh + 14));
    const uint32_t rs2_x = get_arg_val<uint32_t>(sh + 15), rs2_y = get_arg_val<uint32_t>(sh + 16);
    const uint32_t rs2_sem = get_arg_val<uint32_t>(sh + 17);
    const uint32_t lpos = mlashard::line_pos(rank);  // this chip's column = its slice index in the row's head tile
#endif
    const uint32_t arx_base = get_local_cb_interface(CB_ARX).fifo_limit - get_local_cb_interface(CB_ARX).fifo_size;
    const uint32_t arxo_base = get_local_cb_interface(CB_ARXO).fifo_limit - get_local_cb_interface(CB_ARXO).fifo_size;
    const uint32_t ptb = get_tile_size(CB_ARX);
    const uint32_t otb = get_tile_size(CB_ARXO);
    constexpr uint32_t PG = mlashard::TB;  // one page of x_rs / y_rs (bf16 tile bytes)
    constexpr uint32_t OG = 4;
    static_assert(mlaatt::DT % OG == 0, "O groups");
    volatile tt_l1_ptr uint32_t* dato = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DATO));
    // CB_AKEY (scratch, never pushed): [0, 64) histogram / tie count, [64, 64 + POS_BYTES) cur_pos, then the region U
    // of UB bytes = the phase-A keys (MLA_DSA_KEYS words) + compacted candidates (MLA_DSA_KEYS words), overlaid in
    // phase B by the merge helpers' scratch (rev 60, review 2 item 1: dead keys by then), then per lane token j the
    // kept (key, local pool) pairs (2 KW words each, TL blocks: a core keeps <= min(CAND_MAX, its MLA_DSA_KEYS) pools
    // of a token) and a 64-B-aligned record staging area (TB + 64 B) (tt/mk_mla.py _akey_bytes: 64 + POS_BYTES + UB +
    // TL 8 KW + 64
    // + TB + 64 + 64)
    constexpr uint32_t KW = MLA_DSA_KEYS < mlashard::CAND_MAX ? MLA_DSA_KEYS : mlashard::CAND_MAX;
    constexpr uint32_t UB =
        2 * MLA_DSA_KEYS * 4 > mlashard::MERGE_SCR_BYTES + 64 ? 2 * MLA_DSA_KEYS * 4 : mlashard::MERGE_SCR_BYTES + 64;
    const uint32_t kbase = get_write_ptr(CB_AKEY);
    volatile tt_l1_ptr uint32_t* hist = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kbase);
    volatile tt_l1_ptr int32_t* pos_l1 = reinterpret_cast<volatile tt_l1_ptr int32_t*>(kbase + 64);
    const uint32_t ubase = kbase + 64 + mladsa::POS_BYTES;
    volatile tt_l1_ptr uint32_t* keys = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ubase);
    volatile tt_l1_ptr uint32_t* cand = keys + MLA_DSA_KEYS;
    volatile tt_l1_ptr uint32_t* kept = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ubase + UB);  // j: + j 2 KW
    const uint32_t stg = (reinterpret_cast<uint32_t>(kept + mlashard::TL * 2 * KW) + 63) & ~63u;
    // the merge helpers' scratch (rev 59; rev 60: over the dead keys of region U): this chip q = c's candidate pages,
    // its keys, the compacted candidates, the per-core tie quotas (dsa_shard.hpp MERGE_SCR_BYTES)
    const uint32_t mstg = (ubase + 63) & ~63u;
    static_assert(mlashard::MERGE_SCR_BYTES + 63 <= UB, "the merge scratch inside region U");
    volatile tt_l1_ptr uint32_t* mkeys =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mstg + mlashard::CAND_PAGES * mlashard::TB);
    volatile tt_l1_ptr uint32_t* mcand = mkeys + mlashard::CAND_MAX;
    volatile tt_l1_ptr uint32_t* mgq = mcand + mlashard::CAND_MAX;
    const bool helper = g == 0 && c < mlashard::NCHIPS;  // merge helper c holds chip c's list (lane-local c: per lane)
    // (review 2 item 7) helper `rank` writes its MAX_CHUNKS per-core quota words into the coordinator's C histogram
    // slots
    static_assert(C * mladsa::HIST_SLOT_BYTES >= mladsa::MAX_CHUNKS * 4, "the quota report inside the histogram slots");
    static_assert(mlashard::NCHIPS <= C, "one merge helper per chip among the lane's chunk cores");
    uint32_t n_kept[mlashard::TL];
    volatile tt_l1_ptr uint32_t* sel_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEL_SEM));
    const uint64_t hist_dst = get_noc_addr(cx, cy, hist_addr + c * mladsa::HIST_SLOT_BYTES);
    const uint64_t h_sem = get_noc_addr(cx, cy, get_semaphore(H_SEM));
    auto report = [&]() {
        invalidate_l1_cache();
        noc_async_write(kbase, hist_dst, mladsa::HIST_SLOT_BYTES);
        noc_async_write_barrier();
        noc_semaphore_inc(h_sem, 1);
        noc_async_atomic_barrier();
    };
    uint32_t msgs = 0;

    MK_MARK();  // probe 0: the attention phase starts (after mhc_pre)
    noc_async_read_page(0, cur, kbase + 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    auto pos_of = [&](uint32_t t) {
        const int32_t pos = pos_l1[t];
        return pos < 0 ? 0u : static_cast<uint32_t>(pos);
    };
    // THE DENSE WINDOW (MLA_DSA_DENSE_WINDOW, att_dsa_shard_dm0.cpp: the same decision on every core): an all-dense
    // launch runs D's tree over the window FIRST (no pass 0: the root only keeps the RS lane's per-token count) and B
    // (the merge messages -> CB_ASEL, popped by the reader at its end) after it, so the window pass never waits for the
    // AG / merge
#if defined(MLA_DSA_DENSE_WINDOW)
    bool all_dense = true;
    for (uint32_t t = 0; t < t_rows; ++t) {
        all_dense = all_dense && !mladsa::sparse(pos_of(t));
    }
#else
    constexpr bool all_dense = false;
#endif
    // ---- A: the local select of every lane token + the candidate records
    for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
        const uint32_t p = pos_of(t);
        const uint32_t j = mlashard::lane_tok(t);  // the mailbox slots / kept block of token t
        n_kept[j] = 0;
        uint32_t off = 0;
        if (mladsa::sparse(p)) {
            const uint32_t n_closed = mlashard::n_closed_local(p, rank, nchips);
            const uint32_t npt = mladsa::pool_tiles(n_closed, c, C);
            const uint32_t frow = (t / 16) * 512 + (t % 16) * 16;
            uint32_t nvalid = 0;
            for (uint32_t i = 0; i < npt; ++i) {
                const uint32_t pt0 = (c + i * C) * 32;
                const uint32_t ncol = n_closed - pt0 < 32 ? n_closed - pt0 : 32;
                cb_wait_front(CB_ASC, 1);
                invalidate_l1_cache();
                volatile tt_l1_ptr uint32_t* tile =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_ASC));
                for (uint32_t col = 0; col < ncol; ++col) {
                    keys[nvalid++] = mladsa::key_of(tile[frow + (col < 16 ? col : 256 + col - 16)]);
                }
                cb_pop_front(CB_ASC, 1);
            }
            if (npt % mladsa::SB != 0) {
                cb_wait_front(CB_ASC, mladsa::SB - npt % mladsa::SB);
                cb_pop_front(CB_ASC, mladsa::SB - npt % mladsa::SB);
            }
            // ---- I5: radix rounds on the compacted candidates (att_dsa_dm1.cpp rev 36)
            uint32_t prefix = 0, mask = 0;
            volatile tt_l1_ptr uint32_t* src = keys;
            uint32_t nsrc = nvalid;
            for (uint32_t r = 0; r < mladsa::ROUNDS; ++r) {
                const uint32_t shift = 32 - mladsa::DIGIT_BITS * (r + 1);
                if (g == 0) {
                    for (uint32_t b = 0; b < mladsa::BINS; ++b) {
                        hist[b] = 0;
                    }
                    for (uint32_t i = 0; i < nsrc; ++i) {
                        const uint32_t k = src[i];
                        if ((k & mask) == prefix) {
                            hist[(k >> shift) & (mladsa::BINS - 1)] += 1;
                        }
                    }
                    report();
                }
                noc_semaphore_wait_min(sel_sem, ++msgs);
                invalidate_l1_cache();
                prefix = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    mbox + (j * mladsa::MSGS + r) * mladsa::MSG_BYTES)[0];
                mask |= (mladsa::BINS - 1) << shift;
                if (g == 0) {
                    uint32_t m = 0;
                    for (uint32_t i = 0; i < nsrc; ++i) {
                        const uint32_t k = src[i];
                        if ((k & mask) == prefix) {
                            cand[m++] = k;
                        }
                    }
                    src = cand;
                    nsrc = m;
                }
            }
            if (g == 0) {
                uint32_t eq = 0;
                for (uint32_t i = 0; i < nsrc; ++i) {
                    eq += src[i] == prefix ? 1 : 0;
                }
                hist[0] = eq;
                report();
            }
            noc_semaphore_wait_min(sel_sem, ++msgs);
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* qo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                mbox + mladsa::QUOTA_OFF + j * mladsa::MAX_CHUNKS * 4 * mladsa::QUOTA_W);
            uint32_t quota = qo[c * mladsa::QUOTA_W];
            off = qo[c * mladsa::QUOTA_W + 1];
            // ---- this core's local selection: (key, local pool id) pairs in local order (the global tie order)
            volatile tt_l1_ptr uint32_t* kp = kept + j * 2 * KW;
            uint32_t n = 0;
            for (uint32_t i = 0; i < nvalid; ++i) {
                const uint32_t k = keys[i];
                bool take = k > prefix;
                if (!take && k == prefix && quota > 0) {
                    --quota;
                    take = true;
                }
                if (take) {
                    kp[2 * n] = k;
                    kp[2 * n + 1] = mladsa::pool_of(i, c, C);
                    ++n;
                }
            }
            ASSERT(n <= KW);  // (a core's kept pools of a token: <= its keys, <= the local top-512)
            n_kept[j] = n;
        }
        if (g == 0) {  // the records (key, core, 0, 0) at this core's offset of token t's candidate chunk of x_ag
            const uint32_t n = n_kept[j];
            volatile tt_l1_ptr uint32_t* kp = kept + j * 2 * KW;
            uint32_t byte = mlashard::REC_HDR + off * mlashard::REC_BYTES;
            uint32_t i = 0;
            while (i < n) {
                // one write per page: the run of records that stays inside the page of `byte`
                const uint32_t in_page = byte % mlashard::TB;
                uint32_t run = (mlashard::TB - in_page) / mlashard::REC_BYTES;
                if (run > n - i) {
                    run = n - i;
                }
                const uint32_t src = stg + (in_page & 63);  // source congruent to the destination mod 64
                volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
                for (uint32_t j = 0; j < run; ++j) {
                    rec[4 * j] = kp[2 * (i + j)];
                    rec[4 * j + 1] = c;
                    rec[4 * j + 2] = 0;
                    rec[4 * j + 3] = 0;
                }
                invalidate_l1_cache();
                noc_async_write(
                    src,
                    xag.get_noc_addr(mlashard::agx_cand_page(t_rows, 0, t, byte), in_page),
                    run * mlashard::REC_BYTES);
                noc_async_write_barrier();
                i += run;
                byte += run * mlashard::REC_BYTES;
            }
            noc_semaphore_inc(get_noc_addr(ag_x, ag_y, get_semaphore(ag_sem)), 1);
            noc_async_atomic_barrier();
        }
    }
    MK_MARK();  // probe 1: the local select done, this core's records in x_ag
    // rev 59: merge helper c (ix_dm0.cpp): chip c's candidate list from y_ag, the radix rounds with the index core
    // (start, 8 rounds + the done flag, the tie quotas), helper `rank` the per-core quotas of this chip's records
    auto merge_help = [&](uint32_t t, uint32_t j) {  // (t: the y_ag token; j: its mailbox slots)
        noc_semaphore_wait_min(sel_sem, ++msgs);     // start: every chip's records landed in y_ag
        for (uint32_t pg = 0; pg < mlashard::CAND_PAGES; ++pg) {
            noc_async_read_page(
                mlashard::agx_cand_page(t_rows, mlashard::ag_blk(c), t, pg * mlashard::TB),
                xyag,
                mstg + pg * mlashard::TB);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* rq = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mstg);
        const uint32_t n = rq[0] < mlashard::CAND_MAX ? rq[0] : mlashard::CAND_MAX;
        for (uint32_t i = 0; i < n; ++i) {
            mkeys[i] = rq[(mlashard::REC_HDR + i * mlashard::REC_BYTES) / 4];
        }
        uint32_t prefix = 0, mask = 0;
        volatile tt_l1_ptr uint32_t* src = mkeys;
        uint32_t nsrc = n;
        for (uint32_t r = 0; r < mladsa::ROUNDS; ++r) {
            const uint32_t shift = 32 - mladsa::DIGIT_BITS * (r + 1);
            for (uint32_t b = 0; b < mladsa::BINS; ++b) {
                hist[b] = 0;
            }
            for (uint32_t i = 0; i < nsrc; ++i) {
                const uint32_t k = src[i];
                if ((k & mask) == prefix) {
                    hist[(k >> shift) & (mladsa::BINS - 1)] += 1;
                }
            }
            report();  // slot c of the coordinator's histogram tensor
            noc_semaphore_wait_min(sel_sem, ++msgs);
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* mm =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mbox + mlashard::merge_msg(j, 1 + r));
            if (mm[2] != 0) {
                return;  // done: fewer candidates than k (T_g = 0, every one selected; no tie round)
            }
            prefix = mm[0];
            mask |= (mladsa::BINS - 1) << shift;
            uint32_t m = 0;
            for (uint32_t i = 0; i < nsrc; ++i) {
                const uint32_t k = src[i];
                if ((k & mask) == prefix) {
                    mcand[m++] = k;
                }
            }
            src = mcand;
            nsrc = m;
        }
        uint32_t eq = 0;
        for (uint32_t i = 0; i < nsrc; ++i) {
            eq += src[i] == prefix ? 1 : 0;
        }
        hist[0] = eq;
        report();
        noc_semaphore_wait_min(sel_sem, ++msgs);  // the per-helper tie quotas (chip order)
        invalidate_l1_cache();
        if (c != rank) {
            return;
        }
        // this chip's list = its records in (core, local) order: the first `q` keys == T_g, counted per core
        uint32_t q =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mbox + mlashard::merge_msg(j, 1 + mladsa::ROUNDS))[c];
        for (uint32_t k = 0; k < mladsa::MAX_CHUNKS; ++k) {
            mgq[k] = 0;
        }
        for (uint32_t i = 0; i < n && q > 0; ++i) {
            if (mkeys[i] == prefix) {
                mgq[rq[(mlashard::REC_HDR + i * mlashard::REC_BYTES) / 4 + 1] & (mladsa::MAX_CHUNKS - 1)] += 1;
                --q;
            }
        }
        invalidate_l1_cache();
        noc_async_write(reinterpret_cast<uint32_t>(mgq), get_noc_addr(cx, cy, hist_addr), mladsa::MAX_CHUNKS * 4);
        noc_async_write_barrier();
        noc_semaphore_inc(h_sem, 1);
        noc_async_atomic_barrier();
    };
    // ---- B: the global filter -> CB_ASEL page j
    auto sweep_b = [&]() {
        for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
            const uint32_t p = pos_of(t);
            const uint32_t j = mlashard::lane_tok(t);
            if (helper && mladsa::sparse(p)) {
                merge_help(t, j);
            }
            noc_semaphore_wait_min(sel_sem, ++msgs);  // the 10th message: after the all-gather and the merge
            invalidate_l1_cache();
            const uint32_t T_g =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mbox + mlashard::GMSG_OFF + j * mladsa::MSG_BYTES)[0];
            uint32_t quota = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                mbox + mlashard::GQUOTA_OFF + j * mladsa::MAX_CHUNKS * 4)[c];
            cb_reserve_back(CB_ASEL, 1);
            volatile tt_l1_ptr uint32_t* sel = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_ASEL));
            uint32_t n_sel = 0;
            volatile tt_l1_ptr uint32_t* kp = kept + j * 2 * KW;
            for (uint32_t i = 0; i < n_kept[j]; ++i) {
                const uint32_t k = kp[2 * i];
                bool take = k > T_g;
                if (!take && k == T_g && quota > 0) {
                    --quota;
                    take = true;
                }
                if (take) {
                    sel[mladsa::SEL_HDR + n_sel++] = kp[2 * i + 1];
                }
            }
            const bool tail = mladsa::sparse(p) && c == 0 && mlashard::owner(p, nchips) == rank;
            sel[0] = n_sel;
            sel[1] = tail ? (p + 1) % mladsa::KPOOL : 0;
            sel[2] = mlashard::local_pool(p / mladsa::KPOOL, nchips);
            cb_push_back(CB_ASEL, 1);
        }
        MK_MARK();  // probe 2: the global merge done (every token: T_g + quota -> CB_ASEL)
    };
    if (!all_dense) {
        sweep_b();
    } else if (c == 0) {  // the RS lanes' per-launch protocol: one root increment per lane token (no partial shipped)
        for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
            noc_semaphore_inc(get_noc_addr(rs_x, rs_y, get_semaphore(rs_sem)), 1);
#if defined(MLA_DSA_SHARD_LINE)
            noc_semaphore_inc(
                get_noc_addr(rs2_x, rs2_y, get_semaphore(rs2_sem)), 1);  // TP8: the round-2 scatter lane too
#endif
        }
        noc_async_atomic_barrier();
    }
    // ---- C / D: the two passes' partial trees; the root ships (pass 0) / lands (pass 1) the cross-chip partial
    volatile tt_l1_ptr uint32_t* rsc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(RS_SEM));
#if defined(MLA_ATT_OT)
    // rev 60: o_T (att_dsa_dm1.cpp rev 39's writer): rows r0 .. r0 + HR - 1 of the finalised o tile d (L1 at src) =
    // this chip's HR heads -> row t of each head's column tile d of o_T [1, 1, T, H 512]. A row whose source is not
    // congruent mod 64 with its destination goes through a 64-B staging slot of the root's send CB: the root's send CBs
    // are empty whenever it finalises (pass 0's ship and RS round 2's P1 are popped before; 2 HR 64 B <= 2 fp32 tiles).
    // Slots are reused per tile after the previous tile's writes left L1 (the flush)
    auto write_ot = [&](uint32_t src, uint32_t t, uint32_t d, uint32_t r0) {
        const uint32_t drow = mla::row_off(t);
        const uint32_t ostg = (get_write_ptr(CB_ASEND) + 63) & ~63u;
        uint64_t dst[HR];
        uint32_t from[HR];
        uint32_t staged = 0;
        noc_async_writes_flushed();
        invalidate_l1_cache();
        for (uint32_t h = 0; h < HR; ++h) {
            dst[h] = ot.get_noc_addr((g * 32 + h) * mlaatt::DT + d, drow);
            const uint32_t off = static_cast<uint32_t>(dst[h]) & 63;
            from[h] = src + mla::row_off(r0 + h);
            if ((from[h] & 63) != off) {
                const uint32_t slot = ostg + 2 * h * 64 + off;
                mla::copy_row32(from[h], slot);
                mla::copy_row32(from[h] + 512, slot + 64);
                from[h] = slot;
                staged |= 1u << h;
            }
        }
        invalidate_l1_cache();
        for (uint32_t h = 0; h < HR; ++h) {
            noc_async_write(from[h], dst[h], 32);
            noc_async_write(from[h] + (((staged >> h) & 1) ? 64 : 512), dst[h] + 512, 32);
        }
    };
#endif
    for (uint32_t pass = all_dense ? 1 : 0; pass < 2; ++pass) {
        if (pass == 1) {
            MK_MARK();  // probe 3: pass 0 done (the root: its partial shipped into x_rs)
            if (c == 0 && !all_dense) {
                if constexpr (mlashard::LINE > 1) {
                    noc_semaphore_wait_min(rsc, 1);  // RS round 1 landed (round 2's signal comes later: >= waits only)
                } else {
                    mk::wait_handoff(
                        RS_SEM, 1);  // the reduce-scatter landed this chip's block of y_rs (the lane's waiter)
                }
            }
            MK_MARK();  // probe 4: RS round 1 landed (the root)
        }
        for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
            for (uint32_t L = 0; L < LV; ++L) {
                const uint32_t px = get_arg_val<uint32_t>(lv0 + 2 * L), py = get_arg_val<uint32_t>(lv0 + 2 * L + 1);
                if (mlaatt::recv_at(c, L, C)) {
                    cb_reserve_back(CB_ARX, ST);
                    cb_reserve_back(CB_ARXO, DT);
                    noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(SEM_RDY)), 1);
                    noc_async_atomic_barrier();
                    mk::wait_handoff(SEM_DAT, 1);
                    cb_push_back(CB_ARX, ST);
                    for (uint32_t d0_ = 0; d0_ < DT; d0_ += OG) {
                        noc_semaphore_wait_min(dato, d0_ + OG);
                        cb_push_back(CB_ARXO, OG);
                    }
                    noc_semaphore_set(dato, 0);
                } else if (mlaatt::send_at(c, L)) {
                    mk::wait_handoff(SEM_RDY, 1);
                    cb_wait_front(CB_ASEND, ST);
                    noc_async_write(get_read_ptr(CB_ASEND), get_noc_addr(px, py, arx_base), ST * ptb);
                    noc_async_write_barrier();
                    noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(SEM_DAT)), 1);
                    cb_pop_front(CB_ASEND, ST);
                    const uint64_t dst_o = get_noc_addr(px, py, arxo_base);
                    const uint64_t sem_o = get_noc_addr(px, py, get_semaphore(SEM_DATO));
                    for (uint32_t d0_ = 0; d0_ < DT; d0_ += OG) {
                        cb_wait_front(CB_ASENDO, d0_ + OG);
                        noc_async_write(get_read_ptr(CB_ASENDO) + d0_ * otb, dst_o + d0_ * otb, OG * otb);
                        noc_async_write_barrier();
                        noc_semaphore_inc(sem_o, OG);
                    }
                    noc_async_atomic_barrier();
                    cb_pop_front(CB_ASENDO, DT);
                    break;
                }
            }
            if (c != 0) {
                continue;
            }
            // the O / stats tiles of a partial as x_rs / y_rs pages: tile k of otb bytes = otb / PG pages from page p0
            const uint32_t rsb = mlashard::rs_base(t_rows, pass == 0 ? peer_blk : 0, t);
            const uint32_t o_pages = otb / PG, s_pages = ptb / PG;
            ASSERT(DT * o_pages + ST * s_pages <= mlashard::RS_PAGES);
            if (pass == 0) {  // the root ships its partial (the send CBs: the compute's last combine landed there)
                cb_wait_front(CB_ASEND, ST);
                for (uint32_t i = 0; i < ST; ++i) {
                    for (uint32_t h = 0; h < s_pages; ++h) {
                        noc_async_write_page(
                            rsb + DT * o_pages + i * s_pages + h, xrs, get_read_ptr(CB_ASEND) + i * ptb + h * PG);
                    }
                }
                for (uint32_t d0_ = 0; d0_ < DT; d0_ += OG) {
                    cb_wait_front(CB_ASENDO, d0_ + OG);
                    for (uint32_t d = d0_; d < d0_ + OG; ++d) {
                        for (uint32_t h = 0; h < o_pages; ++h) {
                            noc_async_write_page(
                                rsb + d * o_pages + h, xrs, get_read_ptr(CB_ASENDO) + d * otb + h * PG);
                        }
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(CB_ASEND, ST);
                cb_pop_front(CB_ASENDO, DT);
                noc_semaphore_inc(get_noc_addr(rs_x, rs_y, get_semaphore(rs_sem)), 1);
                noc_async_atomic_barrier();
            } else if (all_dense) {  // the window pass: the finalised o tiles of this chip's heads -> o_lat
                const uint32_t tbw = get_tile_size(CB_AOUT);
                for (uint32_t d = 0; d < mlaatt::DT; ++d) {
                    cb_wait_front(CB_AOUT, d + 1);
                    noc_async_write_page((t * HPT + g) * mlaatt::DT + d, olat, get_read_ptr(CB_AOUT) + d * tbw);
#if defined(MLA_ATT_OT)
                    write_ot(get_read_ptr(CB_AOUT) + d * tbw, t, d, 0);
#endif
                }
                noc_async_write_barrier();
                cb_pop_front(CB_AOUT, mlaatt::DT);
            } else {  // the peer's partial of this chip's heads -> the receive CBs, after the children's messages
                cb_reserve_back(CB_ARX, ST);
                cb_reserve_back(CB_ARXO, DT);
                const uint32_t rx = get_write_ptr(CB_ARX), rxo = get_write_ptr(CB_ARXO);
                for (uint32_t i = 0; i < ST; ++i) {
                    for (uint32_t h = 0; h < s_pages; ++h) {
                        noc_async_read_page(rsb + DT * o_pages + i * s_pages + h, yrs, rx + i * ptb + h * PG);
                    }
                }
                for (uint32_t d = 0; d < DT; ++d) {
                    for (uint32_t h = 0; h < o_pages; ++h) {
                        noc_async_read_page(rsb + d * o_pages + h, yrs, rxo + d * otb + h * PG);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(CB_ARX, ST);
                cb_push_back(CB_ARXO, DT);
#if defined(MLA_DSA_SHARD_LINE)
                // TP8 round 2 send: the compute's combine put the column-combined partial P1 in the send CBs; rows
                // 8 q .. 8 q + 7 of every tile -> x_rs2 block q (q = every column of the row, this chip's own too)
                const uint32_t ob = otb / 4, pb = ptb / 4;  // slice bytes of an O / a stats tile (2 runs of tb / 8)
                ASSERT(DT * ob + ST * pb <= mlashard::RS2_PPC * mlashard::TB);
                cb_wait_front(CB_ASEND, ST);
                for (uint32_t q = 0; q < mlashard::LINE; ++q) {
                    const uint32_t pg0 = mlashard::rs2_page(t_rows, q, t);
                    for (uint32_t i = 0; i < ST; ++i) {
                        const uint32_t b = DT * ob + i * pb;
                        const uint64_t dst = xrs2.get_noc_addr(pg0 + b / PG, b % PG);
                        const uint32_t src = get_read_ptr(CB_ASEND) + i * ptb + mlashard::slice_off(q, ptb);
                        noc_async_write(src, dst, ptb / 8);
                        noc_async_write(src + ptb / 4, dst + ptb / 8, ptb / 8);
                    }
                }
                for (uint32_t d0_ = 0; d0_ < DT; d0_ += OG) {
                    cb_wait_front(CB_ASENDO, d0_ + OG);
                    for (uint32_t q = 0; q < mlashard::LINE; ++q) {
                        const uint32_t pg0 = mlashard::rs2_page(t_rows, q, t);
                        for (uint32_t d = d0_; d < d0_ + OG; ++d) {
                            const uint32_t b = d * ob;
                            const uint64_t dst = xrs2.get_noc_addr(pg0 + b / PG, b % PG);
                            const uint32_t src = get_read_ptr(CB_ASENDO) + d * otb + mlashard::slice_off(q, otb);
                            noc_async_write(src, dst, otb / 8);
                            noc_async_write(src + otb / 4, dst + otb / 8, otb / 8);
                        }
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(CB_ASEND, ST);
                cb_pop_front(CB_ASENDO, DT);
                noc_semaphore_inc(get_noc_addr(rs2_x, rs2_y, get_semaphore(rs2_sem)), 1);
                noc_async_atomic_barrier();
#else
                // the root's o tiles as the compute packs them (finalize after the seventh combine)
                const uint32_t tb = get_tile_size(CB_AOUT);
                for (uint32_t d = 0; d < mlaatt::DT; ++d) {
                    cb_wait_front(CB_AOUT, d + 1);
                    noc_async_write_page((t * HPT + g) * mlaatt::DT + d, olat, get_read_ptr(CB_AOUT) + d * tb);
#if defined(MLA_ATT_OT)
                    write_ot(
                        get_read_ptr(CB_AOUT) + d * tb, t, d, 0);  // the pair: this chip's heads = rows 0 .. HR - 1
#endif
                }
                noc_async_write_barrier();
                cb_pop_front(CB_AOUT, mlaatt::DT);
#endif
            }
        }
    }
    MK_MARK();  // probe 5: pass 1 done (TP8 root: + the round-1 combine, P1's slices shipped; the pair: finalised)
    if (all_dense) {
        sweep_b();  // the merge messages of an all-dense launch (CB_ASEL pages for the reader's final pop)
    }
#if defined(MLA_DSA_SHARD_LINE)
    // ---- E: TP8 round 2 on the root -- per token its own slice of P1 (x_rs2 block lpos) then the row peers' slices of
    // its heads (y_rs2 block s = the source column, ascending) into rows 8 lpos .. of the receive CBs; the compute
    // takes the first, combines the others, finalises; rows 8 lpos .. 8 lpos + 7 of the o tiles -> o_lat rows 0..7
    if (c == 0 && !all_dense) {          // (the dense window's pass wrote o_lat in D: no round 2)
        noc_semaphore_wait_min(rsc, 2);  // the scatter's waiter: every row peer's slices landed in y_rs2
        MK_MARK();                       // probe 6: RS round 2 landed (TP8 root)
        const uint32_t ob = otb / 4, pb = ptb / 4;
        const uint32_t tb = get_tile_size(CB_AOUT);
        // one slice (chunk pg0 of x_rs2 / y_rs2) -> rows 8 lpos .. of the receive CBs (a message for the compute)
        auto land = [&](const auto& src_t, uint32_t pg0) {
            cb_reserve_back(CB_ARX, ST);
            cb_reserve_back(CB_ARXO, DT);
            const uint32_t rx = get_write_ptr(CB_ARX), rxo = get_write_ptr(CB_ARXO);
            for (uint32_t i = 0; i < ST; ++i) {
                const uint32_t b = DT * ob + i * pb;
                const uint64_t src = src_t.get_noc_addr(pg0 + b / PG, b % PG);
                const uint32_t dst = rx + i * ptb + mlashard::slice_off(lpos, ptb);
                noc_async_read(src, dst, ptb / 8);
                noc_async_read(src + ptb / 8, dst + ptb / 4, ptb / 8);
            }
            for (uint32_t d = 0; d < DT; ++d) {
                const uint32_t b = d * ob;
                const uint64_t src = src_t.get_noc_addr(pg0 + b / PG, b % PG);
                const uint32_t dst = rxo + d * otb + mlashard::slice_off(lpos, otb);
                noc_async_read(src, dst, otb / 8);
                noc_async_read(src + otb / 8, dst + otb / 4, otb / 8);
            }
            noc_async_read_barrier();
            cb_push_back(CB_ARX, ST);
            cb_push_back(CB_ARXO, DT);
        };
        for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
            land(xrs2, mlashard::rs2_page(t_rows, lpos, t));    // the own slice first
            for (uint32_t s_ = 0; s_ < mlashard::LINE; ++s_) {  // then the row peers in column order
                if (s_ != lpos) {
                    land(yrs2, mlashard::rs2_page(t_rows, s_, t));
                }
            }
            // this chip's 8 heads: rows 8 lpos .. of each finalised bf16 o tile -> o_lat tile (t, d) rows 0..7
            for (uint32_t d = 0; d < mlaatt::DT; ++d) {
                cb_wait_front(CB_AOUT, d + 1);
                const uint32_t src = get_read_ptr(CB_AOUT) + d * tb + mlashard::slice_off(lpos, tb);
                const uint64_t dst = olat.get_noc_addr((t * HPT + g) * mlaatt::DT + d, 0);
                noc_async_write(src, dst, tb / 8);
                noc_async_write(src + tb / 4, dst + tb / 4, tb / 8);
#if defined(MLA_ATT_OT)
                write_ot(get_read_ptr(CB_AOUT) + d * tb, t, d, mlashard::PIECE_ROWS * lpos);  // rows 8 lpos ..
#endif
            }
            noc_async_write_barrier();
            cb_pop_front(CB_AOUT, mlaatt::DT);
        }
        noc_semaphore_set(rsc, 0);
    } else {
        MK_MARK();  // (the same mark count on every core)
    }
#else
    MK_MARK();  // (probe 6: the pair has no round 2)
#endif
    MK_MARK();  // probe 7: finalised, o_lat written (TP8 root: after the round-2 combines)
    if (msgs > 0) {
        noc_semaphore_set(sel_sem, 0);
    }
    MK_MARK();  // probe 8: the phase ends
    if constexpr (NCONS > 0) {
        if (c == 0) {
            mk::signal_handoff(SIGNAL_SEM, lv0 + 2 * LV, NCONS);
        }
    }
}
