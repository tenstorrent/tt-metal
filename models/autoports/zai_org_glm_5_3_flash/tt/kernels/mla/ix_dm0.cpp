// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DSA index core (kernels/mla/ix_common.hpp) — BRISC (NoC0): the constant tiles (LayerNorm weight / bias, ape, row
// mask) and cur_pos first, then after the I1 readers' hand-off the 9 fp32 projection tiles. Then the coordinator of
// the owners' top-512 radix select (kernels/mla/dsa_common.hpp I5), per sparse token: per round, wait for the C
// group-0 histograms (H_SEM), sum them, pick the digit holding the k-th largest key, send (prefix, k_remaining) to
// every owner's mailbox and increment its SEL_SEM; the tie round turns the C counts of keys == T into quotas in chunk
// order. Compile-time args: WAIT_SEM, WAIT_COUNT, T_MAX, H_SEM, SEL_SEM, C (chunk cores per group; with MLA_ATT_LANES
// the
//   lane's), NOWN (owners; the lane's), then the TensorAccessorArgs of y, the constants, cur_pos.
// Runtime args: y, constants, cur_pos addresses, T, the histogram tensor's (local) address, the owners' mailbox
//   address, then NOWN owner NoC (x, y) pairs; then the lane (MLA_ATT_LANES, rev 47); then NR + MCAST_MAX x (x0, y0,
//   x1, y1, n) owner rectangles (MLA_DSA_MCAST, rev 50); then the probe pair (MK_PROBE_ARG).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/dsa_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/ix_common.hpp"
#if defined(MLA_DSA_SHARD)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/dsa_shard.hpp"
#endif
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_common.hpp"

void kernel_main() {
    using namespace mlaix;
    constexpr uint32_t WAIT_SEM = get_compile_time_arg_val(0);
    constexpr uint32_t WAIT_COUNT = get_compile_time_arg_val(1);
    constexpr uint32_t H_SEM = get_compile_time_arg_val(3);
    constexpr uint32_t SEL_SEM = get_compile_time_arg_val(4);
    constexpr uint32_t C = get_compile_time_arg_val(5);
    constexpr uint32_t NOWN = get_compile_time_arg_val(6);
    constexpr auto y_args = TensorAccessorArgs<7>();
    constexpr auto c_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
#if defined(MLA_DSA_SHARD)  // kernels/mla/dsa_shard.hpp: this chip's block (x_ag; rev 60, review 2 item 8: the y_ag
    // accessor of rev 58's serial merge is gone -- the merge helpers read y_ag)
    constexpr auto xag_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
#endif
    static_assert(C <= mladsa::MAX_CHUNKS, "chunks");
    static_assert(mladsa::T_MAX * 4 <= mladsa::POS_BYTES, "cur_pos scratch");
    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(0));
    const auto cst = TensorAccessor(c_args, get_arg_val<uint32_t>(1));
    const auto cur = TensorAccessor(p_args, get_arg_val<uint32_t>(2));
    const uint32_t t_rows = get_arg_val<uint32_t>(3);
    const uint32_t hist_addr = get_arg_val<uint32_t>(4);
    const uint32_t mbox = get_arg_val<uint32_t>(5);
    constexpr uint32_t OWN0 = 6;
#if defined(MLA_ATT_LANES)  // rev 47: this token lane's tokens only; the owners (NOWN) are the lane's
    const uint32_t t0 = get_arg_val<uint32_t>(OWN0 + 2 * NOWN);
#else
    const uint32_t t0 = 0;
#endif
    MK_MARK();
    cb_reserve_back(CB_C, 13);
    const uint32_t c1 = get_write_ptr(CB_C);
    for (uint32_t i = 0; i < 13; ++i) {
        noc_async_read_page(i, cst, c1 + i * get_tile_size(CB_C));
    }
    cb_reserve_back(CB_INFO, 1);
    noc_async_read_page(0, cur, get_write_ptr(CB_INFO));
    const uint32_t scr = get_write_ptr(CB_CSCR);
    ASSERT((scr & 63) == 0);  // cur_pos lands here from DRAM (rev 43: a Blackhole DRAM read target is 64-B aligned)
    noc_async_read_page(0, cur, scr);
    noc_async_read_barrier();
    cb_push_back(CB_C, 13);
    cb_push_back(CB_INFO, 1);
    mk::wait_handoff(WAIT_SEM, WAIT_COUNT);
    MK_MARK();
    cb_reserve_back(CB_Y, 9);
    const uint32_t l1 = get_write_ptr(CB_Y);
    for (uint32_t i = 0; i < 9; ++i) {
        noc_async_read_page(i, y, l1 + i * get_tile_size(CB_Y));
    }
    noc_async_read_barrier();
    cb_push_back(CB_Y, 9);
    MK_MARK();

#if !defined(MLA_DSA_COORD_EXT)  // rev 40: with token lanes each lane's coordinator is dsa_coord.cpp
    // ---- I5 coordinator
    invalidate_l1_cache();
    volatile tt_l1_ptr int32_t* pos_l1 = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr);
    // CB_CSCR (512 B): cur_pos [0, POS_BYTES) (32 rows), the round message [POS_BYTES, + 64), the quotas after it
    constexpr uint32_t MSG_OFF = mladsa::POS_BYTES, QUOTA_L1 = mladsa::POS_BYTES + 64;
    static_assert(QUOTA_L1 + ((C * 4 + 15) & ~15u) <= 512, "CB_CSCR");
    volatile tt_l1_ptr uint32_t* msg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + MSG_OFF);
    volatile tt_l1_ptr uint32_t* quota = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + QUOTA_L1);
    volatile tt_l1_ptr uint32_t* hist = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hist_addr);
#if defined(MLA_DSA_MCAST)
    // rev 50: the owners as NR exact virtual-NoC rectangles (tt/mk_mla.py _owner_rects: each covers only owner cores; a
    // 1 x 1 one is a unicast), runtime args after the lane: NR, then NR x (x0, y0, x1, y1, n), padded to MCAST_MAX. The
    // message is multicast per rectangle, then SEL_SEM is SET to the running message count (the owners wait_min on it:
    // monotonic) -- 2 NR NoC commands per round instead of 2 NOWN. Opt-in: on device it was 7-15 us per layer slower
    // than the unicasts (card0 u52 / u61; the round time was the histogram sum, rev 52). NOC0 corner order (start <=
    // end): ix_dm0 runs on RISCV_0's default NoC.
    constexpr uint32_t MC0 = OWN0 + 2 * NOWN + (mlaatt::LANES > 1 ? 1 : 0);
    const uint32_t n_rect = get_arg_val<uint32_t>(MC0);
    static_assert(QUOTA_L1 + ((C * 4 + 15) & ~15u) <= 496, "CB_CSCR: the semaphore value word at 496");
    volatile tt_l1_ptr uint32_t* semval = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 496);
    uint32_t sent = 0;
    ASSERT(noc_index == 0);  // rev 53 (review S3): get_noc_multicast_addr's corner order below is NOC0's
    auto broadcast = [&](uint32_t src, uint32_t dst_off, uint32_t bytes) {
        *semval = ++sent;
        invalidate_l1_cache();  // the RISC's stores land before the NoC reads them (mla_common.hpp)
        for (uint32_t r = 0; r < n_rect; ++r) {
            const uint32_t a = MC0 + 1 + 5 * r;
            const uint32_t x0 = get_arg_val<uint32_t>(a), y0 = get_arg_val<uint32_t>(a + 1);
            const uint32_t x1 = get_arg_val<uint32_t>(a + 2), y1 = get_arg_val<uint32_t>(a + 3),
                           n = get_arg_val<uint32_t>(a + 4);
            if (n == 1) {
                noc_async_write(src, get_noc_addr(x0, y0, mbox + dst_off), bytes);
            } else {
                noc_async_write_multicast(src, get_noc_multicast_addr(x0, y0, x1, y1, mbox + dst_off), bytes, n);
            }
        }
        noc_async_write_barrier();  // every message landed before any SEL_SEM moves
        for (uint32_t r = 0; r < n_rect; ++r) {
            const uint32_t a = MC0 + 1 + 5 * r;
            const uint32_t x0 = get_arg_val<uint32_t>(a), y0 = get_arg_val<uint32_t>(a + 1);
            const uint32_t x1 = get_arg_val<uint32_t>(a + 2), y1 = get_arg_val<uint32_t>(a + 3),
                           n = get_arg_val<uint32_t>(a + 4);
            if (n == 1) {
                noc_semaphore_set_remote(scr + 496, get_noc_addr(x0, y0, get_semaphore(SEL_SEM)));
            } else {
                noc_semaphore_set_multicast(
                    scr + 496, get_noc_multicast_addr(x0, y0, x1, y1, get_semaphore(SEL_SEM)), n);
            }
        }
        noc_async_write_barrier();  // semval is rewritten by the next broadcast
    };
#else
    // src [scr + off, + bytes) -> every owner's mailbox at mbox + dst_off, then SEL_SEM += 1 on each
    auto broadcast = [&](uint32_t src, uint32_t dst_off, uint32_t bytes) {
        invalidate_l1_cache();  // the RISC's stores land before the NoC reads them (mla_common.hpp)
        for (uint32_t k = 0; k < NOWN; ++k) {
            const uint32_t x = get_arg_val<uint32_t>(OWN0 + 2 * k), yy = get_arg_val<uint32_t>(OWN0 + 2 * k + 1);
            noc_async_write(src, get_noc_addr(x, yy, mbox + dst_off), bytes);
        }
        noc_async_write_barrier();
        for (uint32_t k = 0; k < NOWN; ++k) {
            const uint32_t x = get_arg_val<uint32_t>(OWN0 + 2 * k), yy = get_arg_val<uint32_t>(OWN0 + 2 * k + 1);
            noc_semaphore_inc(get_noc_addr(x, yy, get_semaphore(SEL_SEM)), 1);
        }
        noc_async_atomic_barrier();
    };
#endif
#if defined(MLA_DSA_SHARD)
    // ---- the shard coordinator (kernels/mla/dsa_shard.hpp): the local select rounds as below, plus per core the count
    // of keys above the threshold (above[c], from every round's histograms) so the tie message carries (quota, record
    // offset) per core; then, after the all-gather of the C chips' candidate records, the exact global top-512 of the
    // union (T_g; ties in (chip, core, local) order) and this chip's per-core tie quotas as the 10th message. Rev 60
    // (W): with token lanes (MLA_ATT_LANES) this core is lane t0's index core -- its tokens t0, t0 + LANES, .., its
    // C / NOWN = the lane's chunk cores / owners, the mailbox slots of the lane-local token j = t / LANES
    static_assert(mladsa::QUOTA_W == 2, "the shard tie message: (quota, offset) per core");
    constexpr uint32_t SH0 = OWN0 + 2 * NOWN + (mlaatt::LANES > 1 ? 1 : 0);  // (after the lane arg)
    const uint32_t rank = get_arg_val<uint32_t>(SH0), nchips = get_arg_val<uint32_t>(SH0 + 1);
    const auto xag = TensorAccessor(xag_args, get_arg_val<uint32_t>(SH0 + 2));
    // (SH0 + 3: the y_ag address, unused since rev 59's helper merge; the slot stays: fixed runtime-arg offsets)
    const uint32_t ag_x = get_arg_val<uint32_t>(SH0 + 4), ag_y = get_arg_val<uint32_t>(SH0 + 5);
    const uint32_t ag_sem = get_arg_val<uint32_t>(SH0 + 6), agc_sem = get_arg_val<uint32_t>(SH0 + 7);
    ASSERT(t_rows <= mlashard::T_MAX && nchips <= 8);
    ASSERT(nchips == mlashard::NCHIPS);
    ASSERT(mlashard::lane_tokens(t_rows, t0) <= mlashard::TL);
    // scratch past the C histogram slots (the host sizes the tensor, tt/mk_mla.py shard_coord_words): the 64-B aligned
    // count header (+ the T_g probe words 8 + j, j < TL <= 8), the (quota, offset) pairs of the tie message (MAX_CHUNKS
    // x 8 B: too big for CB_CSCR). Rev 60: the slots span >= MAX_CHUNKS words (helper `rank` writes its per-core quotas
    // there: 256 B, more than C / L = 2 lane slots)
    constexpr uint32_t SLOTS_BYTES =
        C * mladsa::HIST_SLOT_BYTES > mladsa::MAX_CHUNKS * 4 ? C * mladsa::HIST_SLOT_BYTES : mladsa::MAX_CHUNKS * 4;
    const uint32_t shs = (hist_addr + SLOTS_BYTES + 63) & ~63u;
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(shs);
    volatile tt_l1_ptr uint32_t* qpair = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(shs + 64);
    // rev 59: the global merge runs on NCHIPS merge helpers (the owner chunk cores 0 .. NCHIPS - 1 of group 0 = the
    // first NCHIPS owners of the list; helper q holds chip q's candidate list, in y_ag block ag_blk(q)): the radix
    // rounds as the local select's -- each helper histograms its candidates into slot q, this core sums NCHIPS slots,
    // picks the digit and messages the helpers (kernels/mla/dsa_shard.hpp merge_msg); the tie round's per-helper quotas
    // in helper (= chip) order; helper `rank` then turns its quota into this chip's per-core quotas (its list is this
    // chip's records in (core, local) order) and writes them into slots 0..3. The same union histograms, digits, T_g
    // and (chip, core, local) tie order as rev 58's serial merge of the C x 512 keys on this RISC (bitwise-identical
    // results), with each helper's work 1 / C of it and NCHIPS (not 64) participants per round.
    static_assert(mlashard::NCHIPS <= C, "one merge helper per chip");
    auto hsend = [&](uint32_t src, uint32_t off, uint32_t bytes) {
        invalidate_l1_cache();
        for (uint32_t k = 0; k < mlashard::NCHIPS; ++k) {
            const uint32_t x = get_arg_val<uint32_t>(OWN0 + 2 * k), yy = get_arg_val<uint32_t>(OWN0 + 2 * k + 1);
            noc_async_write(src, get_noc_addr(x, yy, mbox + off), bytes);
        }
        noc_async_write_barrier();
        for (uint32_t k = 0; k < mlashard::NCHIPS; ++k) {
            const uint32_t x = get_arg_val<uint32_t>(OWN0 + 2 * k), yy = get_arg_val<uint32_t>(OWN0 + 2 * k + 1);
            noc_semaphore_inc(get_noc_addr(x, yy, get_semaphore(SEL_SEM)), 1);
        }
        noc_async_atomic_barrier();
    };
#if defined(MLA_DSA_SHARD_LINE)
    // TP8: the index core's NCRISC assembles the q tiles (ix_dm1.cpp) and counts the tokens done on this semaphore
    const uint32_t qasm_sem = get_arg_val<uint32_t>(SH0 + 8);
    volatile tt_l1_ptr uint32_t* qasm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(qasm_sem));
#endif
    volatile tt_l1_ptr uint32_t* gmsg =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + MSG_OFF);  // (T_g, 0) reuses the round message slot
    volatile tt_l1_ptr uint32_t* gquota = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + QUOTA_L1);
    static_assert(QUOTA_L1 + ((C * 4 + 15) & ~15u) <= 512, "CB_CSCR: the global quotas");
    auto broadcast2 =
        [&](uint32_t src1, uint32_t off1, uint32_t bytes1, uint32_t src2, uint32_t off2, uint32_t bytes2) {
            invalidate_l1_cache();
            for (uint32_t k = 0; k < NOWN; ++k) {
                const uint32_t x = get_arg_val<uint32_t>(OWN0 + 2 * k), yy = get_arg_val<uint32_t>(OWN0 + 2 * k + 1);
                noc_async_write(src1, get_noc_addr(x, yy, mbox + off1), bytes1);
                noc_async_write(src2, get_noc_addr(x, yy, mbox + off2), bytes2);
            }
            noc_async_write_barrier();
            for (uint32_t k = 0; k < NOWN; ++k) {
                const uint32_t x = get_arg_val<uint32_t>(OWN0 + 2 * k), yy = get_arg_val<uint32_t>(OWN0 + 2 * k + 1);
                noc_semaphore_inc(get_noc_addr(x, yy, get_semaphore(SEL_SEM)), 1);
            }
            noc_async_atomic_barrier();
        };
    for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
        const int32_t pos = pos_l1[t];
        const uint32_t j = mlashard::lane_tok(t);  // the lane-local token: its mailbox slots
        uint32_t n_rec = 0;
        if (mladsa::sparse(pos < 0 ? 0u : static_cast<uint32_t>(pos))) {
            uint32_t prefix = 0, k_rem = mladsa::TOPK;
            uint32_t above[mladsa::MAX_CHUNKS];
            for (uint32_t c = 0; c < C; ++c) {
                above[c] = 0;
            }
            for (uint32_t r = 0; r < mladsa::ROUNDS; ++r) {
                const uint32_t shift = 32 - mladsa::DIGIT_BITS * (r + 1);
                mk::wait_handoff(H_SEM, C);
                invalidate_l1_cache();
                uint32_t h[mladsa::BINS];
                mladsa::sum_hist(hist, C, h);
                uint32_t d = mladsa::BINS - 1, ab = 0;
                while (d > 0 && ab + h[d] < k_rem) {
                    ab += h[d];
                    --d;
                }
                k_rem -= ab;
                prefix |= d << shift;
                for (uint32_t c = 0; c < C; ++c) {  // this round's keys above the chosen digit, per core
                    for (uint32_t b = d + 1; b < mladsa::BINS; ++b) {
                        above[c] += hist[c * (mladsa::HIST_SLOT_BYTES / 4) + b];
                    }
                }
                msg[0] = prefix;
                msg[1] = k_rem;
                broadcast(scr + MSG_OFF, (j * mladsa::MSGS + r) * mladsa::MSG_BYTES, mladsa::MSG_BYTES);
            }
            mk::wait_handoff(H_SEM, C);
            invalidate_l1_cache();
            uint32_t rem = k_rem, off = 0;
            for (uint32_t c = 0; c < C; ++c) {
                const uint32_t n = hist[c * (mladsa::HIST_SLOT_BYTES / 4)];
                const uint32_t q = n < rem ? n : rem;
                qpair[2 * c] = q;
                qpair[2 * c + 1] = off;
                off += above[c] + q;
                rem -= q;
            }
            n_rec = off;
            broadcast(
                shs + 64,
                mladsa::QUOTA_OFF + j * mladsa::MAX_CHUNKS * 4 * mladsa::QUOTA_W,
                (C * 4 * mladsa::QUOTA_W + 15) & ~15u);
        }
        // this chip's record count -> the header of token t's candidate chunk (x_ag), then the lane may send
        hdr[0] = n_rec;
        hdr[1] = hdr[2] = hdr[3] = 0;
        invalidate_l1_cache();
        noc_async_write(shs, xag.get_noc_addr(mlashard::agx_cand_page(t_rows, 0, t, 0), 0), mlashard::REC_HDR);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(ag_x, ag_y, get_semaphore(ag_sem)), 1);
        noc_async_atomic_barrier();
    }
    MK_MARK();  // probe: the local select coordinated, every token's record count in x_ag
    // ---- after the all-gather: the global merge per token (on the helpers)
#if defined(MLA_DSA_SHARD_LINE)
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(agc_sem)), 1);  // (ix_dm1 too)
#else
    mk::wait_handoff(agc_sem, 1);
#endif
    MK_MARK();  // probe: the all-gather landed (both rounds)
    for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
        const int32_t pos = pos_l1[t];
        const uint32_t j = mlashard::lane_tok(t);
        uint32_t T_g = 0;
        for (uint32_t c = 0; c < C; ++c) {
            gquota[c] = 0;
        }
        if (mladsa::sparse(pos < 0 ? 0u : static_cast<uint32_t>(pos))) {
            msg[0] = 0;
            msg[1] = mladsa::TOPK;
            msg[2] = msg[3] = 0;
            hsend(scr + MSG_OFF, mlashard::merge_msg(j, 0), mladsa::MSG_BYTES);  // start: y_ag holds every list
            uint32_t prefix = 0, k_rem = mladsa::TOPK;
            bool done = false;
            for (uint32_t r = 0; r < mladsa::ROUNDS; ++r) {
                const uint32_t shift = 32 - mladsa::DIGIT_BITS * (r + 1);
                mk::wait_handoff(H_SEM, mlashard::NCHIPS);
                invalidate_l1_cache();
                uint32_t h[mladsa::BINS];
                mladsa::sum_hist(hist, mlashard::NCHIPS, h);
                if (r == 0) {
                    uint32_t m = 0;
                    for (uint32_t b = 0; b < mladsa::BINS; ++b) {
                        m += h[b];
                    }
                    if (m <= mladsa::TOPK) {  // fewer candidates than k: every one is selected (T_g below every key)
                        msg[0] = msg[1] = 0;
                        msg[2] = 1;  // done
                        hsend(scr + MSG_OFF, mlashard::merge_msg(j, 1), mladsa::MSG_BYTES);
                        done = true;
                        break;
                    }
                }
                uint32_t d = mladsa::BINS - 1, ab = 0;
                while (d > 0 && ab + h[d] < k_rem) {
                    ab += h[d];
                    --d;
                }
                k_rem -= ab;
                prefix |= d << shift;
                msg[0] = prefix;
                msg[1] = k_rem;
                msg[2] = 0;
                hsend(scr + MSG_OFF, mlashard::merge_msg(j, 1 + r), mladsa::MSG_BYTES);
            }
            if (!done) {
                // ties: the first k_rem keys == T_g in (chip, core, local) order -> per helper (= chip) quotas
                mk::wait_handoff(H_SEM, mlashard::NCHIPS);
                invalidate_l1_cache();
                uint32_t rem = k_rem;
                for (uint32_t q = 0; q < mlashard::NCHIPS; ++q) {
                    const uint32_t n = hist[q * (mladsa::HIST_SLOT_BYTES / 4)];
                    msg[q] = n < rem ? n : rem;
                    rem -= msg[q];
                }
                static_assert(mlashard::NCHIPS * 4 <= 2 * mladsa::MSG_BYTES && MSG_OFF + 32 <= QUOTA_L1, "tie quotas");
                hsend(scr + MSG_OFF, mlashard::merge_msg(j, 1 + mladsa::ROUNDS), 2 * mladsa::MSG_BYTES);
                T_g = prefix;
                mk::wait_handoff(H_SEM, 1);  // helper `rank`: this chip's per-core quotas, words [0, C) of the slots
                invalidate_l1_cache();
                for (uint32_t c = 0; c < C; ++c) {
                    gquota[c] = hist[c];
                }
            }
        }
        gmsg[0] = T_g;
        gmsg[1] = 0;
        hdr[8 + j] =
            T_g;  // probe: the threshold this coordinator computed, host-readable in the hist tensor (shs words
                  // 8 + j, j < TL; tests/s14/jobs/dsa_shard.py compares it with the owners' mailbox word)
#if defined(MLA_DSA_SHARD_LINE)
        // token t's q tiles are in q_asm (the owners read them after this message): this lane's ix_dm1 counts its
        // tokens
        noc_semaphore_wait_min(qasm, j + 1);
#endif
        broadcast2(
            scr + MSG_OFF,
            mlashard::GMSG_OFF + j * mladsa::MSG_BYTES,
            mladsa::MSG_BYTES,
            scr + QUOTA_L1,
            mlashard::GQUOTA_OFF + j * mladsa::MAX_CHUNKS * 4,
            (C * 4 + 15) & ~15u);
    }
#else
    for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
        const int32_t pos = pos_l1[t];
        if (!mladsa::sparse(pos < 0 ? 0u : static_cast<uint32_t>(pos))) {
            continue;
        }
        uint32_t prefix = 0, k_rem = mladsa::TOPK;
        for (uint32_t r = 0; r < mladsa::ROUNDS; ++r) {
            const uint32_t shift = 32 - mladsa::DIGIT_BITS * (r + 1);
            mk::wait_handoff(H_SEM, C);
            MK_MARK();  // probe (rev 51): round r's histograms all in (16 marks max: t = 1 fits; lanes see their first
                        // token)
            invalidate_l1_cache();
            uint32_t h[mladsa::BINS];
#if defined(MLA_DSA_HSUM_OLD)  // GLM53_MLA_DSA_HSUM=0: the rev <= 51 sum
            for (uint32_t b = 0; b < mladsa::BINS; ++b) {
                h[b] = 0;
            }
            for (uint32_t c = 0; c < C; ++c) {
                for (uint32_t b = 0; b < mladsa::BINS; ++b) {
                    h[b] += hist[c * (mladsa::HIST_SLOT_BYTES / 4) + b];
                }
            }
#else
            mladsa::sum_hist(hist, C, h);  // rev 52: register accumulators
#endif
            // the largest digit d with (keys of digit > d) < k_rem <= (keys of digit >= d)
            uint32_t d = mladsa::BINS - 1, above = 0;
            while (d > 0 && above + h[d] < k_rem) {
                above += h[d];
                --d;
            }
            k_rem -= above;
            prefix |= d << shift;
            msg[0] = prefix;
            msg[1] = k_rem;
            broadcast(scr + MSG_OFF, (t * mladsa::MSGS + r) * mladsa::MSG_BYTES, mladsa::MSG_BYTES);
            if (r == 0) {
                MK_MARK();  // probe (rev 51): round 0's sum + digit + broadcast done (the coordinator's share of a
                            // round)
            }
        }
        // tie round: k_rem keys == prefix, taken in chunk order
        mk::wait_handoff(H_SEM, C);
        MK_MARK();  // probe (rev 51): the tie counts in
        invalidate_l1_cache();
        uint32_t rem = k_rem;
        for (uint32_t c = 0; c < C; ++c) {
            const uint32_t n = hist[c * (mladsa::HIST_SLOT_BYTES / 4)];
            const uint32_t q = n < rem ? n : rem;
            quota[c] = q;
            rem -= q;
        }
        broadcast(scr + QUOTA_L1, mladsa::QUOTA_OFF + t * mladsa::MAX_CHUNKS * 4, (C * 4 + 15) & ~15u);
    }
#endif  // MLA_DSA_SHARD
#else
    (void)t_rows, (void)hist_addr, (void)mbox, (void)t0;
#endif
    MK_MARK();
#if defined(MK_PROBE_ARG)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB));
#endif
}
