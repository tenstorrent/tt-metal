// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MLA megakernel attention phase with DSA (kernels/mla/dsa_common.hpp) — NCRISC (NoC1) of owner core (g, c). Dense
// tokens: kernels/mla/att_dm1.cpp (the group's partial tree, o_lat, the o_lat consumers' signal). Sparse tokens first:
//   keys: row t of each fp32 score tile of CB_ASC (the pools < n_closed) -> order-preserving uint32 keys (CB_AKEY);
//   I5: the exact top-512 by radix select, 4-bit digits MSB first: per round, group-0 cores write the 16-bin histogram
//       of their keys that match the prefix so far into slot c of the coordinator's histogram tensor and increment its
//       H_SEM; every core waits for the coordinator's message (SEL_SEM, cumulative count) in its mailbox: (prefix,
//       k_remaining). After 8 rounds the prefix is the 512th key T; the tie round reports count(key == T) and returns
//       per-core quotas (core order). Selected = key > T, plus the first quota keys == T in local order;
//   CB_ASEL (-> the BRISC): n_sel, tail rows (core 0: the (p + 1) % 4 rows of the incomplete pool), tail pool, ids.
// Every core of a chunk (all groups) computes the same keys, so the selection is the same in every head group.
// Compile-time args: att_dm1.cpp's 13 (.. CB_ARXO, CB_ASENDO, SEM_DATO), then CB_ASC, CB_ASEL, CB_AKEY, SEL_SEM, H_SEM,
// then the TensorAccessorArgs of
//   o_lat and cur_pos.
// Runtime args: att_dm1.cpp's (o_lat, g, c, T, 2 LV tree, 2 NCONS consumers), then cur_pos address, coordinator NoC x,
//   y, its histogram tensor address, this core's mailbox address (+ the token lane with MLA_ATT_LANES, rev 40: the
//   coordinator is the lane's, kernels/mla/dsa_coord.cpp; c / C / LV are the lane's) (+ the o_T address with
//   MLA_ATT_OT).
// MLA_ATT_OT = HR (Stage 14 rev 39, as att_dm1.cpp rev 30): the root also writes W_uv's in0 rows into o_T [1, 1, T, H
// 512] (its accessor args follow cur_pos's), so W_uv pulls whole tiles instead of gathering face rows.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/dsa_common.hpp"
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
    constexpr auto o_args = TensorAccessorArgs<PH_CT0 + 18>();
    constexpr auto c_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
#if defined(MLA_ATT_OT)
    constexpr auto ot_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr uint32_t HR = MLA_ATT_OT;
    static_assert(HR >= 1 && HR <= 32, "o_T: real heads per 32-row group");
#endif
    constexpr uint32_t ST = mlaatt::STAT_TILES, DT = mlaatt::DT;
    static_assert(C <= mladsa::MAX_CHUNKS, "chunks");
    static_assert(mladsa::T_MAX * 4 <= mladsa::POS_BYTES, "cur_pos scratch");
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
#if defined(MLA_ATT_LANES)
    const uint32_t t0 = get_arg_val<uint32_t>(d0 + 5);
    constexpr uint32_t LANE_ARGS = 1;
#else
    const uint32_t t0 = 0;
    constexpr uint32_t LANE_ARGS = 0;
#endif
#if defined(MLA_ATT_OT)
    const auto ot = TensorAccessor(ot_args, get_arg_val<uint32_t>(d0 + 5 + LANE_ARGS));
#endif
    (void)LANE_ARGS;
    const uint32_t arx_base = get_local_cb_interface(CB_ARX).fifo_limit - get_local_cb_interface(CB_ARX).fifo_size;
    const uint32_t arxo_base = get_local_cb_interface(CB_ARXO).fifo_limit - get_local_cb_interface(CB_ARXO).fifo_size;
    const uint32_t ptb = get_tile_size(CB_ARX);   // fp32 stats tile
    const uint32_t otb = get_tile_size(CB_ARXO);  // O tile (fp32, or bf16 with GLM53_MLA_ATT_OBF16)
    constexpr uint32_t OG = 4;                    // O tiles per streamed group
    static_assert(mlaatt::DT % OG == 0, "O groups");
    volatile tt_l1_ptr uint32_t* dato = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DATO));
    // CB_AKEY (scratch, never pushed): [0, 64) histogram / tie count (64-B aligned NoC source), [64, 64 + POS_BYTES)
    // cur_pos (32 rows), [64 + POS_BYTES, ...) the keys
    const uint32_t kbase = get_write_ptr(CB_AKEY);
    volatile tt_l1_ptr uint32_t* hist = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kbase);
    volatile tt_l1_ptr int32_t* pos_l1 = reinterpret_cast<volatile tt_l1_ptr int32_t*>(kbase + 64);
    volatile tt_l1_ptr uint32_t* keys = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kbase + 64 + mladsa::POS_BYTES);
#if defined(MLA_DSA_COMPACT)
    // Stage 14 rev 36: the radix rounds scan a compacted candidate list (the keys that still match the prefix) instead
    // of every key; the final selection pass still reads `keys` (selected = key > T, plus the tie quota), so the
    // selected set is unchanged. CB_AKEY holds MLA_DSA_KEYS more words for it.
    volatile tt_l1_ptr uint32_t* cand = keys + MLA_DSA_KEYS;
#endif
    volatile tt_l1_ptr uint32_t* sel_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEL_SEM));
    const uint64_t hist_dst = get_noc_addr(cx, cy, hist_addr + c * mladsa::HIST_SLOT_BYTES);
    const uint64_t h_sem = get_noc_addr(cx, cy, get_semaphore(H_SEM));
    auto report = [&]() {       // the 64-B slot, then the count (the write lands first: barrier)
        invalidate_l1_cache();  // the RISC's histogram stores before the NoC reads them (mla_common.hpp: the fence)
        noc_async_write(kbase, hist_dst, mladsa::HIST_SLOT_BYTES);
        noc_async_write_barrier();
        noc_semaphore_inc(h_sem, 1);
        noc_async_atomic_barrier();
    };
    uint32_t msgs = 0;

    MK_MARK();
    noc_async_read_page(0, cur, kbase + 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
        const int32_t pos = pos_l1[t];
        const uint32_t p = pos < 0 ? 0u : static_cast<uint32_t>(pos);
        if (mladsa::sparse(p)) {
            // ---- keys of this core's pools < n_closed (local index i -> pool mladsa::pool_of(i, c, C))
            const uint32_t n_closed = (p + 1) / mladsa::KPOOL;
            const uint32_t npt = mladsa::pool_tiles(n_closed, c, C);
            const uint32_t frow = (t / 16) * 512 + (t % 16) * 16;  // row t of an fp32 tile, face 0 (words)
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
            if (npt % mladsa::SB != 0) {  // rev 41: the last score batch's padding (att_c.cpp score pushes SB tiles)
                cb_wait_front(CB_ASC, mladsa::SB - npt % mladsa::SB);
                cb_pop_front(CB_ASC, mladsa::SB - npt % mladsa::SB);
            }
            MK_MARK();  // probe (Stage 14 DSA split): the keys of this core's scores are in (I4 done)
            // ---- I5: radix rounds
            uint32_t prefix = 0, mask = 0;
            volatile tt_l1_ptr uint32_t* src = keys;  // the keys a round scans (rev 36 compaction: the candidates)
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
                    MK_MARK();  // probe (rev 51): this round's histogram reported (16 marks max: t = 1 fits)
                }
                noc_semaphore_wait_min(sel_sem, ++msgs);
                invalidate_l1_cache();
                prefix = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    mbox + (t * mladsa::MSGS + r) * mladsa::MSG_BYTES)[0];
                mask |= (mladsa::BINS - 1) << shift;
#if defined(MLA_DSA_COMPACT)
                if (g == 0) {  // keep the keys that match the new prefix (in place once src is cand: m <= i)
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
#endif
            }
            // ---- tie round: count(key == T) -> quota
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
            MK_MARK();  // probe: the radix rounds + the tie round done (I5 done)
            uint32_t quota = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                mbox + mladsa::QUOTA_OFF + t * mladsa::MAX_CHUNKS * 4)[c];
            // ---- this core's selection -> the BRISC
            cb_reserve_back(CB_ASEL, 1);
            volatile tt_l1_ptr uint32_t* sel = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_ASEL));
            uint32_t n_sel = 0;
            for (uint32_t i = 0; i < nvalid; ++i) {
                const uint32_t k = keys[i];
                bool take = k > prefix;
                if (!take && k == prefix && quota > 0) {
                    --quota;
                    take = true;
                }
                if (take) {
                    sel[mladsa::SEL_HDR + n_sel++] = mladsa::pool_of(i, c, C);
                }
            }
            sel[0] = n_sel;
            sel[1] = c == 0 ? (p + 1) % mladsa::KPOOL : 0;
            sel[2] = n_closed;
            cb_push_back(CB_ASEL, 1);
            MK_MARK();  // probe: this core's selection handed to the BRISC
        }
        // ---- the group's partial tree (kernels/mla/att_dm1.cpp)
        for (uint32_t L = 0; L < LV; ++L) {
            const uint32_t px = get_arg_val<uint32_t>(lv0 + 2 * L), py = get_arg_val<uint32_t>(lv0 + 2 * L + 1);
            if (mlaatt::recv_at(c, L, C)) {
                // streamed message: the stats, then O in groups of OG tiles as the child packs them
                cb_reserve_back(CB_ARX, ST);  // the previous message consumed: the regions are free
                cb_reserve_back(CB_ARXO, DT);
                noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(SEM_RDY)), 1);
                noc_async_atomic_barrier();
                mk::wait_handoff(SEM_DAT, 1);
                cb_push_back(CB_ARX, ST);
                for (uint32_t d0 = 0; d0 < DT; d0 += OG) {
                    noc_semaphore_wait_min(dato, d0 + OG);
                    cb_push_back(CB_ARXO, OG);
                }
                noc_semaphore_set(dato, 0);  // before the next level's ready: that child counts from 0
            } else if (mlaatt::send_at(c, L)) {
                mk::wait_handoff(SEM_RDY, 1);
                cb_wait_front(CB_ASEND, ST);
                noc_async_write(get_read_ptr(CB_ASEND), get_noc_addr(px, py, arx_base), ST * ptb);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(SEM_DAT)), 1);
                cb_pop_front(CB_ASEND, ST);
                const uint64_t dst_o = get_noc_addr(px, py, arxo_base);
                const uint64_t sem_o = get_noc_addr(px, py, get_semaphore(SEM_DATO));
                for (uint32_t d0 = 0; d0 < DT; d0 += OG) {
                    cb_wait_front(CB_ASENDO, d0 + OG);
                    noc_async_write(get_read_ptr(CB_ASENDO) + d0 * otb, dst_o + d0 * otb, OG * otb);
                    noc_async_write_barrier();  // the tiles land before the count that announces them
                    noc_semaphore_inc(sem_o, OG);
                }
                noc_async_atomic_barrier();
                cb_pop_front(CB_ASENDO, DT);
                break;
            }
        }
        if (c == 0) {  // the root's o tiles as the compute packs them
            const uint32_t tb = get_tile_size(CB_AOUT);
            for (uint32_t d = 0; d < mlaatt::DT; ++d) {
                cb_wait_front(CB_AOUT, d + 1);
                noc_async_write_page((t * HPT + g) * mlaatt::DT + d, olat, get_read_ptr(CB_AOUT) + d * tb);
#if defined(MLA_ATT_OT)
                // the tile's head rows -> row t of each head's column tile d of o_T (att_dm1.cpp rev 30): a row whose
                // source offset is not congruent mod 64 with its destination goes through a 64-B staging slot of the
                // root's send CB (the root never sends); slots are reused per tile after the previous tile's writes
                const uint32_t src = get_read_ptr(CB_AOUT) + d * tb;
                const uint32_t drow = mla::row_off(t);
                const uint32_t stg = (get_write_ptr(CB_ASEND) + 63) & ~63u;
                uint64_t dst[HR];
                uint32_t from[HR];
                uint32_t staged = 0;
                noc_async_writes_flushed();
                invalidate_l1_cache();
                for (uint32_t h = 0; h < HR; ++h) {
                    dst[h] = ot.get_noc_addr((g * 32 + h) * mlaatt::DT + d, drow);
                    const uint32_t off = static_cast<uint32_t>(dst[h]) & 63;
                    from[h] = src + mla::row_off(h);
                    if ((from[h] & 63) != off) {
                        const uint32_t slot = stg + 2 * h * 64 + off;
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
#endif
            }
            noc_async_write_barrier();
            cb_pop_front(CB_AOUT, mlaatt::DT);
        }
    }
    if (msgs > 0) {
        noc_semaphore_set(sel_sem, 0);  // every message of this run consumed: back to 0 for the next run
    }
    MK_MARK();
    if constexpr (NCONS > 0) {
        if (c == 0) {
            mk::signal_handoff(SIGNAL_SEM, lv0 + 2 * LV, NCONS);
        }
    }
}
