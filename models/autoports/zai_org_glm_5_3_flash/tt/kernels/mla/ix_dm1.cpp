// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DSA index core (kernels/mla/ix_common.hpp) — NCRISC (NoC1), per active token t (slot t, position p): the slot's ring
// tile row read into CB_RING, row p % 4 replaced by row t of [kn | g] (RISC copies of the 32-B face rows), written
// back, handed to the compute; the pooled key (row 0 of CB_POOL) into row p / 4 of the slot's pooled-key cache (tile
// read, two face rows replaced, written back). Then wsc into its L1 tensor and SIGNAL_SEM on the NCONS score cores.
// Compile-time args: TP_TILES (pooled-key tile rows per slot), SIGNAL_SEM, NCONS, then the TensorAccessorArgs of the
//   ring, the pooled-key cache, cur_pos, wsc.
// Runtime args: ring, pools, cur_pos, wsc addresses, T, then NCONS consumer (x, y) pairs; then the lane (MLA_ATT_LANES,
// rev
//   47: this index core's token lane); then the probe pair (MK_PROBE_ARG).
// MLA_DSA_ONE_SLOT (rev 56, Stage 17's speculative verify: the rows are consecutive positions of ONE user): every row
//   addresses the launch's slot (the ring / pools base = slot s0), in row order on this one index core (each row's ring
//   read sees the previous rows' writes; the owners' I4 waits for all of them). Pool q's ring rows go to ring tile rows
//   4 (q - q0) .. + 3, q0 = the first active row's pool: the open pool's rows are the canonical rows 0..3, a later
//   pool's rows stay off them (a rejected draft in pool q0 + 1 must not overwrite a committed row of pool q0) until
//   Stage 17's SpecCtl ACCEPT moves the next verify's pool to rows 0..3 (kernels/spec/spec_ctl.cpp).
// MLA_DSA_SHARD_LINE (rev 59, the TP8 sequence-sharded split-K: kernels/mla/dsa_shard.hpp): afterwards, once the
//   all-gather landed (AGC_SEM, >= 1), the q tiles of both passes: head tile g of token t = the 8-row q pieces of chips
//   4 g .. 4 g + 3 in y_ag -> q_asm (4 tiles per round trip through the pool-tile scratch), then QASM_SEM += 1 on this
//   core per token (ix_dm0.cpp sends token t's 10th message only after QASM_SEM >= j + 1, j = its lane-local index;
//   rev 60: with token lanes each lane's index core assembles its own tokens). Extra compile-time args: the
//   TensorAccessorArgs of y_ag, q_asm; runtime args after rank, C: y_ag, q_asm, AGC_SEM, QASM_SEM.
// MLA_DSA_SHARD runtime args (rev 60): after the NCONS pairs and the lane (MLA_ATT_LANES): rank, C [, TP8 the above].
// MLA_DSA_ONE_SLOT_BATON (rev 62, the speculative verify on the shard WITH token lanes, dsa/spec_on_shard_design.md
//   section 6): the one slot's ring / pooled-key writes stay in ROW order across the lane index cores -- lane l's j-th
//   row t = l + j LANES waits BATON >= j (+1 for l > 0: its predecessor row t - 1's writes landed), and after its
//   writes increments BATON on the next lane's index core (the last row: DONE on every lane core); every lane signals
//   its owners only after DONE (all rows written, as rev 56's one core). Per layer the counts are taken back by atomic
//   subtraction (an increment of the next layer is never lost). Runtime args after the shard block: the next lane's
//   index core (x, y), then the LANES lane index cores (x, y) in lane order. Semaphores MLA_DSA_BATON_SEM /
//   MLA_DSA_DONE_SEM (defines).
// MLA_DSA_PAGED_POOLS (rev 63, doc/ttnn/12-fast-start/paged_pooled_keys.md rev 3: the padded layout): `pools` = the
// per-chip
//   paged store [nb_local, 1, 32, 128]; local pool q of the token's slot is row q % 16 of page ptab_local[slot][q / 16]
//   (its 64-token block; rows 16..31 unused). Extra compile-time args: the TensorAccessorArgs of the LOCAL page table
//   (after every other accessor); its address at runtime arg MLA_DSA_PT_ARG (a define; after the shard / baton blocks).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/ix_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/mla_common.hpp"
#if defined(MLA_DSA_SHARD)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/dsa_shard.hpp"
#endif

void kernel_main() {
    using namespace mlaix;
    constexpr uint32_t TP_TILES = get_compile_time_arg_val(0);
    constexpr uint32_t SIGNAL_SEM = get_compile_time_arg_val(1);
    constexpr uint32_t NCONS = get_compile_time_arg_val(2);
    constexpr auto r_args = TensorAccessorArgs<3>();
    constexpr auto p_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
#if defined(MLA_DSA_SHARD) && defined(MLA_DSA_SHARD_LINE)
    constexpr auto yag_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto qasm_args = TensorAccessorArgs<yag_args.next_compile_time_args_offset()>();
    constexpr uint32_t CT_END = qasm_args.next_compile_time_args_offset();
#else
    constexpr uint32_t CT_END = w_args.next_compile_time_args_offset();
#endif
#if defined(MLA_DSA_PAGED_POOLS)
#if !defined(MLA_DSA_SHARD) || !defined(MLA_DSA_IX_ROWS) || !defined(MLA_DSA_PT_ARG)
#error "MLA_DSA_PAGED_POOLS: the sharded line's row-granular index writes (the page-table address at MLA_DSA_PT_ARG)"
#endif
    // rev 63: the LOCAL page table (row t = slot t's row; a ONE_SLOT verify's rows repeat the slot's row)
    constexpr auto pt_args = TensorAccessorArgs<CT_END>();
    const auto pt = TensorAccessor(pt_args, get_arg_val<uint32_t>(MLA_DSA_PT_ARG));
#endif
    (void)CT_END;
    const auto ring = TensorAccessor(r_args, get_arg_val<uint32_t>(0));
    const auto pools = TensorAccessor(p_args, get_arg_val<uint32_t>(1));
    const auto cur = TensorAccessor(c_args, get_arg_val<uint32_t>(2));
    const auto wsc = TensorAccessor(w_args, get_arg_val<uint32_t>(3));
    const uint32_t t_rows = get_arg_val<uint32_t>(4);
#if defined(MLA_ATT_LANES)  // rev 47: this token lane's rows only (the lane after the NCONS consumer pairs)
    const uint32_t t0 = get_arg_val<uint32_t>(5 + 2 * NCONS);
    constexpr uint32_t LANE_ARGS = 1;
#else
    const uint32_t t0 = 0;
    constexpr uint32_t LANE_ARGS = 0;
#endif
#if defined(MLA_DSA_SHARD)  // after the NCONS consumer pairs and the lane (rev 60: token lanes; the probe after)
    constexpr uint32_t SH1 = 5 + 2 * NCONS + LANE_ARGS;
    const uint32_t rank = get_arg_val<uint32_t>(SH1), nchips = get_arg_val<uint32_t>(SH1 + 1);
#endif
    (void)LANE_ARGS;
    constexpr uint32_t TB = mla::TILE_BYTES_BF16;

    MK_MARK();
    const uint32_t scr = get_write_ptr(CB_SCR);
    const uint32_t cur_l1 = scr + POOL_TILES * TB;
    ASSERT((cur_l1 & 63) == 0);  // a DRAM read target (rev 43)
    noc_async_read_page(0, cur, cur_l1);
    noc_async_read_barrier();
    invalidate_l1_cache();
#if defined(MLA_DSA_IX_ROWS)
    {  // rev 46: CB_RING's rows 4..31 must read as finite zeros (masked out of the pooled-key softmax: 0 x stale = 0)
        const uint32_t base = get_write_ptr(CB_RING);
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < RING_TILES * TB; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, base + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
    }
#endif
#if defined(MLA_DSA_ONE_SLOT)
#if !defined(MLA_DSA_IX_ROWS) || (defined(MLA_ATT_LANES) && !defined(MLA_DSA_ONE_SLOT_BATON))
#error "MLA_DSA_ONE_SLOT: the row-granular index writes on one index core (or the lane baton)"
#endif
    int32_t q0 = -1;  // the first active row's pool: its ring rows are rows 0..3
    for (uint32_t t = 0; t < t_rows && q0 < 0; ++t) {
        const int32_t p = reinterpret_cast<volatile tt_l1_ptr int32_t*>(cur_l1)[t];
        q0 = p < 0 ? -1 : p / 4;
    }
#endif
#if defined(MLA_DSA_ONE_SLOT_BATON)
#if !defined(MLA_DSA_ONE_SLOT) || !defined(MLA_DSA_SHARD) || !defined(MLA_ATT_LANES)
#error "MLA_DSA_ONE_SLOT_BATON: the one-slot verify on the shard with token lanes"
#endif
#if defined(MLA_DSA_SHARD_LINE)
    constexpr uint32_t BT0 = SH1 + 6;  // after rank, C, y_ag, q_asm, AGC, QASM
#else
    constexpr uint32_t BT0 = SH1 + 2;  // after rank, C
#endif
    volatile tt_l1_ptr uint32_t* baton =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(MLA_DSA_BATON_SEM));
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(MLA_DSA_DONE_SEM));
    auto baton_wait = [&](uint32_t t) {  // row t - 1's writes landed (any lane)
        const uint32_t j = (t - t0) / mlaatt::LANES;
        const uint32_t need = t0 == 0 ? j : j + 1;
        if (need > 0) {
            noc_semaphore_wait_min(baton, need);
        }
        invalidate_l1_cache();
    };
    auto baton_pass = [&](uint32_t t) {  // after row t's writes (write barriers done in the row)
        if (t + 1 < t_rows) {
            const uint32_t x = get_arg_val<uint32_t>(BT0), y = get_arg_val<uint32_t>(BT0 + 1);
            noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(MLA_DSA_BATON_SEM)), 1);
        } else {
            for (uint32_t l = 0; l < mlaatt::LANES; ++l) {
                const uint32_t x = get_arg_val<uint32_t>(BT0 + 2 + 2 * l), y = get_arg_val<uint32_t>(BT0 + 3 + 2 * l);
                noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(MLA_DSA_DONE_SEM)), 1);
            }
        }
        noc_async_atomic_barrier();
    };
#endif
    cb_wait_front(CB_ROW, RING_TILES);
    const uint32_t rowl1 = get_read_ptr(CB_ROW);
    auto index_row = [&](uint32_t t) {
        const int32_t p = reinterpret_cast<volatile tt_l1_ptr int32_t*>(cur_l1)[t];
        if (p < 0) {
            return;
        }
        const uint32_t r = static_cast<uint32_t>(p) % 4;
#if defined(MLA_DSA_ONE_SLOT)
        const uint32_t slot = 0;
        const uint32_t grp = static_cast<uint32_t>(p / 4 - q0);  // >= 0: positions rise with the row
        ASSERT(p / 4 >= q0 && 4 * grp + 3 < 32);
        const uint32_t roff = mla::row_off(4 * grp);  // ring tile rows 4 grp .. + 3 (64-B aligned: 128 grp / + 1 KB)
#else
        const uint32_t slot = t;
        const uint32_t roff = 0;
#endif
#if defined(MLA_DSA_IX_ROWS)
        // rev 46: row-granular index writes -- only ring rows 0..3 move (2 x 128 B per tile: face rows 0..3 of faces 0
        // / 1; rows 4..31 of CB_RING stay the zeros written at the start: masked out of the pooled-key softmax), and
        // only row q % 32 of the 4 pooled-key tiles is written (8 x 32 B through 64-B congruent staging slots, no
        // read). Was a full-tile read-modify-write of 8 + 4 tiles per row (~48 KB): 20 us per row, the t = 32 critical
        // path (u42).
        cb_reserve_back(CB_RING, RING_TILES);
        const uint32_t rl1 = get_write_ptr(CB_RING);
        for (uint32_t i = 0; i < RING_TILES; ++i) {
            const uint64_t a = ring.get_noc_addr(slot * RING_TILES + i, roff);
            noc_async_read(a, rl1 + i * TB, 128);
            noc_async_read(a + 512, rl1 + i * TB + 512, 128);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t i = 0; i < RING_TILES; ++i) {
            const uint32_t s = rowl1 + i * TB + mla::row_off(t), d = rl1 + i * TB + mla::row_off(r);
            mla::copy_row32(s, d);
            mla::copy_row32(s + 512, d + 512);
        }
        invalidate_l1_cache();
        for (uint32_t i = 0; i < RING_TILES; ++i) {
            const uint64_t a = ring.get_noc_addr(slot * RING_TILES + i, roff);
            noc_async_write(rl1 + i * TB, a, 128);
            noc_async_write(rl1 + i * TB + 512, a + 512, 128);
        }
        noc_async_write_barrier();
        cb_push_back(CB_RING, RING_TILES);
#if defined(MLA_DSA_SHARD)  // kernels/mla/dsa_shard.hpp: only the owner of the pool writes it, at its local row
        if (mlashard::owner(static_cast<uint32_t>(p), nchips) != rank) {
            cb_wait_front(CB_POOL, POOL_TILES);  // the compute's pooled key is consumed either way
            cb_pop_front(CB_POOL, POOL_TILES);
            return;
        }
        const uint32_t q = mlashard::local_pool(static_cast<uint32_t>(p) / 4, nchips);
#else
        const uint32_t q = static_cast<uint32_t>(p) / 4;
#endif
        const uint32_t stg = (scr + 63) & ~63u;  // 8 staging slots of 64 B in the (unused) pool-tile scratch
#if defined(MLA_DSA_PAGED_POOLS)
        // rev 63 (paged_pooled_keys.md rev 3): pool q = row q % 16 of the page of local block ptab_local[slot][q / 16]
        // (16 pools per 64-token block); the table entry through the 64-B aligned chunk holding it (staging slot 8)
        (void)slot;
        const uint32_t ent = (q / 16) * 4, ptl1 = stg + 8 * 64;
        noc_async_read(pt.get_noc_addr(t, ent & ~63u), ptl1, 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        const uint32_t lb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ptl1 + (ent & 63u))[0];
        const uint32_t tile0 = lb * POOL_TILES, prow = q % 16;
#else
        const uint32_t tile0 = (slot * TP_TILES + q / 32) * POOL_TILES, prow = q % 32;
#endif
        cb_wait_front(CB_POOL, POOL_TILES);
        invalidate_l1_cache();
        const uint32_t pl1 = get_read_ptr(CB_POOL);
        uint64_t dst[2 * POOL_TILES];
        uint32_t from[2 * POOL_TILES];
        for (uint32_t i = 0; i < POOL_TILES; ++i) {
            for (uint32_t h = 0; h < 2; ++h) {
                const uint32_t k = 2 * i + h;
                dst[k] = pools.get_noc_addr(tile0 + i, mla::row_off(prow) + h * 512);
                from[k] = stg + k * 64 + (static_cast<uint32_t>(dst[k]) & 63);
                mla::copy_row32(pl1 + i * TB + h * 512, from[k]);  // row 0 of face h = the pooled key's 16 columns
            }
        }
        invalidate_l1_cache();
        for (uint32_t k = 0; k < 2 * POOL_TILES; ++k) {
            noc_async_write(from[k], dst[k], 32);
        }
        noc_async_write_barrier();
        cb_pop_front(CB_POOL, POOL_TILES);
#else
        cb_reserve_back(CB_RING, RING_TILES);
        const uint32_t rl1 = get_write_ptr(CB_RING);
        for (uint32_t i = 0; i < RING_TILES; ++i) {
            noc_async_read_page(slot * RING_TILES + i, ring, rl1 + i * TB);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t i = 0; i < RING_TILES; ++i) {
            const uint32_t s = rowl1 + i * TB + mla::row_off(t), d = rl1 + i * TB + mla::row_off(r);
            mla::copy_row32(s, d);
            mla::copy_row32(s + 512, d + 512);
        }
        invalidate_l1_cache();
        for (uint32_t i = 0; i < RING_TILES; ++i) {
            noc_async_write_page(slot * RING_TILES + i, ring, rl1 + i * TB);
        }
        noc_async_write_barrier();
        cb_push_back(CB_RING, RING_TILES);
        // pooled-key row p / 4 of the slot
#if defined(MLA_DSA_SHARD)  // kernels/mla/dsa_shard.hpp: only the owner of the pool writes it, at its local row
        if (mlashard::owner(static_cast<uint32_t>(p), nchips) != rank) {
            cb_wait_front(CB_POOL, POOL_TILES);  // the compute's pooled key is consumed either way
            cb_pop_front(CB_POOL, POOL_TILES);
            return;
        }
        const uint32_t q = mlashard::local_pool(static_cast<uint32_t>(p) / 4, nchips);
#else
        const uint32_t q = static_cast<uint32_t>(p) / 4;
#endif
        const uint32_t tile0 = (slot * TP_TILES + q / 32) * POOL_TILES;
        for (uint32_t i = 0; i < POOL_TILES; ++i) {
            noc_async_read_page(tile0 + i, pools, scr + i * TB);
        }
        noc_async_read_barrier();
        cb_wait_front(CB_POOL, POOL_TILES);
        invalidate_l1_cache();
        const uint32_t pl1 = get_read_ptr(CB_POOL);
        for (uint32_t i = 0; i < POOL_TILES; ++i) {
            const uint32_t s = pl1 + i * TB, d = scr + i * TB + mla::row_off(q % 32);
            mla::copy_row32(s, d);
            mla::copy_row32(s + 512, d + 512);
        }
        invalidate_l1_cache();
        for (uint32_t i = 0; i < POOL_TILES; ++i) {
            noc_async_write_page(tile0 + i, pools, scr + i * TB);
        }
        noc_async_write_barrier();
        cb_pop_front(CB_POOL, POOL_TILES);
#endif
    };
    for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
#if defined(MLA_DSA_ONE_SLOT_BATON)
        baton_wait(t);
#endif
        index_row(t);
#if defined(MLA_DSA_ONE_SLOT_BATON)
        baton_pass(t);
#endif
    }
#if defined(MLA_DSA_ONE_SLOT_BATON)
    {  // every row of the slot written (the owners' I4 reads them all); then the per-layer counts taken back
        noc_semaphore_wait_min(done, 1);
        uint32_t mine = 0;
        for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
            ++mine;
        }
        const uint32_t got = t0 == 0 ? (mine > 0 ? mine - 1 : 0) : mine;  // the BATON increments this layer sent here
        if (got > 0) {
            noc_semaphore_inc(
                get_noc_addr(my_x[noc_index], my_y[noc_index], get_semaphore(MLA_DSA_BATON_SEM)), 0u - got);
        }
        noc_semaphore_inc(get_noc_addr(my_x[noc_index], my_y[noc_index], get_semaphore(MLA_DSA_DONE_SEM)), 0u - 1u);
        noc_async_atomic_barrier();
    }
#endif
    MK_MARK();  // probe (rev 45): every row's ring row + pooled key written
    cb_pop_front(CB_ROW, RING_TILES);
    cb_wait_front(CB_W, 1);
    noc_async_write_page(0, wsc, get_read_ptr(CB_W));
    noc_async_write_barrier();
    cb_pop_front(CB_W, 1);
    if constexpr (NCONS > 0) {
        mk::signal_handoff(SIGNAL_SEM, 5, NCONS);
    }
    MK_MARK();
#if defined(MLA_DSA_SHARD) && defined(MLA_DSA_SHARD_LINE)
    {  // the TP8 q assembly (rev 59): both head tiles of every token, once per chip, behind the global merge
        static_assert(mlashard::LINE == 4 && POOL_TILES == 4 && mlashard::PPC % 4 == 0, "4 assembled tiles per trip");
        const auto yag = TensorAccessor(yag_args, get_arg_val<uint32_t>(SH1 + 2));
        const auto qasm = TensorAccessor(qasm_args, get_arg_val<uint32_t>(SH1 + 3));
        const uint32_t agc_sem = get_arg_val<uint32_t>(SH1 + 4), qasm_sem = get_arg_val<uint32_t>(SH1 + 5);
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(agc_sem)), 1);
        MK_MARK();                                               // probe: the all-gather landed
        for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {  // rev 60: this lane's tokens (ix_dm0 counts them)
            for (uint32_t tile = 0; tile < 2; ++tile) {
                for (uint32_t d0 = 0; d0 < mlashard::PPC; d0 += POOL_TILES) {
                    for (uint32_t i = 0; i < POOL_TILES; ++i) {
                        const uint32_t d = d0 + i;
                        for (uint32_t j = 0; j < mlashard::LINE; ++j) {
                            const uint32_t blk = mlashard::ag_blk(tile * mlashard::LINE + j);
                            const uint64_t src =
                                yag.get_noc_addr(mlashard::agx_q_page(t_rows, blk, t, d), mlashard::agx_q_off(d));
                            const uint32_t dst = scr + i * TB + mlashard::slice_off(j, TB);
                            noc_async_read(src, dst, 256);                 // rows 8 j .. of face 0 / 2
                            noc_async_read(src + 256, dst + TB / 4, 256);  // face 1 / 3
                        }
                    }
                    noc_async_read_barrier();
                    for (uint32_t i = 0; i < POOL_TILES; ++i) {
                        noc_async_write_page(mlashard::qasm_page(t_rows, tile, t, d0 + i), qasm, scr + i * TB);
                    }
                    noc_async_write_barrier();  // (the scratch is reused by the next trip)
                }
            }
            noc_semaphore_inc(get_noc_addr(get_semaphore(qasm_sem)), 1);  // this core's ix_dm0: token t is in q_asm
            noc_async_atomic_barrier();
        }
        MK_MARK();  // probe: the q tiles assembled
    }
#endif
#if defined(MK_PROBE_ARG)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB) + 64);
#endif
}
