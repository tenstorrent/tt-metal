// MKA2 writer (BRISC, NOC0), full grid, persistent over NL attention-type layers. Per layer L:
//   STREAM: expose x (producer-fed), land layer L's mixer + MLP weights (trid 1 + (L & 1)), wait the mixer output,
//           bump MIXDONE on the Q root.
//   Q root: wait every STREAM core's mixer (MIXDONE) and the previous layer's SDPA (ADONE), read layer L's Q, multicast
//           it and QFLAG = L + 1 to the full grid.
//   SDPA (all cores): V half of this core's KV slice -> CB_V on NOC0. STREAM cores interleave layer L + 1's weight
//           issue (mixer, odd gate|up groups, down) plus layer L's wo slice in chunks, keeping at most `budget` bytes
//           outstanding (spec v0.3 phase word 13; 0 = issue everything up front).
//   STREAM: wait ADFLAG = L + 1 (all 4 SDPA roots done; multicast by the Q root), wo stand-in + attention AR stall,
//           then the fm11 MLP (h to roots, root broadcast, down output) and the MLP AR stall.
#include "api/debug/kernel_profiler.hpp"
#include "mka2_common.hpp"

void kernel_main() {
    constexpr uint32_t XTB = get_compile_time_arg_val(0), NGU = get_compile_time_arg_val(1);
    constexpr uint32_t NDN = get_compile_time_arg_val(2), KT = get_compile_time_arg_val(3);
    constexpr uint32_t HT = get_compile_time_arg_val(4), WTB = get_compile_time_arg_val(5);
    constexpr uint32_t KB = get_compile_time_arg_val(6), GRP = get_compile_time_arg_val(7);
    constexpr uint32_t HCH = get_compile_time_arg_val(8), NL = get_compile_time_arg_val(9);
    constexpr uint32_t NCORES = get_compile_time_arg_val(10);
    constexpr uint32_t MIXT = get_compile_time_arg_val(11), S_WO = get_compile_time_arg_val(12), S_AR = get_compile_time_arg_val(13);
    constexpr uint32_t KVTB = get_compile_time_arg_val(14), QTB = get_compile_time_arg_val(15), NBB = get_compile_time_arg_val(16);
    constexpr uint32_t WOTB = get_compile_time_arg_val(17);
    constexpr auto wd_args = TensorAccessorArgs<18>();
    constexpr auto wgu_args = TensorAccessorArgs<wd_args.next_compile_time_args_offset()>();
    constexpr auto wm_args = TensorAccessorArgs<wgu_args.next_compile_time_args_offset()>();
    constexpr auto q_args = TensorAccessorArgs<wm_args.next_compile_time_args_offset()>();
    constexpr auto kv_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto wo_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    uint32_t a = 0;
    const uint32_t IS_STREAM = get_arg_val<uint32_t>(a++);
    const uint32_t idx = get_arg_val<uint32_t>(a++);
    Rect r[2];
    for (uint32_t i = 0; i < 2; ++i) {
        r[i].x0 = get_arg_val<uint32_t>(a++); r[i].y0 = get_arg_val<uint32_t>(a++);
        r[i].x1 = get_arg_val<uint32_t>(a++); r[i].y1 = get_arg_val<uint32_t>(a++); r[i].n = get_arg_val<uint32_t>(a++);
    }
    const uint32_t self_rect = get_arg_val<uint32_t>(a++);
    uint32_t rx[NROOT], ry[NROOT];
    for (uint32_t i = 0; i < NROOT; ++i) { rx[i] = get_arg_val<uint32_t>(a++); ry[i] = get_arg_val<uint32_t>(a++); }
    const uint32_t cbh_base = get_arg_val<uint32_t>(a++);
    const uint32_t wd_addr = get_arg_val<uint32_t>(a++), dn_page = get_arg_val<uint32_t>(a++);
    const uint32_t wgu_addr = get_arg_val<uint32_t>(a++), gu_page = get_arg_val<uint32_t>(a++);
    const uint32_t ROOT = get_arg_val<uint32_t>(a++);
    const uint32_t IS_GU = get_arg_val<uint32_t>(a++);
    const uint32_t pgu_layer = get_arg_val<uint32_t>(a++), pdn_layer = get_arg_val<uint32_t>(a++);
    const uint32_t wm_addr = get_arg_val<uint32_t>(a++), wm_page = get_arg_val<uint32_t>(a++), pwm_layer = get_arg_val<uint32_t>(a++);
    uint32_t gs[NGS];
    for (uint32_t i = 0; i < NGS; ++i) gs[i] = get_arg_val<uint32_t>(a++);
    const uint32_t is_qroot = get_arg_val<uint32_t>(a++), q_addr = get_arg_val<uint32_t>(a++);
    uint32_t rf[2][5], rs[2][5];
    for (uint32_t i = 0; i < 2; ++i) for (uint32_t k = 0; k < 5; ++k) rf[i][k] = get_arg_val<uint32_t>(a++);
    const uint32_t self_rf = get_arg_val<uint32_t>(a++);
    for (uint32_t i = 0; i < 2; ++i) for (uint32_t k = 0; k < 5; ++k) rs[i][k] = get_arg_val<uint32_t>(a++);
    uint32_t kv_addr[NL];
    for (uint32_t L = 0; L < NL; ++L) kv_addr[L] = get_arg_val<uint32_t>(a++);
    const uint32_t user = get_arg_val<uint32_t>(a++), vb0 = get_arg_val<uint32_t>(a++), vb1 = get_arg_val<uint32_t>(a++);
    const uint32_t NBK = get_arg_val<uint32_t>(a++);
    const uint32_t wo_addr = get_arg_val<uint32_t>(a++), wo_page = get_arg_val<uint32_t>(a++), wo_tiles = get_arg_val<uint32_t>(a++);
    const uint32_t budget = get_arg_val<uint32_t>(a++), NSTREAM = get_arg_val<uint32_t>(a++);
    const uint32_t qr_x = get_arg_val<uint32_t>(a++), qr_y = get_arg_val<uint32_t>(a++);
    const uint32_t extra_tiles = get_arg_val<uint32_t>(a++);   // qkv bytes the MIXN mixer stand-in does not cover (scratch reads)
    const uint32_t kv_swap = get_arg_val<uint32_t>(a++);
    const uint32_t wcb = kv_swap ? CB_K : CB_V, woff = kv_swap ? 0 : DH;

    const auto wm = TensorAccessor(wm_args, wm_addr, WTB);
    const auto wd = TensorAccessor(wd_args, wd_addr, WTB);
    const auto wgu = TensorAccessor(wgu_args, wgu_addr, WTB);
    const auto qacc = TensorAccessor(q_args, q_addr, QTB);
    const auto wo = TensorAccessor(wo_args, wo_addr, WOTB);
    constexpr uint32_t BLK = KB * 2, NBG = KT / KB, DN_TILES = HT * NDN;
    uint32_t w2_tiles = 0;
    for (uint32_t g0 = GRP; g0 < NBG; g0 += 2 * GRP) w2_tiles += ((NBG - g0) < GRP ? (NBG - g0) : GRP) * BLK;

    {   // ones tile for the softmax denominator (bf16 1.0 = 0x3F80)
        cb_reserve_back(CB_ONE, 1);
        volatile tt_l1_ptr uint32_t* p = l1w(get_write_ptr(CB_ONE));
        for (uint32_t i = 0; i < 2048 / 4; ++i) p[i] = 0x3F803F80u;
        cb_push_back(CB_ONE, 1);
    }

    // ---- next-layer weight issue as a list of chunks (spec v0.3 prefetch budget) ----
    struct Piece { uint64_t src; uint32_t dst, bytes, trid; };
    constexpr uint32_t MAXP = 96, CHB = 16384;
    Piece pcs[MAXP];
    uint32_t npc = 0, ip = 0, outstanding = 0;
    auto add = [&](uint64_t src, uint32_t dst, uint32_t bytes, uint32_t trid, bool scratch) {
        for (uint32_t o = 0; o < bytes && npc < MAXP; o += CHB) {
            const uint32_t n = (bytes - o) < CHB ? (bytes - o) : CHB;
            pcs[npc++] = {src + o, scratch ? dst : dst + o, n, trid};
        }
    };
    auto plan_layer = [&](uint32_t L, bool with_mix) {
        // Reads of layer L's mixer + MLP slices. Called for L = 0 before the loop, and for L + 1 during layer L's SDPA,
        // i.e. AFTER layer L's slots were pushed: each double-buffered CB then has exactly one free slot (layer L - 1
        // popped by the MLP), at the write pointer. (MK1 issued L + 1 before pushing L, hence its 2-slot reserve; doing
        // that here deadlocks, since layer L's slot is only popped by the MLP after this SDPA.)
        const uint32_t trid = 1 + (L & 1);
        if (with_mix) {
            cb_reserve_back(CB_WM, MIXT);
            add(wm.get_noc_addr(wm_page + L * pwm_layer), get_write_ptr(CB_WM), MIXT * WTB, trid, false);
        }
        if (IS_GU) {
            cb_reserve_back(CB_W2, w2_tiles);
            uint32_t dst = get_write_ptr(CB_W2);
            const uint64_t base = wgu.get_noc_addr(gu_page + L * pgu_layer);
            for (uint32_t g0 = GRP; g0 < NBG; g0 += 2 * GRP) {
                const uint32_t nb = (NBG - g0) < GRP ? (NBG - g0) : GRP;
                add(base + (uint64_t)g0 * BLK * WTB, dst, nb * BLK * WTB, trid, false);
                dst += nb * BLK * WTB;
            }
        }
        cb_reserve_back(CB_WD, DN_TILES);
        add(wd.get_noc_addr(dn_page + L * pdn_layer), get_write_ptr(CB_WD), DN_TILES * WTB, trid, false);
    };
    auto plan_wo = [&]() {   // layer L's wo slice + the next layer's uncovered qkv bytes (real bytes into a scratch region)
        for (uint32_t t = 0; t < wo_tiles; t += 8) add(wo.get_noc_addr(wo_page + t), get_write_ptr(CB_SCR), (wo_tiles - t < 8 ? wo_tiles - t : 8) * WOTB, 3, true);
        for (uint32_t t = 0; t < extra_tiles; t += 8) add(wo.get_noc_addr(wo_page + (t % wo_tiles)), get_write_ptr(CB_SCR), (extra_tiles - t < 8 ? extra_tiles - t : 8) * WOTB, 3, true);
    };
    auto issue_some = [&](bool all) {
        // throttled: the previous KV group's interval drained the earlier chunks (wait on the weight trids), then issue
        // at most `budget` bytes for this interval; `all` issues everything left
        if (!all && budget) { for (uint32_t t = 1; t <= 3; ++t) noc_async_read_barrier_with_trid(t); outstanding = 0; }
        while (ip < npc) {
            const Piece& p = pcs[ip];
            if (!all && budget && outstanding > 0 && outstanding + p.bytes > budget) break;
            noc_async_read_set_trid(p.trid);
            noc_async_read(p.src, p.dst, p.bytes);
            noc_async_read_set_trid(0);
            outstanding += p.bytes; ++ip;
        }
    };

    if (IS_STREAM) { plan_layer(0, true); issue_some(true); npc = ip = outstanding = 0; }
    constexpr uint32_t TRID_V = 5;
    for (uint32_t L = 0; L < NL; ++L) {
        const bool att = is_attn(L);
        const uint32_t A = aidx(L);
        if (!IS_STREAM && !att) continue;   // ATTN-only cores idle through GDN-type layers
#if defined(MKA_DBGSPIN)
        if (L > 0) spin_cycles(400000);
#endif
        if (IS_STREAM) {
            DeviceZoneScopedN("W-MIX");
            cb_reserve_back(CB_X, KT);   // producer-fed x (XPRE): re-expose it to compute for this layer
            cb_push_back(CB_X, KT);
            noc_async_read_barrier_with_trid(1 + (L & 1));   // layer L's mixer + MLP weights landed
            cb_push_back(CB_WM, MIXT);
            if (IS_GU) cb_push_back(CB_W2, w2_tiles);
            cb_push_back(CB_WD, DN_TILES);
            cb_wait_front(CB_YM, MIXT / KT);
            cb_pop_front(CB_YM, MIXT / KT);
            if (att) noc_semaphore_inc(get_noc_addr(qr_x, qr_y, gs[GS_MIXDONE]), 1);
        }
        if (!att) {   // GDN-type layer: next layer's weights stream during the GDN stand-in stall (GDN step + out-proj + AR)
            DeviceZoneScopedN("W-GDN");
            npc = ip = outstanding = 0;
            if (L + 1 < NL) plan_layer(L + 1, true);
            issue_some(true);
            spin_cycles(MKA_S_GDN);
            spin_cycles(S_AR);
            cb_reserve_back(CB_GO, 1);
            cb_push_back(CB_GO, 1);
        }
        if (att) 
        {   // SDPA: Q (Q root reads + multicasts, flag = L + 1), then the V stream with the throttled weight issue
            DeviceZoneScopedN("W-SDPA");
            cb_reserve_back(CB_Q, QT);
            const uint32_t qdst = get_write_ptr(CB_Q);
            if (is_qroot) {
                noc_semaphore_wait_min(l1w(gs[GS_MIXDONE]), NSTREAM * (A + 1));
                noc_semaphore_wait_min(l1w(gs[GS_ADONE]), 4 * A);
                for (uint32_t t = 0; t < QT; ++t) noc_async_read(qacc.get_noc_addr(L * QT + t), qdst + t * QTB, QTB);
                noc_async_read_barrier();
                for (uint32_t i = 0; i < 2; ++i) {
                    if (rf[i][4] == 0) continue;
                    const uint64_t m = get_noc_multicast_addr(rf[i][0], rf[i][1], rf[i][2], rf[i][3], qdst);
                    if (i == self_rf) noc_async_write_multicast_loopback_src(qdst, m, QT * QTB, rf[i][4], false);
                    else noc_async_write_multicast(qdst, m, QT * QTB, rf[i][4], false);
                }
                noc_async_writes_flushed();
                mcast_word(gs[GS_QFLAG], A + 1, rf, self_rf, 0);
                noc_async_write_barrier();
            }
            noc_semaphore_wait_min(l1w(gs[GS_QFLAG]), A + 1);
            cb_push_back(CB_Q, QT);
            if (IS_STREAM) {   // plan layer L + 1's mixer + MLP slices and layer L's wo slice
                npc = ip = outstanding = 0;
#if defined(MKA_WO_NOC0)
                plan_wo();
#endif
                if (L + 1 < NL) plan_layer(L + 1, true);
                if (!budget) issue_some(true);
            }
            const auto kv = TensorAccessor(kv_args, kv_addr[L], KVTB);
            for (uint32_t b = vb0; b < vb1; b += NBB) {
                const uint32_t nb = (vb1 - b) < NBB ? (vb1 - b) : NBB;
                cb_reserve_back(wcb, NBB * DH);   // full group (see reader)
                const uint32_t wv = get_write_ptr(wcb);
                noc_async_read_set_trid(TRID_V);
                for (uint32_t i = 0; i < nb; ++i) {
                    const uint32_t pk = ((user * NBK + b + i) * 2) * DH;
                    for (uint32_t t = 0; t < DH; ++t) noc_async_read(kv.get_noc_addr(pk + woff + t), wv + (i * DH + t) * KVTB, KVTB);
                }
                noc_async_read_set_trid(0);
                if (IS_STREAM) issue_some(false);   // one budget's worth of weight chunks per KV group
                noc_async_read_barrier_with_trid(TRID_V);
                cb_push_back(wcb, NBB * DH);
            }
            if (IS_STREAM) issue_some(true);
        }
        if (!IS_STREAM) continue;
        if (att) {   // attention done (all 4 SDPA roots) -> wo stand-in + attention AR -> MLP go
            DeviceZoneScopedN("W-WOAR");
            if (is_qroot) {
                noc_semaphore_wait_min(l1w(gs[GS_ADONE]), 4 * (A + 1));
                mcast_word(gs[GS_ADFLAG], A + 1, rs, self_rect, 0);
                noc_async_write_barrier();
            }
            noc_semaphore_wait_min(l1w(gs[GS_ADFLAG]), A + 1);
#if defined(MKA_WO_NOC0)
            noc_async_read_barrier_with_trid(3);   // layer L's wo slice landed
#endif
            spin_cycles(S_WO);
            spin_cycles(S_AR);
            cb_reserve_back(CB_GO, 1);
            cb_push_back(CB_GO, 1);
        }
#if defined(MKA_NOMLP)
        if (IS_GU) cb_pop_front(CB_W2, w2_tiles);
        cb_pop_front(CB_WD, DN_TILES);
        continue;
#endif
        if (IS_GU) {
            cb_wait_front(CB_HL, 1);
            const uint32_t rt = idx / HCH;
            noc_async_write(get_read_ptr(CB_HL), get_noc_addr(rx[rt], ry[rt], cbh_base + idx * XTB), XTB);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(rx[rt], ry[rt], get_semaphore(SEM_HCNT0 + rt)), 1);
            cb_pop_front(CB_HL, 1);
        }
        if (ROOT < 2) root_bcast_layer(ROOT, L, NCORES, HCH, cbh_base + ROOT * HCH * XTB, HCH * XTB, r, self_rect, 0);
        { DeviceZoneScopedN("W-HWAIT"); wait_hflags(L); }
        cb_reserve_back(CB_H, HT);
        cb_push_back(CB_H, HT);
        cb_wait_front(CB_OUT, NDN);
        cb_pop_front(CB_OUT, NDN);
        { DeviceZoneScopedN("W-ARSTALL"); spin_cycles(S_AR); }
        for (uint32_t i = 0; i < NROOT; ++i) noc_semaphore_inc(get_noc_addr(rx[i], ry[i], get_semaphore(SEM_DONE0 + i)), 1);
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
    // reset for the next pass
    *l1w(gs[GS_QFLAG]) = 0;
    if (IS_STREAM) {
        *l1w(gs[GS_ADFLAG]) = 0;
        for (uint32_t i = 0; i < NROOT; ++i) *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_HFLG0 + i)) = 0;
#if !defined(MKA_NOMLP)
        if (ROOT < 2) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE0 + ROOT)), NCORES * NL);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE0 + ROOT)) = 0;
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_HCNT0 + ROOT)) = 0;
        }
#endif
    }
    if (is_qroot) { *l1w(gs[GS_MIXDONE]) = 0; *l1w(gs[GS_ADONE]) = 0; }
}
