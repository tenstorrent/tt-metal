// MKA2 reader (NCRISC, NOC1), full grid, persistent over NL attention-type layers. Per layer L:
//   SDPA: K half of this core's KV slice (layer L's cache) -> CB_K; then the fanout-6 pull tree of O_aug partials
//         (global semaphores, cumulative per layer); the group root writes the normalized output and bumps ADONE
//         on the Q root.
//   STREAM, gate|up core: the even gate|up groups of layer L (fm11 engine) -> CB_W.
//   STREAM, NOC1 root (2/3): layer L's h root broadcast.
#include "api/debug/kernel_profiler.hpp"
#include "mka2_common.hpp"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0), KB = get_compile_time_arg_val(1);
    constexpr uint32_t WTB = get_compile_time_arg_val(2), GRP = get_compile_time_arg_val(3);
    constexpr uint32_t HCH = get_compile_time_arg_val(4), XTB = get_compile_time_arg_val(5);
    constexpr uint32_t NL = get_compile_time_arg_val(6), NCORES = get_compile_time_arg_val(7);
    constexpr uint32_t KVTB = get_compile_time_arg_val(8), OTB = get_compile_time_arg_val(9), OATB = get_compile_time_arg_val(10);
    constexpr uint32_t NBB = get_compile_time_arg_val(11), WOTB = get_compile_time_arg_val(12);
    constexpr auto wgu_args = TensorAccessorArgs<13>();
    constexpr auto kv_args = TensorAccessorArgs<wgu_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    constexpr auto wo_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    uint32_t a = 0;
    const uint32_t IS_STREAM = get_arg_val<uint32_t>(a++);
    const uint32_t wgu_addr = get_arg_val<uint32_t>(a++), gu_page = get_arg_val<uint32_t>(a++);
    Rect r[2];
    for (uint32_t i = 0; i < 2; ++i) {
        r[i].x0 = get_arg_val<uint32_t>(a++); r[i].y0 = get_arg_val<uint32_t>(a++);
        r[i].x1 = get_arg_val<uint32_t>(a++); r[i].y1 = get_arg_val<uint32_t>(a++); r[i].n = get_arg_val<uint32_t>(a++);
    }
    const uint32_t self_rect = get_arg_val<uint32_t>(a++);
    const uint32_t cbh_base = get_arg_val<uint32_t>(a++);
    const uint32_t ROOT = get_arg_val<uint32_t>(a++);   // 2/3: NOC1 root; else 0xFF
    const uint32_t IS_GU = get_arg_val<uint32_t>(a++);
    const uint32_t pgu_layer = get_arg_val<uint32_t>(a++);
    uint32_t gs[NGS];
    for (uint32_t i = 0; i < NGS; ++i) gs[i] = get_arg_val<uint32_t>(a++);
    uint32_t kv_addr[NL];
    for (uint32_t L = 0; L < NL; ++L) kv_addr[L] = get_arg_val<uint32_t>(a++);
    const uint32_t user = get_arg_val<uint32_t>(a++), b0 = get_arg_val<uint32_t>(a++), b1 = get_arg_val<uint32_t>(a++);
    const uint32_t NBK = get_arg_val<uint32_t>(a++);
    const uint32_t has_parent = get_arg_val<uint32_t>(a++), par_x = get_arg_val<uint32_t>(a++), par_y = get_arg_val<uint32_t>(a++);
    const uint32_t par_lvl = get_arg_val<uint32_t>(a++);
    const uint32_t nc0 = get_arg_val<uint32_t>(a++);
    uint32_t c0x[FANOUT], c0y[FANOUT];
    for (uint32_t i = 0; i < FANOUT - 1; ++i) { c0x[i] = get_arg_val<uint32_t>(a++); c0y[i] = get_arg_val<uint32_t>(a++); }
    const uint32_t nc1 = get_arg_val<uint32_t>(a++);
    uint32_t c1x[FANOUT], c1y[FANOUT];
    for (uint32_t i = 0; i < FANOUT - 1; ++i) { c1x[i] = get_arg_val<uint32_t>(a++); c1y[i] = get_arg_val<uint32_t>(a++); }
    const uint32_t is_root = get_arg_val<uint32_t>(a++), out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t qr_x = get_arg_val<uint32_t>(a++), qr_y = get_arg_val<uint32_t>(a++);
    // STREAM: layer L's wo slice + the next layer's uncovered qkv bytes stream on NOC1 here (scratch), balancing the NoCs
    const uint32_t wo_addr = get_arg_val<uint32_t>(a++), wo_page = get_arg_val<uint32_t>(a++), wo_tiles = get_arg_val<uint32_t>(a++);
    const uint32_t extra_tiles = get_arg_val<uint32_t>(a++);
    const uint32_t kv_swap = get_arg_val<uint32_t>(a++);   // 1: this core's reader streams V (NOC1) and its writer K (NOC0)
    const uint32_t rcb = kv_swap ? CB_V : CB_K, roff = kv_swap ? DH : 0;
    const auto wo = TensorAccessor(wo_args, wo_addr, WOTB);
    const uint32_t scr = get_write_ptr(CB_SCR);
    constexpr uint32_t TRID_WO = 7;

    const auto wgu = TensorAccessor(wgu_args, wgu_addr, WTB);
    const auto out = TensorAccessor(out_args, out_addr, OATB);
    constexpr uint32_t BLK = KB * 2, NBG = KT / KB;
    const uint32_t o_base = get_read_ptr(CB_O);
    constexpr uint32_t TRID_K = 6;
    for (uint32_t L = 0; L < NL; ++L) {
        const bool att = is_attn(L);
        const uint32_t A = aidx(L);
        if (!IS_STREAM && !att) continue;   // ATTN-only cores idle through GDN-type layers
#if defined(MKA_DBGSPIN)
        if (L > 0) spin_cycles(400000);
#endif
#if defined(MKA_GUPRE)
        constexpr uint32_t TRID_GU = 8;
        uint32_t even_t = 0;
        if (att && IS_STREAM && IS_GU) {   // layer L's even gate|up groups land in L1 during SDPA (CB_W holds the whole even half)
            for (uint32_t g0 = 0; g0 < NBG; g0 += 2 * GRP) even_t += ((NBG - g0) < GRP ? (NBG - g0) : GRP) * BLK;
            cb_reserve_back(CB_W, even_t);
            const uint64_t base = wgu.get_noc_addr(gu_page + L * pgu_layer);
            uint32_t dst = get_write_ptr(CB_W);
            noc_async_read_set_trid(TRID_GU);
            for (uint32_t g0 = 0; g0 < NBG; g0 += 2 * GRP) {
                const uint32_t nb = (NBG - g0) < GRP ? (NBG - g0) : GRP;
                noc_async_read(base + (uint64_t)g0 * BLK * WTB, dst, nb * BLK * WTB);
                dst += nb * BLK * WTB;
            }
            noc_async_read_set_trid(0);
        }
#endif
        if (att) {   // SDPA: K stream
            DeviceZoneScopedN("R-KV");
            const auto kv = TensorAccessor(kv_args, kv_addr[L], KVTB);
            const uint32_t wtot = IS_STREAM ? wo_tiles + extra_tiles : 0;
            const uint32_t ngrp = (b1 - b0 + NBB - 1) / NBB;
            const uint32_t wper = ngrp ? (wtot + ngrp - 1) / ngrp : wtot;   // weight tiles per K group, spread evenly
            uint32_t wdone = 0;
            for (uint32_t b = b0; b < b1; b += NBB) {
                const uint32_t nb = (b1 - b) < NBB ? (b1 - b) : NBB;
                if (wdone < wtot) {   // previous group's weight reads landed? (they had a whole K group to complete)
                    noc_async_read_barrier_with_trid(TRID_WO);
                    noc_async_read_set_trid(TRID_WO);
                    const uint32_t n = (wtot - wdone) < wper ? (wtot - wdone) : wper;
                    for (uint32_t t = 0; t < n; t += 8) {
                        const uint32_t m = (n - t) < 8 ? (n - t) : 8;
                        noc_async_read(wo.get_noc_addr(wo_page + ((wdone + t) % wo_tiles)), scr, m * WOTB);
                    }
                    noc_async_read_set_trid(0);
                    wdone += n;
                }
                cb_reserve_back(rcb, NBB * DH);   // always a full group: CB_K is a whole number of groups, so groups never straddle the ring end
                const uint32_t wk = get_write_ptr(rcb);
                noc_async_read_set_trid(TRID_K);
                for (uint32_t i = 0; i < nb; ++i) {
                    const uint32_t pk = ((user * NBK + b + i) * 2) * DH;
                    for (uint32_t t = 0; t < DH; ++t) noc_async_read(kv.get_noc_addr(pk + roff + t), wk + (i * DH + t) * KVTB, KVTB);
                }
                noc_async_read_set_trid(0);
                noc_async_read_barrier_with_trid(TRID_K);
                cb_push_back(rcb, NBB * DH);
            }
            noc_async_read_barrier_with_trid(TRID_WO);
        }
        if (att) {   // SDPA: tree reduce (pull), cumulative global semaphores (over attention layers: A)
            DeviceZoneScopedN("R-RED");
            auto pull = [&](uint32_t sem_addr, uint32_t n, const uint32_t* cx, const uint32_t* cy) {
                if (n == 0) return;
                noc_semaphore_wait_min(l1w(sem_addr), n * (A + 1));
                for (uint32_t i = 0; i < n; ++i) {
                    cb_reserve_back(CB_R, NO);
                    noc_async_read(get_noc_addr(cx[i], cy[i], o_base), get_write_ptr(CB_R), NO * OTB);
                    noc_async_read_barrier();
                    cb_push_back(CB_R, NO);
                    noc_semaphore_inc(get_noc_addr(cx[i], cy[i], gs[GS_DONE]), 1);
                }
            };
            pull(gs[GS_LVL0], nc0, c0x, c0y);
            pull(gs[GS_LVL1], nc1, c1x, c1y);
            if (has_parent) {
                cb_wait_front(CB_O, NO);
                noc_semaphore_inc(get_noc_addr(par_x, par_y, par_lvl ? gs[GS_LVL1] : gs[GS_LVL0]), 1);
                noc_semaphore_wait_min(l1w(gs[GS_DONE]), A + 1);
                cb_pop_front(CB_O, NO);
            }
            if (is_root) {
                cb_wait_front(CB_OA, DH);
                const uint32_t rp = get_read_ptr(CB_OA);
                for (uint32_t t = 0; t < DH; ++t) noc_async_write(rp + t * OATB, out.get_noc_addr((L * 4 + user) * DH + t), OATB);
                noc_async_write_barrier();
                cb_pop_front(CB_OA, DH);
                noc_semaphore_inc(get_noc_addr(qr_x, qr_y, gs[GS_ADONE]), 1);
            }
        }
#if defined(MKA_GUPRE)
        if (att && IS_STREAM && IS_GU) { noc_async_read_barrier_with_trid(TRID_GU); cb_push_back(CB_W, even_t); }
        constexpr bool kGupre = true;
#else
        constexpr bool kGupre = false;
#endif
#if !defined(MKA_NOMLP)
        if (IS_STREAM && IS_GU && !(kGupre && att)) {   // GDN-type layers (and non-GUPRE builds) stream the even groups here
            DeviceZoneScopedN("R-GU");
            const uint64_t base = wgu.get_noc_addr(gu_page + L * pgu_layer);
            for (uint32_t g0 = 0; g0 < NBG; g0 += 2 * GRP) {
                const uint32_t nb = (NBG - g0) < GRP ? (NBG - g0) : GRP;
                cb_reserve_back(CB_W, nb * BLK);
                noc_async_read(base + (uint64_t)g0 * BLK * WTB, get_write_ptr(CB_W), nb * BLK * WTB);
                noc_async_read_barrier();
                cb_push_back(CB_W, nb * BLK);
            }
        }
#endif
#if !defined(MKA_NOMLP)
        if (IS_STREAM && (ROOT == 2 || ROOT == 3)) {
            DeviceZoneScopedN("R-HROOT");
            root_bcast_layer(ROOT, L, NCORES, HCH, cbh_base + ROOT * HCH * XTB, HCH * XTB, r, self_rect, 1);
        }
#endif
    }
#if !defined(MKA_NOMLP)
    if (IS_STREAM && (ROOT == 2 || ROOT == 3)) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE0 + ROOT)), NCORES * NL);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE0 + ROOT)) = 0;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_HCNT0 + ROOT)) = 0;
    }
#endif
    noc_async_atomic_barrier();
    // reset the tree's global semaphores for the next pass: every inc into them for layer NL - 1 happened before the
    // matching wait above returned, so nothing arrives after this point
    *l1w(gs[GS_LVL0]) = 0; *l1w(gs[GS_LVL1]) = 0; *l1w(gs[GS_DONE]) = 0;
}
