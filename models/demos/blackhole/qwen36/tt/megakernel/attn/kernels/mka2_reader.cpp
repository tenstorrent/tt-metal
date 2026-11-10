// MKA2 reader (NCRISC, NOC1), full grid, persistent over NL attention-type layers. Per layer L:
//   SDPA: K half of this core's KV slice (layer L's cache) -> CB_K; then the fanout-6 pull tree of O_aug partials
//         (global semaphores, cumulative per layer); the group root writes the normalized output and bumps ADONE
//         on the Q root.
//   STREAM, gate|up core: the even gate|up groups of layer L (fm11 engine) -> CB_W.
//   STREAM, NOC1 root (2/3): layer L's h root broadcast.
#if defined(MKA_OPTSZ)   // size experiment (spec §17): optimize the whole TU for size; must precede every include
#if MKA_OPTSZ == 1
#pragma GCC optimize("Os")
#else
#pragma GCC optimize("Oz")
#endif
#endif
#include "api/debug/kernel_profiler.hpp"
#include "mka2_common.hpp"
#if defined(MK_V12)
#include "v12e.hpp"
#endif

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
#if defined(MKA_RQKV)
    constexpr auto wq_args = TensorAccessorArgs<wo_args.next_compile_time_args_offset()>();
#if defined(MKA_RQNR)
    constexpr auto qo_args_r = TensorAccessorArgs<wq_args.next_compile_time_args_offset()>();   // bf16 DRAM tiles
#endif
    constexpr uint32_t MWTB = MKA_MWTB, QSL = 2 * KT;   // STREAM_EXT slice: 2 output tiles x full K, BFP8
    constexpr uint32_t TRID_PQ = 9;
#endif
    uint32_t a = 0;
#if defined(MKA_ROLE_STREAM)
    constexpr uint32_t IS_STREAM = 1; a++;   // per-role binary (spec v0.9.17): EXT-only code compiles out
#elif defined(MKA_ROLE_EXT)
    constexpr uint32_t IS_STREAM = 0; a++;   // per-role binary: STREAM-only (MLP / mixer) code compiles out
#else
    const uint32_t IS_STREAM = get_arg_val<uint32_t>(a++);
#endif
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
#if defined(MKA_KVPROD)
    uint32_t kvv_addr[NL];   // paged_v per layer (kv_addr = paged_k)
    for (uint32_t L = 0; L < NL; ++L) kvv_addr[L] = get_arg_val<uint32_t>(a++);
#endif
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
    // kv_mode: 0 reader K (NOC1) + writer V (NOC0); 1 reader V + writer K; 2 reader K and V (all KV on NOC1);
    // 3 writer K and V (all KV on NOC0)
    const uint32_t kv_mode = get_arg_val<uint32_t>(a++);
#if defined(MKA_TREEFLAG)
    const uint32_t tf_k = get_arg_val<uint32_t>(a++);   // this core's index in its parent's child list
    const uint32_t tfl = get_write_ptr(CB_TF);
    for (uint32_t i = 0; i < 2 * FANOUT; ++i) l1w(tfl)[i] = 0;
#endif
    const bool rd_k = kv_mode == 0 || kv_mode == 2, rd_v = kv_mode == 1 || kv_mode == 2;
    uint32_t kv_groups = 0;
#if defined(MKA_RQKV)
    const uint32_t qrole = get_arg_val<uint32_t>(a++), wq_addr = get_arg_val<uint32_t>(a++);
    const uint32_t pq_page = get_arg_val<uint32_t>(a++), pq_layer = get_arg_val<uint32_t>(a++);
    const auto wq = TensorAccessor(wq_args, wq_addr, MWTB);
#if defined(MKA_RWO)
    const uint32_t rw_nt = get_arg_val<uint32_t>(a++), rw_page = get_arg_val<uint32_t>(a++), rw_pl = get_arg_val<uint32_t>(a++);
    const uint32_t wo_addr2 = get_arg_val<uint32_t>(a++), cc_x = get_arg_val<uint32_t>(a++), cc_y = get_arg_val<uint32_t>(a++);
    // MKA_WOHELP: ext cores read only rw_own pages of their Wo slice; STREAM helpers read the rest into their scratch
    // and forward it into the ext ring (spreads the 8.4 MB DRAM issue over 116 readers, like the stand-in's reads)
    const uint32_t rw_own = get_arg_val<uint32_t>(a++), rw_hlp = get_arg_val<uint32_t>(a++);
    const uint32_t h_x = get_arg_val<uint32_t>(a++), h_y = get_arg_val<uint32_t>(a++), h_page = get_arg_val<uint32_t>(a++);
    const uint32_t h_np = get_arg_val<uint32_t>(a++), h_off = get_arg_val<uint32_t>(a++);
#if defined(MKA_RQNR)
    {   // head constants (gamma x 8, cos, sin): 10 bf16 tiles from DRAM, once per pass
        const uint32_t hg_addr = get_arg_val<uint32_t>(a++), hg_page = get_arg_val<uint32_t>(a++);
        const uint32_t mk_addr = get_arg_val<uint32_t>(a++), mk_page = get_arg_val<uint32_t>(a++);
        if (hg_addr || mk_addr) {   // slots 0-9 head constants, slot 10 this core's causal mask tile
            cb_reserve_back(CB_HG, 11);
            if (hg_addr) {
                const auto hg = TensorAccessor(qo_args_r, hg_addr, 2048);
                for (uint32_t t = 0; t < 10; ++t) noc_async_read(hg.get_noc_addr(hg_page + t), get_write_ptr(CB_HG) + t * 2048, 2048);
            }
            if (mk_addr) {
                const auto mk = TensorAccessor(qo_args_r, mk_addr, 2048);
                noc_async_read(mk.get_noc_addr(mk_page), get_write_ptr(CB_HG) + 10 * 2048, 2048);
            }
            noc_async_read_barrier();
            cb_push_back(CB_HG, 11);
        }
    }
#endif
    const auto wow = TensorAccessor(wq_args, wo_addr2, MWTB);   // Wo pages: same BFP8 interleaved layout as wq
    constexpr uint32_t TRID_WOR = 10;
#endif
    bool pq_pending = false;
#endif   // MKA_RQKV (the v12 args follow the qkv args when present; V12 also runs on the MIXN stand-in)
#if defined(MK_V12)
    // ---- v12 MLP args (appended after every other reader arg) ----
    const uint32_t vrole_rt = get_arg_val<uint32_t>(a++);   // 0 none, 1 STREAM, 2 owner
#if defined(MKA_ROLE_STREAM)
    constexpr uint32_t vrole = 1;
#elif defined(MKA_ROLE_EXT)
    const uint32_t vrole = vrole_rt == 2 ? 2u : 0u;
#else
    const uint32_t vrole = vrole_rt;
#endif
    const uint32_t vk = get_arg_val<uint32_t>(a++), vnn = get_arg_val<uint32_t>(a++), vn = get_arg_val<uint32_t>(a++);
    const uint32_t vcx = get_arg_val<uint32_t>(a++);
    uint32_t vcy[V_NROW];
    for (uint32_t i = 0; i < V_NROW; ++i) vcy[i] = get_arg_val<uint32_t>(a++);
    const uint32_t vrsbuf = get_arg_val<uint32_t>(a++), vhbuf = get_arg_val<uint32_t>(a++);
    const uint32_t vobx = get_arg_val<uint32_t>(a++), voby = get_arg_val<uint32_t>(a++), vpbuf = get_arg_val<uint32_t>(a++);
#if defined(MK_V12_RCT)   // R as a compile-time constant (fixed per deployment): the R < 32 face paths compile out
    constexpr uint32_t vR = MK_V12_RCT; a++;
#else
    const uint32_t vR = get_arg_val<uint32_t>(a++);
#endif
    uint32_t vg[NVG];
    for (uint32_t i = 0; i < NVG; ++i) vg[i] = get_arg_val<uint32_t>(a++);
    // owner x delivery (AR stand-in) on NOC1: west inside the row, no wrap (the BRISC/NOC0 path wraps the torus and
    // ran ~25 us late on some rows under the engine's NOC0 weight traffic)
    const uint32_t voslot = get_arg_val<uint32_t>(a++), voxdst = get_arg_val<uint32_t>(a++), voxpk = get_arg_val<uint32_t>(a++);
    uint32_t vorc[2][5];
    for (uint32_t i = 0; i < 2; ++i) for (uint32_t k = 0; k < 5; ++k) vorc[i][k] = get_arg_val<uint32_t>(a++);
    const uint32_t vmyx = get_arg_val<uint32_t>(a++), vmyy = get_arg_val<uint32_t>(a++);
    // x deliveries per layer (stand-in for the real AR's K-row delivery): MLP only (flag L + 1), or with MK_V12_X2 /
    // MK_RS_QKV two (spec §18: the split-K mixer reads x from CB_XK too): mixer x at layer start (flag 2L + 1, after
    // XRDY = the previous layer's MLP AR) and MLP x at MGO (flag 2L + 2). Consumers wait the flag of their phase.
#if defined(MK_RS_QKV) && !defined(MK_V12_X2)
#define MK_V12_X2 1
#endif
    auto v12_ox_to = [&](uint32_t wait_addr, uint32_t wait_val, uint32_t flag_val) {
        if (vrole != 2) return;
        noc_semaphore_wait_min(vw(wait_addr), wait_val);
        DeviceZoneScopedN("R-OX");
        const uint32_t sp = get_read_ptr(CB_VX0);
        if (vR >= 32) {
            for (uint32_t i = 0; i < 2; ++i)
                if (vorc[i][4]) noc_async_write_multicast(sp, v_mc_addr(vorc[i], voxdst, 1), V_OT * V_PTB, vorc[i][4], false);
        } else {
            const uint32_t pk = get_write_ptr(CB_VPK);
            for (uint32_t t = 0; t < V_OT; ++t)
                for (uint32_t f = 0; f < 2; ++f)
                    noc_async_write(sp + t * V_PTB + f * V_FACE_B, get_noc_addr(vmyx, vmyy, pk + v_pk_off(t, f, vR)), vR * V_ROW_B);
            noc_async_write_barrier();
            for (uint32_t i = 0; i < 2; ++i)
                if (vorc[i][4]) noc_async_write_multicast(pk, v_mc_addr(vorc[i], voxpk, 1), 2 * V_OT * vR * V_ROW_B, vorc[i][4], false);
        }
        noc_async_writes_flushed();
        v_mcast_word(vg[VG_XA + voslot], flag_val, vorc, 1);
        noc_async_write_barrier();
    };
#if defined(MK_V12_X2)
    auto v12_ox = [&](uint32_t L) { v12_ox_to(vg[VG_MGO], L + 1, 2 * L + 2); };                    // MLP x
    auto v12_ox_mix = [&](uint32_t L) { v12_ox_to(gs[GS_XRDY], L, 2 * L + 1); };                   // mixer x
#else
    auto v12_ox = [&](uint32_t L) { v12_ox_to(vg[VG_MGO], L + 1, L + 1); };
#endif
#if !defined(MK_NOBUSYGATE)
    // spec v0.8.9 busy gate: the non-owner ext cores hold their PQGDN chunks while the MLP owners are busy (odd)
    // owner cores never gate on their own word: this RISC delivers the owner's x after its PQ chunks, so a self-gate
    // would deadlock (busy clears only after the partials that need that x)
    auto v_gate = [&]() { if (vrole != 2) while (*vw(vg[VG_BUSY]) & 1u) {} };
#else
    auto v_gate = [&]() {};
#endif
#endif   // MK_V12
#if defined(MKA_RQKV)
    // next attention layer after L (NL if none) and the length of the GDN-type run right before attention layer La
    auto next_att = [](uint32_t L) { uint32_t n = L + 1; while (n < NL && !is_attn(n)) ++n; return n; };
    auto gdn_run = [](uint32_t La) { uint32_t r = 0; while (r < La && !is_attn(La - 1 - r)) ++r; return r; };
#if defined(MKA_PQGDN)
    constexpr bool kPqGdn = true;    // spec v0.8.1: the slice is streamed during the GDN-type layers before its layer
#else
    constexpr bool kPqGdn = false;
#endif
#ifndef MKA_PQCH
#define MKA_PQCH 16
#endif
#ifndef MKA_PQN
#define MKA_PQN 2
#endif
    const uint32_t La0 = is_attn(0) ? 0 : next_att(0);
    if (qrole == QR_EXT && La0 < NL && !(kPqGdn && gdn_run(La0) > 0)) {   // the first attention layer's slice up front
        cb_reserve_back(CB_PQ, QSL);
#if defined(MKA_QSK)
        noc_async_read(wow.get_noc_addr(rw_page + La0 * rw_pl), get_write_ptr(CB_PQ), rw_nt * WK * MWTB);
#else
        noc_async_read(wq.get_noc_addr(pq_page + La0 * pq_layer), get_write_ptr(CB_PQ), QSL * MWTB);
#endif
        noc_async_read_barrier();
        cb_push_back(CB_PQ, QSL);
    }
#if defined(MKA_RWO) && defined(MKA_WORING)
    if (rw_nt && La0 < NL && !(kPqGdn && gdn_run(La0) > 0)) {
        cb_reserve_back(CB_WOR, WOR_SLOT);
        noc_async_read(wow.get_noc_addr(rw_page + La0 * rw_pl), get_write_ptr(CB_WOR), rw_nt * WK * MWTB);
        noc_async_read_barrier();
        cb_push_back(CB_WOR, WOR_SLOT);
    }
#endif
#endif
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
#if defined(MK_V12) && defined(MK_V12_X2)
        v12_ox_mix(L);   // owners: this layer's mixer x (the AR stand-in, at the layer boundary)
#endif
        const uint32_t A = aidx(L);
#if defined(MKA_RQKV)
        if (!IS_STREAM && !att && kPqGdn && qrole == QR_EXT) {
            // STREAM_EXT in a GDN-type layer (spec v0.8.1): this layer's share of the next attention layer's slice,
            // throttled (MKA_PQCH-page chunks, MKA_PQN in flight); the ring was popped by that core's last qkv
            const uint32_t Ln = next_att(L);
            if (Ln < NL) {
                DeviceZoneScopedN("R-PQGDN");
                const uint32_t run = gdn_run(Ln), j = L - (Ln - run);
#if defined(MKA_QSK)
                const uint32_t pqn = rw_nt * WK;   // §18: the ext ring is free of qkv -> it carries the Wo slice (WORING-like)
                const uint64_t src = wow.get_noc_addr(rw_page + Ln * rw_pl);
#else
                const uint32_t pqn = QSL;
                const uint64_t src = wq.get_noc_addr(pq_page + Ln * pq_layer);
#endif
                const uint32_t p0 = j * pqn / run, p1 = (j + 1) * pqn / run;
                if (L > 0) noc_semaphore_wait_min(l1w(gs[GS_XRDY]), L);   // layer L has started
#if defined(MKA_PQDELAY)
                spin_cycles(MKA_PQDELAY);
#endif
                if (j == 0) cb_reserve_back(CB_PQ, QSL);
                const uint32_t dst = get_write_ptr(CB_PQ);
                noc_async_read_set_trid(TRID_PQ);
                for (uint32_t p = p0, c = 0; p < p1; p += MKA_PQCH, ++c) {
                    if (c && (c % MKA_PQN) == 0) { noc_async_read_barrier_with_trid(TRID_PQ); noc_async_read_set_trid(TRID_PQ); }
#if defined(MK_V12)
                    if ((c % MKA_PQN) == 0) v_gate();
#endif
                    const uint32_t n = (p1 - p) < MKA_PQCH ? (p1 - p) : MKA_PQCH;
                    noc_async_read(src + p * MWTB, dst + p * MWTB, n * MWTB);
                }
                noc_async_read_set_trid(0);
                noc_async_read_barrier_with_trid(TRID_PQ);
                if (j + 1 == run) cb_push_back(CB_PQ, QSL);
#if defined(MKA_RWO) && defined(MKA_WORING)
                if (rw_nt) {   // the next attention layer's Wo slice, same share / throttle (ring popped by the last wo)
                    const uint32_t wn = rw_nt * WK, w0 = j * wn / run, w1 = (j + 1) * wn / run;
                    if (j == 0) cb_reserve_back(CB_WOR, WOR_SLOT);
                    const uint64_t wsrc = wow.get_noc_addr(rw_page + Ln * rw_pl);
                    const uint32_t wdst = get_write_ptr(CB_WOR);
                    noc_async_read_set_trid(TRID_PQ);
                    for (uint32_t p = w0, c = 0; p < w1; p += MKA_PQCH, ++c) {
                        if (c && (c % MKA_PQN) == 0) { noc_async_read_barrier_with_trid(TRID_PQ); noc_async_read_set_trid(TRID_PQ); }
                        const uint32_t n = (w1 - p) < MKA_PQCH ? (w1 - p) : MKA_PQCH;
                        noc_async_read(wsrc + p * MWTB, wdst + p * MWTB, n * MWTB);
                    }
                    noc_async_read_set_trid(0);
                    noc_async_read_barrier_with_trid(TRID_PQ);
                    if (j + 1 == run) cb_push_back(CB_WOR, WOR_SLOT);
                }
#endif
            }
#if defined(MK_V12)
            v12_ox(L);
#endif
            continue;
        }
#endif
        if (!IS_STREAM && !att) {   // ATTN-only cores idle through GDN-type layers
#if defined(MK_V12)
            v12_ox(L);
#endif
            continue;
        }
#if defined(MKA_DBGSPIN)
        if (L > 0) spin_cycles(400000);
#endif
#if defined(MKA_GUPRE) && !defined(MK_V12)
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
#if defined(MKA_QSK)
        if (att && vrole == 1) {   // spec §18: this core's qkv partial to the column rows above (NOC1); the writer does rows below
            DeviceZoneScopedN("R-QRS");
            cb_wait_front(CB_VA, V_NTM);
            *vw(vg[VG_LA]) = 2 * L + 1;
            v_rs_send(get_read_ptr(CB_VA), 0, vk, qsk_nt(vn), 1, vcy, vcx, vrsbuf, vk, vR, vg[VG_RS]);
            noc_semaphore_wait_min(vw(vg[VG_LB]), 2 * L + 1);
            cb_pop_front(CB_VA, V_NTM);
        }
#endif
#if defined(MKA_RQKV) && defined(MKA_PQPOST)
        if (pq_pending) { noc_async_read_barrier_with_trid(TRID_PQ); cb_push_back(CB_PQ, QSL); pq_pending = false; }
#endif
        if (att) {   // SDPA: K stream
            DeviceZoneScopedN("R-KV");
            const auto kv = TensorAccessor(kv_args, kv_addr[L], KVTB);
#if defined(MKA_KVPROD)
            const auto kvv = TensorAccessor(kv_args, kvv_addr[L], KVTB);
            const uint32_t kvn = MKA_BPU;
#else
            const auto& kvv = kv;
            const uint32_t kvn = NBK;
#endif
            #if defined(MKA_EXTPOST)
            const uint32_t wtot = IS_STREAM ? wo_tiles + extra_tiles : 0;   // STREAM_EXT cores read theirs after the tree
#else
            const uint32_t wtot = wo_tiles + extra_tiles;   // STREAM, and STREAM_EXT (ATTN-only) cores under MKA_EXT (spec v0.5)
#endif
            const uint32_t ngrp = (b1 - b0 + NBB - 1) / NBB;
            const uint32_t wper = ngrp ? (wtot + ngrp - 1) / ngrp : wtot;   // weight tiles per K group, spread evenly
            uint32_t wdone = 0;
#if defined(MKA_RQKV)
            // STREAM_EXT: layer L + 1's slice streams in from the 3rd KV group on (by then compute has run layer L's
            // qkv, which precedes its SDPA, and freed CB_PQ), spread evenly over the remaining groups
#if defined(MKA_PQPOST)
            const bool pq_go = false;   // issued after this core's tree step instead (see below)
#elif defined(MKA_PQSKIP)
            const bool pq_go = false;
            if (qrole == QR_EXT && next_att(L) < NL) { cb_reserve_back(CB_PQ, QSL); cb_push_back(CB_PQ, QSL); }   // DEBUG: stale slice, no bytes
#else
            // in-window fill of the NEXT ATTENTION layer's slice (with MKA_PQGDN only when no GDN layer precedes it)
            const uint32_t Lq = next_att(L);
#if defined(MKA_QSK)
            const bool pq_go = false;
#else
            const bool pq_go = qrole == QR_EXT && Lq < NL && !(kPqGdn && gdn_run(Lq) > 0);
#endif
#endif
            const uint32_t pq_g0 = ngrp > 2 ? 2 : (ngrp ? ngrp - 1 : 0);
            const uint32_t pq_ng = ngrp > pq_g0 ? ngrp - pq_g0 : 1;
            const uint32_t pq_per = (QSL - PQ_WS + pq_ng - 1) / pq_ng;   // pages per group
            uint32_t pq_done = PQ_WS, gi = 0, pq_dst = 0;
            bool pq_rsv = false;
            const uint64_t pq_src = wq.get_noc_addr(pq_page + next_att(L) * pq_layer);
#endif
#if defined(MKA_RWO)
#if defined(MKA_RWOSKIP)
            const uint32_t wo_np = 0;   // DEBUG timing split: Wo slice not read (stale ring; wo PCC invalid)
#elif defined(MKA_WORING) || defined(MKA_QSK)
            const uint32_t wo_np = 0;   // filled during the GDN layers (own ring CB 31; under QSK the ext ring CB_PQ)
#else
            const uint32_t wo_np = rw_own;   // pages of this core's Wo slice it reads itself (rest: STREAM helpers)
#endif
            // helper (STREAM): h_np pages of an ext core's slice, 8 at a time via CB_SCR, once that ring is free
            // (QFLAG >= A + 1: every qkv producer, ext included, finished, so every ring was popped)
            uint32_t h_done = 0;
            const uint64_t h_src = wow.get_noc_addr(h_page + L * rw_pl);
            auto help_some = [&](bool all) {
                if (!h_np || !att) return;
                if (!all && *l1w(gs[GS_QFLAG]) < A + 1) return;
                if (all) noc_semaphore_wait_min(l1w(gs[GS_QFLAG]), A + 1);
                const uint32_t ring = *l1w(gs[GS_RADDR]);
                while (h_done < h_np) {
                    const uint32_t n = (h_np - h_done) < 8 ? (h_np - h_done) : 8;
                    noc_async_read_set_trid(TRID_WOR + 1);
                    noc_async_read(h_src + h_done * MWTB, scr, n * MWTB);
                    noc_async_read_set_trid(0);
                    noc_async_read_barrier_with_trid(TRID_WOR + 1);
                    noc_async_write(scr, get_noc_addr(h_x, h_y, ring + (h_off + h_done) * MWTB), n * MWTB);
                    noc_async_writes_flushed();   // scratch reused next chunk
                    h_done += n;
                    if (!all) break;   // one chunk per KV group
                }
                if (h_done == h_np) { noc_async_write_barrier(); noc_semaphore_inc(get_noc_addr(h_x, h_y, gs[GS_WOHC]), h_np); }
            };
            const uint32_t wo_per = (wo_np + pq_ng - 1) / pq_ng;
            uint32_t wo_done = 0, wo_dst = 0;
            bool wo_rsv = false;
            const uint64_t wo_src = wow.get_noc_addr(rw_page + L * rw_pl);
#endif
            for (uint32_t b = b0; b < b1; b += NBB) {
                const uint32_t nb = (b1 - b) < NBB ? (b1 - b) : NBB;
#if defined(MKA_RWO)
                if (h_np && h_done < h_np) help_some(false);
                if (rw_nt && gi >= pq_g0 && wo_done < wo_np) {   // the ring is free once this layer's qkv popped it
                    if (!wo_rsv) { cb_reserve_back(CB_PQ, QSL); wo_dst = get_write_ptr(CB_PQ); wo_rsv = true; }
                    const uint32_t n = (wo_np - wo_done) < wo_per ? (wo_np - wo_done) : wo_per;
                    noc_async_read_set_trid(TRID_WOR);
                    noc_async_read(wo_src + wo_done * MWTB, wo_dst + wo_done * MWTB, n * MWTB);
                    noc_async_read_set_trid(0);
                    wo_done += n;
                }
#endif
#if defined(MKA_RQKV)
                if (pq_go && gi >= pq_g0 && pq_done < QSL) {
                    if (!pq_rsv) { cb_reserve_back(CB_PQ, QSL); pq_dst = get_write_ptr(CB_PQ); pq_rsv = true; }
                    const uint32_t n = (QSL - pq_done) < pq_per ? (QSL - pq_done) : pq_per;
                    noc_async_read_set_trid(TRID_PQ);
                    noc_async_read(pq_src + pq_done * MWTB, pq_dst + pq_done * MWTB, n * MWTB);
                    noc_async_read_set_trid(0);
                    pq_done += n;
                }
                ++gi;
#endif
                if (wdone < wtot) {   // previous group's weight reads landed? (they had a whole K group to complete)
                    noc_async_read_barrier_with_trid(TRID_WO);
                    noc_async_read_set_trid(TRID_WO);
                    const uint32_t n = (wtot - wdone) < wper ? (wtot - wdone) : wper;
                    for (uint32_t t = 0; t < n; t += 8) {
                        const uint32_t m = (n - t) < 8 ? (n - t) : 8;
#if defined(MKA_RCAP)
                        // cap this reader's in-flight weight bytes (NoC/DRAM congestion: fewer outstanding reads drain faster)
                        if (t && (t % MKA_RCAP) == 0) { noc_async_read_barrier_with_trid(TRID_WO); noc_async_read_set_trid(TRID_WO); }
#endif
                        noc_async_read(wo.get_noc_addr(wo_page + ((wdone + t) % wo_tiles)), scr, m * WOTB);
                    }
                    noc_async_read_set_trid(0);
                    wdone += n;
                }
                if (!(rd_k || rd_v)) continue;
                // always a full group: CB_K / CB_V hold a whole number of groups, so groups never straddle the ring end
                if (rd_k) cb_reserve_back(CB_K, NBB * DH);
                if (rd_v) cb_reserve_back(CB_V, NBB * DH);
                const uint32_t wk = rd_k ? get_write_ptr(CB_K) : 0, wv = rd_v ? get_write_ptr(CB_V) : 0;
                noc_async_read_set_trid(TRID_K);
                for (uint32_t i = 0; i < nb; ++i) {
                    const uint32_t pk = kv_page(user, b + i, kvn, false), pv = kv_page(user, b + i, kvn, true);
                    if (rd_k) for (uint32_t t = 0; t < DH; ++t) noc_async_read(kv.get_noc_addr(pk + t), wk + (i * DH + t) * KVTB, KVTB);
                    if (rd_v) for (uint32_t t = 0; t < DH; ++t) noc_async_read(kvv.get_noc_addr(pv + t), wv + (i * DH + t) * KVTB, KVTB);
                }
                noc_async_read_set_trid(0);
                noc_async_read_barrier_with_trid(TRID_K);
                if (rd_k) cb_push_back(CB_K, NBB * DH);
                if (rd_v) cb_push_back(CB_V, NBB * DH);
                if (kv_mode == 2) *l1w(gs[GS_KVPROG]) = ++kv_groups;   // paces the writer's weight issue (it has no KV)
            }
            noc_async_read_barrier_with_trid(TRID_WO);
#if defined(MKA_RQKV)
            if (pq_go) {
                if (!pq_rsv) { cb_reserve_back(CB_PQ, QSL); pq_dst = get_write_ptr(CB_PQ); pq_rsv = true; }
                if (pq_done < QSL) {
                    noc_async_read_set_trid(TRID_PQ);
                    noc_async_read(pq_src + pq_done * MWTB, pq_dst + pq_done * MWTB, (QSL - pq_done) * MWTB);
                    noc_async_read_set_trid(0);
                }
                noc_async_read_barrier_with_trid(TRID_PQ);
                if (PQ_WS) noc_semaphore_wait_min(l1w(gs[GS_PQH]), L + 1);   // the writer's share landed
                cb_push_back(CB_PQ, QSL);
            }
#endif
#if defined(MKA_RWO) && !defined(MKA_WORING) && !defined(MKA_QSK)   // QSK: the slice was pushed in the GDN layers
            if (rw_nt) {
                if (!wo_rsv) { cb_reserve_back(CB_PQ, QSL); wo_dst = get_write_ptr(CB_PQ); wo_rsv = true; }
                if (wo_done < wo_np) {
                    noc_async_read_set_trid(TRID_WOR);
                    noc_async_read(wo_src + wo_done * MWTB, wo_dst + wo_done * MWTB, (wo_np - wo_done) * MWTB);
                    noc_async_read_set_trid(0);
                }
                noc_async_read_barrier_with_trid(TRID_WOR);
                if (rw_hlp) noc_semaphore_wait_min(l1w(gs[GS_WOHC]), rw_hlp * (A + 1));   // helpers' pages landed
                cb_push_back(CB_PQ, QSL);   // full slot (straddle-safe ring), wo_np pages valid
            }
            if (h_np && h_done < h_np) help_some(true);
#endif
        }
        if (att) {   // SDPA: tree reduce (pull), cumulative global semaphores (over attention layers: A)
            DeviceZoneScopedN("R-RED");
            auto pull = [&](uint32_t lvl, uint32_t sem_addr, uint32_t n, const uint32_t* cx, const uint32_t* cy) {
                if (n == 0) return;
#if !defined(MKA_TREEFLAG)
                noc_semaphore_wait_min(l1w(sem_addr), n * (A + 1));
#endif
                for (uint32_t i = 0; i < n; ++i) {
#if defined(MKA_TREEFLAG)
                    noc_semaphore_wait_min(l1w(tfl + 4 * (lvl * FANOUT + i)), A + 1);   // child i alone
#endif
                    cb_reserve_back(CB_R, NO);
                    noc_async_read(get_noc_addr(cx[i], cy[i], o_base), get_write_ptr(CB_R), NO * OTB);
                    noc_async_read_barrier();
                    cb_push_back(CB_R, NO);
                    noc_semaphore_inc(get_noc_addr(cx[i], cy[i], gs[GS_DONE]), 1);
                }
            };
            pull(0, gs[GS_LVL0], nc0, c0x, c0y);
            pull(1, gs[GS_LVL1], nc1, c1x, c1y);
            if (has_parent) {
                cb_wait_front(CB_O, NO);
#if defined(MKA_TREEFLAG)
                noc_inline_dw_write(get_noc_addr(par_x, par_y, tfl + 4 * (par_lvl * FANOUT + tf_k)), A + 1);
#else
                noc_semaphore_inc(get_noc_addr(par_x, par_y, par_lvl ? gs[GS_LVL1] : gs[GS_LVL0]), 1);
#endif
                noc_semaphore_wait_min(l1w(gs[GS_DONE]), A + 1);
                cb_pop_front(CB_O, NO);
            }
            if (is_root) {
                cb_wait_front(CB_OA, DH);
                const uint32_t rp = get_read_ptr(CB_OA);
                for (uint32_t t = 0; t < DH; ++t) noc_async_write(rp + t * OATB, out.get_noc_addr((L * NUSR + user) * DH + t), OATB);
#if defined(MKA_RWO)
                {   // O rows (4h + j, head h, candidate j) -> A rows 4u + j, cols h * 256 + t * 32 (tile h * 8 + t): 128 B blocks
                    noc_semaphore_wait_min(l1w(gs[GS_AADDR]), 1);
                    const uint32_t a_base = *l1w(gs[GS_AADDR]);
                    const uint32_t drow = (4 * user) % 16, dface = user < 4 ? 0 : 2;
                    for (uint32_t h = 0; h < 6; ++h)
                        for (uint32_t t = 0; t < DH; ++t)
                            for (uint32_t c = 0; c < 2; ++c) {
                                const uint32_t src = rp + t * OATB + ((h < 4 ? 0 : 2) + c) * 512 + ((4 * h) % 16) * 32;
                                const uint32_t dst = a_base + (h * 8 + t) * 2048 + (dface + c) * 512 + drow * 32;
                                noc_async_write(src, get_noc_addr(cc_x, cc_y, dst), 128);
                            }
                }
#endif
                noc_async_write_barrier();
#if defined(MKA_RWO)
                noc_semaphore_inc(get_noc_addr(cc_x, cc_y, gs[GS_ACNT]), 1);
#endif
                cb_pop_front(CB_OA, DH);
                noc_semaphore_inc(get_noc_addr(qr_x, qr_y, gs[GS_ADONE]), 1);
            }
        }
#if defined(MKA_EXTPOST)
        if (!IS_STREAM && wo_tiles + extra_tiles) {   // STREAM_EXT: projection bytes outside this core's SDPA phase
            DeviceZoneScopedN("R-EXTW");
            const uint32_t n = wo_tiles + extra_tiles;
            noc_async_read_set_trid(TRID_WO);
            for (uint32_t t = 0; t < n; t += 8)
                noc_async_read(wo.get_noc_addr(wo_page + (t % wo_tiles)), scr, ((n - t) < 8 ? (n - t) : 8) * WOTB);
            noc_async_read_set_trid(0);
        }
#endif
#if defined(MKA_RQKV) && defined(MKA_PQPOST)
        // STREAM_EXT: issue layer L + 1's whole slice once this core's KV + tree work is done (the STREAM cores are then
        // in wo / AR / MLP); it lands during that tail and is pushed at the start of layer L + 1
        if (qrole == QR_EXT && next_att(L) < NL) {
            DeviceZoneScopedN("R-PQ");
            cb_reserve_back(CB_PQ, QSL);
            const uint64_t src = wq.get_noc_addr(pq_page + next_att(L) * pq_layer);
            const uint32_t dst = get_write_ptr(CB_PQ);
#if defined(MKA_PQCH)
            // throttled: MKA_PQCH-page chunks, at most MKA_PQN chunks in flight (big single reads from 40 cores flood
            // the NoC and stall the STREAM cores' tail multicasts)
            noc_async_read_set_trid(TRID_PQ);
            for (uint32_t p = 0, c = 0; p < QSL; p += MKA_PQCH, ++c) {
                if (c && (c % MKA_PQN) == 0) { noc_async_read_barrier_with_trid(TRID_PQ); noc_async_read_set_trid(TRID_PQ); }
                const uint32_t n = (QSL - p) < MKA_PQCH ? (QSL - p) : MKA_PQCH;
                noc_async_read(src + p * MWTB, dst + p * MWTB, n * MWTB);
            }
            noc_async_read_set_trid(0);
#else
            noc_async_read_set_trid(TRID_PQ);
            noc_async_read(src, dst, QSL * MWTB);
            noc_async_read_set_trid(0);
#endif
            pq_pending = true;
        }
#endif
#if defined(MKA_GUPRE) && !defined(MK_V12)
        if (att && IS_STREAM && IS_GU) { noc_async_read_barrier_with_trid(TRID_GU); cb_push_back(CB_W, even_t); }
        constexpr bool kGupre = true;
#else
        constexpr bool kGupre = false;
#endif
#if !defined(MKA_NOMLP) && !defined(MK_V12)
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
#if !defined(MKA_NOMLP) && !defined(MK_V12)
        if (IS_STREAM && (ROOT == 2 || ROOT == 3)) {
            DeviceZoneScopedN("R-HROOT");
            root_bcast_layer(ROOT, L, NCORES, HCH, cbh_base + ROOT * HCH * XTB, HCH * XTB, r, self_rect, 1);
        }
#endif
#if defined(MK_V12)
        v12_ox(L);   // owners, attention-type layers (after this core's SDPA work)
        if (vrole == 1) {   // v12 MLP, STREAM core (n, k): RS north, h north, down tiles of the first chunk -> its owner
            cb_wait_front(CB_VA, V_NTM);
            *vw(vg[VG_LA]) = V_LV(L);
            v_rs_send(get_read_ptr(CB_VA), 0, vk < vnn ? vk : vnn, vnn, 2, vcy, vcx, vrsbuf, vk, vR, vg[VG_RS]);   // rows above (NOC1)
            noc_semaphore_wait_min(vw(vg[VG_LB]), V_LV(L));
            cb_pop_front(CB_VA, V_NTM);
            if (vk < vnn) {
                cb_wait_front(CB_VHO2, 1);
                if (vk > 0) { v_hu_send(get_read_ptr(CB_VHO2), 0, vk, vcy, vcx, vhbuf + vk * V_PTB, vR, vg[VG_H]); }
                cb_pop_front(CB_VHO2, 1);
            }
            cb_wait_front(CB_VDO2, V_OT);
            {
                DeviceZoneScopedN("R-VDN");
                v_tile_write(get_read_ptr(CB_VDO2), get_noc_addr(vobx, voby, vpbuf + vn * V_OT * V_PTB), V_OT, vR);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(vobx, voby, vg[VG_P]), 1);
            }
            cb_pop_front(CB_VDO2, V_OT);
        }
#endif
    }
#if !defined(MKA_NOMLP) && !defined(MK_V12)
    if (IS_STREAM && (ROOT == 2 || ROOT == 3)) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE0 + ROOT)), NCORES * NL);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE0 + ROOT)) = 0;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_HCNT0 + ROOT)) = 0;
    }
#endif
    noc_async_read_barrier();
    noc_async_atomic_barrier();
    // reset the tree's global semaphores for the next pass: every inc into them for layer NL - 1 happened before the
    // matching wait above returned, so nothing arrives after this point
    *l1w(gs[GS_LVL0]) = 0; *l1w(gs[GS_LVL1]) = 0; *l1w(gs[GS_DONE]) = 0;
#if defined(MKA_RWO)
    if (rw_hlp) *l1w(gs[GS_WOHC]) = 0;
#endif
#if defined(MKA_RQKV)
    if (qrole == QR_EXT && PQ_WS) *l1w(gs[GS_PQH]) = 0;   // the writer's last set (NL - 2) preceded our last wait
#endif
}
