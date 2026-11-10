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
#if defined(MK_V12)
#include "v12e.hpp"
#endif

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
#if defined(MKA_RQKV)
    constexpr auto qo_args = TensorAccessorArgs<wo_args.next_compile_time_args_offset()>();
    constexpr uint32_t MWTB = MKA_MWTB, QOTB = 2048, QN = 112;   // qkv weight page (BFP8), output page (bf16), output tiles
#else
    constexpr uint32_t MWTB = WTB;
#endif
#if defined(MK_V12)
#if !defined(MKA_RQKV)
#error "MK_V12 needs MKA_RQKV (CB 20 rstd stand-in shared with the qkv PROJ)"
#endif
    constexpr auto wv_args = TensorAccessorArgs<qo_args.next_compile_time_args_offset()>();
#endif
    uint32_t a = 0;
#if defined(MKA_ROLE_STREAM)
    constexpr uint32_t IS_STREAM = 1; a++;   // per-role binary (spec v0.9.17): EXT-only code compiles out
#elif defined(MKA_ROLE_EXT)
    constexpr uint32_t IS_STREAM = 0; a++;   // per-role binary: STREAM-only (MLP / mixer) code compiles out
#else
    const uint32_t IS_STREAM = get_arg_val<uint32_t>(a++);
#endif
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
    uint32_t rf[NRF][5], rs[2][5];
    for (uint32_t i = 0; i < NRF; ++i) for (uint32_t k = 0; k < 5; ++k) rf[i][k] = get_arg_val<uint32_t>(a++);
    const uint32_t self_rf = get_arg_val<uint32_t>(a++);
    for (uint32_t i = 0; i < 2; ++i) for (uint32_t k = 0; k < 5; ++k) rs[i][k] = get_arg_val<uint32_t>(a++);
    uint32_t kv_addr[NL];
    for (uint32_t L = 0; L < NL; ++L) kv_addr[L] = get_arg_val<uint32_t>(a++);
#if defined(MKA_KVPROD)
    uint32_t kvv_addr[NL];   // paged_v per layer (kv_addr = paged_k)
    for (uint32_t L = 0; L < NL; ++L) kvv_addr[L] = get_arg_val<uint32_t>(a++);
#endif
    const uint32_t user = get_arg_val<uint32_t>(a++), vb0 = get_arg_val<uint32_t>(a++), vb1 = get_arg_val<uint32_t>(a++);
    const uint32_t NBK = get_arg_val<uint32_t>(a++);
    const uint32_t wo_addr = get_arg_val<uint32_t>(a++), wo_page = get_arg_val<uint32_t>(a++), wo_tiles = get_arg_val<uint32_t>(a++);
    const uint32_t budget = get_arg_val<uint32_t>(a++), NSTREAM = get_arg_val<uint32_t>(a++);
    const uint32_t qr_x = get_arg_val<uint32_t>(a++), qr_y = get_arg_val<uint32_t>(a++);
    const uint32_t extra_tiles = get_arg_val<uint32_t>(a++);   // qkv bytes the MIXN mixer stand-in does not cover (scratch reads)
    const uint32_t kv_mode = get_arg_val<uint32_t>(a++);   // see reader
    const bool wr_k = kv_mode == 1 || kv_mode == 3, wr_v = kv_mode == 0 || kv_mode == 3;
    uint32_t kv_groups = 0;
#if defined(MKA_RQKV)
    const uint32_t qrole = get_arg_val<uint32_t>(a++), qn = get_arg_val<uint32_t>(a++), qnt = get_arg_val<uint32_t>(a++);
    const uint32_t q_ox = get_arg_val<uint32_t>(a++), q_oy = get_arg_val<uint32_t>(a++), qo_addr = get_arg_val<uint32_t>(a++);
    const uint32_t gdn_on = get_arg_val<uint32_t>(a++);   // this core runs the GDN in-proj stand-in in GDN-type layers
#if defined(MKA_RWO)
    // wo: tiles (rot order), concat role, owner chunk, wo-core multicast rects, the 16 owners' coords, debug output
    const uint32_t wo_nt = get_arg_val<uint32_t>(a++);
    uint32_t wo_tile[WO_MAXT];
    for (uint32_t i = 0; i < WO_MAXT; ++i) wo_tile[i] = get_arg_val<uint32_t>(a++);
    const uint32_t is_concat = get_arg_val<uint32_t>(a++), own_chunk = get_arg_val<uint32_t>(a++);
    uint32_t re[3][5];
    for (uint32_t i = 0; i < 3; ++i) for (uint32_t k = 0; k < 5; ++k) re[i][k] = get_arg_val<uint32_t>(a++);
    const uint32_t self_re = get_arg_val<uint32_t>(a++);
    uint32_t ow_x[16], ow_y[16];
    for (uint32_t i = 0; i < 16; ++i) { ow_x[i] = get_arg_val<uint32_t>(a++); ow_y[i] = get_arg_val<uint32_t>(a++); }
    const uint32_t wod_addr = get_arg_val<uint32_t>(a++);
    const auto wod = TensorAccessor(qo_args, wod_addr, 2048);   // debug copy of every owner's CB_XO (bf16 tiles)
    auto mcast_ext = [&](uint32_t src, uint32_t dst, uint32_t bytes) {
        for (uint32_t i = 0; i < 3; ++i) {
            if (re[i][4] == 0) continue;
            const uint64_t m = get_noc_multicast_addr(re[i][0], re[i][1], re[i][2], re[i][3], dst);
            if (i == self_re) noc_async_write_multicast_loopback_src(src, m, bytes, re[i][4], false);
            else noc_async_write_multicast(src, m, bytes, re[i][4], false);
        }
    };
    if (is_concat) {   // A rows of non-own users stay 0; publish the (uniform) CB_A / CB_S addresses to every core
        volatile tt_l1_ptr uint32_t* p = l1w(get_write_ptr(CB_A));
        for (uint32_t i = 0; i < WK * 2048 / 4; ++i) p[i] = 0;
        mcast_word<NRF>(gs[GS_AADDR], get_write_ptr(CB_A), rf, self_rf, 0);
        mcast_word<NRF>(gs[GS_SADDR], get_write_ptr(CB_S), rf, self_rf, 0);
        noc_async_write_barrier();
    }
#endif
    const auto qo = TensorAccessor(qo_args, qo_addr, QOTB);
    constexpr uint32_t QATB = 4096;   // fp32 partial tile
    // the qkv tail of layer L on this core: ship / receive the K-half partial, write the rstd-scaled output tiles
    auto qkv_tail = [&](uint32_t L, uint32_t A) {
        if (qrole == QR_PARTNER) {
            cb_wait_front(CB_QA, 1);
            noc_async_write(get_read_ptr(CB_QA), get_noc_addr(q_ox, q_oy, get_write_ptr(CB_QR)), QATB);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(q_ox, q_oy, gs[GS_QPART]), 1);
            cb_pop_front(CB_QA, 1);
            return;
        }
        if (qrole == QR_OWNER) {
            noc_semaphore_wait_min(l1w(gs[GS_QPART]), A + 1);   // per attention layer
            cb_reserve_back(CB_QR, 1);   // the partner wrote the page in place (1-page CB: write pointer fixed)
            cb_push_back(CB_QR, 1);
        }
        if (qrole == QR_NONE) return;
        cb_wait_front(CB_QO, QO_SLOT ? QO_SLOT : qnt);
        for (uint32_t j = 0; j < qnt; ++j) noc_async_write(get_read_ptr(CB_QO) + j * QOTB, qo.get_noc_addr(L * QN + qn + j), QOTB);
#if defined(MKA_RWO)
        if (qn + qnt > 64) {   // sig(gate) tiles -> every wo core's CB_S slot n - 64
            noc_semaphore_wait_min(l1w(gs[GS_SADDR]), 1);
            const uint32_t s_base = *l1w(gs[GS_SADDR]);
            for (uint32_t j = 0; j < qnt; ++j)
                if (qn + j >= 64) mcast_ext(get_read_ptr(CB_QO) + j * QOTB, s_base + (qn + j - 64) * QOTB, QOTB);
        }
#endif
        noc_async_write_barrier();
        cb_pop_front(CB_QO, QO_SLOT ? QO_SLOT : qnt);
        noc_semaphore_inc(get_noc_addr(qr_x, qr_y, gs[GS_MIXDONE]), 1);
    };
#endif
#if defined(MK_V12)
    // ---- v12 MLP args (appended after every other writer arg) ----
    const uint32_t vrole_rt = get_arg_val<uint32_t>(a++);   // 0 none, 1 STREAM, 2 owner
#if defined(MKA_ROLE_STREAM)
    constexpr uint32_t vrole = 1;
#elif defined(MKA_ROLE_EXT)
    const uint32_t vrole = vrole_rt == 2 ? 2u : 0u;
#else
    const uint32_t vrole = vrole_rt;
#endif
    const uint32_t vk = get_arg_val<uint32_t>(a++), vnn = get_arg_val<uint32_t>(a++), vn = get_arg_val<uint32_t>(a++);
    uint32_t vcol[5];
    for (uint32_t i = 0; i < 5; ++i) vcol[i] = get_arg_val<uint32_t>(a++);
    const uint32_t vcx = get_arg_val<uint32_t>(a++);
    uint32_t vcy[V_NROW];
    for (uint32_t i = 0; i < V_NROW; ++i) vcy[i] = get_arg_val<uint32_t>(a++);
    const uint32_t vhbuf = get_arg_val<uint32_t>(a++), vrsbuf = get_arg_val<uint32_t>(a++);
    const uint32_t voax = get_arg_val<uint32_t>(a++), voay = get_arg_val<uint32_t>(a++), vpbuf = get_arg_val<uint32_t>(a++);
    const uint32_t vR = get_arg_val<uint32_t>(a++), vxpk = get_arg_val<uint32_t>(a++), vmyx = get_arg_val<uint32_t>(a++), vmyy = get_arg_val<uint32_t>(a++);
    const uint32_t vgo = get_arg_val<uint32_t>(a++);
    const uint32_t vgo0x = get_arg_val<uint32_t>(a++), vgo0y = get_arg_val<uint32_t>(a++), vgo1x = get_arg_val<uint32_t>(a++), vgo1y = get_arg_val<uint32_t>(a++);
    const uint32_t vwv_addr = get_arg_val<uint32_t>(a++), vwv_page = get_arg_val<uint32_t>(a++), vpvl = get_arg_val<uint32_t>(a++), vnt = get_arg_val<uint32_t>(a++);
    uint32_t vg[NVG];
    for (uint32_t i = 0; i < NVG; ++i) vg[i] = get_arg_val<uint32_t>(a++);
    // owner fields
    const uint32_t voslot = get_arg_val<uint32_t>(a++), voxdst = get_arg_val<uint32_t>(a++), voxpk = get_arg_val<uint32_t>(a++);
    uint32_t vorc[2][5];
    for (uint32_t i = 0; i < 2; ++i) for (uint32_t k = 0; k < 5; ++k) vorc[i][k] = get_arg_val<uint32_t>(a++);
    uint32_t vrowx[V_NCOL];
    for (uint32_t i = 0; i < V_NCOL; ++i) vrowx[i] = get_arg_val<uint32_t>(a++);
    const uint32_t vrowy = get_arg_val<uint32_t>(a++), vbmirror = get_arg_val<uint32_t>(a++);
    uint32_t vbrc[2][5];
    for (uint32_t i = 0; i < 2; ++i) for (uint32_t k = 0; k < 5; ++k) vbrc[i][k] = get_arg_val<uint32_t>(a++);
    const auto wv = TensorAccessor(wv_args, vwv_addr, WTB);
    const uint32_t vabase = vrole == 1 ? get_write_ptr(CB_VA) : 0;
    if (vrole) { cb_reserve_back(CB_VI, 1); cb_push_back(CB_VI, 1); }
    if (vrole == 1) { cb_reserve_back(CB_VJ, 1); cb_push_back(CB_VJ, 1); }
    if (vrole == 2) { cb_reserve_back(CB_VX0, V_OT); cb_push_back(CB_VX0, V_OT); }
    // owner phase of layer L (stand-in AR): x chunk to the K-row, then the column-partial sum and its release
    auto v12_owner = [&](uint32_t L) {
        if (vrole != 2) return;
        // busy only from MGO on (the pre-MLP AR is done): the same core's reader holds its PQGDN chunks while busy is
        // odd and delivers this layer's x after them, so raising busy earlier deadlocks (x -> partials -> clear)
        { DeviceZoneScopedN("W-OGO"); noc_semaphore_wait_min(vw(vg[VG_MGO]), L + 1); }
        v_busy(vg[VG_BUSY], 2 * L + 1, vbmirror, vbrc);   // x delivery (stand-in) is on the reader (NOC1)
        { DeviceZoneScopedN("W-OP"); noc_semaphore_wait_min(vw(vg[VG_P]), V_NCOL * (L + 1)); }
        cb_reserve_back(CB_VP, V_NCOL * V_OT);
        cb_push_back(CB_VP, V_NCOL * V_OT);
        cb_wait_front(CB_VXO, V_OT);
        {
            DeviceZoneScopedN("W-OREL");   // stand-in release: the K-row's STREAM cores may enter the MLP AR stall
#if defined(MK_V12_RELUNI)
            for (uint32_t i = 0; i < V_NCOL; ++i) noc_semaphore_inc(get_noc_addr(vrowx[i], vrowy, vg[VG_MDONE]), 1);
#else
            for (uint32_t i = 0; i < 2; ++i)
                if (vorc[i][4]) noc_semaphore_inc_multicast(v_mc_addr(vorc[i], vg[VG_MDONE], 0), 1, vorc[i][4]);
#endif
            noc_async_atomic_barrier();
        }
        // AR_RECV(L) here (real AR, spec v0.9.2: READY inc to the RS sender, then the receiver phase on this RISC)
        cb_pop_front(CB_VXO, V_OT);
        v_busy(vg[VG_BUSY], 2 * L + 2, vbmirror, vbrc);
    };
#define V12_OWNER(L) v12_owner(L)
#define V12_PLAN_MLP(L) v12_plan_mlp(L)
// v12: the MLP slice is barriered + pushed at the MLP phase itself (it streams through the whole pre-MLP stall)
#define V_MLP_PUSH() do {} while (0)
#define V_MLP_WAIT_PUSH() do {} while (0)
#else
#define V12_OWNER(L) do {} while (0)
#define V12_PLAN_MLP(L) do {} while (0)
#define V_MLP_PUSH() do { if (IS_GU) cb_push_back(CB_W2, w2_tiles); cb_push_back(CB_WD, DN_TILES); } while (0)
#define V_MLP_WAIT_PUSH() do { noc_async_read_barrier_with_trid(1 + (L & 1)); V_MLP_PUSH(); } while (0)
#endif
    const auto wm = TensorAccessor(wm_args, wm_addr, MWTB);
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

#if defined(MKA_MIXTRID) && MKA_MIXTRID
    // gdnext: the mixer slice gets its own transaction ids (6/7 by layer parity), so the mixer is released as soon as
    // ITS slice landed instead of after the layer's whole mixer + gate|up + down (~26 MB) read.
    constexpr uint32_t MIX_TRID0 = 6;
#else
    constexpr uint32_t MIX_TRID0 = 1;   // same trid as the layer's MLP slices (original behaviour)
#endif
    // ---- next-layer weight issue as a list of chunks (spec v0.3 prefetch budget) ----
    struct Piece { uint64_t src; uint32_t dst, bytes, trid; };
#if defined(MK_V12)
    constexpr uint32_t MAXP = 128, CHB = 16384;   // + the 242 KB MLP slice (16 pieces)
#else
    constexpr uint32_t MAXP = 96, CHB = 16384;
#endif
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
#if defined(MKA_RQKV)
        if (with_mix && (qrole == QR_OWNER || qrole == QR_PARTNER) && (is_attn(L) || gdn_on)) {
#else
        if (with_mix) {
#endif
            cb_reserve_back(CB_WM, MIXT);
            add(wm.get_noc_addr(wm_page + L * pwm_layer), get_write_ptr(CB_WM), MIXT * MWTB, MIX_TRID0 + (L & 1), false);
        }
#if defined(MK_V12)
        if (vrole == 1 && L == 0) {   // later layers: issued by the MLP of the previous layer (single-slot CB_WV, L1)
            cb_reserve_back(CB_WV, V_SLOT);
            add(wv.get_noc_addr(vwv_page + L * vpvl), get_write_ptr(CB_WV), vnt * WTB, trid, false);
        }
        if (false) {
#else
        if (IS_GU) {
#endif
            cb_reserve_back(CB_W2, w2_tiles);
            uint32_t dst = get_write_ptr(CB_W2);
            const uint64_t base = wgu.get_noc_addr(gu_page + L * pgu_layer);
            for (uint32_t g0 = GRP; g0 < NBG; g0 += 2 * GRP) {
                const uint32_t nb = (NBG - g0) < GRP ? (NBG - g0) : GRP;
                add(base + (uint64_t)g0 * BLK * WTB, dst, nb * BLK * WTB, trid, false);
                dst += nb * BLK * WTB;
            }
        }
#if !defined(MK_V12)
        cb_reserve_back(CB_WD, DN_TILES);
        add(wd.get_noc_addr(dn_page + L * pdn_layer), get_write_ptr(CB_WD), DN_TILES * WTB, trid, false);
#endif
    };
#if defined(MK_V12)
    // layer L's MLP slice into the single CB_WV slot, as weight pieces (trid of layer L). Queued in layer L itself:
    // GDN-type layers after plan_layer(L + 1)'s mixer pieces in W-GDN (it lands during the GDN + AR stall),
    // attention layers with the SDPA phase's throttled weight issue. Queued straight after the previous MLP it
    // stalled the next layer's mixer issue behind 242 KB per core of reads (W-GDN +35 us on some cores).
    auto v12_plan_mlp = [&](uint32_t L) {
        if (vrole != 1 || L == 0) return;   // layer 0: plan_layer(0)
        cb_reserve_back(CB_WV, V_SLOT);
        add(wv.get_noc_addr(vwv_page + L * vpvl), get_write_ptr(CB_WV), vnt * WTB, 1 + (L & 1), false);
    };
#endif
    auto plan_wo = [&]() {   // layer L's wo slice + the next layer's uncovered qkv bytes (real bytes into a scratch region)
        for (uint32_t t = 0; t < wo_tiles; t += 8) add(wo.get_noc_addr(wo_page + t), get_write_ptr(CB_SCR), (wo_tiles - t < 8 ? wo_tiles - t : 8) * WOTB, 3, true);
        for (uint32_t t = 0; t < extra_tiles; t += 8) add(wo.get_noc_addr(wo_page + (t % wo_tiles)), get_write_ptr(CB_SCR), (extra_tiles - t < 8 ? extra_tiles - t : 8) * WOTB, 3, true);
    };
    auto issue_some = [&](bool all, uint32_t minp = 0) {
        // throttled: the previous KV group's interval drained the earlier chunks (wait on the weight trids), then issue
        // at most `budget` bytes for this interval; `all` issues everything left
        if (!all && budget) {
            for (uint32_t t = 1; t <= 3; ++t) noc_async_read_barrier_with_trid(t);
            if (MIX_TRID0 != 1) { noc_async_read_barrier_with_trid(MIX_TRID0); noc_async_read_barrier_with_trid(MIX_TRID0 + 1); }
            outstanding = 0;
        }
        while (ip < npc) {
            const Piece& p = pcs[ip];
            if (!all && minp == 0 && budget && outstanding > 0 && outstanding + p.bytes > budget) break;
#if defined(MKA_WCAP)
            // cap in-flight weight bytes on this writer at MKA_WCAP even when issuing many pieces at once
            if (outstanding + p.bytes > MKA_WCAP && outstanding > 0) { for (uint32_t t = 1; t <= 3; ++t) noc_async_read_barrier_with_trid(t); outstanding = 0; }
#endif
            if (minp) --minp;
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
        if (!IS_STREAM && !att) { V12_OWNER(L); continue; }   // ATTN-only cores idle through GDN-type layers (owners: v12 MLP)
#if defined(MKA_DBGSPIN)
        if (L > 0) spin_cycles(400000);
#endif
#if defined(MKA_RQKV)
        if (IS_STREAM) {
            // attention layer: layer L's qkv K-half; GDN-type layer: the same K-half matmul is the GDN in-proj stand-in
            DeviceZoneScopedN("W-MIX");
            cb_reserve_back(CB_X, KT);   // producer-fed x (XPRE): re-expose it to compute for this layer
            cb_push_back(CB_X, KT);
            if (att && qrole == QR_OWNER) { cb_reserve_back(CB_RSP, MKA_NOWN); cb_push_back(CB_RSP, MKA_NOWN); }   // partners skip the norm
#if defined(MKA_MIXTRID) && MKA_MIXTRID
            noc_async_read_barrier_with_trid(MIX_TRID0 + (L & 1));   // layer L's qkv / in-proj half landed
            if (qrole && (att || gdn_on)) cb_push_back(CB_WM, MIXT);
            if (att) {
                V_MLP_WAIT_PUSH();
            }
#else
            noc_async_read_barrier_with_trid(1 + (L & 1));   // layer L's qkv half + MLP weights landed
            if (qrole && (att || gdn_on)) cb_push_back(CB_WM, MIXT);
            V_MLP_PUSH();
#endif
            if (att) qkv_tail(L, A);
        } else if (qrole == QR_EXT) {   // STREAM_EXT: x of layer L is there once the previous layer's MLP AR is done
            DeviceZoneScopedN("W-XQKV");
            if (L > 0) noc_semaphore_wait_min(l1w(gs[GS_XRDY]), L);
            cb_reserve_back(CB_X, KT);
            cb_push_back(CB_X, KT);
            cb_reserve_back(CB_RSP, MKA_NOWN);
            cb_push_back(CB_RSP, MKA_NOWN);
            qkv_tail(L, A);
        }
#else
        if (IS_STREAM) {
            DeviceZoneScopedN("W-MIX");
            cb_reserve_back(CB_X, KT);   // producer-fed x (XPRE): re-expose it to compute for this layer
            cb_push_back(CB_X, KT);
#if defined(MKA_MIXTRID) && MKA_MIXTRID
            noc_async_read_barrier_with_trid(MIX_TRID0 + (L & 1));   // layer L's mixer slice landed
            cb_push_back(CB_WM, MIXT);
            if (att) {   // attention layers: plan_layer(L + 1) runs in SDPA, after the MLP slots are pushed here
                V_MLP_WAIT_PUSH();
            }
#else
            noc_async_read_barrier_with_trid(1 + (L & 1));   // layer L's mixer + MLP weights landed
            cb_push_back(CB_WM, MIXT);
            V_MLP_PUSH();
#endif
            cb_wait_front(CB_YM, MIXT / KT);
            cb_pop_front(CB_YM, MIXT / KT);
            if (att) noc_semaphore_inc(get_noc_addr(qr_x, qr_y, gs[GS_MIXDONE]), 1);
        }
#endif
        if (!att) {   // GDN-type layer: next layer's weights stream during the GDN stand-in stall (GDN step + out-proj + AR)
            DeviceZoneScopedN("W-GDN");
#if defined(MKA_MIXTRID) && MKA_MIXTRID
            // GDN-type layer: the MLP slots of layer L are pushed here (before plan_layer(L + 1) reserves the other slot)
            V_MLP_WAIT_PUSH();
#endif
            npc = ip = outstanding = 0;
            if (L + 1 < NL) plan_layer(L + 1, true);
            V12_PLAN_MLP(L);
#if defined(MK_V12)
            {   // pump the pieces through the GDN stall instead of blocking on the command buffer up front (a blocking
                // issue_some held some writers ~20 us past the stall and delayed their MLP)
                const uint32_t t0 = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
                while ((uint32_t)(reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - t0) < (uint32_t)MKA_S_GDN) {
                    if (ip < npc && noc_cmd_buf_ready(noc_index, read_cmd_buf)) {
                        const Piece& pc = pcs[ip++];
                        noc_async_read_set_trid(pc.trid);
                        noc_async_read(pc.src, pc.dst, pc.bytes);
                        noc_async_read_set_trid(0);
                    }
                }
                issue_some(true);
            }
#else
            issue_some(true);
            spin_cycles(MKA_S_GDN);
#endif
            spin_cycles(S_AR);
#if defined(MKA_RQKV)
            if (is_qroot && L + 1 < NL) mcast_word<NRF>(gs[GS_XRDY], L + 1, rf, self_rf, 0);   // layer boundary -> STREAM_EXT
#endif
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
                noc_semaphore_wait_min(l1w(gs[GS_ADONE]), NUSR * A);
                for (uint32_t t = 0; t < QT; ++t) noc_async_read(qacc.get_noc_addr(L * QT + t), qdst + t * QTB, QTB);
                noc_async_read_barrier();
                for (uint32_t i = 0; i < NRF; ++i) {
                    if (rf[i][4] == 0) continue;
                    const uint64_t m = get_noc_multicast_addr(rf[i][0], rf[i][1], rf[i][2], rf[i][3], qdst);
                    if (i == self_rf) noc_async_write_multicast_loopback_src(qdst, m, QT * QTB, rf[i][4], false);
                    else noc_async_write_multicast(qdst, m, QT * QTB, rf[i][4], false);
                }
                noc_async_writes_flushed();
                mcast_word<NRF>(gs[GS_QFLAG], A + 1, rf, self_rf, 0);
                noc_async_write_barrier();
            }
            noc_semaphore_wait_min(l1w(gs[GS_QFLAG]), A + 1);
            cb_push_back(CB_Q, QT);
            if (IS_STREAM) {   // plan layer L + 1's mixer + MLP slices and layer L's wo slice
                npc = ip = outstanding = 0;
#if defined(MKA_WO_NOC0)
                plan_wo();
#endif
                V12_PLAN_MLP(L);   // needed at the end of THIS layer: before the next layer's mixer pieces
                if (L + 1 < NL) plan_layer(L + 1, true);
                if (!budget) issue_some(true);
            }
            const auto kv = TensorAccessor(kv_args, kv_addr[L], KVTB);
#if defined(MKA_KVPROD)
            const auto kvv = TensorAccessor(kv_args, kvv_addr[L], KVTB);
            const uint32_t kvn = MKA_BPU;
#else
            const auto& kvv = kv;
            const uint32_t kvn = NBK;
#endif
#if defined(MKA_RQKV)
            // STREAM_EXT: the writer's share of layer L + 1's slice (pages [0, PQ_WS)), spread over this core's V groups.
            // Safe from group 0: qkv_tail(L) above waited this layer's qkv output, i.e. compute already popped CB_PQ.
            const bool pqw_go = qrole == QR_EXT && L + 1 < NL && PQ_WS > 0;
            const uint32_t pqw_ng = (vb1 - vb0 + NBB - 1) / NBB;
            const uint32_t pqw_per = pqw_ng ? (PQ_WS + pqw_ng - 1) / pqw_ng : PQ_WS;
            uint32_t pqw_done = 0;
            const uint32_t pqw_dst = get_write_ptr(CB_PQ);   // 1-slot ring: the writer never moves its copy of the pointer
            const uint64_t pqw_src = wm.get_noc_addr(wm_page + (L + 1) * pwm_layer);
#endif
            for (uint32_t b = vb0; b < vb1; b += NBB) {
#if defined(MKA_RQKV)
                if (pqw_go && pqw_done < PQ_WS) {
                    const uint32_t n = (PQ_WS - pqw_done) < pqw_per ? (PQ_WS - pqw_done) : pqw_per;
                    noc_async_read_set_trid(9);
                    noc_async_read(pqw_src + pqw_done * MWTB, pqw_dst + pqw_done * MWTB, n * MWTB);
                    noc_async_read_set_trid(0);
                    pqw_done += n;
                }
#endif
                const uint32_t nb = (vb1 - b) < NBB ? (vb1 - b) : NBB;
#if defined(MKA_SPREAD)
                // spread the planned weight pieces evenly over this core's KV groups (at least `budget` per group):
                // at small KV (u1, short context) a fixed budget left most pieces for the issue-all after the loop
                const uint32_t gl = (vb1 - b + NBB - 1) / NBB;
                const uint32_t minp = (npc - ip + gl - 1) / gl;
#else
                constexpr uint32_t minp = 0;
#endif
                if (!(wr_k || wr_v)) {   // kv_mode 2: the reader streams this core's K and V; pace the weight issue on its progress
                    if (IS_STREAM) issue_some(false, minp);
                    noc_semaphore_wait_min(l1w(gs[GS_KVPROG]), ++kv_groups);
                    continue;
                }
                if (wr_k) cb_reserve_back(CB_K, NBB * DH);   // full group (see reader)
                if (wr_v) cb_reserve_back(CB_V, NBB * DH);
                const uint32_t wk = wr_k ? get_write_ptr(CB_K) : 0, wv = wr_v ? get_write_ptr(CB_V) : 0;
                noc_async_read_set_trid(TRID_V);
                for (uint32_t i = 0; i < nb; ++i) {
                    const uint32_t pk = kv_page(user, b + i, kvn, false), pv = kv_page(user, b + i, kvn, true);
                    if (wr_k) for (uint32_t t = 0; t < DH; ++t) noc_async_read(kv.get_noc_addr(pk + t), wk + (i * DH + t) * KVTB, KVTB);
                    if (wr_v) for (uint32_t t = 0; t < DH; ++t) noc_async_read(kvv.get_noc_addr(pv + t), wv + (i * DH + t) * KVTB, KVTB);
                }
                noc_async_read_set_trid(0);
                if (IS_STREAM) issue_some(false, minp);   // one budget's worth (or the even share) of weight chunks per KV group
                noc_async_read_barrier_with_trid(TRID_V);
                if (wr_k) cb_push_back(CB_K, NBB * DH);
                if (wr_v) cb_push_back(CB_V, NBB * DH);
            }
            if (IS_STREAM) issue_some(true);
#if defined(MKA_RQKV)
            if (pqw_go) {
                if (pqw_done < PQ_WS) {
                    noc_async_read_set_trid(9);
                    noc_async_read(pqw_src + pqw_done * MWTB, pqw_dst + pqw_done * MWTB, (PQ_WS - pqw_done) * MWTB);
                    noc_async_read_set_trid(0);
                }
                noc_async_read_barrier_with_trid(9);
                *l1w(gs[GS_PQH]) = L + 1;   // core-local flag for the reader, which pushes CB_PQ
            }
#endif
        }
#if defined(MKA_RWO)
        if (!IS_STREAM && att && wo_nt) {
            DeviceZoneScopedN("W-WO");
            cb_reserve_back(CB_S, WK);   // sig(gate) landed before QFLAG (producers barrier before MIXDONE)
            cb_push_back(CB_S, WK);
            if (is_concat) {
                noc_semaphore_wait_min(l1w(gs[GS_ACNT]), NUSR * (A + 1));
                mcast_ext(get_write_ptr(CB_A), get_write_ptr(CB_A), WK * 2048);
                noc_async_write_barrier();
                for (uint32_t i = 0; i < 3; ++i) {   // AFLAG (data before flag: same NoC, barrier above)
                    if (re[i][4] == 0) continue;
                    *l1w(gs[GS_AFLAG]) = A + 1;
                    const uint64_t m = get_noc_multicast_addr(re[i][0], re[i][1], re[i][2], re[i][3], gs[GS_AFLAG]);
                    if (i == self_re) noc_semaphore_set_multicast_loopback_src(gs[GS_AFLAG], m, re[i][4], false);
                    else noc_semaphore_set_multicast(gs[GS_AFLAG], m, re[i][4], false);
                }
                noc_async_write_barrier();
            }
            noc_semaphore_wait_min(l1w(gs[GS_AFLAG]), A + 1);
            cb_reserve_back(CB_A, WK);
            cb_push_back(CB_A, WK);
            cb_wait_front(CB_QO, QO_SLOT);   // A_g @ Wo slice (full slot, wo_nt valid)
            const uint32_t xo = get_write_ptr(CB_XO);   // uniform on the ext cores
            uint32_t cnt[16];
            for (uint32_t i = 0; i < 16; ++i) cnt[i] = 0;
            for (uint32_t j = 0; j < wo_nt; ++j) {   // rot order: farthest owning device first
                const uint32_t n = wo_tile[j], ch = n / 10;
                noc_async_write(get_read_ptr(CB_QO) + j * 2048, get_noc_addr(ow_x[ch], ow_y[ch], xo + (n % 10) * 2048), 2048);
                ++cnt[ch];
            }
            noc_async_write_barrier();
            for (uint32_t j = 0; j < wo_nt; ++j) {
                const uint32_t ch = wo_tile[j] / 10;
                if (cnt[ch]) { noc_semaphore_inc(get_noc_addr(ow_x[ch], ow_y[ch], gs[GS_WOCNT]), cnt[ch]); cnt[ch] = 0; }
            }
            cb_pop_front(CB_QO, QO_SLOT);
            if (own_chunk < 16) {   // owner: chunk complete -> READY (AR); bench: debug copy + WODONE on the Q root
                noc_semaphore_wait_min(l1w(gs[GS_WOCNT]), 10 * (A + 1));
                for (uint32_t i = 0; i < 10; ++i) noc_async_write(xo + i * 2048, wod.get_noc_addr(L * WN + own_chunk * 10 + i), 2048);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(qr_x, qr_y, gs[GS_WODONE]), 1);
            }
        }
#endif
        if (!IS_STREAM) { V12_OWNER(L); continue; }
        if (att) {   // attention done (all 4 SDPA roots) -> wo stand-in + attention AR -> MLP go
            DeviceZoneScopedN("W-WOAR");
            if (is_qroot) {
                noc_semaphore_wait_min(l1w(gs[GS_ADONE]), NUSR * (A + 1));
#if defined(MKA_RWO)
                noc_semaphore_wait_min(l1w(gs[GS_WODONE]), 16 * (A + 1));   // every owner chunk of wo complete
#endif
                mcast_word(gs[GS_ADFLAG], A + 1, rs, self_rect, 0);
                noc_async_write_barrier();
            }
            noc_semaphore_wait_min(l1w(gs[GS_ADFLAG]), A + 1);
#if defined(MKA_WO_NOC0)
            noc_async_read_barrier_with_trid(3);   // layer L's wo slice landed
#endif
#if !defined(MKA_RWO)
            spin_cycles(S_WO);
#endif
            spin_cycles(S_AR);
            cb_reserve_back(CB_GO, 1);
            cb_push_back(CB_GO, 1);
        }
#if defined(MK_V12)
        if (vrole == 1) {   // v12 MLP, STREAM core (n, k)
            if (vgo) {   // the pre-MLP AR (stand-in) is done: this K-row's owners deliver x (the real AR does this itself)
                noc_semaphore_inc(get_noc_addr(vgo0x, vgo0y, vg[VG_MGO]), 1);
                noc_semaphore_inc(get_noc_addr(vgo1x, vgo1y, vg[VG_MGO]), 1);
            }
            const bool own_h = vk < vnn;
            { noc_async_read_barrier_with_trid(1 + (L & 1)); }   // layer L's MLP slice landed
            cb_push_back(CB_WV, V_SLOT);
            if (own_h) { cb_reserve_back(CB_VRSA, MKA_NOWN); cb_push_back(CB_VRSA, MKA_NOWN); }
            { DeviceZoneScopedN("W-VX"); noc_semaphore_wait_min(vw(vg[VG_XA]), L + 1); noc_semaphore_wait_min(vw(vg[VG_XB]), L + 1); }
            cb_reserve_back(CB_XK, 2 * V_OT);
            if (vR < 32) {   // unpack the two packed chunks (v0.8.2) into tile layout
                DeviceZoneScopedN("W-VXU");
                const uint32_t xd = get_write_ptr(CB_XK);
                for (uint32_t c = 0; c < 2; ++c)
                    for (uint32_t t = 0; t < V_OT; ++t)
                        for (uint32_t f = 0; f < 2; ++f)
                            noc_async_write(vxpk + c * V_XPK_SLOT_B + v_pk_off(t, f, vR), get_noc_addr(vmyx, vmyy, xd + (c * V_OT + t) * V_PTB + f * V_FACE_B), vR * V_ROW_B);
                noc_async_write_barrier();
            }
            cb_push_back(CB_XK, 2 * V_OT);
            noc_semaphore_wait_min(vw(vg[VG_LA]), L + 1);
            if (vk + 1 < vnn) { v_rs_send(vabase, vk + 1, vnn, vcy, vcx, vrsbuf, vk, vR, vg[VG_RS]); }
            *vw(vg[VG_LB]) = L + 1;
            if (own_h) {
                { DeviceZoneScopedN("W-VRW"); noc_semaphore_wait_min(vw(vg[VG_RS]), (V_NROW - 1) * (L + 1)); }
                cb_reserve_back(CB_VRS, 2 * V_NROW);
                cb_push_back(CB_VRS, 2 * V_NROW);
                cb_wait_front(CB_VHO, 1);
                { v_hu_send(get_read_ptr(CB_VHO), vk, V_NROW, vcy, vcx, vhbuf + vk * V_PTB, vR, vg[VG_H]); }
                cb_pop_front(CB_VHO, 1);
            }
            { DeviceZoneScopedN("W-VH"); noc_semaphore_wait_min(vw(vg[VG_H]), vnn * (L + 1)); }
            cb_reserve_back(CB_VH, V_NNM);
            cb_push_back(CB_VH, V_NNM);
            cb_wait_front(CB_VDO, V_OT);
            {
                DeviceZoneScopedN("W-VDN");
                v_tile_write(get_read_ptr(CB_VDO), get_noc_addr(voax, voay, vpbuf + vn * V_OT * V_PTB), V_OT, vR);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(voax, voay, vg[VG_P]), 1);
            }
            cb_pop_front(CB_VDO, V_OT);
            { DeviceZoneScopedN("W-VMD"); noc_semaphore_wait_min(vw(vg[VG_MDONE]), 2 * (L + 1)); }   // both row owners summed
        }
        { DeviceZoneScopedN("W-ARSTALL"); spin_cycles(S_AR); }
#if defined(MKA_RQKV)
        if (is_qroot && L + 1 < NL) mcast_word(gs[GS_XRDY], L + 1, rf, self_rf, 0);   // x of layer L + 1 -> STREAM_EXT
#endif
#else
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
#if defined(MK_DNPIPE) && MK_DNPIPE
        // -mk-tail port: publish h root chunk by root chunk (fixed root order = landing-buffer order) so the down
        // projection starts on chunk 0 while chunks 1-3 are still in flight
        for (uint32_t i = 0; i < NROOT; ++i) {
            DeviceZoneScopedN("W-HWAIT");
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_HFLG0 + i)), L + 1);
            cb_reserve_back(CB_H, HCH);
            cb_push_back(CB_H, HCH);
        }
#else
        { DeviceZoneScopedN("W-HWAIT"); wait_hflags(L); }
        cb_reserve_back(CB_H, HT);
        cb_push_back(CB_H, HT);
#endif
        cb_wait_front(CB_OUT, NDN);
        cb_pop_front(CB_OUT, NDN);
        { DeviceZoneScopedN("W-ARSTALL"); spin_cycles(S_AR); }
#if defined(MKA_RQKV)
        if (is_qroot && L + 1 < NL) mcast_word<NRF>(gs[GS_XRDY], L + 1, rf, self_rf, 0);   // x of layer L + 1 -> STREAM_EXT
#endif
        for (uint32_t i = 0; i < NROOT; ++i) noc_semaphore_inc(get_noc_addr(rx[i], ry[i], get_semaphore(SEM_DONE0 + i)), 1);
#endif
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
    // reset for the next pass
    *l1w(gs[GS_QFLAG]) = 0;
    if (IS_STREAM) {
        *l1w(gs[GS_ADFLAG]) = 0;
        for (uint32_t i = 0; i < NROOT; ++i) *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_HFLG0 + i)) = 0;
#if !defined(MKA_NOMLP) && !defined(MK_V12)
        if (ROOT < 2) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE0 + ROOT)), NCORES * NL);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE0 + ROOT)) = 0;
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_HCNT0 + ROOT)) = 0;
        }
#endif
    }
    if (is_qroot) { *l1w(gs[GS_MIXDONE]) = 0; *l1w(gs[GS_ADONE]) = 0; }
#if defined(MK_V12)
    // v12 words: every increment into this core's words for layer NL - 1 preceded its last wait above (STREAM: W-VMD;
    // owner: W-OP), and the reader's LA write preceded the writer's last LA wait
    if (vrole) for (uint32_t i = 0; i < NVG; ++i) *vw(vg[i]) = 0;
#endif
    if (kv_mode == 2) *l1w(gs[GS_KVPROG]) = 0;   // the reader's last write preceded our last wait
#if defined(MKA_RQKV)
    // XRDY: every STREAM_EXT core awaited the last multicast (NL - 1) before its last qkv, which precedes this point
    // only on that core; the reset of a late remote copy is covered by the next pass starting from 0 only after the
    // host synchronises, and the Q root never re-sends after NL - 2. QPART: every inc was awaited by its owner.
    // spec v0.9.13: reach the FINAL cumulative value before the reset (QROOT multicasts XRDY = L + 1 for L + 1 < NL to
    // the whole grid; cores that never wait on it would otherwise race the last multicast)
    if (NL > 1) noc_semaphore_wait_min(l1w(gs[GS_XRDY]), NL - 1);
    *l1w(gs[GS_XRDY]) = 0;
    if (qrole == QR_OWNER) *l1w(gs[GS_QPART]) = 0;
#if defined(MKA_RWO)
    *l1w(gs[GS_ACNT]) = 0; *l1w(gs[GS_AFLAG]) = 0; *l1w(gs[GS_WOCNT]) = 0; if (is_qroot) *l1w(gs[GS_WODONE]) = 0;
#endif
    // GS_PQH is reset by the reader after its last wait (see reader)
#endif
}
