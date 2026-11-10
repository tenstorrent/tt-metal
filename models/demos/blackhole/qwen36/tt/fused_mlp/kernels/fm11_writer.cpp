// Fused MLP v11 writer (BRISC, NOC0), persistent over NL layers. Per layer L: the odd gate|up groups and the down
// slice of layer L + 1 are issued (transaction id 1 + ((L + 1) & 1)) before layer L is consumed, into double-buffered
// CB_W2 / CB_WD, so each core's DRAM stream never idles across layers. x is producer-fed (XPRE contract: the
// all-reduce receive stage has written x; here it is the same x for every layer). h: unicast to the chunk's root,
// 4 roots broadcast (roots 0/1 here on NOC0). After its down output for layer L, every core bumps the roots'
// down-done counters.
#include "api/debug/kernel_profiler.hpp"
#include "fm11_common.hpp"

void kernel_main() {
    constexpr uint32_t XTB = get_compile_time_arg_val(0), NGU = get_compile_time_arg_val(1);
    constexpr uint32_t NDN = get_compile_time_arg_val(2), KT = get_compile_time_arg_val(3);
    constexpr uint32_t HT = get_compile_time_arg_val(4), WTB = get_compile_time_arg_val(5);
    constexpr uint32_t KB = get_compile_time_arg_val(6), GRP = get_compile_time_arg_val(7);
    constexpr uint32_t HCH = get_compile_time_arg_val(8), NL = get_compile_time_arg_val(9);
    constexpr uint32_t NCORES = get_compile_time_arg_val(10);
    constexpr auto wd_args = TensorAccessorArgs<11>();
    constexpr auto wgu_args = TensorAccessorArgs<wd_args.next_compile_time_args_offset()>();
    uint32_t a = 0;
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
    const uint32_t ROOT = get_arg_val<uint32_t>(a++);   // 0/1: NOC0 root; else 0xFF
    const uint32_t IS_GU = get_arg_val<uint32_t>(a++);
    const uint32_t pgu_layer = get_arg_val<uint32_t>(a++), pdn_layer = get_arg_val<uint32_t>(a++);
    const auto wd = TensorAccessor(wd_args, wd_addr, WTB);
    const auto wgu = TensorAccessor(wgu_args, wgu_addr, WTB);
    constexpr uint32_t BLK = KB * 2, NB = KT / KB, DN_TILES = HT * NDN;
    uint32_t w2_tiles = 0;
    for (uint32_t b0 = GRP; b0 < NB; b0 += 2 * GRP) w2_tiles += ((NB - b0) < GRP ? (NB - b0) : GRP) * BLK;

    // CB_W2 / CB_WD are two layer slots each. Layer L + 1 is issued before layer L is pushed, so its address is the
    // OTHER slot (get_write_ptr still points at layer L's), and its reserve covers both slots (= layer L - 1 popped).
    const uint32_t w2_base = get_write_ptr(CB_W2), wd_base = get_write_ptr(CB_WD);
    auto issue = [&](uint32_t L) {
        noc_async_read_set_trid(1 + (L & 1));
        if (IS_GU) {
            cb_reserve_back(CB_W2, L == 0 ? w2_tiles : 2 * w2_tiles);
            const uint32_t wp = get_write_ptr(CB_W2);
            uint32_t dst = (L == 0) ? wp : (wp == w2_base ? w2_base + w2_tiles * WTB : w2_base);
            const uint64_t base = wgu.get_noc_addr(gu_page + L * pgu_layer);
            for (uint32_t b0 = GRP; b0 < NB; b0 += 2 * GRP) {
                const uint32_t nb = (NB - b0) < GRP ? (NB - b0) : GRP;
                noc_async_read(base + (uint64_t)b0 * BLK * WTB, dst, nb * BLK * WTB);
                dst += nb * BLK * WTB;
            }
        }
        cb_reserve_back(CB_WD, L == 0 ? DN_TILES : 2 * DN_TILES);
        const uint32_t wpd = get_write_ptr(CB_WD);
        const uint32_t dd = (L == 0) ? wpd : (wpd == wd_base ? wd_base + DN_TILES * WTB : wd_base);
        noc_async_read(wd.get_noc_addr(dn_page + L * pdn_layer), dd, DN_TILES * WTB);
        noc_async_read_set_trid(0);
    };

    issue(0);
    for (uint32_t L = 0; L < NL; ++L) {
        if (L + 1 < NL) issue(L + 1);
        cb_reserve_back(CB_X, KT);  // producer-fed x: already in place; re-expose it to compute for this layer
        cb_push_back(CB_X, KT);
        noc_async_read_barrier_with_trid(1 + (L & 1));
        if (IS_GU) cb_push_back(CB_W2, w2_tiles);
        cb_push_back(CB_WD, DN_TILES);
        if (IS_GU) {
            cb_wait_front(CB_HL, 1);
            const uint32_t rt = idx / HCH;
            noc_async_write(get_read_ptr(CB_HL), get_noc_addr(rx[rt], ry[rt], cbh_base + idx * XTB), XTB);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(rx[rt], ry[rt], get_semaphore(SEM_HCNT0 + rt)), 1);
            cb_pop_front(CB_HL, 1);
        }
        if (ROOT < 2) root_bcast_layer(ROOT, L, NCORES, HCH, cbh_base + ROOT * HCH * XTB, HCH * XTB, r, self_rect, 0);
        wait_hflags(L);
        cb_reserve_back(CB_H, HT);
        cb_push_back(CB_H, HT);
        cb_wait_front(CB_OUT, NDN);   // layer L's down output (in the model: the next all-reduce's send buffer)
        cb_pop_front(CB_OUT, NDN);
        for (uint32_t i = 0; i < NROOT; ++i) noc_semaphore_inc(get_noc_addr(rx[i], ry[i], get_semaphore(SEM_DONE0 + i)), 1);
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
    // reset for the next launch: every core clears its received h flags; NOC0 roots wait until every core is past
    // the last layer, then clear their counters (NOC1 roots do the same in the reader)
    for (uint32_t i = 0; i < NROOT; ++i) *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_HFLG0 + i)) = 0;
    if (ROOT < 2) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE0 + ROOT)), NCORES * NL);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE0 + ROOT)) = 0;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_HCNT0 + ROOT)) = 0;
    }
}
