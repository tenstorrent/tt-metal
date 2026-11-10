// Fused MLP v9 writer (root role is a runtime arg: 2 kernel groups instead of 6)
// was: Fused MLP v8 writer (BRISC, NOC0). x: pre-replicated by the producer (XPRE, the all-reduce-fed case) or gathered
// by 4 roots and re-broadcast; h: unicast to its chunk's root, 4 roots (2 on NOC0, 2 on NOC1) broadcast one chunk each.
#include "api/debug/kernel_profiler.hpp"
#include "fm8_common.hpp"

void kernel_main() {
    constexpr uint32_t XTB = get_compile_time_arg_val(0), NGU = get_compile_time_arg_val(1);
    constexpr uint32_t NDN = get_compile_time_arg_val(2), IS_GU = get_compile_time_arg_val(3);
    constexpr uint32_t KT = get_compile_time_arg_val(4), HT = get_compile_time_arg_val(5);
    constexpr uint32_t WTB = get_compile_time_arg_val(6), KB = get_compile_time_arg_val(7);
    constexpr uint32_t GRP = get_compile_time_arg_val(8), XPRE = get_compile_time_arg_val(9);
    constexpr uint32_t XSW = get_compile_time_arg_val(10), XCH = get_compile_time_arg_val(11), HCH = get_compile_time_arg_val(12);
    constexpr auto wd_args = TensorAccessorArgs<13>();
    constexpr auto wgu_args = TensorAccessorArgs<wd_args.next_compile_time_args_offset()>();
    uint32_t a = 0;
    const uint32_t idx = get_arg_val<uint32_t>(a++);
    const uint32_t x_shard = get_arg_val<uint32_t>(a++);
    Rect r[2];
    for (uint32_t i = 0; i < 2; ++i) {
        r[i].x0 = get_arg_val<uint32_t>(a++); r[i].y0 = get_arg_val<uint32_t>(a++);
        r[i].x1 = get_arg_val<uint32_t>(a++); r[i].y1 = get_arg_val<uint32_t>(a++); r[i].n = get_arg_val<uint32_t>(a++);
    }
    const uint32_t self_rect = get_arg_val<uint32_t>(a++);
    uint32_t rx[NROOT], ry[NROOT];
    for (uint32_t i = 0; i < NROOT; ++i) { rx[i] = get_arg_val<uint32_t>(a++); ry[i] = get_arg_val<uint32_t>(a++); }
    const uint32_t cbx_base = get_arg_val<uint32_t>(a++), cbh_base = get_arg_val<uint32_t>(a++);
    const uint32_t wd_addr = get_arg_val<uint32_t>(a++), dn_page = get_arg_val<uint32_t>(a++);
    const uint32_t wgu_addr = get_arg_val<uint32_t>(a++), gu_page = get_arg_val<uint32_t>(a++);
    const uint32_t ROOT = get_arg_val<uint32_t>(a++);   // 0/1: NOC0 root; else 0xFF
    const auto wd = TensorAccessor(wd_args, wd_addr, WTB);
    const auto wgu = TensorAccessor(wgu_args, wgu_addr, WTB);
    cb_reserve_back(CB_X, KT);
    cb_reserve_back(CB_H, HT);
    // weights first: odd gate|up groups and the down slice stream from t = 0
    constexpr uint32_t BLK = KB * 2, NB = KT / KB, DN_TILES = HT * NDN;
    uint32_t w2_tiles = 0;
    if constexpr (IS_GU) {
        for (uint32_t b0 = GRP; b0 < NB; b0 += 2 * GRP) w2_tiles += ((NB - b0) < GRP ? (NB - b0) : GRP) * BLK;
        cb_reserve_back(CB_W2, w2_tiles);
        uint32_t dst = get_write_ptr(CB_W2);
        for (uint32_t b0 = GRP; b0 < NB; b0 += 2 * GRP) {
            const uint32_t nb = (NB - b0) < GRP ? (NB - b0) : GRP;
            noc_async_read(wgu.get_noc_addr(gu_page) + (uint64_t)b0 * BLK * WTB, dst, nb * BLK * WTB);
            dst += nb * BLK * WTB;
        }
    }
    cb_reserve_back(CB_WD, DN_TILES);
    noc_async_read(wd.get_noc_addr(dn_page), get_write_ptr(CB_WD), DN_TILES * WTB);
    {
        DeviceZoneScopedN("W-X");
        if constexpr (XPRE) {
            cb_push_back(CB_X, KT);  // the producer (all-reduce receive stage) already wrote x here
        } else {
            const uint32_t rt = (idx * XSW) / XCH;  // root owning this core's x tiles
            noc_async_write(x_shard, get_noc_addr(rx[rt], ry[rt], cbx_base + idx * XSW * XTB), XSW * XTB);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(rx[rt], ry[rt], get_semaphore(SEM_XCNT0 + rt)), 1);
            if (ROOT < 2) root_bcast(SEM_XCNT0 + ROOT, SEM_XFLG0 + ROOT, XCH / XSW, cbx_base + ROOT * XCH * XTB, XCH * XTB, r, self_rect, 0);
            wait_flags(SEM_XFLG0);
            cb_push_back(CB_X, KT);
        }
    }
    noc_async_read_barrier();
    if constexpr (IS_GU) cb_push_back(CB_W2, w2_tiles);
    cb_push_back(CB_WD, DN_TILES);
    if constexpr (IS_GU) {
        cb_wait_front(CB_HL, 1);
        DeviceZoneScopedN("W-H");
        const uint32_t rt = idx / HCH;
        noc_async_write(get_read_ptr(CB_HL), get_noc_addr(rx[rt], ry[rt], cbh_base + idx * XTB), XTB);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(rx[rt], ry[rt], get_semaphore(SEM_HCNT0 + rt)), 1);
        cb_pop_front(CB_HL, 1);
    }
    if (ROOT < 2) { DeviceZoneScopedN("W-HROOT"); root_bcast(SEM_HCNT0 + ROOT, SEM_HFLG0 + ROOT, HCH, cbh_base + ROOT * HCH * XTB, HCH * XTB, r, self_rect, 0); }
    { DeviceZoneScopedN("W-WAITH"); wait_flags(SEM_HFLG0); }
    cb_push_back(CB_H, HT);
    noc_async_atomic_barrier();
    noc_async_write_barrier();
    cb_wait_front(CB_OUT, NDN);
    clear_flags(SEM_XFLG0); clear_flags(SEM_HFLG0);
}
