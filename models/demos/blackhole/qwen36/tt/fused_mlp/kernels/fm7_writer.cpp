// Fused MLP v3 writer (BRISC, NOC0): owns the x / h landing CBs and the down weights. Multicasts x (and h), prefetches
// this core's down slice from its bank, publishes x / h to compute when the coordinator's flags arrive.
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "fm2_common.hpp"

struct Rect { uint32_t x0, y0, x1, y1, n; };
FORCE_INLINE void mcast_data(uint32_t src, uint32_t dst_l1, uint32_t bytes, const Rect* r, uint32_t self_rect) {
    for (uint32_t i = 0; i < 2; ++i) {
        if (r[i].n == 0) continue;
        uint64_t a = get_noc_multicast_addr(r[i].x0, r[i].y0, r[i].x1, r[i].y1, dst_l1);
        if (i == self_rect) noc_async_write_multicast_loopback_src(src, a, bytes, r[i].n);
        else noc_async_write_multicast(src, a, bytes, r[i].n);
    }
}
FORCE_INLINE void mcast_flag(uint32_t flag_l1, const Rect* r, uint32_t self_rect) {
    for (uint32_t i = 0; i < 2; ++i) {
        if (r[i].n == 0) continue;
        uint64_t a = get_noc_multicast_addr(r[i].x0, r[i].y0, r[i].x1, r[i].y1, flag_l1);
        if (i == self_rect) noc_semaphore_set_multicast_loopback_src(flag_l1, a, r[i].n);
        else noc_semaphore_set_multicast(flag_l1, a, r[i].n);
    }
}

void kernel_main() {
    constexpr uint32_t XSW = get_compile_time_arg_val(0), XTB = get_compile_time_arg_val(1);
    constexpr uint32_t NCORES = get_compile_time_arg_val(2), NGU = get_compile_time_arg_val(3);
    constexpr uint32_t NDN = get_compile_time_arg_val(4), IS_GU = get_compile_time_arg_val(5);
    constexpr uint32_t IS_COORD = get_compile_time_arg_val(6), KT = get_compile_time_arg_val(7);
    constexpr uint32_t HT = get_compile_time_arg_val(8), WTB = get_compile_time_arg_val(9);
    constexpr uint32_t KB = get_compile_time_arg_val(10), GRP = get_compile_time_arg_val(11);
    constexpr auto wd_args = TensorAccessorArgs<12>();
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
    const uint32_t cx = get_arg_val<uint32_t>(a++), cy = get_arg_val<uint32_t>(a++);
    const uint32_t cbx_base = get_arg_val<uint32_t>(a++), cbh_base = get_arg_val<uint32_t>(a++);
    const uint32_t wd_addr = get_arg_val<uint32_t>(a++), dn_page = get_arg_val<uint32_t>(a++);
    const uint32_t wgu_addr = get_arg_val<uint32_t>(a++), gu_page = get_arg_val<uint32_t>(a++);
    const auto wd = TensorAccessor(wd_args, wd_addr, WTB);
    const auto wgu = TensorAccessor(wgu_args, wgu_addr, WTB);
    volatile tt_l1_ptr uint32_t* xcnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_XCNT));
    volatile tt_l1_ptr uint32_t* xflag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_XFLAG));
    volatile tt_l1_ptr uint32_t* hcnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_HCNT));
    volatile tt_l1_ptr uint32_t* hflag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_HFLAG));
    cb_reserve_back(CB_X, KT);
    cb_reserve_back(CB_H, HT);
    // odd gate|up groups (NOC0): issue all reads now (they fit CB_W2), so DRAM streams while x is gathered
    constexpr uint32_t BLK = KB * 2, NB = KT / KB;
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
    constexpr uint32_t DN_TILES0 = HT * NDN;
    cb_reserve_back(CB_WD, DN_TILES0);
    noc_async_read(wd.get_noc_addr(dn_page), get_write_ptr(CB_WD), DN_TILES0 * WTB);
#ifdef DRAIN_FIRST
    { DeviceZoneScopedN("W-DRAIN"); noc_async_read_barrier(); }  // timing probe: this core's whole writer share landed
#endif
    {
        // x: every core unicasts its shard into the root's landing zone; the root multicasts the whole x once.
        DeviceZoneScopedN("W-X");
        noc_async_write(x_shard, get_noc_addr(cx, cy, cbx_base + idx * XSW * XTB), XSW * XTB);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(cx, cy, (uint32_t)xcnt), 1);
        if constexpr (IS_COORD) {
            noc_semaphore_wait(xcnt, NCORES);
            *xcnt = 0;
            mcast_data(cbx_base, cbx_base, KT * XTB, r, self_rect);
            noc_async_write_barrier();
            *xflag = 1;
            mcast_flag((uint32_t)xflag, r, self_rect);
        }
    }
    { DeviceZoneScopedN("W-WAITX"); noc_semaphore_wait(xflag, 1); }
    cb_push_back(CB_X, KT);
    constexpr uint32_t DN_TILES = HT * NDN;
    if constexpr (IS_GU) {
        DeviceZoneScopedN("W-GU");
        noc_async_read_barrier();
        cb_push_back(CB_W2, w2_tiles);
    }
    if constexpr (IS_GU) {
        cb_wait_front(CB_HL, 1);
        DeviceZoneScopedN("W-H");
        noc_async_write(get_read_ptr(CB_HL), get_noc_addr(cx, cy, cbh_base + idx * XTB), XTB);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(cx, cy, (uint32_t)hcnt), 1);
        cb_pop_front(CB_HL, 1);
    }
    { DeviceZoneScopedN("W-DN"); noc_async_read_barrier(); }
    cb_push_back(CB_WD, DN_TILES);
    if constexpr (IS_COORD) {
        noc_semaphore_wait(hcnt, NGU);
        *hcnt = 0;
        mcast_data(cbh_base, cbh_base, HT * XTB, r, self_rect);
        noc_async_write_barrier();
        *hflag = 1;
        mcast_flag((uint32_t)hflag, r, self_rect);
    }
    { DeviceZoneScopedN("W-WAITH"); noc_semaphore_wait(hflag, 1); }
    cb_push_back(CB_H, HT);
    noc_async_atomic_barrier();
    noc_async_write_barrier();
    cb_wait_front(CB_OUT, NDN);
    *xflag = 0; *hflag = 0;
}
