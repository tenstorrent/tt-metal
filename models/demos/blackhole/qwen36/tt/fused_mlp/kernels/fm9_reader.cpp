// Fused MLP v9 reader (root role runtime)
// was: Fused MLP v8 reader (NCRISC, NOC1): gate_up cores stream the even gate|up groups; NOC1 root cores (2, 3) broadcast
// their x chunk (unless XPRE) and h chunk on NOC1.
#include "api/debug/kernel_profiler.hpp"
#include "fm8_common.hpp"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0), KB = get_compile_time_arg_val(1);
    constexpr uint32_t WTB = get_compile_time_arg_val(2), GRP = get_compile_time_arg_val(3);
    constexpr uint32_t IS_GU = get_compile_time_arg_val(4);
    constexpr uint32_t XPRE = get_compile_time_arg_val(5), XCH = get_compile_time_arg_val(6), HCH = get_compile_time_arg_val(7);
    constexpr uint32_t XSW = get_compile_time_arg_val(8), XTB = get_compile_time_arg_val(9);
    constexpr auto wgu_args = TensorAccessorArgs<10>();
    uint32_t a = 0;
    const uint32_t wgu_addr = get_arg_val<uint32_t>(a++), gu_page = get_arg_val<uint32_t>(a++);
    Rect r[2];
    for (uint32_t i = 0; i < 2; ++i) {
        r[i].x0 = get_arg_val<uint32_t>(a++); r[i].y0 = get_arg_val<uint32_t>(a++);
        r[i].x1 = get_arg_val<uint32_t>(a++); r[i].y1 = get_arg_val<uint32_t>(a++); r[i].n = get_arg_val<uint32_t>(a++);
    }
    const uint32_t self_rect = get_arg_val<uint32_t>(a++);
    const uint32_t cbx_base = get_arg_val<uint32_t>(a++), cbh_base = get_arg_val<uint32_t>(a++);
    const uint32_t ROOT = get_arg_val<uint32_t>(a++);   // 2/3: NOC1 root; else 0xFF
    if constexpr (IS_GU) {
        DeviceZoneScopedN("R-GU");
        const auto wgu = TensorAccessor(wgu_args, wgu_addr, WTB);
        constexpr uint32_t BLK = KB * 2, NB = KT / KB;
        for (uint32_t b0 = 0; b0 < NB; b0 += 2 * GRP) {
            const uint32_t nb = (NB - b0) < GRP ? (NB - b0) : GRP;
            cb_reserve_back(CB_W, nb * BLK);
            noc_async_read(wgu.get_noc_addr(gu_page) + (uint64_t)b0 * BLK * WTB, get_write_ptr(CB_W), nb * BLK * WTB);
            noc_async_read_barrier();
            cb_push_back(CB_W, nb * BLK);
        }
    }
    if (ROOT == 2 || ROOT == 3) {
        if constexpr (!XPRE) root_bcast(SEM_XCNT0 + ROOT, SEM_XFLG0 + ROOT, XCH / XSW, cbx_base + ROOT * XCH * XTB, XCH * XTB, r, self_rect, 1);
        DeviceZoneScopedN("R-HROOT");
        root_bcast(SEM_HCNT0 + ROOT, SEM_HFLG0 + ROOT, HCH, cbh_base + ROOT * HCH * XTB, HCH * XTB, r, self_rect, 1);
    }
}
