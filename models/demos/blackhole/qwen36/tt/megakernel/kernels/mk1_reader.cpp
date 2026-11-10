// Fused MLP v11 reader (NCRISC, NOC1), persistent over NL layers: gate_up cores stream the even gate|up groups of
// every layer back to back through CB_W (the ring keeps the next layer's reads flowing); NOC1 root cores (roles 2/3)
// broadcast their h chunk for every layer.
#include "api/debug/kernel_profiler.hpp"
#include "mk1_common.hpp"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0), KB = get_compile_time_arg_val(1);
    constexpr uint32_t WTB = get_compile_time_arg_val(2), GRP = get_compile_time_arg_val(3);
    constexpr uint32_t HCH = get_compile_time_arg_val(4), XTB = get_compile_time_arg_val(5);
    constexpr uint32_t NL = get_compile_time_arg_val(6), NCORES = get_compile_time_arg_val(7);
    constexpr auto wgu_args = TensorAccessorArgs<8>();
    uint32_t a = 0;
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
    const uint32_t pgu_layer = get_arg_val<uint32_t>(a++);  // pages per layer in the gate|up tensor
    const auto wgu = TensorAccessor(wgu_args, wgu_addr, WTB);
    constexpr uint32_t BLK = KB * 2, NB = KT / KB;
    if (IS_GU) {
        DeviceZoneScopedN("R-GU");
        for (uint32_t L = 0; L < NL; ++L) {
            const uint64_t base = wgu.get_noc_addr(gu_page + L * pgu_layer);
            for (uint32_t b0 = 0; b0 < NB; b0 += 2 * GRP) {
                const uint32_t nb = (NB - b0) < GRP ? (NB - b0) : GRP;
                cb_reserve_back(CB_W, nb * BLK);
                noc_async_read(base + (uint64_t)b0 * BLK * WTB, get_write_ptr(CB_W), nb * BLK * WTB);
                noc_async_read_barrier();
                cb_push_back(CB_W, nb * BLK);
            }
        }
    }
    if (ROOT == 2 || ROOT == 3) {
        DeviceZoneScopedN("R-HROOT");
        for (uint32_t L = 0; L < NL; ++L)
            root_bcast_layer(ROOT, L, NCORES, HCH, cbh_base + ROOT * HCH * XTB, HCH * XTB, r, self_rect, 1);
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE0 + ROOT)), NCORES * NL);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE0 + ROOT)) = 0;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_HCNT0 + ROOT)) = 0;
    }
}
