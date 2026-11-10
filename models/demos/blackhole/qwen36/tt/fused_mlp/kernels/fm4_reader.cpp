// Fused MLP v3 reader (NCRISC, NOC1): only the gate|up DRAM stream (bank-local, contiguous) into CB_W.
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "fm2_common.hpp"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0), KB = get_compile_time_arg_val(1);
    constexpr uint32_t WTB = get_compile_time_arg_val(2), GRP = get_compile_time_arg_val(3);
    constexpr auto wgu_args = TensorAccessorArgs<4>();
    const uint32_t wgu_addr = get_arg_val<uint32_t>(0), gu_page = get_arg_val<uint32_t>(1);
    const auto wgu = TensorAccessor(wgu_args, wgu_addr, WTB);
    DeviceZoneScopedN("R-GU");
    constexpr uint32_t BLK = KB * 2, NB = KT / KB;
    for (uint32_t b0 = 0; b0 < NB; b0 += 2 * GRP) {  // even groups (NOC1); the writer streams the odd ones (NOC0)
        const uint32_t nb = (NB - b0) < GRP ? (NB - b0) : GRP;
        cb_reserve_back(CB_W, nb * BLK);
        noc_async_read(wgu.get_noc_addr(gu_page) + (uint64_t)b0 * BLK * WTB, get_write_ptr(CB_W), nb * BLK * WTB);
        noc_async_read_barrier();
        cb_push_back(CB_W, nb * BLK);
    }
}
