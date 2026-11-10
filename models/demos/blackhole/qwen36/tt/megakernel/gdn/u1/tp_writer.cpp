// GDN_STEP_U1 op-layout prefix writer: the job's fp32 tile to up to 3 pages of `prefixed` (CB `cb` = compile arg 0).
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t CB = get_compile_time_arg_val(0), N = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    constexpr uint32_t TB = 4096;
    const uint32_t out_addr = get_arg_val<uint32_t>(0), ndst = get_arg_val<uint32_t>(1);
    const auto out = TensorAccessor(out_args, out_addr, TB);
    DeviceZoneScopedN("TP-W");
    cb_wait_front(CB, N);
    for (uint32_t d = 0; d < ndst; ++d) {
        const uint32_t page0 = get_arg_val<uint32_t>(2 + d);
        for (uint32_t i = 0; i < N; ++i) noc_async_write(get_read_ptr(CB) + i * TB, out.get_noc_addr(page0 + i), TB);
    }
    noc_async_write_barrier();
    cb_pop_front(CB, N);
}
