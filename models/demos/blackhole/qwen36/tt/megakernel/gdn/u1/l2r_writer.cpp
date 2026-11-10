// L2-norm root writer: the N normalized tiles to DRAM (megakernel: multicast to the head's step cores).
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "l2r_common.hpp"

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0), tile0 = get_arg_val<uint32_t>(1);
    const auto out = TensorAccessor(out_args, out_addr, 4096);
    DeviceZoneScopedN("L2R-W");
    cb_wait_front(L_DST, N);
    for (uint32_t i = 0; i < N; ++i) noc_async_write(get_read_ptr(L_DST) + i * 4096, out.get_noc_addr(tile0 + i), 4096);
    noc_async_write_barrier();
    cb_pop_front(L_DST, N);
}
