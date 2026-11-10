// lever 1 producer reader: the raw a|b tile holding head h's a / b columns (user 0's tile row).
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tg_common.hpp"

void kernel_main() {
    constexpr auto qkv_args = TensorAccessorArgs<0>();
    const uint32_t qkv_addr = get_arg_val<uint32_t>(0), page = get_arg_val<uint32_t>(1);
    const auto qkv = TensorAccessor(qkv_args, qkv_addr, 2048);
    DeviceZoneScopedN("TG-R");
    cb_reserve_back(TG_AB, 1);
    noc_async_read(qkv.get_noc_addr(page), get_write_ptr(TG_AB), 2048);
    noc_async_read_barrier();
    cb_push_back(TG_AB, 1);
}
