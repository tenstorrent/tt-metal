#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t seed_page = get_compile_time_arg_val(1);
    constexpr uint32_t active_page = get_compile_time_arg_val(2);
    constexpr auto sa = TensorAccessorArgs<3>();
    constexpr auto aa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    auto seeds = TensorAccessor(sa, get_arg_val<uint32_t>(0), seed_page);
    auto active = TensorAccessor(aa, get_arg_val<uint32_t>(1), active_page);
    uint32_t s = get_write_ptr(0), a = s + 128;
    noc_async_read(seeds.get_noc_addr(0), s, seed_page);
    noc_async_read(active.get_noc_addr(0), a, active_page);
    noc_async_read_barrier();
    auto counters = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s);
    auto mask = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a);
    for (uint32_t i = 0; i < rows; ++i) {
        if (mask[i]) {
            // UINT32_MAX tells manual_seed to retain per-core PRNG state.
            // Keep request counters in the same modulus as host restoration.
            counters[i] = counters[i] >= UINT32_MAX - 1 ? 0 : counters[i] + 1;
        }
    }
    noc_async_write(s, seeds.get_noc_addr(0), seed_page);
    noc_async_write_barrier();
}
