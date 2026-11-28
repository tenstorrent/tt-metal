// Probe: sub-64-B NOC writes into DRAM (byte-granular patching of a cache tile in place).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    constexpr auto a = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(a, addr, 2048);
    const uint32_t l1 = get_write_ptr(0);
    volatile tt_l1_ptr uint8_t* p = (volatile tt_l1_ptr uint8_t*)l1;
    for (uint32_t i = 0; i < 2048; ++i) {
        p[i] = (uint8_t)(i * 7 + 1);
    }
    const uint64_t d = s.get_noc_addr(0);
    // (offset, size) pairs, src at the same offset
    const uint32_t ws[][2] = {{5, 1}, {17, 1}, {80, 16}, {200, 16}, {320, 2}, {1000, 32}, {1501, 3}};
    for (auto& w : ws) {
        noc_async_write(l1 + w[0], d + w[0], w[1]);
    }
    noc_async_write_barrier();
}
