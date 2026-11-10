// engine/argblock.py ArgBlocks.write_op: one core rewrites the arg block of one kind on every core of its grid.
// runtime args: [block L1 addr, width words, n cores, n x (noc x << 16 | noc y), width x block word]
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t width = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    const uint32_t scratch = get_write_ptr(24);
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    for (uint32_t i = 0; i < width; ++i) w[i] = get_arg_val<uint32_t>(3 + n + i);
    for (uint32_t c = 0; c < n; ++c) {
        const uint32_t xy = get_arg_val<uint32_t>(3 + c);
        noc_async_write(scratch, get_noc_addr(xy >> 16, xy & 0xFFFF, addr), width * 4);
    }
    noc_async_write_barrier();
}
