// engine/program.py make_xio: the engine's residual I/O on the 16 owner cores (AR contract, R = 32): owner j holds x
// columns [320 j, 320 j + 320) = DRAM tiles 10 j .. 10 j + 9 of the [32, 5120] bf16 TILE residual as its v_x0 shard.
// runtime args: [dram base, shard L1 addr, first tile, n tiles, tile bytes, dir (0: DRAM -> L1 = x_in seed, 1: L1 -> DRAM = x_out)]
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dram = get_arg_val<uint32_t>(0);
    const uint32_t l1 = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);
    const uint32_t tb = get_arg_val<uint32_t>(4);
    const uint32_t dir = get_arg_val<uint32_t>(5);
    const InterleavedAddrGenFast<true> g = {.bank_base_address = dram, .page_size = tb, .data_format = DataFormat::Float16_b};
    for (uint32_t i = 0; i < n; ++i) {
        if (dir == 0) {
            noc_async_read_tile(t0 + i, g, l1 + i * tb);
        } else {
            noc_async_write_tile(t0 + i, g, l1 + i * tb);
        }
    }
    if (dir == 0) {
        noc_async_read_barrier();
    } else {
        noc_async_write_barrier();
    }
}
