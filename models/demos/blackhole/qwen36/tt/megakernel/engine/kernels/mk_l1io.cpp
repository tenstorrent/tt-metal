// engine/program.py l1io (spec §20 phase 2): DRAM <-> a planned engine CB address, one core per entry. Runs as its own
// generic_op right before (seed) / after (drain) the engine op: the engine's static CBs never touch data at setup, and
// nothing runs between the two ops, so the bytes stay valid across the launch boundary.
// runtime args: [dram base, L1 addr, first page, n pages, page bytes, dir (0: DRAM -> L1, 1: L1 -> DRAM)]
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dram = get_arg_val<uint32_t>(0);
    const uint32_t l1 = get_arg_val<uint32_t>(1);
    const uint32_t p0 = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);
    const uint32_t pb = get_arg_val<uint32_t>(4);
    const uint32_t dir = get_arg_val<uint32_t>(5);
    const InterleavedAddrGen<true> g = {.bank_base_address = dram, .page_size = pb};
    for (uint32_t i = 0; i < n; ++i) {
        if (dir == 0) {
            noc_async_read(get_noc_addr(p0 + i, g), l1 + i * pb, pb);
        } else {
            noc_async_write(l1 + i * pb, get_noc_addr(p0 + i, g), pb);
        }
    }
    if (dir == 0) {
        noc_async_read_barrier();
    } else {
        noc_async_write_barrier();
    }
}
