// Deliberately triggers NoC sanitizer — reads from misaligned address
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);

    // Force a misaligned NOC read: add 7 to make address ≡ 7 mod 16
    uint64_t noc_addr = get_noc_addr(0, 0, src_addr + 7);
    uint32_t l1_dest = get_write_ptr(0);

    // This should trip the NoC sanitizer — address is not 16-byte aligned
    noc_async_read(noc_addr, l1_dest, 16);
    noc_async_read_barrier();
}
