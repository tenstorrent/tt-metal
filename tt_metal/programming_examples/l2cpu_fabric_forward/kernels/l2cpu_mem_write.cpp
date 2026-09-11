// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// Generic host -> L2CPU memory write. Copies `size` bytes (multiple of 16) from a
// DRAM buffer into the L2CPU tile at dst_addr via this core's L1, then optionally
// writes one 32-bit word (flag_val) to flag_addr — used to publish a request or a
// sequence number after its body has landed.
void kernel_main() {
    const uint32_t l1_scratch = get_arg_val<uint32_t>(0);
    const uint32_t dram_src = get_arg_val<uint32_t>(1);
    const uint32_t size = get_arg_val<uint32_t>(2);
    const uint32_t l2cpu_x = get_arg_val<uint32_t>(3);
    const uint32_t l2cpu_y = get_arg_val<uint32_t>(4);
    const uint32_t dst_addr = get_arg_val<uint32_t>(5);
    const uint32_t flag_addr = get_arg_val<uint32_t>(6);
    const uint32_t flag_val = get_arg_val<uint32_t>(7);

    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, dram_src, size);

    if (size != 0) {
        noc_async_read_page(0, in0, l1_scratch);
        noc_async_read_barrier();
        noc_async_write(l1_scratch, get_noc_addr(l2cpu_x, l2cpu_y, dst_addr), size);
        noc_async_write_barrier();
    }
    if (flag_addr != 0) {
        noc_inline_dw_write(get_noc_addr(l2cpu_x, l2cpu_y, flag_addr), flag_val);
        noc_async_write_barrier();
    }
}
