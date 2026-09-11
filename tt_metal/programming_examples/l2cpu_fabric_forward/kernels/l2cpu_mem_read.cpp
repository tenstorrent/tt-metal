// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// Generic L2CPU memory -> host read. Copies `size` bytes (multiple of 16) from the
// L2CPU tile at src_addr into this core's L1 and on to a DRAM buffer the host reads.
void kernel_main() {
    const uint32_t l1_scratch = get_arg_val<uint32_t>(0);
    const uint32_t dram_dst = get_arg_val<uint32_t>(1);
    const uint32_t size = get_arg_val<uint32_t>(2);
    const uint32_t l2cpu_x = get_arg_val<uint32_t>(3);
    const uint32_t l2cpu_y = get_arg_val<uint32_t>(4);
    const uint32_t src_addr = get_arg_val<uint32_t>(5);

    constexpr auto out0_args = TensorAccessorArgs<0>();
    const auto out0 = TensorAccessor(out0_args, dram_dst, size);

    noc_async_read(get_noc_addr(l2cpu_x, l2cpu_y, src_addr), l1_scratch, size);
    noc_async_read_barrier();
    invalidate_l1_cache();
    noc_async_write_page(0, out0, l1_scratch);
    noc_async_write_barrier();
}
