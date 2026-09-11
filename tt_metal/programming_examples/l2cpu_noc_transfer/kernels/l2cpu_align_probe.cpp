// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// Alignment probe for Tensix <-> L2CPU NOC transfers.
//   WRITE test: fill L1 at (l1_base + src_off) with pattern words, NOC-write `size` bytes to
//               L2CPU address (dst_base + dst_off).
//   READ test:  NOC-read `size` bytes from L2CPU (dst_base + dst_off) into (l1_base + 0x400 +
//               src_off), then copy that L1 region (from its 64 B-aligned base) to the L2CPU
//               readback area (rb_base) with an aligned write so the host can see what landed.
void kernel_main() {
    const uint32_t l1_base = get_arg_val<uint32_t>(0);  // 64 B aligned
    const uint32_t l2cpu_x = get_arg_val<uint32_t>(1);
    const uint32_t l2cpu_y = get_arg_val<uint32_t>(2);
    const uint32_t dst_base = get_arg_val<uint32_t>(3);  // 64 B aligned L2CPU address
    const uint32_t size = get_arg_val<uint32_t>(4);
    const uint32_t src_off = get_arg_val<uint32_t>(5);
    const uint32_t dst_off = get_arg_val<uint32_t>(6);
    const uint32_t do_read = get_arg_val<uint32_t>(7);
    const uint32_t rb_base = get_arg_val<uint32_t>(8);  // 64 B aligned L2CPU readback area (read test)

    if (!do_read) {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_base + src_off);
        for (uint32_t i = 0; i < size / 4; i++) {
            p[i] = 0xA5000000u | i;
        }
        noc_async_write(l1_base + src_off, get_noc_addr(l2cpu_x, l2cpu_y, dst_base + dst_off), size);
        noc_async_write_barrier();
    } else {
        // Landing zone: 256 B at l1_base + 0x400, pre-filled with a marker.
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_base + 0x400);
        for (uint32_t i = 0; i < 64; i++) {
            z[i] = 0xEEEEEEEEu;
        }
        noc_async_read(get_noc_addr(l2cpu_x, l2cpu_y, dst_base + dst_off), l1_base + 0x400 + src_off, size);
        noc_async_read_barrier();
        invalidate_l1_cache();
        // Aligned write of the whole landing zone so the host can inspect it.
        noc_async_write(l1_base + 0x400, get_noc_addr(l2cpu_x, l2cpu_y, rb_base), 256);
        noc_async_write_barrier();
    }
}
