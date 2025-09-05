// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    // Read parameters from the kernel arguments
    std::uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);

    l1_buffer_addr += 8;

    uint64_t arc_noc_addr = ((uint64_t)0 << (36 + 6)) | ((uint64_t)8 << 36) | 0x80030408;

    noc_async_read(arc_noc_addr, l1_buffer_addr, 4);
    noc_async_read_barrier();
}
