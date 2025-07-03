// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    // Read parameters from the kernel arguments
    std::uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);

    // Address and the DRAM bank ID of the source buffer
    std::uint32_t dram_buffer_src_addr  = get_arg_val<uint32_t>(1);
    std::uint32_t dram_buffer_src_bank  = get_arg_val<uint32_t>(2);

    // Address and the DRAM bank ID of the destination buffer
    std::uint32_t dram_buffer_dst_addr  = get_arg_val<uint32_t>(3);
    std::uint32_t dram_buffer_dst_bank  = get_arg_val<uint32_t>(4);

    // Size of the buffer in bytes
    std::uint32_t dram_buffer_size      = get_arg_val<uint32_t>(5);

    std::uint64_t dram_buffer_src_noc_addr =
        get_noc_addr_from_bank_id</*dram=*/true>(dram_buffer_src_bank, dram_buffer_src_addr);
    // Read data into L1 buffer and wait for transfer to complete
    noc_async_read(dram_buffer_src_noc_addr, l1_buffer_addr, dram_buffer_size);
    noc_async_read_barrier();

    std::uint64_t dram_buffer_dst_noc_addr =
        get_noc_addr_from_bank_id</*dram=*/true>(dram_buffer_dst_bank, dram_buffer_dst_addr);
    // Write data from L1 back into DRAM and wait
    noc_async_write(l1_buffer_addr, dram_buffer_dst_noc_addr, dram_buffer_size);
    noc_async_write_barrier();
}
