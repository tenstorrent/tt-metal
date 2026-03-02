// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

// Pre-fill kernel: reads from DRAM into L1 buffers.
// Runs on BRISC only.
//
// Compile-time args:
//   0: dram_buffer0_addr        - DRAM address of buffer 0 (bfloat16)
//   1: dram_buffer1_addr        - DRAM address of buffer 1 (bfloat16)
//   2: dram_buffer_0xAAAA_addr  - DRAM address of 0xAAAAAAAA pattern
//   3: dram_buffer_0x5555_addr  - DRAM address of 0x55555555 pattern
//   4: l1_buffer0_addr          - L1 destination for buffer 0 (bfloat16)
//   5: l1_buffer1_addr          - L1 destination for buffer 1 (bfloat16)
//   6: l1_buffer3_addr          - L1 destination for 0xAAAA pattern
//   7: l1_buffer4_addr          - L1 destination for 0x5555 pattern
//   8: l1_buffer5_addr          - L1 destination for 0xAAAA pattern
//   9: l1_buffer6_addr          - L1 destination for 0x5555 pattern
//   10: tile_size_bytes         - 2048 for bfloat16
//   11: transfer_size           - transfer size (8KB)
//   12: num_tiles               - number of tiles to read (8)
// Buffers 2, 7, 8, 9, 10: addr in cfg only, no init in prefill kernel.

void kernel_main() {
    constexpr uint32_t dram_buffer0_addr = get_compile_time_arg_val(0);
    constexpr uint32_t dram_buffer1_addr = get_compile_time_arg_val(1);
    constexpr uint32_t dram_buffer_0xAAAA_addr = get_compile_time_arg_val(2);
    constexpr uint32_t dram_buffer_0x5555_addr = get_compile_time_arg_val(3);
    constexpr uint32_t l1_buffer0_addr = get_compile_time_arg_val(4);
    constexpr uint32_t l1_buffer1_addr = get_compile_time_arg_val(5);
    constexpr uint32_t l1_buffer3_addr = get_compile_time_arg_val(6);
    constexpr uint32_t l1_buffer4_addr = get_compile_time_arg_val(7);
    constexpr uint32_t l1_buffer5_addr = get_compile_time_arg_val(8);
    constexpr uint32_t l1_buffer6_addr = get_compile_time_arg_val(9);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t transfer_size = get_compile_time_arg_val(11);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(12);

    // Create address generators for DRAM buffers
    const InterleavedAddrGen<true> dram0_addr_gen = {
        .bank_base_address = dram_buffer0_addr,
        .page_size = tile_size_bytes,
    };

    const InterleavedAddrGen<true> dram1_addr_gen = {
        .bank_base_address = dram_buffer1_addr,
        .page_size = tile_size_bytes,
    };

    const InterleavedAddrGen<true> dram_0xAAAA_addr_gen = {
        .bank_base_address = dram_buffer_0xAAAA_addr,
        .page_size = transfer_size,
    };

    const InterleavedAddrGen<true> dram_0x5555_addr_gen = {
        .bank_base_address = dram_buffer_0x5555_addr,
        .page_size = transfer_size,
    };

    // Read buffer 0 from DRAM to L1
    for (uint32_t t = 0; t < num_tiles; ++t) {
        uint32_t l1_write_addr = l1_buffer0_addr + (t * tile_size_bytes);
        uint64_t noc_addr = get_noc_addr(t, dram0_addr_gen);
        noc_async_read(noc_addr, l1_write_addr, tile_size_bytes);
        noc_async_read_barrier();
    }

    // Read buffer 1 from DRAM to L1
    for (uint32_t t = 0; t < num_tiles; ++t) {
        uint32_t l1_write_addr = l1_buffer1_addr + (t * tile_size_bytes);
        uint64_t noc_addr = get_noc_addr(t, dram1_addr_gen);
        noc_async_read(noc_addr, l1_write_addr, tile_size_bytes);
        noc_async_read_barrier();
    }

    // Read buffer 3 and 5 from same DRAM (0xAAAA pattern)
    uint64_t noc_addr = get_noc_addr(0, dram_0xAAAA_addr_gen);
    noc_async_read(noc_addr, l1_buffer3_addr, transfer_size);
    noc_async_read_barrier();
    noc_async_read(noc_addr, l1_buffer5_addr, transfer_size);
    noc_async_read_barrier();

    // Read buffer 4 and 6 from same DRAM (0x5555 pattern)
    noc_addr = get_noc_addr(0, dram_0x5555_addr_gen);
    noc_async_read(noc_addr, l1_buffer4_addr, transfer_size);
    noc_async_read_barrier();
    noc_async_read(noc_addr, l1_buffer6_addr, transfer_size);
    noc_async_read_barrier();
}
