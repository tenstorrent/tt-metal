// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

// Pre-fill kernel: reads from DRAM into L1 buffers.
// Runs on BRISC only.
//
// Compile-time args:
//   0: dram_buffer0_addr        - DRAM address of buffer 0 (bfloat16)
//   1: dram_buffer1_addr        - DRAM address of buffer 1 (bfloat16)
//   2: l1_buffer0_addr          - L1 destination for buffer 0 (bfloat16)
//   3: l1_buffer1_addr          - L1 destination for buffer 1 (bfloat16)
//   4: tile_size_bytes          - 2048 for bfloat16
//   5: num_tiles                - number of tiles to read (8)
//   6: l1_super_sync_addr       - L1 destination for super sync semaphore
//   7: l1_fpu_timing_addr       - L1 destination for FPU timing results

void kernel_main() {
    constexpr uint32_t dram_buffer0_addr = get_compile_time_arg_val(0);
    constexpr uint32_t dram_buffer1_addr = get_compile_time_arg_val(1);
    constexpr uint32_t l1_buffer0_addr = get_compile_time_arg_val(2);
    constexpr uint32_t l1_buffer1_addr = get_compile_time_arg_val(3);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t l1_super_sync_addr = get_compile_time_arg_val(6);
    constexpr uint32_t l1_fpu_timing_addr = get_compile_time_arg_val(7);
    volatile tt_l1_ptr uint32_t* l1_super_sync_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_super_sync_addr);

    volatile tt_l1_ptr uint32_t* l1_fpu_timing_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_fpu_timing_addr);

    constexpr auto dram0_args = TensorAccessorArgs<8>();
    constexpr auto dram1_args = TensorAccessorArgs<dram0_args.next_compile_time_args_offset()>();

    const auto dram0_addr_gen = TensorAccessor(dram0_args, dram_buffer0_addr);
    const auto dram1_addr_gen = TensorAccessor(dram1_args, dram_buffer1_addr);

    // Read buffer 0 from DRAM to L1
    for (uint32_t t = 0; t < num_tiles; ++t) {
        uint32_t l1_write_addr = l1_buffer0_addr + (t * tile_size_bytes);
        noc_async_read_page(t, dram0_addr_gen, l1_write_addr);
        noc_async_read_barrier();
    }

    // Read buffer 1 from DRAM to L1
    for (uint32_t t = 0; t < num_tiles; ++t) {
        uint32_t l1_write_addr = l1_buffer1_addr + (t * tile_size_bytes);
        noc_async_read_page(t, dram1_addr_gen, l1_write_addr);
        noc_async_read_barrier();
    }

    *(l1_super_sync_addr_ptr) = 0;
    for (uint32_t i = 0; i < 4; ++i) {
        l1_fpu_timing_addr_ptr[i] = 0;
    }
}
