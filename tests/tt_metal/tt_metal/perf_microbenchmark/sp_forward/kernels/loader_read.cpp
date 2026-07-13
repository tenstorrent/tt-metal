// SPDX-License-Identifier: Apache-2.0
// in0 loader READ stage (BRISC/NOC0): read in0 K-block by K-block (interleaved DRAM) into cb1, deeply
// pipelined (RD tiles in flight), push each block for the mcast stage. K-block streamed => bounded L1.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_kblocks = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_kb = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb1 = get_compile_time_arg_val(3);
    constexpr uint32_t num_banks = get_compile_time_arg_val(4);
    constexpr uint32_t use_cb = get_compile_time_arg_val(5);  // 0 = read-only (local ring, no mcast consumer)
    constexpr uint32_t contig = get_compile_time_arg_val(6);  // 1 = DRAM-sharded contiguous read (one bank)

    const uint32_t in0_addr = get_arg_val<uint32_t>(0);
    const uint32_t base_tile = get_arg_val<uint32_t>(1);
    const uint32_t my_bank = get_arg_val<uint32_t>(2);  // contig: this loader's DRAM bank
    const uint32_t base_l1 = get_write_ptr(cb1);

    uint32_t tile = base_tile;
    uint32_t coff = base_tile * tile_bytes;  // contig: running byte offset within my_bank
    for (uint32_t kb = 0; kb < num_kblocks; ++kb) {
        uint32_t l1;
        if constexpr (use_cb) {
            cb_reserve_back(cb1, tiles_per_kb);
            l1 = get_write_ptr(cb1);
        } else {
            l1 = base_l1 + (kb % 2) * tiles_per_kb * tile_bytes;
        }
        uint32_t p = l1;
        for (uint32_t i = 0; i < tiles_per_kb; ++i) {  // tiles_per_kb reads in flight per block
            uint64_t src;
            if constexpr (contig) {
                src = get_noc_addr_from_bank_id<true>(my_bank, in0_addr + coff);
                coff += tile_bytes;
            } else {
                uint32_t t = tile + i;
                src = get_noc_addr_from_bank_id<true>(t % num_banks, in0_addr + (t / num_banks) * tile_bytes);
            }
            noc_async_read(src, p, tile_bytes);
            p += tile_bytes;
        }
        noc_async_read_barrier();
        if constexpr (use_cb) {
            cb_push_back(cb1, tiles_per_kb);
        }
        tile += tiles_per_kb;
    }
}
