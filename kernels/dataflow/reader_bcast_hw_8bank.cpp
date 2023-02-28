#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr  = get_arg_val<uint32_t>(0);
    uint32_t src0_num_tiles  = get_arg_val<uint32_t>(3);
    uint32_t src1_addr  = get_arg_val<uint32_t>(4);
    // skip args 1,2,5,6,7 for compat with single bank readers and reader_diff_lengths
    uint32_t NCHtWt     = get_arg_val<uint32_t>(8);
    uint32_t NC         = get_arg_val<uint32_t>(9);
    uint32_t Ht         = get_arg_val<uint32_t>(10);
    uint32_t Wt         = get_arg_val<uint32_t>(11);
    uint32_t nc1        = get_arg_val<uint32_t>(12); // if 1 we expect the bcast tensor to have NC=1 and wrap around in NC

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t onetile = 1;

    // single-tile ublocks
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    uint32_t num_tiles = src0_num_tiles;
    uint32_t i = 0;
    uint32_t i1 = 0;

    const InterleavedPow2AddrGen s0 = {
        .bank_base_address = src0_addr,
        .num_used_banks = 8,
        .log_base_2_of_num_used_banks = 3,
        .log_base_2_of_bank_unit_size = 11
    };

    const InterleavedPow2AddrGen s1 = {
        .bank_base_address = src1_addr,
        .num_used_banks = 8,
        .log_base_2_of_num_used_banks = 3,
        .log_base_2_of_bank_unit_size = 11
    };

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t ht = 0; ht < Ht; ht++) {
        for (uint32_t wt = 0; wt < Wt; wt++) {
            uint64_t src0_noc_addr = get_noc_addr(i, s0);
            cb_reserve_back(cb_id_in0, onetile);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            noc_async_read(src0_noc_addr, l1_write_addr_in0, tile_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);

            // for each H,W-tile of the first tensor we push one tile from the second arg tile list
            // but we don't advance the second tile index for H,W
            cb_reserve_back(cb_id_in1, onetile);
            uint64_t src1_noc_addr = get_noc_addr(i1, s1);
            l1_write_addr_in1 = get_write_ptr(cb_id_in1);
            noc_async_read(src1_noc_addr, l1_write_addr_in1, tile_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_id_in1, onetile);

            i ++; // input tile iterates over NC Ht Wt
        } // wt loop
        } // ht loop
        if (nc1 == 0)
            i1 ++; // bcast-HW tile iterates only for nc loop and only if NC>1
    } // nc loop
}
