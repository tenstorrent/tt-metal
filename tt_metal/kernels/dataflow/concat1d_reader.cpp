#include <stdint.h>
#include "dataflow_api.h"


// Make n reads defined by num_reads
// Writes to Specified Circular Buffers in L1
// Expects n provided src_addr, src_noc_x, src_noc_y, and cb_id_in
void kernel_main() {
    // compile time args
    DataFormat fmt = static_cast<DataFormat>(get_compile_time_arg_val(0));
    constexpr bool IS_DRAM_A = static_cast<bool>(get_compile_time_arg_val(1));
    constexpr bool IS_DRAM_B = static_cast<bool>(get_compile_time_arg_val(2));

    // runtime args
    // TODO: This can be generalized to concat on any dim, as long as input is tiled without padding
    // padding should work if padding is on the dims not being concated?
    // Instead of num rows, and num_tiles_per_row, you would pass in num_units, which
    // is the multiplication of all dims before concat dim
    // and num_tiles_per_unit, which is the multiplication of all dims starting from concat dim
    // H and W should use Ht and Wt since we are working with tiles
    uint32_t num_tensors = get_arg_val<uint32_t>(0); //hard coded to 2 right now.
    uint32_t num_rows  = get_arg_val<uint32_t>(1);
    uint32_t cb_id_in  = get_arg_val<uint32_t>(2);

    // ublocks size defined in tiles
    uint32_t ublock_size_tiles = 1;
    uint32_t tile_size_bytes = get_tile_size(cb_id_in);

    InterleavedAddrGenFast<false> l1_src_addr_gens[num_tensors];
    InterleavedAddrGenFast<true> dram_src_addr_gens[num_tensors];
    uint32_t num_tiles_per_row[num_tensors];
    uint32_t tile_id_per_tensor[num_tensors];
    const bool is_dram[num_tensors] = {IS_DRAM_A,IS_DRAM_B};

    for (uint32_t i = 0; i < num_tensors; i++) {
        uint32_t src_addr  = get_arg_val<uint32_t>(3 + i * 2);
        num_tiles_per_row[i] = get_arg_val<uint32_t>(4 + i * 2);
        dram_src_addr_gens[i] = {
            .bank_base_address = src_addr,
            .page_size = tile_size_bytes,
            .data_format = fmt
        };

        l1_src_addr_gens[i] = {
            .bank_base_address = src_addr,
            .page_size = tile_size_bytes,
            .data_format = fmt
        };
        tile_id_per_tensor[i] = 0;
    }
    for (uint32_t i = 0; i<num_rows; i++) {
        for(uint32_t j = 0; j < num_tensors; j++) {
            for(uint32_t k = 0; k < num_tiles_per_row[j]; k++) {
                cb_reserve_back(cb_id_in, ublock_size_tiles);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in);
                if ( is_dram[j] ) {
                    noc_async_read_tile(tile_id_per_tensor[j], dram_src_addr_gens[j], l1_write_addr);
                } else {
                    noc_async_read_tile(tile_id_per_tensor[j], l1_src_addr_gens[j], l1_write_addr);
                }
                noc_async_read_barrier();
                cb_push_back(cb_id_in, ublock_size_tiles);
                tile_id_per_tensor[j]++;
            }
        }
    }

}
