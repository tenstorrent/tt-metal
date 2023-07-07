#include <stdint.h>
#include "dataflow_kernel_api.h"

#include "debug_print.h"

using u32 = std::uint32_t;

// tile index to address
inline u32 TADDR(u32 ti) {
    return ti << 11;
}

void kernel_main() {
    u32 src0_addr    = get_arg_val<uint32_t>(0);
    u32 input_Wt     = get_arg_val<uint32_t>(1);
    u32 output_N     = get_arg_val<uint32_t>(2);
    u32 output_C     = get_arg_val<uint32_t>(3);
    u32 output_Ht    = get_arg_val<uint32_t>(4);
    u32 output_Wt    = get_arg_val<uint32_t>(5);

    u32 num_sticks_per_input_tile_row = input_Wt << 5; // Tile height is 32
    u32 num_sticks_per_output_tile_row = output_Wt << 5;

    constexpr u32 SUBTILE_LINE_BYTES = (16<<1);
    constexpr u32 onetile = 1;
    constexpr u32 operand0 = 0;


    const dataflow::InterleavedPow2AddrGen<true> s0 = {
        .bank_base_address = src0_addr,


        .log_base_2_of_page_size = 11
    };

    // Sticks are a row of elements in a single tile (32 elements)
    // Stick id increments row-wise
    // We loop through output tiles row wise, and then read one tile row at a tile to form the output tile
    uint32_t base_tile_row_stick_id = 0; // ID of the first stick in the target output tile row
    for (uint32_t n = 0; n < output_N; n++) { // Iterate over output N
        for (uint32_t c = 0; c < output_C; c++) { // Iterate over output C TODO: Combine into single loop of batch
            for (uint32_t h = 0; h < output_Ht; h++) { // Iterate over output Ht
                uint32_t base_tile_stick_id = base_tile_row_stick_id;
                for (uint32_t w = 0; w < output_Wt; w++) {
                    uint32_t output_stick_id = base_tile_stick_id; // Offset tile id of the current sub tile row
                    dataflow::cb_reserve_back(operand0, onetile);
                    for (uint32_t tile_h = 0; tile_h < 32; tile_h++) {

                        uint32_t input_tile_row_to_read = output_stick_id / num_sticks_per_input_tile_row;
                        uint32_t input_tile_col_to_read = output_stick_id % input_Wt;
                        uint32_t input_tile_to_read = input_tile_row_to_read * input_Wt + input_tile_col_to_read;
                        uint32_t input_tile_sub_row_to_read = output_stick_id % num_sticks_per_input_tile_row / input_Wt;

                        uint64_t banked_addr = dataflow::get_noc_addr(input_tile_to_read, s0);
                        banked_addr += (((input_tile_sub_row_to_read >> 4) << 1) << 9); // if intra-tile source h is > 16, add 2*512 to subtile offset
                        banked_addr += ((input_tile_sub_row_to_read & 15) << 5); // 16 * 2 bytes per face row

                        uint32_t dest_tr0_l1 = dataflow::get_write_ptr(operand0);
                        dest_tr0_l1 += (((tile_h >> 4) << 1) << 9); // if intra-tile source h is > 16, add 2*512 to subtile offset
                        dest_tr0_l1 += ((tile_h & 15) << 5); // 16 * 2 bytes per face row

                        dataflow::noc_async_read(banked_addr, dest_tr0_l1, SUBTILE_LINE_BYTES);
                        // Read the 16 elements for the row of the face directly adjacent since this comes from the same input tile
                        dest_tr0_l1 += 512; // 16 subtile rows of 16 elements of 2 bytes for bfloat16 (16 * 16 * 2)
                        banked_addr += 512;
                        dataflow::noc_async_read(banked_addr, dest_tr0_l1, SUBTILE_LINE_BYTES);

                        output_stick_id += output_Wt;

                    }
                    dataflow::noc_async_read_barrier();
                    // notifies the unpacker that the buffer is populated
                    dataflow::cb_push_back(operand0, onetile);

                    base_tile_stick_id += 1;
                }

                base_tile_row_stick_id += num_sticks_per_output_tile_row;
            }
        }
    }
}
