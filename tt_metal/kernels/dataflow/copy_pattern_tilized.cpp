#include <cstdint>
/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or other RISCs
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */
void kernel_main() {
    std::uint32_t dram_buffer_src_addr_base        = get_arg_val<uint32_t>(0);
    std::uint32_t dram_src_noc_x                   = get_arg_val<uint32_t>(1);
    std::uint32_t dram_src_noc_y                   = get_arg_val<uint32_t>(2);

    std::uint32_t N                                = get_arg_val<uint32_t>(3);
    std::uint32_t C                                = get_arg_val<uint32_t>(4);
    std::uint32_t H                                = get_arg_val<uint32_t>(5);
    std::uint32_t W                                = get_arg_val<uint32_t>(6);
    std::uint32_t R                                = get_arg_val<uint32_t>(7);
    std::uint32_t S                                = get_arg_val<uint32_t>(8);
    std::uint32_t U                                = get_arg_val<uint32_t>(9);
    std::uint32_t V                                = get_arg_val<uint32_t>(10);
    std::uint32_t input_vertical_stride_bytes      = get_arg_val<uint32_t>(11); // W * C * 2B
    std::uint32_t input_horizontal_stride_bytes    = get_arg_val<uint32_t>(12); // C * 2B
    std::uint32_t num_tiles_c                      = get_arg_val<uint32_t>(13); // C * R * S / 32
    std::uint32_t num_bytes_per_row_of_tiles       = get_arg_val<uint32_t>(14); // num_tiles_c * 32x32 * 2B
    std::uint32_t num_repetitions                  = get_arg_val<uint32_t>(15);


    for(std::uint32_t i = 0; i < num_repetitions; i++) {
        // l1 address to write to
        std::uint32_t l1_address = dataflow::get_write_ptr(0);

        std::uint32_t stick_size_bytes = C << 1; // C * 2B
        noc_fast_read_set_len(stick_size_bytes);
        // DRAM NOC src address
        std::uint64_t dram_buffer_src_noc_addr = dataflow::get_noc_addr(dram_src_noc_x, dram_src_noc_y, dram_buffer_src_addr_base);
        noc_fast_read_set_src_xy(dram_buffer_src_noc_addr);

        std::uint32_t num_reads_issued = 0; // number of noc reads issued
        // next l1 address to write to for the first tile in the current row of tiles
        std::uint32_t first_tile_write_address = l1_address;
        std::uint32_t num_bytes_in_current_row_of_tiles = 0;
        std::uint32_t num_bytes_in_current_tile_row = 0;
        std::uint32_t tile_column_index = 0;
        // address counters
        // vertical iteration over the activation (valid) rows
        std::uint32_t activation_row_start_address = 0; // increments by vertical stride
        // horizontal iteration over the activation row
        std::uint32_t overlapping_activation_top_left_address; // increments by horizontal stride
        // vertical iteration over the filter rows
        std::uint32_t filter_row_start_address; // increments by vertical dilation (ignored for now)
        // horizontal iteration over the filter row
        std::uint32_t filter_datum_address; // increments by horizontal dilation (ignored for now)

        // to keep track of the dram address to read from
        std::uint32_t dram_buffer_src_addr;
        for(std::uint32_t n = 0; n < N; n++) {
            for(std::uint32_t h = 0; h < H - (R - 1); h=h+U) {
                overlapping_activation_top_left_address = activation_row_start_address;
                for(std::uint32_t w = 0; w < W - (S - 1); w=w+V) {
                    filter_row_start_address = overlapping_activation_top_left_address;
                    for(std::uint32_t r = 0; r < R; r += 1) {
                        filter_datum_address = filter_row_start_address;
                        for(std::uint32_t s = 0; s < S; s += 1) {
                            dram_buffer_src_addr = dram_buffer_src_addr_base + filter_datum_address;
                            noc_fast_read(dram_buffer_src_addr, l1_address);

                            num_reads_issued += 1;
                            num_bytes_in_current_row_of_tiles += stick_size_bytes;
                            num_bytes_in_current_tile_row += stick_size_bytes;
                            // if we have not finished writing a row within the tile
                            if(num_bytes_in_current_tile_row != 64) {
                                l1_address += stick_size_bytes;
                            }
                            // if we have finished writing a row within the tile, and we need to move to the next tile
                            else if(tile_column_index != (num_tiles_c - 1)) {
                                l1_address += stick_size_bytes + 1984; // increment stick_size_bytes to complete the row, then jump 31 rows to the next tile
                                num_bytes_in_current_tile_row = 0;
                                tile_column_index += 1;
                            }
                            // if we have finished writing a row within the last tile of the current row of tiles, we have to jump back to the first tile write address
                            else if(num_bytes_in_current_row_of_tiles != num_bytes_per_row_of_tiles) {
                                first_tile_write_address += 64;
                                l1_address = first_tile_write_address; // jump back to the first tile write address
                                tile_column_index = 0;
                                num_bytes_in_current_tile_row = 0;
                            }
                            // Once done copying a row of tiles, now wait for reads to get flushed then push tiles into stream
                            else {
                                dataflow::cb_reserve_back(0, num_tiles_c);
                                noc_fast_read_inc_num_issued(num_reads_issued);
                                dataflow::noc_async_read_barrier();
                                dataflow::cb_push_back(0, num_tiles_c);
                                l1_address = dataflow::get_write_ptr(0);
                                first_tile_write_address = l1_address;

                                num_reads_issued = 0;
                                num_bytes_in_current_row_of_tiles = 0;
                                num_bytes_in_current_tile_row = 0;
                                tile_column_index = 0;
                            }
                            filter_datum_address += input_horizontal_stride_bytes;
                        } // s
                        filter_row_start_address += input_vertical_stride_bytes;
                    } // r
                    overlapping_activation_top_left_address += input_horizontal_stride_bytes;
                } // w
                activation_row_start_address += input_vertical_stride_bytes;
            } // h
        } // n
    }
}
