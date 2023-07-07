#include <cstdint>
#define OPT_READ 1
/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or other RISCs
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */
void kernel_main() {
    std::uint32_t dram_buffer_src_addr_base  = get_arg_val<uint32_t>(0);
    std::uint32_t dram_src_noc_x             = get_arg_val<uint32_t>(1);
    std::uint32_t dram_src_noc_y             = get_arg_val<uint32_t>(2);

    std::uint32_t l1_buffer_dst_addr_base    = get_arg_val<uint32_t>(3);
    std::uint32_t N                          = get_arg_val<uint32_t>(4);
    std::uint32_t C                          = get_arg_val<uint32_t>(5);
    std::uint32_t H                          = get_arg_val<uint32_t>(6);
    std::uint32_t W                          = get_arg_val<uint32_t>(7);
    std::uint32_t R                          = get_arg_val<uint32_t>(8);
    std::uint32_t S                          = get_arg_val<uint32_t>(9);
    std::uint32_t U                          = get_arg_val<uint32_t>(10);
    std::uint32_t V                          = get_arg_val<uint32_t>(11);
    // The product of W and U, i.e. the offset to move to the next row after doing stride U
    std::uint32_t WU                         = get_arg_val<uint32_t>(12);
    std::uint32_t log2_of_C                  = get_arg_val<uint32_t>(13);
    std::uint32_t log2_of_bytes_per_datum    = get_arg_val<uint32_t>(14);
    std::uint32_t num_repetitions            = get_arg_val<uint32_t>(15);


    for(std::uint32_t i = 0; i < num_repetitions; i++) {

    #ifndef OPT_READ
        // TODO: fix hardcoding. << 2 because each value is 4Bytes
        std::uint32_t chunk_size = C << log2_of_bytes_per_datum; // z * 4B

        // the cordinate of the beginning of the current row
        std::uint32_t row_start = 0;
        // to iterate over the R rows of the filter
        std::uint32_t idx = 0;
        // l1 address to write to
        std::uint32_t l1_address = l1_buffer_dst_addr_base;
        // to keep track of the dram address to read from
        std::uint32_t dram_buffer_src_addr;
        std::uint64_t dram_buffer_src_noc_addr;

        for(std::uint32_t n = 0; n < N; n++) {
            for(std::uint32_t h = 0; h < H - (R - 1); h=h+U) {
                for(std::uint32_t w = 0; w < W - (S - 1); w=w+V) {
                    idx = row_start + w;
                    for(std::uint32_t r = 0; r < R; r += 1) {
                        for(std::uint32_t s = 0; s < S; s += 1) {
                        // shift (idx+s) by (log2_of_C + 2) means multiply by (C * 4B)
                        dram_buffer_src_addr = dram_buffer_src_addr_base + ((idx + s) << (log2_of_C + log2_of_bytes_per_datum));
                        // DRAM NOC src address
                        dram_buffer_src_noc_addr = dataflow::get_noc_addr(dram_src_noc_x, dram_src_noc_y, dram_buffer_src_addr);

                        dataflow::noc_async_read(dram_buffer_src_noc_addr, l1_address, chunk_size);
                        l1_address = l1_address + chunk_size;
                        } // s
                        idx = idx + W;
                    } // r
                } // w
                row_start = row_start + WU;
            } // h
        } // n

        // wait all reads flushed (ie received)
        dataflow::noc_async_read_barrier();
    #else
        // TODO: fix hardcoding. << 2 because each value is 4Bytes
        std::uint32_t chunk_size = C << log2_of_bytes_per_datum; // z * 4B
        noc_fast_read_set_len(chunk_size);
        // DRAM NOC src address
        std::uint64_t dram_buffer_src_noc_addr = dataflow::get_noc_addr(dram_src_noc_x, dram_src_noc_y, dram_buffer_src_addr_base);
        noc_fast_read_set_src_xy(dram_buffer_src_noc_addr);

        // the cordinate of the beginning of the current row
        std::uint32_t row_start = 0;
        // to iterate over the R rows of the filter
        std::uint32_t idx = 0;
        // l1 address to write to
        std::uint32_t l1_address = l1_buffer_dst_addr_base;
        // to keep track of the dram address to read from
        std::uint32_t dram_buffer_src_addr;

        std::uint32_t num_issued = 0;
        for(std::uint32_t n = 0; n < N; n++) {
            for(std::uint32_t h = 0; h < H - (R - 1); h=h+U) {
                for(std::uint32_t w = 0; w < W - (S - 1); w=w+V) {
                    idx = row_start + w;
                    for(std::uint32_t r = 0; r < R; r += 1) {
                        for(std::uint32_t s = 0; s < S; s += 1) {
                            // shift (idx+s) by (log2_of_C + 2) means multiply by (C * 4B)
                            dram_buffer_src_addr = dram_buffer_src_addr_base + ((idx + s) << (log2_of_C + log2_of_bytes_per_datum));

                            noc_fast_read(dram_buffer_src_addr, l1_address);
                            dram_buffer_src_addr += chunk_size;
                            l1_address += chunk_size;
                            num_issued += 1;
                        } // s
                        idx = idx + W;
                    } // r
                } // w
                row_start = row_start + WU;
            } // h
        } // n

        noc_fast_read_inc_num_issued(num_issued);
        // wait all reads from all transactions to be flushed (ie received)
        dataflow::noc_async_read_barrier();

    #endif
    }
}
