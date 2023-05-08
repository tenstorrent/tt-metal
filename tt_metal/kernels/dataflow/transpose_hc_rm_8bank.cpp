#include <cstdlib>
#include "dataflow_api.h"
//#include "debug_print.h"

int __multiply(int n, int m) {
    int res = 0, count = 0;
    while (m) {
        if ((m & 1) == 1)
            res += (n << count);
        count++;
        m >>= 1;
    }
    return res;
}

int __min(int a, int b) {
    if (a < b)
        return a;
    else
        return b;
}

inline __attribute__((always_inline))
std::uint64_t get_noc_addr_rm(
    uint32_t row, uint32_t col, uint32_t bank_base_address, uint32_t num_used_banks, uint32_t W)
{
    uint32_t bank_id = row & (num_used_banks - 1);
    uint32_t dram_x = dram_bank_to_noc_x[bank_id];
    uint32_t dram_y = dram_bank_to_noc_y[bank_id];
    // >>3 is because of 8 banks
    // TODO(AP): replace multiply with increments
    uint32_t dram_addr = bank_base_address + (__multiply(row>>3, (W<<1))) + (col<<1);
    std::uint64_t noc_addr = get_noc_addr(dram_x, dram_y, dram_addr);
    return noc_addr;
}

void kernel_main() {
    // Kernel args
    // This kernel accepts a RM row-interleaved tensor laid out as NC,H,(Wt*32)-RM
    // H should be < 32 at the moment
    // It will write out a tensor NC,32,Wt*32

    // Note: this kernel is written with maximum simplicity in mind and (deliberately) doesn't pursue performance

    uint32_t src_addr     = get_arg_val<uint32_t>(0);
    uint32_t dst_addr     = get_arg_val<uint32_t>(1);
    uint32_t N            = get_arg_val<uint32_t>(2);
    uint32_t C            = get_arg_val<uint32_t>(3);
    uint32_t H            = get_arg_val<uint32_t>(4);
    uint32_t W            = get_arg_val<uint32_t>(5);
    uint32_t CH           = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    // How many bytes along a row in the original tensor
    uint32_t num_bytes_per_tile = get_tile_size(cb_id_in0);
    uint32_t num_bytes_per_tile_row = 64;
    uint32_t Wt = (W>>5);

    // Variables
    uint64_t replicate_dest_addr;
    uint32_t start_dram_addr_offset_for_tensor_row = 0;

    cb_reserve_back(cb_id_in0, 16); // in this kernel we are not pushing anything into CBs, just using the space
    uint32_t l1_tmp_addr = get_write_ptr(cb_id_in0);

    //DPRINT << "----- N=" << N << " C=" << C << " H=" << H << " W=" << W << ENDL();

    uint32_t nch_src = 0;
    uint32_t nch_dst = 0;
    // input is row major, we iterate over output address and compute the input address
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t h = 0; h < H; h++) {
            for (uint32_t c = 0; c < C; c++) {
                uint64_t src_noc_addr = get_noc_addr_rm(nch_src, 0, src_addr, 8, W);
                //DPRINT << "nch_dst=" << nch_dst << ENDL();
                //DPRINT << "nch_src=" << nch_src << ENDL();
                //DPRINT << "src_addr=" << uint32_t(src_noc_addr>>32) << "," << uint32_t(src_noc_addr&0xffffFFFF) << ENDL();
                noc_async_read(src_noc_addr, l1_tmp_addr, (W<<1)); // TODO(AP): segment this read
                noc_async_read_barrier();

                uint64_t dst_noc_addr = get_noc_addr_rm(nch_dst, 0, dst_addr, 8, W);

                //DPRINT << "  dst_addr=" << uint32_t(dst_noc_addr>>32) << "," << uint32_t(dst_noc_addr&0xffffFFFF) << ENDL();
                noc_async_write(l1_tmp_addr, dst_noc_addr, (W<<1)); // TODO(AP): segment this write
                noc_async_write_barrier();
                nch_dst ++;
                nch_src += H;
            } // c loop
            nch_src -= CH;
            nch_src += 1; // h increment
        } // h loop
        nch_src -= H;  // undo +1 H times
        nch_src += CH; // n increment for NCH tensor with W-sized elements
    } // n loop
}
