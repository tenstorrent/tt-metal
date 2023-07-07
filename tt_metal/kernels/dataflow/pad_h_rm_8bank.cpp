#include <cstdlib>
#include "dataflow_kernel_api.h"
//#include "debug_print.h"

void kernel_main() {
    // Kernel args
    // This kernel accepts a RM row-interleaved tensor laid out as NC,H,(Wt*32)-RM
    // H should be < 32 at the moment
    // It will write out a tensor NC,32,Wt*32

    // Note: this kernel is written with maximum simplicity in mind and (deliberately) doesn't pursue performance

    uint32_t src_addr     = get_arg_val<uint32_t>(0);
    uint32_t dst_addr     = get_arg_val<uint32_t>(1);
    uint32_t NC           = get_arg_val<uint32_t>(2);
    uint32_t H            = get_arg_val<uint32_t>(3);
    uint32_t paddedH      = get_arg_val<uint32_t>(4);
    uint32_t W            = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    // How many bytes along a row in the original tensor
    uint32_t num_bytes_per_tile = get_tile_size(cb_id_in0);
    uint32_t num_bytes_per_tile_row = 64;
    uint32_t Wt = (W>>5);

    // Variables
    uint64_t replicate_dest_addr;
    uint32_t start_dram_addr_offset_for_tensor_row = 0;

    dataflow::cb_reserve_back(cb_id_in0, 16); // in this kernel we are not pushing anything into CBs, just using the space
    dataflow::cb_reserve_back(cb_id_in1, 16);
    uint32_t l1_tmp_addr = dataflow::get_write_ptr(cb_id_in0);
    uint32_t l1_zeros_addr = dataflow::get_write_ptr(cb_id_in1);
    constexpr uint32_t num_elems0 = (2048<<4) / sizeof(uint32_t); // tile size*16 tiles/4
    volatile uint32_t* zeros = reinterpret_cast<volatile uint32_t*>(l1_zeros_addr);
    for (uint32_t iz = 0; iz < num_elems0; iz++)
        zeros[iz] = 0;

    uint32_t nch_src = 0;
    uint32_t nch_dst = 0;
    // input is NCH(Wt*32) unpadded RM
    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t h = 0; h < paddedH; h++) {
            if (h < H) {
                uint64_t src_noc_addr = dataflow::get_noc_addr_rm(nch_src, 0, src_addr, 8, W);
                //DPRINT << "nc=" << nc << "h=" << h << ENDL();
                //DPRINT << "src_addr=" << uint32_t(src_noc_addr>>32) << "," << uint32_t(src_noc_addr&0xffffFFFF) << ENDL();
                dataflow::noc_async_read(src_noc_addr, l1_tmp_addr, (W<<1)); // TODO(AP): segment this read
                dataflow::noc_async_read_barrier();
                nch_src ++;
            }

            uint64_t dst_noc_addr = dataflow::get_noc_addr_rm(nch_dst, 0, dst_addr, 8, W);
                //DPRINT << "  dst_addr=" << uint32_t(dst_noc_addr>>32) << "," << uint32_t(dst_noc_addr&0xffffFFFF) << ENDL();
            if (h < H) {
                dataflow::noc_async_write(l1_tmp_addr, dst_noc_addr, (W<<1)); // TODO(AP): segment this write
            } else {
                dataflow::noc_async_write(l1_zeros_addr, dst_noc_addr, (W<<1)); // TODO(AP): segment this write
            }
            dataflow::noc_async_write_barrier();
            nch_dst ++;
        } // h<paddedH
    } // nc
}
