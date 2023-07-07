#include "dataflow_kernel_api.h"


void generate_bcast_scaler() {
    constexpr uint32_t cb_in_2 = 2;
    uint32_t scaler = get_arg_val<uint32_t>(8);
    union { float f; uint32_t u; } u; u.u = scaler;
    //DPRINT << "basic Scaler = " << F32(u.f) << ENDL();
    dataflow::cb_reserve_back(cb_in_2, 1);
    auto ptr = reinterpret_cast<uint16_t*>(dataflow::get_write_ptr(cb_in_2));
    for (int j = 0; j < 1024; j++)
        ptr[j] = uint16_t(0);

    for (int k = 0; k < 4; k++)
    for (int j = 0; j < 16; j++)
        ptr[k*256 + j] = uint16_t(u.u>>16);
    dataflow::cb_push_back(cb_in_2, 1);
}

// HW-bcast scale for fused scale-attn-softmax
void generate_inv_sqrt_hw_bcast_tile() {
    constexpr uint32_t scale_cb_id = 3;
    uint32_t u = get_arg_val<uint32_t>(2);
    dataflow::cb_reserve_back(scale_cb_id, 1);
    auto ptr = reinterpret_cast<uint16_t*>(dataflow::get_write_ptr(scale_cb_id));
    // for (int j = 0; j < 1024; j++)
    //     ptr[j] = uint16_t(0);
    ptr[0] = u>>16;
    dataflow::cb_push_back(scale_cb_id, 1);
}

void kernel_main() {

    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(3); // same arg index as in reader_unary and in reader_unary_transpose_wh_8bank
    uint32_t tile_offset = get_arg_val<uint32_t>(4);

    constexpr DataFormat src0_data_format = static_cast<DataFormat>(get_compile_time_arg_val(0));
    constexpr bool src0_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t blk = get_compile_time_arg_val(2); // 8 for perf for fused kernels, this imposes constraints on Wt
    constexpr uint32_t cb_id_in0 = 0, cb_id_in1 = 1;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);

    #if FUSED_SCALE_MASK
    uint32_t partHt = get_arg_val<uint32_t>(5);
    uint32_t Wt = get_arg_val<uint32_t>(6);
    uint32_t mask_addr = get_arg_val<uint32_t>(7);
    constexpr DataFormat mask_data_format = static_cast<DataFormat>(get_compile_time_arg_val(3));
    constexpr bool mask_is_dram = get_compile_time_arg_val(4) == 1;

    constexpr uint32_t cb_id_attn = 4;
    uint32_t mask_tile_bytes = get_tile_size(cb_id_attn);

    const dataflow::InterleavedAddrGenFast<mask_is_dram> addr_mask = {
        .bank_base_address = mask_addr,
        .page_size = mask_tile_bytes,
        .data_format = mask_data_format
    };


    uint32_t ht = 0, wt = 0, nc = 0;
    generate_inv_sqrt_hw_bcast_tile();
    #endif

    const dataflow::InterleavedAddrGenFast<src0_is_dram> src_a = {
        .bank_base_address = src_addr,
        .page_size = src0_tile_bytes,
        .data_format = src0_data_format
    };


    // TODO(AP): cleanup, probably with named args/param pack/reflection.
    generate_bcast_scaler();

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t i_tile = 0;
    uint32_t curr_tile = tile_offset;
    for (uint32_t i = 0; i<num_tiles; i += blk) {
        uint32_t rem = blk; // (i + blk > num_tiles) ? num_tiles - i : blk;
        dataflow::cb_reserve_back(cb_id_in0, rem);
        uint32_t l1_write_addr = dataflow::get_write_ptr(cb_id_in0);

        for (uint32_t r = 0; r<rem; r++) {
            dataflow::noc_async_read_tile(curr_tile, src_a, l1_write_addr); // TODO(AP): data type size
            curr_tile++;
            l1_write_addr += src0_tile_bytes;
        }
        dataflow::noc_async_read_barrier();
        dataflow::cb_push_back(cb_id_in0, rem);

        #if FUSED_SCALE_MASK
        // Recall that the total attention tensor size in tiles is NC,1,Wt
        // For fused scale-mask softmax we write Wt attention tiles for every partHt*Wt
        // of slice of tensor that was assigned to our core, then we skip to next batch
        if (ht == 0) {
            // This is only executed every blk wts
            dataflow::cb_reserve_back(cb_id_attn, blk);
            l1_write_addr = dataflow::get_write_ptr(cb_id_attn);
            for (uint32_t wb = 0; wb<blk; wb++) {
                dataflow::noc_async_read_tile(wt+wb, addr_mask, l1_write_addr);
                l1_write_addr += mask_tile_bytes;
            }
            dataflow::noc_async_read_barrier();
            dataflow::cb_push_back(cb_id_attn, blk);
        }

        wt += blk; // the i<numtiles loop is using blk stride
        // TODO(AP): could be easier to just structure the loop as multiple loops
        // Probably split the kernel
        if (wt == Wt) {
            wt = 0; ht++;
            if (ht == partHt) { nc ++; ht = 0; }
        }
        #endif // FUSED_SCALE_MASK
    }
}
