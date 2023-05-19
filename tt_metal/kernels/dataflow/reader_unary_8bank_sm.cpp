#define GENERATE_BCAST_SCALER 1
#define TILE_OFFSET get_arg_val<uint32_t>(4)

#ifndef BLOCK_SIZE // can be alread defined via add_define
#error "Block size must be defined"
#endif

void generate_bcast_scaler() {
    constexpr uint32_t cb_in_2 = 2;
    uint32_t scaler = get_arg_val<uint32_t>(8);
    union { float f; uint32_t u; } u; u.u = scaler;
    //DPRINT << "basic Scaler = " << F32(u.f) << ENDL();
    cb_reserve_back(cb_in_2, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_in_2));
    for (int j = 0; j < 1024; j++)
        ptr[j] = uint16_t(0);

    for (int k = 0; k < 4; k++)
    for (int j = 0; j < 16; j++)
        ptr[k*256 + j] = uint16_t(u.u>>16);
    cb_push_back(cb_in_2, 1);
}

// HW-bcast scale for fused scale-attn-softmax
void generate_inv_sqrt_hw_bcast_tile() {
    constexpr uint32_t scale_cb_id = 3;
    union { float f; uint32_t u; } u; u.u = get_arg_val<uint32_t>(2);
    //DPRINT << "NC fused mask scale = " << F32(u.f) << ENDL();
    cb_reserve_back(scale_cb_id, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(scale_cb_id));
    for (int j = 0; j < 1024; j++)
        ptr[j] = uint16_t(0);
    ptr[0] = uint16_t(u.u>>16);
    cb_push_back(scale_cb_id, 1);
}

void kernel_main() {
    //auto s16 = SliceRange::hw0_32_16();

    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(3); // same arg index as in reader_unary and in reader_unary_transpose_wh_8bank

    constexpr uint32_t cb_id_in0 = 0, cb_id_in1 = 1;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    #if FUSED_SCALE_MASK
    uint32_t partHt = get_arg_val<uint32_t>(5);
    uint32_t Wt = get_arg_val<uint32_t>(6);
    InterleavedPow2AddrGen<MASK_DRAM> addr_mask {get_arg_val<uint32_t>(7), 11};

    uint32_t ht = 0, wt = 0, nc = 0, wtblk = 0;
    constexpr uint32_t cb_id_attn = 4;
    generate_inv_sqrt_hw_bcast_tile();
    #endif

    const InterleavedPow2AddrGen<A_DRAM> src_a = { src_addr, 11 };

    #if GENERATE_BCAST_SCALER
    // TODO(AP): cleanup, probably with named args/param pack/reflection.
    generate_bcast_scaler();
    constexpr uint32_t blk = BLOCK_SIZE; // 8 for perf for fused kernels, this imposes constraints on Wt
    #else
    constexpr uint32_t blk = 1; // 1 for correctness for unfused kernels
    #endif

    #ifdef TILE_OFFSET
    uint32_t tile_offset = TILE_OFFSET;
    #else
    constexpr uint32_t tile_offset = 0;
    #endif
    //DPRINT << "Reader Tile offset=" << tile_offset << ENDL();

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t i_tile = 0;
    for (uint32_t i = 0; i<num_tiles; i += blk) {
        uint32_t rem = blk; // (i + blk > num_tiles) ? num_tiles - i : blk;
        cb_reserve_back(cb_id_in0, rem);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

        // DPRINT << i << ENDL();
        for (uint32_t r = 0; r<rem; r++) {
            uint64_t src_noc_addr = get_noc_addr(i+r+tile_offset, src_a); // not contiguous for sequential r, can be banked
            auto addr = l1_write_addr + (r<<11);
            // DPRINT << i << ENDL();
            // DPRINT << 'k' << ENDL();
            noc_async_read(src_noc_addr, addr, tile_bytes); // TODO(AP): data type size
            // DPRINT << i << ENDL();
            // DPRINT << 'l' << ENDL();
            //DPRINT << "  dest_addr=" << addr << ENDL();
        }
        // DPRINT << uint(my_x[loading_noc]) << ", " << uint(my_y[loading_noc]) << ENDL();
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, rem);

        #if FUSED_SCALE_MASK
        // Recall that the total attention tensor size in tiles is NC,1,Wt
        // For fused scale-mask softmax we write Wt attention tiles for every partHt*Wt
        // of slice of tensor that was assigned to our core, then we skip to next batch
        if (ht == 0 && wtblk == 0) {
            // This is only executed every blk wts
            cb_reserve_back(cb_id_attn, blk);
            l1_write_addr = get_write_ptr(cb_id_attn);
            for (uint32_t wb = 0; wb<blk; wb++) {
                uint64_t src_noc_addr = get_noc_addr(wt+wb, addr_mask); // not contiguous for sequential r, can be banked
                auto addr = l1_write_addr + (wb<<11);
                noc_async_read(src_noc_addr, addr, tile_bytes);
            }
            noc_async_read_barrier();
            //DPRINT << "NC pushing mask tiles " << blk << ENDL();
            //DPRINT << TSLICE(cb_id_attn, 0, SliceRange::hw041());
            cb_push_back(cb_id_attn, blk);
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
