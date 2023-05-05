#include <stdint.h>
#include "dataflow_api.h"

#include "debug_print.h"

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

void generate_epsilon() {
    constexpr uint32_t eps_cb_id = 3;
    union { float f; uint32_t u; } u; u.u = get_arg_val<uint32_t>(9);
    //DPRINT << "epsilon = " << F32(u.f) << ENDL();
    cb_reserve_back(eps_cb_id, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(eps_cb_id));
    for (int j = 0; j < 1024; j++)
        ptr[j] = uint16_t(0);

    for (int k = 0; k < 4; k+=2)
    for (int j = 0; j < 16; j++)
        ptr[k*256 + j*16] = uint16_t(u.u>>16);
    cb_push_back(eps_cb_id, 1);
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

void generate_col_ones() {
    constexpr uint32_t cb_in_4 = 4;
    union { float f; uint32_t u; } u; u.u = 0x3f800000;
    //DPRINT << "one = " << F32(u.f) << ENDL();
    cb_reserve_back(cb_in_4, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_in_4));
    for (int j = 0; j < 1024; j++)
        ptr[j] = uint16_t(0);

    for (int k = 0; k < 4; k+=2)
    for (int j = 0; j < 16; j++)
        ptr[k*256 + j*16] = uint16_t(u.u>>16);
    cb_push_back(cb_in_4, 1);
}

void kernel_main() {
    auto s16 = SliceRange::hw0_32_16();

    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(3); // same arg index as in reader_unary and in reader_unary_transpose_wh_8bank

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    #if GAMMA_BETA
        uint32_t num_gamma_written = 0, num_beta_written = 0;
        uint32_t num_gamma_tiles = get_arg_val<uint32_t>(10); // gamma, beta are (32,W), bcast from W->NCH , result = x*gamma+beta
        uint32_t gamma_addr = get_arg_val<uint32_t>(11);
        uint32_t num_beta_tiles = get_arg_val<uint32_t>(12);
        uint32_t beta_addr = get_arg_val<uint32_t>(13);
        constexpr uint32_t cb_id_gamma = 5;
        constexpr uint32_t cb_id_beta = 6;
        InterleavedPow2AddrGen<true> addrg {gamma_addr, 11}, addrb {beta_addr, 11};
    #endif

    #if FUSED_SCALE_MASK
    uint32_t partHt = get_arg_val<uint32_t>(5);
    uint32_t Wt = get_arg_val<uint32_t>(6);
    InterleavedPow2AddrGen<true> addr_mask {get_arg_val<uint32_t>(7), 11};

    uint32_t ht = 0, wt = 0, nc = 0, wtblk = 0;
    constexpr uint32_t cb_id_attn = 4;
    generate_inv_sqrt_hw_bcast_tile();
    #endif

    constexpr bool read_from_dram =
    #ifdef get_compile_time_arg_val
    get_compile_time_arg_val(0)
    #else
    true
    #endif
    ;

    const InterleavedPow2AddrGen<read_from_dram> s = { src_addr, 11 };

    #if GENERATE_BCAST_SCALER
    // TODO(AP): cleanup, probably with named args/param pack/reflection.
    generate_bcast_scaler();
    constexpr uint32_t blk = BLOCK_SIZE; // 8 for perf for fused kernels, this imposes constraints on Wt
    #else
    constexpr uint32_t blk = 1; // 1 for correctness for unfused kernels
    #endif
    #if GENERATE_EPSILON // for LN
    generate_epsilon();
    generate_col_ones();
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
            // DPRINT << i+r+tile_offset << ENDL();

            uint64_t src_noc_addr = get_noc_addr(i+r+tile_offset, s); // not contiguous for sequential r, can be banked
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
        //DPRINT << "NC in0 pushing " << rem << ENDL();
        cb_push_back(cb_id_in0, rem);

        #if GAMMA_BETA // TODO(AP): refactor
        if (num_gamma_written < num_gamma_tiles) {
            cb_reserve_back(cb_id_gamma, blk);
            l1_write_addr = get_write_ptr(cb_id_gamma);
            for (uint32_t r = 0; r<blk; r++) {
                uint64_t src_noc_addr = get_noc_addr(num_gamma_written+r, addrg); // not contiguous for sequential r, can be banked
                auto addr = l1_write_addr + (r<<11);
                noc_async_read(src_noc_addr, addr, tile_bytes);
                //DPRINT << "  gamma dest_addr=" << addr << ENDL();
            }
            num_gamma_written += blk;
            noc_async_read_barrier();
            //DPRINT << "  NGW= " << num_gamma_written << ENDL();
            cb_push_back(cb_id_gamma, blk);
        }

        if (num_beta_written < num_beta_tiles) {
            cb_reserve_back(cb_id_beta, blk);
            l1_write_addr = get_write_ptr(cb_id_beta);
            for (uint32_t r = 0; r<blk; r++) {
                uint64_t src_noc_addr = get_noc_addr(num_beta_written+r, addrb); // not contiguous for sequential r, can be banked
                auto addr = l1_write_addr + (r<<11);
                noc_async_read(src_noc_addr, addr, tile_bytes);
                //DPRINT << "  beta dest_addr=" << addr << ENDL();
            }
            num_beta_written += blk;
            noc_async_read_barrier();
            //DPRINT << "  NBW= " << num_beta_written << ENDL();
            cb_push_back(cb_id_beta, blk);
        }
        #endif

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
