#include <stdint.h>
#include "dataflow_api.h"

#include "debug_print.h"

void generate_scaler() {
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
    constexpr uint32_t cb_in_3 = 3;
    uint32_t eps = get_arg_val<uint32_t>(9);
    union { float f; uint32_t u; } u; u.u = eps;
    //DPRINT << "epsilon = " << F32(u.f) << ENDL();
    cb_reserve_back(cb_in_3, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_in_3));
    for (int j = 0; j < 1024; j++)
        ptr[j] = uint16_t(0);

    for (int k = 0; k < 4; k+=2)
    for (int j = 0; j < 16; j++)
        ptr[k*256 + j*16] = uint16_t(u.u>>16);
    cb_push_back(cb_in_3, 1);
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
        //DPRINT << "NC ngt=" << num_gamma_tiles << " nbt=" << num_beta_tiles << ENDL();
        constexpr uint32_t cb_id_gamma = 5;
        constexpr uint32_t cb_id_beta = 6;
        InterleavedPow2AddrGen addrg {gamma_addr, 8, 3, 11}, addrb {beta_addr, 8, 3, 11};
    #endif

    const InterleavedPow2AddrGen s = {
        .bank_base_address = src_addr,
        .num_used_banks = 8,
        .log_base_2_of_num_used_banks = 3,
        .log_base_2_of_bank_unit_size = 11
    };

    #if GENERATE_SCALER
    // TODO(AP): cleanup, probably with named args/param pack/reflection.
    generate_scaler();
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
            //DPRINT << "L1ADDR=" << l1_write_addr << " " << tile_bytes << ENDL();
        for (uint32_t r = 0; r<rem; r++) {
            uint64_t src_noc_addr = get_noc_addr(i+r+tile_offset, s); // not contiguous for sequential r, can be banked
            auto addr = l1_write_addr + (r<<11);
            noc_async_read(src_noc_addr, addr, tile_bytes); // TODO(AP): data type size
            //DPRINT << "  dest_addr=" << addr << ENDL();
        }
        noc_async_read_barrier();
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
            //DPRINT << "  NBW= " << num_beta_written << ENDL();
            cb_push_back(cb_id_beta, blk);
        }
        #endif
    }
}
