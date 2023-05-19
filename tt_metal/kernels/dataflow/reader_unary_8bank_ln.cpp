#define TILE_OFFSET get_arg_val<uint32_t>(4)

#ifndef BLOCK_SIZE // can be defined via add_define
#error "Block size must be defined"
#endif

#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#define MT(id) // kernel_profiler::mark_time(id);

#include <stdint.h>
#include "dataflow_api.h"

#include "debug_print.h"
#undef DPRINT

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
    //auto s16 = SliceRange::hw0_32_16();
    nc_set_core_xy();

    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t NCHt      = get_arg_val<uint32_t>(1);
    uint32_t Wt        = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = 0, cb_id_in1 = 1;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    uint32_t num_gamma_written = 0, num_beta_written = 0;
    uint32_t num_gamma_tiles = get_arg_val<uint32_t>(10); // gamma, beta are (1,1,32,W), bcast from W->NCH , result = x*gamma+beta
    uint32_t gamma_addr = get_arg_val<uint32_t>(11);
    uint32_t num_beta_tiles = get_arg_val<uint32_t>(12);
    uint32_t beta_addr = get_arg_val<uint32_t>(13);
    constexpr uint32_t cb_id_gamma = 5;
    constexpr uint32_t cb_id_beta = 6;
    InterleavedPow2AddrGen<GAMMA_DRAM> addrg {gamma_addr, 11};
    InterleavedPow2AddrGen<BETA_DRAM> addrb {beta_addr, 11};

    //DPRINT << "NGT=" << num_gamma_tiles << " NBT=" << num_beta_tiles << ENDL();

    #ifdef KERNEL_COMPILE_TIME_ARG_0
    constexpr bool read_from_dram = get_compile_time_arg_val(0);
    #else
    constexpr bool read_from_dram = true;
    #endif

    const InterleavedPow2AddrGen<A_DRAM> src_a = { src_addr, 11 };
    #ifdef FUSE_PRE_ADD
    const InterleavedPow2AddrGen<B_DRAM> src_b = { get_arg_val<uint32_t>(14), 11 };
    #endif

    constexpr uint32_t blk = BLOCK_SIZE; // 8 for perf for fused kernels, this imposes constraints on Wt
    generate_bcast_scaler();
    generate_epsilon();
    generate_col_ones();

    MT(100)

    uint32_t tile_offset = TILE_OFFSET;
    //DPRINT << "Reader Tile offset=" << tile_offset << " " << core_x() << core_y() << ENDL();
    //DPRINT << "LN Reader NCHt=" << NCHt << " Wt=" << Wt << " blk=" << blk << "corex=" << core_x() << ENDL();

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t offs = 0;
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        //DPRINT << "ncht= " << ncht << ENDL();
        for (uint32_t wt = 0; wt<Wt; wt += blk) {
            cb_reserve_back(cb_id_in0, blk);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

            for (uint32_t r = 0; r<blk; r++) {
                uint64_t src_noc_addr = get_noc_addr(offs+wt+r+tile_offset, src_a); // not contiguous for sequential r, can be banked
                auto addr = l1_write_addr + (r<<11);
                noc_async_read(src_noc_addr, addr, tile_bytes); // TODO(AP): data type size
            }
            noc_async_read_barrier();
            //if (core_x() == 2 && core_y() == 1 && ncht == 0 && wt == 0) DPRINT << "TILE_OFFSET: " << TILE_OFFSET << ENDL();
            //if (core_x() == 2 && core_y() == 1 && ncht == 0 && wt == 0) DPRINT << "READER A: " << TSLICE(cb_id_in0, 0, SliceRange::hw0_32_16());
            cb_push_back(cb_id_in0, blk);
            //DPRINT << "LN Reader A pushed " << blk << " wt=" << wt << ENDL();

            #ifdef FUSE_PRE_ADD
            // TODO(AP): refactor the ifdefs
            cb_reserve_back(cb_id_in1, blk);
            l1_write_addr = get_write_ptr(cb_id_in1);
            for (uint32_t r = 0; r<blk; r++) {
                uint64_t src_noc_addr = get_noc_addr(offs+wt+r+tile_offset, src_b); // not contiguous for sequential r, can be banked
                auto addr = l1_write_addr + (r<<11);
                noc_async_read(src_noc_addr, addr, tile_bytes); // TODO(AP): data type size
            }
            noc_async_read_barrier();
            //DPRINT << "NC in1 pushing tiles " << blk << ENDL();
            //if (core_x() == 2 && core_y() == 1 && ncht == 0 && wt == 0) DPRINT << "TILE_OFFSET: " << TILE_OFFSET << ENDL();
            //if (core_x() == 2 && core_y() == 1 && ncht == 0 && wt == 0) DPRINT << "READER B: " << TSLICE(cb_id_in1, 0, SliceRange::hw0_32_16());
            cb_push_back(cb_id_in1, blk);
            #endif
        } // wt loop

        for (uint32_t wt = 0; wt<Wt; wt += blk) {
            if (num_gamma_written < num_gamma_tiles) {
                cb_reserve_back(cb_id_gamma, blk);
                uint32_t l1_write_addr = get_write_ptr(cb_id_gamma);
                for (uint32_t r = 0; r<blk; r++) {
                    uint64_t src_noc_addr = get_noc_addr(num_gamma_written+r, addrg); // not contiguous for sequential r, can be banked
                    auto addr = l1_write_addr + (r<<11);
                    noc_async_read(src_noc_addr, addr, tile_bytes);
                    //DPRINT << "  gamma dest_addr=" << addr << ENDL();
                }
                noc_async_read_barrier();
                cb_push_back(cb_id_gamma, blk);
                num_gamma_written += blk;
                //DPRINT << "  NGW= " << num_gamma_written << ENDL();
            }

            if (num_beta_written < num_beta_tiles) {
                cb_reserve_back(cb_id_beta, blk);
                uint32_t l1_write_addr = get_write_ptr(cb_id_beta);
                for (uint32_t r = 0; r<blk; r++) {
                    uint64_t src_noc_addr = get_noc_addr(num_beta_written+r, addrb); // not contiguous for sequential r, can be banked
                    auto addr = l1_write_addr + (r<<11);
                    noc_async_read(src_noc_addr, addr, tile_bytes);
                    //DPRINT << "  beta dest_addr=" << addr << ENDL();
                }
                noc_async_read_barrier();
                cb_push_back(cb_id_beta, blk);
                num_beta_written += blk;
                //DPRINT << "  NBW= " << num_beta_written << ENDL();
            }
        } // wt loop
        offs += Wt;
    } // ncht loop
}
