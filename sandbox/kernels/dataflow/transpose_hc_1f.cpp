// backup of the old single-tile single-subtile version of transpose_hc kernel
#include <stdint.h>
#include "dataflow_api.h"

using u32 = std::uint32_t;

// tile index to address
inline u32 TADDR(u32 ti) {
    return ti << 11;
}

void kernel_main() {
    u32 dram_buffer_src_addr  = *((volatile u32*)(L1_ARG_BASE));
    u32 dram_src_noc_x        = *((volatile u32*)(L1_ARG_BASE+4));
    u32 dram_src_noc_y        = *((volatile u32*)(L1_ARG_BASE+8));
    u32 W                     = *((volatile u32*)(L1_ARG_BASE+12));
    u32 H                     = *((volatile u32*)(L1_ARG_BASE+16));
    u32 C                     = *((volatile u32*)(L1_ARG_BASE+20));
    u32 HW                    = *((volatile u32*)(L1_ARG_BASE+24));

    auto WT = (W >> 5); // number of tiles in W
    auto HT = (H >> 5); // number of tiles in H
    auto CT = (C >> 5); // number of tiles in C
    auto HTWT = (HW >> 10); // product of HT*WT
    auto HW2 = (HW << 1); // HW stride in bytes
    constexpr u32 TILE_LINE_BYTES = 32*2;
    constexpr u32 onetile = 1;
    constexpr u32 operand0 = 0;
    std::uint64_t input_tensor_dram_addr =
        NOC_XY_ADDR(NOC_X(dram_src_noc_x), NOC_Y(dram_src_noc_y), dram_buffer_src_addr);

    // The basic idea here is to iterate over output tiles (that will be over CT,WT) and H
    // this will generate a linearly incremented output address in the inner loop
    // we then reverse map this linear dest address to src address
    u32 htWT = 0;
    for (u32 h = 0; h < H; h++) {
        u32 ctoffs = 0;
        for (u32 ct = 0; ct < CT; ct++) {
            for (u32 wt = 0; wt < WT; wt++) {
                // what is the source address for the current tile?
                // c32 = intra-C-tile loop
                // every 32 C's acquire a new output tile address
                cb_reserve_back(operand0, onetile);

                u32 dest_tr0_l1 = get_write_ptr(operand0);
                u32 c32offs = 0;
                for (u32 c32 = 0; c32 < 32; c32++) {
                    // A version with multiplications would look like this:
                    // First convert C-tile to c
                    // auto c = (ct << 5) + c32;
                    // The formula for tilized layout address from chw is:
                    // src_addr = c*HW2 + (ht*WT + wt)*2048 + (h32*32 + w32)
                    //auto htWT = loopmul(h >> 5, WT);
                    auto h32 = (h&31);
                    auto src_offs = ctoffs + c32offs + TADDR(htWT + wt) + (h32<<6); // bytes offset
                    auto src_addr = input_tensor_dram_addr + src_offs;
                    c32offs += HW2;

                    // this starts async NOC dma from DRAM to TR0_L1 buffer
                    ncrisc_noc_fast_read_any_len(loading_noc, NCRISC_RD_CMD_BUF,
                                                src_addr, dest_tr0_l1, TILE_LINE_BYTES);
                    // the output address is just linearly incremented
                    dest_tr0_l1 += TILE_LINE_BYTES;
                }

                // block on all outstanding noc DMA requests to complete
                while (!ncrisc_noc_reads_flushed(loading_noc));

                // notifies the unpacker that the buffer is populated
                cb_push_back(operand0, onetile);
            }
            ctoffs += (HW2 << 5); // since we increment ct, we need to mlutiply by 32
        }
        // multiplication-free computation of ht*WT, since ht = h/32
        if ((h&31) == 31)
            htWT += WT;
    }
}
