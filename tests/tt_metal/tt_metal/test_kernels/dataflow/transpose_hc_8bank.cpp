// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

using uint32_t = std::uint32_t;

// tile index to address
inline uint32_t TADDR(uint32_t ti) { return ti << 11; }

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src_bank_id = get_arg_val<uint32_t>(1);
    uint32_t W = get_arg_val<uint32_t>(2);
    uint32_t H = get_arg_val<uint32_t>(3);
    uint32_t C = get_arg_val<uint32_t>(4);
    uint32_t HW = get_arg_val<uint32_t>(5);
    uint32_t N = get_arg_val<uint32_t>(6);
    uint32_t CHW = get_arg_val<uint32_t>(7);

    auto WT = (W >> 5);      // number of tiles in W
    auto HT = (H >> 5);      // number of tiles in H
    auto CT = (C >> 5);      // number of tiles in C
    auto HTWT = (HW >> 10);  // product of HT*WT
    auto HW2 = (HW << 1);    // HW stride in bytes
    auto CHW2 = (CHW << 1);  // batch stride in bytes
    constexpr uint32_t SUBTILE_LINE_BYTES = (16 << 1);
    constexpr uint32_t onetile = 1;
    constexpr uint32_t operand0 = 0;

    // The basic idea here is to iterate over output tiles (that will be over CT,WT) and H
    // this will generate a linearly incremented output address in the inner loop
    // we then reverse map this linear dest address to src address

    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(src_args, src0_addr, 2048);

    uint64_t batch_addr = src0_addr;
    for (uint32_t n = 0; n < N; n++) {
        uint32_t htWT = 0;
        for (uint32_t h = 0; h < H; h++) {
            uint32_t ctoffs = 0;
            for (uint32_t ct = 0; ct < CT; ct++) {
                for (uint32_t wt = 0; wt < WT; wt++) {
                    // what is the source address for the current tile?
                    // c32 = intra-C-tile loop
                    // every 32 C's acquire a new output tile address
                    //    DPRINT << "8B h=" << h << " ct=" << ct << " wt=" << wt << " W=" << W << " HW2=" << HW2 <<
                    //    ENDL();

                    cb_reserve_back(operand0, onetile);

                    uint32_t dest_tr0_l1 = get_write_ptr(operand0);
                    uint32_t save_dest = dest_tr0_l1;
                    uint32_t cSubtileOffs = 0;
                    for (uint32_t sub = 0; sub < 4; sub++) {
                        uint32_t c16offs = cSubtileOffs;
                        for (uint32_t c16 = 0; c16 < 16; c16++) {
                            // In this loop sub, c16 are source subtile, c16
                            // dest in this loop is varying h implicitly via dest address increment

                            // Dest is HCW
                            // We are iterating over it as H Ct Wt-tiles
                            // intra-tile FC16 for F going over 4-subtiles
                            // the source address is (bytes):
                            // src_addr = c*HW2 + (ht*Wt + wt)*1024*2 + f*256*2 + (h16*16 + w16)*2
                            // we have 512 bytes per subtile and 32 bytes per subtile row of 16 elems
                            // here sub<<9 is multiply by 512 which offset in bytes of a subtile
                            // note that dest h is decomposed as h = ht+h32 and htWT is incremented by WT in the outer H
                            // loop
                            auto h32 = (h & 31);
                            // TODO(AP): not really trivial need better comments here
                            auto sub_src_offs = (sub & 1) << 9;  // if dest subtile w==16, add 512 to src subtile offset
                            sub_src_offs +=
                                (((h32 >> 4) << 1)
                                 << 9);  // if intra-tile source h is > 16, add 2*512 to subtile offset
                            // below we only use the lower 4 bits out of 5-bit range for h, shift by 5 because 2 bytes
                            // per element
                            auto src_offs =
                                ctoffs + c16offs + TADDR(htWT + wt) + sub_src_offs + ((h32 & 15) << 5);  // bytes offset
                            auto bsrc_offs = (batch_addr + src_offs) - src0_addr;
                            uint32_t batch_itile = (bsrc_offs >> 11);
                            uint32_t rem = (bsrc_offs & 2047);

                            // if (h == 0 && ct == 0 && wt == 0) {
                            //     DPRINT << "  Sub=" << sub << " c16=" << c16 << ENDL();
                            //     DPRINT << "    Reading from src_offs=" << src_offs << ENDL();
                            //     DPRINT << "    Writing to   dst_offs=" << dest_tr0_l1-save_dest << ENDL();
                            // }

                            uint64_t banked_addr = get_noc_addr(batch_itile, s0);
                            banked_addr += rem;

                            // this starts async NOC dma from DRAM to TR0_L1 buffer
                            noc_async_read(banked_addr, dest_tr0_l1, SUBTILE_LINE_BYTES);

                            // if (h == 0 && ct == 0 && wt == 0)
                            //     DPRINT << uint32_t( reinterpret_cast<uint16_t*>( dest_tr0_l1 )[0] ) << ENDL();

                            // the output address is just linearly incremented
                            dest_tr0_l1 += SUBTILE_LINE_BYTES;
                            c16offs += HW2;
                        }
                        // subtiles are ordered like this:
                        // 0 1
                        // 2 3
                        // Here we offset C by 16 starting with subtile=2
                        if (sub == 1) {                  // after we are done with subtile 1, increment for sub=2
                            cSubtileOffs += (HW2 << 4);  // 16*HWbytes, which is subtile vertical size
                        }
                    }  // sub<4

                    // block on all outstanding noc DMA requests to complete
                    noc_async_read_barrier();

                    // notifies the unpacker that the buffer is populated
                    cb_push_back(operand0, onetile);
                }
                ctoffs += (HW2 << 5);  // since we increment ct, we need to mlutiply by 32
            }  // ct loop
            // multiplication-free computation of ht*WT, since ht = h/32
            if ((h & 31) == 31) {
                htWT += WT;
            }
        }  // h < H loop
        batch_addr += CHW2;
    }  // n<N loop
}
