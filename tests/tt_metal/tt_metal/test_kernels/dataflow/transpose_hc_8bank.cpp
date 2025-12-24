// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

#include <cstdint>
#include <cstring>

// tile index to address
inline uint32_t TADDR(uint32_t ti) { return ti << 11; }

void kernel_main() {
    const uint32_t src0_addr = get_arg_val<uint32_t>(0U);
    const uint32_t src_bank_id = get_arg_val<uint32_t>(1U);
    const uint32_t W = get_arg_val<uint32_t>(2U);
    const uint32_t H = get_arg_val<uint32_t>(3U);
    const uint32_t C = get_arg_val<uint32_t>(4U);
    const uint32_t HW = get_arg_val<uint32_t>(5U);
    const uint32_t N = get_arg_val<uint32_t>(6U);
    const uint32_t CHW = get_arg_val<uint32_t>(7U);

    constexpr uint32_t tile_elements = 32U;
    const uint32_t WT = W / tile_elements;                       // number of tiles in W
    const uint32_t HT = H / tile_elements;                       // number of tiles in H
    const uint32_t CT = C / tile_elements;                       // number of tiles in C
    const uint32_t HTWT = HW / (tile_elements * tile_elements);  // product of HT*WT
    const uint32_t HW2 = HW * 2U;                                // HW stride in bytes, * 2U since FP16
    const uint32_t CHW2 = CHW * 2U;                              // batch stride in bytes, *2U since FP16

    constexpr uint32_t subtile_elements = 16U;
    constexpr uint32_t SUBTILE_LINE_BYTES = subtile_elements * 2U;  // FP16 is 2 bytes
    constexpr uint32_t subtile_size_bytes = subtile_elements * SUBTILE_LINE_BYTES;
    constexpr uint32_t tile_size_bytes = tile_elements * tile_elements * 2U;  // * 2U since FP16 is 2 bytes

    constexpr uint32_t ALIGNMENT = get_compile_time_arg_val(0U);
    constexpr uint32_t ALIGNMENT_MASK = ALIGNMENT - 1U;
    constexpr bool MISALIGNED = ALIGNMENT > SUBTILE_LINE_BYTES;

    const uint32_t intermed_l1_scratch = MISALIGNED ? get_write_ptr(1U) : 0U;
    uint8_t* intermed_l1_scratch_ptr = reinterpret_cast<uint8_t*>(intermed_l1_scratch);

    constexpr uint32_t onetile = 1U;
    constexpr uint32_t operand0 = 0U;
    constexpr auto src_args = TensorAccessorArgs<1U>();
    const auto s0 = TensorAccessor(src_args, src0_addr, tile_size_bytes);

    // The original tensor shape is [N, C, H, W]
    // It is laid out in memory as addr = W + ShapeW * H + ShapeW * ShapeH * C + ShapeW * ShapeH * ShapeC * N

    // This will be converted to a tensor shape of [N, H, C, W]
    // With memory laid out as addr = W + ShapeW * C + ShapeW * ShapeC * H + ShapeW * ShapeC * ShapeH * N
    // Thus we loop over N, H, C, W to produce a linear incremented output address
    uint64_t batch_addr = src0_addr;
    for (uint32_t n = 0U; n < N; ++n) {
        uint32_t htWT = 0U;
        for (uint32_t h = 0U; h < H; ++h) {
            uint32_t ctoffs = 0U;
            for (uint32_t ct = 0U; ct < CT; ++ct) {
                for (uint32_t wt = 0U; wt < WT; ++wt) {
                    // what is the source address for the current tile?
                    // c32 = intra-C-tile loop
                    // every 32 C's acquire a new output tile address
                    //    DPRINT << "8B h=" << h << " ct=" << ct << " wt=" << wt << " W=" << W << " HW2=" << HW2 <<
                    //    ENDL();

                    cb_reserve_back(operand0, onetile);

                    uint32_t dest_tr0_l1 = get_write_ptr(operand0);
                    uint32_t cSubtileOffs = 0U;

                    // loop over the 4 sub tiles of each tile
                    for (uint32_t sub = 0U; sub < 4U; ++sub) {
                        uint32_t c16offs = cSubtileOffs;
                        for (uint32_t c16 = 0U; c16 < 16U; ++c16) {
                            // In this loop sub, c16 are source subtile, c16
                            // dest in this loop is varying h implicitly via dest address increment

                            // Dest is HCW
                            // We are iterating over it as H Ct Wt-tiles
                            // intra-tile FC16 for F going over 4-subtiles
                            // the source address is (bytes):
                            // src_addr = c*HW2 + (ht*Wt + wt)*tile_size_bytes + sub*subtile_size_bytes + (h16*16 +
                            // w16)*2 we have 512 bytes per subtile and 32 bytes per subtile row of 16 elems note that
                            // dest h is decomposed as h = ht+h32 and htWT is incremented by WT in the outer H loop
                            const uint32_t h32 = static_cast<uint32_t>(h & (tile_elements - 1U));

                            // subtiles are ordered like this:
                            // 0 1
                            // 2 3
                            // So this will offset the address by a sub tile size for tiles 1 and 3
                            // if intra-tile source h is >= 16, add 2*512 to subtile offset
                            const uint32_t sub_src_offs =
                                (sub & 0x1U) * subtile_size_bytes + (h32 / subtile_elements) * 2U * subtile_size_bytes;

                            // below we only use the lower 4 bits out of 5-bit range for h, shift by 5 because 2 bytes
                            // per element
                            const uint32_t src_offs = ctoffs + c16offs + TADDR(htWT + wt) + sub_src_offs +
                                                      ((h32 & (subtile_elements - 1U)) * tile_elements);
                            const uint64_t bsrc_offs = (batch_addr + src_offs) - src0_addr;
                            const uint32_t batch_itile = static_cast<uint32_t>(bsrc_offs / tile_size_bytes);
                            const uint32_t rem = static_cast<uint32_t>(bsrc_offs & (tile_size_bytes - 1U));

                            const uint64_t banked_addr = get_noc_addr(batch_itile, s0) + rem;

                            if (MISALIGNED) {
                                // if banked addr and dest addr don't share alignment then we need to read to the
                                // intermediate buffer and then copy it to the correct location
                                const uint32_t banked_alignment =
                                    static_cast<uint32_t>(banked_addr & static_cast<uint64_t>(ALIGNMENT_MASK));
                                if ((dest_tr0_l1 & ALIGNMENT_MASK) != banked_alignment) {
                                    // we write to the top of the intermediate buffer as that's aligned, and we write
                                    // from the closest align source address if source is not aligned to ALIGNMENT then
                                    // we go to the nearest address that is aligned and copy from there
                                    noc_async_read(banked_addr - banked_alignment, intermed_l1_scratch, ALIGNMENT);
                                    uint8_t* dest_tr0_l1_ptr = reinterpret_cast<uint8_t*>(dest_tr0_l1);
                                    // need the barrier to ensure that we can copy from the intermediate buffer
                                    noc_async_read_barrier();
                                    // if source is not aligned to ALIGNMENT then we need to skip forward by the amount
                                    // needed to align to get to the correct data
                                    ::memcpy(
                                        dest_tr0_l1_ptr,
                                        intermed_l1_scratch_ptr + banked_alignment,
                                        SUBTILE_LINE_BYTES);
                                } else {
                                    // this starts async NOC dma from DRAM to TR0_L1 buffer
                                    noc_async_read(banked_addr, dest_tr0_l1, SUBTILE_LINE_BYTES);
                                }
                            } else {
                                noc_async_read(banked_addr, dest_tr0_l1, SUBTILE_LINE_BYTES);
                            }

                            // the output address is just linearly incremented
                            dest_tr0_l1 += SUBTILE_LINE_BYTES;
                            c16offs += HW2;
                        }
                        // subtiles are ordered like this:
                        // 0 1
                        // 2 3
                        // Here we offset C by 16 starting with subtile=2
                        if (sub == 1U) {  // after we are done with subtile 1, increment for sub=2
                            cSubtileOffs += (HW2 * subtile_elements);  // 16*HWbytes, which is subtile vertical size
                        }
                    }  // sub<4

                    // block on all outstanding noc DMA requests to complete
                    noc_async_read_barrier();

                    // notifies the unpacker that the buffer is populated
                    cb_push_back(operand0, onetile);
                }
                // since src_addr for c term is ShapeW * ShapeH * C,
                // We increment C Offset by HW2 * tile_elements
                ctoffs += HW2 * tile_elements;
            }  // ct loop
            // multiplication-free computation of ht*WT, since ht = h/32
            // since last h of tile increment source address W size
            if ((h & (tile_elements - 1U)) == tile_elements - 1U) {
                htWT += WT;
            }
        }  // h < H loop
        batch_addr += CHW2;  // increment for N which is the same for the source and destination address
    }  // n<N loop
}
