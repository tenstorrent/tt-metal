// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// Chip A, Tensix {0,0}: stages a payload from DRAM into the L2CPU's LIM/GDDR
// scratchpad, then posts a fabric-forward request into the x280 firmware's
// mailbox so it forwards the payload to a Tensix core on chip B over the
// fabric. See ../x280/fabric_mbox.h for the mailbox layout this kernel writes
// into (not #included here: Tensix kernels are JIT-compiled and the relative
// path does not resolve cleanly from this build).
void kernel_main() {
    uint32_t l1_src = get_arg_val<uint32_t>(0);
    uint32_t dram_src = get_arg_val<uint32_t>(1);
    uint32_t size = get_arg_val<uint32_t>(2);
    uint32_t lim_addr = get_arg_val<uint32_t>(3);
    uint32_t l2cpu_x = get_arg_val<uint32_t>(4);
    uint32_t l2cpu_y = get_arg_val<uint32_t>(5);
    uint32_t mbox_base = get_arg_val<uint32_t>(6);
    uint32_t seq = get_arg_val<uint32_t>(7);
    uint32_t dest_noc_x = get_arg_val<uint32_t>(8);
    uint32_t dest_noc_y = get_arg_val<uint32_t>(9);
    uint32_t dest_l1_addr = get_arg_val<uint32_t>(10);

    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, dram_src, size);

    // 1. Stage the payload: DRAM -> local L1.
    noc_async_read_page(0, in0, l1_src);
    noc_async_read_barrier();

    // 2. Bulk NOC write: L1 -> L2CPU LIM. The L2CPU tile keeps its NOC0
    //    coordinates under Blackhole's translation tables, so (l2cpu_x,
    //    l2cpu_y) is used as-is.
    noc_async_write(l1_src, get_noc_addr(l2cpu_x, l2cpu_y, lim_addr), size);
    noc_async_write_barrier();

    // 3. Build the fabric-forward request block in L1 scratch just past the
    //    payload staging area, then post it in one contiguous NOC write.
    //    Offsets below mirror the FF_REQ_* fields in ../x280/fabric_mbox.h:
    //      +0x00 seq            (FF_REQ_SEQ)
    //      +0x04 payload_lim    (FF_REQ_PAYLOAD_LIM)
    //      +0x08 size           (FF_REQ_SIZE)
    //      +0x0c dest_noc_x     (FF_REQ_DEST_NOC_X)
    //      +0x10 dest_noc_y     (FF_REQ_DEST_NOC_Y)
    //      +0x14 dest_l1_addr   (FF_REQ_DEST_L1)
    uint32_t l1_req = l1_src + size;
    volatile tt_l1_ptr uint32_t* req = (volatile tt_l1_ptr uint32_t*)l1_req;
    req[0] = seq;
    req[1] = lim_addr;
    req[2] = size;
    req[3] = dest_noc_x;
    req[4] = dest_noc_y;
    req[5] = dest_l1_addr;

    noc_async_write(l1_req, get_noc_addr(l2cpu_x, l2cpu_y, mbox_base + 0x80), 24);
    noc_async_write_barrier();
}
