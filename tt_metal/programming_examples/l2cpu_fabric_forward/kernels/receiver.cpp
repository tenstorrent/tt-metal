// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// Chip B, Tensix {0,0}: the far end of the fabric-forward path. By the time
// this kernel runs, the x280 firmware on the L2CPU tile has already pushed
// the payload across the fabric EDM connection and it has landed in this
// core's L1 at dst_l1_addr. This first cut does not itself wait for or
// verify delivery — sequencing producer -> fabric send -> this kernel is the
// host's job (poll the x280 status mailbox before launching this program).
// All this kernel does is invalidate the local L1 cache and copy the
// delivered bytes out to DRAM for host verification.
void kernel_main() {
    uint32_t dst_l1_addr = get_arg_val<uint32_t>(0);
    uint32_t dram_dst = get_arg_val<uint32_t>(1);
    uint32_t size = get_arg_val<uint32_t>(2);

    constexpr auto out0_args = TensorAccessorArgs<0>();
    const auto out0 = TensorAccessor(out0_args, dram_dst, size);

    // BH RISC L1 reads are cached; the fabric EDM's write landed via NOC, so
    // invalidate before reading it back out.
    invalidate_l1_cache();

    noc_async_write_page(0, out0, dst_l1_addr);
    noc_async_write_barrier();
}
