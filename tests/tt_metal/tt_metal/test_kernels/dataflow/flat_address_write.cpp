// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

// LOCAL horizon_x2 spike: writes `size` bytes of this core's L1 at `src` to the 64-bit NoC operand
// dst_hi:dst_lo, used as is. Under the ATT that operand is a flat address, e.g. another chip's tile.
// A posted write only waits until it has left; a non-posted one waits for the target's ack.
void kernel_main() {
    const uint32_t src = get_arg(args::src);
    const uint64_t dst = (uint64_t{get_arg(args::dst_hi)} << 32) | get_arg(args::dst_lo);
    const uint32_t size = get_arg(args::size);
    if (get_arg(args::posted)) {
        noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(src, dst, size);
        noc_async_posted_writes_flushed();
    } else {
        noc_async_write(src, dst, size);
        noc_async_write_barrier();
    }
}
