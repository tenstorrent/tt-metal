// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Bisect step 3: does anything my kernel writes reach the host at all?
//
// The previous copy test produced *identical* wrong values before and after a
// source change, which smells like the output is stale garbage rather than my
// kernel's result. So drop all data movement: the compute side fills Dst with a
// known constant and packs it, and BRISC writes that out. If the host does not
// read back 1.0, the output path is broken and no amount of matmul debugging
// will help; if it does, the output path is fine and the fault is upstream.

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/fill.h"
#endif

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    // nothing
#elif defined(COMPILE_FOR_BRISC)
    constexpr uint32_t cb_o = 16;
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t o_bytes = 2048;

    const InterleavedAddrGenFast<true> o_gen{out_addr, o_bytes};
    cb_wait_front(cb_o, 1);
    noc_async_write_page(0, o_gen, get_read_ptr(cb_o));
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_TRISC)
    constexpr uint32_t cb_o = 16;

    // Every compute kernel must call this once, before any other compute API.
    // Without it the packer/unpacker/math threads are unconfigured and a packed
    // tile comes out as zeros -- which is exactly what the first attempts did.
    compute_kernel_hw_startup(cb_o, cb_o, cb_o);

    cb_reserve_back(cb_o, 1);
    tile_regs_acquire();
    // SFPU fill of the whole tile; needs no input at all.
    fill_tile(0, 1.0f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_o);
    cb_push_back(cb_o, 1);
    tile_regs_release();
#endif
}
