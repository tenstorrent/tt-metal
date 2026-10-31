// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Bisect step 2: copy a tile through, writing the output with an explicit BRISC
// NOC write to the output tensor's address.
//
// Step 1 packed into a CB backed by a sharded output tensor and the host read
// zeros. Rather than debug CB-backed output buffers, this uses the arrangement the
// working DeepSeek compressed matmul uses: the compute side packs into an ordinary
// CB, and BRISC copies that CB out to the output tensor's address with
// noc_async_write. That is a pattern known to work in this tree.

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#endif

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t cb_a = 0;
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t a_bytes = 2048;

    const InterleavedAddrGenFast<true> a_gen{a_addr, a_bytes};
    cb_reserve_back(cb_a, 1);
    noc_async_read_page(0, a_gen, get_write_ptr(cb_a));
    noc_async_read_barrier();
    cb_push_back(cb_a, 1);
#elif defined(COMPILE_FOR_BRISC)
    // Write the packed tile out to wherever the output tensor lives.
    constexpr uint32_t cb_o = 16;
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t o_bytes = 2048;

    const InterleavedAddrGenFast<true> o_gen{out_addr, o_bytes};
    cb_wait_front(cb_o, 1);
    noc_async_write_page(0, o_gen, get_read_ptr(cb_o));
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_TRISC)
    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_o = 16;

    compute_kernel_hw_startup(cb_a, cb_a, cb_o);
    cb_wait_front(cb_a, 1);
    // Without this the packer is left configured by whatever ran before, so the
    // tile comes out in the wrong format or face order. pack_tile does not
    // initialize the packer; an op-specific init normally does it for you.
    pack_reconfig_data_format(cb_o);
    tile_regs_acquire();
    copy_tile(cb_a, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_o, 1);
    pack_tile(0, cb_o);
    cb_push_back(cb_o, 1);
    tile_regs_release();
    cb_pop_front(cb_a, 1);
#endif
}
