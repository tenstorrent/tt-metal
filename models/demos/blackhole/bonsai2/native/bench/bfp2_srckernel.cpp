// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Bisect step 6: compute reads a tile from a sharded-CB and packs it back out.
//
// Established so far:
//   * constant write: compute fills Dst with a known value, packs, BRISC writes
//     -> 1024/1024 correct. So compute, pack, BRISC write and the output path work.
//   * copy via NCRISC read: the output comes back as zeros, i.e. written but empty.
//     So the DRAM read into a plain CB is what fails.
//
// This removes NCRISC from the picture entirely: the input CB *is* a sharded input
// tensor's own L1 buffer (the arrangement the paused ternary_matmul op used for its
// activation), so the tile is already where compute expects it.
//
//   works  -> the tile pipeline is fine and only the NCRISC read needs fixing,
//             which is a local problem.
//   zeros  -> the compute-side tile path itself is wrong, and the matmul should not
//             be trusted until that is understood.

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
    // nothing: the input arrives via the sharded CB
#elif defined(COMPILE_FOR_BRISC)
    constexpr uint32_t cb_o = 16;
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t o_bytes = 2048;

    const InterleavedAddrGenFast<true> o_gen{out_addr, o_bytes};
    cb_wait_front(cb_o, 1);
    noc_async_write_page(0, o_gen, get_read_ptr(cb_o));
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_TRISC)
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_o = 16;

    compute_kernel_hw_startup(cb_in, cb_o);
    cb_wait_front(cb_in, 1);
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_o);
    cb_reserve_back(cb_o, 1);
    pack_tile(0, cb_o);
    cb_push_back(cb_o, 1);
    tile_regs_release();
#endif
}
