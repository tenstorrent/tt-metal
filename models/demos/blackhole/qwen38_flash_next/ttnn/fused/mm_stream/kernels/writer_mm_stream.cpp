// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// x52 streamed decode matmul, writer (NOC 1, one per compute core): releases the x CB's K blocks to compute as the
// hub's multicast lands them (semaphore >= block + 1), then writes the core's output tiles to page (bank * bank_tiles +
// col0 + j) of the DRAM tile tensor.  Runtime args: out address, first global output tile, ncols.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "../../kernels/zones.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#ifdef X52_DPRINT
#include "api/debug/dprint.h"
#endif
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t kt = get_named_compile_time_arg_val("kt");
    constexpr uint32_t out_l1 = get_named_compile_time_arg_val("out_l1");  // 1: the output is an L1 width shard
    constexpr auto out_args = TensorAccessorArgs<0, 0>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_tile = get_arg_val<uint32_t>(1);
    const uint32_t ncols = get_arg_val<uint32_t>(2);
    const auto out_tensor = TensorAccessor(out_args, out_addr);
    // L1 width shard: tile n is tile (n % shard_tiles) of shard core n / shard_tiles (the stock op's layout: a shard is
    // one row tile of shard_tiles columns, row-major); the cores' NoC coordinates follow the per-core runtime args 3..
    const uint32_t shard_tiles = out_l1 ? get_named_compile_time_arg_val("shard_out_tiles") : 1;

    Noc noc;
    CircularBuffer x(cb_x);
    CircularBuffer out(cb_out);
#ifdef X52_DPRINT
    DPRINT << "w start sem_x=" << *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0))
           << " first=" << first_tile << ENDL();
#endif
    // tell the hub this core's kernel is running (its semaphore is initialised): the hub multicasts only after every
    // core of the rectangle has said so, so the data and the semaphore never reach a core still being launched
    noc_semaphore_inc(
        get_noc_addr(
            get_arg_val<uint32_t>(3),
            get_arg_val<uint32_t>(4),
            get_semaphore(get_named_compile_time_arg_val("sem_hub"))),
        1);
    x.reserve_back(kt);
    {
        FUSED_ZONE("fz_mm_w_x");
        Semaphore<> sem(0);
        sem.wait_min(1);  // the hub's one multicast of the compact activation (kt tiles of 1 x 32) has landed
        sem.set(0);
        x.push_back(kt);
    }
    FUSED_ZONE("fz_mm_w_out");
    out.wait_front(ncols);
    // the output tiles are 1 x 32 (64 B: row 0 of face 0, row 0 of face 1): row 0 of page first_tile + j at bytes 0 and
    // 512
    for (uint32_t j = 0; j < ncols; ++j) {
        const uint32_t tile = first_tile + j;
#ifdef X52_SKIP_WRITES
        continue;
#endif
        if constexpr (out_l1) {
            const uint32_t shard = tile / shard_tiles;
            const uint32_t dst = out_addr + (tile % shard_tiles) * 2048;
            const uint32_t cx = get_arg_val<uint32_t>(5 + 2 * shard);
            const uint32_t cy = get_arg_val<uint32_t>(6 + 2 * shard);
            UnicastEndpoint remote;
            noc.async_write(
                CoreLocalMem<uint32_t>(out.get_read_ptr() + j * 64),
                remote,
                32,
                {},
                {.noc_x = cx, .noc_y = cy, .addr = dst});
            noc.async_write(
                CoreLocalMem<uint32_t>(out.get_read_ptr() + j * 64 + 32),
                remote,
                32,
                {},
                {.noc_x = cx, .noc_y = cy, .addr = dst + 512});
        } else {
            noc.async_write(
                CoreLocalMem<uint32_t>(out.get_read_ptr() + j * 64),
                out_tensor,
                32,
                {},
                {.page_id = tile, .offset_bytes = 0});
            noc.async_write(
                CoreLocalMem<uint32_t>(out.get_read_ptr() + j * 64 + 32),
                out_tensor,
                32,
                {},
                {.page_id = tile, .offset_bytes = 512});
        }
    }
    noc.async_write_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();  // the L1 form issues legacy writes
    out.pop_front(ncols);
}
