// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream, lane NCRISC (tt/fabric_stream.py; ported into the DSA shard tree with FS_WAIT: the producers'
// semaphore gate before the first read): streams this lane's share of the source rows (DRAM tiles of
// x from chunk src_base on) into the staging CB in fixed NB-page batches (NB divides the pages per chunk and the CB
// holds a whole number of batches: every push lands inside the CB), for the lane's BRISC (ag_send.cpp) to write and
// send. FS_SCATTER (tt/fabric_stream.py FabricStreamScatter): after the wait count, n_dst then the n_dst source chunk
// bases (block q of x for destination q): the lane streams its share of each destination's block in turn, in the order
// fs_send.cpp sends them.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PPC = get_compile_time_arg_val(0);  // pages per chunk
constexpr uint32_t NB = get_compile_time_arg_val(1);   // pages per batch
constexpr uint32_t CB_STAGE = get_compile_time_arg_val(2);
constexpr uint32_t TB = 2048;
static_assert(PPC % NB == 0, "a batch never straddles a chunk");

void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<3>();
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t c_lo = get_arg_val<uint32_t>(1);  // this lane's first chunk (of the block)
    const uint32_t c_n = get_arg_val<uint32_t>(2);
    const uint32_t src_base = get_arg_val<uint32_t>(3);  // the block's first chunk (row tile) in x
    const auto x = TensorAccessor(x_args, x_addr);
#if defined(FS_WAIT)  // tt/fabric_stream.py wait=: the block is complete once FS_WAIT reached runtime arg 4
    const uint32_t wait_count = get_arg_val<uint32_t>(4);
    volatile tt_l1_ptr uint32_t* wsem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FS_WAIT));
    noc_semaphore_wait_min(wsem, wait_count);
#endif
    const uint32_t nbat = c_n * PPC / NB;
#if defined(FS_SCATTER)
#if defined(FS_WAIT)
    constexpr uint32_t SC0 = 5;
#else
    constexpr uint32_t SC0 = 4;
#endif
    const uint32_t n_dst = get_arg_val<uint32_t>(SC0);
    for (uint32_t k = 0; k < n_dst; k++) {
        const uint32_t base = get_arg_val<uint32_t>(SC0 + 1 + k);
#else
    {
        const uint32_t base = src_base;
#endif
        for (uint32_t b = 0; b < nbat; b++) {
            cb_reserve_back(CB_STAGE, NB);
            const uint32_t wp = get_write_ptr(CB_STAGE);
            for (uint32_t i = 0; i < NB; i++) {
                noc_async_read_page((base + c_lo) * PPC + b * NB + i, x, wp + i * TB);
            }
            noc_async_read_barrier();
            cb_push_back(CB_STAGE, NB);
        }
    }
}
