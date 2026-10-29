// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// x52 streamed decode matmul, x hub (NOC 0, one core outside the compute rectangle): the shard cores write their
// compacted activation (row 0 of every K tile as a 1 x 32 tile, 64 B) into this core's compact buffer; once all
// `shards` chunks are in (semaphore), ONE multicast of the whole buffer (kt x 64 B) into the x CB of every core of the
// compute rectangle and the receivers' semaphore.  Runtime args: rect NoC corners x0 y0 x1 y1 (NOC 0 order), number of
// dests.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t cb_stage = get_named_compile_time_arg_val("cb_stage");
    constexpr uint32_t sem_in = get_named_compile_time_arg_val("sem_hub");
    constexpr uint32_t sem_out = get_named_compile_time_arg_val("sem_x");
    constexpr uint32_t shards = get_named_compile_time_arg_val("shards");
    constexpr uint32_t kt = get_named_compile_time_arg_val("kt");
    const uint32_t x0 = get_arg_val<uint32_t>(0);
    const uint32_t y0 = get_arg_val<uint32_t>(1);
    const uint32_t x1 = get_arg_val<uint32_t>(2);
    const uint32_t y1 = get_arg_val<uint32_t>(3);
    const uint32_t dests = get_arg_val<uint32_t>(4);
    CircularBuffer x(cb_x);
    CircularBuffer stage(cb_stage);
    const uint32_t x_base = x.get_write_ptr();
    const uint32_t out_sem_addr = get_semaphore(sem_out);
    volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_in));
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_sem_addr);
    FUSED_ZONE("fz_mm_hub");
    noc_semaphore_wait(in, shards + dests);  // the shard chunks and a ready from every core of the rectangle
    noc_semaphore_set(in, 0);
    noc_async_write_multicast(
        stage.get_write_ptr(), get_noc_multicast_addr(x0, y0, x1, y1, x_base), kt * 64, dests, true);
    *out = 1;
    noc_semaphore_set_multicast(out_sem_addr, get_noc_multicast_addr(x0, y0, x1, y1, out_sem_addr), dests);
    noc_async_write_barrier();
    *out = 0;
}
