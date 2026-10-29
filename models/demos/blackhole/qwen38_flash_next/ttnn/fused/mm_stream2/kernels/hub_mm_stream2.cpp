// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// x60 streamed decode matmul, hub (one core): collects the compact activation (kt 1x32 tiles = 64 B each) the shard
// cores wrote into its staging CB, waits until every compute core reported ready, then multicasts the tiles + the "x
// ready" semaphore to each destination RANGE (the compute cores are placed by DRAM bank, not as one rectangle). Runtime
// args: number of ranges, ranges * (x0, y0, x1, y1, cores in the range), total compute cores.
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
    const uint32_t ranges = get_arg_val<uint32_t>(0);
    const uint32_t total = get_arg_val<uint32_t>(1);
    CircularBuffer x(cb_x);
    CircularBuffer stage(cb_stage);
    const uint32_t x_base = x.get_write_ptr();
    const uint32_t out_sem_addr = get_semaphore(sem_out);
    volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_in));
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_sem_addr);
    FUSED_ZONE("fz_mm_hub");
    noc_semaphore_wait(in, shards + total);  // the shard chunks and a ready from every compute core
    noc_semaphore_set(in, 0);
    *out = 1;
    for (uint32_t r = 0; r < ranges; ++r) {
        const uint32_t a = 2 + 5 * r;
        const uint32_t x0 = get_arg_val<uint32_t>(a), y0 = get_arg_val<uint32_t>(a + 1);
        const uint32_t x1 = get_arg_val<uint32_t>(a + 2), y1 = get_arg_val<uint32_t>(a + 3);
        const uint32_t n = get_arg_val<uint32_t>(a + 4);
        noc_async_write_multicast(
            stage.get_write_ptr(), get_noc_multicast_addr(x0, y0, x1, y1, x_base), kt * 64, n, true);
        noc_semaphore_set_multicast(out_sem_addr, get_noc_multicast_addr(x0, y0, x1, y1, out_sem_addr), n);
    }
    noc_async_write_barrier();
    *out = 0;
}
