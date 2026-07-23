// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tensix_types.h"
#include "api/tensor/tensor_accessor.h"
#include "barrier_sync.hpp"

// L1 to DRAM write
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t l1_read_addr = get_arg_val<uint32_t>(1);
    uint32_t page_offset = get_arg_val<uint32_t>(2);
    // Barrier synchronization args
    uint32_t barrier_sem_id = get_arg_val<uint32_t>(3);
    uint32_t barrier_coord_x = get_arg_val<uint32_t>(4);
    uint32_t barrier_coord_y = get_arg_val<uint32_t>(5);
    uint32_t num_cores = get_arg_val<uint32_t>(6);
    uint32_t local_barrier_addr = get_arg_val<uint32_t>(7);  // Local scratch space for polling

    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(3);
    constexpr uint32_t test_id = get_compile_time_arg_val(4);
    constexpr bool sync = get_compile_time_arg_val(5) == 1;
    // default_noc selects which RISC/NOC the writer runs on so the profiler zone name
    // matches the physical RISC: false -> RISCV1/NOC1 (default writer), true -> RISCV0/NOC0.
    constexpr bool default_noc = get_compile_time_arg_val(6) == 1;
    // enable_phase_counters: compile in the t0..t3 write-phase instrumentation (batched !sync path).
    // Compiled out when 0 so timing is clean by default (host sets it from TT_DM_PHASE_COUNTERS).
    constexpr bool enable_phase_counters = get_compile_time_arg_val(7) == 1;
    // enable_page_counters: emit one timestamp marker right after every noc_async_write (host sets it
    // from TT_DM_PAGE_COUNTERS). The gap between consecutive markers is the per-page issue cadence; the
    // payload is the cumulative write-progress count (see wr_progress_reg) at that point. Independent of
    // enable_phase_counters. WARNING: emits one marker per page, so a large Q overflows the per-RISC
    // profiler L1 buffer and the tail is dropped.
    constexpr bool enable_page_counters = get_compile_time_arg_val(8) == 1;
    // posted: issue posted writes (host sets it from TT_DM_POSTED_WRITES).
    //   false (default) -> non-posted: each write requests an ack; noc_async_write_barrier() waits on
    //                      the ack flush and NIU_MST_WR_ACK_RECEIVED counts completed writes.
    //   true            -> posted: no ack is requested, so the non-posted ack barrier would hang. We
    //                      wait on noc_async_posted_writes_flushed() (requests departed, not landed) and
    //                      track NIU_MST_POSTED_WR_REQ_SENT instead. NOTE: posted completion only
    //                      guarantees departure from the NIU, not arrival at the destination.
    constexpr bool posted = get_compile_time_arg_val(9) == 1;

    // HW cumulative counter used by the phase/page instrumentation for the selected write mode:
    //   non-posted -> NIU_MST_WR_ACK_RECEIVED    (destination ack returned; also gates the write barrier)
    //   posted     -> NIU_MST_POSTED_WR_REQ_SENT (posted writes never ack, so track requests departed)
    constexpr uint32_t wr_progress_reg = posted ? NIU_MST_POSTED_WR_REQ_SENT : NIU_MST_WR_ACK_RECEIVED;

    // Tensor accessor compile time args appended to kernel's compile time args
    // so the index is offset to start at 10
    auto args = TensorAccessorArgs<10>();
    auto s = TensorAccessor(args, dst_addr);

    constexpr uint32_t transaction_size_bytes = page_size_bytes;
    // These user timestamped-data markers carry bandwidth metadata for the DM profiler CSV, but they
    // are emitted as TS_DATA markers. When NoC-event tracing is on the profiler treats every TS_DATA
    // marker as serialized NoC-event metadata, so these arbitrary payloads trip the "Invalid NoC
    // transfer type" TT_FATAL. Compile them out while tracing (PROFILE_NOC_EVENTS is auto-defined
    // whenever TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1); the bandwidth CSV is not needed for tt-npe runs.
#if !defined(PROFILE_NOC_EVENTS)
    DeviceTimestampedData("Number of transactions", num_of_transactions * num_pages);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Posted writes", (uint64_t)posted);
#endif

    // Wait for all cores to reach this point before starting data movement
    barrier_sync(barrier_sem_id, barrier_coord_x, barrier_coord_y, num_cores, local_barrier_addr);

    // Completion wait: posted writes never ack, so the non-posted ack barrier would hang. Wait for the
    // posted requests to depart instead. Non-posted uses the standard ack barrier.
    auto write_barrier = [&]() {
        if constexpr (posted) {
            noc_async_posted_writes_flushed();
        } else {
            noc_async_write_barrier();
        }
    };

    auto do_writes = [&]() {
        // Phase instrumentation only applies to the write-only (!sync) batched path, where a single
        // barrier drains the whole batch. In sync mode each page barriers individually, so the t0..t3
        // split is not meaningful. wr_progress_reg is the HW cumulative write counter; each marker
        // carries "writes so far" (relative to the pre-issue baseline) for host phase splitting.
        [[maybe_unused]] uint32_t prog_baseline = 0;
        if constexpr (!sync && (enable_phase_counters || enable_page_counters)) {
            prog_baseline = NOC_STATUS_READ_REG(noc_index, wr_progress_reg);
        }
        if constexpr (!sync && enable_phase_counters) {
            DeviceTimestampedData("dm_t0_issue_start", (uint64_t)noc_index);
        }

        for (uint32_t i = 0; i < num_of_transactions; i++) {
            for (uint32_t p = 0; p < num_pages; p++) {
                if constexpr (sync) {
                    cb_wait_front(cb_id_out0, 1);
                }
                uint64_t noc_addr = s.get_noc_addr(page_offset + p);
                noc_async_write<NOC_MAX_BURST_SIZE + 1, true, posted>(
                    l1_read_addr + p * page_size_bytes, noc_addr, page_size_bytes);
                if constexpr (!sync && enable_page_counters) {
                    // NOTE: keep on a single source line (GCC __LINE__ marker-name hash quirk).
                    DeviceTimestampedData(
                        "dm_page_issued", (uint64_t)(NOC_STATUS_READ_REG(noc_index, wr_progress_reg) - prog_baseline));
                }
                if constexpr (sync) {
                    write_barrier();
                    cb_pop_front(cb_id_out0, 1);
                }
            }
        }
        if constexpr (!sync) {
            if constexpr (enable_phase_counters) {
                // NOTE: each DeviceTimestampedData(...) call MUST stay on a single source line
                // (GCC __LINE__ expansion quirk in the marker-name hash).
                DeviceTimestampedData(
                    "dm_t1_issue_end", (uint64_t)(NOC_STATUS_READ_REG(noc_index, wr_progress_reg) - prog_baseline));
            }
            if constexpr (enable_phase_counters) {
                while (NOC_STATUS_READ_REG(noc_index, wr_progress_reg) == prog_baseline) {
                }
                DeviceTimestampedData(
                    "dm_t2_first_return", (uint64_t)(NOC_STATUS_READ_REG(noc_index, wr_progress_reg) - prog_baseline));
            }
            write_barrier();
            if constexpr (enable_phase_counters) {
                DeviceTimestampedData(
                    "dm_t3_barrier_clear", (uint64_t)(NOC_STATUS_READ_REG(noc_index, wr_progress_reg) - prog_baseline));
            }
        }
    };

    if constexpr (default_noc) {
        DeviceZoneScopedN("RISCV0");
        do_writes();
    } else {
        DeviceZoneScopedN("RISCV1");
        do_writes();
    }
}
