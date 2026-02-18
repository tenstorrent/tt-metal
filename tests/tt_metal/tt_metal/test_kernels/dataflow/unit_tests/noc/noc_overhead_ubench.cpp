// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Microbenchmark to measure cycle cost of individual NOC operations:
//   - fence (invalidate_l1_cache)
//   - NOC status register read (NOC_STATUS_READ_REG)
//   - L1 software counter read
//   - noc_semaphore_inc (forced non-posted atomic on BH)
//   - noc_semaphore_wait (single iteration, sem already set)
//   - full barrier (noc_async_full_barrier) with nothing outstanding
//   - ncrisc_noc_nonposted_atomics_flushed check
//   - Each individual barrier sub-check
//
// Results written to L1 timestamp buffer as pairs of [op_id, cycles].

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "risc_common.h"
#include "noc_nonblocking_api.h"

#define TS() reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L)

void kernel_main() {
    set_l1_data_cache<false>();

    uint32_t arg_idx = 0;
    const uint32_t timestamp_buf_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_iterations = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t target_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t target_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t target_sem_addr = get_arg_val<uint32_t>(arg_idx++);

    volatile tt_l1_ptr uint32_t* ts_buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(timestamp_buf_addr);

    // Number of tests
    constexpr uint32_t NUM_TESTS = 14;
    // Layout: [test_id(0), min(1), max(2), sum_lo(3), sum_hi(4)] x NUM_TESTS = 70 words header
    // Then raw samples follow if needed
    constexpr uint32_t HEADER_WORDS = NUM_TESTS * 5;

    // Initialize header
    for (uint32_t t = 0; t < NUM_TESTS; t++) {
        ts_buf[t * 5 + 0] = t;           // test_id
        ts_buf[t * 5 + 1] = 0xFFFFFFFF;  // min
        ts_buf[t * 5 + 2] = 0;           // max
        ts_buf[t * 5 + 3] = 0;           // sum_lo
        ts_buf[t * 5 + 4] = 0;           // sum_hi
    }

    auto record = [&](uint32_t test_id, uint32_t cycles) {
        uint32_t base = test_id * 5;
        if (cycles < ts_buf[base + 1]) {
            ts_buf[base + 1] = cycles;
        }
        if (cycles > ts_buf[base + 2]) {
            ts_buf[base + 2] = cycles;
        }
        uint32_t old_lo = ts_buf[base + 3];
        ts_buf[base + 3] = old_lo + cycles;
        if (ts_buf[base + 3] < old_lo) {
            ts_buf[base + 4]++;  // carry
        }
    };

    uint64_t target_noc_addr = get_noc_addr(target_noc_x, target_noc_y, target_sem_addr);

    // ---- Test 0: fence (invalidate_l1_cache) ----
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t t0 = TS();
        invalidate_l1_cache();
        uint32_t t1 = TS();
        record(0, t1 - t0);
    }

    // ---- Test 1: wall clock read (baseline for timestamp overhead) ----
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t t0 = TS();
        uint32_t t1 = TS();
        record(1, t1 - t0);
    }

    // ---- Test 2: NOC_STATUS_READ_REG - NIU_MST_RD_RESP_RECEIVED ----
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t t0 = TS();
        volatile uint32_t x = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED);
        uint32_t t1 = TS();
        record(2, t1 - t0);
        (void)x;
    }

    // ---- Test 3: NOC_STATUS_READ_REG - NIU_MST_NONPOSTED_WR_REQ_SENT ----
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t t0 = TS();
        volatile uint32_t x = NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT);
        uint32_t t1 = TS();
        record(3, t1 - t0);
        (void)x;
    }

    // ---- Test 4: NOC_STATUS_READ_REG - NIU_MST_WR_ACK_RECEIVED ----
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t t0 = TS();
        volatile uint32_t x = NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED);
        uint32_t t1 = TS();
        record(4, t1 - t0);
        (void)x;
    }

    // ---- Test 5: NOC_STATUS_READ_REG - NIU_MST_ATOMIC_RESP_RECEIVED ----
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t t0 = TS();
        volatile uint32_t x = NOC_STATUS_READ_REG(noc_index, NIU_MST_ATOMIC_RESP_RECEIVED);
        uint32_t t1 = TS();
        record(5, t1 - t0);
        (void)x;
    }

    // ---- Test 6: NOC_STATUS_READ_REG - NIU_MST_POSTED_WR_REQ_SENT ----
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t t0 = TS();
        volatile uint32_t x = NOC_STATUS_READ_REG(noc_index, NIU_MST_POSTED_WR_REQ_SENT);
        uint32_t t1 = TS();
        record(6, t1 - t0);
        (void)x;
    }

    // ---- Test 7: L1 read of software counter (noc_reads_num_issued) ----
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t t0 = TS();
        volatile uint32_t x = noc_reads_num_issued[noc_index];
        uint32_t t1 = TS();
        record(7, t1 - t0);
        (void)x;
    }

    // ---- Test 8: ncrisc_noc_reads_flushed (MMIO read + L1 read + compare) ----
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t t0 = TS();
        volatile bool x = ncrisc_noc_reads_flushed(noc_index);
        uint32_t t1 = TS();
        record(8, t1 - t0);
        (void)x;
    }

    // ---- Test 9: ncrisc_noc_nonposted_atomics_flushed ----
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t t0 = TS();
        volatile bool x = ncrisc_noc_nonposted_atomics_flushed(noc_index);
        uint32_t t1 = TS();
        record(9, t1 - t0);
        (void)x;
    }

    // ---- Test 10: noc_async_full_barrier (nothing outstanding) ----
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t t0 = TS();
        noc_async_full_barrier();
        uint32_t t1 = TS();
        record(10, t1 - t0);
    }

    // ---- Test 11: noc_semaphore_inc (to self, forced non-posted on BH) ----
    // Target a local sem so the atomic completes quickly
    volatile tt_l1_ptr uint32_t* local_dummy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(target_sem_addr);
    *local_dummy = 0;
    uint64_t self_noc_addr = get_noc_addr(
        NOC_XY_ENCODING(my_x[noc_index], my_y[noc_index]) >> NOC_ADDR_NODE_ID_BITS,
        NOC_XY_ENCODING(my_x[noc_index], my_y[noc_index]) & ((1 << NOC_ADDR_NODE_ID_BITS) - 1),
        target_sem_addr);
    // Actually, just use get_noc_addr with my_x/my_y directly
    // Warmup to get counters in steady state
    noc_semaphore_inc(target_noc_addr, 1);
    while (!ncrisc_noc_nonposted_atomics_flushed(noc_index));

    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t t0 = TS();
        noc_semaphore_inc(target_noc_addr, 1);
        uint32_t t1 = TS();
        record(11, t1 - t0);
        // Wait for atomic to complete before next iteration
        while (!ncrisc_noc_nonposted_atomics_flushed(noc_index));
    }

    // ---- Test 12: noc_semaphore_inc + wait for atomic flush (full round-trip to target) ----
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t t0 = TS();
        noc_semaphore_inc(target_noc_addr, 1);
        while (!ncrisc_noc_nonposted_atomics_flushed(noc_index));
        uint32_t t1 = TS();
        record(12, t1 - t0);
    }

    // ---- Test 13: volatile L1 read (semaphore poll, single read, no fence) ----
    *local_dummy = 1;  // pre-set so we don't block
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t t0 = TS();
        volatile uint32_t x = *local_dummy;
        uint32_t t1 = TS();
        record(13, t1 - t0);
        (void)x;
    }
}
