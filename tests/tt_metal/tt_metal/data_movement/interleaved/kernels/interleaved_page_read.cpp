// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tensix_types.h"

// #include "api/debug/dprint.h"
// #include "api/debug/dprint_pages.h"

template <
    uint32_t num_of_transactions,
    uint32_t num_pages,
    uint32_t page_size_bytes,
    bool enable_phase_counters,
    typename AddrGen>
FORCE_INLINE void noc_read_helper(const uint32_t l1_write_addr, const AddrGen& s) {
    // Phase instrumentation (compiled out unless enable_phase_counters): split Tlat into
    // measured phases.
    //   t0 = issue loop start, t1 = issue loop end (all read commands sent),
    //   t2 = first read response observed, t3 = barrier clear (all responses in).
    // Host derives: T_issue = t1-t0, T_fill = t2-t0, T_stream = t3-t2, T_drain = t3-t1.
    // NIU_MST_RD_RESP_RECEIVED is the HW cumulative read-response counter for this NoC;
    // the "responses so far" value carried by each marker lets the host see how many
    // reads had already completed at each phase boundary.
    [[maybe_unused]] uint32_t resp_baseline = 0;
    if constexpr (enable_phase_counters) {
        resp_baseline = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED);
        DeviceTimestampedData("dm_t0_issue_start", (uint64_t)noc_index);
    }

    for (uint32_t i = 0; i < num_of_transactions; i++) {
        for (uint32_t p = 0; p < num_pages; p++) {
            noc_async_read_page(p, s, l1_write_addr + p * page_size_bytes);
        }
    }

    if constexpr (enable_phase_counters) {
        // NOTE: each DeviceTimestampedData(...) call MUST stay on a single source line.
        // For a multi-line macro invocation, GCC expands __LINE__ differently for the
        // device-side Hash16_CT() (macro-name line) vs the emitted #pragma message text
        // (argument line), so the host hash->name map never resolves the marker name.
        DeviceTimestampedData(
            "dm_t1_issue_end", (uint64_t)(NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED) - resp_baseline));

        // Wait for the first read response of this batch. It may already have arrived
        // during the issue loop (near channels / small pages), in which case t2 ~ t1.
        while (NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED) == resp_baseline) {
        }

        DeviceTimestampedData(
            "dm_t2_first_return", (uint64_t)(NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED) - resp_baseline));
    }

    noc_async_read_barrier();

    if constexpr (enable_phase_counters) {
        DeviceTimestampedData(
            "dm_t3_barrier_clear",
            (uint64_t)(NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED) - resp_baseline));
    }
}

// DRAM to L1 read
void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t l1_write_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(3);
    constexpr uint32_t test_id = get_compile_time_arg_val(4);
    constexpr bool sync = get_compile_time_arg_val(5) == 1;
    constexpr bool default_noc = get_compile_time_arg_val(6) == 1;
    constexpr bool enable_phase_counters = get_compile_time_arg_val(7) == 1;

    constexpr auto src_args = TensorAccessorArgs<8>();
    const auto s = TensorAccessor(src_args, src_addr);

    constexpr uint32_t transaction_size_bytes = page_size_bytes;
    DeviceTimestampedData("Number of transactions", num_of_transactions * num_pages);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    if constexpr (sync) {
        cb_reserve_back(cb_id_in0, 1);
    }
    if constexpr (default_noc) {
        {
            DeviceZoneScopedN("RISCV1");
            noc_read_helper<num_of_transactions, num_pages, page_size_bytes, enable_phase_counters>(l1_write_addr, s);
        }
    } else {
        {
            DeviceZoneScopedN("RISCV0");
            noc_read_helper<num_of_transactions, num_pages, page_size_bytes, enable_phase_counters>(l1_write_addr, s);
        }
    }
    if constexpr (sync) {
        cb_push_back(cb_id_in0, 1);
    }
}
