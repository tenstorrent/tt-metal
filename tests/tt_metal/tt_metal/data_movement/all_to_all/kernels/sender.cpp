// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "ckernel.h"
#include "tensix_types.h"
#include "barrier_sync.hpp"

// Read the 64-bit wall-clock register (same source the profiler uses). Used by the W=1
// latency probe to time each non-posted write's ack round trip explicitly.
FORCE_INLINE uint64_t read_wall_clock() {
    volatile uint32_t tt_reg_ptr* clock_lo =
        reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile uint32_t tt_reg_ptr* clock_hi =
        reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return clock_lo[0] | ((uint64_t)clock_hi[0] << 32);
}

// Blocking busy-wait for `cycles` NoC cycles (wraparound-safe). Open-loop stall knob.
FORCE_INLINE void spin_cycles(uint32_t cycles) {
    uint64_t start = read_wall_clock();
    while (read_wall_clock() < start + cycles) {
    }
}

void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t test_id = get_compile_time_arg_val(0);
    constexpr uint32_t mst_l1_base_address = get_compile_time_arg_val(1);
    constexpr uint32_t sub_l1_base_address = get_compile_time_arg_val(2);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(3);
    constexpr uint32_t bytes_per_transaction_per_master = get_compile_time_arg_val(4);
    constexpr uint32_t num_subordinates = get_compile_time_arg_val(5);
    constexpr uint32_t num_virtual_channels = get_compile_time_arg_val(6);
    // Injection-rate knobs for the fixed-full-grid throttled sweep (test 90 -> id 321).
    // All 0 for the pre-existing configs (300-309), which compile the original path below
    // byte-for-byte (the sweep branch is `if constexpr`-eliminated when max_outstanding == 0).
    constexpr uint32_t stall_cycles = get_compile_time_arg_val(7);
    constexpr uint32_t max_outstanding = get_compile_time_arg_val(8);  // closed-loop window W
    constexpr bool latency_probe = get_compile_time_arg_val(9) == 1;
    // Distinct receiver-offset striping: each master writes to receiver slot master_index%recv_slots
    // on every subordinate, so masters do not all serialize on one L1 address (bug fix).
    constexpr uint32_t recv_slots = get_compile_time_arg_val(10);

    if constexpr (max_outstanding == 0) {
        // ============================= ORIGINAL PATH (configs 300-309) =============================
        uint32_t master_l1_local_address = mst_l1_base_address;
        uint32_t subordinate_l1_local_address = sub_l1_base_address;

        uint32_t subordinate_x_coord;
        uint32_t subordinate_y_coord;
        uint64_t subordinate_l1_noc_address;

        {
            DeviceZoneScopedN("RISCV0");
            for (uint32_t j = 0; j < num_subordinates; j++) {
                subordinate_x_coord = get_arg_val<uint32_t>(j * 2);
                subordinate_y_coord = get_arg_val<uint32_t>(j * 2 + 1);

                subordinate_l1_noc_address =
                    get_noc_addr(subordinate_x_coord, subordinate_y_coord, subordinate_l1_local_address);

                for (uint32_t i = 0; i < num_of_transactions; i++) {
                    uint32_t current_virtual_channel = i % num_virtual_channels;
                    noc_async_write(
                        master_l1_local_address,
                        subordinate_l1_noc_address,
                        bytes_per_transaction_per_master,
                        noc_index,
                        current_virtual_channel);
                }
            }
            noc_async_write_barrier();
        }

        DeviceTimestampedData("Test id", test_id);
        DeviceTimestampedData("NoC Index", noc_index);
        DeviceTimestampedData("Number of transactions", num_of_transactions * num_subordinates);
        DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction_per_master);
        DeviceTimestampedData("Number of Virtual Channels", num_virtual_channels);
        DeviceTimestampedData("Number of subordinates", num_subordinates);
        return;
    }

    // ============================= THROTTLED INJECTION SWEEP (test 90 -> id 321) =============================
    // Runtime args: [0 .. 2*num_subordinates) = subordinate (x,y) coords, then barrier + master_index.
    constexpr uint32_t barrier_arg_base = num_subordinates * 2;
    const uint32_t barrier_sem_id = get_arg_val<uint32_t>(barrier_arg_base + 0);
    const uint32_t barrier_coord_x = get_arg_val<uint32_t>(barrier_arg_base + 1);
    const uint32_t barrier_coord_y = get_arg_val<uint32_t>(barrier_arg_base + 2);
    const uint32_t num_cores = get_arg_val<uint32_t>(barrier_arg_base + 3);
    const uint32_t local_barrier_addr = get_arg_val<uint32_t>(barrier_arg_base + 4);
    const uint32_t master_index = get_arg_val<uint32_t>(barrier_arg_base + 5);

    constexpr uint32_t N = bytes_per_transaction_per_master;
    // Per-master distinct receiver offset on each subordinate (bug fix): masters spread across
    // `recv_slots` L1 offsets instead of every master writing the same address.
    const uint32_t recv_offset = (master_index % recv_slots) * N;

    // Align the timed region across all masters so the wall-clock window is genuinely concurrent.
    barrier_sync(barrier_sem_id, barrier_coord_x, barrier_coord_y, num_cores, local_barrier_addr);

    if constexpr (latency_probe) {
        // W=1 latency probe: serialize to one outstanding non-posted write and time each ack
        // round trip explicitly. The stall (after timing) only lowers the grid-wide offered rate.
        DeviceZoneScopedN("RISCV0");
        const uint32_t ack_baseline = NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED);
        uint64_t roundtrip_sum = 0;
        uint32_t issued = 0;
        for (uint32_t j = 0; j < num_subordinates; j++) {
            uint32_t sub_x = get_arg_val<uint32_t>(j * 2);
            uint32_t sub_y = get_arg_val<uint32_t>(j * 2 + 1);
            uint64_t sub_noc_addr = get_noc_addr(sub_x, sub_y, sub_l1_base_address + recv_offset);
            for (uint32_t i = 0; i < num_of_transactions; i++) {
                const uint64_t t_issue = read_wall_clock();
                noc_async_write(mst_l1_base_address, sub_noc_addr, N, noc_index, 0);
                issued++;
                while ((NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED) - ack_baseline) < issued) {
                }
                roundtrip_sum += (read_wall_clock() - t_issue);
                if constexpr (stall_cycles > 0) {
                    spin_cycles(stall_cycles);
                }
            }
        }
        noc_async_write_barrier();
        constexpr uint32_t total_writes = num_of_transactions * num_subordinates;
        // NOTE: each DeviceTimestampedData(...) call MUST stay on a single source line.
        DeviceTimestampedData("Read roundtrip mean", (uint64_t)(roundtrip_sum / total_writes));
    } else {
        // Closed-loop window: hold at most W non-posted writes in flight, paced to acks via the
        // HW write-ack counter, so the measured busy time is genuine latency (Little's law).
        DeviceZoneScopedN("RISCV0");
        const uint32_t ack_baseline = NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED);
        uint32_t issued = 0;
        for (uint32_t j = 0; j < num_subordinates; j++) {
            uint32_t sub_x = get_arg_val<uint32_t>(j * 2);
            uint32_t sub_y = get_arg_val<uint32_t>(j * 2 + 1);
            uint64_t sub_noc_addr = get_noc_addr(sub_x, sub_y, sub_l1_base_address + recv_offset);
            for (uint32_t i = 0; i < num_of_transactions; i++) {
                while ((issued - (NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED) - ack_baseline)) >=
                       max_outstanding) {
                }
                noc_async_write(mst_l1_base_address, sub_noc_addr, N, noc_index, 0);
                issued++;
                if constexpr (stall_cycles > 0) {
                    spin_cycles(stall_cycles);
                }
            }
        }
        noc_async_write_barrier();
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("NoC Index", noc_index);
    DeviceTimestampedData("Number of transactions", num_of_transactions * num_subordinates);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction_per_master);
    DeviceTimestampedData("Number of subordinates", num_subordinates);
    DeviceTimestampedData("Stall cycles", (uint64_t)stall_cycles);
    DeviceTimestampedData("Max outstanding", (uint64_t)max_outstanding);
    DeviceTimestampedData("Latency probe", (uint64_t)(latency_probe ? 1 : 0));
}
