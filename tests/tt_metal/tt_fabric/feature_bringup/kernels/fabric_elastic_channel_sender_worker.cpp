// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include <cstdint>
namespace tt::tt_fabric {
// temporary to avoid a decent chunk of conflicting code reorg that would be better done as an isolated change
constexpr uint8_t worker_handshake_noc = 0;
}  // namespace tt::tt_fabric

#include "tests/tt_metal/tt_fabric/feature_bringup/kernels/fabric_elastic_channels.hpp"
#include "core_config.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tensix.h"

// Timing tracking structure for worker operations
struct WorkerTimingStats {
    uint64_t total_idle_cycles;
    uint32_t idle_count;

    WorkerTimingStats() : total_idle_cycles(0), idle_count(0) {}
};

// Global timing stats stored in L1 - will be initialized in kernel_main
volatile WorkerTimingStats* worker_timing_stats = nullptr;

inline uint64_t read_wall_clock() {
    uint32_t low = memory_read(RISCV_DEBUG_REG_WALL_CLOCK_L);  // latches high
    uint32_t high = memory_read(RISCV_DEBUG_REG_WALL_CLOCK_H);

    return ((uint64_t)high << 32) | low;
}

void kernel_main() {
    constexpr size_t N_CHUNKS = get_compile_time_arg_val(0);
    constexpr size_t CHUNK_N_PKTS = get_compile_time_arg_val(1);

    size_t arg_idx = 0;
    size_t n_pkts = get_arg_val<size_t>(arg_idx++);
    size_t src_addr = get_arg_val<size_t>(arg_idx++);
    uint32_t dest_eth_noc_x = get_arg_val<size_t>(arg_idx++);
    uint32_t dest_eth_noc_y = get_arg_val<size_t>(arg_idx++);
    size_t payload_size_bytes = get_arg_val<size_t>(arg_idx++);
    size_t payload_size_words = payload_size_bytes / sizeof(uint32_t);

    auto next_chunk_ptr = reinterpret_cast<volatile uint32_t*>(
        get_semaphore<ProgrammableCoreType::TENSIX>(get_arg_val<uint32_t>(arg_idx++)));
    auto from_eth_flow_control_ptr = reinterpret_cast<volatile uint32_t*>(
        get_semaphore<ProgrammableCoreType::TENSIX>(get_arg_val<uint32_t>(arg_idx++)));
    size_t to_eth_flow_control_stream_id = get_arg_val<size_t>(arg_idx++);
    const uint32_t timing_stats_addr = get_arg_val<uint32_t>(arg_idx++);

    // Initialize timing stats at address provided by host
    worker_timing_stats = reinterpret_cast<volatile WorkerTimingStats*>(timing_stats_addr);
    worker_timing_stats->total_idle_cycles = 0;
    worker_timing_stats->idle_count = 0;

    tt::tt_fabric::WorkerFabricWriterAdapter fabric_writer_adapter(next_chunk_ptr, CHUNK_N_PKTS, payload_size_bytes);

    const uint64_t dest_sem_noc_addr =
        get_noc_addr(dest_eth_noc_x, dest_eth_noc_y, get_stream_reg_write_addr(to_eth_flow_control_stream_id));
    size_t pkts_sent = 0;

    bool sent_once = false;
    bool timer_open = false;
    uint64_t idle_start_cycles = 0;
    while (pkts_sent < n_pkts) {
        if (fabric_writer_adapter.has_valid_destination()) {
            auto dest_bank_addr = fabric_writer_adapter.get_next_write_address();
            auto dest_noc_addr = get_noc_addr(dest_eth_noc_x, dest_eth_noc_y, dest_bank_addr);
            noc_async_write(src_addr, dest_noc_addr, payload_size_bytes);
            noc_inline_dw_write(dest_sem_noc_addr, pack_value_for_inc_on_write_stream_reg_write(1));

            fabric_writer_adapter.advance_to_next_buffer_slot();
            pkts_sent++;

            sent_once = true;
        } else if (fabric_writer_adapter.new_chunk_is_available()) {
            if (sent_once && timer_open) {
                timer_open = false;
                uint64_t idle_end_cycles = read_wall_clock();
                worker_timing_stats->total_idle_cycles += (idle_end_cycles - idle_start_cycles);
                worker_timing_stats->idle_count++;
            }
            fabric_writer_adapter.update_to_new_chunk();
        } else if (sent_once && !timer_open) {
            timer_open = true;
            // Idle time - waiting for new chunk or valid destination
            idle_start_cycles = read_wall_clock();
        }
    }
}
