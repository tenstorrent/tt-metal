// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <bit>
/**
 * @brief Enumeration of code profiling timer types as bitfield
 * Each timer type is a unique bit position, allowing multiple timers to be enabled simultaneously
 */
enum class CodeProfilingTimerType : uint32_t {
    NONE = 0,
    RECEIVER_CHANNEL_FORWARD = 1 << 0,

    // Speedy sender path timers
    SPEEDY_SENDER_FULL = 1 << 1,               // Entire run_sender_channel_step_speedy call
    SPEEDY_SENDER_SEND_DATA = 1 << 2,          // send_next_data path (when can_send)
    SPEEDY_SENDER_CHECK_COMPLETIONS = 1 << 3,  // get_num_unprocessed_completions_from_receiver
    SPEEDY_SENDER_CREDITS_UPSTREAM = 1 << 4,   // send_credits_to_upstream_workers

    // Speedy receiver path timers
    SPEEDY_RECEIVER_FULL = 1 << 5,     // Entire run_receiver_channel_step_speedy call
    SPEEDY_RECEIVER_FORWARD = 1 << 6,  // Inner loop packet forwarding
    SPEEDY_RECEIVER_FLUSH = 1 << 7,    // Batched completion flush

    // Sender send_next_data sub-timers (breakdown of SPEEDY_SENDER_SEND_DATA)
    SPEEDY_SENDER_SEND_ETH = 1 << 8,      // L1 reads + spin-wait + eth_send_packet_bytes_unsafe
    SPEEDY_SENDER_SEND_ADV = 1 << 9,      // Pointer advances + bookkeeping
    SPEEDY_SENDER_SEND_NOTIFY = 1 << 10,  // Second spin-wait + remote_update_ptr_val

    // Receiver forward sub-timers (breakdown of SPEEDY_RECEIVER_FORWARD)
    SPEEDY_RECEIVER_FWD_HDR = 1 << 11,   // Cache invalidate + header load + packed load
    SPEEDY_RECEIVER_FWD_NOC = 1 << 12,   // execute_chip_unicast_to_local_chip_impl
    SPEEDY_RECEIVER_FWD_BOOK = 1 << 13,  // Counter increment + decrement pkts

    // Spin iteration counters (total_cycles = total iterations, num_instances = loop entries)
    SPEEDY_SENDER_ETH_TXQ_SPIN_1 = 1 << 14,        // eth_txq_is_busy before eth_send (send_next_data)
    SPEEDY_SENDER_ETH_TXQ_SPIN_2 = 1 << 15,        // eth_txq_is_busy before remote_update_ptr_val
    SPEEDY_SENDER_NOC_FLUSH_SPIN = 1 << 16,        // noc writes_sent flush in sender credits
    SPEEDY_SENDER_NOC_CMD_BUF_SPIN = 1 << 17,      // noc_cmd_buf_ready in sender credits
    SPEEDY_RECEIVER_NOC_CMD_BUF_SPIN = 1 << 18,    // noc_cmd_buf_ready in receiver forward
    SPEEDY_RECEIVER_FLUSH_ETH_TXQ_SPIN = 1 << 19,  // eth_txq_is_busy in receiver flush completion ack

    // Receiver flush sub-timers (breakdown of SPEEDY_RECEIVER_FLUSH)
    SPEEDY_RECEIVER_FLUSH_TRID = 1 << 20,  // trid check loop
    SPEEDY_RECEIVER_FLUSH_SEND = 1 << 21,  // completion ack ETH send

    LAST = 1 << 22  // Sentinel for size calculation
};

/**
 * @brief Structure to store accumulated code profiling results for a single timer type
 */
struct CodeProfilingTimerResult {
    uint64_t total_cycles;    // Total cycles accumulated across all captures
    uint64_t num_instances;   // Number of timer captures that occurred
    uint64_t min_cycles;      // Minimum cycle duration observed (initialized to UINT64_MAX)
    uint64_t max_cycles;      // Maximum cycle duration observed (initialized to 0)
};

/**
 * @brief Get the number of timer types defined in the enum
 * @return Number of timer types (excluding NONE and LAST)
 */
constexpr uint32_t get_num_code_profiling_timer_types() {
    return 22;  // 14 timers + 6 spin counters + 2 flush sub-timers
}

/**
 * @brief Get the maximum number of timer types supported
 * @return Maximum number of timer types
 */
constexpr uint32_t get_max_code_profiling_timer_types() {
    // get the bit offset of LAST
    return __builtin_ctz(static_cast<uint32_t>(CodeProfilingTimerType::LAST));
}
