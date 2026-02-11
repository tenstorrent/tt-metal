// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ethernet_write_worker_latency_ubench_common.hpp"

void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_messages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t message_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_buffer_addr = get_arg_val<uint32_t>(arg_idx++);

    uint32_t buffer_offset = 0;
    if constexpr (benchmark_type == BenchmarkType::DualEriscBiDir) {
        buffer_offset = get_arg_val<uint32_t>(arg_idx++);
    }

    ASSERT(is_power_of_two(NUM_BUFFER_SLOTS));

    uint32_t buffer_start_addr = handshake_addr + sizeof(eth_buffer_slot_sync_t) + buffer_offset;

    std::array<uint32_t, NUM_BUFFER_SLOTS> receiver_buffer_slot_addrs;
    buffer_start_addr = setup_receiver_buffer(receiver_buffer_slot_addrs, buffer_start_addr, message_size);

    // Only used for bi-directional cases
    std::array<uint32_t, NUM_BUFFER_SLOTS> sender_buffer_slot_addrs;
    if constexpr (benchmark_type == BenchmarkType::EthOnlyBiDir or benchmark_type == BenchmarkType::EthEthTensixBiDir) {
        setup_sender_buffer(sender_buffer_slot_addrs, buffer_start_addr, message_size);
    }

    // For TensixEthEthTensixUniDir, initialize push_counter before handshake
    if constexpr (benchmark_type == BenchmarkType::TensixEthEthTensixUniDir) {
        uint32_t push_counter_addr = handshake_addr + sizeof(eth_buffer_slot_sync_t) + message_size;
        volatile tt_l1_ptr uint32_t* push_counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(push_counter_addr);
        *push_counter = 0;
    }

    // Initialize overlay registers BEFORE handshake
    init_flow_control_registers();

    uint64_t worker_noc_addr = get_noc_addr(worker_noc_x, worker_noc_y, worker_buffer_addr);

    if constexpr (benchmark_type != BenchmarkType::DualEriscBiDir) {
        eth_setup_handshake(handshake_addr, false);
    }

    switch (benchmark_type) {
        case EthOnlyUniDir: {
            DeviceZoneScopedN("MAIN-TEST-BODY");
            receiver_uni_dir<false>(receiver_buffer_slot_addrs, message_size, num_messages, worker_noc_addr);
        } break;
        case EthOnlyBiDir: {
            DeviceZoneScopedN("MAIN-TEST-BODY");
            send_receiver_bi_dir<false>(
                sender_buffer_slot_addrs, receiver_buffer_slot_addrs, message_size, num_messages, worker_noc_addr);
        } break;
        case EthEthTensixUniDir: {
            DeviceZoneScopedN("MAIN-TEST-BODY");
            receiver_uni_dir<true>(receiver_buffer_slot_addrs, message_size, num_messages, worker_noc_addr);
        } break;
        case EthEthTensixBiDir: {
            DeviceZoneScopedN("MAIN-TEST-BODY");
            send_receiver_bi_dir<true>(
                sender_buffer_slot_addrs, receiver_buffer_slot_addrs, message_size, num_messages, worker_noc_addr);

        } break;
        case TensixEthEthTensixUniDir: {
            uint32_t push_counter_addr = handshake_addr + sizeof(eth_buffer_slot_sync_t) + message_size;
            uint64_t tensix_landing_noc_addr = get_noc_addr(worker_noc_x, worker_noc_y, worker_buffer_addr);
            uint64_t tensix_sem_noc_addr = get_noc_addr(worker_noc_x, worker_noc_y, worker_buffer_addr + message_size);

            DeviceZoneScopedN("MAIN-TEST-BODY");
            tensix_eth_receiver_loop(
                receiver_buffer_slot_addrs[0],
                push_counter_addr,
                tensix_landing_noc_addr,
                tensix_sem_noc_addr,
                message_size,
                num_messages);
        } break;
        case DualEriscBiDir: {
            receiver_uni_dir<false>(receiver_buffer_slot_addrs, message_size, num_messages, worker_noc_addr);
        } break;
        case TensixPushEth: {
            ASSERT(0);
        } break;
        case EthMcastTensix: {
            ASSERT(0);
        } break;
        case EthToLocalEth: {
            ASSERT(0);
        } break;
        case EthToLocalEthAndMcastTensix: {
            ASSERT(0);
        } break;
        default: WAYPOINT("!ETH"); ASSERT(0);
    }

    // need to do a delay as trid writes are not waiting for acks, so need to make sure noc response is back.
    for (int i = 0; i < 1000; ++i) {
        asm volatile("nop");
    }
    ncrisc_noc_counters_init();

    internal_::risc_context_switch();
}
