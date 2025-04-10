// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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

    ASSERT(is_power_of_two(NUM_BUFFER_SLOTS));

    const uint32_t full_payload_size = message_size + sizeof(eth_buffer_slot_sync_t);
    const uint32_t full_payload_size_eth_words = full_payload_size >> 4;

    uint32_t buffer_start_addr = handshake_addr + sizeof(eth_buffer_slot_sync_t);

    std::array<uint32_t, NUM_BUFFER_SLOTS> receiver_buffer_slot_addrs;
    std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS> receiver_buffer_slot_sync_addrs;
    buffer_start_addr = setup_receiver_buffer(
        receiver_buffer_slot_addrs, receiver_buffer_slot_sync_addrs, buffer_start_addr, message_size);

    // Only used for bi-directional cases
    std::array<uint32_t, NUM_BUFFER_SLOTS> sender_buffer_slot_addrs;
    std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS> sender_buffer_slot_sync_addrs;
    if constexpr (benchmark_type == BenchmarkType::EthOnlyBiDir or benchmark_type == BenchmarkType::EthEthTensixBiDir) {
        setup_sender_buffer(sender_buffer_slot_addrs, sender_buffer_slot_sync_addrs, buffer_start_addr, message_size);
    }

    // Avoids hang in issue https://github.com/tenstorrent/tt-metal/issues/9963
    for (uint32_t i = 0; i < 2000000000; i++) {
        asm volatile("nop");
    }

    uint64_t worker_noc_addr = get_noc_addr(worker_noc_x, worker_noc_y, worker_buffer_addr);

    eth_setup_handshake(handshake_addr, false);

    switch (benchmark_type) {
        case EthOnlyUniDir: {
            DeviceZoneScopedN("MAIN-TEST-BODY");
            receiver_uni_dir<false>(
                receiver_buffer_slot_addrs,
                receiver_buffer_slot_sync_addrs,
                message_size,
                full_payload_size,
                num_messages,
                worker_noc_addr);
        } break;
        case EthOnlyBiDir: {
            DeviceZoneScopedN("MAIN-TEST-BODY");
            send_receiver_bi_dir<false>(
                sender_buffer_slot_addrs,
                sender_buffer_slot_sync_addrs,
                receiver_buffer_slot_addrs,
                receiver_buffer_slot_sync_addrs,
                full_payload_size,
                message_size,
                num_messages,
                worker_noc_addr);
        } break;
        case EthEthTensixUniDir: {
            DeviceZoneScopedN("MAIN-TEST-BODY");
            receiver_uni_dir<true>(
                receiver_buffer_slot_addrs,
                receiver_buffer_slot_sync_addrs,
                message_size,
                full_payload_size,
                num_messages,
                worker_noc_addr);
        } break;
        case EthEthTensixBiDir: {
            DeviceZoneScopedN("MAIN-TEST-BODY");
            send_receiver_bi_dir<true>(
                sender_buffer_slot_addrs,
                sender_buffer_slot_sync_addrs,
                receiver_buffer_slot_addrs,
                receiver_buffer_slot_sync_addrs,
                full_payload_size,
                message_size,
                num_messages,
                worker_noc_addr);

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
