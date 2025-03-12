// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
// clang-format on

using namespace tt::tt_fabric;

#define CONTROLLER_HANDSHAKE_START 0x4
#define CONTROLLER_HANDSHAKE_END 0x8

void kernel_main() {
    uint32_t rt_args_idx = 0;
    uint32_t time_seed = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t num_tx_workers = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t tx_signal_addr = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t host_signal_address = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t num_mcast_dests = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t mcast_encoding = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t sync_with_remote = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));

    // wait for sync from tx kernels
    while (*(volatile tt_l1_ptr uint32_t*)tx_signal_addr != num_tx_workers);

    // wait for signal from host
    // this is needed to know that all the routers are up and running on all the chips
    while (*(volatile tt_l1_ptr uint32_t*)host_signal_address == 0);

    tt_l1_ptr uint32_t* mcast_sem = reinterpret_cast<tt_l1_ptr uint32_t*>(0x100000);
    *mcast_sem = 1;

#ifdef FVC_MODE_PULL
    if (sync_with_remote) {
        uint32_t dest_device = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
        uint32_t remote_controller_noc_encoding = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
        uint32_t outbound_eth_chan = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));

        uint32_t remote_notification_addr = host_signal_address + 16;
        uint32_t pkt_hdr_address = remote_notification_addr + 16;
        uint32_t payload_address = pkt_hdr_address + PACKET_HEADER_SIZE_BYTES;
        uint32_t payload_size_bytes = 16;
        uint32_t client_interface_addr = payload_address + payload_size_bytes;

        volatile tt_l1_ptr uint32_t* data_buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload_address);
        volatile tt_l1_ptr uint32_t* poll_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_notification_addr);

        volatile tt_l1_ptr fabric_pull_client_interface_t* client_interface =
            reinterpret_cast<volatile tt_l1_ptr fabric_pull_client_interface_t*>(client_interface_addr);

        fabric_endpoint_init<decltype(client_interface), RoutingType::ROUTING_TABLE>(
            client_interface, outbound_eth_chan);

        uint64_t dst_noc_addr = get_noc_addr_helper(remote_controller_noc_encoding, remote_notification_addr);

        data_buf[0] = CONTROLLER_HANDSHAKE_START;
        fabric_async_write<AsyncWriteMode::ALL, RoutingType::ROUTING_TABLE>(
            client_interface,
            0,
            pkt_hdr_address,
            dest_device >> 16,
            dest_device & 0xFFFF,
            dst_noc_addr,
            payload_size_bytes + PACKET_HEADER_SIZE_BYTES);
        fabric_wait_for_pull_request_flushed(client_interface);

        while (*poll_addr != CONTROLLER_HANDSHAKE_START);

        data_buf[0] = CONTROLLER_HANDSHAKE_END;
        fabric_async_write<AsyncWriteMode::ALL, RoutingType::ROUTING_TABLE>(
            client_interface,
            0,
            pkt_hdr_address,
            dest_device >> 16,
            dest_device & 0xFFFF,
            dst_noc_addr,
            payload_size_bytes + PACKET_HEADER_SIZE_BYTES);
        fabric_wait_for_pull_request_flushed(client_interface);

        while (*poll_addr != CONTROLLER_HANDSHAKE_END);
    }
#endif

    // do a noc multicast to tx kernels
    uint64_t mcast_dest_addr = get_noc_addr_helper(mcast_encoding, tx_signal_addr);
    noc_async_write_multicast_loopback_src((uint32_t)mcast_sem, mcast_dest_addr, sizeof(uint32_t), num_mcast_dests);
    noc_async_writes_flushed();
}
