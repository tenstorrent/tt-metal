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

template <typename T>
inline void send_packet(
    T client_interface,
    uint32_t routing_plane,
    uint32_t src_addr,
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t size) {
#ifdef FVC_MODE_PULL
    fabric_async_write<ClientDataMode::PACKETIZED_DATA, AsyncWriteMode::ALL, RoutingType::ROUTING_TABLE>(
#else
    fabric_async_write<ClientDataMode::PACKETIZED_DATA, AsyncWriteMode::ALL>(
#endif
        client_interface, routing_plane, src_addr, dst_mesh_id, dst_dev_id, dst_addr, size);

#ifdef FVC_MODE_PULL
    fabric_wait_for_pull_request_flushed(client_interface);
#else
    noc_async_writes_flushed();
#endif
}

void kernel_main() {
    uint32_t rt_args_idx = 0;
    uint32_t time_seed = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t num_tx_workers = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t tx_signal_address = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t host_signal_address = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t num_mcast_dests = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t mcast_encoding = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t sync_with_remote = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));

    // wait for sync from tx kernels
    volatile tt_l1_ptr uint32_t* tx_signal_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tx_signal_address);
    while (*tx_signal_ptr != num_tx_workers);

    // wait for signal from host
    // this is needed to know that all the routers are up and running on all the chips
    volatile tt_l1_ptr uint32_t* host_signal_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(host_signal_address);
    while (*host_signal_ptr == 0);

    if (sync_with_remote) {
        uint32_t dest_device = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
        uint32_t remote_controller_noc_encoding = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
        uint32_t outbound_eth_chan = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));

        uint32_t remote_notification_address = host_signal_address + 16;
        uint32_t pkt_hdr_address = remote_notification_address + 16;
        uint32_t payload_address = pkt_hdr_address + PACKET_HEADER_SIZE_BYTES;
        uint32_t payload_size_bytes = 16;
        uint32_t client_interface_addr = payload_address + payload_size_bytes;

        uint32_t routing_plane = 0;
        uint16_t dest_mesh_id = dest_device >> 16;
        uint16_t dest_dev_id = dest_device & 0xFFFF;

        tt_l1_ptr uint32_t* data_buf = reinterpret_cast<tt_l1_ptr uint32_t*>(payload_address);
        volatile tt_l1_ptr uint32_t* poll_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_notification_address);

#ifdef FVC_MODE_PULL
        volatile fabric_pull_client_interface_t* client_interface =
            reinterpret_cast<volatile fabric_pull_client_interface_t*>(client_interface_addr);
#else
        volatile fabric_push_client_interface_t* client_interface =
            reinterpret_cast<volatile fabric_push_client_interface_t*>(client_interface_addr);
#endif

        fabric_endpoint_init<RoutingType::ROUTING_TABLE>(client_interface, outbound_eth_chan);

#ifndef FVC_MODE_PULL
        fabric_client_connect(client_interface, routing_plane, dest_mesh_id, dest_dev_id);
#endif

        uint64_t dst_noc_addr = get_noc_addr_helper(remote_controller_noc_encoding, remote_notification_address);

        data_buf[0] = CONTROLLER_HANDSHAKE_START;
        send_packet(
            client_interface,
            routing_plane,
            pkt_hdr_address,
            dest_mesh_id,
            dest_dev_id,
            dst_noc_addr,
            payload_size_bytes + PACKET_HEADER_SIZE_BYTES);

        while (*poll_addr != CONTROLLER_HANDSHAKE_START);

        data_buf[0] = CONTROLLER_HANDSHAKE_END;
        send_packet(
            client_interface,
            routing_plane,
            pkt_hdr_address,
            dest_mesh_id,
            dest_dev_id,
            dst_noc_addr,
            payload_size_bytes + PACKET_HEADER_SIZE_BYTES);

        while (*poll_addr != CONTROLLER_HANDSHAKE_END);

#ifndef FVC_MODE_PULL
        fabric_client_disconnect(client_interface);
#endif
    }

    tt_l1_ptr uint32_t* mcast_sem = reinterpret_cast<tt_l1_ptr uint32_t*>(0x100000);
    *mcast_sem = 1;

    // do a noc multicast to tx kernels
    uint64_t mcast_dest_addr = get_noc_addr_helper(mcast_encoding, tx_signal_address);
    noc_async_write_multicast_loopback_src((uint32_t)mcast_sem, mcast_dest_addr, sizeof(uint32_t), num_mcast_dests);
    noc_async_writes_flushed();
}
