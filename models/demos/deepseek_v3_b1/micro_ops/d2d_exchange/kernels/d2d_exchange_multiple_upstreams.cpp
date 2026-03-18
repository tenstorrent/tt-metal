// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/debug/dprint.h"

constexpr uint32_t sender_socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t num_upstream_sockets = get_compile_time_arg_val(1);
constexpr uint32_t upstream_page_size = get_compile_time_arg_val(2);
constexpr uint32_t termination_semaphore_addr = get_compile_time_arg_val(3);
constexpr uint32_t page_size = get_compile_time_arg_val(4);
constexpr uint32_t num_whole_fabric_packets_per_link = get_compile_time_arg_val(5);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(6);
constexpr uint32_t partial_packet_size = get_compile_time_arg_val(7);
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(8);
constexpr bool use_fabric_on_receiver = get_compile_time_arg_val(9);
constexpr bool use_fabric_on_sender = get_compile_time_arg_val(10);

constexpr uint32_t receiver_socket_config_addr_0 = get_compile_time_arg_val(11);
constexpr uint32_t receiver_socket_config_addr_1 = get_compile_time_arg_val(12);
constexpr uint32_t receiver_socket_config_addr_2 = get_compile_time_arg_val(13);
constexpr uint32_t receiver_socket_config_addr_3 = get_compile_time_arg_val(14);
constexpr uint32_t receiver_socket_config_addr_4 = get_compile_time_arg_val(15);
constexpr uint32_t receiver_socket_config_addr_5 = get_compile_time_arg_val(16);
constexpr uint32_t receiver_socket_config_addr_6 = get_compile_time_arg_val(17);
constexpr uint32_t receiver_socket_config_addr_7 = get_compile_time_arg_val(18);

constexpr uint32_t receiver_socket_config_addrs[8] = {
    receiver_socket_config_addr_0,
    receiver_socket_config_addr_1,
    receiver_socket_config_addr_2,
    receiver_socket_config_addr_3,
    receiver_socket_config_addr_4,
    receiver_socket_config_addr_5,
    receiver_socket_config_addr_6,
    receiver_socket_config_addr_7,
};

FORCE_INLINE void write_data_to_local_core_with_ack(uint32_t l1_read_addr, uint64_t dst_addr, uint32_t size) {
    noc_async_write(l1_read_addr, dst_addr, size);
    noc_async_writes_flushed();
}

template <bool flush = true>
FORCE_INLINE void write_data_to_remote_core_with_ack(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header_addr,
    uint32_t l1_read_addr,
    uint64_t dst_addr,
    uint64_t downstream_bytes_sent_noc_addr,
    uint32_t packet_size) {
    DPRINT << "writing packet with size: " << (uint32_t)packet_size << " TO REMOTE CORE\n";
    packet_header_addr->to_noc_fused_unicast_write_atomic_inc(
        NocUnicastAtomicIncFusedCommandHeader{dst_addr, downstream_bytes_sent_noc_addr, packet_size, false},
        packet_size);
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_without_header_non_blocking_from_address(l1_read_addr, packet_size);
    if constexpr (flush) {
        fabric_connection.send_payload_flush_blocking_from_address(
            (uint32_t)packet_header_addr, sizeof(PACKET_HEADER_TYPE));
    } else {
        fabric_connection.send_payload_flush_non_blocking_from_address(
            (uint32_t)packet_header_addr, sizeof(PACKET_HEADER_TYPE));
    }
    DPRINT << "finished writing packet with size: " << (uint32_t)packet_size << " TO REMOTE CORE\n";
}

FORCE_INLINE void send_worker_data_over_fabric(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header_addr,
    uint32_t l1_read_addr,
    uint64_t dst_addr,
    uint64_t downstream_bytes_sent_noc_addr) {
    uint32_t src = l1_read_addr;
    uint64_t dst = dst_addr;
    DPRINT << "num whole fabric packets per link: " << (uint32_t)num_whole_fabric_packets_per_link << "\n";
    if constexpr (partial_packet_size > 0) {
        DPRINT << "partial packet size: " << (uint32_t)(partial_packet_size) << "\n";
        for (uint32_t i = 0; i < num_whole_fabric_packets_per_link; ++i) {
            DPRINT << "num whole packets sent: " << (uint32_t)(num_whole_fabric_packets_per_link) << "\n";
            DPRINT << "whole packet size: " << (uint32_t)(whole_packet_size) << "\n";

            write_data_to_remote_core_with_ack<false>(
                fabric_connection, packet_header_addr, src, dst, downstream_bytes_sent_noc_addr, whole_packet_size);
            src += whole_packet_size;
            dst += whole_packet_size;
        }
        write_data_to_remote_core_with_ack(
            fabric_connection, packet_header_addr, src, dst, downstream_bytes_sent_noc_addr, partial_packet_size);
    } else if constexpr (num_whole_fabric_packets_per_link > 0) {
        DPRINT << "in else num whole fabric packet per link: " << (uint32_t)(num_whole_fabric_packets_per_link) << "\n";
        DPRINT << "whole packet size: " << (uint32_t)(whole_packet_size) << "\n";

        for (uint32_t i = 0; i < num_whole_fabric_packets_per_link - 1; ++i) {
            write_data_to_remote_core_with_ack<false>(
                fabric_connection, packet_header_addr, src, dst, downstream_bytes_sent_noc_addr, whole_packet_size);
            src += whole_packet_size;
            dst += whole_packet_size;
        }
        write_data_to_remote_core_with_ack(
            fabric_connection, packet_header_addr, src, dst, downstream_bytes_sent_noc_addr, whole_packet_size);
    }
}

void kernel_main() {
    DPRINT << "start of reduce to one kernel with multiple upstreams\n";
    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender downstream_fabric_connection;
    tt::tt_fabric::WorkerToFabricEdmSender downstream_fabric_connection_2;
    tt::tt_fabric::WorkerToFabricEdmSender upstream_fabric_connection;

    if constexpr (use_fabric_on_sender) {
        downstream_fabric_connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
        downstream_fabric_connection_2 =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    }
    if constexpr (use_fabric_on_receiver) {
        upstream_fabric_connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    }

    SocketSenderInterface sender_socket = create_sender_socket_interface(sender_socket_config_addr);
    set_sender_socket_page_size(sender_socket, page_size);
    DPRINT << "set sender socket page size to :" << (uint32_t)page_size << "\n";
    sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);

    SocketReceiverInterface receiver_sockets[num_upstream_sockets];
    DPRINT << "num upstream sockets: " << (uint32_t)num_upstream_sockets << "\n";
    for (uint32_t i = 0; i < num_upstream_sockets; i++) {
        DPRINT << "receiver socket " << i << " config addr: " << (uint32_t)receiver_socket_config_addrs[i] << "\n";
        receiver_sockets[i] = create_receiver_socket_interface(receiver_socket_config_addrs[i]);
        set_receiver_socket_page_size(receiver_sockets[i], upstream_page_size);
        DPRINT << "set receiver socket " << i << " page size to :" << (uint32_t)upstream_page_size << "\n";
    }

    uint64_t downstream_bytes_sent_noc_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x,
        downstream_enc.d2d.downstream_noc_y,
        sender_socket.downstream_bytes_sent_addr);
    uint64_t downstream_data_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x, downstream_enc.d2d.downstream_noc_y, sender_socket.downstream_fifo_addr);

    uint64_t upstream_bytes_acked_noc_addrs[num_upstream_sockets];
    if constexpr (use_fabric_on_receiver) {
        for (uint32_t i = 0; i < num_upstream_sockets; i++) {
            upstream_bytes_acked_noc_addrs[i] = get_noc_addr(
                receiver_sockets[i].d2d.upstream_noc_x,
                receiver_sockets[i].d2d.upstream_noc_y,
                receiver_sockets[i].d2d.upstream_bytes_acked_addr);
        }
    }

    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_data_packet_header_addr = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_data_packet_header_addr_2 = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* upstream_socket_packet_header_addr = nullptr;

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);

    if constexpr (use_fabric_on_sender) {
        downstream_data_packet_header_addr =
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
        downstream_data_packet_header_addr_2 = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + sizeof(PACKET_HEADER_TYPE));

        downstream_fabric_connection.open();
        downstream_fabric_connection_2.open();

        fabric_set_unicast_route(downstream_data_packet_header_addr, downstream_enc);
        fabric_set_unicast_route(downstream_data_packet_header_addr_2, downstream_enc);
    }
    if constexpr (use_fabric_on_receiver) {
        upstream_socket_packet_header_addr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + 2 * sizeof(PACKET_HEADER_TYPE));

        upstream_fabric_connection.open();
    }

    tt::tt_fabric::WorkerToFabricEdmSender* fabric_links[2] = {
        &downstream_fabric_connection, &downstream_fabric_connection_2};
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_headers[2] = {
        downstream_data_packet_header_addr, downstream_data_packet_header_addr_2};

    uint32_t current_link = 0;
    bool terminated = false;

    while (!terminated) {
        socket_reserve_pages(sender_socket, 1);

        invalidate_l1_cache();
        if (termination_semaphore[0] == 1) {
            break;
        }

        uint64_t dst_addr_base = downstream_data_addr + sender_socket.write_ptr;
        uint32_t remaining = num_upstream_sockets;
        uint32_t worker_idx = 0;
        uint32_t processed_mask = 0;
        DPRINT << "remaining: " << (uint32_t)remaining << "\n";
        while (remaining > 0) {
            invalidate_l1_cache();
            if (termination_semaphore[0] == 1) {
                terminated = true;
                break;
            }

            if (!(processed_mask & (1 << worker_idx)) && socket_wait_for_pages(receiver_sockets[worker_idx], 1, 1000)) {
                DPRINT << "Processing worker idx: " << worker_idx << "\n";
                uint32_t l1_read_addr = receiver_sockets[worker_idx].read_ptr;
                uint64_t dst_addr = dst_addr_base + worker_idx * upstream_page_size;

                if constexpr (use_fabric_on_sender) {
                    DPRINT << "Sending data of worker idx " << worker_idx << " over fabric\n";
                    send_worker_data_over_fabric(
                        *fabric_links[current_link],
                        packet_headers[current_link],
                        l1_read_addr,
                        dst_addr,
                        downstream_bytes_sent_noc_addr);
                } else {
                    DPRINT << "Sending data of worker idx " << worker_idx << " over socket\n";
                    write_data_to_local_core_with_ack(l1_read_addr, dst_addr, upstream_page_size);
                }

                socket_pop_pages(receiver_sockets[worker_idx], 1);
                DPRINT << "Popped pages for worker idx " << worker_idx << "\n";

                if constexpr (use_fabric_on_receiver) {
                    DPRINT << "Notifying sender for worker idx " << worker_idx << " over fabric\n";
                    fabric_set_unicast_route(upstream_socket_packet_header_addr, receiver_sockets[worker_idx]);
                    fabric_socket_notify_sender_stateful(
                        receiver_sockets[worker_idx],
                        upstream_fabric_connection,
                        upstream_socket_packet_header_addr,
                        upstream_bytes_acked_noc_addrs[worker_idx]);
                } else {
                    DPRINT << "Notifying sender for worker idx " << worker_idx << " over socket\n";
                    socket_notify_sender(receiver_sockets[worker_idx]);
                }

                processed_mask |= (1 << worker_idx);
                current_link = (current_link + 1) % 2;
                remaining--;
            }

            worker_idx = (worker_idx + 1) % num_upstream_sockets;
            // DPRINT << "Next worker idx: " << worker_idx << "\n";
        }

        if (!terminated) {
            if constexpr (use_fabric_on_sender) {
                socket_push_pages(sender_socket, 1);
            } else {
                socket_push_pages(sender_socket, 1);
                socket_notify_receiver(sender_socket);
            }
            DPRINT << "Pushed pages to sender socket\n";
        }
    }

    update_socket_config(sender_socket);
    for (uint32_t i = 0; i < num_upstream_sockets; i++) {
        update_socket_config(receiver_sockets[i]);
    }

    if constexpr (use_fabric_on_receiver) {
        upstream_fabric_connection.close();
    }

    if constexpr (use_fabric_on_sender) {
        downstream_fabric_connection.close();
        downstream_fabric_connection_2.close();
    }
    DPRINT << "end of d2d kernel with multiple upstreams\n";
}
