// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "../../../unified_kernels/termination.hpp"

constexpr uint32_t sender_socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t receiver_socket_config_addr = get_compile_time_arg_val(1);
constexpr uint32_t termination_semaphore_addr = get_compile_time_arg_val(2);
constexpr uint32_t page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_whole_fabric_packets_per_link = get_compile_time_arg_val(4);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(5);
constexpr uint32_t partial_packet_size = get_compile_time_arg_val(6);
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(7);
constexpr bool use_fabric_on_receiver = get_compile_time_arg_val(8);
constexpr bool use_fabric_on_sender = get_compile_time_arg_val(9);
constexpr uint32_t num_fabric_connections = 2;
constexpr uint32_t downstream_header_variants_per_link = partial_packet_size > 0 ? 2 : 1;
constexpr uint32_t num_downstream_packet_headers =
    use_fabric_on_sender ? num_fabric_connections * downstream_header_variants_per_link : 0;

FORCE_INLINE void write_data_to_local_core_with_ack(
    SocketSenderInterface& sender_socket, uint32_t l1_read_addr, uint64_t dst_addr, uint32_t page_size) {
    noc_async_write(l1_read_addr, dst_addr, page_size);
    socket_push_pages(sender_socket, 1);
    socket_notify_receiver(sender_socket);
    // Flush here to ensure that NOC has picked up data before we pop pages in receiver socket.
    noc_async_writes_flushed();
}

struct DownstreamLinkState {
    volatile tt_l1_ptr PACKET_HEADER_TYPE* whole_packet_header_addr = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* partial_packet_header_addr = nullptr;
    uint32_t cached_free_write_slots = 0;
};

struct UpstreamNotifyState {
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header_addr = nullptr;
    uint32_t cached_free_write_slots = 0;
    bool flush_pending = false;
};

template <typename FabricConnection>
FORCE_INLINE void refill_free_write_slots(FabricConnection& fabric_connection, uint32_t& cached_free_write_slots) {
    do {
        cached_free_write_slots = fabric_connection.get_num_free_write_slots();
    } while (cached_free_write_slots == 0);
}

template <typename FabricConnection>
FORCE_INLINE void wait_for_cached_free_write_slot(
    FabricConnection& fabric_connection, uint32_t& cached_free_write_slots) {
    if (cached_free_write_slots == 0) {
        refill_free_write_slots(fabric_connection, cached_free_write_slots);
    }
}

FORCE_INLINE void initialize_downstream_packet_header(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header_addr,
    const sender_downstream_encoding& downstream_enc,
    uint64_t downstream_data_addr,
    uint64_t downstream_bytes_sent_noc_addr,
    uint32_t packet_size) {
    fabric_set_unicast_route(packet_header_addr, downstream_enc);
    packet_header_addr->to_noc_fused_unicast_write_atomic_inc(
        NocUnicastAtomicIncFusedCommandHeader{downstream_data_addr, downstream_bytes_sent_noc_addr, packet_size, false},
        packet_size);
}

FORCE_INLINE void send_packet_over_fabric(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header_addr,
    uint32_t& cached_free_write_slots,
    uint32_t l1_read_addr,
    uint64_t dst_addr,
    uint32_t packet_size) {
    packet_header_addr->set_fused_unicast_write_atomic_inc_write_noc_address(dst_addr);
    wait_for_cached_free_write_slot(fabric_connection, cached_free_write_slots);
    fabric_connection.send_current_slot_non_blocking(
        l1_read_addr, packet_size, reinterpret_cast<uint32_t>(packet_header_addr));
    cached_free_write_slots--;
}

FORCE_INLINE void send_whole_packet_over_fabric(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    DownstreamLinkState& downstream_link_state,
    uint32_t l1_read_addr,
    uint64_t dst_addr) {
    send_packet_over_fabric(
        fabric_connection,
        downstream_link_state.whole_packet_header_addr,
        downstream_link_state.cached_free_write_slots,
        l1_read_addr,
        dst_addr,
        whole_packet_size);
}

FORCE_INLINE void send_partial_packet_over_fabric(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    DownstreamLinkState& downstream_link_state,
    uint32_t l1_read_addr,
    uint64_t dst_addr) {
    send_packet_over_fabric(
        fabric_connection,
        downstream_link_state.partial_packet_header_addr,
        downstream_link_state.cached_free_write_slots,
        l1_read_addr,
        dst_addr,
        partial_packet_size);
}

FORCE_INLINE void notify_sender_over_fabric(
    const SocketReceiverInterface& receiver_socket,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    UpstreamNotifyState& upstream_notify_state) {
    upstream_notify_state.packet_header_addr->set_unicast_inline_write_value(receiver_socket.bytes_acked);
    wait_for_cached_free_write_slot(fabric_connection, upstream_notify_state.cached_free_write_slots);
    fabric_connection.send_payload_flush_non_blocking_from_address(
        reinterpret_cast<uint32_t>(upstream_notify_state.packet_header_addr), sizeof(PACKET_HEADER_TYPE));
    upstream_notify_state.cached_free_write_slots--;
    upstream_notify_state.flush_pending = true;
}

FORCE_INLINE void send_pages_over_socket(
    SocketSenderInterface& sender_socket,
    tt::tt_fabric::WorkerToFabricEdmSender& downstream_fabric_connection,
    tt::tt_fabric::WorkerToFabricEdmSender& downstream_fabric_connection_2,
    DownstreamLinkState& downstream_link_state,
    DownstreamLinkState& downstream_link_state_2,
    uint32_t l1_read_addr,
    uint64_t dst_addr) {
    if constexpr (use_fabric_on_sender) {
        // Initialize base src + dst addrs pers link
        constexpr uint32_t page_size_per_link = page_size / num_fabric_connections;
        uint32_t l1_read_addr_0 = l1_read_addr;
        uint32_t l1_read_addr_1 = l1_read_addr + page_size_per_link;
        uint64_t dst_addr_0 = dst_addr;
        uint64_t dst_addr_1 = dst_addr + page_size_per_link;

        for (uint32_t i = 0; i < num_whole_fabric_packets_per_link; ++i) {
            send_whole_packet_over_fabric(
                downstream_fabric_connection, downstream_link_state, l1_read_addr_0, dst_addr_0);
            send_whole_packet_over_fabric(
                downstream_fabric_connection_2, downstream_link_state_2, l1_read_addr_1, dst_addr_1);
            noc_async_writes_flushed();
            l1_read_addr_0 += whole_packet_size;
            l1_read_addr_1 += whole_packet_size;
            dst_addr_0 += whole_packet_size;
            dst_addr_1 += whole_packet_size;
        }
        if constexpr (partial_packet_size > 0) {
            send_partial_packet_over_fabric(
                downstream_fabric_connection, downstream_link_state, l1_read_addr_0, dst_addr_0);
            send_partial_packet_over_fabric(
                downstream_fabric_connection_2, downstream_link_state_2, l1_read_addr_1, dst_addr_1);
            noc_async_writes_flushed();
        }
        socket_push_pages(sender_socket, 1);
    } else {
        write_data_to_local_core_with_ack(sender_socket, l1_read_addr, dst_addr, page_size);
    }
}

void kernel_main() {
    // Build Fabric Connections
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
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(receiver_socket_config_addr);
    set_sender_socket_page_size(sender_socket, page_size);
    set_receiver_socket_page_size(receiver_socket, page_size);
    sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);

    uint64_t downstream_bytes_sent_noc_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x,
        downstream_enc.d2d.downstream_noc_y,
        sender_socket.downstream_bytes_sent_addr);
    uint64_t upstream_bytes_acked_noc_addr = get_noc_addr(
        receiver_socket.d2d.upstream_noc_x,
        receiver_socket.d2d.upstream_noc_y,
        receiver_socket.d2d.upstream_bytes_acked_addr);
    uint64_t downstream_data_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x, downstream_enc.d2d.downstream_noc_y, sender_socket.downstream_fifo_addr);

    DownstreamLinkState downstream_link_state;
    DownstreamLinkState downstream_link_state_2;
    [[maybe_unused]] UpstreamNotifyState upstream_notify_state;

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);

    [[maybe_unused]] uint32_t packet_header_cb_base = 0;
    if constexpr (use_fabric_on_sender || use_fabric_on_receiver) {
        packet_header_cb_base = get_write_ptr(fabric_packet_header_cb_id);
    }

    if constexpr (use_fabric_on_sender) {
        uint32_t header_addr = packet_header_cb_base;
        downstream_link_state.whole_packet_header_addr =
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header_addr);
        header_addr += sizeof(PACKET_HEADER_TYPE);
        if constexpr (partial_packet_size > 0) {
            downstream_link_state.partial_packet_header_addr =
                reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header_addr);
            header_addr += sizeof(PACKET_HEADER_TYPE);
        }
        downstream_link_state_2.whole_packet_header_addr =
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header_addr);
        header_addr += sizeof(PACKET_HEADER_TYPE);
        if constexpr (partial_packet_size > 0) {
            downstream_link_state_2.partial_packet_header_addr =
                reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header_addr);
        }

        downstream_fabric_connection.open();
        downstream_fabric_connection_2.open();

        initialize_downstream_packet_header(
            downstream_link_state.whole_packet_header_addr,
            downstream_enc,
            downstream_data_addr,
            downstream_bytes_sent_noc_addr,
            whole_packet_size);
        if constexpr (partial_packet_size > 0) {
            initialize_downstream_packet_header(
                downstream_link_state.partial_packet_header_addr,
                downstream_enc,
                downstream_data_addr,
                downstream_bytes_sent_noc_addr,
                partial_packet_size);
        }

        initialize_downstream_packet_header(
            downstream_link_state_2.whole_packet_header_addr,
            downstream_enc,
            downstream_data_addr,
            downstream_bytes_sent_noc_addr,
            whole_packet_size);
        if constexpr (partial_packet_size > 0) {
            initialize_downstream_packet_header(
                downstream_link_state_2.partial_packet_header_addr,
                downstream_enc,
                downstream_data_addr,
                downstream_bytes_sent_noc_addr,
                partial_packet_size);
        }
    }
    if constexpr (use_fabric_on_receiver) {
        upstream_notify_state.packet_header_addr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            packet_header_cb_base + num_downstream_packet_headers * sizeof(PACKET_HEADER_TYPE));

        upstream_fabric_connection.open();

        fabric_set_unicast_route(upstream_notify_state.packet_header_addr, receiver_socket);
        upstream_notify_state.packet_header_addr->to_noc_unicast_inline_write(
            NocUnicastInlineWriteCommandHeader{upstream_bytes_acked_noc_addr, receiver_socket.bytes_acked});
    }

    while (true) {
        socket_reserve_pages(sender_socket, 1);
        if constexpr (use_fabric_on_receiver) {
            if (upstream_notify_state.flush_pending) {
                // Ensure the prior ack is visible before we wait for the next page on this single upstream socket.
                noc_async_writes_flushed();
                upstream_notify_state.flush_pending = false;
            }
        }
        if (!deepseek_b1_ops::socket_wait_for_pages_with_termination(receiver_socket, 1, termination_semaphore)) {
            break;
        }

        auto l1_read_addr = receiver_socket.read_ptr;
        uint64_t dst_addr = downstream_data_addr + sender_socket.write_ptr;

        send_pages_over_socket(
            sender_socket,
            downstream_fabric_connection,
            downstream_fabric_connection_2,
            downstream_link_state,
            downstream_link_state_2,
            l1_read_addr,
            dst_addr);
        socket_pop_pages(receiver_socket, 1);
        if constexpr (use_fabric_on_receiver) {
            notify_sender_over_fabric(receiver_socket, upstream_fabric_connection, upstream_notify_state);
        } else {
            socket_notify_sender(receiver_socket);
        }
    }

    if constexpr (use_fabric_on_receiver) {
        if (upstream_notify_state.flush_pending) {
            noc_async_writes_flushed();
            upstream_notify_state.flush_pending = false;
        }
    }

    update_socket_config(sender_socket);
    update_socket_config(receiver_socket);

    if constexpr (use_fabric_on_receiver) {
        upstream_fabric_connection.close();
    }

    if constexpr (use_fabric_on_sender) {
        downstream_fabric_connection.close();
        downstream_fabric_connection_2.close();
    }
}
