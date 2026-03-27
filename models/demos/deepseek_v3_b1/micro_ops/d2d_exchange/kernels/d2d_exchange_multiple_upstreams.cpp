// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Dual-RISC D2D exchange with multiple upstream sockets.
//
// BRISC (Writer, link 0) and NCRISC (Reader, link 1) each handle half the
// upstream sockets in parallel
// BRISC owns the sender (downstream) socket and coordinates via two L1
// semaphore words:
//   page_ready_sem  – BRISC -> NCRISC: downstream page is reserved
//   ncrisc_done_sem – NCRISC -> BRISC: NCRISC finished its sockets
//

#include <array>
#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

constexpr uint32_t sender_socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t num_sockets_this_risc = get_compile_time_arg_val(1);
constexpr uint32_t upstream_page_size = get_compile_time_arg_val(2);
constexpr uint32_t termination_semaphore_addr = get_compile_time_arg_val(3);
constexpr uint32_t page_size = get_compile_time_arg_val(4);
constexpr uint32_t num_whole_fabric_packets_per_link = get_compile_time_arg_val(5);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(6);
constexpr uint32_t partial_packet_size = get_compile_time_arg_val(7);
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(8);
constexpr bool use_fabric_on_receiver = get_compile_time_arg_val(9);
constexpr bool use_fabric_on_sender = get_compile_time_arg_val(10);
constexpr uint32_t page_ready_sem_addr = get_compile_time_arg_val(11);
constexpr uint32_t ncrisc_done_sem_addr = get_compile_time_arg_val(12);
constexpr uint32_t socket_start_idx = get_compile_time_arg_val(13);
constexpr uint32_t packet_header_slot_start = get_compile_time_arg_val(14);

constexpr uint32_t receiver_socket_addrs_start_idx = 15;

template <size_t START_IDX, size_t COUNT, size_t I = 0>
struct CTAArrayFiller {
    static constexpr void fill(std::array<uint32_t, COUNT>& arr) {
        arr[I] = get_compile_time_arg_val(START_IDX + I);
        if constexpr (I + 1 < COUNT) {
            CTAArrayFiller<START_IDX, COUNT, I + 1>::fill(arr);
        }
    }
};

template <size_t START_IDX, size_t COUNT>
constexpr std::array<uint32_t, COUNT> fill_ct_args_array() {
    std::array<uint32_t, COUNT> arr{};
    if constexpr (COUNT > 0) {
        CTAArrayFiller<START_IDX, COUNT>::fill(arr);
    }
    return arr;
}

constexpr auto receiver_socket_config_addrs =
    fill_ct_args_array<receiver_socket_addrs_start_idx, num_sockets_this_risc>();

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
}

FORCE_INLINE void send_worker_data_over_fabric(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header_addr,
    uint32_t l1_read_addr,
    uint64_t dst_addr,
    uint64_t downstream_bytes_sent_noc_addr) {
    uint32_t src = l1_read_addr;
    uint64_t dst = dst_addr;
    if constexpr (partial_packet_size > 0) {
        for (uint32_t i = 0; i < num_whole_fabric_packets_per_link; ++i) {
            write_data_to_remote_core_with_ack<false>(
                fabric_connection, packet_header_addr, src, dst, downstream_bytes_sent_noc_addr, whole_packet_size);
            src += whole_packet_size;
            dst += whole_packet_size;
        }
        write_data_to_remote_core_with_ack(
            fabric_connection, packet_header_addr, src, dst, downstream_bytes_sent_noc_addr, partial_packet_size);
    } else if constexpr (num_whole_fabric_packets_per_link > 0) {
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

// Process this RISC's subset of upstream sockets: receive data, forward via
// fabric (or local NOC), pop the upstream socket, and ack the upstream sender.
// Returns true if termination was detected during processing.
FORCE_INLINE bool process_upstream_sockets(
    SocketReceiverInterface* receiver_sockets,
    tt::tt_fabric::WorkerToFabricEdmSender& downstream_fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_packet_header,
    uint64_t dst_addr_base,
    uint64_t downstream_bytes_sent_noc_addr,
    tt::tt_fabric::WorkerToFabricEdmSender& upstream_fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* upstream_packet_header,
    uint64_t* upstream_bytes_acked_noc_addrs,
    volatile tt_l1_ptr uint32_t* termination_semaphore) {
    uint32_t remaining = num_sockets_this_risc;
    uint32_t worker_idx = 0;
    uint32_t processed_mask = 0;

    while (remaining > 0) {
        invalidate_l1_cache();
        if (termination_semaphore[0] == 1) {
            return true;
        }

        if (!(processed_mask & (1 << worker_idx)) && socket_wait_for_pages(receiver_sockets[worker_idx], 1, 1000)) {
            uint32_t l1_read_addr = receiver_sockets[worker_idx].read_ptr;
            uint64_t dst_addr = dst_addr_base + (socket_start_idx + worker_idx) * upstream_page_size;

            if constexpr (use_fabric_on_sender) {
                send_worker_data_over_fabric(
                    downstream_fabric_connection,
                    downstream_packet_header,
                    l1_read_addr,
                    dst_addr,
                    downstream_bytes_sent_noc_addr);
            } else {
                write_data_to_local_core_with_ack(l1_read_addr, dst_addr, upstream_page_size);
            }

            socket_pop_pages(receiver_sockets[worker_idx], 1);

            if constexpr (use_fabric_on_receiver) {
                fabric_set_unicast_route(upstream_packet_header, receiver_sockets[worker_idx]);
                fabric_socket_notify_sender_stateful(
                    receiver_sockets[worker_idx],
                    upstream_fabric_connection,
                    upstream_packet_header,
                    upstream_bytes_acked_noc_addrs[worker_idx]);
            } else {
                socket_notify_sender(receiver_sockets[worker_idx]);
            }

            processed_mask |= (1 << worker_idx);
            remaining--;
        }

        worker_idx = (worker_idx + 1) % num_sockets_this_risc;
    }
    return false;
}

// ============================================================================
// BRISC – owns the sender socket, handles upstream sockets [0..N/2), link 0
// ============================================================================
#if defined(COMPILE_FOR_BRISC)
void kernel_main() {
    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender downstream_fabric_connection;
    tt::tt_fabric::WorkerToFabricEdmSender upstream_fabric_connection;

    if constexpr (use_fabric_on_sender) {
        downstream_fabric_connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    }
    if constexpr (use_fabric_on_receiver) {
        upstream_fabric_connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    }

    SocketSenderInterface sender_socket = create_sender_socket_interface(sender_socket_config_addr);
    set_sender_socket_page_size(sender_socket, page_size);
    sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);

    SocketReceiverInterface receiver_sockets[num_sockets_this_risc];
    for (uint32_t i = 0; i < num_sockets_this_risc; i++) {
        receiver_sockets[i] = create_receiver_socket_interface(receiver_socket_config_addrs[i]);
        set_receiver_socket_page_size(receiver_sockets[i], upstream_page_size);
    }

    uint64_t downstream_bytes_sent_noc_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x,
        downstream_enc.d2d.downstream_noc_y,
        sender_socket.downstream_bytes_sent_addr);
    uint64_t downstream_data_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x, downstream_enc.d2d.downstream_noc_y, sender_socket.downstream_fifo_addr);

    uint64_t upstream_bytes_acked_noc_addrs[num_sockets_this_risc];
    if constexpr (use_fabric_on_receiver) {
        for (uint32_t i = 0; i < num_sockets_this_risc; i++) {
            upstream_bytes_acked_noc_addrs[i] = get_noc_addr(
                receiver_sockets[i].d2d.upstream_noc_x,
                receiver_sockets[i].d2d.upstream_noc_y,
                receiver_sockets[i].d2d.upstream_bytes_acked_addr);
        }
    }

    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_packet_header = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* upstream_packet_header = nullptr;

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);
    volatile tt_l1_ptr uint32_t* page_ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_ready_sem_addr);
    volatile tt_l1_ptr uint32_t* ncrisc_done_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ncrisc_done_sem_addr);

    if constexpr (use_fabric_on_sender) {
        uint32_t hdr_base =
            get_write_ptr(fabric_packet_header_cb_id) + packet_header_slot_start * sizeof(PACKET_HEADER_TYPE);
        downstream_packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_base);
        downstream_fabric_connection.open();
        fabric_set_unicast_route(downstream_packet_header, downstream_enc);
    }
    if constexpr (use_fabric_on_receiver) {
        uint32_t hdr_base =
            get_write_ptr(fabric_packet_header_cb_id) + (packet_header_slot_start + 1) * sizeof(PACKET_HEADER_TYPE);
        upstream_packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_base);
        upstream_fabric_connection.open();
    }

    bool terminated = false;
    while (!terminated) {
        socket_reserve_pages(sender_socket, 1);

        invalidate_l1_cache();
        if (termination_semaphore[0] == 1) {
            break;
        }

        // sync with NCRISC to start processing sockets
        noc_semaphore_set(page_ready_sem, 1);

        uint64_t dst_addr_base = downstream_data_addr + sender_socket.write_ptr;

        terminated = process_upstream_sockets(
            receiver_sockets,
            downstream_fabric_connection,
            downstream_packet_header,
            dst_addr_base,
            downstream_bytes_sent_noc_addr,
            upstream_fabric_connection,
            upstream_packet_header,
            upstream_bytes_acked_noc_addrs,
            termination_semaphore);

        if (!terminated) {
            noc_semaphore_wait_min(ncrisc_done_sem, 1);
            noc_semaphore_set(ncrisc_done_sem, 0);

            if constexpr (use_fabric_on_sender) {
                socket_push_pages(sender_socket, 1);
            } else {
                socket_push_pages(sender_socket, 1);
                socket_notify_receiver(sender_socket);
            }
        }
    }

    // Ensure NCRISC can observe termination
    noc_semaphore_set(page_ready_sem, 1);

    update_socket_config(sender_socket);
    for (uint32_t i = 0; i < num_sockets_this_risc; i++) {
        update_socket_config(receiver_sockets[i]);
    }

    if constexpr (use_fabric_on_receiver) {
        upstream_fabric_connection.close();
    }
    if constexpr (use_fabric_on_sender) {
        downstream_fabric_connection.close();
    }
}

// ============================================================================
// NCRISC – handles upstream sockets [N/2..N), link 1
// ============================================================================
#elif defined(COMPILE_FOR_NCRISC)
void kernel_main() {
    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender downstream_fabric_connection;
    tt::tt_fabric::WorkerToFabricEdmSender upstream_fabric_connection;

    if constexpr (use_fabric_on_sender) {
        downstream_fabric_connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    }
    if constexpr (use_fabric_on_receiver) {
        upstream_fabric_connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    }

    // Read sender socket config once for static downstream address info
    SocketSenderInterface sender_socket_snapshot = create_sender_socket_interface(sender_socket_config_addr);
    sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket_snapshot, 0);

    SocketReceiverInterface receiver_sockets[num_sockets_this_risc];
    for (uint32_t i = 0; i < num_sockets_this_risc; i++) {
        receiver_sockets[i] = create_receiver_socket_interface(receiver_socket_config_addrs[i]);
        set_receiver_socket_page_size(receiver_sockets[i], upstream_page_size);
    }

    uint64_t downstream_bytes_sent_noc_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x,
        downstream_enc.d2d.downstream_noc_y,
        sender_socket_snapshot.downstream_bytes_sent_addr);
    uint64_t downstream_data_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x,
        downstream_enc.d2d.downstream_noc_y,
        sender_socket_snapshot.downstream_fifo_addr);

    uint64_t upstream_bytes_acked_noc_addrs[num_sockets_this_risc];
    if constexpr (use_fabric_on_receiver) {
        for (uint32_t i = 0; i < num_sockets_this_risc; i++) {
            upstream_bytes_acked_noc_addrs[i] = get_noc_addr(
                receiver_sockets[i].d2d.upstream_noc_x,
                receiver_sockets[i].d2d.upstream_noc_y,
                receiver_sockets[i].d2d.upstream_bytes_acked_addr);
        }
    }

    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_packet_header = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* upstream_packet_header = nullptr;

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);
    volatile tt_l1_ptr uint32_t* page_ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_ready_sem_addr);
    volatile tt_l1_ptr uint32_t* ncrisc_done_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ncrisc_done_sem_addr);

    if constexpr (use_fabric_on_sender) {
        uint32_t hdr_base =
            get_write_ptr(fabric_packet_header_cb_id) + packet_header_slot_start * sizeof(PACKET_HEADER_TYPE);
        downstream_packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_base);
        downstream_fabric_connection.open();
        fabric_set_unicast_route(downstream_packet_header, downstream_enc);
    }
    if constexpr (use_fabric_on_receiver) {
        uint32_t hdr_base =
            get_write_ptr(fabric_packet_header_cb_id) + (packet_header_slot_start + 1) * sizeof(PACKET_HEADER_TYPE);
        upstream_packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_base);
        upstream_fabric_connection.open();
    }

    bool terminated = false;
    while (!terminated) {
        noc_semaphore_wait_min(page_ready_sem, 1);
        noc_semaphore_set(page_ready_sem, 0);

        invalidate_l1_cache();
        if (termination_semaphore[0] == 1) {
            break;
        }

        SocketSenderInterface sender_socket_int = create_sender_socket_interface(sender_socket_config_addr);
        uint64_t dst_addr_base = downstream_data_addr + sender_socket_int.write_ptr;

        terminated = process_upstream_sockets(
            receiver_sockets,
            downstream_fabric_connection,
            downstream_packet_header,
            dst_addr_base,
            downstream_bytes_sent_noc_addr,
            upstream_fabric_connection,
            upstream_packet_header,
            upstream_bytes_acked_noc_addrs,
            termination_semaphore);

        // Signal BRISC that our sockets are done (even on termination path)
        noc_semaphore_set(ncrisc_done_sem, 1);
    }

    // Ensure BRISC doesn't hang waiting for NCRISC
    noc_semaphore_set(ncrisc_done_sem, 1);

    for (uint32_t i = 0; i < num_sockets_this_risc; i++) {
        update_socket_config(receiver_sockets[i]);
    }

    if constexpr (use_fabric_on_receiver) {
        upstream_fabric_connection.close();
    }
    if constexpr (use_fabric_on_sender) {
        downstream_fabric_connection.close();
    }
}
#endif
