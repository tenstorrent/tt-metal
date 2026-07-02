// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
constexpr uint32_t forward_metadata_size_bytes = get_compile_time_arg_val(11);

constexpr uint32_t page_ready_sem_id = get_compile_time_arg_val(12);
constexpr uint32_t ncrisc_done_sem_id = get_compile_time_arg_val(13);
constexpr uint32_t socket_start_idx = get_compile_time_arg_val(14);
constexpr uint32_t packet_header_slot_start = get_compile_time_arg_val(15);

constexpr uint32_t receiver_socket_addrs_start_idx = 16;
constexpr uint32_t downstream_header_ring_size = 2;
constexpr uint32_t downstream_header_slot_count = use_fabric_on_sender ? downstream_header_ring_size : 0;
constexpr uint8_t downstream_stateful_data_cmd_buf = write_reg_cmd_buf;
constexpr uint8_t downstream_stateful_sync_cmd_buf = write_at_cmd_buf;
constexpr uint8_t upstream_dual_stateful_data_cmd_buf = write_cmd_buf;
constexpr uint8_t upstream_dual_stateful_sync_cmd_buf = read_cmd_buf;

struct DownstreamSendState {
    std::array<volatile tt_l1_ptr PACKET_HEADER_TYPE*, downstream_header_ring_size> packet_headers = {};
    uint32_t next_packet_header_idx = 0;
    uint32_t packet_headers_in_use = 0;
    uint32_t cached_free_write_slots = 0;
};

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

FORCE_INLINE void write_data_to_remote_core_with_ack(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    DownstreamSendState& downstream_send_state,
    uint32_t l1_read_addr,
    uint64_t dst_addr,
    uint32_t packet_size) {
    if (downstream_send_state.packet_headers_in_use == downstream_header_ring_size) {
        // Drain once per ring wrap so a header slot is only reused after its prior send has departed.
        noc_async_writes_flushed();
        downstream_send_state.packet_headers_in_use = 0;
    }

    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header_addr =
        downstream_send_state.packet_headers[downstream_send_state.next_packet_header_idx];
    packet_header_addr->set_fused_unicast_write_atomic_inc_write_noc_address(dst_addr);
    packet_header_addr->set_fused_unicast_write_atomic_inc_value(packet_size);
    packet_header_addr->set_payload_size_bytes(static_cast<uint16_t>(packet_size));

    wait_for_cached_free_write_slot(fabric_connection, downstream_send_state.cached_free_write_slots);

    fabric_connection.send_current_slot_stateful_non_blocking(
        l1_read_addr, packet_size, reinterpret_cast<uint32_t>(packet_header_addr));

    downstream_send_state.cached_free_write_slots--;
    if (++downstream_send_state.next_packet_header_idx == downstream_header_ring_size) {
        downstream_send_state.next_packet_header_idx = 0;
    }
    downstream_send_state.packet_headers_in_use++;
}

FORCE_INLINE void send_worker_data_over_fabric(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    DownstreamSendState& downstream_send_state,
    uint32_t l1_read_addr,
    uint64_t dst_addr,
    uint32_t total_size) {
    uint32_t src = l1_read_addr;
    uint64_t dst = dst_addr;
    uint32_t remaining = total_size;
    while (remaining > whole_packet_size) {
        write_data_to_remote_core_with_ack(fabric_connection, downstream_send_state, src, dst, whole_packet_size);
        src += whole_packet_size;
        dst += whole_packet_size;
        remaining -= whole_packet_size;
    }
    write_data_to_remote_core_with_ack(fabric_connection, downstream_send_state, src, dst, remaining);
}

FORCE_INLINE void flush_downstream_fabric_writes(DownstreamSendState& downstream_send_state) {
    if (downstream_send_state.packet_headers_in_use > 0) {
        noc_async_writes_flushed();
        downstream_send_state.packet_headers_in_use = 0;
    }
}

FORCE_INLINE void notify_sender_over_fabric(
    const SocketReceiverInterface& socket,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header_addr,
    uint32_t& cached_free_write_slots,
    bool& flush_pending) {
    packet_header_addr->set_unicast_inline_write_value(socket.bytes_acked);
    wait_for_cached_free_write_slot(fabric_connection, cached_free_write_slots);
    fabric_connection.send_current_slot_stateful_non_blocking_from_address(
        reinterpret_cast<uint32_t>(packet_header_addr), sizeof(PACKET_HEADER_TYPE));
    cached_free_write_slots--;
    flush_pending = true;
}

// Process this RISC's subset of upstream sockets: receive data, forward via
// fabric (or local NOC), pop the upstream socket, and ack the upstream sender.
// Returns true if termination was detected during processing.
FORCE_INLINE bool process_upstream_sockets(
    SocketReceiverInterface* receiver_sockets,
    tt::tt_fabric::WorkerToFabricEdmSender& downstream_fabric_connection,
    DownstreamSendState& downstream_send_state,
    uint64_t dst_addr_base,
    tt::tt_fabric::WorkerToFabricEdmSender& upstream_fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE** upstream_packet_headers,
    volatile tt_l1_ptr uint32_t* termination_semaphore) {
    if constexpr (num_sockets_this_risc == 0) {
        return false;
    }
    uint32_t remaining = num_sockets_this_risc;
    uint32_t worker_idx = 0;
    uint32_t processed_mask = 0;
    [[maybe_unused]] uint32_t upstream_cached_free_write_slots = 0;
    [[maybe_unused]] bool upstream_flush_pending = false;

    auto flush_upstream_notifications = [&]() __attribute__((always_inline)) {
        if constexpr (use_fabric_on_receiver) {
            if (upstream_flush_pending) {
                noc_async_writes_flushed();
                upstream_flush_pending = false;
            }
        }
    };

    while (remaining > 0) {
        invalidate_l1_cache();
        if (termination_semaphore[0] == 1) {
            flush_upstream_notifications();
            return true;
        }
        if (!(processed_mask & (1 << worker_idx)) && socket_wait_for_pages(receiver_sockets[worker_idx], 1, 1)) {
            uint32_t l1_read_addr = receiver_sockets[worker_idx].read_ptr;
            uint64_t dst_addr = dst_addr_base + (socket_start_idx + worker_idx) * upstream_page_size;
            if constexpr (use_fabric_on_sender) {
                send_worker_data_over_fabric(
                    downstream_fabric_connection,
                    downstream_send_state,
                    l1_read_addr,
                    dst_addr,
                    receiver_sockets[worker_idx].page_size);
                // Preserve current downstream-before-upstream ordering even with disjoint stateful cmd-buf pairs.
                flush_downstream_fabric_writes(downstream_send_state);
            } else {
                write_data_to_local_core_with_ack(
                    l1_read_addr, dst_addr, receiver_sockets[worker_idx].page_size);
            }

            socket_pop_pages(receiver_sockets[worker_idx], 1);

            if constexpr (use_fabric_on_receiver) {
                notify_sender_over_fabric(
                    receiver_sockets[worker_idx],
                    upstream_fabric_connection,
                    upstream_packet_headers[worker_idx],
                    upstream_cached_free_write_slots,
                    upstream_flush_pending);
            } else {
                socket_notify_sender(receiver_sockets[worker_idx]);
            }

            processed_mask |= (1 << worker_idx);
            remaining--;
        }

        if constexpr (num_sockets_this_risc > 1) {
            worker_idx = (worker_idx + 1) % num_sockets_this_risc;
        }
    }
    flush_upstream_notifications();
    return false;
}

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
void kernel_main() {
    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender downstream_fabric_connection;
    tt::tt_fabric::WorkerToFabricEdmSender upstream_fabric_connection;

    if constexpr (use_fabric_on_sender) {
        downstream_fabric_connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
        tt::tt_fabric::fabric_detail::set_stateful_cmd_buf_pair(
            downstream_fabric_connection, downstream_stateful_data_cmd_buf, downstream_stateful_sync_cmd_buf);
    }
    if constexpr (use_fabric_on_receiver) {
        upstream_fabric_connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
        if constexpr (use_fabric_on_sender) {
            tt::tt_fabric::fabric_detail::set_stateful_cmd_buf_pair(
                upstream_fabric_connection, upstream_dual_stateful_data_cmd_buf, upstream_dual_stateful_sync_cmd_buf);
        }
    }

    constexpr uint32_t downstream_page_size = page_size;

    SocketSenderInterface sender_socket = create_sender_socket_interface(sender_socket_config_addr);
    set_sender_socket_page_size(sender_socket, downstream_page_size);
    sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);
    constexpr uint32_t last_upstream_page_size = upstream_page_size + forward_metadata_size_bytes;

    SocketReceiverInterface receiver_sockets[num_sockets_this_risc > 0 ? num_sockets_this_risc : 1];
    for (uint32_t i = 0; i < num_sockets_this_risc; i++) {
        receiver_sockets[i] = create_receiver_socket_interface(receiver_socket_config_addrs[i]);
#if defined(COMPILE_FOR_NCRISC)
        const uint32_t rx_page_size = (i == num_sockets_this_risc - 1) ? last_upstream_page_size : upstream_page_size;
#else
        const uint32_t rx_page_size = upstream_page_size;
#endif
        set_receiver_socket_page_size(receiver_sockets[i], rx_page_size);
    }

    uint64_t downstream_bytes_sent_noc_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x,
        downstream_enc.d2d.downstream_noc_y,
        sender_socket.downstream_bytes_sent_addr);
    uint64_t downstream_data_addr = get_noc_addr(
        downstream_enc.d2d.downstream_noc_x, downstream_enc.d2d.downstream_noc_y, sender_socket.downstream_fifo_addr);

    constexpr uint32_t header_slots_per_risc = num_sockets_this_risc > 0 ? num_sockets_this_risc : 1;
    std::array<uint64_t, header_slots_per_risc> upstream_bytes_acked_noc_addrs = {};
    if constexpr (use_fabric_on_receiver) {
        for (uint32_t i = 0; i < num_sockets_this_risc; i++) {
            upstream_bytes_acked_noc_addrs[i] = get_noc_addr(
                receiver_sockets[i].d2d.upstream_noc_x,
                receiver_sockets[i].d2d.upstream_noc_y,
                receiver_sockets[i].d2d.upstream_bytes_acked_addr);
        }
    }

    DownstreamSendState downstream_send_state = {};
    std::array<volatile tt_l1_ptr PACKET_HEADER_TYPE*, header_slots_per_risc> upstream_packet_headers = {};

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);
    volatile tt_l1_ptr uint32_t* page_ready_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(page_ready_sem_id));
    volatile tt_l1_ptr uint32_t* ncrisc_done_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ncrisc_done_sem_id));

    [[maybe_unused]] uint32_t packet_header_cb_base = 0;
    if constexpr (use_fabric_on_sender || use_fabric_on_receiver) {
        packet_header_cb_base = get_write_ptr(fabric_packet_header_cb_id);
    }
    if constexpr (use_fabric_on_sender) {
        downstream_fabric_connection.open();
        uint32_t header_addr = packet_header_cb_base + packet_header_slot_start * sizeof(PACKET_HEADER_TYPE);
        for (uint32_t i = 0; i < downstream_header_ring_size; i++) {
            downstream_send_state.packet_headers[i] =
                reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header_addr);
            fabric_set_unicast_route(downstream_send_state.packet_headers[i], downstream_enc);
            downstream_send_state.packet_headers[i]->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader{
                    downstream_data_addr, downstream_bytes_sent_noc_addr, whole_packet_size, false},
                whole_packet_size);
            header_addr += sizeof(PACKET_HEADER_TYPE);
        }
    }
    if constexpr (use_fabric_on_receiver) {
        upstream_fabric_connection.open();
        uint32_t upstream_header_slot_start = packet_header_slot_start + downstream_header_slot_count;
        uint32_t header_addr = packet_header_cb_base + upstream_header_slot_start * sizeof(PACKET_HEADER_TYPE);
        for (uint32_t i = 0; i < num_sockets_this_risc; i++) {
            upstream_packet_headers[i] = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header_addr);
            fabric_set_unicast_route(upstream_packet_headers[i], receiver_sockets[i]);
            upstream_packet_headers[i]->to_noc_unicast_inline_write(
                NocUnicastInlineWriteCommandHeader{upstream_bytes_acked_noc_addrs[i], receiver_sockets[i].bytes_acked});
            header_addr += sizeof(PACKET_HEADER_TYPE);
        }
    }
    if constexpr (use_fabric_on_sender) {
        downstream_fabric_connection.setup_stateful_send_cmd_bufs();
    }
    if constexpr (use_fabric_on_receiver) {
        upstream_fabric_connection.setup_stateful_send_cmd_bufs();
    }

    bool terminated = false;
    while (!terminated) {
#if defined(COMPILE_FOR_BRISC)
        socket_reserve_pages(sender_socket, 1);
#elif defined(COMPILE_FOR_NCRISC)
        noc_semaphore_wait_min(page_ready_sem, 1);
        noc_semaphore_set(page_ready_sem, 0);
#endif

        invalidate_l1_cache();
        if (termination_semaphore[0] == 1) {
            break;
        }

        uint64_t dst_addr_base;
#if defined(COMPILE_FOR_BRISC)
        noc_semaphore_set(page_ready_sem, 1);
#endif
        dst_addr_base = downstream_data_addr + sender_socket.write_ptr;

        terminated = process_upstream_sockets(
            receiver_sockets,
            downstream_fabric_connection,
            downstream_send_state,
            dst_addr_base,
            upstream_fabric_connection,
            upstream_packet_headers.data(),
            termination_semaphore);

#if defined(COMPILE_FOR_BRISC)
        if (!terminated) {
            noc_semaphore_wait_min(ncrisc_done_sem, 1);
            noc_semaphore_set(ncrisc_done_sem, 0);
            socket_push_pages(sender_socket, 1);
            if constexpr (!use_fabric_on_sender) {
                socket_notify_receiver(sender_socket);
            }
        }
#elif defined(COMPILE_FOR_NCRISC)
        noc_semaphore_set(ncrisc_done_sem, 1);
        // Used to update NCRISC local copy of write_ptr
        // Only brisc will update the actual socket config in l1 with the new write_ptr
        socket_push_pages(sender_socket, 1);
#endif
    }

#if defined(COMPILE_FOR_BRISC)
    noc_semaphore_set(page_ready_sem, 1);
    update_socket_config(sender_socket);
#endif

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
