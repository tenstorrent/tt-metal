// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Sender kernel for the pure-D2D pipeline test.
//
// Reads `data_size` bytes from a DRAM tensor (one page at a time, walked via
// TensorAccessor) into a local L1 CB, then pushes each page into a cross-mesh
// MeshSocket by splitting the page across TWO forward fabric links (matching
// the production d2d_exchange.cpp pattern: num_fwd_links=2, num_bwd_links=1).
// Companion: socket_to_dram_receiver.cpp.

#include <cstdint>
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"

// Write `xfer_size` bytes on a single fabric link, breaking into FABRIC_MAX_PACKET_SIZE chunks.
void fabric_write_any_len(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    uint32_t src_addr,
    uint64_t dst_addr,
    uint32_t xfer_size,
    sender_downstream_encoding& downstream_enc) {
    fabric_set_unicast_route(data_packet_header_addr, downstream_enc);
    while (xfer_size > FABRIC_MAX_PACKET_SIZE) {
        data_packet_header_addr->to_noc_unicast_write(NocUnicastCommandHeader{dst_addr}, FABRIC_MAX_PACKET_SIZE);
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_without_header_non_blocking_from_address(src_addr, FABRIC_MAX_PACKET_SIZE);
        fabric_connection.send_payload_flush_blocking_from_address(
            (uint32_t)data_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
        dst_addr += FABRIC_MAX_PACKET_SIZE;
        src_addr += FABRIC_MAX_PACKET_SIZE;
        xfer_size -= FABRIC_MAX_PACKET_SIZE;
    }
    data_packet_header_addr->to_noc_unicast_write(NocUnicastCommandHeader{dst_addr}, xfer_size);
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_without_header_non_blocking_from_address(src_addr, xfer_size);
    fabric_connection.send_payload_flush_blocking_from_address(
        (uint32_t)data_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
}

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t data_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t page_size = get_compile_time_arg_val(2);
constexpr uint32_t num_pages = get_compile_time_arg_val(3);
// Toggles whether the sender waits for all pages to be acked before exiting.
// See test_dram_to_dram_smoke.py for why this is a per-topology knob: cross-host
// fabric acks have a credit-return gap that hangs the barrier; data path itself
// is unaffected.
constexpr uint32_t wait_for_acks = get_compile_time_arg_val(4);
// TensorAccessor CT args for the DRAM source tensor start here.
constexpr uint32_t input_args_cta_idx = 5;
constexpr uint32_t input_args_crta_idx = 0;

// Each page is split into 2 halves, one half per forward link. Page size must
// be even.
static_assert(page_size % 2 == 0, "page_size must be even — pages are split across 2 forward links");

void kernel_main() {
    size_t rt_args_idx = 0;
    uint32_t input_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);

    // Two forward fabric links (matches production d2d_exchange.cpp pattern).
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection_0 =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection_1 =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    auto input_addr_gen_args = TensorAccessorArgs<input_args_cta_idx, input_args_crta_idx>();
    auto input_addr_gen = TensorAccessor(input_addr_gen_args, input_tensor_addr);

    // Packet header CB layout: [data_link_0 | data_link_1 | socket_notify].
    auto* data_packet_header_addr_0 =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    auto* data_packet_header_addr_1 = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
        get_write_ptr(fabric_packet_header_cb_id) + sizeof(PACKET_HEADER_TYPE));
    auto* socket_packet_header_addr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
        get_write_ptr(fabric_packet_header_cb_id) + 2 * sizeof(PACKET_HEADER_TYPE));

    fabric_connection_0.open();
    fabric_connection_1.open();

    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, page_size);

    constexpr uint32_t half_page = page_size / 2;

    for (uint32_t p = 0; p < num_pages; ++p) {
        // Stage page p of the DRAM source into the L1 CB.
        cb_reserve_back(data_cb_id, 1);
        auto cb_addr = get_write_ptr(data_cb_id);
        auto noc_read_addr = input_addr_gen.get_noc_addr(p);
        noc_async_read<page_size>(noc_read_addr, cb_addr, page_size);
        noc_async_read_barrier();
        cb_push_back(data_cb_id, 1);

        // Push page p across the fabric — first half over link 0, second half
        // over link 1. Receiver writes both halves contiguously into the socket
        // FIFO at the downstream slot.
        socket_reserve_pages(sender_socket, 1);
        uint32_t l1_read_addr = get_read_ptr(data_cb_id);
        for (uint32_t i = 0; i < sender_socket.num_downstreams; ++i) {
            sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, i);
            uint64_t receiver_noc_coord_addr_base = get_noc_addr(
                downstream_enc.d2d.downstream_noc_x,
                downstream_enc.d2d.downstream_noc_y,
                sender_socket.write_ptr + sender_socket.downstream_fifo_addr);
            fabric_write_any_len(
                data_packet_header_addr_0,
                fabric_connection_0,
                l1_read_addr,
                receiver_noc_coord_addr_base,
                half_page,
                downstream_enc);
            fabric_write_any_len(
                data_packet_header_addr_1,
                fabric_connection_1,
                l1_read_addr + half_page,
                receiver_noc_coord_addr_base + half_page,
                half_page,
                downstream_enc);
        }
        socket_push_pages(sender_socket, 1);

        // Socket-notify goes on link 0 (the bwd-direction ack uses its own link
        // on the receiver side, but the fwd-direction page-available notify
        // shares one of the fwd links — same convention as production).
        fabric_socket_notify_receiver(sender_socket, fabric_connection_0, socket_packet_header_addr);

        cb_pop_front(data_cb_id, 1);
    }

    if constexpr (wait_for_acks) {
        socket_barrier(sender_socket);
    }
    update_socket_config(sender_socket);
    fabric_connection_0.close();
    fabric_connection_1.close();
}
