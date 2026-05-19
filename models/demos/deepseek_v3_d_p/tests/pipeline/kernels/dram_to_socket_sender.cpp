// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Sender kernel for the pure-D2D pipeline test.
//
// Reads `data_size` bytes from a DRAM tensor (one page at a time, walked via
// TensorAccessor) into a local L1 CB, then pushes each page into a cross-mesh
// MeshSocket via single-link fabric writes. Companion: socket_to_dram_receiver.cpp.
//
// Modeled on tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_sender.cpp,
// extended to walk DRAM pages via TensorAccessor (so the source lives in DRAM,
// not L1).

#include <cstdint>
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"

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
// TensorAccessor CT args for the DRAM source tensor start here.
constexpr uint32_t input_args_cta_idx = 4;
constexpr uint32_t input_args_crta_idx = 0;

void kernel_main() {
    size_t rt_args_idx = 0;
    uint32_t input_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);

    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    auto input_addr_gen_args = TensorAccessorArgs<input_args_cta_idx, input_args_crta_idx>();
    auto input_addr_gen = TensorAccessor(input_addr_gen_args, input_tensor_addr);

    auto* data_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    auto* socket_packet_header_addr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
        get_write_ptr(fabric_packet_header_cb_id) + sizeof(PACKET_HEADER_TYPE));

    fabric_connection.open();

    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, page_size);

    for (uint32_t p = 0; p < num_pages; ++p) {
        // Stage page p of the DRAM source into the L1 CB.
        cb_reserve_back(data_cb_id, 1);
        auto cb_addr = get_write_ptr(data_cb_id);
        auto noc_read_addr = input_addr_gen.get_noc_addr(p);
        noc_async_read<page_size>(noc_read_addr, cb_addr, page_size);
        noc_async_read_barrier();
        cb_push_back(data_cb_id, 1);

        // Push page p across the fabric to the downstream socket FIFO.
        socket_reserve_pages(sender_socket, 1);
        uint32_t l1_read_addr = get_read_ptr(data_cb_id);
        for (uint32_t i = 0; i < sender_socket.num_downstreams; ++i) {
            sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, i);
            uint64_t receiver_noc_coord_addr = get_noc_addr(
                downstream_enc.d2d.downstream_noc_x,
                downstream_enc.d2d.downstream_noc_y,
                sender_socket.write_ptr + sender_socket.downstream_fifo_addr);
            fabric_write_any_len(
                data_packet_header_addr,
                fabric_connection,
                l1_read_addr,
                receiver_noc_coord_addr,
                page_size,
                downstream_enc);
        }
        socket_push_pages(sender_socket, 1);

        fabric_socket_notify_receiver(sender_socket, fabric_connection, socket_packet_header_addr);

        cb_pop_front(data_cb_id, 1);
    }

    socket_barrier(sender_socket);
    update_socket_config(sender_socket);
    fabric_connection.close();
}
