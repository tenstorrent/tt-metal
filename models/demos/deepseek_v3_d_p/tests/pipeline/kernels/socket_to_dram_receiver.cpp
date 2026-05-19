// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Receiver kernel for the pure-D2D pipeline test.
//
// For each of `num_pages` pages: wait for one page on the socket, `noc_async_write`
// it from the socket FIFO straight into the DRAM destination tensor (walked via
// TensorAccessor), pop the page, and ack the sender over the backward fabric link.
//
// Modeled on tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_receiver_worker.cpp,
// changed to write to DRAM via TensorAccessor (so the destination lives in DRAM,
// not L1).

#include <cstdint>
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t page_size = get_compile_time_arg_val(1);
constexpr uint32_t num_pages = get_compile_time_arg_val(2);
// TensorAccessor CT args for the DRAM destination tensor start here.
constexpr uint32_t output_args_cta_idx = 3;
constexpr uint32_t output_args_crta_idx = 0;

void kernel_main() {
    size_t rt_args_idx = 0;
    uint32_t recv_socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);

    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    auto output_addr_gen_args = TensorAccessorArgs<output_args_cta_idx, output_args_crta_idx>();
    auto output_addr_gen = TensorAccessor(output_addr_gen_args, output_tensor_addr);

    fabric_connection.open_start();
    auto* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));

    SocketReceiverInterface recv_socket = create_receiver_socket_interface(recv_socket_config_addr);
    set_receiver_socket_page_size(recv_socket, page_size);

    fabric_connection.open_finish();

    for (uint32_t p = 0; p < num_pages; ++p) {
        socket_wait_for_pages(recv_socket, 1);
        uint64_t dst_noc_addr = output_addr_gen.get_noc_addr(p);
        noc_async_write<page_size>(recv_socket.read_ptr, dst_noc_addr, page_size);
        noc_async_write_barrier();
        socket_pop_pages(recv_socket, 1);
        fabric_socket_notify_sender(recv_socket, fabric_connection, socket_packet_header_addr);
    }

    update_socket_config(recv_socket);
    fabric_connection.close();
}
