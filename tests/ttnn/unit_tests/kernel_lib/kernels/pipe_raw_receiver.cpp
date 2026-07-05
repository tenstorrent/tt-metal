// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// mcast_pipe helper unit test: RAW RECEIVER kernel — ReceiverPipe constructed BY HAND.
//
// No host helper (Mcast1D/Mcast2D) and no McastArgs decoder: this kernel reads its own CT/RT, hands
// the sender coords to the ReceiverPipe ctor (which keeps them), then calls receive() per round.
// (Contrast pipe_receiver.cpp, which decodes the Mcast2D wire via McastArgs.)
// The data-ready signal is Flag: receive() waits for VALID, then resets the flag each round.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_dst = get_compile_time_arg_val(0);
    constexpr uint32_t data_ready_sem_id = get_compile_time_arg_val(1);
    constexpr uint32_t consumer_ready_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t payload_pages = get_compile_time_arg_val(3);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t num_iters = get_compile_time_arg_val(5);
    constexpr uint32_t pre_handshake = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_start_id = get_arg_val<uint32_t>(1);
    const uint32_t sender_x = get_arg_val<uint32_t>(2);
    const uint32_t sender_y = get_arg_val<uint32_t>(3);

    Noc noc;
    CircularBuffer cb_dst_obj(cb_dst);

    // reserve the landing region: write_ptr == base == the address the sender mcasts to
    cb_dst_obj.reserve_back(payload_pages);

    // BY HAND: sem ids + pre_handshake + signal are template params; the sender coords (target of this
    // receiver's consumer-ready ack) go to the ReceiverPipe ctor, which copies + keeps them (NUM_SENDERS
    // defaults to 1). receive() then acks/waits the stored sender each round — no coords passed per call.
    const uint32_t sender_coords[2] = {sender_x, sender_y};
    ReceiverPipe<data_ready_sem_id, pre_handshake != 0, consumer_ready_sem_id, DataReadySignal::Flag> pipe(
        noc, sender_coords);

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        pipe.receive();
    }

    cb_dst_obj.push_back(payload_pages);

    // write the received payload out to DRAM for host verification
    const auto out = TensorAccessor(out_args, output_addr);
    for (uint32_t i = 0; i < payload_pages; ++i) {
        cb_dst_obj.wait_front(1);
        noc.async_write(cb_dst_obj, out, page_bytes, {}, {.page_id = output_start_id + i});
        noc.async_writes_flushed();
        cb_dst_obj.pop_front(1);
    }
    noc.async_write_barrier();
}
