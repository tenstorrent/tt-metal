// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// mcast_pipe helper unit test: RECEIVER kernel driving dataflow_kernel_lib::ReceiverPipe.
// Ported from bakeoff_mcast_receiver.cpp. The ReceiverPipe takes only the noc + (template) sem
// ids; the sender's coords (the target of the consumer-ready ack) are passed to receive().
//   STAGING_COUNTER(0/1) -> DataReadySignal::Flag (wait+reset) | DataReadySignal::Counter (wait_min)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

#ifndef STAGING_COUNTER
#define STAGING_COUNTER 0
#endif

using namespace dataflow_kernel_lib;

constexpr DataReadySignal STG = STAGING_COUNTER ? DataReadySignal::Counter : DataReadySignal::Flag;

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

    // The receiver takes no rectangle / count — just the noc; the sem ids are template params.
    // The sender's coords (target of the consumer-ready ack) are passed to receive().
    ReceiverPipe<data_ready_sem_id, pre_handshake != 0, consumer_ready_sem_id, STG> pipe(noc);

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        pipe.receive(sender_x, sender_y);
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
