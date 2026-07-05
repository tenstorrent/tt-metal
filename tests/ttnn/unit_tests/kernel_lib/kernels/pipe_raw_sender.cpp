// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// mcast_pipe helper unit test: RAW SENDER kernel — SenderPipe constructed BY HAND.
//
// No host helper (Mcast1D/Mcast2D) and no McastArgs decoder: this kernel reads its own CT/RT and
// spells the SenderPipe template + McastRect itself, so the pipe primitive is exercised in isolation
// from the host arg-emitter. (Contrast pipe_sender.cpp, which decodes the Mcast2D wire via McastArgs.)
// The fence is baked in to flush; linking is always on; the data-ready signal is Flag. The sender is
// out-of-rect, so the pipe infers a plain (no-loopback) mcast.
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
    constexpr uint32_t cb_src = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dst = get_compile_time_arg_val(1);
    constexpr uint32_t data_ready_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t consumer_ready_sem_id = get_compile_time_arg_val(3);
    // The mcast fan-out is derived from the rect area; this slot carries the consumer-ack count.
    // ACK_EQUALS_FANOUT (0xFFFFFFFF) means "ack == the EXCLUDE fan-out" (the dense default).
    constexpr uint32_t consumer_ack_count = get_compile_time_arg_val(4);
    constexpr uint32_t payload_pages = get_compile_time_arg_val(5);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t num_iters = get_compile_time_arg_val(7);
    constexpr uint32_t pre_handshake = get_compile_time_arg_val(8);
    constexpr auto in_args = TensorAccessorArgs<9>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_start_id = get_arg_val<uint32_t>(1);
    const uint32_t x0 = get_arg_val<uint32_t>(2);
    const uint32_t y0 = get_arg_val<uint32_t>(3);
    const uint32_t x1 = get_arg_val<uint32_t>(4);
    const uint32_t y1 = get_arg_val<uint32_t>(5);

    constexpr uint32_t payload_bytes = payload_pages * page_bytes;

    Noc noc;
    CircularBuffer cb_src_obj(cb_src);
    CircularBuffer cb_dst_obj(cb_dst);

    // stage the payload into cb_src (read from DRAM input)
    const auto in = TensorAccessor(in_args, input_addr);
    cb_src_obj.reserve_back(payload_pages);
    for (uint32_t i = 0; i < payload_pages; ++i) {
        noc.async_read(in, cb_src_obj, page_bytes, {.page_id = input_start_id + i}, {.offset_bytes = i * page_bytes});
    }
    noc.async_read_barrier();
    cb_src_obj.push_back(payload_pages);
    cb_src_obj.wait_front(payload_pages);

    const uint32_t src_addr = cb_src_obj.get_read_ptr();
    const uint32_t dst_addr = cb_dst_obj.get_write_ptr();

    // BY HAND: compile-time, core-uniform values (noc id + sem ids + pre_handshake + signal) are
    // template params; the receiver rectangle (its area gives the fan-out) and the consumer-ack count
    // are runtime ctor args. Arg order: NOC_ID, data-ready id, PRE_HANDSHAKE gate, consumer-ready id
    // (used iff PRE_HANDSHAKE), then the signal.
    SenderPipe<noc_index, data_ready_sem_id, pre_handshake != 0, consumer_ready_sem_id, DataReadySignal::Flag> pipe(
        noc, McastRect<>{x0, y0, x1, y1}, consumer_ack_count);

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        pipe.send(src_addr, dst_addr, payload_bytes);
    }
}
