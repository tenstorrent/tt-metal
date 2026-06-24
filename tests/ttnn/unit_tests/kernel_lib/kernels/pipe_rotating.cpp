// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// mcast_pipe helper unit test: ROTATING-ROLE regression kernel.
//
// Two cores ping-pong over a SINGLE shared data_ready cell: each is a sender on some iters and a
// receiver on others (role 0 sends on even iters, role 1 on odd). A core's receiver turn clears the
// shared cell, so its next sender turn must re-assert VALID before broadcasting it — otherwise it
// broadcasts a stale INVALID and the partner's receive() hangs. Sender is always out of its 1x1
// partner rect (no loopback), isolating the flag staleness.
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
    constexpr uint32_t payload_pages = get_compile_time_arg_val(4);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t num_iters = get_compile_time_arg_val(6);
    constexpr auto in_args = TensorAccessorArgs<7>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_start_id = get_arg_val<uint32_t>(1);
    const uint32_t partner_x = get_arg_val<uint32_t>(2);
    const uint32_t partner_y = get_arg_val<uint32_t>(3);
    const uint32_t output_addr = get_arg_val<uint32_t>(4);
    const uint32_t output_start_id = get_arg_val<uint32_t>(5);  // first DRAM slot this core writes
    const uint32_t role = get_arg_val<uint32_t>(6);             // 0: sends on even iters; 1: on odd

    constexpr uint32_t payload_bytes = payload_pages * page_bytes;

    Noc noc;
    CircularBuffer cb_src_obj(cb_src);
    CircularBuffer cb_dst_obj(cb_dst);

    // stage the payload into cb_src (the send source, reused every sender turn)
    const auto in = TensorAccessor(in_args, input_addr);
    cb_src_obj.reserve_back(payload_pages);
    for (uint32_t i = 0; i < payload_pages; ++i) {
        noc.async_read(in, cb_src_obj, page_bytes, {.page_id = input_start_id + i}, {.offset_bytes = i * page_bytes});
    }
    noc.async_read_barrier();
    cb_src_obj.push_back(payload_pages);
    cb_src_obj.wait_front(payload_pages);

    const uint32_t src_addr = cb_src_obj.get_read_ptr();
    cb_dst_obj.reserve_back(payload_pages);  // landing region; write_ptr == base == the mcast target
    const uint32_t dst_addr = cb_dst_obj.get_write_ptr();

    // SenderPipe built ONCE (ctor sets the local data_ready cell VALID). The partner is the 1x1 dest
    // rect; sender is out-of-rect => plain mcast (area 1, excl 1), no loopback. PRE_HANDSHAKE so the
    // ping-pong serializes through the consumer_ready ack — the ack count defaults to the EXCLUDE
    // fan-out (1, the partner), so no explicit ack arg is needed.
    SenderPipe<noc_index, data_ready_sem_id, /*PRE_HANDSHAKE=*/true, consumer_ready_sem_id> send_pipe(
        noc, McastRect<>{partner_x, partner_y, partner_x, partner_y});

    const auto out = TensorAccessor(out_args, output_addr);
    uint32_t recv_slot = output_start_id;  // DRAM page id for the next received block this core writes

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const bool am_sender = (((iter + role) & 1u) == 0u);
        if (am_sender) {
            send_pipe.send(src_addr, dst_addr, payload_bytes);
        } else {
            // per-round ReceiverPipe on the SAME shared data_ready cell; its receive() clears the cell,
            // which is what makes this core's next sender turn need the per-send VALID re-assert.
            ReceiverPipe<data_ready_sem_id, /*PRE_HANDSHAKE=*/true, consumer_ready_sem_id> recv_pipe(noc);
            recv_pipe.receive(partner_x, partner_y);
            // copy the freshly received block out to DRAM for host verification, before the next round
            // can overwrite the landing buffer.
            for (uint32_t i = 0; i < payload_pages; ++i) {
                noc.async_write(
                    cb_dst_obj, out, page_bytes, {.offset_bytes = i * page_bytes}, {.page_id = recv_slot + i});
            }
            noc.async_write_barrier();
            recv_slot += payload_pages;
        }
    }
}
