// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// mcast_pipe helper unit test: F3 SENDER-IN-RECT kernel driving Pipe with INCLUDE_SRC
// (the bake-off winner for sender-in-rect loopback). The sender is one core of the mcast
// rectangle and ends up with the payload in its OWN cb_dst via hardware loopback, then
// writes that to its own output shard for verification. The other rect cores run
// pipe_receiver.cpp. Also exercises the degenerate guard when rect_len==1 (area==1, excl==0).
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
    // Handshake is off here (McastConfig(handshake=false)), so the sender broadcasts without the ack.
    constexpr auto mc = McastArgs</*CT=*/2, /*RT=*/2>();              // mcast config (CT 2..) + dest rect (RT 2..)
    constexpr uint32_t SCALARS = mc.next_compile_time_args_offset();  // = 7, right after the mcast CT block
    constexpr uint32_t payload_pages = get_compile_time_arg_val(SCALARS + 0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(SCALARS + 1);
    constexpr uint32_t num_iters = get_compile_time_arg_val(SCALARS + 2);
    constexpr auto in_args = TensorAccessorArgs<SCALARS + 3>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_start_id = get_arg_val<uint32_t>(1);
    // RT 2..5 = the dest rect INCLUDING this sender's own core (loopback); a 1x1 self-rect (R==1) is the
    // degenerate case the SenderPipe collapses to a local copy. The sender's own output words follow the
    // 4-word rect (RT 6,7 = mc.next_runtime_args_offset()+0/+1).
    const uint32_t output_addr = get_arg_val<uint32_t>(mc.next_runtime_args_offset() + 0);
    const uint32_t self_start_id = get_arg_val<uint32_t>(mc.next_runtime_args_offset() + 1);

    constexpr uint32_t payload_bytes = payload_pages * page_bytes;

    Noc noc;
    CircularBuffer cb_src_obj(cb_src);
    CircularBuffer cb_dst_obj(cb_dst);

    // stage payload into cb_src from DRAM
    const auto in = TensorAccessor(in_args, input_addr);
    cb_src_obj.reserve_back(payload_pages);
    for (uint32_t i = 0; i < payload_pages; ++i) {
        noc.async_read(in, cb_src_obj, page_bytes, {.page_id = input_start_id + i}, {.offset_bytes = i * page_bytes});
    }
    noc.async_read_barrier();
    cb_src_obj.push_back(payload_pages);
    cb_src_obj.wait_front(payload_pages);

    cb_dst_obj.reserve_back(payload_pages);
    const uint32_t dst_addr = cb_dst_obj.get_write_ptr();
    const uint32_t src_addr = cb_src_obj.get_read_ptr();

    // Sender is IN the rect -> the pipe infers loopback (hardware writes the payload to self too), and a
    // 1x1 self-only box (R==1) collapses to a local copy (the degenerate guard).
    auto pipe = mc.sender(noc);

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        pipe.send(src_addr, dst_addr, payload_bytes);
    }

    cb_dst_obj.push_back(payload_pages);

    // write our own received payload out to DRAM (sender's shard) for verification
    const auto out = TensorAccessor(out_args, output_addr);
    for (uint32_t i = 0; i < payload_pages; ++i) {
        cb_dst_obj.wait_front(1);
        noc.async_write(cb_dst_obj, out, page_bytes, {}, {.page_id = self_start_id + i});
        noc.async_writes_flushed();
        cb_dst_obj.pop_front(1);
    }
    noc.async_write_barrier();
}
