// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// mcast_pipe + mcast_host fixed-sender LINE test kernel, LOOPBACK variant.
//
// pipe_fixed_line.cpp multicasts in place (src == dst), so SenderPipe's loopback path is never
// taken: the sender already holds the payload at the destination address and needs no self-copy.
// This kernel stages into a SEPARATE source CB and multicasts into the landing CB, so src != dst
// and a sender that lies inside its own destination rectangle MUST take the loopback path to end
// up with the payload in its own landing CB.
//
// Every core -- sender included -- writes its landing CB to its own DRAM slot, so the sender's slot
// is only correct if the loopback self-copy happened. That makes this the regression test for the
// destination rectangle a fixed Mcast1D sender emits: a rectangle that excludes the sender leaves
// its landing CB stale and its slot wrong.
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
    constexpr uint32_t cb_src = get_compile_time_arg_val(0);  // staging area (sender only)
    constexpr uint32_t cb_dst = get_compile_time_arg_val(1);  // mcast landing region (every core)
    constexpr auto mc = McastArgs</*CT=*/2, /*RT=*/5>();      // mcast config (CT 2..) + per-core coords (RT 5..)
    constexpr uint32_t SCALARS = mc.next_compile_time_args_offset();
    constexpr uint32_t num_blocks = get_compile_time_arg_val(SCALARS + 0);
    constexpr uint32_t payload_pages = get_compile_time_arg_val(SCALARS + 1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(SCALARS + 2);
    constexpr auto in_args = TensorAccessorArgs<SCALARS + 3>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_start_id = get_arg_val<uint32_t>(1);  // this line's first block (sender only)
    const uint32_t output_addr = get_arg_val<uint32_t>(2);
    const uint32_t output_start_id = get_arg_val<uint32_t>(3);  // this core's first DRAM slot
    const uint32_t is_sender = get_arg_val<uint32_t>(4);        // host's mc.is_sender(core)

    constexpr uint32_t payload_bytes = payload_pages * page_bytes;

    Noc noc;
    CircularBuffer cb_src_obj(cb_src);
    CircularBuffer cb_dst_obj(cb_dst);
    const auto in = TensorAccessor(in_args, input_addr);
    const auto out = TensorAccessor(out_args, output_addr);

    // The landing address is the CB base, identical on every core of the line -- that is what makes
    // one multicast write land at the same offset everywhere.
    cb_dst_obj.reserve_back(payload_pages);
    const uint32_t dst_addr = cb_dst_obj.get_write_ptr();

    if (is_sender) {
        cb_src_obj.reserve_back(payload_pages);
        const uint32_t src_addr = cb_src_obj.get_write_ptr();
        auto pipe = mc.sender(noc);
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            for (uint32_t i = 0; i < payload_pages; ++i) {
                noc.async_read(
                    in,
                    cb_src_obj,
                    page_bytes,
                    {.page_id = input_start_id + blk * payload_pages + i},
                    {.offset_bytes = i * page_bytes});
            }
            noc.async_read_barrier();
            if constexpr (mc.active) {
                // src != dst: an in-rect sender takes the loopback path and lands its own copy.
                pipe.send(src_addr, dst_addr, payload_bytes);
            }
            for (uint32_t i = 0; i < payload_pages; ++i) {
                noc.async_write(
                    cb_dst_obj,
                    out,
                    page_bytes,
                    {.offset_bytes = i * page_bytes},
                    {.page_id = output_start_id + blk * payload_pages + i});
            }
            noc.async_write_barrier();
        }
    } else {
        auto pipe = mc.receiver(noc);
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            pipe.receive();
            for (uint32_t i = 0; i < payload_pages; ++i) {
                noc.async_write(
                    cb_dst_obj,
                    out,
                    page_bytes,
                    {.offset_bytes = i * page_bytes},
                    {.page_id = output_start_id + blk * payload_pages + i});
            }
            noc.async_write_barrier();
        }
    }
}
