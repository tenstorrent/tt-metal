// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// mcast_pipe + mcast_host END-TO-END fixed-sender LINE test kernel.
//
// The fixed-mode counterpart of pipe_rotating_line.cpp. Every core on the grid runs this ONE kernel
// and decodes the host::Mcast1D (fixed edge sender) wire with McastArgs.
//
// This is the 2D dual-mcast matmul in0/in1 shape: one edge sender per line broadcasts to the rest.
// The sender streams `num_blocks` blocks of its line (the K-block loop of a matmul), each staged from
// DRAM and multicast; every receiver receives each block. The dest rect the helper emits EXCLUDES the
// sender, so send() is a plain EXCLUDE_SRC broadcast (no loopback).
//
// PIPE LIFETIME. A core's role is FIXED for the whole block loop (sender iff grid col 0), so the pipe
// is built ONCE above the loop and reused every block, rather than reconstructed per iteration.
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
    constexpr uint32_t cb = get_compile_time_arg_val(0);  // mcast source (in place) + landing region
    constexpr auto mc = McastArgs</*CT=*/1, /*RT=*/5>();  // mcast config (CT 1..) + per-core coords (RT 5..)
    constexpr uint32_t SCALARS = mc.next_compile_time_args_offset();  // = 6, right after the mcast CT block
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
    CircularBuffer cb_obj(cb);
    const auto in = TensorAccessor(in_args, input_addr);
    const auto out = TensorAccessor(out_args, output_addr);

    cb_obj.reserve_back(payload_pages);  // write_ptr == base == the mcast address (same on every core)
    const uint32_t cb_addr = cb_obj.get_write_ptr();

    if (is_sender) {
        // SENDER — built ONCE above the block loop. Its rect EXCLUDES the sender, so send() is a plain
        // EXCLUDE_SRC broadcast; an inactive single-line family (no receivers) skips send() below.
        auto pipe = mc.sender(noc);
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            for (uint32_t i = 0; i < payload_pages; ++i) {
                noc.async_read(
                    in,
                    cb_obj,
                    page_bytes,
                    {.page_id = input_start_id + blk * payload_pages + i},
                    {.offset_bytes = i * page_bytes});
            }
            noc.async_read_barrier();
            if constexpr (mc.active) {
                pipe.send(cb_addr, cb_addr, payload_bytes);
            }
            for (uint32_t i = 0; i < payload_pages; ++i) {
                noc.async_write(
                    cb_obj,
                    out,
                    page_bytes,
                    {.offset_bytes = i * page_bytes},
                    {.page_id = output_start_id + blk * payload_pages + i});
            }
            noc.async_write_barrier();
        }
    } else {
        // RECEIVER — built ONCE above the block loop and reused every block (no per-block reconstruction).
        auto pipe = mc.receiver(noc);
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            pipe.receive();
            for (uint32_t i = 0; i < payload_pages; ++i) {
                noc.async_write(
                    cb_obj,
                    out,
                    page_bytes,
                    {.offset_bytes = i * page_bytes},
                    {.page_id = output_start_id + blk * payload_pages + i});
            }
            noc.async_write_barrier();
        }
    }
}
