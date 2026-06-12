// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// mcast_pipe helper unit test: SENDER kernel driving dataflow_kernel_lib::SenderPipe.
// Ported from bakeoff_mcast_sender.cpp — same program shape, but the mcast+handshake
// block is now SenderPipe::send(). Style axis selected at compile time:
//   STAGING_COUNTER (0/1) -> Staging::Flag | Staging::Counter
// (F1 fence is BAKED IN to flush; F4 linking is ALWAYS on — the helper has no barrier/link knob.)
// EXCLUDE_SRC vs INCLUDE_SRC is NOT a knob: the Pipe infers it at runtime from whether this
// sender's core lies in the rect. Here the sender is out-of-rect, so the Pipe infers EXCLUDE.
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

constexpr Staging STG = STAGING_COUNTER ? Staging::Counter : Staging::Flag;

void kernel_main() {
    constexpr uint32_t cb_src = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dst = get_compile_time_arg_val(1);
    constexpr uint32_t data_ready_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t consumed_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t num_active_cores = get_compile_time_arg_val(4);
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

    // Compile-time, core-uniform values (count + sem ids + staging/pre_handshake) are template
    // params; the only runtime ctor input is the receiver rectangle.
    SenderPipe<num_active_cores, data_ready_sem_id, consumed_sem_id, STG, pre_handshake != 0> pipe(
        noc, McastRect{x0, y0, x1, y1});

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        pipe.send(src_addr, dst_addr, payload_bytes);
    }
}
