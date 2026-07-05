// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// mcast_pipe helper unit test: SENDER kernel. The host emits the mcast wire via ttnn.Mcast2D; this
// kernel decodes it with McastArgs and drives SenderPipe::send() for `num_iters` rounds. The sender is
// out-of-rect here, so the broadcast is a plain (no-loopback) mcast to the receiver rect.
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
    constexpr auto mc = McastArgs</*CT=*/2, /*RT=*/2>();              // mcast config (CT 2..) + dest rect (RT 2..)
    constexpr uint32_t SCALARS = mc.next_compile_time_args_offset();  // = 7, right after the mcast CT block
    constexpr uint32_t payload_pages = get_compile_time_arg_val(SCALARS + 0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(SCALARS + 1);
    constexpr uint32_t num_iters = get_compile_time_arg_val(SCALARS + 2);
    constexpr auto in_args = TensorAccessorArgs<SCALARS + 3>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_start_id = get_arg_val<uint32_t>(1);
    // RT 2..5 = the dest rect (virtual, NOC-ordered), consumed by mc.sender().

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

    auto pipe = mc.sender(noc);

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        pipe.send(src_addr, dst_addr, payload_bytes);
    }
}
