// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// mcast_pipe helper unit test: RECEIVER kernel. Decodes the ttnn.Mcast2D wire with McastArgs and drives
// ReceiverPipe::receive() for `num_iters` rounds, then writes the received payload out for verification.
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
    constexpr auto mc = McastArgs</*CT=*/1, /*RT=*/2>();              // mcast config (CT 1..) + sender coords (RT 2..)
    constexpr uint32_t SCALARS = mc.next_compile_time_args_offset();  // = 6, right after the mcast CT block
    constexpr uint32_t payload_pages = get_compile_time_arg_val(SCALARS + 0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(SCALARS + 1);
    constexpr uint32_t num_iters = get_compile_time_arg_val(SCALARS + 2);
    constexpr auto out_args = TensorAccessorArgs<SCALARS + 3>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_start_id = get_arg_val<uint32_t>(1);
    // RT 2,3 = the sender's coords (the ack target), consumed by mc.receiver().

    Noc noc;
    CircularBuffer cb_dst_obj(cb_dst);

    // reserve the landing region: write_ptr == base == the address the sender mcasts to
    cb_dst_obj.reserve_back(payload_pages);

    auto pipe = mc.receiver(noc);

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
