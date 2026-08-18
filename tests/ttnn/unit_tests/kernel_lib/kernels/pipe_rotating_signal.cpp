// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Rotating control-only multicast regression. Every core sends its round number on its sender round
// and receives on every other round. A rotating Flag sender must clear its local multicast source
// after send_signal(), or its next receive_signal() observes that stale value.
#include <stdint.h>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_result = get_compile_time_arg_val(0);
    constexpr auto mc = McastArgs</*CT=*/1, /*RT=*/2>();
    constexpr auto out_args = TensorAccessorArgs<mc.next_compile_time_args_offset()>();
    constexpr uint32_t span = mc.rotating_span;

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_sender_round = get_arg_val<uint32_t>(1);

    Noc noc;
    auto send_pipe = mc.sender(noc);
    auto receive_pipe = mc.receiver(noc);
    const auto out = TensorAccessor(out_args, output_addr);
    CircularBuffer result_cb(cb_result);

    for (uint32_t round = 0; round < span; ++round) {
        const uint32_t expected = round + 1;
        uint32_t observed = expected;
        if (round == my_sender_round) {
            send_pipe.send_signal(expected);
        } else {
            observed = receive_pipe.receive_signal(round);
        }

        result_cb.reserve_back(1);
        volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_cb.get_write_ptr());
        for (uint32_t i = 0; i < 8; ++i) {
            result[i] = observed;
        }
        result_cb.push_back(1);
        result_cb.wait_front(1);
        noc.async_write(result_cb, out, /*size=*/32, {}, {.page_id = my_sender_round * span + round});
        noc.async_write_barrier();
        result_cb.pop_front(1);
    }
}
