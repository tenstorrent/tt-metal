// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Control-only Counter coverage: consume monotone rounds through receive_signal() and publish the
// final returned round for host verification.
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
    constexpr uint32_t cb_result = get_compile_time_arg_val(0);
    constexpr auto mc = McastArgs</*CT=*/1, /*RT=*/2>();
    constexpr uint32_t SCALARS = mc.next_compile_time_args_offset();
    constexpr uint32_t num_iters = get_compile_time_arg_val(SCALARS);
    constexpr uint32_t control_value = get_compile_time_arg_val(SCALARS + 1);
    constexpr auto out_args = TensorAccessorArgs<SCALARS + 2>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_page_id = get_arg_val<uint32_t>(1);

    Noc noc;
    auto pipe = mc.receiver(noc);
    uint32_t final_round = 0;
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        final_round = pipe.receive_signal();
        if constexpr (control_value == INVALID) {
            ASSERT(final_round == iter + 1);
        } else {
            ASSERT(final_round == control_value);
        }
    }

    CircularBuffer result_cb(cb_result);
    result_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_cb.get_write_ptr());
    for (uint32_t i = 0; i < 8; ++i) {
        result[i] = final_round;
    }
    result_cb.push_back(1);
    result_cb.wait_front(1);

    const auto out = TensorAccessor(out_args, output_addr);
    noc.async_write(result_cb, out, /*size=*/32, {}, {.page_id = output_page_id});
    noc.async_write_barrier();
    result_cb.pop_front(1);
}
