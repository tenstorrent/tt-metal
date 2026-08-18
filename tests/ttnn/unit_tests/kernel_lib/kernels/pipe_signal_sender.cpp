// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Control-only Counter coverage: issue every signal without a receiver acknowledgement between
// rounds. A receiver may therefore observe several accumulated increments before its next wait.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr auto mc = McastArgs</*CT=*/0, /*RT=*/0>();
    constexpr uint32_t num_iters = get_compile_time_arg_val(mc.next_compile_time_args_offset());
    constexpr uint32_t control_value = get_compile_time_arg_val(mc.next_compile_time_args_offset() + 1);

    Noc noc;
    auto pipe = mc.sender(noc);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        if constexpr (control_value == INVALID) {
            pipe.send_signal();
        } else {
            pipe.send_signal(control_value);
        }
    }
}
