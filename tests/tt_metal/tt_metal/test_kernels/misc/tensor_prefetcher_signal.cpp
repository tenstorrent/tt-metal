// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/tensor_prefetcher_signal.h"

// Raises the Tensor prefetcher op signal at `signal_addr` `num_signals` times on NoC `signal_noc`.
void kernel_main() {
    constexpr uint8_t signal_noc = get_compile_time_arg_val(0);
    const uint32_t signal_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_signals = get_arg_val<uint32_t>(1);
    for (uint32_t i = 0; i < num_signals; ++i) {
        experimental::tensor_prefetcher_signal<signal_noc>(signal_addr);
    }
    noc_async_atomic_barrier(signal_noc);
}
