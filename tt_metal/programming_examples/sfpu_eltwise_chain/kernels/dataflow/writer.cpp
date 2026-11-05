// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include <cstdint>

void kernel_main() {
    // Dataflow buffers
    DataflowBuffer result_dfb(dfb::result);

    // Input data config
    // The output tensor is bound to this kernel by name in the host code.
    const auto interleaved_accessor = TensorAccessor(tensor::output);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Save output data
    result_dfb.wait_front(one_tile);
    const uint32_t l1_read_addr = result_dfb.get_read_ptr();
    noc_async_write_page(0, interleaved_accessor, l1_read_addr);
    noc_async_write_barrier();
    result_dfb.pop_front(one_tile);
}
