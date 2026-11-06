// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tile_constants.h"
#include "api/numeric/bfloat16.h"
#include "experimental/kernel_args.h"

#include <cstdint>
#include <cstring>

void kernel_main() {
    // Dataflow buffers
    DataflowBuffer src_dfb(dfb::src);
    DataflowBuffer ones_dfb(dfb::ones);

    // Input data config
    // The input tensor is bound to this kernel by name in the host code.
    const auto interleaved_accessor = TensorAccessor(tensor::input);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Read input value data
    src_dfb.reserve_back(one_tile);
    const uint32_t l1_write_addr = src_dfb.get_write_ptr();
    noc_async_read_page(0, interleaved_accessor, l1_write_addr);
    noc_async_read_barrier();
    src_dfb.push_back(one_tile);

    // Create tile with ones
    ones_dfb.reserve_back(one_tile);
    const uint32_t ones_l1_write_addr = ones_dfb.get_write_ptr();
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ones_l1_write_addr);
    for (uint32_t i = 0; i < tt::constants::TILE_HW; i++) {
        ptr[i] = fp32_to_bf16_truncate(1.0f);
    }
    ones_dfb.push_back(one_tile);
}
