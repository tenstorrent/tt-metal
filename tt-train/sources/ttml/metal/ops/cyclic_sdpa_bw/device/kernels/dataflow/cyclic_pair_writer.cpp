// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Results of one block pair, from L1 back to DRAM: the stage's probe
// intermediate, and at the final stage the three gradients.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

#ifndef COMPUTE_STAGE
#define COMPUTE_STAGE 5
#endif

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t probe_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_query_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_key_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_value_addr = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t qWt = get_compile_time_arg_val(0);
    constexpr uint32_t vWt = get_compile_time_arg_val(1);
    constexpr auto probe_args = TensorAccessorArgs<2>();
    constexpr auto grad_query_args = TensorAccessorArgs<probe_args.next_compile_time_args_offset()>();
    constexpr auto grad_key_args = TensorAccessorArgs<grad_query_args.next_compile_time_args_offset()>();
    constexpr auto grad_value_args = TensorAccessorArgs<grad_key_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_probe = tt::CBIndex::c_15;
    constexpr uint32_t cb_grad_query = tt::CBIndex::c_16;
    constexpr uint32_t cb_grad_key = tt::CBIndex::c_17;
    constexpr uint32_t cb_grad_value = tt::CBIndex::c_18;

    const uint32_t probe_bytes = get_tile_size(cb_probe);
    const uint32_t grad_bytes = get_tile_size(cb_grad_query);

    const auto probe = TensorAccessor(probe_args, probe_addr, probe_bytes);
    write_tiles_by_row(cb_probe, probe, 0, 1, probe_bytes, 1);

#if COMPUTE_STAGE >= 5
    const auto grad_query = TensorAccessor(grad_query_args, grad_query_addr, grad_bytes);
    const auto grad_key = TensorAccessor(grad_key_args, grad_key_addr, grad_bytes);
    const auto grad_value = TensorAccessor(grad_value_args, grad_value_addr, grad_bytes);
    write_tiles_by_row(cb_grad_query, grad_query, 0, qWt, grad_bytes, qWt);
    write_tiles_by_row(cb_grad_value, grad_value, 0, vWt, grad_bytes, vWt);
    write_tiles_by_row(cb_grad_key, grad_key, 0, qWt, grad_bytes, qWt);
#endif
}
