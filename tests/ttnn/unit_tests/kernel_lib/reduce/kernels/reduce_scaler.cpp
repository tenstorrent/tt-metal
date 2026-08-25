// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {

constexpr uint32_t cb_scaler = 1;

}  // namespace

void kernel_main() {
    const uint32_t valid_elements = get_arg_val<uint32_t>(0);

#ifdef REDUCE_CUSTOM_SCALER_BITS
    constexpr float scaler = __builtin_bit_cast(float, static_cast<uint32_t>(REDUCE_CUSTOM_SCALER_BITS));
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, REDUCE_OP, REDUCE_DIM>(scaler, valid_elements);
#else
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, REDUCE_OP, REDUCE_DIM, REDUCE_FACTOR>(
        valid_elements);
#endif
}
