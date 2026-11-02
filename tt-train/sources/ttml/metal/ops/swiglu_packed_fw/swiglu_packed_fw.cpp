// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_packed_fw.hpp"

#include "device/swiglu_packed_fw_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor swiglu_packed_fw(
    const ttnn::Tensor& packed,
    const std::optional<ttnn::Tensor>& preallocated_output,
    uint32_t out_cols,
    bool transposed_output,
    bool transposed_input) {
    return ttnn::prim::ttml_swiglu_packed_fw(
        packed, preallocated_output, out_cols, transposed_output, transposed_input);
}

}  // namespace ttml::metal
