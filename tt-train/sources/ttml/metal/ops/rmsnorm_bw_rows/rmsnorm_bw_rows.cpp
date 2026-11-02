// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_rows.hpp"

#include "device/rmsnorm_bw_rows_device_operation.hpp"

namespace ttml::metal {

std::vector<ttnn::Tensor> rmsnorm_bw_rows(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gamma,
    const ttnn::Tensor& rms,
    const ttnn::Tensor& dL_dout,
    uint32_t num_slices,
    uint32_t slice_tiles,
    const std::optional<ttnn::Tensor>& residual_grad,
    bool transposed_grad) {
    return ttnn::prim::ttml_rmsnorm_bw_rows(
        input, gamma, rms, dL_dout, num_slices, slice_tiles, residual_grad, transposed_grad);
}

}  // namespace ttml::metal
