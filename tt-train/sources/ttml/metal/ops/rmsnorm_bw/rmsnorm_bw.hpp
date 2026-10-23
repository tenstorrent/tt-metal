// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Returns {dL_dinput, dL_dgamma}. A grad that is not requested is nullopt and its kernel work is skipped; at least one
// must be requested.
std::vector<std::optional<ttnn::Tensor>> rmsnorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,  // intermediate from fw
    const ttnn::Tensor& dL_dout_tensor,
    bool compute_grad_input = true,
    bool compute_grad_gamma = true);

}  // namespace ttml::metal
