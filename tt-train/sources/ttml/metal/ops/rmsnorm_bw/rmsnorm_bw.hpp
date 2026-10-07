// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// RMSNorm backward. Returns {dL/dinput, dL/dgamma}; dL/dgamma is nullopt when compute_dgamma is false
// (frozen gamma), which also skips the dgamma-components pass and its reduction.
std::vector<std::optional<ttnn::Tensor>> rmsnorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,  // intermediate from fw
    const ttnn::Tensor& dL_dout_tensor,
    bool compute_dgamma = true);

}  // namespace ttml::metal
