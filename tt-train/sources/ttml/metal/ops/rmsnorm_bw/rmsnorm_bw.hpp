// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Returns {dL_dinput, dL_dgamma, dL_dgate}. dL_dgamma is computed only with `compute_dgamma` (skip it for a frozen
// gamma) and dL_dgate only in gated mode, where `dL_dout` and dL_dgate have the gate's layout; see rmsnorm_fw.
std::vector<std::optional<ttnn::Tensor>> rmsnorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,  // intermediate from fw
    const ttnn::Tensor& dL_dout_tensor,
    bool compute_dgamma = true,
    const std::optional<ttnn::Tensor>& gate_tensor = std::nullopt);

}  // namespace ttml::metal
