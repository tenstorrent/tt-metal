// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Which grads to compute; skip one for a frozen tensor.
struct RMSNormBackwardGrads {
    bool input = true;
    bool gamma = true;
};

// Returns {dL_dinput, dL_dgamma, dL_dgate}. dL_dinput is computed only with `grads.input`, dL_dgamma only with
// `grads.gamma` and dL_dgate only in gated mode, where `dL_dout` and dL_dgate have the gate's layout; see rmsnorm_fw.
// A grad that is not computed is nullopt. Without a gate, at least one of the flags must be set.
std::vector<std::optional<ttnn::Tensor>> rmsnorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,  // intermediate from fw
    const ttnn::Tensor& dL_dout_tensor,
    const std::optional<ttnn::Tensor>& gate_tensor = std::nullopt,
    const RMSNormBackwardGrads& grads = {});

}  // namespace ttml::metal
