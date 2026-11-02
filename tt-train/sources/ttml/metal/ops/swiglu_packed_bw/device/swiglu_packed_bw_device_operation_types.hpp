// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::swiglu_packed_bw::device {

struct SwigluPackedBwParams {
    // iteration 38: dL_dh arrives transposed ([.., >= I + 32, R]: dh^T in rows [0, I), the scaled LoRA tail
    // s_dh^T in rows [I, I + 32)) and the low-rank term lowrank_scale * s_dh @ lora_a is folded into dh in DEST
    bool transposed_grad = false;
    float lowrank_scale = 1.0F;
    // iteration 39: packed is packed^T [1, 1, 2I, T], dL_dh is dh^T wide [1, 1, >= I + 32, T] (rows [I, I + 32) the
    // LoRA tail s_dh^T) and the output dpacked^T [1, 1, 2I, T]; dh^T += lowrank_scale * lora_a^T @ s_dh^T is folded
    // in while the dh^T tiles are copied into DEST (no transpose anywhere)
    bool transposed_io = false;
};

// Backward of swiglu_packed_fw. `packed` [.., R, 2*I] = [gate | up], `dL_dh` [.., R, I];
// produces dL_dpacked [.., R, 2*I] = [dgate | dup].
struct SwigluPackedBwInputs {
    ttnn::Tensor packed;
    ttnn::Tensor dL_dh;
    std::optional<ttnn::Tensor> preallocated_dL_dpacked = std::nullopt;
    std::optional<ttnn::Tensor> lora_a = std::nullopt;  // [1, 1, <= 32, I] bf16 (TRANSPOSED_GRAD / TRANSPOSED_IO)
};

using operation_attributes_t = SwigluPackedBwParams;
using tensor_args_t = SwigluPackedBwInputs;

using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::swiglu_packed_bw::device
