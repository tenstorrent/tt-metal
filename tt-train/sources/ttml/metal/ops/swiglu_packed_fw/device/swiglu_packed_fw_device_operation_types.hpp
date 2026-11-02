// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::swiglu_packed_fw::device {

struct SwigluPackedFwParams {
    uint32_t out_cols = 0;  // output width (>= I, multiple of 32; 0 = I): the wide LoRA input buffer (iteration 24)
    // iteration 38: the output transposed, [.., out_cols, R] with h^T in rows [0, I) and the rows past I zeroed
    // (the transposed wide LoRA input buffer of a transposed-orientation w2)
    bool transposed_output = false;
    // iteration 39: the input is packed^T [1, 1, 2I, T] (gate rows [0, I), up rows [I, 2I)) and the output h^T
    // [1, 1, out_cols, T] (rows past I zeroed): the transposed wide buffer without any transpose in the kernel
    bool transposed_input = false;
};

// `packed` [.., R, 2*I] = [gate | up]; output h = silu(gate) * up, [.., R, I].
struct SwigluPackedFwInputs {
    ttnn::Tensor packed;
    std::optional<ttnn::Tensor> preallocated_output = std::nullopt;
};

using operation_attributes_t = SwigluPackedFwParams;
using tensor_args_t = SwigluPackedFwInputs;

using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::swiglu_packed_fw::device
