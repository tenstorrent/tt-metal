// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>

namespace ttml::metal::ops::k_split_gram_matmul::device {

enum class OutputMode : uint32_t {
    UpperTriangle = 0,  // Write only upper triangle + diagonal (default)
    Full = 1,           // Write full symmetric matrix (upper + transposed mirror to lower)
};

struct operation_attributes_t {
    OutputMode output_mode = OutputMode::UpperTriangle;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t override_kb = 0;  // 0 = auto
    uint32_t override_mb = 0;
    uint32_t override_db = 0;
};

struct tensor_args_t {
    ttnn::Tensor input_tensor;
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::k_split_gram_matmul::device
