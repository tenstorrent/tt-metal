// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::ring_lse_merge::device {

// The merge has no scalar parameters: everything it needs is in the four tensors.
struct operation_attributes_t {};

// Online-softmax (logsumexp) merge of one ring step's partial attention result into a running
// accumulator, in place:
//   output_accum       FP32 (B, H, S, D), updated in place.
//   running_lse        FP32 (B, H, S, 32), logsumexp in column 0, zeros elsewhere (the SDPA
//                      intermediates layout), updated in place.
//   step_output        BF16 (B, H, S, D), this step's normalized attention output O_i.
//   step_intermediates FP32 (B, H, S, 32), this step's logsumexp lse_i in column 0.
struct tensor_args_t {
    const ttnn::Tensor& output_accum;
    const ttnn::Tensor& running_lse;
    const ttnn::Tensor& step_output;
    const ttnn::Tensor& step_intermediates;
};

// {output_accum, running_lse}: the in-place outputs, aliasing the inputs of the same name.
using tensor_return_value_t = std::vector<ttnn::Tensor>;

using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::ring_lse_merge::device
