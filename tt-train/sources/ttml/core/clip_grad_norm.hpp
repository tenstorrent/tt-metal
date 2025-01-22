// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "serialization/serializable.hpp"

namespace ttml::core {

// Clip the gradients of the parameters up to a given maximum norm. If
// error_if_nonfinite is true, an error is thrown if the sum of the parameters
// is in {nan,inf,-inf}. p_norm_type specifies which p-norm
// (https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm) to use in the norm
// calculation. Gradients are clipped in place in keeping with pytorch:
// https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
// Returns the summed norm of the gradients after clipping.
autograd::TensorPtr clip_grad_norm(
    const serialization::NamedParameters& parameters,
    float max_norm,
    float p_norm_type = 2.0F,
    bool error_if_nonfinite = true);
}  // namespace ttml::core
