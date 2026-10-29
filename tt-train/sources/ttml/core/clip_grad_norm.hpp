// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

// Tensor-parallel clip_grad_norm. `sharded` gradients are split along mesh axis `cluster_axis`
// (their Sum[|g|^p] is all-reduced across it); `replicated` ones are identical on that axis and
// counted once. Clips in place; returns the pre-clip norm on device without a host sync, so a
// non-finite norm NaNs the gradients instead of throwing.
autograd::TensorPtr clip_grad_norm_tp(
    const serialization::NamedParameters& sharded,
    const serialization::NamedParameters& replicated,
    float max_norm,
    uint32_t cluster_axis,
    float p_norm_type = 2.0F);

}  // namespace ttml::core
