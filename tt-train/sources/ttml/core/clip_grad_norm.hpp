// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "serialization/serializable.hpp"

namespace ttml::core {

// Clip the gradients of the parameters up to a given maximum norm. If
// error_if_nonfinite is true, an error is thrown if the norm of the gradients
// is in {nan,inf,-inf}; checking reads the norm back to the host. p_norm_type
// specifies which p-norm (https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm)
// to use in the norm calculation. Gradients are clipped in place in keeping with pytorch:
// https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
// Returns the norm of the gradients before clipping, as a [1, 1] tensor on device.
//
// On a mesh, the norm is that of the whole model, and every device gets the same one. Each gradient's Sum[|g|^p] is
// all-reduced over the mesh axes its parameter's placement says it is sharded on (tensor parallel, FSDP, or both),
// and counted once along the axes it is replicated on. Replicated gradients must therefore be identical on every
// device: call this after DDP gradient sync, and all-reduce partial-sum grads first (e.g. Qwen3 QK-norm); otherwise
// the norm is wrong. Per-gradient sums accumulate in bf16 (moreh step1), so the norm is underestimated for gradients
// of more than ~1k tiles per device (#63505).
autograd::TensorPtr clip_grad_norm(
    const serialization::NamedParameters& parameters,
    float max_norm,
    float p_norm_type = 2.0F,
    bool error_if_nonfinite = true);

}  // namespace ttml::core
