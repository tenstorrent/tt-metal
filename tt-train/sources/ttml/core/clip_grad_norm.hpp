// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

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
// Without cluster_axis, each device computes the norm of the gradients it holds. With cluster_axis,
// the parameters are tensor-parallel over that mesh axis: gradients sharded on it have their
// Sum[|g|^p] all-reduced across it, and replicated ones are counted once, so they must be identical
// on every device of that axis. All-reduce partial-sum grads first (e.g. Qwen3 QK-norm, and
// LoraColumnParallelLinear's lora_A until #62958 lands); otherwise the norm is wrong. No parameter
// may be sharded on another mesh axis (so FSDP is not supported). Call after DDP gradient sync.
// Per-gradient sums accumulate in bf16 (moreh step1), so the norm is underestimated for gradients
// of more than ~1k tiles per device (#63505).
autograd::TensorPtr clip_grad_norm(
    const serialization::NamedParameters& parameters,
    float max_norm,
    float p_norm_type = 2.0F,
    bool error_if_nonfinite = true,
    std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ttml::core
