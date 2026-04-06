// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr layernorm_moreh(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, const autograd::TensorPtr& beta);

autograd::TensorPtr composite_layernorm(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, std::optional<autograd::TensorPtr> beta_opt);

autograd::TensorPtr layernorm(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, std::optional<autograd::TensorPtr> beta_opt);

}  // namespace ttml::ops
