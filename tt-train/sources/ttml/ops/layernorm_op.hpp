// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr composite_layernorm(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, std::optional<autograd::TensorPtr> beta_opt);

autograd::TensorPtr layernorm(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, std::optional<autograd::TensorPtr> beta_opt);

}  // namespace ttml::ops
