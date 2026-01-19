// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops::distributed {

autograd::TensorPtr reduce_scatter(const autograd::TensorPtr& tensor, int dim, std::optional<uint32_t> cluster_axis = std::nullopt);
autograd::TensorPtr scatter(const autograd::TensorPtr& tensor, int dim, std::optional<uint32_t> cluster_axis = std::nullopt);
autograd::TensorPtr all_reduce(const autograd::TensorPtr& tensor, bool noop_backward = false, std::optional<uint32_t> cluster_axis = std::nullopt);
autograd::TensorPtr all_gather(const autograd::TensorPtr& tensor, int dim, std::optional<uint32_t> cluster_axis = std::nullopt);
autograd::TensorPtr broadcast(const autograd::TensorPtr& tensor, std::optional<uint32_t> cluster_axis = std::nullopt);
autograd::TensorPtr ring_shift(
    const autograd::TensorPtr& tensor, std::optional<uint32_t> cluster_axis = std::nullopt, bool forward = true);

}  // namespace ttml::ops::distributed
