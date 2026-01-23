// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

autograd::TensorPtr reduce_scatter(const autograd::TensorPtr& tensor, int dim);
autograd::TensorPtr all_reduce(const autograd::TensorPtr& tensor, bool noop_backward = false);
autograd::TensorPtr all_gather(const autograd::TensorPtr& tensor, int dim);
autograd::TensorPtr broadcast(const autograd::TensorPtr& tensor);
autograd::TensorPtr ring_shift(
    const autograd::TensorPtr& tensor,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    ttnn_fixed::distributed::RingShiftDirection direction = ttnn_fixed::distributed::RingShiftDirection::Forward);

}  // namespace ttml::ops::distributed
