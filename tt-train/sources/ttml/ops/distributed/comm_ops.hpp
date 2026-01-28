// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

// dim: tensor dimension to scatter/gather/reduce across (which tensor dimension will change after the operation)
// cluster_axis: mesh device shape axis to scatter/gather/reduce across (which parts of the tensor participate in the
// operation), default is none (all axes)
autograd::TensorPtr reduce_scatter(
    const autograd::TensorPtr& tensor, const int dim, const std::optional<uint32_t> cluster_axis = std::nullopt);
autograd::TensorPtr scatter(
    const autograd::TensorPtr& tensor, const int dim, const std::optional<uint32_t> cluster_axis = std::nullopt);
autograd::TensorPtr all_reduce(
    const autograd::TensorPtr& tensor,
    const bool noop_backward = false,
    const std::optional<uint32_t> cluster_axis = std::nullopt);
autograd::TensorPtr all_gather(
    const autograd::TensorPtr& tensor, const int dim, const std::optional<uint32_t> cluster_axis = std::nullopt);
autograd::TensorPtr broadcast(
    const autograd::TensorPtr& tensor, const std::optional<uint32_t> cluster_axis = std::nullopt);
autograd::TensorPtr ring_shift(
    const autograd::TensorPtr& tensor,
    const std::optional<uint32_t> cluster_axis = std::nullopt,
    const ttnn_fixed::distributed::RingShiftDirection direction = ttnn_fixed::distributed::RingShiftDirection::Forward);
}  // namespace ttml::ops::distributed
