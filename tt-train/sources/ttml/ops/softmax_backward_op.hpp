// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/core_coord.hpp>

#include "autograd/tensor.hpp"

namespace ttml::ops {

/** Computes the gradient of softmax: output = y * (grad - sum(y * grad, dim, keepdim=True)).
 * Uses the tt-metal softmax backward kernel. Supports the last dimension only.
 *
 * @param softmax_output The softmax output tensor (y) from the forward pass.
 * @param grad Upstream gradient with respect to y.
 * @param dim Dimension along which softmax was applied in the forward pass (e.g. -1 for last dim).
 * @param sub_core_grids Optional. Restricts which device cores run this op. When std::nullopt
 *        (default), the full compute grid is used. When set, only cores in the given CoreRangeSet
 *        are used; work (tile rows) is split across them in a deterministic order. Useful for:
 *        - Reserving part of the grid for other ops.
 *        - Non-rectangular or disjoint core sets (e.g. L-shapes, or several CoreRanges).
 *        The set can be a single CoreRange, multiple CoreRanges (e.g. from a vector), or any
 *        CoreRangeSet. Row distribution order: ranges are traversed in order; within each
 *        CoreRange, cores are traversed row-major (x then y). All cores in the set must be valid
 *        for the device. Pass std::nullopt to use the full device grid.
 * @return autograd::TensorPtr holding the gradient with respect to the pre-softmax logits.
 */
autograd::TensorPtr softmax_backward(
    const autograd::TensorPtr& softmax_output,
    const autograd::TensorPtr& grad,
    int dim,
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids = std::nullopt);

}  // namespace ttml::ops
