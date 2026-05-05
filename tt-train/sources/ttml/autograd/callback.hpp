// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "tensor.hpp"

namespace ttml::autograd {

/**
 * @brief Insert an identity autograd node that fires a user-provided callback during backward.
 *
 * Forward: returns a new Tensor that shares the underlying ttnn tensor handle with @p input
 * (no device-side copy). The returned tensor is wired into the autograd graph with a single
 * parent (@p input), so its backward node sits at the same topological position as @p input's
 * consumers.
 *
 * Backward: the graph node invokes @p callback first, then forwards the out-grad to the input
 * via add_grad. Because callbacks on a module's output fire *before* the module's internal
 * backward closures in the reversed topological order, and callbacks on a module's input fire
 * *after* them, this primitive enables module-level backward-pre / backward-post hooks
 * (e.g. FSDP unshard/reshard + reduce_scatter) to run at the correct moment.
 *
 * Memory: one autograd::Tensor wrapper + one GraphNode. The ttnn tensor handle is ref-counted,
 * so no device buffer is allocated by this op.
 *
 * @param input    Tensor to pass through. Must have an associated autograd node for the
 *                 callback to ever fire (otherwise the graph has nothing to traverse
 *                 through this op).
 * @param callback Nullary side-effect function invoked during backward.
 * @return         Identity-forwarded tensor carrying the callback node.
 */
TensorPtr autograd_callback(const TensorPtr& input, std::function<void()> callback);

}  // namespace ttml::autograd
