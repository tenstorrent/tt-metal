// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

// Composed | Fused | NoComm, selected process-wide with ttnn_fixed::distributed::set_sp_linear_impl (both sites)
// and set_sp_linear_backward_impl (the backward alone); the two-stream backward is ops/distributed/sp_overlap.hpp.
using ttnn_fixed::distributed::SPLinearImpl;
using ttnn_fixed::distributed::SPLinearSite;

// The Megatron sequence-parallel linears: the collective on dim 2 of (B, 1, S, X) issued together with the
// matmul it feeds (column) or that feeds it (row), across mesh axis `cluster_axis` with T ranks. Under
// SPLinearImpl::Composed each is the exact autograd equivalent of the all_gather -> linear_op and
// linear_op -> reduce_scatter sequences the modules used before; Fused runs the fused ttnn ops instead.

// x [B,1,S/T,K] sequence-sharded, weight [1,1,N/T,K] output-feature-sharded, bias [1,1,1,N/T] or null.
// Forward: all_gather(x) @ W^T (+ bias) -> [B,1,S,N/T]; the gathered x is kept for the weight gradient.
// Backward: dgrad = reduce_scatter(grad @ W) -> [B,1,S/T,K]; wgrad = grad^T @ gathered_x -> [1,1,N/T,K];
// bias grad = the column sums of grad, complete on every rank (the full sequence, the rank's own features).
autograd::TensorPtr sp_column_parallel_linear(
    const autograd::TensorPtr& x,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    uint32_t cluster_axis);

// x [B,1,S,K/T] input-feature-sharded, weight [1,1,N,K/T].
// Forward: reduce_scatter over T of x @ W^T, split on the sequence -> [B,1,S/T,N]. The row-parallel bias is
// added by the module afterwards (it is marked sequence-parallel already).
// Backward: {grad_full, dgrad} = all_gather_matmul(grad, W): dgrad = all_gather(grad) @ W -> [B,1,S,K/T];
// wgrad = grad_full^T @ x -> [1,1,N,K/T].
autograd::TensorPtr sp_row_parallel_linear(
    const autograd::TensorPtr& x, const autograd::TensorPtr& weight, uint32_t cluster_axis);

}  // namespace ttml::ops::distributed
