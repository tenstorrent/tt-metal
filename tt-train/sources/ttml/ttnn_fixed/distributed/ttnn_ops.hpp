// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>

namespace ttml::ttnn_fixed::distributed {

ttnn::Tensor all_gather(
    const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis = std::nullopt);
ttnn::Tensor all_reduce(const ttnn::Tensor& tensor, const std::optional<uint32_t> cluster_axis = std::nullopt);
ttnn::Tensor reduce_scatter(
    const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis = std::nullopt);

// Local, communication-free per-device shard extraction along `dim` on `cluster_axis`.
// Each device slices out its own `size(dim) / axis_size` partition based on its mesh
// coordinate — the inverse of all_gather with NO collective. Only correct/meaningful
// when the input is replicated across `cluster_axis`; requires `dim` divisible by the
// axis size and tile-aligned in TILE layout.
ttnn::Tensor mesh_partition(
    const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis = std::nullopt);

/**
 * Direction for ring shift operation.
 */
enum class RingShiftDirection {
    Forward,  // device i sends to device (i+1) % ring_size
    Backward  // device i sends to device (i-1+ring_size) % ring_size
};

/**
 * Ring shift operation - shifts tensor to next/previous device in the ring.
 *
 * @param tensor The input tensor to shift
 * @param cluster_axis Optional axis of the device mesh along which to perform the ring shift.
          E.g. for 2d case and direction == RingShiftDirection::Forward, if cluster axis == 0, then
          each device with coordinate (idx0, idx1) sends to ((idx0 + 1) % mesh_shape[0], idx1),
          otherwise if cluster axis == 1, then
          each device with coordinate (idx0, idx1) sends to (idx0, (idx1 + 1) % mesh_shape[1])
 *        If std::nullopt (the default) and the device fabric is 1D, axis 1 is used.
 *        For multi-dimensional fabrics, this parameter must be explicitly specified.
 * @param direction Direction to shift: Forward (i -> i+1) or Backward (i -> i-1)
 * @return The tensor received from the neighbor device
 */
ttnn::Tensor ring_shift(
    const ttnn::Tensor& tensor,
    const std::optional<uint32_t> cluster_axis = std::nullopt,
    const RingShiftDirection direction = RingShiftDirection::Forward);

// Which implementation backs the sequence-parallel linears (ops/distributed/sp_linear_ops.hpp), process-wide.
//   Composed: today's unfused sequence -- the collective, then the matmul (or the reverse) -- op for op.
//   Fused:    the fused ttnn ops of issue #52944 (all_gather_matmul_sp_async / matmul_reduce_scatter_sp_async).
enum class SPLinearImpl { Composed, Fused };
void set_sp_linear_impl(SPLinearImpl impl);
SPLinearImpl get_sp_linear_impl();

// Sequence-parallel matmul fusions on dim 2 of (B, 1, S, X) across `cluster_axis`; T = the mesh extent on it.
// The result is the Composed sequence's result; the implementation only decides how it is computed.
//
// x [B,1,S/T,K] sequence-sharded -> {all_gather(x) [B,1,S,K], all_gather(x) @ (transpose_b ? W^T : W) (+ bias)}.
// The gathered activation is returned because the caller's weight gradient needs it. A bias is only
// supported with transpose_b (the forward linear).
std::pair<ttnn::Tensor, ttnn::Tensor> all_gather_matmul(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    uint32_t cluster_axis,
    bool transpose_b,
    const std::optional<ttnn::Tensor>& bias = std::nullopt);

// x [B,1,S,K] -> reduce_scatter over T of x @ (transpose_b ? W^T : W), split on the sequence: [B,1,S/T,N].
ttnn::Tensor matmul_reduce_scatter(
    const ttnn::Tensor& x, const ttnn::Tensor& w, uint32_t cluster_axis, bool transpose_b);

}  // namespace ttml::ttnn_fixed::distributed
