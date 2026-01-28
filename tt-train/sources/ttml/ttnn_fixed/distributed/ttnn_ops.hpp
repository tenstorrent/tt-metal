// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>

namespace ttml::ttnn_fixed::distributed {

tt::tt_metal::Tensor all_gather(
    const tt::tt_metal::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis = std::nullopt);
tt::tt_metal::Tensor all_reduce(
    const tt::tt_metal::Tensor& tensor, const std::optional<uint32_t> cluster_axis = std::nullopt);
tt::tt_metal::Tensor reduce_scatter(
    const tt::tt_metal::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis = std::nullopt);

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
tt::tt_metal::Tensor ring_shift(
    const tt::tt_metal::Tensor& tensor,
    const std::optional<uint32_t> cluster_axis = std::nullopt,
    const RingShiftDirection direction = RingShiftDirection::Forward);

}  // namespace ttml::ttnn_fixed::distributed
