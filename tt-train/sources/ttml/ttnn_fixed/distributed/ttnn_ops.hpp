// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>

namespace ttml::ttnn_fixed::distributed {

tt::tt_metal::Tensor all_gather(const tt::tt_metal::Tensor& tensor, int dim, std::optional<uint32_t> cluster_axis = std::nullopt);
tt::tt_metal::Tensor all_reduce(const tt::tt_metal::Tensor& tensor, std::optional<uint32_t> cluster_axis = std::nullopt);
tt::tt_metal::Tensor reduce_scatter(const tt::tt_metal::Tensor& tensor, int dim, std::optional<uint32_t> cluster_axis = std::nullopt);

/**
 * Ring shift operation - shifts tensor to next/previous device in the ring.
 *
 * @param tensor The input tensor to shift
 * @param cluster_axis Optional axis of the device mesh along which to perform the ring shift.
 *        If std::nullopt (the default) and the device fabric is 1D, axis 1 is used.
 *        For multi-dimensional fabrics, this parameter must be explicitly specified.
 * @param forward If true, shift forward (device i sends to i+1), else backward (device i sends to i-1)
 * @return The tensor received from the neighbor device
 */
tt::tt_metal::Tensor ring_shift(
    const tt::tt_metal::Tensor& tensor, std::optional<uint32_t> cluster_axis = std::nullopt, bool forward = true);

/**
 * Ring shift operation - shifts tensor to next/previous device in the ring.
 *
 * @param tensor The input tensor to shift
 * @param cluster_axis Optional axis of the device mesh along which to perform the ring shift.
 *        If std::nullopt (the default) and the device fabric is 1D, axis 1 is used.
 *        For multi-dimensional fabrics, this parameter must be explicitly specified.
 * @param forward If true, shift forward (device i sends to i+1), else backward (device i sends to i-1)
 * @return The tensor received from the neighbor device
 */
tt::tt_metal::Tensor ring_shift(
    const tt::tt_metal::Tensor& tensor, std::optional<uint32_t> cluster_axis = std::nullopt, bool forward = true);

}  // namespace ttml::ttnn_fixed::distributed
