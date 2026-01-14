// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>

namespace ttml::ttnn_fixed::distributed {

tt::tt_metal::Tensor all_gather(const tt::tt_metal::Tensor& tensor, int dim);
tt::tt_metal::Tensor all_reduce(const tt::tt_metal::Tensor& tensor);
tt::tt_metal::Tensor reduce_scatter(const tt::tt_metal::Tensor& tensor, int dim);

/**
 * Ring shift operation - shifts tensor to next/previous device in the ring.
 *
 * @param tensor The input tensor to shift
 * @param cluster_axis The axis of the mesh to perform the ring shift on
 * @param forward If true, shift forward (device i sends to i+1), else backward (device i sends to i-1)
 * @return The tensor received from the neighbor device
 */
tt::tt_metal::Tensor ring_shift(const tt::tt_metal::Tensor& tensor, uint32_t cluster_axis, bool forward = true);

}  // namespace ttml::ttnn_fixed::distributed
