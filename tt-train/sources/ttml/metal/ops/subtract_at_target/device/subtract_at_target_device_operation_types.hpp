// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::subtract_at_target::device {

// The shard window for each device is derived per-coord inside the program factory:
//
//   tp_rank        = cluster_axis.has_value() ? mesh_coord[*cluster_axis] : mesh_coord.to_linear_index(mesh_shape)
//   device_first_v = first_v + tp_rank * local_V
//   device_last_v  = device_first_v + local_V
//
// Real callers pass (local_V, cluster_axis) and leave first_v = 0; first_v exists so
// single-device unit tests can still exercise non-zero shard windows.
struct SubtractAtTargetParams {
    uint32_t first_v{0U};
    uint32_t local_V{0U};
    std::optional<uint32_t> cluster_axis{};
    float subtract_value{1.0F};
};

struct SubtractAtTargetInputs {
    const ttnn::Tensor& input;   // [N, 1, S, local_V] TILE BFLOAT16
    const ttnn::Tensor& target;  // [N, S]              ROW_MAJOR UINT32  (global indices)

    std::optional<ttnn::Tensor> preallocated_output;
};

using operation_attributes_t = SubtractAtTargetParams;
using tensor_args_t = SubtractAtTargetInputs;

// output: same shape as input [N, 1, S, local_V] TILE BFLOAT16
// output[n, 0, s, c] = input[n, 0, s, c] - subtract_value
//                       if c + device_first_v == target[n, s] && target[n, s] in [device_first_v, device_last_v)
//                     = input[n, 0, s, c] otherwise
using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::subtract_at_target::device
