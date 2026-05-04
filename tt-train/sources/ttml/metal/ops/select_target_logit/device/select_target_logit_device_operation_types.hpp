// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::select_target_logit::device {

// The shard window for each device is derived per-coord inside the program factory:
//
//   tp_rank        = cluster_axis.has_value() ? mesh_coord[*cluster_axis] : linear_index
//   device_first_v = first_v + tp_rank * local_V
//   device_last_v  = device_first_v + local_V
//
// Real callers pass (local_V, cluster_axis) and leave first_v = 0; first_v exists so
// single-device unit tests can still exercise non-zero shard windows.
struct operation_attributes_t {
    uint32_t first_v{0U};
    uint32_t local_V{0U};
    std::optional<uint32_t> cluster_axis{};
};

struct tensor_args_t {
    const ttnn::Tensor& logit;   // [N, 1, S, local_V] TILE BFLOAT16  (local_V = last_v - first_v)
    const ttnn::Tensor& target;  // [N, S]              ROW_MAJOR UINT32  (global indices)

    std::optional<ttnn::Tensor> preallocated_output;
};

// output: [N, 1, S, 1] TILE BFLOAT16
// output[n, 0, s, 0] = logit[n, 0, s, target[n, s] - device_first_v]
//                       if target[n, s] in [device_first_v, device_last_v)
//                     = 0.0 otherwise
using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::select_target_logit::device
