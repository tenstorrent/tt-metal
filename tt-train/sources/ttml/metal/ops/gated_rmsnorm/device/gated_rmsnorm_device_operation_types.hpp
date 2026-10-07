// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gated_rmsnorm::device {

// Shared geometry/validation for the forward and backward device operations.
//   input / gate / dL_dout : [B, 1, T, W] TILE bf16, DRAM interleaved
//   gamma                  : [1, 1, 1, group] TILE bf16, group | W, group % 32 == 0
struct GatedRmsNormGeometry {
    uint32_t rows_tiles = 0;   // B * T / 32
    uint32_t width_tiles = 0;  // W / 32
    uint32_t group_tiles = 0;  // group / 32
    uint32_t num_groups = 0;   // W / group
    uint32_t group = 0;
};

GatedRmsNormGeometry validate_and_get_geometry(
    const char* op_name,
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& gamma,
    const std::optional<ttnn::Tensor>& dL_dout);

// ---------------------------------------------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------------------------------------------
namespace fw {

struct Params {
    float epsilon = 1e-6F;
};

struct Inputs {
    ttnn::Tensor input;
    ttnn::Tensor gate;
    ttnn::Tensor gamma;
};

using operation_attributes_t = Params;
using tensor_args_t = Inputs;
using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace fw

// ---------------------------------------------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------------------------------------------
namespace bw {

struct Params {
    float epsilon = 1e-6F;
    bool compute_dgamma = true;
};

struct Inputs {
    ttnn::Tensor input;
    ttnn::Tensor gate;
    ttnn::Tensor gamma;
    ttnn::Tensor dL_dout;
};

using operation_attributes_t = Params;
using tensor_args_t = Inputs;
// {dL_dinput, dL_dgate, dL_dgamma_components (unreduced, [B,1,T,W]; absent when !compute_dgamma)}
using spec_return_value_t = std::vector<std::optional<tt::tt_metal::TensorSpec>>;
using tensor_return_value_t = std::vector<std::optional<ttnn::Tensor>>;

}  // namespace bw

}  // namespace ttml::metal::ops::gated_rmsnorm::device
