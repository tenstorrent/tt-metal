// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gated_rmsnorm::device {

// Head-merged activations [.., T, W] with W = H * V: work item (r, h) is the Gt consecutive tiles
// of head h on tile-row r, pages r * width_tiles + h * group_tiles + [0, group_tiles).
struct GatedRmsNormGeometry {
    uint32_t rows_tiles{};   // number of 32-row tile-rows over all leading dims
    uint32_t width_tiles{};  // Wt = W / 32
    uint32_t group_tiles{};  // Gt = V / 32
    uint32_t num_groups{};   // H = W / V
    uint32_t group{};        // V, the reduction width

    [[nodiscard]] uint32_t total_work() const {
        return rows_tiles * num_groups;
    }
};

// Checks the contract shared by fw and bw and returns the geometry. `dL_dout` is bw only.
GatedRmsNormGeometry validate_and_get_geometry(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& gamma,
    const std::optional<ttnn::Tensor>& dL_dout,
    std::string_view op_name);

namespace fw {

struct Params {
    float epsilon = 1e-6F;
};

struct Inputs {
    ttnn::Tensor input;
    ttnn::Tensor gate;
    ttnn::Tensor gamma;
};

using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace fw

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

// Specs of the outputs that are produced: {dx, dgate} plus dgamma_components when compute_dgamma.
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
// {dx, dgate, dgamma_components}; dgamma_components is nullopt unless compute_dgamma.
using tensor_return_value_t = std::vector<std::optional<ttnn::Tensor>>;

}  // namespace bw

}  // namespace ttml::metal::ops::gated_rmsnorm::device
