// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <core/ttnn_all_includes.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "layernorm_op.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr rmsnorm(const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, float epsilon) {
    auto squares = ttml::ops::matmul(
        tensor,
        tensor,
        /* transpose_a */ false,
        /* transpose_b */ true);
    auto eps_tensor =
        autograd::create_tensor(core::from_xtensor(xt::xarray<float>{epsilon}, &autograd::ctx().get_device()));
    auto mean_of_squares = ttml::ops::mean(squares);
    auto mean_of_squares_plus_epsilon = ttml::ops::add(mean_of_squares, eps_tensor);
    auto rms_eps = ttml::ops::sqrt(mean_of_squares_plus_epsilon);
    auto gamma_times_activations = gamma * tensor;
    auto out = gamma_times_activations / rms_eps;

    return out;
}

}  // namespace ttml::ops
