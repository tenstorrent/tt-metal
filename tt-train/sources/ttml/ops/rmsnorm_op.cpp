// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <core/ttnn_all_includes.hpp>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <ttnn/operations/eltwise/unary/unary.hpp>

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

autograd::TensorPtr rmsnorm(const autograd::TensorPtr &tensor, const autograd::TensorPtr &gamma, float epsilon) {
    auto device = &autograd::ctx().get_device();
    ttnn::Tensor squares = ttnn::square(tensor->get_value());
    std::array<uint32_t, 4> eps_shape = {1, 1, 1, 1};
    ttnn::Tensor eps_tensor = core::from_vector({epsilon}, core::create_shape(eps_shape), device);
    ttnn::Tensor mean_of_squares = ttnn::mean(squares);
    ttnn::Tensor mean_of_squares_plus_epsilon = ttnn::experimental::add(mean_of_squares, eps_tensor);
    ttnn::Tensor rms_eps = ttnn::sqrt(mean_of_squares_plus_epsilon);
    ttnn::Tensor gamma_times_activations = ttnn::experimental::mul(gamma->get_value(), tensor->get_value());
    ttnn::Tensor out_tensor = ttnn::experimental::div(gamma_times_activations, rms_eps);

    auto out = autograd::create_tensor(out_tensor);
    out->set_value(out_tensor);

    autograd::GradFunction grad = [tensor, gamma, out, rms_eps, eps_tensor, device]() {
        auto a = tensor->get_value();
        auto g = gamma->get_value();
        auto n = static_cast<float>(a.logical_shape()[-1]);
        auto dL_dout = out->get_grad();

        auto rms_a = rms_eps;
        TT_ASSERT(
            std::ranges::all_of(rms_a.logical_shape().view(), [](const auto &d) { return d == 1; }),
            "Expected a scalar in RMS(a).");

        auto outer =
            ttnn::matmul(a, a, /*transpose_a=*/true, /*transpose_b=*/false);  // row vectors so the order is flipped
        TT_ASSERT(
            [=]() {
                auto outer_shape = outer.logical_shape();
                return outer_shape[0] == static_cast<int>(n);
            }(),
            "Expected outer product to be an n-square tensor.");

        auto ms_a = ttnn::square(rms_a);
        auto n_by_ms_a = ttnn::experimental::mul(ms_a, n);
        auto scaled_outer = ttnn::experimental::div(outer, n_by_ms_a);
        auto gained_dL_dout = ttnn::experimental::mul(ttnn::experimental::div(g, rms_a), dL_dout);
        auto dL_da = ttnn::experimental::sub(gained_dL_dout, ttnn::matmul(gained_dL_dout, scaled_outer));
        tensor->add_grad(dL_da);
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
