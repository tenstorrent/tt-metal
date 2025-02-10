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

autograd::TensorPtr rmsnorm(const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, float epsilon) {
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

    autograd::GradFunction grad = [tensor, gamma, out, eps_tensor, device]() {
        auto tensor_to_str = [](const auto &tensor) {
            std::ostringstream oss;
            oss << core::to_xtensor(tensor);
            return oss.str();
        };

        auto print_tensor = [=](std::string name, const auto &tensor) {
            std::cout << name << ": " << tensor_to_str(tensor) << "\n";
        };

        auto a = tensor->get_value();
        auto g = gamma->get_value();
        auto n = static_cast<float>(a.logical_shape().rank());
        auto dL_dout = out->get_grad();
        auto rms_a = out->get_value();

        // let tensor = {a_i | i = 0, 1, ..., n}
        // and gamma = {g_i | i = 0, 1, ..., n}

        auto g_times_dL_dout = ttnn::experimental::mul(g, dL_dout);
        auto l = ttnn::experimental::div(g_times_dL_dout, rms_a);
        auto scale = ttnn::matmul(a, g_times_dL_dout, /*transpose a*/ true, /*transpose b*/ false);
        auto r = ttnn::experimental::div(
            ttnn::experimental::mul(a, scale), ttnn::experimental::mul(ttnn::pow(rms_a, 3.0F), n));
        auto dL_da = ttnn::experimental::sub(l, r);
        tensor->add_grad(dL_da);
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
