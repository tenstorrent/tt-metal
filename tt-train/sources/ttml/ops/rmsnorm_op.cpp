// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <core/ttnn_all_includes.hpp>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
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

    autograd::GradFunction grad = [tensor, gamma, out, eps_tensor]() {
        auto a = tensor->get_value();
        auto g = gamma->get_value();
        auto dout = out->get_grad();
        // let tensor = {a_i | i = 0, 1, ..., n}
        // and gamma = {g_i | i = 0, 1, ..., n}

        // backward grads in terms of dout:
        // dL/da_i = dL/dout * eps * gamma_i / (eps + a_i^2)^(3/2)
        // dL/dg_i = dL/dout * a_i / sqrt(eps + a_i^2)

        auto dtensor_divisor = ttnn::pow(ttnn::experimental::add(eps_tensor, ttnn::square(a)), 3.0F / 2.0F);
        auto dtensor_dividend = ttnn::experimental::mul(ttnn::experimental::mul(dout, g), eps_tensor);
        auto dtensor = ttnn::experimental::div(dtensor_dividend, dtensor_divisor);
        
        auto dgamma_dividend = ttnn::experimental::mul(dout, a);
        auto dgamma_divisor = ttnn::sqrt(ttnn::experimental::add(eps_tensor, ttnn::square(a))); // using a^2 + eps for scalar add in ttnn.
        auto dgamma = ttnn::experimental::div(dgamma_dividend, dgamma_divisor);

        tensor->add_grad(dtensor);
        gamma->add_grad(dgamma);
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
