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
    auto squares = tensor * tensor;
    std::array<uint32_t, 4> eps_shape = {1, 1, 1, 1};
    auto eps_tensor = autograd::create_tensor(
        core::from_vector({epsilon}, core::create_shape(eps_shape), &autograd::ctx().get_device()));
    auto mean_of_squares = ttml::ops::mean(squares);
    auto mean_of_squares_plus_epsilon = mean_of_squares + eps_tensor;
    auto rms_eps = ttml::ops::sqrt(mean_of_squares_plus_epsilon);
    auto gamma_times_activations = (gamma * tensor)->get_value();
    float rms_eps_value = core::to_xtensor(rms_eps->get_value())[0];
    auto out_tensor = ttnn::divide(gamma_times_activations, rms_eps_value);
    
    auto out = autograd::create_tensor(out_tensor);

    autograd::GradFunction grad = [tensor, out, gamma, epsilon]() {
        auto dout = out->get_grad();
        // let tensor = {a_i | i = 0, 1, ..., n}
        // and gamma = {g_i | i = 0, 1, ..., n}

        // backward grads in terms of dout:
        // dL/da_i = dL/dout * eps * gamma_i / (eps + a_i^2)^(3/2)
        // dL/dg_i = dL/dout * a_i / sqrt(eps + a_i^2)

        auto dtensor_divisor = ttnn::pow(ttnn::add(epsilon, ttnn::square(tensor->get_value())), 3.0F / 2.0F);
        auto dtensor_dividend = ttnn::multiply(ttnn::multiply(dout, gamma), epsilon);
        auto dtensor = ttnn::divide(dtensor_dividend, dtensor_divisor);
        
        auto dgamma_dividend = ttnn::multiply(dout, tensor);
        auto dgamma_divisor = ttnn::sqrt(ttnn::add(epsilon, ttnn::square(tensor->get_value())));
        auto dgamma = ttnn::divide(dgamma_dividend, dgamma_divisor);

        tensor->add_grad(dtensor);
        gamma->add_grad(dgamma);
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
