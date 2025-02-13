// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_op.hpp"

#include <core/ttnn_all_includes.hpp>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <ttnn/operations/eltwise/unary/unary.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr rmsnorm(const autograd::TensorPtr &tensor, const autograd::TensorPtr &gamma, float epsilon) {
    auto a_shape = tensor->get_value().logical_shape();
    if (a_shape.rank() != 4) {
        throw std::runtime_error{"rmsnorm only supports rank-4 input tensors."};
    }
    auto [B, N, S, C] = a_shape.to_array_4D();

    auto device = &autograd::ctx().get_device();
    ttnn::Tensor squares = ttnn::square(tensor->get_value());  // [B,N,S,C]
    std::array<uint32_t, 4> eps_shape = {1, 1, 1, 1};
    ttnn::Tensor eps_tensor = core::from_vector({epsilon}, core::create_shape(eps_shape), device);
    ttnn::Tensor seq_means_of_squares = ttnn::mean(squares, /*dim_arg=*/-1, /*keep_dim=*/true);  // [B,N,S,1]
    std::cout << "mean of square shape: " << seq_means_of_squares.logical_shape() << std::endl;
    ttnn::Tensor seq_means_of_squares_plus_epsilon =
        ttnn::experimental::add(seq_means_of_squares, eps_tensor);       // [B,N,S,1]
    ttnn::Tensor rms_a = ttnn::sqrt(seq_means_of_squares_plus_epsilon);  // [B,N,S,1]
    ttnn::Tensor gamma_times_activations =
        ttnn::experimental::mul(gamma->get_value(), tensor->get_value());               // [B,N,S,C]
    ttnn::Tensor out_tensor = ttnn::experimental::div(gamma_times_activations, rms_a);  // [B,N,S,C]

    auto out = autograd::create_tensor(out_tensor);

    autograd::GradFunction grad = [B, N, S, C, tensor, gamma, out, rms_a, eps_tensor, device]() {
        auto a = tensor->get_value();  // [B,N,S,C]
        auto g = gamma->get_value();   // [B,N,S,C]

        // c is the number of activations; in the RMSNorm paper they call this
        // "n". it is renamed here to avoid confusion with N.
        auto c = static_cast<float>(a.logical_shape()[-1]);

        auto dL_dout = out->get_grad();  // Grad w.r.t normalized arctivations, hence [B,N,S,C]

        auto rms_a_all_dims_one =
            std::ranges::all_of(rms_a.logical_shape().view(), [](const auto &d) { return d == 1; });
        if (!rms_a_all_dims_one) {
            throw std::runtime_error{"Expected a scalar in RMS(a, epsilon)."};
        }

        // have a : [B,N,S,C]
        // want outer product tensor [B,N,S,C,C]
        // reshape to get [B,N,S,1,C], mul by its own transpose.

        auto a_vectors = ttnn::reshape(a, ttnn::Shape{B, N, S, C, 1});
        fmt::println("a_vectors shape: {}", a_vectors.logical_shape());
        auto outer = ttnn::matmul(a_vectors, a_vectors, /*transpose_a=*/false, /*transpose_b=*/true);
        auto desired_outer_shape = {B, N, S, C, C};
        auto outer_shape = outer.logical_shape();
        if (outer_shape != desired_outer_shape) {
            throw std::runtime_error{
                fmt::format("Expected outer product to be a tensor with shape [B,N,S,C,C] but got {}.", outer_shape)};
        }

        auto ms_a = ttnn::square(rms_a);
        auto c_by_ms_a = ttnn::experimental::mul(ms_a, c);
        auto scaled_outer = ttnn::experimental::div(outer, c_by_ms_a);
        auto gained_dL_dout = ttnn::experimental::mul(ttnn::experimental::div(g, rms_a), dL_dout);
        auto dL_da = ttnn::experimental::sub(gained_dL_dout, ttnn::matmul(gained_dL_dout, scaled_outer));
        tensor->add_grad(dL_da);

        auto dL_dg = ttnn::experimental::div(a, rms_a);
        gamma->add_grad(dL_dg);
    };

    auto links = autograd::get_links(tensor, gamma);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
