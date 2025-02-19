// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_op.hpp"

#include <cassert>
#include <core/ttnn_all_includes.hpp>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr rmsnorm(const autograd::TensorPtr &tensor, const autograd::TensorPtr &gamma, float epsilon) {
    auto a_shape = tensor->get_value().logical_shape();
    if (a_shape.rank() != 4) {
        throw std::runtime_error("rmsnorm only supports rank-4 input tensors.");
    }

    auto ashape_arr = a_shape.to_array_4D();
    auto [B, N, S, C] = ashape_arr;
    assert((N == 1));  // one sequence per batch

    // one gain parameter per channel
    assert((gamma->get_value().logical_shape().to_array_4D() == std::array<uint32_t, 4>{1, 1, 1, C}));

    auto device = &autograd::ctx().get_device();

    ttnn::Tensor squares = ttnn::square(tensor->get_value());  // [B,1,S,C] -> [B,1,S,C]

    ttnn::Tensor seq_means_of_squares = ttnn::mean(squares, /*dim_arg=*/-1, /*keep_dim=*/true);  // [B,1,S,1]

    ttnn::Tensor seq_means_of_squares_plus_epsilon =
        ttnn::experimental::add(seq_means_of_squares, epsilon);  // [B,1,S,1] x. [1] -> [B,1,S,1] (bcast)

    ttnn::Tensor rms_a = ttnn::sqrt(seq_means_of_squares_plus_epsilon);  // [B,1,S,1] -> [B,1,S,1]

    ttnn::Tensor gamma_times_activations =
        ttnn::experimental::mul(gamma->get_value(), tensor->get_value());  // [1,1,1,C] x [B,1,S,C] -> [B,1,S,C] (bcast)

    ttnn::Tensor out_tensor =
        ttnn::experimental::div(gamma_times_activations, rms_a);  // [B,1,S,C] x [B,1,S,C] -> [B,1,S,C]

    auto out = autograd::create_tensor(out_tensor);

    autograd::GradFunction grad = [B, S, C, tensor, gamma, out, rms_a, device]() {
        auto a = tensor->get_value();  // [B,1,S,C]
        auto g = gamma->get_value();   // [1,1,1,C]

        // c is the number of activations; in the RMS1orm paper they call this
        // "n". it is renamed here to avoid confusion with 1.
        auto c = static_cast<float>(a.logical_shape()[-1]);

        auto dL_dout = out->get_grad();  // Grad w.r.t normalized arctivations, hence [B,1,S,C]

        auto scaled_gain = ttnn::experimental::div(g, rms_a);  // [1,1,1,C] x [B,1,S,1] -> [B,1,S,C] (bcast)
        auto gained_dL_dout = ttnn::experimental::mul(scaled_gain, dL_dout);  // [B,1,S,C] x [B,1,S,C] -> [B,1,S,C]

        // notation:
        // _ · _ <- usual dot product
        // _ @ _ <- matrix multiplication
        // _ *. _ <- Hadamard product/eltwise multiplication with broadcasting
        // _ /. _ <- eltwise division with broadcasting

        // have a : [B,1,S,C]

        // want to obtain scaled_outer = gained_dL_dout @ ((a@a^T)/n*rms(a)^2)

        // to avoid computing the large outer product matrix explicitly, we
        // instead compute
        // scale = (a^T · gained_dL_dout) : [B,1,S,C] x [B,1,S,C] -> [1]
        // scaled_outer = scale *. a : [1] x [B,1,S,C] -> [B,1,S,C]

        auto scale = ttml::ttnn_fixed::sum_over_dim(
            ttnn::experimental::mul(a, gained_dL_dout), 3);  // [B,1,S,C] x [B,1,S,C] -> [B,1,S,C] -> [B,1,S,1]

        auto scaled_outer = ttnn::experimental::mul(scale, a);  // [B,1,S,1] x [B,1,S,C] -> [B,1,S,C] (bcast)

        auto ms_a = ttnn::square(rms_a);  // [B,1,S,1] -> [B,1,S,1]

        auto c_by_ms_a = ttnn::experimental::mul(ms_a, c);  // [B,1,S,1] x [1] -> [B,1,S,1] (bcast)

        auto rhs = ttnn::experimental::div(scaled_outer, c_by_ms_a);  // [B,1,S,C] x [B,1,S,1] -> [B,1,S,C] (bcast)

        auto dL_da =
            ttnn::experimental::sub(gained_dL_dout, rhs);  // [B,1,S,C] x [B,1,S,C] -> [B,1,S,C]; checked by add_grad
        tensor->add_grad(dL_da);

        // dL_dgamma = (a / rms(a)) * dL_dout -> requires sum over batch due to broadcasting
        auto dL_dg_components = ttnn::experimental::mul(
            dL_dout,
            ttnn::experimental::div(a, rms_a));  // [B,1,S,C] x [B,1,S,1] -> [B,1,S,C] (bcast); checked by add_grad
        auto dL_dg = ttnn::sum(
            dL_dg_components,
            /* dim_arg */ ttnn::SmallVector<int>{0, 1, 2},
            /* keep_dim */ true,
            /* output_mem_config */ std::nullopt,
            /*compute_kernel_config */ core::ComputeKernelConfig::precise());  // [B,1,S,C] -> [1,1,1,C]
        gamma->add_grad(dL_dg);
    };

    auto links = autograd::get_links(tensor, gamma);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
