// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ops.hpp"

#include <memory>
#include <stdexcept>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/binary_backward/binary_backward.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

namespace {

bool was_broadcasted(const autograd::TensorPtr& input, const ttnn::Tensor& grad) {
    auto input_shape = input->get_value().logical_shape();
    auto grad_shape = grad.logical_shape();

    // Rank mismatch means the input was broadcast with implicit leading 1s.
    if (input_shape.rank() != grad_shape.rank()) {
        return true;
    }

    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (input_shape[i] != grad_shape[i]) {
            return true;
        }
    }

    return false;
}

// Reduce grad along all broadcast dimensions and reshape back to the input's
// original shape.  Handles both same-rank broadcasting (e.g. [1,1,1,1] vs
// [1,1,1,4]) and cross-rank broadcasting (e.g. [4] vs [2,1,1,4]) by aligning
// shapes from the right and left-padding with implicit 1s.
ttnn::Tensor unbroadcast_grad(const autograd::TensorPtr& input, const ttnn::Tensor& grad) {
    auto input_shape = input->get_value().logical_shape();
    auto grad_shape = grad.logical_shape();
    auto input_rank = input_shape.rank();
    auto grad_rank = grad_shape.rank();
    auto max_rank = std::max(input_rank, grad_rank);

    ttsl::SmallVector<int64_t> broadcast_dims;
    for (size_t i = 0; i < max_rank; ++i) {
        auto input_dim = (i < max_rank - input_rank) ? 1u : input_shape[i - (max_rank - input_rank)];
        auto grad_dim = (i < max_rank - grad_rank) ? 1u : grad_shape[i - (max_rank - grad_rank)];
        if (input_dim != grad_dim) {
            broadcast_dims.push_back(static_cast<int64_t>(i));
        }
    }

    auto reduced = ttnn::moreh_sum(
        grad,
        broadcast_dims,
        /* keep_dim */ true,
        /* output_tensor */ std::nullopt,
        /* memory_config_arg */ std::nullopt,
        core::ComputeKernelConfig::precise());

    // When ranks differ, moreh_sum with keep_dim preserves the grad's rank.
    // Reshape back to the input's original shape so add_grad doesn't throw.
    if (input_rank != grad_rank) {
        reduced = ttnn::reshape(reduced, input_shape);
    }

    return reduced;
}

}  // namespace

autograd::TensorPtr operator+(const autograd::TensorPtr& a, const ttnn::Tensor& b) {
    auto out = autograd::create_tensor(ttnn::add(a->get_value(), b));
    autograd::GradFunction grad = [a, out]() { a->add_grad(out->get_grad()); };
    out->set_node(autograd::add_backward_node(std::move(grad), out, a));
    return out;
}

autograd::TensorPtr operator+(const autograd::TensorPtr& a, const autograd::AutocastTensor& b) {
    auto out = autograd::create_tensor(ttnn::add(a->get_value(), b.get_tensor()));
    autograd::GradFunction grad = [a, out]() { a->add_grad(out->get_grad()); };
    out->set_node(autograd::add_backward_node(std::move(grad), out, a));
    return out;
}

autograd::TensorPtr operator+(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    auto out = autograd::create_tensor();

    constexpr ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> none{};
    out->set_value(
        ttnn::add(a->get_value(), b->get_value(), std::nullopt, std::nullopt, std::nullopt, none, none, none, false));
    autograd::GradFunction grad = [a, b, out]() {
        if (was_broadcasted(a, out->get_grad())) {
            a->add_grad(unbroadcast_grad(a, out->get_grad()));
        } else {
            a->add_grad(out->get_grad());
        }

        if (was_broadcasted(b, out->get_grad())) {
            b->add_grad(unbroadcast_grad(b, out->get_grad()));
        } else {
            b->add_grad(out->get_grad());
        }
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, a, b));

    return out;
}

autograd::TensorPtr operator-(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    auto out = autograd::create_tensor();

    out->set_value(ttnn::subtract(a->get_value(), b->get_value()));
    autograd::GradFunction grad = [a, b, out]() {
        if (was_broadcasted(a, out->get_grad())) {
            a->add_grad(unbroadcast_grad(a, out->get_grad()));
        } else {
            a->add_grad(out->get_grad());
        }

        auto neg_grad = ttnn::neg(out->get_grad());
        if (was_broadcasted(b, neg_grad)) {
            b->add_grad(unbroadcast_grad(b, neg_grad));
        } else {
            b->add_grad(neg_grad);
        }
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, a, b));

    return out;
}

autograd::TensorPtr operator*(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    auto out = autograd::create_tensor();

    out->set_value(ttnn::multiply(
        a->get_value(),
        b->get_value(),
        /* fast_and_approximate_mode*/ true));
    autograd::GradFunction grad = [a, b, out]() {
        auto a_grad = ttnn::multiply(
            out->get_grad(),
            b->get_value(),
            /* fast_and_approximate_mode*/ true);
        auto b_grad = ttnn::multiply(
            out->get_grad(),
            a->get_value(),
            /* fast_and_approximate_mode*/ true);

        if (was_broadcasted(a, a_grad)) {
            a->add_grad(unbroadcast_grad(a, a_grad));
        } else {
            a->add_grad(a_grad);
        }

        if (was_broadcasted(b, b_grad)) {
            b->add_grad(unbroadcast_grad(b, b_grad));
        } else {
            b->add_grad(b_grad);
        }
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, a, b));

    return out;
}

autograd::TensorPtr operator*(const autograd::TensorPtr& a, float b) {
    auto out = autograd::create_tensor(ttnn::multiply(a->get_value(), b));
    autograd::GradFunction grad = [a, b, out]() {
        auto a_grad = ttnn::multiply(out->get_grad(), b, /* fast_and_approximate_mode*/ true);

        a->add_grad(a_grad);
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, a));

    return out;
}

autograd::TensorPtr operator/(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    auto out = autograd::create_tensor();

    out->set_value(ttnn::divide(a->get_value(), b->get_value()));
    autograd::GradFunction grad = [a, b, out]() {
        if (was_broadcasted(a, out->get_grad()) || was_broadcasted(b, out->get_grad())) {
            throw std::runtime_error("Broadcasting is not supported in the backward pass of operator/");
        }
        auto res = ttnn::div_bw(out->get_grad(), a->get_value(), b->get_value());
        a->add_grad(res[0].value());
        b->add_grad(res[1].value());
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, a, b));

    return out;
}

autograd::TensorPtr add(const autograd::TensorPtr& a, const ttnn::Tensor& b) {
    return a + b;
}

autograd::TensorPtr add(const autograd::TensorPtr& a, const autograd::AutocastTensor& b) {
    return a + b;
}

autograd::TensorPtr add(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    return a + b;
}

autograd::TensorPtr sub(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    return a - b;
}

autograd::TensorPtr mul(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    return a * b;
}

autograd::TensorPtr div(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    return a / b;
}

autograd::TensorPtr mul(const autograd::TensorPtr& a, float b) {
    return a * b;
}

}  // namespace ttml::ops
