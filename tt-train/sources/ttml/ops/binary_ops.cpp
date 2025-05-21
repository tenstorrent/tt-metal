// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ops.hpp"

#include <core/ttnn_all_includes.hpp>
#include <memory>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/binary_backward/binary_backward.hpp>
#include <ttnn/tensor/types.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

namespace {

bool was_broadcasted(const autograd::TensorPtr& input, const ttnn::Tensor& grad) {
    auto input_shape = input->get_value().logical_shape();
    auto grad_shape = grad.logical_shape();
    if (input_shape.rank() != grad_shape.rank()) {
        return false;
    }

    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (input_shape[i] != grad_shape[i]) {
            return true;
        }
    }

    return false;
}

ttnn::SmallVector<int64_t> get_broadcast_dimensions(const autograd::TensorPtr& input, const ttnn::Tensor& grad) {
    ttnn::SmallVector<int64_t> broadcast_dims;
    auto input_shape = input->get_value().logical_shape();
    auto grad_shape = grad.logical_shape();
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (input_shape[i] != grad_shape[i]) {
            broadcast_dims.push_back(static_cast<int64_t>(i));
        }
    }

    return broadcast_dims;
}

}  // namespace

autograd::TensorPtr operator+(const autograd::TensorPtr& a, const autograd::AutocastTensor& b) {
    auto out = autograd::create_tensor(ttnn::add(a->get_value(), b.get_tensor()));
    autograd::GradFunction grad = [a, out]() { a->add_grad(out->get_grad()); };
    auto links = autograd::get_links(a);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr operator+(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    auto out = autograd::create_tensor();

    constexpr tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> none{};
    out->set_value(
        ttnn::add(a->get_value(), b->get_value(), std::nullopt, std::nullopt, std::nullopt, none, none, none, false));
    autograd::GradFunction grad = [a, b, out]() {
        if (was_broadcasted(a, out->get_grad())) {
            a->add_grad(ttnn::moreh_sum(
                out->get_grad(),
                get_broadcast_dimensions(a, out->get_grad()),
                /* keep_dim */ true,
                /* output_tensor */ std::nullopt,
                /* memory_config_arg */ std::nullopt,
                core::ComputeKernelConfig::precise()));
        } else {
            a->add_grad(out->get_grad());
        }

        if (was_broadcasted(b, out->get_grad())) {
            b->add_grad(ttnn::moreh_sum(
                out->get_grad(),
                get_broadcast_dimensions(b, out->get_grad()),
                /* keep_dim */ true,
                /* output_tensor */ std::nullopt,
                /* memory_config_arg */ std::nullopt,
                core::ComputeKernelConfig::precise()));
        } else {
            b->add_grad(out->get_grad());
        }
    };
    auto links = autograd::get_links(a, b);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr operator-(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    auto out = autograd::create_tensor();

    out->set_value(ttnn::subtract(a->get_value(), b->get_value()));
    autograd::GradFunction grad = [a, b, out]() {
        tt::tt_metal::MemoryConfig mem_config;
        // TODO: support broadcasting
        a->add_grad(out->get_grad());
        b->add_grad(ttnn::neg(out->get_grad()));
    };
    auto links = autograd::get_links(a, b);

    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr operator*(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    auto out = autograd::create_tensor();

    out->set_value(ttnn::multiply(a->get_value(), b->get_value()));
    autograd::GradFunction grad = [a, b, out]() {
        tt::tt_metal::MemoryConfig mem_config;
        // TODO: support broadcasting (or not)
        auto a_grad = ttnn::multiply(out->get_grad(), b->get_value());
        auto b_grad = ttnn::multiply(out->get_grad(), a->get_value());

        a->add_grad(a_grad);
        b->add_grad(b_grad);
    };
    auto links = autograd::get_links(a, b);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr operator*(const autograd::TensorPtr& a, float b) {
    auto out = autograd::create_tensor(ttnn::multiply(a->get_value(), b));
    autograd::GradFunction grad = [a, b, out]() {
        auto a_grad = ttnn::multiply(out->get_grad(), b);

        a->add_grad(a_grad);
    };
    auto links = autograd::get_links(a);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr operator/(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    auto out = autograd::create_tensor();

    out->set_value(ttnn::divide(a->get_value(), b->get_value()));
    autograd::GradFunction grad = [a, b, out]() {
        auto res = ttnn::div_bw(out->get_grad(), a->get_value(), b->get_value());
        a->add_grad(res[0].value());
        b->add_grad(res[1].value());
    };
    auto links = autograd::get_links(a, b);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
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
