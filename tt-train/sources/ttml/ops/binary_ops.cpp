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
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr operator+(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    auto out = autograd::create_tensor();

    out->set_value(ttnn::add(a->get_value(), b->get_value()));
    autograd::GradFunction grad = [a, b, out]() {
        a->add_grad(out->get_grad());
        b->add_grad(out->get_grad());
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

autograd::TensorPtr operator/(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    auto out = autograd::create_tensor();

    out->set_value(ttnn::divide(a->get_value(), b->get_value()));
    autograd::GradFunction grad = [a, b, out]() {
        auto res = ttnn::div_bw(out->get_grad(), a->get_value(), b->get_value(), "None");
        a->add_grad(res[0].value());
        b->add_grad(res[1].value());
    };
    auto links = autograd::get_links(a, b);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
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

}  // namespace ttml::ops
