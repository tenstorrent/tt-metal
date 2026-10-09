// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_rmsnorm_op.hpp"

#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "metal/ops/gated_rmsnorm/gated_rmsnorm.hpp"

namespace ttml::ops {

autograd::TensorPtr gated_rmsnorm(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& gate,
    const autograd::TensorPtr& gamma,
    const float epsilon) {
    auto out = autograd::create_tensor(
        ttml::metal::gated_rmsnorm_fw(input->get_value(), gate->get_value(), gamma->get_value(), epsilon));

    autograd::GradFunction grad = [input, gate, gamma, out, epsilon]() {
        auto [dx, dgate, dgamma] = ttml::metal::gated_rmsnorm_bw(
            input->get_value(),
            gate->get_value(),
            gamma->get_value(),
            out->get_grad(),
            epsilon,
            /* compute_dgamma */ gamma->get_requires_grad());
        input->add_grad(dx);
        gate->add_grad(dgate);
        if (dgamma.has_value()) {
            gamma->add_grad(dgamma.value());
        }
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, input, gate, gamma));
    return out;
}

}  // namespace ttml::ops
