// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nb_memeff_runner.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>

#include <core/ttnn_all_includes.hpp>
#include <memory>
#include <utility>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/scoped.hpp"

namespace ttml::nanobind {

namespace {

autograd::TensorPtr call_forward(
    const ForwardPyCallable& fn, const autograd::TensorPtr& x, std::optional<autograd::TensorPtr> m) {
    nb::gil_scoped_acquire guard;
    return nb::cast<autograd::TensorPtr>(fn(x, m));
}

}  // anonymous namespace

autograd::TensorPtr memory_efficient_runner(
    ForwardPyCallable forward_impl, const autograd::TensorPtr& input, std::optional<autograd::TensorPtr> mask) {
    if (autograd::ctx().get_gradient_mode() == autograd::GradMode::DISABLED) {
        return call_forward(forward_impl, input, mask);
    }

    auto forward_impl_ptr = std::make_shared<ForwardPyCallable>(std::move(forward_impl));

    // make a copy of a generator before running forward pass
    auto generator = autograd::ctx().get_generator();

    // running forward pass
    autograd::TensorPtr out;
    {
        auto scoped = ttml::core::Scoped(
            []() { autograd::ctx().set_gradient_mode(autograd::GradMode::DISABLED); },
            []() { autograd::ctx().set_gradient_mode(autograd::GradMode::ENABLED); });
        out = call_forward(*forward_impl_ptr, input, mask);
    }

    // define grad function and copy generator (in the state before forward pass)
    autograd::GradFunction grad = [input, mask, out, forward_impl_ptr, generator]() {
        // detach input from existing graph
        auto input_detached = autograd::create_tensor(input->get_value());
        // run forward pass again
        autograd::TensorPtr output;
        {
            // set generator to the state before forward pass during construction
            // restore generator state after grad function is executed
            auto scoped = ttml::core::Scoped(
                [&generator]() { autograd::ctx().set_generator(generator); },
                [generator = autograd::ctx().get_generator()]() { autograd::ctx().set_generator(generator); });
            output = call_forward(*forward_impl_ptr, input_detached, mask);
        }
        // use gradients from new output
        output->set_grad(out->get_grad());
        output->backward();
        // reuse gradients from detached input
        input->add_grad(input_detached->get_grad());
    };

    auto links = autograd::get_links(input);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::nanobind
