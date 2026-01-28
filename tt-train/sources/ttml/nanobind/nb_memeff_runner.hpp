// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

#include "autograd/tensor.hpp"
#include "nb_fwd.hpp"

namespace ttml::nanobind {

using ForwardFunc = autograd::TensorPtr(const autograd::TensorPtr&, std::optional<autograd::TensorPtr>);
using ForwardPyCallable = nb::typed<nb::callable, ForwardFunc>;

autograd::TensorPtr memory_efficient_runner(
    ForwardPyCallable forward_impl, const autograd::TensorPtr& input, std::optional<autograd::TensorPtr> mask);

}  // namespace ttml::nanobind
