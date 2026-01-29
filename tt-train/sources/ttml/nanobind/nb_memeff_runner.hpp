// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

#include "autograd/tensor.hpp"
#include "nb_fwd.hpp"

namespace ttml::nanobind::detail {

// C++ signature of forward_impl, passed into memory_efficient_runner
using ForwardFunc = autograd::TensorPtr(const autograd::TensorPtr&, std::optional<autograd::TensorPtr>);

// Python forward_impl representation, passed into memory_efficient_runner from Python
using ForwardPyCallable = nb::typed<nb::callable, ForwardFunc>;

// Tweaked version of the native C++ memory_efficient_runner
// that properly handles Python callable as forward_impl
autograd::TensorPtr memory_efficient_runner(
    ForwardPyCallable forward_impl, const autograd::TensorPtr& input, std::optional<autograd::TensorPtr> mask);

}  // namespace ttml::nanobind::detail
