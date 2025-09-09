// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// #include "ttnn-pybind/pybind_fwd.hpp"
#include <pybind11/pybind11.h>

namespace ttml::autograd {
namespace py = pybind11;

void py_autograd_module_types(py::module_& m);
void py_autograd_module(py::module_& m);

}  // namespace ttml::autograd
