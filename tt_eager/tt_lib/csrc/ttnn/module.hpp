// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

#include "tensor/module.hpp"

namespace py = pybind11;

namespace ttnn {

void py_module(py::module& m_ttnn) {
    auto m_tensor = m_ttnn.def_submodule("tensor", "Tensor");
    tensor::py_module(m_tensor);
}

}  // namespace ttnn
