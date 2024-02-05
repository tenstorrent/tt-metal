// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

#include "binary.hpp"

namespace py = pybind11;

namespace ttnn {

namespace operations {

void py_module(py::module& m_operations) {
    auto m_binary = m_operations.def_submodule("binary", "binary operations");
    binary::py_module(m_binary);
}

}  // namespace operations

}  // namespace ttnn
