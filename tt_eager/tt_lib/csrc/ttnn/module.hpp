// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

#include "operations/module.hpp"
#include "types.hpp"

namespace py = pybind11;

namespace ttnn {

void py_module(py::module& m_ttnn) {
    auto m_types = m_ttnn.def_submodule("types", "ttnn Types");
    types::py_module(m_types);

    auto m_operations = m_ttnn.def_submodule("operations", "ttnn Operations");
    operations::py_module(m_operations);
}

}  // namespace ttnn
