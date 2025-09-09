// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttml-pybind/__init__.hpp"

#include <pybind11/pybind11.h>

namespace ttml {

void py_module(py::module& module) {
}

}  // namespace ttml

PYBIND11_MODULE(_ttnn, module) {
    module.doc() = "Python bindings for TTML";
    module.def("hi", []() { return "hi"; });
}
