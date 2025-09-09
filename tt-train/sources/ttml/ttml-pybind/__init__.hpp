// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

namespace ttml {
namespace py = pybind11;

void py_module(py::module_& m);

}  // namespace ttml
