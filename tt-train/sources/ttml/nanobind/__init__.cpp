// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "__init__.hpp"

#include <nanobind/nanobind.h>

#include "nb_autograd.hpp"

NB_MODULE(_ttml, m) {
    auto m_autograd = m.def_submodule("autograd", "autograd");

    // TYPES
    ttml::autograd::py_module_types(m_autograd);

    // FUNCTIONS / OPERATIONS
    ttml::autograd::py_module(m_autograd);
}
