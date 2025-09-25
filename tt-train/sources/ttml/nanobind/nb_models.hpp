// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nb_fwd.hpp"

namespace ttml::nanobind::models {

void py_module_types(nb::module_& m, nb::module_& m_modules);
void py_module(nb::module_& m, nb::module_& m_modules);

}  // namespace ttml::nanobind::models
