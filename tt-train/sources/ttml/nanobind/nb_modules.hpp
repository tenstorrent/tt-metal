// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nb_fwd.hpp"

namespace ttml::nanobind::modules {

void py_module_types(nb::module_& m);
void py_module(nb::module_& m);

}  // namespace ttml::nanobind::modules
