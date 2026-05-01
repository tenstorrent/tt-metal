// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nb_fwd.hpp"

namespace ttml::nanobind::schedulers {

void py_module_types(nb::module_& m);
void py_module(nb::module_& m);

}  // namespace ttml::nanobind::schedulers
