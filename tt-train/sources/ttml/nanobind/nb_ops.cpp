// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>

#include "nb_export_enum.hpp"
#include "nb_fwd.hpp"
#include "ops/losses.hpp"

namespace ttml::ops {

void py_module_types(nb::module_& m) {
    nb::export_enum<ReduceType>(m);
    m.def("cross_entropy_loss", &ttml::ops::cross_entropy_loss);
    m.def("mse_loss", &ttml::ops::mse_loss);
}

void py_module(nb::module_& m) {
}

}  // namespace ttml::ops
