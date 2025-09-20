// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "models/gpt2.hpp"
#include "models/linear_regression.hpp"
#include "nb_fwd.hpp"

namespace ttml::models {

void py_module_types(nb::module_& m) {
    m.def("create_linear_regression_model", &linear_regression::create);
    // m.def("load_gpt2_model_from_safetensors", &gpt2::load_model_from_safetensors);
}

void py_module(nb::module_& m) {
}

}  // namespace ttml::models
