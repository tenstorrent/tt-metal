// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

struct Tensor {};
struct LayerNorm {};

NB_MODULE(_op_documenter_bindings, m) {
    nb::class_<Tensor>(m, "Tensor").def(nb::init<>());
    nb::class_<LayerNorm>(m, "LayerNorm")
        .def(nb::init<>())
        .def(
            "__call__",
            [](LayerNorm&, const Tensor&, double, nb::object, nb::object) { return Tensor(); },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("epsilon") = 1e-12,
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("recip_tensor") = nb::none());

    m.def("overloaded", [](int value) { return value; }, nb::arg("value") = 0);
    m.def("overloaded", [](const std::string& value) { return value; }, nb::arg("value") = "literal\\0");
    m.def("custom_default", [](int value) { return value; }, nb::arg("value").sig("DEFAULT_VALUE") = 4);
}
