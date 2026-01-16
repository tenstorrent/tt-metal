// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "autograd/autocast_tensor.hpp"
#include "modules/grouped_query_attention.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"
#include "serialization/serializable.hpp"

// Make NamedParameters opaque - must be before unordered_map include
NB_MAKE_OPAQUE(ttml::serialization::NamedParameters)

#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/unordered_map.h>

#include "nb_export_enum.hpp"
#include "nb_fwd.hpp"
#include "nb_util.hpp"

namespace ttml::nanobind::modules {
using namespace ttml::modules;

void py_module_types(nb::module_& m) {
    ttml::nanobind::util::export_enum<RunMode>(m);
    ttml::nanobind::util::export_enum<InferenceMode>(m);

    nb::class_<ModuleBase>(m, "ModuleBase");
    nb::class_<LinearLayer, ModuleBase>(m, "LinearLayer");
}

void py_module(nb::module_& m) {
    {
        auto py_module_base = static_cast<nb::class_<ModuleBase>>(m.attr("ModuleBase"));
        py_module_base.def(nb::init<>());
        py_module_base.def(nb::init<const ModuleBase&>());
        py_module_base.def(nb::init<ModuleBase&&>());
        py_module_base.def("get_name", &ModuleBase::get_name, "Get name");
        py_module_base.def("parameters", &ModuleBase::parameters, "Get parameters");
        py_module_base.def("train", &ModuleBase::train, "Set mode to train");
        py_module_base.def("eval", &ModuleBase::eval, "Set mode to eval");
        py_module_base.def("set_run_mode", &ModuleBase::set_run_mode, "Set run mode");
        py_module_base.def("get_run_mode", &ModuleBase::get_run_mode, "Get run mode");
        py_module_base.def(
            "__call__",
            static_cast<autograd::TensorPtr (ModuleBase::*)(const autograd::TensorPtr&)>(&ModuleBase::operator()),
            nb::arg("tensor"));
        py_module_base.def(
            "__call__",
            static_cast<autograd::TensorPtr (ModuleBase::*)(const autograd::TensorPtr&, const autograd::TensorPtr&)>(
                &ModuleBase::operator()),
            nb::arg("tensor"),
            nb::arg("other"));
    }

    {
        auto py_linear_layer = static_cast<nb::class_<LinearLayer, ModuleBase>>(m.attr("LinearLayer"));
        py_linear_layer.def(
            nb::init<uint32_t, uint32_t, bool>(),
            nb::arg("in_features"),
            nb::arg("out_features"),
            nb::arg("has_bias") = true);
        py_linear_layer.def(
            nb::init<const autograd::TensorPtr&, const autograd::TensorPtr&>(), nb::arg("weight"), nb::arg("bias"));
        py_linear_layer.def(
            nb::init<const autograd::TensorPtr&, bool>(), nb::arg("weight"), nb::arg("has_bias") = true);
        py_linear_layer.def("get_weight", &LinearLayer::get_weight, "Get weight");
        py_linear_layer.def(
            "get_weight_numpy",
            [](const LinearLayer& layer) {
                auto const w = layer.get_weight();
                return ttml::nanobind::util::make_numpy_tensor(w->get_value(autograd::PreferredPrecision::FULL));
            },
            "Get weight as numpy tensor");
    }
}

}  // namespace ttml::nanobind::modules
