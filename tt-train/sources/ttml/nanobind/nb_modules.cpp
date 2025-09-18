// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "models/gpt2.hpp"
#include "models/linear_regression.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"
#include "nb_export_enum.hpp"
#include "nb_fwd.hpp"
#include "nb_util.hpp"

namespace ttml::modules {

void py_module_types(nb::module_& m) {
    nb::export_enum<RunMode>(m);

    nb::class_<ModuleBase>(m, "ModuleBase");
    nb::class_<LinearLayer, ModuleBase>(m, "LinearLayer");
}

void py_module(nb::module_& m) {
    {
        auto py_module_base = static_cast<nb::class_<ModuleBase>>(m.attr("ModuleBase"));
        py_module_base.def(nb::init<>());
        py_module_base.def(nb::init<const ModuleBase&>());
        py_module_base.def(nb::init<ModuleBase&&>());
        py_module_base.def("get_name", &ModuleBase::get_name);
        py_module_base.def("parameters", &ModuleBase::parameters);
        py_module_base.def("train", &ModuleBase::train);
        py_module_base.def("eval", &ModuleBase::eval);
        py_module_base.def("set_run_mode", &ModuleBase::set_run_mode);
        py_module_base.def("get_run_mode", &ModuleBase::get_run_mode);
    }

    {
        auto py_linear_layer = static_cast<nb::class_<LinearLayer, ModuleBase>>(m.attr("LinearLayer"));
        py_linear_layer.def(nb::init<uint32_t, uint32_t, bool>());
        py_linear_layer.def("__call__", &LinearLayer::operator());
        py_linear_layer.def("get_weight", &LinearLayer::get_weight);
        py_linear_layer.def("get_weight_numpy", [](const LinearLayer& layer) {
            auto const w = layer.get_weight();
            return make_numpy_tensor(w->get_value(autograd::PreferredPrecision::FULL));
        });
    }

    m.def("create_linear_regression_model", &models::linear_regression::create);
    m.def("load_gpt2_model_from_safetensors", &models::gpt2::load_model_from_safetensors);
}

}  // namespace ttml::modules
