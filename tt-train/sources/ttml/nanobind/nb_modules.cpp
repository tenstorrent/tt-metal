// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/autocast_tensor.hpp"
#include "autograd/tensor.hpp"
#include "modules/grouped_query_attention.hpp"
#include "modules/module_base.hpp"
#include "serialization/serializable.hpp"

// Make NamedParameters opaque - must be before unordered_map include
NB_MAKE_OPAQUE(ttml::serialization::NamedParameters)

#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/unordered_map.h>

#include "nb_export_enum.hpp"
#include "nb_fwd.hpp"
#include "nb_util.hpp"

// Trampoline class to enable Python subclassing of ModuleBase
// Must be outside ttml::nanobind namespace so NB_OVERRIDE macros resolve to ::nanobind
struct PyModuleBase : ttml::modules::ModuleBase {
    NB_TRAMPOLINE(ttml::modules::ModuleBase, 2);  // 2 virtual methods to override

    // Make protected methods accessible
    using ttml::modules::ModuleBase::create_name;
    using ttml::modules::ModuleBase::override_module;
    using ttml::modules::ModuleBase::override_tensor;
    using ttml::modules::ModuleBase::register_module;
    using ttml::modules::ModuleBase::register_tensor;

    // Override virtual operator() to allow Python implementation
    // Use NB_OVERRIDE_NAME to map C++ operator() to Python __call__
    ttml::autograd::TensorPtr operator()(const ttml::autograd::TensorPtr& tensor) override {
        NB_OVERRIDE_NAME("__call__", operator(), tensor);
    }

    ttml::autograd::TensorPtr operator()(
        const ttml::autograd::TensorPtr& tensor, const ttml::autograd::TensorPtr& other) override {
        NB_OVERRIDE_NAME("__call__", operator(), tensor, other);
    }
};

namespace ttml::nanobind::modules {
using namespace ttml::modules;

void py_module_types(nb::module_& m) {
    ttml::nanobind::util::export_enum<RunMode>(m);
    ttml::nanobind::util::export_enum<InferenceMode>(m);

    // Enable Python subclassing via PyModuleBase trampoline
    nb::class_<ModuleBase, PyModuleBase>(m, "ModuleBase");
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

        // Expose registration methods for Python subclassing
        py_module_base.def("create_name", &PyModuleBase::create_name, nb::arg("name"), "Set the module's name");
        py_module_base.def(
            "register_tensor",
            &PyModuleBase::register_tensor,
            nb::arg("tensor"),
            nb::arg("name"),
            "Register a tensor parameter");
        py_module_base.def(
            "register_module",
            &PyModuleBase::register_module,
            nb::arg("module"),
            nb::arg("name"),
            "Register a submodule");
        py_module_base.def(
            "override_tensor",
            &PyModuleBase::override_tensor,
            nb::arg("tensor"),
            nb::arg("name"),
            "Override an existing tensor parameter");
        py_module_base.def(
            "override_module",
            &PyModuleBase::override_module,
            nb::arg("module"),
            nb::arg("name"),
            "Override an existing submodule");
    }
}

}  // namespace ttml::nanobind::modules
