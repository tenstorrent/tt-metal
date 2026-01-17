// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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

        // Pickle support for LinearLayer
        py_linear_layer.def(
            "__getstate__",
            [](const LinearLayer& layer) {
                // Get parameters from the layer
                auto params = layer.parameters();

                // Extract weight tensor
                auto weight_it = params.find(layer.get_name() + "/weight");
                if (weight_it == params.end()) {
                    throw std::runtime_error("LinearLayer weight not found in parameters");
                }
                auto weight_numpy = ttml::nanobind::util::make_numpy_tensor(
                    weight_it->second->get_value(autograd::PreferredPrecision::FULL));

                // Check for bias
                auto bias_it = params.find(layer.get_name() + "/bias");
                bool has_bias = (bias_it != params.end());

                nb::dict state;
                state["weight"] = weight_numpy;
                state["has_bias"] = has_bias;
                if (has_bias) {
                    auto bias_numpy = ttml::nanobind::util::make_numpy_tensor(
                        bias_it->second->get_value(autograd::PreferredPrecision::FULL));
                    state["bias"] = bias_numpy;
                }

                return state;
            },
            "Serialize LinearLayer state for pickling");

        py_linear_layer.def(
            "__setstate__",
            [](LinearLayer& layer, nb::dict state) {
                // Extract weight from state
                auto weight_numpy = nb::cast<nb::ndarray<nb::numpy>>(state["weight"]);
                bool has_bias = nb::cast<bool>(state["has_bias"]);

                // Create weight tensor from numpy
                auto weight_tensor = autograd::create_tensor(ttml::nanobind::util::make_metal_tensor(
                    weight_numpy, tt::tt_metal::Layout::TILE, std::nullopt, nullptr));

                if (has_bias) {
                    // Create bias tensor from numpy
                    auto bias_numpy = nb::cast<nb::ndarray<nb::numpy>>(state["bias"]);
                    auto bias_tensor = autograd::create_tensor(ttml::nanobind::util::make_metal_tensor(
                        bias_numpy, tt::tt_metal::Layout::TILE, std::nullopt, nullptr));

                    // Use placement new to reconstruct the layer in-place
                    new (&layer) LinearLayer(weight_tensor, bias_tensor);
                } else {
                    new (&layer) LinearLayer(weight_tensor, false);
                }
            },
            "Deserialize LinearLayer state from pickle");
    }
}

}  // namespace ttml::nanobind::modules
