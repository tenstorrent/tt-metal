// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "serialization/serializable.hpp"

// Make NamedParameters opaque - must be before unordered_map include
NB_MAKE_OPAQUE(ttml::serialization::NamedParameters)

#include <nanobind/stl/unordered_map.h>

#include "nanobind/nb_export_enum.hpp"
#include "nanobind/nb_fwd.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/optimizer_base.hpp"
#include "optimizers/sgd.hpp"

namespace ttml::nanobind::optimizers {
using namespace ttml::optimizers;

void py_module_types(nb::module_& m) {
    nb::class_<OptimizerBase>(m, "OptimizerBase");
    nb::class_<SGDConfig>(m, "SGDConfig");
    nb::class_<SGD, OptimizerBase>(m, "SGD");
    nb::class_<AdamWConfig>(m, "AdamWConfig");
    nb::class_<MorehAdamW, OptimizerBase>(m, "AdamW");
}

void py_module(nb::module_& m) {
    {
        auto py_optimizer_base = static_cast<nb::class_<OptimizerBase>>(m.attr("OptimizerBase"));
        py_optimizer_base.def("zero_grad", &OptimizerBase::zero_grad, "Zero out gradient");
        py_optimizer_base.def("step", &OptimizerBase::step, "Step function");
        py_optimizer_base.def("get_state_dict", &OptimizerBase::get_state_dict, "Get state dictionary");
        py_optimizer_base.def(
            "set_state_dict", &OptimizerBase::set_state_dict, nb::arg("dict"), "Set state dictionary");
        py_optimizer_base.def("get_lr", &OptimizerBase::get_lr, "Get learning rate");
        py_optimizer_base.def("set_lr", &OptimizerBase::set_lr, nb::arg("lr"), "Set learning rate");
        py_optimizer_base.def("print_stats", &OptimizerBase::print_stats, "Print statistics");
    }

    {
        auto py_sgd_config = static_cast<nb::class_<SGDConfig>>(m.attr("SGDConfig"));
        py_sgd_config.def(nb::init<>());
        py_sgd_config.def_static(
            "make",
            [](float lr, float momentum, float dampening, float weight_decay, bool nesterov) {
                return SGDConfig{
                    .lr = lr,
                    .momentum = momentum,
                    .dampening = dampening,
                    .weight_decay = weight_decay,
                    .nesterov = nesterov};
            },
            nb::arg("lr"),
            nb::arg("momentum"),
            nb::arg("dampening"),
            nb::arg("weight_decay"),
            nb::arg("nesterov"),
            "Create a SGDConfig object");
    }

    {
        auto py_sgd = static_cast<nb::class_<SGD, OptimizerBase>>(m.attr("SGD"));
        py_sgd.def(
            nb::init<serialization::NamedParameters, const SGDConfig&>(), nb::arg("parameters"), nb::arg("config"));
    }

    {
        auto py_adamw_config = static_cast<nb::class_<AdamWConfig>>(m.attr("AdamWConfig"));
        py_adamw_config.def(nb::init<>());
        py_adamw_config.def_static(
            "make",
            [](float lr, float beta1, float beta2, float epsilon, float weight_decay) {
                return AdamWConfig{
                    .lr = lr, .beta1 = beta1, .beta2 = beta2, .epsilon = epsilon, .weight_decay = weight_decay};
            },
            nb::arg("lr"),
            nb::arg("beta1"),
            nb::arg("beta2"),
            nb::arg("epsilon"),
            nb::arg("weight_decay"),
            "Make an AdamWConfig object");
    }

    {
        auto py_adamw = static_cast<nb::class_<MorehAdamW, OptimizerBase>>(m.attr("AdamW"));
        py_adamw.def(
            nb::init<serialization::NamedParameters, const AdamWConfig&>(), nb::arg("parameters"), nb::arg("config"));
    }
}

}  // namespace ttml::nanobind::optimizers
