// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>

#include "nanobind/nb_export_enum.hpp"
#include "nanobind/nb_fwd.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/sgd.hpp"

namespace ttml::optimizers {

void py_module_types(nb::module_& m) {
    nb::class_<OptimizerBase>(m, "OptimizerBase");
    nb::class_<SGDConfig>(m, "SGDConfig");
    nb::class_<SGD, OptimizerBase>(m, "SGD");
    nb::class_<AdamWConfig>(m, "AdamWConfig");
    nb::class_<MorehAdamW, OptimizerBase>(m, "AdamW");
}

void py_module(nb::module_& m) {
    // {
    //     auto py_optimizer_base = static_cast<nb::class_<OptimizerBase>>(m.attr("OptimizerBase"));
    //     py_optimizer_base.def(nb::init<serialization::NamedParameters&&>());
    //     // TODO
    // }
    {
        auto py_sgd_config = static_cast<nb::class_<SGDConfig>>(m.attr("SGDConfig"));
        py_sgd_config.def_static(
            "make", [](float lr, float momentum, float dampening, float weight_decay, bool nesterov) {
                return SGDConfig{
                    .lr = lr,
                    .momentum = momentum,
                    .dampening = dampening,
                    .weight_decay = weight_decay,
                    .nesterov = nesterov};
            });

        auto py_sgd = static_cast<nb::class_<SGD, optimizers::OptimizerBase>>(m.attr("SGD"));
        py_sgd.def(nb::init<serialization::NamedParameters, const SGDConfig&>());
        py_sgd.def("zero_grad", &SGD::zero_grad);
        py_sgd.def("step", &SGD::step);
        py_sgd.def("get_state_dict", &SGD::get_state_dict);
        py_sgd.def("set_state_dict", &SGD::set_state_dict);

        auto py_adamw_config = static_cast<nb::class_<AdamWConfig>>(m.attr("AdamWConfig"));
        py_adamw_config.def_static("make", [](float lr, float beta1, float beta2, float epsilon, float weight_decay) {
            return AdamWConfig{
                .lr = lr, .beta1 = beta1, .beta2 = beta2, .epsilon = epsilon, .weight_decay = weight_decay};
        });

        auto py_adamw = static_cast<nb::class_<MorehAdamW, optimizers::OptimizerBase>>(m.attr("AdamW"));
        py_adamw.def(nb::init<serialization::NamedParameters, const AdamWConfig&>());
        py_adamw.def("zero_grad", &MorehAdamW::zero_grad);
        py_adamw.def("step", &MorehAdamW::step);
        py_adamw.def("get_state_dict", &MorehAdamW::get_state_dict);
        py_adamw.def("set_state_dict", &MorehAdamW::set_state_dict);
    }
}

}  // namespace ttml::optimizers
