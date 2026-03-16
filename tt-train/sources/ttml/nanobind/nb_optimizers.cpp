// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>
#include <nanobind/trampoline.h>

#include "serialization/serializable.hpp"

// Make NamedParameters opaque - must be before unordered_map include
NB_MAKE_OPAQUE(ttml::serialization::NamedParameters)

#include <nanobind/stl/pair.h>
#include <nanobind/stl/unordered_map.h>
#include <yaml-cpp/yaml.h>

#include "nanobind/nb_export_enum.hpp"
#include "nanobind/nb_fwd.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/adamw_full_precision.hpp"
#include "optimizers/muon.hpp"
#include "optimizers/no_op.hpp"
#include "optimizers/optimizer_base.hpp"
#include "optimizers/optimizer_registry.hpp"
#include "optimizers/remote_optimizer.hpp"
#include "optimizers/sgd.hpp"

class OptimizerBaseTrampoline : public ttml::optimizers::OptimizerBase {
    NB_TRAMPOLINE(OptimizerBase, 10);

public:
    std::string get_name() const override {
        NB_OVERRIDE_PURE(get_name);
    }
    void zero_grad() override {
        NB_OVERRIDE_PURE(zero_grad);
    }
    void step() override {
        NB_OVERRIDE_PURE(step);
    }
    ttml::serialization::StateDict get_state_dict() const override {
        NB_OVERRIDE_PURE(get_state_dict);
    }
    void set_state_dict(const ttml::serialization::StateDict& dict) override {
        NB_OVERRIDE_PURE(set_state_dict, dict);
    }
    size_t get_steps() const override {
        NB_OVERRIDE_PURE(get_steps);
    }
    void set_steps(size_t steps) override {
        NB_OVERRIDE_PURE(set_steps, steps);
    }
    void set_lr(float lr) override {
        NB_OVERRIDE_PURE(set_lr, lr);
    }
    float get_lr() const override {
        NB_OVERRIDE_PURE(get_lr);
    }
    void print_stats() const override {
        NB_OVERRIDE(print_stats);
    }
};

namespace ttml::nanobind::optimizers {
using namespace ttml::optimizers;

void py_module_types(nb::module_& m) {
    nb::class_<OptimizerBase, OptimizerBaseTrampoline>(m, "OptimizerBase");
    nb::class_<SGDConfig>(m, "SGDConfig");
    nb::class_<SGD, OptimizerBase>(m, "SGD");
    nb::class_<AdamWConfig>(m, "AdamWConfig");
    nb::class_<AdamWFullPrecisionConfig>(m, "AdamWFullPrecisionConfig");
    nb::class_<AdamW, OptimizerBase>(m, "AdamW");
    nb::class_<AdamWFullPrecision, OptimizerBase>(m, "AdamWFullPrecision");
    nb::class_<MuonConfig>(m, "MuonConfig");
    nb::class_<MuonComposite, OptimizerBase>(m, "MuonComposite");
    nb::class_<NoOp, OptimizerBase>(m, "NoOp");
    nb::class_<RemoteOptimizer, OptimizerBase>(m, "RemoteOptimizer");
}

namespace {

YAML::Node obj_to_yaml(nb::object obj) {
    if (nb::isinstance<nb::dict>(obj)) {
        YAML::Node node;
        for (auto [key, val] : nb::cast<nb::dict>(obj)) {
            node[nb::cast<std::string>(key)] = obj_to_yaml(nb::borrow(val));
        }
        return node;
    }
    if (nb::isinstance<nb::list>(obj)) {
        YAML::Node node;
        for (auto item : nb::cast<nb::list>(obj)) {
            node.push_back(obj_to_yaml(nb::borrow(item)));
        }
        return node;
    }
    if (nb::isinstance<nb::bool_>(obj)) {
        return YAML::Node(nb::cast<bool>(obj));
    }
    if (nb::isinstance<nb::int_>(obj)) {
        return YAML::Node(nb::cast<int64_t>(obj));
    }
    if (nb::isinstance<nb::float_>(obj)) {
        return YAML::Node(nb::cast<double>(obj));
    }
    if (nb::isinstance<nb::str>(obj)) {
        return YAML::Node(nb::cast<std::string>(obj));
    }
    return YAML::Node();
}

}  // namespace

void py_module(nb::module_& m) {
    // Python-side registry for optimizers defined in Python.
    // Kept separate from the C++ registry to avoid unique_ptr ownership issues
    // (Python-constructed trampoline objects can't be transferred to C++ unique_ptr).
    nb::dict py_registry;
    m.attr("_py_optimizer_registry") = py_registry;

    m.def(
        "create_optimizer",
        [py_registry](nb::dict config, serialization::NamedParameters params) -> nb::object {
            auto type = nb::cast<std::string>(config.attr("get")("type", "AdamW"));
            if (py_registry.contains(type.c_str())) {
                return py_registry[type.c_str()](config, std::move(params));
            }
            return nb::cast(create_optimizer(obj_to_yaml(config), std::move(params)));
        },
        nb::arg("config"),
        nb::arg("params"),
        "Create an optimizer from a config dict and named parameters");

    m.def(
        "register_optimizer",
        [py_registry](const std::string& type, nb::object creator) { py_registry[type.c_str()] = creator; },
        nb::arg("type"),
        nb::arg("creator"),
        "Register a custom optimizer creator function");

    {
        auto py_optimizer_base =
            static_cast<nb::class_<OptimizerBase, OptimizerBaseTrampoline>>(m.attr("OptimizerBase"));
        py_optimizer_base.def(nb::init<serialization::NamedParameters>(), nb::arg("parameters"));
        py_optimizer_base.def("get_name", &OptimizerBase::get_name, "Get optimizer name");
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
            [](float lr,
               float beta1,
               float beta2,
               float epsilon,
               float weight_decay,
               bool amsgrad,
               bool stochastic_rounding) {
                return AdamWConfig{
                    .lr = lr,
                    .beta1 = beta1,
                    .beta2 = beta2,
                    .epsilon = epsilon,
                    .weight_decay = weight_decay,
                    .amsgrad = amsgrad,
                    .stochastic_rounding = stochastic_rounding};
            },
            nb::arg("lr"),
            nb::arg("beta1"),
            nb::arg("beta2"),
            nb::arg("epsilon"),
            nb::arg("weight_decay"),
            nb::arg("amsgrad") = false,
            nb::arg("stochastic_rounding") = false,
            "Make an AdamWConfig object");
    }

    {
        auto py_adamw_full_precision_config =
            static_cast<nb::class_<AdamWFullPrecisionConfig>>(m.attr("AdamWFullPrecisionConfig"));
        py_adamw_full_precision_config.def(nb::init<>());
        py_adamw_full_precision_config.def_static(
            "make",
            [](float lr, float beta1, float beta2, float epsilon, float weight_decay, bool amsgrad) {
                return AdamWFullPrecisionConfig{
                    .lr = lr,
                    .beta1 = beta1,
                    .beta2 = beta2,
                    .epsilon = epsilon,
                    .weight_decay = weight_decay,
                    .amsgrad = amsgrad};
            },
            nb::arg("lr"),
            nb::arg("beta1"),
            nb::arg("beta2"),
            nb::arg("epsilon"),
            nb::arg("weight_decay"),
            nb::arg("amsgrad") = false,
            "Make an AdamWFullPrecisionConfig object");
    }

    {
        auto py_adamw = static_cast<nb::class_<AdamW, OptimizerBase>>(m.attr("AdamW"));
        py_adamw.def(
            nb::init<serialization::NamedParameters, const AdamWConfig&>(), nb::arg("parameters"), nb::arg("config"));
    }

    {
        auto py_adamw_full_precision =
            static_cast<nb::class_<AdamWFullPrecision, OptimizerBase>>(m.attr("AdamWFullPrecision"));
        py_adamw_full_precision.def(
            nb::init<serialization::NamedParameters, const AdamWFullPrecisionConfig&>(),
            nb::arg("parameters"),
            nb::arg("config"));
    }

    {
        auto py_muon_config = static_cast<nb::class_<MuonConfig>>(m.attr("MuonConfig"));
        py_muon_config.def(nb::init<>());
        py_muon_config.def_static(
            "make",
            [](float lr, float momentum, int ns_steps) {
                return MuonConfig{.lr = lr, .momentum = momentum, .ns_steps = ns_steps};
            },
            nb::arg("lr") = 1e-3F,
            nb::arg("momentum") = 0.95F,
            nb::arg("ns_steps") = 5,
            "Create a MuonConfig object");
    }

    {
        auto py_muon = static_cast<nb::class_<MuonComposite, OptimizerBase>>(m.attr("MuonComposite"));
        py_muon.def(
            nb::init<serialization::NamedParameters, const MuonConfig&>(), nb::arg("parameters"), nb::arg("config"));
    }

    {
        auto py_no_op = static_cast<nb::class_<NoOp, OptimizerBase>>(m.attr("NoOp"));
        py_no_op.def(nb::init<serialization::NamedParameters>(), nb::arg("parameters"));
    }

    {
        auto py_remote_optimizer = static_cast<nb::class_<RemoteOptimizer, OptimizerBase>>(m.attr("RemoteOptimizer"));
        py_remote_optimizer.def(
            nb::init<serialization::NamedParameters, int>(), nb::arg("parameters"), nb::arg("aggregator_rank"));
        py_remote_optimizer.def("send_gradients", &RemoteOptimizer::send_gradients);
        py_remote_optimizer.def("receive_weights", &RemoteOptimizer::receive_weights);
        py_remote_optimizer.def("get_sorted_parameters", &RemoteOptimizer::get_sorted_parameters);
    }
}

}  // namespace ttml::nanobind::optimizers
