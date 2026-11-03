// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include "serialization/serializable.hpp"

// Make NamedParameters opaque - must be before unordered_map include
NB_MAKE_OPAQUE(ttml::serialization::NamedParameters)

#include <nanobind/stl/pair.h>
#include <nanobind/stl/unordered_map.h>
#include <yaml-cpp/yaml.h>

#include <stdexcept>

#include "nanobind/nb_export_enum.hpp"
#include "nanobind/nb_fwd.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/adamw_full_precision.hpp"
#include "optimizers/muon_composite.hpp"
#include "optimizers/no_op.hpp"
#include "optimizers/optimizer_base.hpp"
#include "optimizers/optimizer_registry.hpp"
#include "optimizers/remote_optimizer.hpp"
#include "optimizers/sgd.hpp"

namespace {

// Python dict -> StateDict, picking each value's alternative explicitly. nanobind's own variant caster tries the
// alternatives in order, and with nanobind/stl/vector.h in this file ValueType's std::vector<...> alternatives accept
// any sequence: an empty NamedParameters (e.g. momentum with no params) would become an empty std::vector<char>.
ttml::serialization::StateDict state_dict_from_py(const nb::dict& py_dict) {
    ttml::serialization::StateDict dict;
    for (auto [py_key, py_value] : py_dict) {
        auto key = nb::cast<std::string>(py_key);
        ttml::serialization::NamedParameters named_parameters;
        ttml::autograd::TensorPtr tensor_ptr;
        ttnn::Tensor tensor;
        if (nb::try_cast(py_value, named_parameters, /*convert=*/false)) {
            dict[key] = std::move(named_parameters);
        } else if (nb::try_cast(py_value, tensor_ptr, /*convert=*/false)) {
            dict[key] = std::move(tensor_ptr);
        } else if (nb::try_cast(py_value, tensor, /*convert=*/false)) {
            dict[key] = std::move(tensor);
        } else {
            dict[key] = nb::cast<ttml::serialization::ValueType>(py_value);
        }
    }
    return dict;
}

}  // namespace

class OptimizerBaseTrampoline : public ttml::optimizers::OptimizerBase {
    NB_TRAMPOLINE(OptimizerBase, 13);

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
        // NB_OVERRIDE_PURE without its result cast, which would go through the ambiguous variant caster.
        nb::detail::ticket nb_ticket(nb_trampoline, "get_state_dict", /*pure=*/true);
        return state_dict_from_py(nb::cast<nb::dict>(nb_trampoline.base().attr(nb_ticket.key)()));
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
    std::vector<float> get_group_lrs() const override {
        // Fallback for Python optimizers that only implement the deprecated get_lr; goes away with the shims.
        {
            nb::detail::ticket nb_ticket(nb_trampoline, "get_group_lrs", /*pure=*/false);
            if (!nb_ticket.key.is_valid()) {
                nb::detail::ticket legacy(nb_trampoline, "get_lr", /*pure=*/false);
                if (legacy.key.is_valid()) {
                    return {nb::cast<float>(nb_trampoline.base().attr(legacy.key)())};
                }
            }
        }
        NB_OVERRIDE_PURE(get_group_lrs);
    }
    void set_group_lrs(const std::vector<float>& lrs) override {
        // Fallback for Python optimizers that only implement the deprecated set_lr; goes away with the shims.
        {
            nb::detail::ticket nb_ticket(nb_trampoline, "set_group_lrs", /*pure=*/false);
            if (!nb_ticket.key.is_valid()) {
                nb::detail::ticket legacy(nb_trampoline, "set_lr", /*pure=*/false);
                if (legacy.key.is_valid()) {
                    if (lrs.size() != 1) {
                        throw std::invalid_argument(
                            "this optimizer only implements the single-LR set_lr, but got " +
                            std::to_string(lrs.size()) + " LRs");
                    }
                    nb_trampoline.base().attr(legacy.key)(lrs.front());
                    return;
                }
            }
        }
        NB_OVERRIDE_PURE(set_group_lrs, lrs);
    }
    std::vector<float> get_initial_group_lrs() override {
        NB_OVERRIDE(get_initial_group_lrs);
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

std::vector<ParamGroup> param_groups_from_py(const nb::list& groups) {
    std::vector<ParamGroup> result;
    for (auto group_obj : groups) {
        auto group_dict = nb::cast<nb::dict>(group_obj);
        if (!group_dict.contains("params")) {
            throw std::invalid_argument("Every param group dict needs a \"params\" entry");
        }
        ParamGroup group;
        for (auto [key, value] : group_dict) {
            const auto key_str = nb::cast<std::string>(key);
            if (key_str == "params") {
                group.params = nb::cast<std::vector<std::string>>(value);
            } else {
                group.options[key_str] = nb::cast<float>(value);
            }
        }
        result.push_back(std::move(group));
    }
    return result;
}

nb::list param_groups_to_py(const std::vector<ParamGroup>& groups) {
    nb::list result;
    for (const auto& group : groups) {
        nb::dict group_dict;
        group_dict["params"] = nb::cast(group.params);
        for (const auto& [key, value] : group.options) {
            group_dict[key.c_str()] = value;
        }
        result.append(group_dict);
    }
    return result;
}

template <typename Optimizer, typename Config>
void def_init_with_param_groups(nb::class_<Optimizer, OptimizerBase>& py_optimizer) {
    py_optimizer.def(
        "__init__",
        [](Optimizer* self, serialization::NamedParameters parameters, const Config& config, nb::list param_groups) {
            new (self) Optimizer(std::move(parameters), config, param_groups_from_py(param_groups));
        },
        nb::arg("parameters"),
        nb::arg("config"),
        nb::arg("param_groups") = nb::list());
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
                // Python-registered optimizers bypass OptimizerRegistry::create, so repeat its
                // weight_decay_skip_1d guard or the removed key silently does nothing.
                if (config.contains("weight_decay_skip_1d")) {
                    throw std::runtime_error(
                        "weight_decay_skip_1d was removed, use a param group with weight_decay: 0.0 instead "
                        "(see tt-train/configs/README.md)");
                }
                // They also have no param groups, so reject the key or it silently does nothing.
                if (config.contains("param_groups")) {
                    throw std::runtime_error("param_groups is not supported by optimizer type: " + type);
                }
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
            "set_state_dict",
            [](OptimizerBase& self, const nb::dict& dict) { self.set_state_dict(state_dict_from_py(dict)); },
            nb::arg("dict"),
            "Set state dictionary");
        py_optimizer_base.def("print_stats", &OptimizerBase::print_stats, "Print statistics");
        py_optimizer_base.def(
            "get_group_lrs", &OptimizerBase::get_group_lrs, "LR of every param group, indexed like get_param_groups");
        py_optimizer_base.def(
            "set_group_lrs",
            &OptimizerBase::set_group_lrs,
            nb::arg("lrs"),
            "Set the LR of every param group (indexed like get_group_lrs). This is what LR schedulers call");
        py_optimizer_base.def(
            "get_group_options",
            &OptimizerBase::get_group_options,
            nb::arg("key"),
            "Value of a param group option ('lr', 'weight_decay', ...) in every param group");
        py_optimizer_base.def(
            "set_group_options",
            &OptimizerBase::set_group_options,
            nb::arg("key"),
            nb::arg("values"),
            "Set a param group option in every param group, one value per group (indexed like get_param_groups)");
        py_optimizer_base.def(
            "get_initial_group_lrs",
            &OptimizerBase::get_initial_group_lrs,
            "Per-group LRs recorded the first time this is called (PyTorch's param_group['initial_lr'])");
        py_optimizer_base.def(
            "get_lr", &OptimizerBase::get_lr, "Deprecated: the first param group's LR; use get_group_lrs");
        py_optimizer_base.def(
            "set_lr",
            &OptimizerBase::set_lr,
            nb::arg("lr"),
            "Deprecated: scale every param group's LR so the first one is lr, keeping their ratios; use set_group_lrs");
        py_optimizer_base.def(
            "get_initial_lr",
            &OptimizerBase::get_initial_lr,
            "Deprecated: the first param group's initial LR; use get_initial_group_lrs");
        py_optimizer_base.def(
            "get_param_groups",
            [](const OptimizerBase& self) { return param_groups_to_py(self.get_param_groups()); },
            "Param groups as PyTorch-style dicts: {'params': [names], <option>: value, ...}");
        py_optimizer_base.def("get_defaults", &OptimizerBase::get_defaults, "Default param group options");
        py_optimizer_base.def(
            "set_param_group_option",
            &OptimizerBase::set_param_group_option,
            nb::arg("group_idx"),
            nb::arg("key"),
            nb::arg("value"),
            "Set one option of one param group");
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
        def_init_with_param_groups<SGD, SGDConfig>(py_sgd);
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
        def_init_with_param_groups<AdamW, AdamWConfig>(py_adamw);
    }

    {
        auto py_adamw_full_precision =
            static_cast<nb::class_<AdamWFullPrecision, OptimizerBase>>(m.attr("AdamWFullPrecision"));
        def_init_with_param_groups<AdamWFullPrecision, AdamWFullPrecisionConfig>(py_adamw_full_precision);
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
        def_init_with_param_groups<MuonComposite, MuonConfig>(py_muon);
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
