// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizer_base.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::optimizers {

void require_bf16_parameters(const serialization::NamedParameters& parameters, std::string_view optimizer_name) {
    for (const auto& [name, tensor_ptr] : parameters) {
        if (!tensor_ptr->get_requires_grad()) {
            continue;
        }
        const auto dtype = tensor_ptr->get_value(autograd::PreferredPrecision::NATIVE).dtype();
        TT_FATAL(
            dtype == ttnn::DataType::BFLOAT16,
            "{} supports bf16 parameters only, but parameter '{}' is stored as {}. fp32 parameters are supported by "
            "AdamW, and by other optimizers once the precision config lands (#56513).",
            optimizer_name,
            name,
            dtype);
    }
}

OptimizerBase::OptimizerBase(serialization::NamedParameters&& parameters) : m_parameters(std::move(parameters)) {
}

namespace {
// A group's param names as one string: sorted (so the order the group was built in does not matter).
std::string joined_param_names(std::vector<std::string> names) {
    std::ranges::sort(names);
    return fmt::format("{}", fmt::join(names, "\n"));
}

std::invalid_argument unknown_option_error(
    const std::string& key, const ParamGroupOptions& supported, const std::string& where = "") {
    std::vector<std::string> names;
    names.reserve(supported.size());
    for (const auto& [name, value] : supported) {
        names.push_back(name);
    }
    std::ranges::sort(names);
    return std::invalid_argument(
        fmt::format("Unknown param group option '{}'{} (supported: {})", key, where, fmt::join(names, ", ")));
}
}  // namespace

std::string OptimizerBase::param_group_key(size_t group_idx, const std::string& key) {
    return fmt::format("param_groups.{}.{}", group_idx, key);
}

std::vector<float> OptimizerBase::get_group_options(const std::string& key) const {
    std::vector<float> values;
    values.reserve(m_param_groups.size());
    for (const auto& group : m_param_groups) {
        const auto it = group.options.find(key);
        if (it == group.options.end()) {
            throw unknown_option_error(key, group.options);
        }
        values.push_back(it->second);
    }
    return values;
}

void OptimizerBase::set_group_options(const std::string& key, const std::vector<float>& values) {
    if (values.size() != m_param_groups.size()) {
        throw std::invalid_argument(fmt::format(
            "{} has {} param groups, but got {} values for '{}'",
            get_name(),
            m_param_groups.size(),
            values.size(),
            key));
    }
    for (size_t group_idx = 0; group_idx < m_param_groups.size(); ++group_idx) {
        set_param_group_option(group_idx, key, values[group_idx]);
    }
}

std::vector<float> OptimizerBase::get_group_lrs() const {
    return get_group_options("lr");
}

void OptimizerBase::set_group_lrs(const std::vector<float>& lrs) {
    set_group_options("lr", lrs);
}

std::vector<float> OptimizerBase::get_initial_group_lrs() {
    if (!m_initial_lrs.has_value()) {
        m_initial_lrs = get_group_lrs();
    }
    return *m_initial_lrs;
}

float OptimizerBase::get_lr() const {
    return get_group_lrs().front();
}

void OptimizerBase::set_lr(float lr) {
    const auto initial_lrs = m_initial_lrs.value_or(get_group_lrs());
    const float first_initial_lr = initial_lrs.front();
    const bool all_zero = std::ranges::all_of(initial_lrs, [](float value) { return value == 0.0F; });
    if (first_initial_lr == 0.0F && !all_zero) {
        throw std::invalid_argument(fmt::format(
            "{}::set_lr cannot scale param groups whose first group has an initial lr of 0; set each group's lr "
            "with set_group_lrs instead",
            get_name()));
    }
    std::vector<float> lrs;
    lrs.reserve(initial_lrs.size());
    for (const float initial_lr : initial_lrs) {
        lrs.push_back(all_zero ? lr : lr * (initial_lr / first_initial_lr));
    }
    set_group_lrs(lrs);
}

float OptimizerBase::get_initial_lr() {
    return get_initial_group_lrs().front();
}

void OptimizerBase::save_param_groups(serialization::StateDict& dict) const {
    // If no scheduler has recorded base LRs yet, the current LRs are the base.
    const auto initial_lrs = m_initial_lrs.value_or(get_group_lrs());
    dict["num_param_groups"] = m_param_groups.size();
    for (size_t group_idx = 0; group_idx < m_param_groups.size(); ++group_idx) {
        const auto& group = m_param_groups[group_idx];
        dict[param_group_key(group_idx, "params")] = joined_param_names(group.params);
        for (const auto& [key, value] : group.options) {
            dict[param_group_key(group_idx, key)] = value;
        }
        dict[param_group_key(group_idx, "initial_lr")] = initial_lrs[group_idx];
    }
}

void OptimizerBase::restore_param_groups(const serialization::StateDict& dict) {
    const auto require = [&](const std::string& key) {
        if (!dict.contains(key)) {
            throw std::runtime_error(fmt::format("{} state dict has no \"{}\" entry", get_name(), key));
        }
    };

    if (!dict.contains("num_param_groups")) {
        throw std::runtime_error(fmt::format(
            "{} state dict has no \"num_param_groups\" entry: it was written before param groups were added "
            "(hyperparameters as top-level keys). Such checkpoints are not supported and must be regenerated.",
            get_name()));
    }
    const auto num_groups = serialization::get_value_type<size_t>(dict, "num_param_groups");
    if (num_groups != m_param_groups.size()) {
        throw std::runtime_error(fmt::format(
            "{} checkpoint has {} param groups, but the optimizer has {}",
            get_name(),
            num_groups,
            m_param_groups.size()));
    }
    // Read and check every group first, then apply, so a rejected checkpoint leaves every group (and the state
    // AdamW derives from its betas) unchanged.
    std::vector<ParamGroupOptions> options(num_groups);
    std::vector<float> initial_lrs;
    initial_lrs.reserve(num_groups);
    for (size_t group_idx = 0; group_idx < num_groups; ++group_idx) {
        const auto& group = m_param_groups[group_idx];
        // Group membership comes from how the optimizer was constructed; it must match the checkpoint's, or the
        // saved options would be applied to different params.
        const auto params_key = param_group_key(group_idx, "params");
        require(params_key);
        if (serialization::get_value_type<std::string>(dict, params_key) != joined_param_names(group.params)) {
            throw std::runtime_error(fmt::format(
                "{} param group {} holds different params in the checkpoint than in the optimizer; the optimizer "
                "must be built with the same param groups",
                get_name(),
                group_idx));
        }
        for (const auto& [key, value] : group.options) {
            const auto option_key = param_group_key(group_idx, key);
            require(option_key);
            options[group_idx][key] = serialization::get_value_type<float>(dict, option_key);
        }
        const auto initial_lr_key = param_group_key(group_idx, "initial_lr");
        require(initial_lr_key);
        initial_lrs.push_back(serialization::get_value_type<float>(dict, initial_lr_key));
    }

    for (size_t group_idx = 0; group_idx < num_groups; ++group_idx) {
        for (const auto& [key, value] : options[group_idx]) {
            set_param_group_option(group_idx, key, value);
        }
    }
    m_initial_lrs = std::move(initial_lrs);
}

const std::vector<ParamGroup>& OptimizerBase::get_param_groups() const {
    return m_param_groups;
}

const ParamGroupOptions& OptimizerBase::get_defaults() const {
    return m_defaults;
}

void OptimizerBase::set_param_group_option(size_t group_idx, const std::string& key, float value) {
    if (group_idx >= m_param_groups.size()) {
        throw std::out_of_range(
            fmt::format("Param group index {} out of range ({} groups)", group_idx, m_param_groups.size()));
    }
    auto& options = m_param_groups[group_idx].options;
    const auto it = options.find(key);
    if (it == options.end()) {
        throw unknown_option_error(key, options);
    }
    it->second = value;
}

void OptimizerBase::init_param_groups(ParamGroupOptions defaults, std::vector<ParamGroup> groups) {
    std::unordered_set<std::string> grouped;
    // group 0 is the ungrouped one, so the listed groups start at 1
    for (size_t idx = 0; idx < groups.size(); ++idx) {
        auto& group = groups[idx];
        for (const auto& [key, value] : group.options) {
            if (!defaults.contains(key)) {
                throw unknown_option_error(key, defaults, fmt::format(" in group {}", idx + 1));
            }
        }
        for (const auto& [key, value] : defaults) {
            group.options.try_emplace(key, value);
        }
        for (const auto& name : group.params) {
            if (!m_parameters.contains(name)) {
                throw std::invalid_argument(fmt::format("Param group {} lists unknown param '{}'", idx + 1, name));
            }
            if (!grouped.insert(name).second) {
                throw std::invalid_argument(fmt::format("Param '{}' is listed in more than one param group", name));
            }
        }
    }

    std::vector<std::string> ungrouped;
    for (const auto& [name, tensor] : m_parameters) {
        if (!grouped.contains(name)) {
            ungrouped.push_back(name);
        }
    }

    groups.insert(groups.begin(), ParamGroup{.params = std::move(ungrouped), .options = defaults});

    m_defaults = std::move(defaults);
    m_param_groups = std::move(groups);
}

void restore_topology(
    std::initializer_list<autograd::TensorPtr> written, const tt::tt_metal::TensorTopology& topology) {
    for (const auto& tensor_ptr : written) {
        // A Tensor is a handle: this copy shares the stored tensor's attributes, so the relabel lands on the stored
        // tensor (the one the step wrote) and on every alias of it. No set_value: nothing else changes.
        auto value = tensor_ptr->get_value(autograd::PreferredPrecision::NATIVE);
        value.update_tensor_topology(topology);
    }
}

void OptimizerBase::print_stats() const {
    fmt::print("\n\nOptimization parameters values and gradients:\n");
    for (const auto& [name, tensor] : m_parameters) {
        core::print_tensor_stats(tensor->get_value(), fmt::format("{}/value", name));
        if (tensor->is_grad_initialized()) {
            core::print_tensor_stats(tensor->get_grad(), fmt::format("{}/gradient", name));
        }
    }
    fmt::print("=================================================\n");
}

}  // namespace ttml::optimizers
