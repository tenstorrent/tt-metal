// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <initializer_list>
#include <optional>
#include <reflect>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "serialization/serializable.hpp"

namespace ttml::optimizers {

using ParamGroupOptions = std::unordered_map<std::string, float>;

// Param group: model's parameters names plus the hyperparameters they are stepped with
struct ParamGroup {
    std::vector<std::string> params;
    ParamGroupOptions options;
};

// The per-group options of an optimizer config
template <typename Config>
[[nodiscard]] ParamGroupOptions param_group_defaults(const Config& config) {
    ParamGroupOptions defaults;
    reflect::for_each(
        [&](auto field) {
            const auto& value = reflect::get<field>(config);
            if constexpr (std::is_same_v<std::decay_t<decltype(value)>, float>) {
                defaults.emplace(reflect::member_name<field>(config), value);
            }
        },
        config);
    return defaults;
}

// TT_FATAL unless every trainable parameter is stored in bf16. For optimizers whose update only exists for bf16
// parameters: fp32 parameters are supported by AdamW, and by the others once the precision config lands (#56513).
void require_bf16_parameters(const serialization::NamedParameters& parameters, std::string_view optimizer_name);

class OptimizerBase {
public:
    [[nodiscard]] virtual std::string get_name() const = 0;

    explicit OptimizerBase(serialization::NamedParameters&& parameters);
    OptimizerBase(const OptimizerBase&) = delete;
    OptimizerBase& operator=(const OptimizerBase&) = delete;
    OptimizerBase(OptimizerBase&&) = delete;
    OptimizerBase& operator=(OptimizerBase&&) = delete;
    virtual ~OptimizerBase() = default;

    virtual void zero_grad() = 0;

    virtual void step() = 0;

    [[nodiscard]] virtual serialization::StateDict get_state_dict() const = 0;
    virtual void set_state_dict(const serialization::StateDict& dict) = 0;

    [[nodiscard]] virtual size_t get_steps() const = 0;
    virtual void set_steps(size_t steps) = 0;

    // Value of hyperparameter in every param group, indexed like the groups.
    [[nodiscard]] std::vector<float> get_group_options(const std::string& key) const;
    // Sets hyperparameter for every param group, one value per group.
    void set_group_options(const std::string& key, const std::vector<float>& values);

    [[nodiscard]] virtual std::vector<float> get_group_lrs() const;
    virtual void set_group_lrs(const std::vector<float>& lrs);

    // Per-group LRs recorded the first time this is called. Schedulers read their base LRs from here so that several
    // schedulers attached to the same optimizer share one base even after an earlier scheduler has already scaled the
    // LRs at construction.
    [[nodiscard]] virtual std::vector<float> get_initial_group_lrs();

    // Deprecated single-LR API, kept for existing callers and to be removed; use the per-group methods above instead.
    // - get_lr(): the first group's LR.
    // - set_lr(lr): sets the first group to lr and scales the others by the same factor, keeping their ratios. It
    //   scales relative to the initial LRs once a scheduler has recorded them, and to the current LRs before that.
    //   It doesn't record them itself, so a scheduler attached afterwards takes the LRs set_lr left as its base.
    //   Without a scheduler, set_lr(0) loses the ratios: a later set_lr(lr) gives every group lr.
    // - get_initial_lr(): the first group's initial LR (see get_initial_group_lrs).
    [[nodiscard]] float get_lr() const;
    void set_lr(float lr);
    [[nodiscard]] float get_initial_lr();

    virtual void print_stats() const;

    // Every optimizer has at least one group.
    [[nodiscard]] const std::vector<ParamGroup>& get_param_groups() const;
    [[nodiscard]] const ParamGroupOptions& get_defaults() const;

    // Sets hyperparameter for a specific param group by key.
    virtual void set_param_group_option(size_t group_idx, const std::string& key, float value);

protected:
    // State dict key of one param group entry: "param_groups.{group_idx}.{key}".
    [[nodiscard]] static std::string param_group_key(size_t group_idx, const std::string& key);

    // Fills every group's missing options from defaults and validates the groups: each option must
    // be one of the default options' keys, and each param must exist and belong to at most one group. Group 0 always
    // holds the params in no group (possibly none) with the defaults.
    void init_param_groups(ParamGroupOptions defaults, std::vector<ParamGroup> groups);

    std::vector<ParamGroup> m_param_groups;
    ParamGroupOptions m_defaults;

    // Param group state: num_param_groups and per group param_groups.{i}.{option}, param_groups.{i}.initial_lr and
    // param_groups.{i}.params.
    void save_param_groups(serialization::StateDict& dict) const;
    void restore_param_groups(const serialization::StateDict& dict);

    serialization::NamedParameters m_parameters;

private:
    std::optional<std::vector<float>> m_initial_lrs;
};

// Puts `topology` back on the value of every tensor in `written`.
//
// A ttnn op labels its output with the UNION of its inputs' mesh topologies
// (ttnn::device_operation::detail::compute_output_placements_and_shape), gradient included, and a composite
// optimizer step stores those outputs in the parameter and its state through Tensor::set_value. A gradient with a
// wrong label -- a CCL output that kept a stale Shard, or a collapsed 1-D label from a default mapper -- would so
// relabel the parameter that the checkpointer later gathers by that label. Capture the parameter's topology BY
// VALUE before the step (Tensor::tensor_topology() returns a reference into attributes the ops overwrite), leave
// the step body as it is, and call this at the end of the iteration over everything the step wrote. Optimizer
// state follows its parameter's distribution, so it takes the same label.
//
// Relabels the native tensor only. The other-precision copy picks the label up when it is next re-derived from the
// native tensor after a write, so call this only on tensors the step wrote; a tensor the step did not write would
// keep a stale copy.
void restore_topology(std::initializer_list<autograd::TensorPtr> written, const tt::tt_metal::TensorTopology& topology);

}  // namespace ttml::optimizers
