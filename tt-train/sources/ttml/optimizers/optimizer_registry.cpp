// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizer_registry.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <sstream>
#include <unordered_set>
#include <vector>

#include "optimizers/adamw.hpp"
#include "optimizers/adamw_composite.hpp"
#include "optimizers/adamw_full_precision.hpp"
#include "optimizers/muon_composite.hpp"
#include "optimizers/no_op.hpp"
#include "optimizers/sgd.hpp"
#include "optimizers/sgd_composite.hpp"

namespace ttml::optimizers {

namespace {
bool name_has_segment_in(const std::string& name, const std::unordered_set<std::string>& segments) {
    std::istringstream parts(name);
    for (std::string part; std::getline(parts, part, '/');) {
        if (segments.contains(part)) {
            return true;
        }
    }
    return false;
}

// Resolves the "param_groups" config list into param groups. Each entry lists "modules" plus the
// options it overrides (validated and completed from the defaults by the optimizer). A param joins the
// first group with a "/"-separated name segment in "modules", so both module names ("tok_emb") and leaf
// tensor names ("gamma", "bias") work. Unmatched params use the top-level config.
std::vector<ParamGroup> resolve_param_groups(const YAML::Node& config, const serialization::NamedParameters& params) {
    const auto& groups_config = config["param_groups"];
    if (!groups_config) {
        return {};
    }
    std::vector<ParamGroup> groups;
    std::vector<std::unordered_set<std::string>> group_modules;
    for (const auto& group_config : groups_config) {
        if (!group_config["modules"]) {
            throw std::runtime_error("Every param group needs a \"modules\" list");
        }
        const auto modules = group_config["modules"].as<std::vector<std::string>>();
        group_modules.emplace_back(modules.begin(), modules.end());
        ParamGroup group;
        for (const auto& entry : group_config) {
            const auto key = entry.first.as<std::string>();
            if (key == "modules") {
                continue;
            }
            try {
                group.options[key] = entry.second.as<float>();
            } catch (const YAML::BadConversion&) {
                throw std::runtime_error("Param group option '" + key + "' must be a number");
            }
        }
        groups.push_back(std::move(group));
    }
    for (const auto& [name, tensor] : params) {
        for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
            if (name_has_segment_in(name, group_modules[group_idx])) {
                groups[group_idx].params.push_back(name);
                break;
            }
        }
    }
    // A group that matches nothing (a typo, or a name the model does not use) would silently leave its params on
    // the top-level settings. With no params at all there is nothing to match, so that case is allowed. Under
    // pipeline parallelism each rank's optimizer only sees its own stage's params (tok_emb only on the first stage,
    // fc only on the last), so a valid group can be empty on some ranks: trainers set allow_empty_param_groups.
    if (!params.empty() && !config["allow_empty_param_groups"].as<bool>(false)) {
        for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
            if (groups[group_idx].params.empty()) {
                throw std::runtime_error(fmt::format(
                    "Param group {} (modules: {}) matches no parameters; check the module names, and that an earlier "
                    "group does not already claim all of them",
                    group_idx + 1,  // group 0 is the ungrouped one, so the listed groups start at 1
                    fmt::join(groups_config[group_idx]["modules"].as<std::vector<std::string>>(), ", ")));
            }
        }
    }
    return groups;
}

// The options every AdamW variant has. Keys missing from the YAML keep the config struct's own defaults.
template <typename Config>
Config parse_adamw_config(const YAML::Node& config) {
    Config adamw_config;
    adamw_config.lr = config["lr"].as<float>(adamw_config.lr);
    adamw_config.beta1 = config["beta1"].as<float>(adamw_config.beta1);
    adamw_config.beta2 = config["beta2"].as<float>(adamw_config.beta2);
    adamw_config.epsilon = config["epsilon"].as<float>(adamw_config.epsilon);
    adamw_config.weight_decay = config["weight_decay"].as<float>(adamw_config.weight_decay);
    adamw_config.amsgrad = config["amsgrad"].as<bool>(adamw_config.amsgrad);
    if constexpr (requires { adamw_config.stochastic_rounding; }) {
        adamw_config.stochastic_rounding = config["stochastic_rounding"].as<bool>(adamw_config.stochastic_rounding);
    }
    if constexpr (requires { adamw_config.kahan_summation; }) {
        adamw_config.kahan_summation = config["kahan_summation"].as<bool>(adamw_config.kahan_summation);
    }
    return adamw_config;
}

// The options SGD and SGDComposite share. Keys missing from the YAML keep the config struct's own defaults.
template <typename Config>
Config parse_sgd_config(const YAML::Node& config) {
    Config sgd_config;
    sgd_config.lr = config["lr"].as<float>(sgd_config.lr);
    sgd_config.momentum = config["momentum"].as<float>(sgd_config.momentum);
    sgd_config.dampening = config["dampening"].as<float>(sgd_config.dampening);
    sgd_config.weight_decay = config["weight_decay"].as<float>(sgd_config.weight_decay);
    sgd_config.nesterov = config["nesterov"].as<bool>(sgd_config.nesterov);
    return sgd_config;
}

MuonConfig parse_muon_config(const YAML::Node& config) {
    MuonConfig muon_config;
    muon_config.lr = config["lr"].as<float>(muon_config.lr);
    muon_config.momentum = config["momentum"].as<float>(muon_config.momentum);
    muon_config.ns_steps = config["ns_steps"].as<int>(muon_config.ns_steps);
    return muon_config;
}

// Creator for an optimizer constructed from its params, the config parse_config reads from the YAML, and the param
// groups resolved from the YAML's param_groups.
template <typename Optimizer, typename ParseConfig>
OptimizerCreator param_groups_creator(ParseConfig parse_config) {
    return [parse_config](
               const YAML::Node& config, serialization::NamedParameters params) -> std::unique_ptr<OptimizerBase> {
        auto optimizer_config = parse_config(config);
        auto param_groups = resolve_param_groups(config, params);
        return std::make_unique<Optimizer>(std::move(params), optimizer_config, std::move(param_groups));
    };
}
}  // namespace

OptimizerRegistry& OptimizerRegistry::instance() {
    static OptimizerRegistry registry;
    return registry;
}

OptimizerRegistry::OptimizerRegistry() {
    register_optimizer("NoOp", [](const YAML::Node&, serialization::NamedParameters params) {
        return std::make_unique<NoOp>(std::move(params));
    });

    register_optimizer("AdamW", param_groups_creator<AdamW>(parse_adamw_config<AdamWConfig>));
    register_optimizer("MorehAdamW", param_groups_creator<MorehAdamW>(parse_adamw_config<AdamWCompositeConfig>));
    register_optimizer(
        "AdamWComposite", param_groups_creator<AdamWComposite>(parse_adamw_config<AdamWCompositeConfig>));
    register_optimizer(
        "AdamWFullPrecision", param_groups_creator<AdamWFullPrecision>(parse_adamw_config<AdamWFullPrecisionConfig>));
    register_optimizer("SGD", param_groups_creator<SGD>(parse_sgd_config<SGDConfig>));
    register_optimizer("MuonComposite", param_groups_creator<MuonComposite>(parse_muon_config));
    register_optimizer("SGDComposite", param_groups_creator<SGDComposite>(parse_sgd_config<SGDCompositeConfig>));
}

void OptimizerRegistry::register_optimizer(const std::string& type, OptimizerCreator creator) {
    m_creators[type] = std::move(creator);
}

std::unique_ptr<OptimizerBase> OptimizerRegistry::create(
    const YAML::Node& config, serialization::NamedParameters params) const {
    auto type = config["type"].as<std::string>("AdamW");
    auto it = m_creators.find(type);
    if (it == m_creators.end()) {
        throw std::runtime_error("Unknown optimizer type: " + type);
    }
    if (config["weight_decay_skip_1d"]) {
        throw std::runtime_error(
            "weight_decay_skip_1d was removed, use a param group with weight_decay: 0.0 instead "
            "(see tt-train/configs/README.md)");
    }

    return it->second(config, std::move(params));
}

std::unique_ptr<OptimizerBase> create_optimizer(const YAML::Node& config, serialization::NamedParameters params) {
    return OptimizerRegistry::instance().create(config, std::move(params));
}

}  // namespace ttml::optimizers
