// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizer_registry.hpp"

#include "optimizers/adamw.hpp"
#include "optimizers/adamw_composite.hpp"
#include "optimizers/adamw_full_precision.hpp"
#include "optimizers/muon.hpp"
#include "optimizers/no_op.hpp"
#include "optimizers/sgd.hpp"
#include "optimizers/sgd_composite.hpp"

namespace ttml::optimizers {

OptimizerRegistry& OptimizerRegistry::instance() {
    static OptimizerRegistry registry;
    return registry;
}

OptimizerRegistry::OptimizerRegistry() {
    register_optimizer("NoOp", [](const YAML::Node&, serialization::NamedParameters params) {
        return std::make_unique<NoOp>(std::move(params));
    });

    register_optimizer("AdamW", [](const YAML::Node& config, serialization::NamedParameters params) {
        return std::make_unique<AdamW>(
            std::move(params),
            AdamWConfig{
                .lr = config["lr"].as<float>(1e-3F),
                .beta1 = config["beta1"].as<float>(0.9F),
                .beta2 = config["beta2"].as<float>(0.999F),
                .epsilon = config["epsilon"].as<float>(1e-8F),
                .weight_decay = config["weight_decay"].as<float>(1e-2F),
                .amsgrad = config["amsgrad"].as<bool>(false),
                .stochastic_rounding = config["stochastic_rounding"].as<bool>(false)});
    });

    register_optimizer("MorehAdamW", [](const YAML::Node& config, serialization::NamedParameters params) {
        return std::make_unique<MorehAdamW>(
            std::move(params),
            AdamWCompositeConfig{
                .lr = config["lr"].as<float>(1e-3F),
                .beta1 = config["beta1"].as<float>(0.9F),
                .beta2 = config["beta2"].as<float>(0.999F),
                .epsilon = config["epsilon"].as<float>(1e-8F),
                .weight_decay = config["weight_decay"].as<float>(0.01F),
                .amsgrad = config["amsgrad"].as<bool>(false),
                .kahan_summation = config["kahan_summation"].as<bool>(false)});
    });

    register_optimizer("AdamWComposite", [](const YAML::Node& config, serialization::NamedParameters params) {
        return std::make_unique<AdamWComposite>(
            std::move(params),
            AdamWCompositeConfig{
                .lr = config["lr"].as<float>(1e-3F),
                .beta1 = config["beta1"].as<float>(0.9F),
                .beta2 = config["beta2"].as<float>(0.999F),
                .epsilon = config["epsilon"].as<float>(1e-8F),
                .weight_decay = config["weight_decay"].as<float>(0.01F),
                .amsgrad = config["amsgrad"].as<bool>(false),
                .kahan_summation = config["kahan_summation"].as<bool>(false)});
    });

    register_optimizer("AdamWFullPrecision", [](const YAML::Node& config, serialization::NamedParameters params) {
        return std::make_unique<AdamWFullPrecision>(
            std::move(params),
            AdamWFullPrecisionConfig{
                .lr = config["lr"].as<float>(1e-3F),
                .beta1 = config["beta1"].as<float>(0.9F),
                .beta2 = config["beta2"].as<float>(0.999F),
                .epsilon = config["epsilon"].as<float>(1e-8F),
                .weight_decay = config["weight_decay"].as<float>(1e-2F),
                .amsgrad = config["amsgrad"].as<bool>(false)});
    });

    register_optimizer("SGD", [](const YAML::Node& config, serialization::NamedParameters params) {
        return std::make_unique<SGD>(
            std::move(params),
            SGDConfig{
                .lr = config["lr"].as<float>(1e-3F),
                .momentum = config["momentum"].as<float>(0.0F),
                .dampening = config["dampening"].as<float>(0.0F),
                .weight_decay = config["weight_decay"].as<float>(0.0F),
                .nesterov = config["nesterov"].as<bool>(false)});
    });

    register_optimizer("Muon", [](const YAML::Node& config, serialization::NamedParameters params) {
        return std::make_unique<Muon>(
            std::move(params),
            MuonConfig{
                .lr = config["lr"].as<float>(1e-3F),
                .momentum = config["momentum"].as<float>(0.95F),
                .ns_steps = config["ns_steps"].as<int>(5)});
    });

    register_optimizer("SGDComposite", [](const YAML::Node& config, serialization::NamedParameters params) {
        return std::make_unique<SGDComposite>(
            std::move(params),
            SGDCompositeConfig{
                .lr = config["lr"].as<float>(1e-3F),
                .momentum = config["momentum"].as<float>(0.0F),
                .dampening = config["dampening"].as<float>(0.0F),
                .weight_decay = config["weight_decay"].as<float>(0.0F),
                .nesterov = config["nesterov"].as<bool>(false)});
    });
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
    return it->second(config, std::move(params));
}

std::unique_ptr<OptimizerBase> create_optimizer(const YAML::Node& config, serialization::NamedParameters params) {
    return OptimizerRegistry::instance().create(config, std::move(params));
}

}  // namespace ttml::optimizers
