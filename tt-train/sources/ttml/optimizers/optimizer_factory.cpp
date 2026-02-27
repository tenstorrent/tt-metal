// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizer_factory.hpp"

#include "optimizers/adamw.hpp"
#include "optimizers/adamw_composite.hpp"
#include "optimizers/adamw_full_precision.hpp"
#include "optimizers/no_op.hpp"
#include "optimizers/sgd.hpp"
#include "optimizers/sgd_composite.hpp"

namespace ttml::optimizers {

std::unique_ptr<OptimizerBase> create_optimizer(const YAML::Node& config, serialization::NamedParameters params) {
    auto type = config["type"].as<std::string>("AdamW");

    if (type == "NoOp") {
        return std::make_unique<NoOp>(std::move(params));
    }
    if (type == "AdamW") {
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
    }
    if (type == "MorehAdamW") {
        return std::make_unique<MorehAdamW>(
            std::move(params),
            AdamWCompositeConfig{
                .lr = config["lr"].as<float>(1e-3F),
                .beta1 = config["beta1"].as<float>(0.9F),
                .beta2 = config["beta2"].as<float>(0.999F),
                .epsilon = config["epsilon"].as<float>(1e-8F),
                .weight_decay = config["weight_decay"].as<float>(0.01F),
                .amsgrad = config["amsgrad"].as<bool>(false),
                .use_kahan_summation = config["use_kahan_summation"].as<bool>(false)});
    }
    if (type == "AdamWComposite") {
        return std::make_unique<AdamWComposite>(
            std::move(params),
            AdamWCompositeConfig{
                .lr = config["lr"].as<float>(1e-3F),
                .beta1 = config["beta1"].as<float>(0.9F),
                .beta2 = config["beta2"].as<float>(0.999F),
                .epsilon = config["epsilon"].as<float>(1e-8F),
                .weight_decay = config["weight_decay"].as<float>(0.01F),
                .amsgrad = config["amsgrad"].as<bool>(false),
                .use_kahan_summation = config["use_kahan_summation"].as<bool>(false)});
    }
    if (type == "AdamWFullPrecision") {
        return std::make_unique<AdamWFullPrecision>(
            std::move(params),
            AdamWFullPrecisionConfig{
                .lr = config["lr"].as<float>(1e-3F),
                .beta1 = config["beta1"].as<float>(0.9F),
                .beta2 = config["beta2"].as<float>(0.999F),
                .epsilon = config["epsilon"].as<float>(1e-8F),
                .weight_decay = config["weight_decay"].as<float>(1e-2F),
                .amsgrad = config["amsgrad"].as<bool>(false)});
    }
    if (type == "SGD") {
        return std::make_unique<SGD>(
            std::move(params),
            SGDConfig{
                .lr = config["lr"].as<float>(1e-3F),
                .momentum = config["momentum"].as<float>(0.0F),
                .dampening = config["dampening"].as<float>(0.0F),
                .weight_decay = config["weight_decay"].as<float>(0.0F),
                .nesterov = config["nesterov"].as<bool>(false)});
    }
    if (type == "SGDComposite") {
        return std::make_unique<SGDComposite>(
            std::move(params),
            SGDCompositeConfig{
                .lr = config["lr"].as<float>(1e-3F),
                .momentum = config["momentum"].as<float>(0.0F),
                .dampening = config["dampening"].as<float>(0.0F),
                .weight_decay = config["weight_decay"].as<float>(0.0F),
                .nesterov = config["nesterov"].as<bool>(false)});
    }
    throw std::runtime_error("Unknown optimizer type: " + type);
}

}  // namespace ttml::optimizers
