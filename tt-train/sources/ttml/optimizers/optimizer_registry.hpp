// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "optimizers/optimizer_base.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

using OptimizerCreator =
    std::function<std::unique_ptr<OptimizerBase>(const YAML::Node&, serialization::NamedParameters)>;

class OptimizerRegistry {
public:
    static OptimizerRegistry& instance();

    void register_optimizer(const std::string& type, OptimizerCreator creator);

    std::unique_ptr<OptimizerBase> create(const YAML::Node& config, serialization::NamedParameters params) const;

private:
    OptimizerRegistry();
    std::unordered_map<std::string, OptimizerCreator> m_creators;
};

std::unique_ptr<OptimizerBase> create_optimizer(const YAML::Node& config, serialization::NamedParameters params);

}  // namespace ttml::optimizers
