// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>

#include "optimizers/optimizer_base.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

std::unique_ptr<OptimizerBase> create_optimizer(const YAML::Node& config, serialization::NamedParameters params);

}  // namespace ttml::optimizers
