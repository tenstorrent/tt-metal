// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "core/scoped.hpp"
#include "init/tensor_initializers.hpp"

namespace ttml::models::common::transformer {

enum class RunnerType {
    MemoryEfficient,
    Default,
};

enum class WeightTyingType {
    Disabled,
    Enabled,
};

autograd::TensorPtr memory_efficient_runner(
    auto&& forward_impl, const autograd::TensorPtr& input, const autograd::TensorPtr& mask);

void weights_initialization(autograd::ModuleBase& model);

RunnerType read_runner_type(const YAML::Node& config);
WeightTyingType read_weight_tying_type(const YAML::Node& config);

}  // namespace ttml::models::common::transformer
