// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "autograd/tensor.hpp"
#include "optimizers/optimizer_base.hpp"
#include "serialization/serializable.hpp"

namespace ttml::test_utils {

using NamedTensors = std::vector<std::pair<std::string, autograd::TensorPtr>>;

// The tensors in an optimizer's state, such as its moments or momentum buffers, by "<entry>/<parameter>".
inline NamedTensors state_tensors(const optimizers::OptimizerBase& optimizer) {
    NamedTensors tensors;
    for (const auto& [key, value] : optimizer.get_state_dict()) {
        if (const auto* named = std::get_if<serialization::NamedParameters>(&value)) {
            for (const auto& [name, tensor] : *named) {
                tensors.emplace_back(key + "/" + name, tensor);
            }
        }
    }
    return tensors;
}

}  // namespace ttml::test_utils
