// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dropout_module.hpp"

#include <cmath>
#include <stdexcept>

#include "modules/module_base.hpp"
#include "ops/dropout_op.hpp"
namespace ttml::modules {

DropoutLayer::DropoutLayer(float probability, bool use_per_device_seed) :
    m_prob(probability), m_use_per_device_seed(use_per_device_seed) {
    if (!std::isfinite(probability) || probability < 0.0F || probability >= 1.0F) {
        throw std::invalid_argument("DropoutLayer probability must be finite and in [0, 1).");
    }
    create_name("dropout");
}

[[nodiscard]] autograd::TensorPtr DropoutLayer::operator()(const autograd::TensorPtr& tensor) {
    if (this->get_run_mode() == RunMode::EVAL) {
        return tensor;
    }

    return ttml::ops::dropout(tensor, m_prob, m_use_per_device_seed);
}

}  // namespace ttml::modules
