// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dropout_module.hpp"

#include "autograd/module_base.hpp"
#include "ops/dropout_op.hpp"
namespace ttml::modules {

DropoutLayer::DropoutLayer(float probability, bool use_per_device_seed) :
    m_prob(probability), m_use_per_device_seed(use_per_device_seed) {
    create_name("dropout");
}

[[nodiscard]] autograd::TensorPtr DropoutLayer::operator()(const autograd::TensorPtr& tensor) {
    if (this->get_run_mode() == autograd::RunMode::EVAL) {
        return tensor;
    }

    return ttml::ops::dropout(tensor, m_prob, m_use_per_device_seed);
}

}  // namespace ttml::modules
