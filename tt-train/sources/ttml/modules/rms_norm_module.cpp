// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_norm_module.hpp"

#include "core/tt_tensor_utils.hpp"
#include "ops/rmsnorm_op.hpp"

namespace ttml::modules {

void RMSNormLayer::initialize_tensors(uint32_t features) {
    m_gamma =
        autograd::create_tensor(core::ones(core::create_shape({1, 1, 1, features}), &autograd::ctx().get_device()));
}

RMSNormLayer::RMSNormLayer(uint32_t features, float epsilon) : m_epsilon(epsilon) {
    initialize_tensors(features);

    create_name("rmsnorm");
    register_tensor(m_gamma, "gamma");
}

autograd::TensorPtr RMSNormLayer::operator()(const autograd::TensorPtr& tensor) {
    return ops::rmsnorm(tensor, m_gamma, m_epsilon);
}

}  // namespace ttml::modules
