// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layer_norm_module.hpp"

#include "core/tt_tensor_utils.hpp"

namespace ttml::modules {

void LayerNormLayer::initialize_tensors(uint32_t features) {
    m_gamma = autograd::create_tensor(core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));
    m_beta = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));
}

LayerNormLayer::LayerNormLayer(
    uint32_t features, float eps, bool use_composite_op, bool enable_hardware_clamp, float min_safe_eps) :
    m_use_composite_op(use_composite_op),
    m_eps(eps),
    m_enable_hardware_clamp(enable_hardware_clamp),
    m_min_safe_eps(min_safe_eps) {
    initialize_tensors(features);

    create_name("layernorm");
    register_tensor(m_gamma, "gamma");
    register_tensor(m_beta, "beta");
}

autograd::TensorPtr LayerNormLayer::operator()(const autograd::TensorPtr& tensor) {
    if (m_use_composite_op) {
        return ops::composite_layernorm(tensor, m_gamma, m_beta, m_eps, m_enable_hardware_clamp, m_min_safe_eps);
    }
    return ops::layernorm(tensor, m_gamma, m_beta, m_eps, m_enable_hardware_clamp, m_min_safe_eps);
}

}  // namespace ttml::modules
