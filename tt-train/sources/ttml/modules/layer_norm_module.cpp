// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layer_norm_module.hpp"

#include "core/tt_tensor_utils.hpp"

namespace ttml::modules {

void LayerNormLayer::initialize_tensors(uint32_t features) {
    m_gamma =
        autograd::create_tensor(core::ones(core::create_shape({1, 1, 1, features}), &autograd::ctx().get_device()));
    m_beta =
        autograd::create_tensor(core::zeros(core::create_shape({1, 1, 1, features}), &autograd::ctx().get_device()));
}

LayerNormLayer::LayerNormLayer(uint32_t features, bool use_composite_op) : m_use_composite_op(use_composite_op) {
    initialize_tensors(features);

    create_name("layernorm");
    register_tensor(m_gamma, "gamma");
    register_tensor(m_beta, "beta");
}

autograd::TensorPtr LayerNormLayer::operator()(const autograd::TensorPtr& tensor) {
    if (m_use_composite_op) {
        return ops::composite_layernorm(tensor, m_gamma, m_beta);
    }
    return ops::layernorm(tensor, m_gamma, m_beta);
}

}  // namespace ttml::modules
