// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layer_norm_module.hpp"

#include "core/tt_tensor_utils.hpp"

namespace ttml::modules {

void LayerNormLayer::initialize_tensors(uint32_t features) {
    m_gamma = autograd::create_tensor(core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));
    m_beta = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));
}

LayerNormLayer::LayerNormLayer(uint32_t features, bool use_composite_op) : m_use_composite_op(use_composite_op) {
    initialize_tensors(features);

    create_name("layernorm");
    register_tensor(m_gamma, "gamma");
    register_tensor(m_beta, "beta");
}

autograd::TensorPtr LayerNormLayer::operator()(const autograd::TensorPtr& tensor) {
    // TODO(nuked-op layer_norm): restore real call (composite_layernorm / layernorm)
    static_cast<void>(m_use_composite_op);
    return tensor;
}

}  // namespace ttml::modules
