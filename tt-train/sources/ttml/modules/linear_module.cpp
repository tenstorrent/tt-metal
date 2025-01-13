// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "linear_module.hpp"

#include <core/ttnn_all_includes.hpp>

#include "core/tt_tensor_utils.hpp"
#include "init/cpu_initializers.hpp"
#include "init/tensor_initializers.hpp"

namespace ttml::modules {

void LinearLayer::initialize_tensors(uint32_t in_features, uint32_t out_features) {
    auto* device = &autograd::ctx().get_device();
    auto weight_shape = core::create_shape({1, 1, out_features, in_features});
    m_weight = ttml::autograd::create_tensor();
    const float init_k = std::sqrtf(1.F / static_cast<float>(in_features));
    init::uniform_init(m_weight, weight_shape, init::UniformRange{-init_k, init_k});
    auto bias_shape = core::create_shape({1, 1, 1, out_features});
    m_bias = ttml::autograd::create_tensor();
    init::uniform_init(m_bias, bias_shape, init::UniformRange{-init_k, init_k});
}

LinearLayer::LinearLayer(uint32_t in_features, uint32_t out_features) {
    initialize_tensors(in_features, out_features);

    create_name("linear");
    register_tensor(m_weight, "weight");
    register_tensor(m_bias, "bias");
}

autograd::TensorPtr LinearLayer::operator()(const autograd::TensorPtr& tensor) {
    return ops::linear_op(tensor, m_weight, m_bias);
}

}  // namespace ttml::modules
