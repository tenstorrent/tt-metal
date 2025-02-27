// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "linear_module.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "init/cpu_initializers.hpp"
#include "init/tensor_initializers.hpp"
#include "ops/linear_op.hpp"

namespace {
ttml::autograd::TensorPtr create_bias(uint32_t out_features, float init_k) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto bias_shape = ttml::core::create_shape({1, 1, 1, out_features});
    auto bias = ttml::autograd::create_tensor();
    ttml::init::uniform_init(bias, bias_shape, ttml::init::UniformRange{-init_k, init_k});
    return bias;
}
}  // namespace
namespace ttml::modules {

void LinearLayer::initialize_tensors(uint32_t in_features, uint32_t out_features, bool has_bias) {
    auto* device = &autograd::ctx().get_device();
    auto weight_shape = core::create_shape({1, 1, out_features, in_features});
    m_weight = ttml::autograd::create_tensor();
    const float init_k = std::sqrtf(1.F / static_cast<float>(in_features));
    init::uniform_init(m_weight, weight_shape, init::UniformRange{-init_k, init_k});
    if (has_bias) {
        m_bias = create_bias(out_features, init_k);
    }
}

LinearLayer::LinearLayer(uint32_t in_features, uint32_t out_features, bool has_bias) {
    initialize_tensors(in_features, out_features, has_bias);

    create_name("linear");
    register_tensor(m_weight, "weight");
    if (m_bias != nullptr) {
        register_tensor(m_bias, "bias");
    }
}

LinearLayer::LinearLayer(const autograd::TensorPtr& weight, bool has_bias) {
    m_weight = weight;
    create_name("linear");
    register_tensor(m_weight, "weight");
    if (has_bias) {
        int in_features = m_weight->get_value().get_logical_shape()[3];
        int out_features = m_weight->get_value().get_logical_shape()[2];
        const float init_k = std::sqrtf(1.F / static_cast<float>(in_features));
        m_bias = create_bias(out_features, init_k);
        register_tensor(m_bias, "bias");
    }
}

LinearLayer::LinearLayer(const autograd::TensorPtr& weight, const autograd::TensorPtr& bias) {
    m_weight = weight;
    m_bias = bias;

    create_name("linear");
    register_tensor(m_weight, "weight");
    if (m_bias != nullptr) {
        register_tensor(m_bias, "bias");
    }
}

autograd::TensorPtr LinearLayer::get_weight() const {
    return m_weight;
}

autograd::TensorPtr LinearLayer::operator()(const autograd::TensorPtr& tensor) {
    return ops::linear_op(tensor, m_weight, m_bias);
}

}  // namespace ttml::modules
