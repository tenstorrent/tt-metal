// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "linear_module.hpp"

#include <cmath>
#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "init/cpu_initializers.hpp"
#include "init/tensor_initializers.hpp"
#include "ops/linear_op.hpp"

namespace ttml::modules {

namespace {
ttml::autograd::TensorPtr create_weight(uint32_t in_features, uint32_t out_features) {
    auto* device = &autograd::ctx().get_device();
    auto weight_shape = ttnn::Shape({1, 1, out_features, in_features});
    auto weight = ttml::autograd::create_tensor();
    const float init_k = std::sqrt(1.F / static_cast<float>(in_features));
    init::uniform_init(weight, weight_shape, init::UniformRange{-init_k, init_k});
    return weight;
}
ttml::autograd::TensorPtr create_bias(uint32_t in_features, uint32_t out_features) {
    const float init_k = std::sqrt(1.F / static_cast<float>(in_features));
    auto* device = &ttml::autograd::ctx().get_device();
    auto bias_shape = ttnn::Shape({1, 1, 1, out_features});
    auto bias = ttml::autograd::create_tensor();
    ttml::init::uniform_init(bias, bias_shape, ttml::init::UniformRange{-init_k, init_k});
    return bias;
}
}  // namespace

void LinearLayer::register_tensors() {
    create_name("linear");
    register_tensor(m_weight, "weight");
    if (m_bias != nullptr) {
        register_tensor(m_bias, "bias");
    }
}

LinearLayer::LinearLayer(uint32_t in_features, uint32_t out_features, bool has_bias) {
    m_weight = create_weight(in_features, out_features);
    if (has_bias) {
        m_bias = create_bias(in_features, out_features);
    }
    register_tensors();
}

LinearLayer::LinearLayer(const autograd::TensorPtr& weight, bool has_bias) : m_weight(weight) {
    if (has_bias) {
        auto weight_shape = m_weight->get_value().logical_shape();
        uint32_t in_features = weight_shape[3];
        uint32_t out_features = weight_shape[2];
        m_bias = create_bias(in_features, out_features);
    }
    register_tensors();
}

LinearLayer::LinearLayer(const autograd::TensorPtr& weight, const autograd::TensorPtr& bias) :
    m_weight(weight), m_bias(bias) {
    register_tensors();
}

autograd::TensorPtr LinearLayer::get_weight() const {
    return m_weight;
}

autograd::TensorPtr LinearLayer::operator()(const autograd::TensorPtr& tensor) {
    return ops::linear_op(tensor, m_weight, m_bias);
}

}  // namespace ttml::modules
