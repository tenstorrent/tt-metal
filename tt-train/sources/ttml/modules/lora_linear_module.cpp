// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "lora_linear_module.hpp"

#include <cmath>
#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "init/cpu_initializers.hpp"
#include "init/tensor_initializers.hpp"
#include "modules/dropout_module.hpp"
#include "ops/binary_ops.hpp"
#include "ops/linear_op.hpp"
#include "ops/matmul_op.hpp"
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

ttml::autograd::TensorPtr create_lora_a(uint32_t rank, uint32_t out_features) {
    auto* device = &autograd::ctx().get_device();
    auto weight_shape = ttnn::Shape({1, 1, out_features, rank});
    auto weight = ttml::autograd::create_tensor();
    init::constant_init(weight, weight_shape, 0.F);
    return weight;
}

ttml::autograd::TensorPtr create_lora_b(uint32_t in_features, uint32_t rank) {
    auto* device = &autograd::ctx().get_device();
    auto weight_shape = ttnn::Shape({1, 1, rank, in_features});
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

void LoRALinearLayer::register_tensors() {
    create_name("linear");
    register_tensor(m_weight, "weight");

    register_tensor(m_lora_a, "lora_a");
    register_tensor(m_lora_b, "lora_b");
    register_module(m_dropout, "dropout");
    if (m_bias != nullptr) {
        register_tensor(m_bias, "bias");
    }
}

LoRALinearLayer::LoRALinearLayer(
    const LoRALayerConfig& config, uint32_t in_features, uint32_t out_features, bool has_bias) {
    m_weight = create_weight(in_features, out_features);
    m_weight->set_requires_grad(false);
    m_scale = config.alpha / static_cast<float>(config.rank);
    m_lora_a = create_lora_a(config.rank, out_features);
    m_lora_b = create_lora_b(in_features, config.rank);
    if (has_bias) {
        m_bias = create_bias(in_features, out_features);
        m_bias->set_requires_grad(config.is_bias_trainable);
    }
    register_tensors();
}

LoRALinearLayer::LoRALinearLayer(const LoRALayerConfig& config, const autograd::TensorPtr& weight, bool has_bias) :
    m_weight(weight) {
    m_weight->set_requires_grad(false);
    m_scale = config.alpha / static_cast<float>(config.rank);
    auto weight_shape = m_weight->get_value().logical_shape();
    uint32_t in_features = weight_shape[3];
    uint32_t out_features = weight_shape[2];
    m_lora_a = create_lora_a(config.rank, out_features);
    m_lora_b = create_lora_b(in_features, config.rank);
    m_dropout = std::make_shared<DropoutLayer>(config.dropout);
    if (has_bias) {
        m_bias = create_bias(in_features, out_features);
        m_bias->set_requires_grad(config.is_bias_trainable);
    }
    register_tensors();
}

autograd::TensorPtr LoRALinearLayer::operator()(const autograd::TensorPtr& tensor) {
    auto base_output = ops::linear_op(tensor, m_weight, m_bias);
    auto lora_down = ops::matmul_op(tensor, m_lora_a);
    auto lora_down_dropout = (*m_dropout)(lora_down);

    auto lora_update = ops::matmul_op(lora_down_dropout, m_lora_b);
    return base_output + (lora_update * m_scale);
}

}  // namespace ttml::modules
