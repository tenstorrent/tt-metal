// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "linear_module.hpp"

#include <cmath>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "init/cpu_initializers.hpp"
#include "init/tensor_initializers.hpp"
#include "ops/linear_op.hpp"

namespace ttml::modules {

namespace {
void validate_weight_shape(const ttnn::Shape& weight_shape) {
    if (weight_shape.rank() < 2U || weight_shape.rank() > 4U) {
        throw std::runtime_error("LinearLayer expects weight rank 2 through 4 with singleton leading dimensions.");
    }
    for (uint32_t dim = 0; dim < weight_shape.rank() - 2U; ++dim) {
        if (weight_shape[dim] != 1U) {
            throw std::runtime_error("LinearLayer expects weight to have singleton leading dimensions.");
        }
    }
}

void validate_parameter_shapes(const autograd::TensorPtr& weight, const autograd::TensorPtr& bias = nullptr) {
    const auto& weight_shape = weight->get_value().logical_shape();
    validate_weight_shape(weight_shape);

    if (bias == nullptr) {
        return;
    }

    const auto& bias_shape = bias->get_value().logical_shape();
    if (bias_shape.rank() < 1U || bias_shape.rank() > 4U) {
        throw std::runtime_error("LinearLayer expects bias rank 1 through 4 with singleton leading dimensions.");
    }
    for (uint32_t dim = 0; dim < bias_shape.rank() - 1U; ++dim) {
        if (bias_shape[dim] != 1U) {
            throw std::runtime_error("LinearLayer expects bias to have singleton leading dimensions.");
        }
    }
    if (bias_shape[-1] != weight_shape[-2]) {
        throw std::runtime_error("LinearLayer expects bias[-1] to match weight[-2].");
    }
}

ttml::autograd::TensorPtr create_weight(uint32_t in_features, uint32_t out_features) {
    auto weight_shape = ttnn::Shape({1, 1, out_features, in_features});
    auto weight = ttml::autograd::create_tensor();
    const float init_k = std::sqrt(1.F / static_cast<float>(in_features));
    init::uniform_init(weight, weight_shape, init::UniformRange{-init_k, init_k});
    return weight;
}
ttml::autograd::TensorPtr create_bias(uint32_t in_features, uint32_t out_features) {
    const float init_k = std::sqrt(1.F / static_cast<float>(in_features));
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
    validate_parameter_shapes(m_weight);
    if (has_bias) {
        const auto& weight_shape = m_weight->get_value().logical_shape();
        uint32_t in_features = weight_shape[-1];
        uint32_t out_features = weight_shape[-2];
        m_bias = create_bias(in_features, out_features);
    }
    register_tensors();
}

LinearLayer::LinearLayer(const autograd::TensorPtr& weight, const autograd::TensorPtr& bias) :
    m_weight(weight), m_bias(bias) {
    validate_parameter_shapes(m_weight, m_bias);
    register_tensors();
}

autograd::TensorPtr LinearLayer::get_weight() const {
    return m_weight;
}

autograd::TensorPtr LinearLayer::operator()(const autograd::TensorPtr& tensor) {
    return ops::linear_op(tensor, m_weight, m_bias);
}

}  // namespace ttml::modules
