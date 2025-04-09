// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/dytanh_module.hpp"

#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

void DyTanhLayer::initialize_tensors(uint32_t features, float scale) {
    m_gain =
        autograd::create_tensor(core::ones(core::create_shape({1, 1, 1, features}), &autograd::ctx().get_device()));
    m_bias =
        autograd::create_tensor(core::zeros(core::create_shape({1, 1, 1, features}), &autograd::ctx().get_device()));
    m_scale = autograd::create_tensor(
        ttnn::experimental::mul(core::ones(core::create_shape({1, 1, 1, 1}), &autograd::ctx().get_device()), scale));
}

DyTanhLayer::DyTanhLayer(uint32_t features, float scale) {
    initialize_tensors(features, scale);

    create_name("dytanh");
    register_tensor(m_gain, "gain");
    register_tensor(m_bias, "bias");
    register_tensor(m_scale, "scale");
}

autograd::TensorPtr DyTanhLayer::operator()(const autograd::TensorPtr& tensor) {
    return ops::dytanh(tensor, m_scale, m_gain, m_bias);
}

}  // namespace ttml::modules
