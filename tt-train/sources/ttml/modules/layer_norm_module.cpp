// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layer_norm_module.hpp"

#include "core/tt_tensor_utils.hpp"
#include "ops/layernorm_op.hpp"

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
    if (m_use_composite_op) {
        return ops::composite_layernorm(tensor, m_gamma, m_beta);
    }
    return ops::layernorm_bw_fused(tensor, m_gamma, m_beta);
}

}  // namespace ttml::modules
// Step: 1, Loss: 11
// Compilation finished at step 1
// Full step time 87819.168 ms, cache entries: 91
// dataloader host only step time 0.033 ms
// dataloader step time 0.365 ms
// Step: 2, Loss: 9.625
// Full step time 247.624 ms, cache entries: 91
// dataloader host only step time 0.021 ms
// dataloader step time 0.106 ms
// Step: 3, Loss: 9.25
// Full step time 701.557 ms, cache entries: 91
// dataloader host only step time 0.04 ms
// dataloader step time 0.151 ms
// Step: 4, Loss: 9
// Full step time 702.953 ms, cache entries: 91
// dataloader host only step time 0.024 ms
// dataloader step time 0.094 ms
// Step: 5, Loss: 8.75
// Full step time 704.907 ms, cache entries: 91
// dataloader host only step time 0.022 ms
// dataloader step time 0.091 ms
// Step: 6, Loss: 8.5625
// Full step time 710.419 ms, cache entries: 91
// dataloader host only step time 0.031 ms
// dataloader step time 0.285 ms
// Step: 7, Loss: 8.3125
// Full step time 697.183 ms, cache entries: 91
// dataloader host only step time 0.022 ms
// dataloader step time 0.09 ms
// Step: 8, Loss: 8.0625
// Full step time 703.035 ms, cache entries: 91
// dataloader host only step time 0.02 ms
// dataloader step time 0.086 ms
// Step: 9, Loss: 7.96875
// Full step time 704.697 ms, cache entries: 91
// dataloader host only step time 0.018 ms
// dataloader step time 0.089 ms
// Step: 10, Loss: 7.71875
// 10 Steps training time: 5.877853 s, cache entries: 91
// 10 Steps training time: 116.086976 s, cache entries: 91

// Step: 1, Loss: 11
// Compilation finished at step 1
// Full step time 96690.899 ms, cache entries: 92
// dataloader host only step time 0.078 ms
// dataloader step time 0.169 ms
// Step: 2, Loss: 11.4375
// Full step time 262.002 ms, cache entries: 92
// dataloader host only step time 0.022 ms
// dataloader step time 0.086 ms
// Step: 3, Loss: 10.9375
// Full step time 693.403 ms, cache entries: 92
// dataloader host only step time 0.025 ms
// dataloader step time 0.12 ms
// Step: 4, Loss: 10.75
// Full step time 689.817 ms, cache entries: 92
// dataloader host only step time 0.023 ms
// dataloader step time 0.123 ms
// Step: 5, Loss: 10.6875
// Full step time 694.5 ms, cache entries: 92
// dataloader host only step time 0.022 ms
// dataloader step time 0.086 ms
// Step: 6, Loss: 10.5625
// Full step time 700.998 ms, cache entries: 92
// dataloader host only step time 0.044 ms
// dataloader step time 0.139 ms
// Step: 7, Loss: 10.4375
// Full step time 694.712 ms, cache entries: 92
// dataloader host only step time 0.026 ms
// dataloader step time 0.08 ms
// Step: 8, Loss: 10.3125
// Full step time 697.91 ms, cache entries: 92
// dataloader host only step time 0.033 ms
// dataloader step time 0.114 ms
// Step: 9, Loss: 10.125
// Full step time 688.773 ms, cache entries: 92
// dataloader host only step time 0.023 ms
// dataloader step time 0.103 ms
// Step: 10, Loss: 10.0625
// 10 Steps training time: 5.825129 s, cache entries: 92
// 10 Steps training time: 124.693414 s, cache entries: 92
