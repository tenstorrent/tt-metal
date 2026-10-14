// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Largest tile-aligned divisor of Q+K+V that is <= 256 and fits the per-core L1 budget. Measured on a p300 at
// T=4096 (fused fw / fused bw, ms):
//   Q=K=V=2048: chunk 64 -> 1.12 / 1.17, 128 -> 0.82 / 1.05, 256 -> 0.42 / 0.94
//   Q=K=V=4096: chunk 64 -> 2.24 / 2.48, 128 -> 1.60 / 2.33, 256 -> 0.80 / 2.70
// The forward always prefers 256 and the training step (fw + bw, one chunk for both) is fastest at 256 for both
// widths; at Q=K=V=4096 the backward alone is 14% faster at 128. Chunk 32 makes the reader issue 64-byte row
// reads and is 2.4x slower than 256 on the backward.
uint32_t qkv_causal_conv1d_silu_chunk_size(uint32_t channels, const ttnn::distributed::MeshDevice& device);

// Returns (grad_input [1,T,Q+K+V] TILE, grad_tap0..3 shaped like the taps). History gets no gradient.
// channel_chunk_size = 0 picks qkv_causal_conv1d_silu_chunk_size; sequence_splits = 0 picks enough sequence
// splits to fill the core grid.
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> qkv_causal_conv1d_silu_bw(
    const ttnn::Tensor& grad_q,
    const ttnn::Tensor& grad_k,
    const ttnn::Tensor& grad_v,
    const ttnn::Tensor& input,
    const ttnn::Tensor& history,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    uint32_t channel_chunk_size = 0,
    uint32_t sequence_splits = 0);

}  // namespace ttml::metal
