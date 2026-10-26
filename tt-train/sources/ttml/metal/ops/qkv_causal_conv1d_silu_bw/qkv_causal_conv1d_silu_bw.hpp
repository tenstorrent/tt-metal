// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Largest tile-aligned divisor of Q+K+V that is <= 256 and fits the per-core L1 budget. Measured on a p300 with a
// 110-core compute grid at T=4096 with automatic row_splits (fused fw / fused bw, ms):
//   Q=K=V=2048: chunk 64 -> 1.12 / 1.12, 128 -> 0.82 / 1.00, 256 -> 0.42 / 0.92
//   Q=K=V=4096: chunk 64 -> 2.24 / 2.26, 128 -> 1.60 / 1.92, 256 -> 0.80 / 1.73
// 256 is fastest for both directions at both widths. Chunk 32 makes the reader issue 64-byte row reads and is
// 2.4x slower than 256 on the backward.
uint32_t qkv_causal_conv1d_silu_chunk_size(uint32_t channels, const ttnn::distributed::MeshDevice& device);

// True when a caller-fixed channel_chunk_size is a tile-aligned divisor of channels whose backward L1 budget fits the
// device. The forward checks its own budget, which is smaller; the autograd op uses this to reject at forward time a
// chunk that would only fail in backward().
bool qkv_causal_conv1d_silu_bw_chunk_fits(
    uint32_t channels,
    uint32_t channel_chunk_size,
    const ttnn::distributed::MeshDevice& device,
    bool compute_tap_grads = true);

struct QkvCausalConv1dSiluBackwardWorkSplit {
    uint32_t channel_chunk_size;
    uint32_t row_splits;
};

// Picks the (channel_chunk_size, row_splits) pair with the lowest modelled cost over `cores` cores; a nonzero
// argument fixes that dimension. Candidate chunks are the tile-aligned divisors of channels <= 256 that fit
// l1_bytes. See the cost model in qkv_causal_conv1d_silu_bw.cpp.
QkvCausalConv1dSiluBackwardWorkSplit qkv_causal_conv1d_silu_bw_work_split(
    uint32_t channels,
    uint32_t row_count,
    uint32_t cores,
    uint64_t l1_bytes,
    uint32_t channel_chunk_size = 0,
    uint32_t row_splits = 0,
    bool compute_tap_grads = true);

// input [1,T,Q+K+V], grad_q/k/v [1,T,.] and the taps are TILE bf16; history [1,3,Q+K+V] is ROW_MAJOR bf16.
// Returns (grad_input [1,T,Q+K+V] TILE, grad_tap0..3 shaped like the taps). History gets no gradient.
// Unlike the forward there is no sequence-parallel or actual_start support: each device's input is one
// independent sequence that starts from history, so do not pair it with a sequence-parallel forward.
// channel_chunk_size = 0 and row_splits = 0 are picked by qkv_causal_conv1d_silu_bw_work_split.
// With compute_tap_grads = false the tap gradients are nullopt and their work is skipped.
// preallocated_grad_input must not share a buffer with input: a split's halo step reads input rows that the next
// split's writer has already overwritten.
std::tuple<
    ttnn::Tensor,
    std::optional<ttnn::Tensor>,
    std::optional<ttnn::Tensor>,
    std::optional<ttnn::Tensor>,
    std::optional<ttnn::Tensor>>
qkv_causal_conv1d_silu_bw(
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
    uint32_t row_splits = 0,
    bool compute_tap_grads = true,
    const std::optional<ttnn::Tensor>& preallocated_grad_input = std::nullopt);

}  // namespace ttml::metal
