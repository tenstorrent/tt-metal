// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_bw.hpp"

#include "device/ring_sdpa_bw_kv_device_operation.hpp"
#include "device/ring_sdpa_bw_q_device_operation.hpp"

namespace ttml::metal {
using RingDirection = ttml::metal::ops::ring_sdpa_bw::RingDirection;

void ring_sdpa_bw(
    std::vector<RingSDPABwChunk> chunks,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type,
    RingDirection ring_direction,
    ops::RingLoadBalancing load_balancing) {
    std::vector<ops::ring_sdpa_bw::q::RingSDPABwQChunk> q_chunks;
    std::vector<ops::ring_sdpa_bw::kv::RingSDPABwKVChunk> kv_chunks;
    q_chunks.reserve(chunks.size());
    kv_chunks.reserve(chunks.size());
    for (auto& chunk : chunks) {
        q_chunks.push_back(ops::ring_sdpa_bw::q::RingSDPABwQChunk{
            .grad_output = chunk.grad_output,
            .attn_output = std::move(chunk.attn_output),
            .query = chunk.query,
            .key = chunk.key,
            .value = chunk.value,
            .intermediates = chunk.intermediates,
            .grad_query = std::move(chunk.grad_query)});
        kv_chunks.push_back(ops::ring_sdpa_bw::kv::RingSDPABwKVChunk{
            .grad_output = std::move(chunk.grad_output),
            .query = std::move(chunk.query),
            .key = std::move(chunk.key),
            .value = std::move(chunk.value),
            .intermediates = std::move(chunk.intermediates),
            .grad_key = std::move(chunk.grad_key),
            .grad_value = std::move(chunk.grad_value)});
    }

    // Q kernel first: computes grad_Q and u_scaler = rowsum(dO * O)
    const auto bw_q_outputs = ttnn::prim::ttml_ring_sdpa_bw_q(
        std::move(q_chunks), ring_size, ring_axis, step, mask_type, ring_direction, load_balancing);

    // KV kernel: uses precomputed u_scaler, which each device filled for the chunk it computes
    ttnn::prim::ttml_ring_sdpa_bw_kv(
        std::move(kv_chunks),
        bw_q_outputs.u_scaler,
        ring_size,
        ring_axis,
        step,
        mask_type,
        ring_direction,
        load_balancing);
}

}  // namespace ttml::metal
