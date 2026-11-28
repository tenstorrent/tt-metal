// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string_view>
#include <tt-metalium/data_formats/tile_constants.hpp>
#include <tt_stl/assert.hpp>
#include <utility>

#include "metal/common/const_utils.hpp"
#include "metal/common/tensor_validation.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::metal::ops {

// ZigZag: with N = ring_size, device d holds global blocks [d | 2N-1-d]; chunks[0] = block d, chunks[1] = block 2N-1-d.
// Same device layout as ttnn's ring_joint_sdpa is_balanced and ring_distributed_sdpa chunk_1 / chunk_2.
// At step >= 1, with src = get_ring_source_device(d, step), the device computes chunk 0 if src < d, chunk 1 if src > d,
// unmasked. At step 0 the caller makes one causal call with RingLoadBalancing::None; at each step >= 1 it adds an
// unmasked call for Q chunk 1 x K/V chunk 0.
enum class RingLoadBalancing : uint8_t { None = 0, ZigZag = 1 };

inline uint32_t get_ring_chunk_count(RingLoadBalancing load_balancing) {
    switch (load_balancing) {
        case RingLoadBalancing::None: return 1U;
        case RingLoadBalancing::ZigZag: return 2U;
    }
    TT_THROW("Unknown load balancing {}", static_cast<int>(load_balancing));
}

inline uint32_t get_ring_source_device(
    uint32_t device_ring_id,
    uint32_t step,
    uint32_t ring_size,
    ttnn_fixed::distributed::RingShiftDirection ring_direction) {
    switch (ring_direction) {
        case ttnn_fixed::distributed::RingShiftDirection::Backward: return (device_ring_id + step) % ring_size;
        case ttnn_fixed::distributed::RingShiftDirection::Forward:
            // step % ring_size first: d - s would wrap in uint32 when s > d, which is wrong unless ring_size is 2^k.
            return (device_ring_id + ring_size - (step % ring_size)) % ring_size;
    }
    TT_THROW("Unknown ring direction {}", static_cast<int>(ring_direction));
}

// Determine if a device should execute at this ring step and which mask type to use.
// Returns: (should_execute, mask_type_to_use)
//
// Causal logic on the source chunk, src = get_ring_source_device(d, step):
// - src == d: diagonal chunk, apply causal mask
// - src < d:  earlier chunk, full attention (no mask)
// - src > d:  later chunk, fully masked, skip
inline std::pair<bool, AttentionMaskType> get_device_execution_info(
    uint32_t device_ring_id,
    uint32_t step,
    uint32_t ring_size,
    AttentionMaskType mask_type,
    ttnn_fixed::distributed::RingShiftDirection ring_direction) {
    if (mask_type != AttentionMaskType::Causal) {
        // Non-causal: all devices execute with full attention
        return {true, AttentionMaskType::None};
    }

    const uint32_t src_device = get_ring_source_device(device_ring_id, step, ring_size, ring_direction);

    if (src_device == device_ring_id) {
        return {true, AttentionMaskType::Causal};  // Diagonal: use causal mask
    } else if (src_device < device_ring_id) {
        return {true, AttentionMaskType::None};  // Earlier: full attention (no mask)
    } else {
        return {false, AttentionMaskType::None};  // Later: skip
    }
}

struct RingStepWork {
    bool execute = false;
    AttentionMaskType mask_type = AttentionMaskType::None;
    uint32_t chunk = 0;

    bool operator==(const RingStepWork&) const = default;
};

// fw, bw_q and bw_kv must all use this function so that each device gets the same chunk at each step: bw_kv reads the
// u_scaler that bw_q wrote for that chunk.
inline RingStepWork get_ring_step_work(
    uint32_t device_ring_id,
    uint32_t step,
    uint32_t ring_size,
    AttentionMaskType mask_type,
    ttnn_fixed::distributed::RingShiftDirection ring_direction,
    RingLoadBalancing load_balancing) {
    switch (load_balancing) {
        case RingLoadBalancing::None: {
            const auto [execute, effective_mask] =
                get_device_execution_info(device_ring_id, step, ring_size, mask_type, ring_direction);
            return {.execute = execute, .mask_type = effective_mask, .chunk = 0U};
        }
        case RingLoadBalancing::ZigZag: {
            TT_FATAL(
                mask_type == AttentionMaskType::Causal,
                "Zig-zag load balancing is only defined for causal attention, got mask type {}",
                static_cast<int>(mask_type));
            const uint32_t src_device = get_ring_source_device(device_ring_id, step, ring_size, ring_direction);
            if (src_device == device_ring_id) {
                return {.execute = false, .mask_type = AttentionMaskType::None, .chunk = 0U};
            }
            return {
                .execute = true,
                .mask_type = AttentionMaskType::None,
                .chunk = src_device < device_ring_id ? 0U : 1U,
            };
        }
    }
    TT_THROW("Unknown load balancing {}", static_cast<int>(load_balancing));
}

// The work of the device at `mesh_coordinate` at this ring step. The ring SDPA program factories build one program per
// device from it, and an empty one for a device that sits the step out.
template <typename RingAttrs>
inline RingStepWork work_for_device_at_step(const RingAttrs& attrs, const ttnn::MeshCoordinate& mesh_coordinate) {
    return get_ring_step_work(
        mesh_coordinate[attrs.ring_axis],
        attrs.step,
        attrs.ring_size,
        attrs.mask_type,
        attrs.ring_direction,
        attrs.load_balancing);
}

// Common attribute validation for the ring SDPA device operations. The ring factories
// dispatch straight to the SDPA program factories, so the SDPA device operations'
// validation never runs for the ring path — these checks are its replacement.
template <typename RingAttrs>
inline void validate_ring_attributes(const RingAttrs& attrs, const ttnn::Tensor& query) {
    TT_FATAL(query.device() != nullptr, "Query tensor must be on a mesh device");
    TT_FATAL(attrs.ring_size > 0, "Ring size must be > 0");
    TT_FATAL(attrs.step < attrs.ring_size, "Step {} must be < ring_size {}", attrs.step, attrs.ring_size);
    // The ring path has no way to feed a mask tensor to the underlying SDPA kernels, so
    // Arbitrary would silently degrade to full attention.
    TT_FATAL(
        attrs.mask_type != AttentionMaskType::Arbitrary,
        "Ring SDPA does not support Arbitrary attention masks, use None or Causal");

    const auto mesh_shape = query.device()->shape();
    TT_FATAL(
        attrs.ring_axis < mesh_shape.dims(),
        "Ring axis {} must be < mesh dimensions {}",
        attrs.ring_axis,
        mesh_shape.dims());
    TT_FATAL(
        attrs.ring_size == mesh_shape[attrs.ring_axis],
        "Ring size {} must match mesh extent {} along ring axis {}",
        attrs.ring_size,
        mesh_shape[attrs.ring_axis],
        attrs.ring_axis);
}

template <typename RingAttrs>
inline void validate_ring_load_balancing(const RingAttrs& attrs, size_t chunk_count) {
    TT_FATAL(
        chunk_count == get_ring_chunk_count(attrs.load_balancing),
        "Load balancing {} expects {} chunks, got {}",
        static_cast<int>(attrs.load_balancing),
        get_ring_chunk_count(attrs.load_balancing),
        chunk_count);
    switch (attrs.load_balancing) {
        case RingLoadBalancing::None: return;
        case RingLoadBalancing::ZigZag:
            TT_FATAL(
                attrs.mask_type == AttentionMaskType::Causal,
                "Zig-zag load balancing requires a causal mask, got mask type {}",
                static_cast<int>(attrs.mask_type));
            TT_FATAL(
                attrs.step != 0U,
                "Zig-zag load balancing is not defined for step 0 (every device holds its own K/V); run step 0 as "
                "one causal call over the full local sequence");
            return;
    }
    TT_THROW("Unknown load balancing {}", static_cast<int>(attrs.load_balancing));
}

// Each device builds its program from the chunk it computes, and chunk-sized buffers (u_scaler) serve every chunk.
inline void validate_same_spec(
    const ttnn::Tensor& tensor, const ttnn::Tensor& reference, std::string_view name, size_t c) {
    TT_FATAL(
        tensor.device() == reference.device(), "Chunk {} '{}' must be on the same mesh device as chunk 0", c, name);
    TT_FATAL(
        tensor.tensor_spec() == reference.tensor_spec(),
        "Chunk {} '{}' spec {} must match chunk 0's spec {}",
        c,
        name,
        tensor.tensor_spec(),
        reference.tensor_spec());
}

// Per-tensor device contract of the reused SDPA kernels: on device, allocated,
// co-located with the query, rank 4, TILE layout, expected dtype, interleaved.
inline void validate_sdpa_tensor(
    const ttnn::Tensor& tensor,
    std::string_view name,
    const ttnn::Tensor& query,
    ttnn::DataType required_dtype = ttnn::DataType::BFLOAT16) {
    check_device_tensor(tensor, "RingSDPA", name, {.dtypes = {required_dtype}});
    TT_FATAL(tensor.device() == query.device(), "Tensor '{}' must be on the same mesh device as the query", name);
    TT_FATAL(
        tensor.logical_shape().rank() == 4U,
        "Tensor '{}' must have rank 4, got rank {}",
        name,
        tensor.logical_shape().rank());
}

// Q/K/V contract shared by the ring forward and backward operations: the per-tensor
// device contract plus the shape consistency the reused SDPA kernels assume.
inline void validate_ring_qkv(const ttnn::Tensor& query, const ttnn::Tensor& key, const ttnn::Tensor& value) {
    validate_sdpa_tensor(query, "Query", query);
    validate_sdpa_tensor(key, "Key", query);
    validate_sdpa_tensor(value, "Value", query);

    const auto [qB, qH, qS, qE] = query.logical_shape().to_array_4D();
    const auto [kB, kH, kS, kE] = key.logical_shape().to_array_4D();
    const auto [vB, vH, vS, vE] = value.logical_shape().to_array_4D();

    TT_FATAL(qH > 0 && kH > 0, "Number of heads must be greater than zero. Got q_heads={}, kv_heads={}", qH, kH);
    TT_FATAL(
        qH % kH == 0,
        "Number of query heads ({}) must be divisible by number of key/value heads ({}) for grouped attention",
        qH,
        kH);
    TT_FATAL(kH == vH, "Key and Value must have the same number of heads. Got key_heads={}, value_heads={}", kH, vH);
    TT_FATAL(
        qB == kB && qB == vB && qS == kS && qS == vS && qE == kE,
        "Query/Key/Value must agree on batch, sequence length, and Q/K inner dim. Got Query={}, Key={}, Value={}",
        query.logical_shape(),
        key.logical_shape(),
        value.logical_shape());
}

// Contract for tensors shaped like the attention output — grad_output, attn_output,
// and preallocated forward output: query B/H/S with the value's inner dim.
inline void validate_output_like_tensor(
    const ttnn::Tensor& tensor, std::string_view name, const ttnn::Tensor& query, const ttnn::Tensor& value) {
    validate_sdpa_tensor(tensor, name, query);
    const auto query_dims = query.logical_shape().to_array_4D();
    const auto value_inner_dim = value.logical_shape().to_array_4D()[3];
    const ttnn::Shape expected_shape{query_dims[0], query_dims[1], query_dims[2], value_inner_dim};
    TT_FATAL(
        tensor.logical_shape() == expected_shape,
        "Tensor '{}' shape {} must be {} (query B/H/S with value inner dim)",
        name,
        tensor.logical_shape(),
        expected_shape);
}

// Contract for preallocated gradient tensors: same device contract and shape as the
// tensor they are the gradient of.
inline void validate_grad_like_tensor(
    const ttnn::Tensor& tensor, std::string_view name, const ttnn::Tensor& reference, const ttnn::Tensor& query) {
    validate_sdpa_tensor(tensor, name, query);
    TT_FATAL(
        tensor.logical_shape() == reference.logical_shape(),
        "Tensor '{}' shape {} must match the shape {} of the tensor it is the gradient of",
        name,
        tensor.logical_shape(),
        reference.logical_shape());
}

// The SDPA kernels read the logsumexp from column 0 of a (B, H, S, TILE_WIDTH) FP32
// tile-layout tensor; anything else is silently misinterpreted on device.
inline void validate_intermediates_tensor(const ttnn::Tensor& intermediates, const ttnn::Tensor& query) {
    TT_FATAL(
        intermediates.dtype() == ttnn::DataType::FLOAT32,
        "Intermediates must be FLOAT32, got {}",
        intermediates.dtype());
    TT_FATAL(
        intermediates.layout() == ttnn::Layout::TILE,
        "Intermediates must be TILE layout, got {}",
        intermediates.layout());
    const auto [batch, heads, seq_len, dim] = query.logical_shape().to_array_4D();
    const ttnn::Shape expected_shape{batch, heads, seq_len, tt::constants::TILE_WIDTH};
    TT_FATAL(
        intermediates.logical_shape() == expected_shape,
        "Intermediates shape {} must be {} (one logsumexp tile column per query row)",
        intermediates.logical_shape(),
        expected_shape);
}

}  // namespace ttml::metal::ops
