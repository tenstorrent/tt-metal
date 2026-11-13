// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string_view>
#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>
#include <utility>

#include "metal/common/const_utils.hpp"
#include "metal/common/tensor_validation.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::metal::ops {

// Source device of the K/V a device holds at a ring step:
// - Backward ring direction: src = (d + s) mod ring_size
// - Forward ring direction:  src = (d - s) mod ring_size
inline uint32_t get_ring_source_device(
    uint32_t device_ring_id,
    uint32_t step,
    uint32_t ring_size,
    ttnn_fixed::distributed::RingShiftDirection ring_direction) {
    if (ring_direction == ttnn_fixed::distributed::RingShiftDirection::Backward) {
        return (device_ring_id + step) % ring_size;
    }
    // Reduce step mod ring_size before subtracting so the uint32 expression cannot
    // underflow: plain (d - s + ring_size) wraps mod 2^32 when s > d, which is only
    // correct for power-of-2 ring sizes.
    return (device_ring_id + ring_size - (step % ring_size)) % ring_size;
}

// Mask a device applies at one ring step, keyed by where its current K/V came from
// relative to it in the ring. std::nullopt means the device skips the step.
struct RingStepMasks {
    std::optional<AttentionMaskType> src_before = std::nullopt;  // src < d
    std::optional<AttentionMaskType> src_self = std::nullopt;    // src == d
    std::optional<AttentionMaskType> src_after = std::nullopt;   // src > d

    bool operator==(const RingStepMasks&) const = default;

    // Explicit so that a skip never hashes like AttentionMaskType::None.
    ttsl::hash::hash_t to_hash() const {
        const auto encode = [](const std::optional<AttentionMaskType>& mask) -> uint32_t {
            return mask.has_value() ? 1U + static_cast<uint32_t>(mask.value()) : 0U;
        };
        return ttsl::hash::hash_objects(
            ttsl::hash::hash_t{0}, encode(src_before), encode(src_self), encode(src_after));
    }
};

// Contiguous sharding (rank d holds the d-th slice of the sequence). Causal: the diagonal
// chunk is causal, earlier chunks are unmasked, later chunks are skipped. Any other mask
// applies to every chunk.
inline RingStepMasks contiguous_ring_step_masks(AttentionMaskType mask_type) {
    if (mask_type == AttentionMaskType::Causal) {
        return {.src_before = AttentionMaskType::None, .src_self = AttentionMaskType::Causal};
    }
    return {.src_before = mask_type, .src_self = mask_type, .src_after = mask_type};
}

// Mask a device applies at this ring step, or std::nullopt if it skips the step. Shared by
// the ring_sdpa_fw and ring_sdpa_bw program factories, which must agree: the backward relies
// on skipping exactly the (device, step) pairs the forward skipped.
inline std::optional<AttentionMaskType> get_ring_step_mask(
    uint32_t device_ring_id,
    uint32_t step,
    uint32_t ring_size,
    ttnn_fixed::distributed::RingShiftDirection ring_direction,
    const RingStepMasks& masks) {
    const uint32_t src_device = get_ring_source_device(device_ring_id, step, ring_size, ring_direction);
    if (src_device < device_ring_id) {
        return masks.src_before;
    }
    return src_device == device_ring_id ? masks.src_self : masks.src_after;
}

// Contiguous schedule as (should_execute, mask_type_to_use).
inline std::pair<bool, AttentionMaskType> get_device_execution_info(
    uint32_t device_ring_id,
    uint32_t step,
    uint32_t ring_size,
    AttentionMaskType mask_type,
    ttnn_fixed::distributed::RingShiftDirection ring_direction) {
    const auto mask =
        get_ring_step_mask(device_ring_id, step, ring_size, ring_direction, contiguous_ring_step_masks(mask_type));
    return {mask.has_value(), mask.value_or(AttentionMaskType::None)};
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
    for (const auto& mask : {attrs.step_masks.src_before, attrs.step_masks.src_self, attrs.step_masks.src_after}) {
        TT_FATAL(
            mask != AttentionMaskType::Arbitrary,
            "Ring SDPA does not support Arbitrary attention masks, use None or Causal");
    }

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
