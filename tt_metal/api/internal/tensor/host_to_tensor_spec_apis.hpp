// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>

namespace tt::tt_metal {

// ======================================================================================
//                         Host to_tensor_spec
// ======================================================================================
//
// Host-side TensorSpec conversion with an explicit pad value. Intended for tests;
// custom pad values are not part of the Runtime Tensor graduation surface.

// Same convention as HostTensor::from_vector: no default T; pad_value defaults to 0.
// T is the logical encode / pad element type.
// Unlike from_vector (T deduced from the buffer), callers must supply T explicitly
// (to_tensor_spec<float>(t, spec)) or pass a typed pad_value for deduction.
// Explicit instantiations: float, bfloat16, int32_t, uint32_t, uint16_t, uint8_t (same as from_vector).

/**
 * Convert **tensor** to match **dest_spec**, filling newly introduced pad regions
 * with **pad_value**.
 *
 * Return: a HostTensor whose TensorSpec exactly matches **dest_spec**.
 *
 * **pad_value** is injected when padded shape / alignment changes under **dest_spec**.
 * Typical causes: ROW_MAJOR → TILE layout conversion, or sharding / packing updates
 * that introduce non-logical fill bytes.
 *
 * pre-conditions:
 * - **tensor** and **dest_spec** logical shapes must match and have rank > 0.
 * - Neither source nor destination dtype may be FP8_E4M3.
 * - Pad/encode type **T** must match the working encode dtype.
 *
 * post-conditions:
 * - Result TensorSpec exactly matches **dest_spec**.
 */
template <typename T>
HostTensor to_tensor_spec(const HostTensor& tensor, const TensorSpec& dest_spec, T pad_value = 0);

}  // namespace tt::tt_metal
