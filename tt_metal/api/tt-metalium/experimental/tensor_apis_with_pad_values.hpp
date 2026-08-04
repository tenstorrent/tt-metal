// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>
#include <vector>

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>

namespace tt::tt_metal {

// ======================================================================================
//                    Tensor APIs with explicit pad values
// ======================================================================================
//
// Runtime Tensor API offers a variety of ways to change/create the padding of an
// existing Tensor. The value of the padded value can be specified using this set of
// experimental APIs.
//
// These functions are in the experimental stage as specifying padding value is a
// questionable endeavor to begin with; access to these values should be treated as
// undefined behavior and usage of these functions should be avoided.
//
// T is the logical encode / pad element type.
// Explicit instantiations: float, bfloat16, int32_t, uint32_t, uint16_t, uint8_t.

/**
 * Same as HostTensor::from_span, but the padded values are filled with **pad_value**.
 */
template <typename T>
HostTensor host_tensor_from_span_with_pad_value(std::span<const T> buffer, TensorSpec spec, T pad_value);

/**
 * Same as HostTensor::from_vector, but the padded values are filled with **pad_value**.
 */
template <typename T>
HostTensor host_tensor_from_vector_with_pad_value(const std::vector<T>& buffer, TensorSpec spec, T pad_value);

/**
 * Same as HostTensor::from_vector, but the padded values are filled with **pad_value**.
 */
template <typename T>
HostTensor host_tensor_from_vector_with_pad_value(std::vector<T>&& buffer, TensorSpec spec, T pad_value);

/**
 * Same as to_tensor_spec, but the padded values are filled with **pad_value**.
 */
template <typename T>
HostTensor host_tensor_to_tensor_spec_with_pad_value(
    const HostTensor& tensor, const TensorSpec& dest_spec, T pad_value);

}  // namespace tt::tt_metal
