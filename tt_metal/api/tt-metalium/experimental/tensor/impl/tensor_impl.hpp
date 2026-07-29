// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>

#include <tt_stl/span.hpp>
#include <vector>

/**
 * Functions in this file are internal utilities for Runtime Tensors.
 * They are exported out in the public API area as a transiet state,
 * many of them are used by ttnn python binding.
 * We should disperse these functions as public APIs in tensor_apis.hpp or make them private to tt_metal.
 */

namespace tt::tt_metal::tensor_impl {

// ======================================================================================
//                           Data reader, writer, and initializers
// ======================================================================================

// Converts physical data into logical data based on tensor spec
// - Physical data: Flat container of physical data corresponding to tensor spec
//   * Assumes that the physical data already matches tensor spec
//   * There is a bare minimum check that size of physical data matches size indicated by tensor_spec.physical_shape()
// - Logical data: Flat container of row major data corresponding to some ND logical shape
//   * Inverse of the private encode_tensor_data helper (padding / untilize)
//   * Resulting data is safe to be converted to python tensors or general consumption with just a ND logical shape
template <typename T>
std::vector<T> decode_tensor_data(ttsl::Span<const T> physical_data, const TensorSpec& tensor_spec);

}  // namespace tt::tt_metal::tensor_impl
