// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt_stl/span.hpp>

/**
 * Private tensor_impl utilities for Runtime Tensors.
 *
 * Not part of the installed public API. Prefer HostTensor / MeshTensor /
 * tensor_apis surfaces at call sites outside this directory.
 */

namespace tt::tt_metal::tensor_impl {

template <typename T>
std::vector<T> encode_tensor_data(ttsl::Span<const T> logical_data, const TensorSpec& tensor_spec, T pad_value = 0);

}  // namespace tt::tt_metal::tensor_impl
