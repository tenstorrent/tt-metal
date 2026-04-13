// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/host_buffer.hpp>

#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {

// Returns true if the logical tensor data matches the physical tensor data:
// 1. Row major layout is used.
// 2. Logical 2D shape matches physical shape.
// Used for optimizing conversion operations.
//
// TODO(#40348): Does this function make sense to be in the general utility?
// If so, better name?
bool logical_matches_physical(const TensorSpec& tensor_spec);

namespace host_buffer {

HostBuffer get_host_buffer(const HostTensor& tensor);

template <typename T>
tt::stl::Span<const T> get_as(const HostBuffer& buffer);

template <typename T>
tt::stl::Span<T> get_as(HostBuffer& buffer);

template <typename T>
tt::stl::Span<const T> get_as(const HostTensor& tensor);

template <typename T>
tt::stl::Span<T> get_as(HostTensor& tensor);

}  // namespace host_buffer

}  // namespace tt::tt_metal
