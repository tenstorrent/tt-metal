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

namespace tt::tt_metal::host_buffer {

HostBuffer get_host_buffer(const HostTensor& tensor);

template <typename T>
tt::stl::Span<const T> get_as(const HostBuffer& buffer);

template <typename T>
tt::stl::Span<T> get_as(HostBuffer& buffer);

template <typename T>
tt::stl::Span<const T> get_as(const HostTensor& tensor);

template <typename T>
tt::stl::Span<T> get_as(HostTensor& tensor);

}  // namespace tt::tt_metal::host_buffer
