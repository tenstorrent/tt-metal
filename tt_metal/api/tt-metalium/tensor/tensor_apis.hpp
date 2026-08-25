// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tile.hpp>

#include <tt_stl/span.hpp>

namespace tt::tt_metal {

// ======================================================================================
//                                  .to_layout()
// ======================================================================================

HostTensor to_layout(const HostTensor& tensor, Layout target_layout);
HostTensor to_tile_layout(const HostTensor& tensor, const Tile& tile);
HostTensor to_row_major_layout(const HostTensor& tensor);

// ======================================================================================
//                                  .to_dtype()
// ======================================================================================

HostTensor to_dtype(const HostTensor& input_tensor, DataType dtype);

// ======================================================================================
//                                  .to_tensor_spec()
// ======================================================================================

template <typename T>
HostTensor to_tensor_spec(const HostTensor& tensor, const TensorSpec& dest_spec);

// ======================================================================================
//                                  Utility functions
// ======================================================================================

namespace host_buffer {

// TODO(#40348): This function has single device assumptions over inheritely multi-device constructs.
HostBuffer get_host_buffer(const HostTensor& tensor);

template <typename T>
ttsl::Span<const T> get_as(const HostBuffer& buffer);

template <typename T>
ttsl::Span<T> get_as(HostBuffer& buffer);

template <typename T>
ttsl::Span<const T> get_as(const HostTensor& tensor);

template <typename T>
ttsl::Span<T> get_as(HostTensor& tensor);

}  // namespace host_buffer

}  // namespace tt::tt_metal
