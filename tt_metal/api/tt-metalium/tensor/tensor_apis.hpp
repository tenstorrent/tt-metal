// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tile.hpp>

#include <tt_stl/span.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/tensor/spec/layout/layout.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>

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
//                                  GGUF weights
// ======================================================================================

// A GGUF tensor of `spec` (a GGUF dtype, shape [K, N]) from its blocks as a GGUF file stores them: N rows of
// K / block_elems blocks. Only bits move; the pages take exactly the blocks' bytes.
HostTensor from_gguf(ttsl::Span<const std::byte> blocks, const TensorSpec& spec);

// The blocks of a GGUF tensor, the exact inverse of from_gguf.
std::vector<std::byte> to_gguf(const HostTensor& tensor);

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
