// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/tile.hpp>

#include <tt_stl/optional_reference.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {
class MemoryConfig;
}

namespace tt::tt_metal {

// ======================================================================================
//                   Uniform enqueue_read/write_tensor
// ======================================================================================

HostTensor enqueue_read_tensor(
    distributed::MeshCommandQueue& cq, const MeshTensor& device_tensor, bool blocking = true);

void enqueue_read_tensor(
    distributed::MeshCommandQueue& cq, const MeshTensor& device_tensor, HostTensor& host_tensor, bool blocking = true);

MeshTensor enqueue_write_tensor(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    distributed::MeshDevice& mesh_device,
    ttsl::optional_reference<const MemoryConfig> memory_config = std::nullopt);

void enqueue_write_tensor(distributed::MeshCommandQueue& cq, const HostTensor& host_tensor, MeshTensor& device_tensor);

// ======================================================================================
//                                  .to_layout()
// ======================================================================================

HostTensor to_layout(const HostTensor& tensor, Layout target_layout);
HostTensor to_tile_layout(const HostTensor& tensor, const Tile& tile);
HostTensor to_row_major_layout(const HostTensor& tensor);

// ======================================================================================
//                                  .pad() and .unpad()
// ======================================================================================

HostTensor pad(
    const HostTensor& tensor, const Shape& output_padded_shape, const Shape& input_tensor_start, float pad_value);

HostTensor unpad(const HostTensor& tensor, const Shape& output_tensor_start, const Shape& output_tensor_end);

HostTensor pad_to_tile(const HostTensor& input_tensor, float pad_value);

HostTensor unpad_from_tile(const HostTensor& input_tensor, const Shape& output_tensor_shape);

// ======================================================================================
//                                  .to_dtype()
// ======================================================================================

HostTensor to_dtype(const HostTensor& input_tensor, DataType dtype);

// Same convention as HostTensor::from_vector: no default T; pad_value defaults to 0.
// T is the logical encode / pad element type.
// Unlike from_vector (T deduced from the buffer), callers must supply T explicitly
// (to_tensor_spec<float>(t, spec)) or pass a typed pad_value for deduction.
// Explicit instantiations: float, bfloat16, int32_t, uint32_t, uint16_t, uint8_t (same as from_vector).
template <typename T>
HostTensor to_tensor_spec(const HostTensor& tensor, const TensorSpec& dest_spec, T pad_value = 0);

// ======================================================================================
//                                  Utility functions
// ======================================================================================

// Returns true if the logical tensor data matches the physical tensor data:
// 1. Row major layout is used.
// 2. Logical 2D shape matches physical shape.
// Used for optimizing conversion operations.
//
// TODO(#40348): This is an internal utility function, we should close this up.
bool logical_matches_physical(const TensorSpec& tensor_spec);

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
