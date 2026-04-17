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
//                        Transfer classification
// ======================================================================================

// Returns true if a H2D between the HostTensor and target MeshDevice is uniform.
// A transfer is uniform if the host shards cover the entire shape of the MeshDevice.
//
// Example of uniform transfer:
// HostTensor with a DistributedHostBuffer of shards [0, 0], [0, 1], [1, 0], [1, 1] (shape 2x2).
// MeshDevice of shape 2x2.
// Here the shards map exactly to the shape of the MeshDevice.
//
// Example of non-uniform transfers:
//
// 1: one to many replicas
// HostTensor with a single shard at [0,0].
// MeshDevice of shape 2x2.
// This is a replica-based non-uniform transfer.
//
// 2: partial coverage:
// HostTensor with a DistributedHostBuffer of shards [0, 0], and [1, 0]
// MeshDevice of shape 2x2.
// This is a partial coverage non-uniform transfer.
// Only opposite sides of the MeshDevice will receive new data.
bool is_uniform_write(const HostTensor& host_tensor, const distributed::MeshDevice& device);

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

void enqueue_write_tensor(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    MeshTensor& device_tensor,
    const CoreRangeSet& logical_core_filter);

// ======================================================================================
//                    Unit Tensor enqueue_read/write_tensor
// ======================================================================================

void enqueue_read_tensor(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    std::byte* dst,
    const std::optional<BufferRegion>& region = std::nullopt,
    bool blocking = true);

void enqueue_write_tensor(
    distributed::MeshCommandQueue& queue,
    const std::byte* src,
    MeshTensor& device_tensor,
    const std::optional<BufferRegion>& region = std::nullopt);

// ======================================================================================
//                Non-uniform enqueue_read/write_tensor
// ======================================================================================

// Data movement for tensors whose shards don't cover the entire MeshDevice.
// The host-side DistributedHostBuffer only populates a subset of MeshCoordinates,
// so the resulting DeviceStorage must track which coordinates were actually written.
namespace non_uniform_data_movement {

HostTensor enqueue_read_tensor(
    distributed::MeshCommandQueue& cq,
    const MeshTensor& device_tensor,
    std::span<const distributed::MeshCoordinate> coords,
    bool blocking = true);

void enqueue_read_tensor(
    distributed::MeshCommandQueue& cq,
    const MeshTensor& device_tensor,
    HostTensor& host_tensor,
    std::span<const distributed::MeshCoordinate> coords,
    bool blocking = true);

std::pair<MeshTensor, std::vector<distributed::MeshCoordinate>> enqueue_write_tensor(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    distributed::MeshDevice& mesh_device,
    ttsl::optional_reference<const MemoryConfig> memory_config = std::nullopt);

std::vector<distributed::MeshCoordinate> enqueue_write_tensor(
    distributed::MeshCommandQueue& cq, const HostTensor& host_tensor, MeshTensor& device_tensor);

std::vector<distributed::MeshCoordinate> enqueue_write_tensor(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    MeshTensor& device_tensor,
    const CoreRangeSet& logical_core_filter);

}  // namespace non_uniform_data_movement

// ======================================================================================
//                                  .to_layout()
// ======================================================================================

HostTensor to_layout(const HostTensor& tensor, Layout target_layout);

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
tt::stl::Span<const T> get_as(const HostBuffer& buffer);

template <typename T>
tt::stl::Span<T> get_as(HostBuffer& buffer);

template <typename T>
tt::stl::Span<const T> get_as(const HostTensor& tensor);

template <typename T>
tt::stl::Span<T> get_as(HostTensor& tensor);

}  // namespace host_buffer

}  // namespace tt::tt_metal
