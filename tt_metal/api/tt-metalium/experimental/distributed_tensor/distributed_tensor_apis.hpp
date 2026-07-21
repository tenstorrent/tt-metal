// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>

#include <tt_stl/optional_reference.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {
class MemoryConfig;
}

namespace tt::tt_metal {

// ======================================================================================
//                        Factories
// ======================================================================================

// NOLINTNEXTLINE(readability-redundant-declaration)
HostTensor host_tensor_from_buffer_with_topology(
    DistributedHostBuffer buffer, TensorSpec spec, TensorTopology topology);

// NOLINTNEXTLINE(readability-redundant-declaration)
MeshTensor mesh_tensor_from_buffer_with_topology(
    distributed::MeshBuffer mesh_buffer, TensorSpec spec, TensorTopology topology);

// NOLINTNEXTLINE(readability-redundant-declaration)
MeshTensor allocate_mesh_tensor_on_device_with_topology(
    distributed::MeshDevice& mesh_device, const TensorSpec& spec, const TensorTopology& topology);

// ======================================================================================
//                        Topology accessors
// ======================================================================================

const TensorTopology& get_tensor_topology(const MeshTensor& tensor);
const TensorTopology& get_tensor_topology(const HostTensor& tensor);

void update_tensor_topology(MeshTensor& tensor, TensorTopology topology);
void update_tensor_topology(HostTensor& tensor, TensorTopology topology);

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

}  // namespace non_uniform_data_movement

}  // namespace tt::tt_metal
