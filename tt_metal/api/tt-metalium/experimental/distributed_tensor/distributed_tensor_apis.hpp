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

// Construct Runtime Tensor with TensorTopology.

// NOLINTNEXTLINE(readability-redundant-declaration)
HostTensor host_tensor_from_buffer_with_topology(
    DistributedHostBuffer buffer, TensorSpec spec, TensorTopology topology);

// NOLINTNEXTLINE(readability-redundant-declaration)
MeshTensor mesh_tensor_from_buffer_with_topology(
    distributed::MeshBuffer mesh_buffer, TensorSpec spec, TensorTopology topology);

// NOLINTNEXTLINE(readability-redundant-declaration)
MeshTensor allocate_mesh_tensor_on_device_with_topology(
    distributed::MeshDevice& mesh_device, TensorSpec spec, TensorTopology topology);

// ======================================================================================
//                        Topology accessors
// ======================================================================================

// Get and mutate the TensorTopology of a Runtime Tensor.

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

/**
 * Read the data of **device_tensor** at the given **coords** into a new HostTensor.
 *
 * Return: a new HostTensor with the data from **device_tensor** populated only at the given **coords**.
 *
 * pre-condition: **coords** must be in bounds of **cq**'s MeshDevice.
 */
HostTensor enqueue_read_tensor(
    distributed::MeshCommandQueue& cq,
    const MeshTensor& device_tensor,
    std::span<const distributed::MeshCoordinate> coords,
    bool blocking = true);

/**
 * Read the data of **device_tensor** at the given **coords** into the given **host_tensor**.
 *
 * pre-conditions:
 * - **host_tensor** must have populated shards that cover the given **coords**
 *   (extra shards beyond **coords** may be present, but see post-condition).
 * - **coords** must be in bounds of **cq**'s MeshDevice.
 *
 * post-conditions:
 * - **host_tensor** is rebuilt from a buffer that contains only **coords**.
 * - Shards at **coords** are overwritten with data from **device_tensor**.
 * - Any host shards outside **coords** are discarded (shed), not preserved.
 */
void enqueue_read_tensor(
    distributed::MeshCommandQueue& cq,
    const MeshTensor& device_tensor,
    HostTensor& host_tensor,
    std::span<const distributed::MeshCoordinate> coords,
    bool blocking = true);

/**
 * Allocate a MeshTensor on **mesh_device** and write populated shards of **host_tensor** into it.
 *
 * Allocates using **host_tensor**'s TensorSpec (dtype / page config / alignment / topology),
 * optionally overriding only MemoryConfig, then delegates to the in-place overload below.
 *
 * Return: `{allocated MeshTensor, coordinates that were written}`.
 *
 * pre-conditions:
 * - **mesh_device** must be **cq**'s MeshDevice (`*cq.device()`).
 * - Host shard coordinates must be in bounds of **mesh_device**, unless host storage shape is
 *   exactly (1,1) (replicate path).
 * - If **memory_config** is set, it must form a valid TensorLayout with the host dtype / page
 *   config / alignment (validated by TensorLayout), and the resulting packed per-device size
 *   (`TensorSpec::compute_packed_buffer_size_bytes()`) must fit each populated host shard
 *   (exact match required for the (1,1) replicate path).
 * - If **memory_config** is omitted, the same size constraint applies using the host TensorSpec.
 *
 * post-conditions:
 * - Same write semantics as the in-place overload (1x1 replicate vs partial shard write).
 */
std::pair<MeshTensor, std::vector<distributed::MeshCoordinate>> enqueue_write_tensor(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    distributed::MeshDevice& mesh_device,
    ttsl::optional_reference<const MemoryConfig> memory_config = std::nullopt);

/**
 * Write populated shards of **host_tensor** into **device_tensor**.
 *
 * Return: the mesh coordinates that were written.
 *
 * pre-conditions:
 * - **host_tensor** and **device_tensor** must have matching logical shape, dtype, and page config.
 * - **device_tensor** must be allocated on **cq**'s MeshDevice (`*cq.device()`).
 * - Host shard coordinates must be in bounds of **device_tensor**'s MeshDevice, unless host
 *   storage shape is exactly (1,1) (replicate path).
 * - Each populated host shard's byte size must fit **device_tensor**'s per-device allocation
 *   (`device_tensor.tensor_spec().compute_packed_buffer_size_bytes()`). The (1,1) replicate path
 *   requires an exact size match.
 *
 * post-conditions:
 * - If host storage shape is (1,1) and smaller than the device mesh, that shard is replicated
 *   across the entire mesh, topology becomes fully replicated, and all mesh coordinates are
 *   returned.
 * - Otherwise, only populated host shards are written; those coordinates are returned;
 *   **device_tensor** is rebuilt with the host topology and the existing device memory config.
 */
std::vector<distributed::MeshCoordinate> enqueue_write_tensor(
    distributed::MeshCommandQueue& cq, const HostTensor& host_tensor, MeshTensor& device_tensor);

}  // namespace non_uniform_data_movement

}  // namespace tt::tt_metal
