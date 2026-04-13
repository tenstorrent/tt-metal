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

}  // namespace tt::tt_metal
