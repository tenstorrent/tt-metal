// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>

namespace tt::tt_metal {

class HostTensor;
class DeviceTensor;

namespace distributed {
class MeshCommandQueue;
}

// Returns true if the logical tensor data matches the physical tensor data:
// 1. Row major layout is used.
// 2. Logical 2D shape matches physical shape.
// Used for optimizing conversion operations.
bool logical_matches_physical(const TensorSpec& tensor_spec);

// Converts data type of a HostTensor to the specified dtype.
HostTensor to_dtype(const HostTensor& input_tensor, DataType dtype);

// High-level H2D transfer for HostTensor/DeviceTensor
void TransferToDevice(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    DeviceTensor& device_tensor,
    bool blocking = true);

// High-level D2H transfer for DeviceTensor/HostTensor
void TransferToHost(
    distributed::MeshCommandQueue& cq,
    const DeviceTensor& device_tensor,
    HostTensor& host_tensor,
    bool blocking = true);

}  // namespace tt::tt_metal
