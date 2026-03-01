// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/tensor/details/legacy_data_movements.hpp>
#include <tt-metalium/experimental/tensor/details/tensor_impl.hpp>
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
//                                         .to_host() and .to_device()
// ======================================================================================

HostTensor to_host(distributed::MeshCommandQueue& queue, const MeshTensor& tensor, bool blocking = true);

void copy_to_host(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    HostTensor& host_tensor,
    bool blocking = true);

MeshTensor to_device(
    distributed::MeshCommandQueue& queue,
    const HostTensor& tensor,
    ttsl::optional_reference<const MemoryConfig> memory_config = std::nullopt);

void copy_to_device(distributed::MeshCommandQueue& queue, const HostTensor& host_tensor, MeshTensor& device_tensor);

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
