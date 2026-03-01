// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

namespace tt::tt_metal::tensor_impl {

// ======================================================================================
// Legacy data movement APIs - these exist to support the async runtime.
// TODO: Consider removing or refactoring. See: #38592
// ======================================================================================

void copy_to_host(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    std::byte* dst,
    const std::optional<BufferRegion>& region = std::nullopt,
    bool blocking = true);

void copy_to_device(
    distributed::MeshCommandQueue& queue,
    const std::byte* src,
    MeshTensor& device_tensor,
    const std::optional<BufferRegion>& region = std::nullopt);

}  // namespace tt::tt_metal::tensor_impl
