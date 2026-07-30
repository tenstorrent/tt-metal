// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>

namespace tt::tt_metal::experimental {

// Allocates a `MeshBuffer` and returns it by value.
//
// Unlike `MeshBuffer::create`, which returns a `std::shared_ptr<MeshBuffer>`, this API hands back an owning value.
// The caller decides how the buffer is owned -- keep it on the stack, move it into a member, or wrap it in
// `std::unique_ptr`/`std::shared_ptr` -- rather than being forced into shared ownership by the creation API. See
// issue #38691.
//
// The allocation semantics are identical to `MeshBuffer::create`.
distributed::MeshBuffer allocate_mesh_buffer(
    const distributed::MeshBufferConfig& mesh_buffer_config,
    const distributed::DeviceLocalBufferConfig& device_local_config,
    distributed::MeshDevice* mesh_device,
    std::optional<DeviceAddr> address = std::nullopt);

}  // namespace tt::tt_metal::experimental
