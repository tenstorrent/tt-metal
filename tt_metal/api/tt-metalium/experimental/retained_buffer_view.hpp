// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <tt-metalium/mesh_buffer.hpp>

// Experimental and subject to change: this header carries no API-stability guarantee.
namespace tt::tt_metal::experimental::retained_buffer_view {

/// Creates an SRAM buffer view whose local addresses are `shard_offset` bytes into each owner shard.
///
/// The owner and view must be sharded L1 buffers with the same allocation mode and sub-device. The view core set must
/// be a subset of the owner core set, and each view interval must fit within its owner shard. The view retains the
/// owner allocation. Explicit owner deallocation invalidates the view.
std::shared_ptr<distributed::MeshBuffer> create(
    std::shared_ptr<distributed::MeshBuffer> owner,
    const distributed::MeshBufferConfig& mesh_buffer_config,
    const distributed::DeviceLocalBufferConfig& device_local_config,
    DeviceAddr shard_offset);

}  // namespace tt::tt_metal::experimental::retained_buffer_view
