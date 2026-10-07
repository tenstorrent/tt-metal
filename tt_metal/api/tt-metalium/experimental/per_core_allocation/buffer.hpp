// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal::experimental::per_core_allocation {

// Buffer free functions — friended by Buffer to access private per-core state.

bool is_per_core_allocation(const Buffer& buffer);
DeviceAddr get_per_core_address(const Buffer& buffer, CoreCoord core);
const std::unordered_map<CoreCoord, DeviceAddr>& get_per_core_addresses(const Buffer& buffer);
void copy_per_core_addresses(Buffer& dst, const Buffer& src);

// Base address of ``buffer``'s shard on ``core``, for either allocation mode.
//
// Per-core-allocated buffers give each core an INDEPENDENT shard address, while
// Buffer::address() returns only cores[0]'s. Host data movement must target the same address
// the kernel reads, otherwise a relocated core reads/writes at the wrong offset. Falls back to
// Buffer::address() for ordinary lockstep buffers, so callers need no mode check.
DeviceAddr get_shard_base_address(const Buffer& buffer, CoreCoord core);

// BufferShardingArgs free functions.

BufferShardingArgs& set_per_core_allocation(BufferShardingArgs& args, bool enable);
bool is_per_core_allocation(const BufferShardingArgs& args);

// Uniform address: a per-core allocation in which every core of the shard grid, on every device
// of a mesh, takes the same address. Each core still reserves the address only in its own per-core
// allocator, so cores outside the grid can reuse it, while a kernel can address every core with
// one value (a multicast inside the grid, a compile-time semaphore address).
//
// Requires set_per_core_allocation(args, true) first. Like per-core allocation, it is only safe
// when nothing reaches the buffer on a core outside the grid.
BufferShardingArgs& set_uniform_address(BufferShardingArgs& args, bool enable);
bool is_uniform_address(const BufferShardingArgs& args);
bool is_uniform_address(const Buffer& buffer);

}  // namespace tt::tt_metal::experimental::per_core_allocation
