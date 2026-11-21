// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Trace store: write a captured mesh trace to a file and install it again in a later process
// without capturing it. A trace is the device command stream of a fixed sequence of programs with
// fixed buffer addresses, so a stored trace is valid only when the restoring process has the same
// programs (same build and kernels, loaded by a prep run) and the same buffers at the same
// addresses. The caller keys the store on everything that decides those and checks the addresses
// itself (see models/tt_dit/utils/tracing.py); this API only moves the bytes.

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/mesh_trace_id.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace tt::tt_metal::experimental::trace_store {

// Write the host descriptor of a captured trace to `path`. Throws if the trace cannot be stored:
// it holds capture frees, or captured tensor prefetcher requests.
void dump_mesh_trace(distributed::MeshDevice* mesh_device, const distributed::MeshTraceId& trace_id, const std::string& path);

// Read a trace written by dump_mesh_trace and install it on `cq_id`: create the trace, write it to
// the trace region of every device and register it, as end_mesh_trace does after a capture.
// Only for a fixed trace region (trace_region_size > 0). Throws on a malformed or truncated file.
distributed::MeshTraceId load_mesh_trace(distributed::MeshDevice* mesh_device, uint8_t cq_id, const std::string& path);

// Address and size of the trace's buffer in the trace region.
std::pair<uint64_t, uint64_t> mesh_trace_buffer_range(distributed::MeshDevice* mesh_device, const distributed::MeshTraceId& trace_id);

// Every live buffer of the mesh allocator: (buffer type, address, size, page size).
std::vector<std::tuple<int, uint64_t, uint64_t, uint64_t>> allocated_buffers(distributed::MeshDevice* mesh_device);

// Place the next DRAM allocation of exactly `size` bytes at `address`. The pin is one-shot and is
// ignored by allocations of any other size; set_next_allocation_address(0, 0) clears it. An
// allocation that matches but cannot be placed there throws.
void set_next_allocation_address(uint64_t address, uint64_t size);

}  // namespace tt::tt_metal::experimental::trace_store
