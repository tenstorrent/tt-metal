// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <initializer_list>
#include <variant>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {
class Buffer;
class Program;
namespace distributed {
class MeshBuffer;
}  // namespace distributed
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {

enum class BufferRole : uint8_t { Read, Write };

// Pack the runtime arguments for a kernel that wants per-tile DRAM access to
// `buffer` without manually computing the DRAM page stride.
//
// Behavior: builds the runtime-arg vector
//     {addr, bank_id, num_tiles, per_tile_dram_stride, ...extra_args}
// where per_tile_dram_stride = buffer.size() / num_tiles. That formula works
// for both common buffer configurations:
//   * page_size == tile_size      -> equals buffer.aligned_page_size()
//   * page_size == whole_buffer   -> equals total_bytes / num_tiles
// so callers no longer have to know which idiom their buffer is in or where
// DRAM page-alignment padding kicks in.
//
// The kernel reads these via the BoundBuffer<DRAM> wrapper in
// "experimental/bound_buffer.h", e.g. `BoundBuffer<DRAM> src(/*slot=*/0)`.
// Extra args are read at runtime-arg index 4+ via the standard get_arg_val<>.
//
// Note: SetRuntimeArgs may only be called once per kernel/core, so this
// function calls it exactly once. Do not call SetRuntimeArgs separately for
// the same kernel/core after BindBufferToKernel; pass any additional args via
// `extra_args`.
void BindBufferToKernel(
    Program& program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const Buffer& buffer,
    uint32_t num_tiles,
    BufferRole role,
    tt::stl::Span<const uint32_t> extra_args = {});

void BindBufferToKernel(
    Program& program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const Buffer& buffer,
    uint32_t num_tiles,
    BufferRole role,
    std::initializer_list<uint32_t> extra_args);

// Overloads for distributed::MeshBuffer. Uses MeshBuffer::address() and
// device_local_size() — equivalent semantics to the Buffer overload above.
void BindBufferToKernel(
    Program& program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const distributed::MeshBuffer& buffer,
    uint32_t num_tiles,
    BufferRole role,
    tt::stl::Span<const uint32_t> extra_args = {});

void BindBufferToKernel(
    Program& program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const distributed::MeshBuffer& buffer,
    uint32_t num_tiles,
    BufferRole role,
    std::initializer_list<uint32_t> extra_args);

}  // namespace tt::tt_metal::experimental
