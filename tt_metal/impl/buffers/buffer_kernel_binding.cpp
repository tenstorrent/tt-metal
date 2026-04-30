// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/buffer_kernel_binding.hpp>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt_stl/assert.hpp>

#include <vector>

namespace tt::tt_metal::experimental {

namespace {

constexpr uint32_t kBindingArgCount = 4;  // {addr, bank_id, num_tiles, stride}

void bind_raw(
    Program& program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    uint64_t buffer_address,
    uint64_t buffer_size,
    uint32_t num_tiles,
    BufferRole role,
    tt::stl::Span<const uint32_t> extra_args) {
    TT_FATAL(num_tiles > 0, "BindBufferToKernel: num_tiles must be > 0");
    TT_FATAL(
        buffer_size % num_tiles == 0,
        "BindBufferToKernel: buffer size {} not evenly divisible by num_tiles {}",
        buffer_size,
        num_tiles);

    // Universal per-tile DRAM stride: works whether the buffer was configured
    // with page_size == tile_size (allocator may pad to L1/DRAM alignment, in
    // which case size == num_pages * aligned_page_size) or with
    // page_size == whole_buffer (in which case size == total bytes the user
    // intends to interpret as num_tiles tiles).
    const uint32_t per_tile_stride = static_cast<uint32_t>(buffer_size / num_tiles);

    std::vector<uint32_t> args;
    args.reserve(kBindingArgCount + extra_args.size());
    args.push_back(static_cast<uint32_t>(buffer_address));
    args.push_back(0);  // bank_id; PoC assumes single-bank addressing as today's kernels do
    args.push_back(num_tiles);
    args.push_back(per_tile_stride);
    args.insert(args.end(), extra_args.begin(), extra_args.end());

    SetRuntimeArgs(program, kernel, core_spec, args);
    (void)role;  // reserved for future use (e.g. read-only vs write-only validation)
}

}  // namespace

void BindBufferToKernel(
    Program& program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const Buffer& buffer,
    uint32_t num_tiles,
    BufferRole role,
    tt::stl::Span<const uint32_t> extra_args) {
    bind_raw(
        program,
        kernel,
        core_spec,
        static_cast<uint64_t>(buffer.address()),
        static_cast<uint64_t>(buffer.size()),
        num_tiles,
        role,
        extra_args);
}

void BindBufferToKernel(
    Program& program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const Buffer& buffer,
    uint32_t num_tiles,
    BufferRole role,
    std::initializer_list<uint32_t> extra_args) {
    bind_raw(
        program,
        kernel,
        core_spec,
        static_cast<uint64_t>(buffer.address()),
        static_cast<uint64_t>(buffer.size()),
        num_tiles,
        role,
        tt::stl::Span<const uint32_t>(extra_args.begin(), extra_args.size()));
}

void BindBufferToKernel(
    Program& program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const distributed::MeshBuffer& buffer,
    uint32_t num_tiles,
    BufferRole role,
    tt::stl::Span<const uint32_t> extra_args) {
    bind_raw(
        program,
        kernel,
        core_spec,
        static_cast<uint64_t>(buffer.address()),
        static_cast<uint64_t>(buffer.device_local_size()),
        num_tiles,
        role,
        extra_args);
}

void BindBufferToKernel(
    Program& program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const distributed::MeshBuffer& buffer,
    uint32_t num_tiles,
    BufferRole role,
    std::initializer_list<uint32_t> extra_args) {
    bind_raw(
        program,
        kernel,
        core_spec,
        static_cast<uint64_t>(buffer.address()),
        static_cast<uint64_t>(buffer.device_local_size()),
        num_tiles,
        role,
        tt::stl::Span<const uint32_t>(extra_args.begin(), extra_args.size()));
}

}  // namespace tt::tt_metal::experimental
