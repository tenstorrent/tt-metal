// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/regions/tensix_l1.hpp -- per-tile sharded L1 MeshBuffer allocation,
// zeroing, and address validation for Tensix characterization participants.

#ifndef X280_HARNESS_REGIONS_TENSIX_L1_HPP
#define X280_HARNESS_REGIONS_TENSIX_L1_HPP

#include <tt-metalium/core_coord.hpp>

#include <mesh_buffer.hpp>
#include <mesh_device.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace x280::harness::regions {

/**
 * One height-sharded L1 MeshBuffer covering a set of Tensix tiles, one page per
 * tile at a common tile-local base address.
 *
 * The caller owns `buffer`: the allocation lives until the last shared_ptr to it
 * is released, and `l1_tile_base` is only valid for that lifetime -- the
 * allocator may hand the same address to a later buffer.
 */
struct TileL1Allocation {
    /** The allocation itself; keep it alive for as long as device code uses the region. */
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer;

    /** Device-local L1 byte address of this tile's page; identical on every tile in `worker_set`. */
    uint32_t l1_tile_base = 0;

    /** The tiles the buffer was allocated over, for CreateKernel / SetRuntimeArgs. */
    tt::tt_metal::CoreRangeSet worker_set;
};

/**
 * Allocate and zero one height-sharded L1 MeshBuffer over `cores`, giving every
 * tile a `page_bytes` page at the same device-local base address.
 *
 * The buffer holds one page per core, in the order `cores.ranges()` stores them,
 * so a host transfer of the whole buffer yields the pages in that same order. A
 * caller that depends on that order must build `cores` from CoreRanges, which
 * preserves it: the CoreCoord constructor merges and coalesces ranges, which
 * silently permutes the pages relative to the coordinate list it was given.
 *
 * @param cores      Participating tiles, in logical Tensix coordinates; must be non-empty.
 * @param page_bytes Per-tile extent in bytes; must be non-zero and a multiple of 4.
 *
 * Validates that the mesh and extent are usable before allocating, then that the
 * resulting base fits in 32 bits, is 64-byte aligned, and that a `page_bytes`
 * page fits within `l1_size_per_core`, and that the allocation really is one
 * page per tile across `cores`. Zeroes the whole buffer with one blocking write
 * before returning, so device code sees cleared L1 on every tile.
 *
 * @throws std::runtime_error on any of the above, naming the offending value.
 *         Failure is never reported through the return value.
 */
inline TileL1Allocation allocate_tile_l1_buffer(
    tt::tt_metal::distributed::MeshDevice* mesh, const tt::tt_metal::CoreRangeSet& cores, uint32_t page_bytes) {
    if (!mesh) {
        throw std::runtime_error("allocate_tile_l1_buffer: mesh is null");
    }
    if (page_bytes == 0) {
        throw std::runtime_error("allocate_tile_l1_buffer: page_bytes is zero");
    }
    if (page_bytes % sizeof(uint32_t) != 0) {
        throw std::runtime_error(
            "allocate_tile_l1_buffer: page_bytes " + std::to_string(page_bytes) + " is not a multiple of 4");
    }
    const uint32_t tile_count = cores.num_cores();
    if (tile_count == 0) {
        throw std::runtime_error("allocate_tile_l1_buffer: core set is empty");
    }

    const uint64_t total_bytes = (uint64_t)page_bytes * (uint64_t)tile_count;
    const tt::tt_metal::ShardSpecBuffer shard(
        cores, {1, 1}, tt::tt_metal::ShardOrientation::ROW_MAJOR, {1, 1}, {tile_count, 1});

    auto buffer = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{.size = total_bytes},
        tt::tt_metal::distributed::DeviceLocalBufferConfig{
            .page_size = page_bytes,
            .buffer_type = tt::tt_metal::BufferType::L1,
            .sharding_args = tt::tt_metal::BufferShardingArgs(shard, tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED),
            .bottom_up = false},
        mesh);

    uint64_t addr = buffer->address();
    if (addr > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("allocate_tile_l1_buffer: L1 address exceeds 32 bits");
    }
    if ((addr & 63u) != 0) {
        throw std::runtime_error("allocate_tile_l1_buffer: L1 base is not 64-byte aligned");
    }
    uint64_t l1_size = mesh->get_device(0, 0)->l1_size_per_core();
    if (addr + (uint64_t)page_bytes > l1_size) {
        throw std::runtime_error("allocate_tile_l1_buffer: buffer extent exceeds core L1 size");
    }
    if (buffer->num_pages() != tile_count) {
        throw std::runtime_error(
            "allocate_tile_l1_buffer: allocation holds " + std::to_string(buffer->num_pages()) + " pages for " +
            std::to_string(tile_count) + " tiles");
    }

    std::vector<uint32_t> zeros(total_bytes / sizeof(uint32_t), 0);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(mesh->mesh_command_queue(), buffer, zeros, /*blocking=*/true);

    return TileL1Allocation{std::move(buffer), static_cast<uint32_t>(addr), cores};
}

}  // namespace x280::harness::regions

#endif  // X280_HARNESS_REGIONS_TENSIX_L1_HPP
