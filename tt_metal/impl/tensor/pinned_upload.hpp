// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace tt::tt_metal {
class DistributedHostBuffer;
namespace distributed {
class MeshBuffer;
class MeshCommandQueue;
class MeshDevice;
}  // namespace distributed
}  // namespace tt::tt_metal

namespace tt::tt_metal::pinned_upload {

// Tensor uploads at or below this size copy through the command queue; pinning costs more than it saves.
inline constexpr size_t k_pin_write_threshold_bytes = 32 * 1024 * 1024;

// Upper bound on one chunk pin of a chunked upload. Pins of this size keep a chunk's pin time well under the transfer
// time of the chunk before it, so pinning hides behind the transfer, while the pipeline still ramps up within the
// first few MiB.
inline constexpr size_t k_max_chunk_bytes = 8 * 1024 * 1024;

// Chunk size a chunked upload uses for one shard of `shard_bytes` written in pages of `page_bytes`: the shard splits
// into the fewest near-equal chunks of at most k_max_chunk_bytes, each a multiple of lcm(page_bytes, OS page size),
// with a shorter last chunk. Chunks are page multiples because the device writes whole pages of a buffer region, and
// OS-page multiples so chunks that start page-aligned pin disjoint pages. Returns 0 when that granule exceeds
// k_max_chunk_bytes.
size_t chunk_bytes_for(size_t shard_bytes, size_t page_bytes);

// True when an upload of `size_bytes` to `mesh_device` should try to have the device read pinned host memory.
bool should_use_pinned_write_path(distributed::MeshDevice& mesh_device, size_t size_bytes);

// Writes this host's shards of `host_buffer` into `mesh_buffer` with the device reading pinned host memory.
//
// Returns false, having written nothing, when no shard could be pinned; the caller then writes through the copy
// path. On true, every shard has been written. The call returns once the device has read all of the host memory,
// except when every shard's storage is device-immutable (experimental::MemoryPinMarkDeviceImmutable): then it returns
// as soon as the writes are enqueued, and keeps the storage alive until they complete.
//
// Large uploads on Blackhole with IOMMU are pinned in chunks by a worker pool (TT_METAL_PINNED_UPLOAD_THREADS) while
// earlier chunks transfer; others pin each shard whole through experimental::PinnedMemoryCache.
bool write_shards(
    distributed::MeshCommandQueue& cq,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const DistributedHostBuffer& host_buffer);

// Waits for the writes of device-immutable uploads that returned early and releases their pins. Their storage (for a
// file mapping, the munmap) is released on a background thread; see wait_for_storage_release.
// `cq_id` limits this to uploads through that command queue; nullopt covers every command queue of `mesh_device`.
void drain(const distributed::MeshDevice& mesh_device, std::optional<uint32_t> cq_id = std::nullopt);

// Number of device-immutable uploads to `mesh_device` whose pins are still held after returning early.
size_t num_pending(const distributed::MeshDevice& mesh_device);

// Waits until the storage of every drained upload has been released.
void wait_for_storage_release();

}  // namespace tt::tt_metal::pinned_upload
