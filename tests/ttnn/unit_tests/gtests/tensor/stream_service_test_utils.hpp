// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

#include <memory>

#include <tt_stl/small_vector.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <umd/device/types/arch.hpp>
#include <ttnn/api/ttnn/distributed/distributed_configs.hpp>

#include "impl/context/metal_context.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::distributed::test {

// Fully-replicated placements sized to a mesh's dimensionality (identity on a
// 1x1 mesh; full tensor on every device otherwise). Shared by the H2D and D2D
// stream-service gtests, which both distribute their inputs with this mapping.
inline ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> replicate_all(
    const tt::tt_metal::distributed::MeshDevice& mesh) {
    return ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>(
        mesh.shape().dims(), tt::tt_metal::distributed::MeshMapperConfig::Replicate{});
}

// Mirror the ServiceCoreManager precondition (Blackhole / UBB Galaxy under Fast
// Dispatch) so tests skip cleanly instead of fataling inside create_pair.
inline bool service_cores_supported() {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    return cluster.is_ubb_galaxy() || cluster.arch() == tt::ARCH::BLACKHOLE;
}

// The H2D front-end pins a host DMA buffer for its FIFO. On a host booted with the
// IOMMU in passthrough/identity mode (e.g. iommu=pt), UMD must pin physically-
// contiguous pages and rejects the multi-page, non-contiguous H2D buffer with
// EINVAL. This is a host-config / H2D-infra limitation, NOT a D2D bug. Tests that
// build an H2DStreamService skip cleanly when the IOMMU isn't in DMA-translation
// mode. See notes/d2d_galaxy_h2d_pinning_failure.md.
inline bool h2d_host_pinning_supported() {
    return tt::tt_metal::MetalContext::instance().get_cluster().is_iommu_enabled();
}

// Number of cores in a (rectangular) CoreRange.
inline uint32_t core_range_volume(const tt::tt_metal::CoreRange& cr) {
    return (cr.end_coord.x - cr.start_coord.x + 1) * (cr.end_coord.y - cr.start_coord.y + 1);
}

// The full compute grid of a (1x1) stage device, as a CoreRange.
inline tt::tt_metal::CoreRange all_cores_for(const tt::tt_metal::distributed::MeshDevice& mesh) {
    const auto grid = mesh.compute_with_storage_grid_size();
    return tt::tt_metal::CoreRange{tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{grid.x - 1, grid.y - 1}};
}

// Row-major worker index within a CoreRange (robust to iterator order).
inline uint32_t worker_index(const tt::tt_metal::CoreCoord& wc, const tt::tt_metal::CoreRange& worker_cores) {
    const uint32_t width = worker_cores.end_coord.x - worker_cores.start_coord.x + 1;
    return (wc.y - worker_cores.start_coord.y) * width + (wc.x - worker_cores.start_coord.x);
}

// Per-worker [start_page, end_page) over num_pages, distributing the remainder to
// the first `rem` workers. Empty ranges (num_pages < num_workers) are valid: that
// worker copies nothing but still handshakes, so the service's num_workers ack
// count is always satisfied.
inline std::pair<uint32_t, uint32_t> worker_page_range(uint32_t worker_idx, uint32_t num_workers, uint32_t num_pages) {
    const uint32_t base = num_pages / num_workers;
    const uint32_t rem = num_pages % num_workers;
    const uint32_t start = worker_idx * base + std::min(worker_idx, rem);
    const uint32_t end = start + base + (worker_idx < rem ? 1u : 0u);
    return {start, end};
}

// Socket / scratch FIFO size for a given spec. derive_chunk_plan FATALs unless the
// FIFO holds at least one tensor page, so wide last dims (page > 4096 B) need a
// bigger FIFO than the historical 4096. Round the page up to a 4096 multiple:
// narrow pages keep a 4096 FIFO (so it still holds several pages, exercising chunk
// packing), while a wide page gets a FIFO sized to exactly one page. 4096 is a
// multiple of the L1 alignment, so this is always >= the buffer's aligned_page_size.
inline uint32_t fifo_bytes_for(const tt::tt_metal::TensorSpec& spec) {
    constexpr uint32_t kMinFifo = 4096u;
    const uint32_t page = static_cast<uint32_t>(spec.compute_page_size_bytes());
    return std::max(kMinFifo, ((page + kMinFifo - 1u) / kMinFifo) * kMinFifo);
}

// DRAM-interleaved spec with an explicit dtype + layout (block-float formats
// require TILE). Defaults to UINT32 ROW_MAJOR.
inline tt::tt_metal::TensorSpec make_spec(
    const ttnn::Shape& global_shape,
    tt::tt_metal::DataType dtype = tt::tt_metal::DataType::UINT32,
    tt::tt_metal::Layout layout = tt::tt_metal::Layout::ROW_MAJOR) {
    return tt::tt_metal::TensorSpec(
        global_shape,
        tt::tt_metal::TensorLayout(
            dtype,
            tt::tt_metal::PageConfig(layout),
            tt::tt_metal::MemoryConfig{
                tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM, std::nullopt}));
}

// Read/WriteShard need a shared_ptr<MeshBuffer>, but a device Tensor only exposes
// its buffer as a `const MeshBuffer&` (DeviceStorage retains sole ownership since
// the removal of get_mesh_buffer_leak_ownership in #47291). Build a non-owning
// MeshBuffer "view" over the tensor's existing device allocation: same address, no
// new allocation, and no const_cast. The view must not outlive the tensor whose
// memory it points at.
inline std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> mesh_buffer_view(const ttnn::Tensor& tensor) {
    const auto& backing = tensor.device_storage().get_mesh_buffer();
    return tt::tt_metal::distributed::MeshBuffer::create(
        backing.global_config(), backing.device_local_config(), backing.device(), backing.address());
}

}  // namespace ttnn::distributed::test
