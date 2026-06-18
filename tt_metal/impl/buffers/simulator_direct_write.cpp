// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "buffers/simulator_direct_write.hpp"

#if defined(TT_UMD_BUILD_SIMULATION)

#include "llrt/rtoptions.hpp"
#include "impl/allocator/allocator.hpp"
#include <tt-metalium/experimental/core_subset_write/buffer_write.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal::tt_sim {

namespace {

bool has_direct_write_runtime_requirements(const DirectWriteGuard& guard, const void* src, const BufferRegion& region) {
    if (guard.target != tt::TargetDevice::Simulator) {
        return false;
    }
    if (guard.rtoptions == nullptr || !guard.rtoptions->get_simulator_direct_tensor_writes()) {
        return false;
    }
    if (src == nullptr) {
        return false;
    }
    if (region.offset != 0) {
        return false;
    }
    return true;
}

}  // namespace

bool is_direct_write_enabled(const DirectWriteGuard& guard, const void* src, const BufferRegion& region) {
    if (!has_direct_write_runtime_requirements(guard, src, region)) {
        return false;
    }
    // Most direct writes must not bypass work that has already been queued through FD.
    return guard.cq_idle;
}

void write_shard(
    Buffer& shard_view, const void* src, const BufferRegion& region, const CoreRangeSet* logical_core_filter) {
    auto payload = tt::stl::Span<const uint8_t>(static_cast<const uint8_t*>(src), static_cast<size_t>(region.size));
    if (logical_core_filter != nullptr) {
        experimental::core_subset_write::WriteToBuffer(shard_view, payload, *logical_core_filter);
    } else {
        detail::WriteToBuffer(shard_view, payload);
    }
}

bool try_direct_write(
    const DirectWriteGuard& guard,
    Buffer& shard_view,
    const void* src,
    const BufferRegion& region,
    const CoreRangeSet* logical_core_filter) {
    if (!has_direct_write_runtime_requirements(guard, src, region)) {
        return false;
    }
    auto buffer_type = shard_view.buffer_type();
    bool is_l1_sharded_upload = buffer_type == BufferType::L1 && is_sharded(shard_view.buffer_layout());
    bool can_direct_write_buffer = is_l1_sharded_upload;
    // Keep interleaved L1 on the normal CQ path. Those tensors can be transient
    // decode intermediates whose ordering is observable by host fallback paths.
    // Keep DRAM on the normal CQ path too: on WH simulator, direct DRAM writes
    // can mis-handle DRAM view/bank placement for replicated tiled tensors.
    if (!can_direct_write_buffer) {
        return false;
    }
    // Sharded L1 full-buffer uploads are TTNN input-staging writes in the
    // DeepSeek decode path. Allow them to bypass unrelated pending FD work in
    // simulator debug mode; keep DRAM and interleaved L1 ordered.
    if (!guard.cq_idle && !is_l1_sharded_upload) {
        return false;
    }
    // Trace-enabled tests depend on normal CQ ordering/capture semantics even for tensor writes.
    if (shard_view.device()->allocator_impl()->get_config().trace_region_size != 0) {
        return false;
    }
    write_shard(shard_view, src, region, logical_core_filter);
    return true;
}

}  // namespace tt::tt_metal::tt_sim

#endif  // defined(TT_UMD_BUILD_SIMULATION)
