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
#include <atomic>
#include <cstdlib>
#include <cstdio>

namespace tt::tt_metal::tt_sim {

namespace {

void trace_direct_write_decision(
    const char* reason, const DirectWriteGuard& guard, const Buffer* shard_view, const BufferRegion& region) {
    if (std::getenv("TT_METAL_SIM_DIRECT_WRITE_TRACE") == nullptr) {
        return;
    }
    static std::atomic<uint32_t> trace_count = 0;
    uint32_t idx = trace_count.fetch_add(1, std::memory_order_relaxed);
    if (idx >= 512) {
        return;
    }
    int buffer_type = -1;
    int buffer_layout = -1;
    uint32_t device_id = 0xffffffffu;
    if (shard_view != nullptr) {
        buffer_type = static_cast<int>(shard_view->buffer_type());
        buffer_layout = static_cast<int>(shard_view->buffer_layout());
        device_id = shard_view->device()->id();
    }
    std::fprintf(
        stderr,
        "SIM_DIRECT_WRITE_TRACE idx=%u reason=%s target=%d cq_idle=%u size=%llu offset=%llu buffer_type=%d "
        "layout=%d device=%u\n",
        idx,
        reason,
        static_cast<int>(guard.target),
        guard.cq_idle ? 1u : 0u,
        static_cast<unsigned long long>(region.size),
        static_cast<unsigned long long>(region.offset),
        buffer_type,
        buffer_layout,
        device_id);
    std::fflush(stderr);
}

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

const char* direct_write_candidate_reject_reason(
    const DirectWriteGuard& guard, Buffer& shard_view, const void* src, const BufferRegion& region) {
    if (!has_direct_write_runtime_requirements(guard, src, region)) {
        return "runtime_requirements";
    }
    if (shard_view.buffer_type() != BufferType::DRAM) {
        return "unsupported_buffer";
    }
    return nullptr;
}

}  // namespace

bool is_direct_write_enabled(const DirectWriteGuard& guard, const void* src, const BufferRegion& region) {
    if (!has_direct_write_runtime_requirements(guard, src, region)) {
        return false;
    }
    // Most direct writes must not bypass work that has already been queued through FD.
    return guard.cq_idle;
}

bool is_direct_write_candidate(
    const DirectWriteGuard& guard, Buffer& shard_view, const void* src, const BufferRegion& region) {
    return direct_write_candidate_reject_reason(guard, shard_view, src, region) == nullptr;
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
        trace_direct_write_decision("runtime_requirements", guard, &shard_view, region);
        return false;
    }
    if (const char* reject_reason = direct_write_candidate_reject_reason(guard, shard_view, src, region)) {
        trace_direct_write_decision(reject_reason, guard, &shard_view, region);
        return false;
    }
    // Sharded L1 full-buffer uploads are TTNN input-staging writes in the
    // DeepSeek decode path. Allow them to bypass unrelated pending FD work in
    // simulator debug mode; keep DRAM ordered because queued kernels can observe
    // tensor and metadata uploads.
    if (!guard.cq_idle) {
        trace_direct_write_decision("pending_ordered_work", guard, &shard_view, region);
        return false;
    }
    write_shard(shard_view, src, region, logical_core_filter);
    trace_direct_write_decision("direct", guard, &shard_view, region);
    return true;
}

}  // namespace tt::tt_metal::tt_sim

#endif  // defined(TT_UMD_BUILD_SIMULATION)
