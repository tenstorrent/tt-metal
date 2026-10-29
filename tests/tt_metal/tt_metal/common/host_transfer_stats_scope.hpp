// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/host_transfer_stats.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal::test {

// Clears the context's host transfer stats and turns them on for the scope, restoring the previous setting after.
class HostTransferStatsScope {
public:
    explicit HostTransferStatsScope(ContextId context_id = DEFAULT_CONTEXT_ID) :
        stats_(MetalContext::instance(context_id).host_transfer_stats()), was_enabled_(stats_.enabled()) {
        stats_.reset();
        stats_.set_enabled(true);
    }
    ~HostTransferStatsScope() { stats_.set_enabled(was_enabled_); }

    HostTransferStatsScope(const HostTransferStatsScope&) = delete;
    HostTransferStatsScope& operator=(const HostTransferStatsScope&) = delete;

    HostTransferStats::Totals totals(HostTransfer kind) const { return stats_.totals(kind); }

private:
    HostTransferStats& stats_;
    bool was_enabled_;
};

// Whether mesh command queue buffer transfers are recorded: fast dispatch times them with events, which mock and
// emulated devices lack, and slow dispatch skips transfers to mock devices.
inline bool buffer_transfers_recorded(ContextId context_id = DEFAULT_CONTEXT_ID) {
    auto& context = MetalContext::instance(context_id);
    const auto target = context.get_cluster().get_target_device_type();
    if (target == tt::TargetDevice::Mock) {
        return false;
    }
    return !(context.rtoptions().get_fast_dispatch() && target == tt::TargetDevice::Emule);
}

inline HostTransfer buffer_write_transfer(ContextId context_id = DEFAULT_CONTEXT_ID) {
    return MetalContext::instance(context_id).rtoptions().get_fast_dispatch() ? HostTransfer::BufferWriteFastDispatch
                                                                              : HostTransfer::BufferWriteSlowDispatch;
}

inline HostTransfer buffer_read_transfer(ContextId context_id = DEFAULT_CONTEXT_ID) {
    return MetalContext::instance(context_id).rtoptions().get_fast_dispatch() ? HostTransfer::BufferReadFastDispatch
                                                                              : HostTransfer::BufferReadSlowDispatch;
}

}  // namespace tt::tt_metal::test
