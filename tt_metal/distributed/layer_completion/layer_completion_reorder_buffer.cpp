// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/layer_completion_reorder_buffer.hpp>

#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::internal {

uint32_t LayerCompletionReorderBuffer::insert(
    const LayerCompletionMessage& msg, std::vector<LayerCompletionMessage>& drained) {
    drained.clear();
    if (msg.seq < next_expected_) {
        return 0;  // stale — already emitted
    }
    const auto [it_ins, inserted] = pending_.emplace(msg.seq, msg);  // duplicate seq → keeps first
    if (!inserted && (it_ins->second.source_rank != msg.source_rank || it_ins->second.layer_idx != msg.layer_idx)) {
        // Two *distinct* completions claimed the same seq — the dense-seq invariant (each rank owns a
        // disjoint global-layer slice) is broken (e.g. a layer-split off-by-one, or a rank passing its
        // local slice length as the seq stride). The first is kept and this one dropped, so a real
        // completion is lost; surface it loudly instead of silently.
        log_error(
            LogMetal,
            "LayerCompletionReorderBuffer: seq {} collision — kept {{rank={}, layer={}}}, dropped "
            "{{rank={}, layer={}}}; dense-seq invariant violated, a completion is lost",
            msg.seq,
            it_ins->second.source_rank,
            it_ins->second.layer_idx,
            msg.source_rank,
            msg.layer_idx);
    }

    auto it = pending_.find(next_expected_);
    while (it != pending_.end()) {
        drained.push_back(it->second);
        pending_.erase(it);
        ++next_expected_;
        it = pending_.find(next_expected_);
    }
    return static_cast<uint32_t>(drained.size());
}

}  // namespace tt::tt_metal::internal
