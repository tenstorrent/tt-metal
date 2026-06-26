// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/layer_completion_reorder_buffer.hpp>

namespace tt::tt_metal::distributed {

uint32_t LayerCompletionReorderBuffer::insert(
    const LayerCompletionMessage& msg, std::vector<LayerCompletionMessage>& drained) {
    drained.clear();
    if (msg.seq < next_expected_) {
        return 0;  // stale — already emitted
    }
    pending_.emplace(msg.seq, msg);  // duplicate seq → no-op (map keeps first)

    auto it = pending_.find(next_expected_);
    while (it != pending_.end()) {
        drained.push_back(it->second);
        pending_.erase(it);
        ++next_expected_;
        it = pending_.find(next_expected_);
    }
    return static_cast<uint32_t>(drained.size());
}

}  // namespace tt::tt_metal::distributed
