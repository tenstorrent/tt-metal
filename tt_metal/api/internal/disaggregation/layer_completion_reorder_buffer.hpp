// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// LayerCompletionReorderBuffer — single-threaded sequencer. Completions
// arrive out of order (different hosts, MPI timing); the master must
// emit them to the scheduler strictly in ascending `seq`. insert()
// buffers a message and drains every now-contiguous message starting at
// next_expected. Pure logic — no SHM, no threads, no MPI — so it is unit
// tested in isolation.
//
// LIMITATION (future work): next_expected_ only advances over a contiguous
// run, so a single permanently-lost `seq` (e.g. a teardown drop or a rank
// SIGKILL) head-of-line-blocks every later completion forever — they pile up
// in pending_ and the scheduler stalls. Proposed fix: a per-sequence watchdog
// — arm a ~5s timer when the first completion of a sequence arrives and clear
// it when its last layer arrives; on expiry, skip the gap (advance
// next_expected_ past the missing seq, logging the skip). Not implemented yet.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include <internal/disaggregation/layer_completion_message.hpp>

namespace tt::tt_metal::internal {

class LayerCompletionReorderBuffer {
public:
    explicit LayerCompletionReorderBuffer(uint64_t start_seq = 0) : next_expected_(start_seq) {}

    // Buffer `msg`, then move every contiguous message from next_expected_
    // into `drained` (cleared first), advancing the cursor. Returns the
    // count drained. Stale (seq < next_expected_) and duplicate
    // (already-buffered seq) messages are dropped → return 0.
    uint32_t insert(const LayerCompletionMessage& msg, std::vector<LayerCompletionMessage>& drained);

    uint64_t next_expected() const noexcept { return next_expected_; }
    std::size_t buffered() const noexcept { return pending_.size(); }

private:
    uint64_t next_expected_;
    std::map<uint64_t, LayerCompletionMessage> pending_;
};

}  // namespace tt::tt_metal::internal
