// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/dry_run.hpp>

#include <atomic>

#include "jit_build/build.hpp"

namespace tt::tt_metal::experimental::dry_run {

namespace {
// Process-wide, so work started from any thread sees it.
std::atomic<int> depth{0};
}  // namespace

void begin() { depth.fetch_add(1, std::memory_order_acq_rel); }

void end() {
    int current = depth.load(std::memory_order_acquire);
    while (current > 0 && !depth.compare_exchange_weak(current, current - 1, std::memory_order_acq_rel)) {
    }
}

bool active() { return depth.load(std::memory_order_acquire) > 0; }

void wait_for_queued_builds() { wait_for_ahead_builds(); }

}  // namespace tt::tt_metal::experimental::dry_run
