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
std::atomic<bool> pass{false};
}  // namespace

void begin() { depth.fetch_add(1, std::memory_order_acq_rel); }

void end() {
    int current = depth.load(std::memory_order_acquire);
    while (current > 0 && !depth.compare_exchange_weak(current, current - 1, std::memory_order_acq_rel)) {
    }
}

bool active() { return depth.load(std::memory_order_acquire) > 0; }

Suspended::Suspended() : depth_(depth.exchange(0, std::memory_order_acq_rel)) {}
Suspended::~Suspended() { depth.fetch_add(depth_, std::memory_order_acq_rel); }

size_t finish_queued_builds(bool drop_queued) { return wait_for_ahead_builds(drop_queued); }

std::vector<std::string> take_queued_builds() { return take_queued_ahead_builds(); }

void set_pass(bool in_dry_pass) { pass.store(in_dry_pass, std::memory_order_release); }

bool in_pass() { return pass.load(std::memory_order_acquire); }

}  // namespace tt::tt_metal::experimental::dry_run
