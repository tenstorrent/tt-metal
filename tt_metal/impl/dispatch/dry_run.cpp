// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/dry_run.hpp>

#include <atomic>
#include <stdexcept>
#include <string>

#include <tt_stl/assert.hpp>

#include "jit_build/build.hpp"

namespace tt::tt_metal::experimental::dry_run {

namespace {
// Process-wide, so work started from any thread sees it.
std::atomic<int> depth{0};
std::atomic<bool> pass{false};
std::atomic<bool> host_only_devices{false};
std::atomic<size_t> blocked{0};
}  // namespace

void begin() { depth.fetch_add(1, std::memory_order_acq_rel); }

void end() {
    int current = depth.load(std::memory_order_acquire);
    while (current > 0 && !depth.compare_exchange_weak(current, current - 1, std::memory_order_acq_rel)) {
    }
}

bool active() { return depth.load(std::memory_order_acquire) > 0 || host_only(); }

Suspended::Suspended() : depth_(depth.exchange(0, std::memory_order_acq_rel)) {}
Suspended::~Suspended() { depth.fetch_add(depth_, std::memory_order_acq_rel); }

size_t finish_queued_builds(bool drop_queued) { return wait_for_ahead_builds(drop_queued); }

void hold_queued_builds(bool hold) { hold_ahead_builds(hold); }

std::vector<std::string> take_queued_builds() { return take_queued_ahead_builds(); }

void set_pass(bool in_dry_pass) {
    pass.store(in_dry_pass, std::memory_order_release);
    if (!in_dry_pass) {
        set_host_only(false);
    }
}

bool host_only() { return host_only_devices.load(std::memory_order_acquire); }
void set_host_only(bool on) { host_only_devices.store(on, std::memory_order_release); }

void chip_access(const char* what) {
    if (!host_only()) {
        return;
    }
    // The first few get a backtrace in the log, so a missing gate can be found; the rest only count.
    if (blocked.fetch_add(1, std::memory_order_acq_rel) < 40) {
        TT_THROW("Dry run with a host-only device: chip access blocked at Cluster::{}", what);
    }
    throw std::runtime_error(std::string("Dry run with a host-only device: chip access blocked at Cluster::") + what);
}

size_t blocked_chip_accesses() { return blocked.load(std::memory_order_acquire); }

bool in_pass() { return pass.load(std::memory_order_acquire); }

}  // namespace tt::tt_metal::experimental::dry_run
