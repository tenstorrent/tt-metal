// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace tt::tt_metal {

// The kinds of host <-> device transfer HostTransferStats reports, one summary row each.
enum class HostTransfer : uint8_t {
    BufferWriteFastDispatch,
    BufferWriteSlowDispatch,
    BufferReadFastDispatch,
    BufferReadSlowDispatch,
    CoreWriteFastDispatch,
    CoreReadFastDispatch,
    ProgramBinariesFastDispatch,
    ProgramBinariesSlowDispatch,
    // Program commands copied into the issue queue, which the device reads later. Staged like SocketPull.
    ProgramConfigFastDispatch,
    ProgramConfigSlowDispatch,
    SocketPush,
    // Bytes copied into host memory that the device reads later, so they have bytes but no transfer time.
    SocketPull,
};

// Bytes and busy time of host <-> device transfers, logged when the MetalContext tears down if
// TT_METAL_HOST_TRANSFER_STATS is set. Each transfer is an interval opened with begin() and closed with end(). Busy
// time is the wall time with at least one transfer of a kind open, so transfers that overlap across threads, devices
// or command queues count once, and the rate is a total across all of them.
class HostTransferStats {
public:
    using Clock = std::chrono::steady_clock;

    struct Totals {
        uint64_t bytes = 0;
        uint64_t transfers = 0;
        Clock::duration busy{};
    };

    explicit HostTransferStats(bool enabled = false) : enabled_(enabled) {}

    bool enabled() const { return enabled_.load(std::memory_order_relaxed); }
    void set_enabled(bool enabled) { enabled_.store(enabled, std::memory_order_relaxed); }

    void begin(HostTransfer kind);
    // Closes an interval opened with begin(). Callers take `start` and `end` on different threads, so intervals can
    // close out of time order. A transfer that moved no bytes closes its interval without adding time.
    void end(HostTransfer kind, Clock::time_point start, Clock::time_point end, uint64_t bytes);
    void add_bytes(HostTransfer kind, uint64_t bytes);

    // Includes the part of a busy stretch that is still open.
    Totals totals(HostTransfer kind) const;
    std::vector<std::pair<std::string, std::string>> summary_rows() const;
    void log_summary() const;
    void reset();

private:
    struct Stream {
        mutable std::mutex mutex;
        uint64_t bytes = 0;
        uint64_t transfers = 0;
        Clock::duration busy{};
        uint32_t in_flight = 0;
        Clock::time_point stretch_start = Clock::time_point::max();
        Clock::time_point stretch_end = Clock::time_point::min();
    };

    std::atomic<bool> enabled_;
    std::array<Stream, static_cast<size_t>(HostTransfer::SocketPull) + 1> streams_;
};

// Times the scope it lives in as one transfer of `kind` while `stats` is on: for transfers that are done when the host
// call returns. `stats` may be null, which records nothing.
class ScopedHostTransfer {
public:
    ScopedHostTransfer(HostTransferStats* stats, HostTransfer kind, uint64_t bytes);
    ~ScopedHostTransfer();

    ScopedHostTransfer(const ScopedHostTransfer&) = delete;
    ScopedHostTransfer& operator=(const ScopedHostTransfer&) = delete;
    ScopedHostTransfer(ScopedHostTransfer&&) = delete;
    ScopedHostTransfer& operator=(ScopedHostTransfer&&) = delete;

private:
    HostTransferStats* stats_ = nullptr;
    HostTransfer kind_;
    uint64_t bytes_;
    HostTransferStats::Clock::time_point start_;
};

}  // namespace tt::tt_metal
