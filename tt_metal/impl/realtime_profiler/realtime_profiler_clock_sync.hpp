// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/realtime_profiler.hpp>

#include "context/context_types.hpp"
#include "tt_metal/impl/realtime_profiler/realtime_profiler_clock_model.hpp"

namespace tt::umd {
class TlbWindow;
}

namespace tt::tt_metal {

class IDevice;

namespace experimental {
class PinnedMemory;
}

namespace distributed {
class MeshDevice;
}  // namespace distributed

// Process-global per-physical-chip cache of the fitted clock frequency, so a rapid MeshDevice reopen can skip the ~0.5s
// bring-up fit and take one anchor probe instead (device WALL_CLOCK is free-running across close). The offset is not
// cached: it is re-anchored every kClockSyncInterval, so a stored one would be stale before first use.
class RealtimeProfilerFrequencyCache {
public:
    std::optional<double> try_get(
        uint32_t chip_id, std::chrono::steady_clock::time_point now, std::chrono::steady_clock::duration max_age) const;
    void put(uint32_t chip_id, double frequency, std::chrono::steady_clock::time_point now);

private:
    struct Entry {
        double frequency = 0.0;
        std::chrono::steady_clock::time_point updated_at;
    };
    mutable std::mutex mu_;
    std::unordered_map<uint32_t, Entry> by_chip_;
};

RealtimeProfilerFrequencyCache& rt_profiler_frequency_cache();

// Everything RealtimeProfilerClockSync needs to reach one chip's handshake fields.
struct RealtimeProfilerClockSyncConfig {
    ContextId context_id{};
    IDevice* device = nullptr;  // the chip being synced; its chip id is taken from device->id()
    // Borrowed for the duration of configure() only, to pin the host ACK buffer and resolve its NOC address.
    distributed::MeshDevice* mesh_device = nullptr;
    CoreCoord profiler_core;  // the reserved tensix running the profiler kernels
    distributed::MeshCoordinate mesh_coord = distributed::MeshCoordinate(0);
    // No-IOMMU + 64-bit-PCIe host: the ACK word falls back to a CQ-sysmem slot whose device PCIe writes may be
    // non-snooped, so reads must evict the cache line first.
    bool hugepage_fallback = false;
    // Base of realtime_profiler_msg_t on profiler_core. The individual sync field addresses are resolved from it via
    // the HAL, so only this class needs to know which of those fields the handshake uses.
    uint32_t msg_base_addr = 0;
};

// The host end of one device's clock-sync handshake: owns the transport (token write, pinned host ACK word, round-trip
// probe) and drives a ClockModel from the samples it measures. Everything about what those samples mean -- the fit, the
// re-anchor policy, the error bar -- belongs to ClockModel; this class is how a sample gets taken.
//
// resync() and mapping() must run on a single thread (the receiver thread in steady state), which is what lets
// mapping() be lock-free; bring-up (configure/run_fit) runs before that thread starts.
class RealtimeProfilerClockSync {
public:
    RealtimeProfilerClockSync() = default;

    void configure(const RealtimeProfilerClockSyncConfig& config);

    // Adopts a cached frequency and takes one probe to anchor on it, skipping the multi-sample fit. False when there is
    // no usable cache entry, or the anchor probe failed and a full fit is needed.
    bool try_restore_calibration(std::chrono::steady_clock::time_point now);
    // Fits frequency and an initial anchor from a batch of handshakes. How many samples that takes is a property of
    // the fit, so the count lives with it rather than at the call site.
    bool run_fit();
    // Runs one handshake and re-anchors the mapping if the round trip was good enough to be worth it. False only when
    // the device did not answer; declining to re-anchor on a slow round trip is normal and returns true.
    bool resync(std::chrono::steady_clock::time_point now);

    // The mapping to publish on records drained at `now`.
    experimental::ProgramRealtimeClockSync mapping(std::chrono::steady_clock::time_point now) const {
        return model_.mapping(now);
    }
    double frequency() const { return model_.frequency(); }

private:
    // L1 addresses of the handshake fields within the profiler core's realtime_profiler_msg_t.
    struct SyncL1Addrs {
        uint32_t host_timestamp = 0;  // host->device handshake token
        uint32_t ack_host_addr = 0;   // base of the [lo, hi] host ACK buffer address pair
    };

    std::optional<ClockSyncSample> probe();
    void write_timestamp(uint32_t value);
    // Times the round trip to the timestamp landing, then waits for the token. Empty when the device stopped
    // responding.
    std::optional<std::chrono::nanoseconds> measure_rtt(
        std::chrono::steady_clock::time_point host_before, uint32_t host_time_id);
    uint32_t read_ack() const;
    uint64_t read_device_time() const;
    SyncL1Addrs resolve_l1_addrs(uint32_t msg_base_addr) const;
    void configure_write_path();
    void configure_ack_word(distributed::MeshDevice& mesh_device);

    ContextId context_id_{};
    IDevice* device_ = nullptr;
    uint32_t chip_id_ = 0;
    CoreCoord profiler_core_;
    distributed::MeshCoordinate mesh_coord_ = distributed::MeshCoordinate(0);
    // No-IOMMU + 64-bit-PCIe host: the ACK word is a CQ-sysmem slot whose device PCIe writes may be non-snooped, so
    // reads must evict the cache line first.
    bool hugepage_fallback_ = false;
    SyncL1Addrs l1_;
    uint32_t sync_seq_ = 0;          // host->device request flag, never 0 so the device can tell a request is pending
    uint64_t last_device_time_ = 0;  // last timestamp read out of the ACK buffer; a change is the acknowledgement
    // Sole owner of the pinned-path ACK buffer: PinnedMemory maps only a raw pointer, so dropping this frees it.
    std::shared_ptr<uint32_t[]> ack_host_backing_;
    std::shared_ptr<experimental::PinnedMemory> ack_pinned_;
    volatile uint32_t* ack_host_ptr_ = nullptr;
    // Blackhole fast path: static-L1 TLB window for a one-store MMIO token write; null elsewhere. Owned by UMD.
    tt::umd::TlbWindow* sync_tlb_ = nullptr;
    // The token's mapped address inside sync_tlb_, resolved once at configure time.
    volatile uint32_t* sync_doorbell_ = nullptr;

    // Seeded with the commanded AICLK in configure(), refined by run_fit(), re-anchored by resync().
    ClockModel model_;
};

}  // namespace tt::tt_metal
