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
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/realtime_profiler.hpp>

#include "context/context_types.hpp"

namespace tt::umd {
class TlbWindow;
}

namespace tt::tt_metal {
class IDevice;
namespace experimental {
class PinnedMemory;
}
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {

class MeshDevice;

// A fitted device->host clock mapping: host_ns = (device_cycles - device_cycle_offset) / frequency.
struct RealtimeProfilerClockFit {
    double frequency = 0.0;
    int64_t device_cycle_offset = 0;
    int64_t rtt_ticks = 0;
};

// Process-global per-physical-chip calibration cache so a rapid MeshDevice reopen reuses a recent fit instead of
// re-running the handshake (device WALL_CLOCK is free-running across close).
class RealtimeProfilerCalibrationCache {
public:
    std::optional<RealtimeProfilerClockFit> try_get(
        uint32_t chip_id, std::chrono::steady_clock::time_point now, std::chrono::steady_clock::duration max_age) const;
    void put(uint32_t chip_id, const RealtimeProfilerClockFit& fit, std::chrono::steady_clock::time_point now);

private:
    struct Entry {
        RealtimeProfilerClockFit fit;
        std::chrono::steady_clock::time_point updated_at;
    };
    mutable std::mutex mu_;
    std::unordered_map<uint32_t, Entry> by_chip_;
};

RealtimeProfilerCalibrationCache& rt_profiler_calibration_cache();

// Per-device host<->device clock sync: fits the device->host mapping, re-anchors it on a periodic servo, and owns the
// handshake transport (token write, host ACK word, round-trip probe). service_servo() and mapping() must run on a
// single thread (the receiver thread in steady state), which is what lets mapping() be lock-free; bring-up
// (configure/run_fit) runs before that thread starts.
class RealtimeProfilerClockSync {
public:
    RealtimeProfilerClockSync() = default;

    void configure(
        ContextId context_id,
        IDevice* device,
        uint32_t chip_id,
        const CoreCoord& profiler_core,
        const MeshCoordinate& mesh_coord,
        bool hugepage_fallback,
        uint32_t sync_host_ts_addr,
        const std::shared_ptr<MeshDevice>& mesh_device,
        uint32_t ack_enc_addr,
        uint32_t ack_lo_addr,
        uint32_t ack_hi_addr);

    bool try_restore_from_cache(std::chrono::steady_clock::time_point now);
    bool run_fit(uint32_t num_samples);
    // Also serves the first anchor at bring-up (due, since not yet anchored).
    void service_servo(std::chrono::steady_clock::time_point now);

    experimental::ProgramRealtimeClockSync mapping() const;
    double frequency() const { return frequency_; }
    bool has_ack() const { return ack_host_ptr_ != nullptr; }

private:
    struct Probe {
        int64_t host_before;
        int64_t rtt_ticks;
        uint64_t device_time;
    };
    std::optional<Probe> probe();
    void write_timestamp(uint32_t value);
    int64_t measure_rtt(int64_t host_before, uint32_t host_time_id);
    uint32_t read_ack() const;
    uint64_t read_device_time() const;
    void configure_write_path();
    void configure_ack_word(
        const std::shared_ptr<MeshDevice>& mesh_device, uint32_t enc_addr, uint32_t lo_addr, uint32_t hi_addr);
    void fit_from_samples(const std::vector<Probe>& samples, int64_t host_start_time, double default_frequency);
    void reanchor(int64_t host_anchor, uint64_t device_anchor);
    bool servo_due(std::chrono::steady_clock::time_point now) const;

    ContextId context_id_{};
    IDevice* device_ = nullptr;
    uint32_t chip_id_ = 0;
    CoreCoord profiler_core_;
    MeshCoordinate mesh_coord_ = MeshCoordinate(0);
    // No-IOMMU + 64-bit-PCIe host: the ACK word is a CQ-sysmem slot whose device PCIe writes may be non-snooped, so
    // reads must evict the cache line first.
    bool hugepage_fallback_ = false;
    uint32_t sync_host_ts_addr_ = 0;
    uint32_t sync_seq_ = 0;  // per-handshake token, never 0 so a stale ACK can't false-match
    // Sole owner of the pinned-path ACK buffer: PinnedMemory maps only a raw pointer, so dropping this frees it.
    std::shared_ptr<uint32_t[]> ack_host_backing_;
    std::shared_ptr<experimental::PinnedMemory> ack_pinned_;
    volatile uint32_t* ack_host_ptr_ = nullptr;
    // Blackhole fast path: static-L1 TLB window for a one-store MMIO token write; null elsewhere. Owned by UMD.
    tt::umd::TlbWindow* sync_tlb_ = nullptr;

    double frequency_ = 0.0;
    int64_t device_cycle_offset_ = 0;  // device_cycle = frequency * host_ns + offset
    int64_t rtt_ticks_ = 0;            // most recent accepted handshake RTT; half is the reported sync uncertainty
    std::optional<std::chrono::steady_clock::time_point> last_reanchor_at_;
};

}  // namespace tt::tt_metal::distributed
