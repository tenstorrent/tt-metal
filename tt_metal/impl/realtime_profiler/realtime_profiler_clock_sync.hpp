// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>

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

// Everything RealtimeProfilerClockSync needs to reach one chip's handshake fields.
struct RealtimeProfilerClockSyncConfig {
    ContextId context_id{};
    IDevice* device = nullptr;  // the chip being synced; its chip id is taken from device->id()
    // Borrowed for the duration of configure() only, to pin the host ACK buffer and resolve its NOC address.
    distributed::MeshDevice* mesh_device = nullptr;
    CoreCoord profiler_core;  // the reserved tensix running the profiler kernels
    distributed::MeshCoordinate mesh_coord = distributed::MeshCoordinate(0);
    // Set on a 32-bit-PCIe arch with no IOMMU (Wormhole): the ACK word falls back to a CQ-sysmem slot whose device
    // PCIe writes may be non-snooped, so reads must evict the cache line first. The 64-bit-PCIe archs have no such
    // fallback and are disabled outright when the IOMMU is off, so they never set this.
    bool hugepage_fallback = false;
    // Base of realtime_profiler_msg_t on profiler_core. The individual sync field addresses are resolved from it via
    // the HAL, so only this class needs to know which of those fields the handshake uses.
    uint32_t msg_base_addr = 0;
};

// The host end of one device's clock-sync handshake: owns the transport (token write, pinned host ACK word, round-trip
// probe) and drives a ClockModel from the samples it measures. Everything about what those samples mean -- the fit, the
// re-anchor policy, the error bar -- belongs to ClockModel; this class is how a sample gets taken.
//
// resync() runs on the sync thread and calibration() is read from the receiver's, served from a seqlock so the two
// never touch the same state. Bring-up (configure/calibrate) precedes both threads.
class RealtimeProfilerClockSync {
public:
    // The device clock as currently calibrated: the frequency it runs at, and the offset placing it on the host clock
    // together with how far that placement could be out. Everything a record needs to carry.
    struct Calibration {
        experimental::ProgramRealtimeClockSync mapping{};
        double frequency = 0.0;
    };

    RealtimeProfilerClockSync() = default;

    void configure(const RealtimeProfilerClockSyncConfig& config);

    // Adopts a cached frequency and takes one probe to anchor on it, skipping the multi-sample fit. False when there is
    // no usable cache entry, or the anchor probe failed and a full fit is needed.
    bool try_restore_calibration(std::chrono::steady_clock::time_point now);
    // Fits frequency and an initial anchor from a batch of handshakes. How many samples that takes is a property of
    // the fit, so the count lives with it rather than at the call site.
    bool calibrate();
    // Runs one handshake and re-anchors the mapping if the round trip was good enough to be worth it. False only when
    // the device did not answer; declining to re-anchor on a slow round trip is normal and returns true.
    bool resync();

    // Reader side, safe to call from any thread while the sync thread is re-anchoring. Returns both halves together;
    // read separately they could come from different handshakes.
    [[nodiscard]] Calibration calibration() const;

private:
    // L1 addresses of the handshake fields within the profiler core's realtime_profiler_msg_t.
    struct SyncL1Addrs {
        uint32_t token = 0;          // host->device handshake token
        uint32_t ack_host_addr = 0;  // base of the [lo, hi] host ACK buffer address pair
    };

    // `timeout` bounds the busy-poll only; a handshake slower than what the model would accept is wasted work, so
    // bring-up and steady state pass different bounds.
    std::optional<ClockSyncSample> probe(std::chrono::nanoseconds timeout);
    void write_token(uint32_t value);
    // Times the round trip to the device timestamp landing, then waits for the token. Empty when the device stopped
    // responding.
    std::optional<std::chrono::nanoseconds> measure_rtt(
        std::chrono::steady_clock::time_point host_before, uint32_t token, std::chrono::nanoseconds timeout);
    // No-op unless hugepage_fallback_; see the definition for why it sits inside the poll loop.
    void evict_ack_line() const;
    uint32_t read_ack() const;
    uint64_t read_device_time() const;
    SyncL1Addrs resolve_l1_addrs(uint32_t msg_base_addr) const;
    void configure_write_path();
    void configure_ack_word(distributed::MeshDevice& mesh_device);
    // Copies the model's current mapping into the fields calibration() serves.
    void publish_mapping();

    ContextId context_id_{};
    IDevice* device_ = nullptr;
    uint32_t chip_id_ = 0;
    CoreCoord profiler_core_;
    distributed::MeshCoordinate mesh_coord_ = distributed::MeshCoordinate(0);
    bool hugepage_fallback_ = false;  // see RealtimeProfilerClockSyncConfig::hugepage_fallback
    SyncL1Addrs l1_;
    uint32_t sync_seq_ = 0;          // host->device request flag, never 0 so the device can tell a request is pending
    uint64_t last_device_time_ = 0;  // last timestamp read out of the ACK buffer; a change is the acknowledgement
    // Sole owner of the pinned-path ACK buffer: PinnedMemory maps only a raw pointer, so dropping this frees it.
    std::shared_ptr<uint32_t[]> ack_host_backing_;
    std::shared_ptr<experimental::PinnedMemory> ack_pinned_;
    volatile uint32_t* ack_host_ptr_ = nullptr;
    // Static TLB window for a one-store MMIO token write, where the profiler core has one; null otherwise. Owned by
    // UMD, not by this class.
    tt::umd::TlbWindow* sync_tlb_ = nullptr;
    // The token's mapped address inside sync_tlb_, resolved once at configure time.
    volatile uint32_t* sync_doorbell_ = nullptr;

    // Seeded with the commanded AICLK in configure(), refined by calibrate(), re-anchored by resync(). Owned by the
    // sync thread; the receiver never touches it, only the published fields below.
    ClockModel model_;

    // A seqlock: the sync thread writes these, the receiver reads them on every drain. An even sequence means the
    // fields are settled; the payload is atomic and read relaxed so a torn read is not undefined behaviour, and the
    // sequence's acquire/release is what orders it. Too wide for a lock-free atomic, and far too hot for a mutex.
    std::atomic<uint32_t> mapping_seq_{0};
    std::atomic<int64_t> mapping_device_cycle_offset_{0};
    std::atomic<uint64_t> mapping_sync_error_ns_{0};
    std::atomic<double> mapping_frequency_{0.0};
};

}  // namespace tt::tt_metal
