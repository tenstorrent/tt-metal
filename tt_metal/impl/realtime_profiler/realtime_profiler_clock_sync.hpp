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
#include <tt-metalium/experimental/realtime_profiler.hpp>

#include "context/context_types.hpp"
#include "tt_metal/impl/realtime_profiler/realtime_profiler_clock_model.hpp"

namespace tt::umd {
class TlbWindow;
}

namespace tt::tt_metal {

class IDevice;

// Everything RealtimeProfilerClockSync needs to reach one chip's clock.
struct RealtimeProfilerClockSyncConfig {
    ContextId context_id{};
    IDevice* device = nullptr;  // the chip being synced; its chip id is taken from device->id()
    CoreCoord profiler_core;    // the reserved tensix running the profiler kernels
};

// The host end of one device's clock sync: reads the profiler core's free-running cycle counter, bracketed between two
// host clock reads, and drives a ClockModel from the samples it measures. Everything about what those samples mean --
// the fit, the re-anchor policy, the error bar -- belongs to ClockModel; this class is how a sample gets taken.
//
// Nothing runs on device for this. The counter is a hardware register the NOC serves directly, so a read costs the
// profiler core nothing and cannot be delayed by whatever its push loop is doing -- which is what bounds the sample's
// accuracy at half the bracket rather than half a software round trip.
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

    RealtimeProfilerClockSync();
    ~RealtimeProfilerClockSync();

    void configure(const RealtimeProfilerClockSyncConfig& config);

    // Adopts a cached frequency and takes one probe to anchor on it, skipping the multi-sample fit. False when there is
    // no usable cache entry, or the anchor probe failed and a full fit is needed.
    bool try_restore_calibration(std::chrono::steady_clock::time_point now);
    // Fits frequency and an initial anchor from a batch of samples. How many samples that takes is a property of the
    // fit, so the count lives with it rather than at the call site.
    bool calibrate();
    // Takes a burst of samples and re-anchors the mapping on the tightest. False only when the device did not answer;
    // declining to re-anchor on a loose bracket is normal and returns true.
    bool resync();

    // Reader side, safe to call from any thread while the sync thread is re-anchoring. Returns both halves together;
    // read separately they could come from different handshakes.
    [[nodiscard]] Calibration calibration() const;

private:
    // Maps the counter into host address space so a read is one load. Leaves the mapping null if it cannot be had,
    // which probe() falls back on.
    void configure_clock_read_path();
    // One bracketed read of the counter: host clock, counter, host clock. The sample's rtt is the bracket width, so
    // ClockModel places the anchor at its midpoint and reports half of it as the error, exactly as for a round trip.
    // Empty when the read failed.
    std::optional<ClockSyncSample> probe();
    // The tightest of `probes` reads. Ranked rather than compared against a threshold: under record load the whole
    // bracket distribution shifts, and an absolute threshold would reject every read in a pass instead of its best.
    std::optional<ClockSyncSample> best_of(int probes);
    // Copies the model's current mapping into the fields calibration() serves.
    void publish_mapping();

    ContextId context_id_{};
    IDevice* device_ = nullptr;
    uint32_t chip_id_ = 0;
    CoreCoord profiler_core_;
    // Virtual coordinates of profiler_core_, resolved once so the resolve does not sit inside the bracket.
    CoreCoord profiler_core_virtual_;
    uint32_t wall_clock_addr_lo_ = 0;
    uint32_t wall_clock_addr_hi_ = 0;
    // A UC window onto the counter, so a sample is a load rather than a UMD call. The generic register read holds a
    // chip-wide mutex and rewrites the TLB configuration registers over PCIe on every call, which lands inside the
    // bracket and widens it by ~450ns -- the bracket is the error bound, so that is the whole quantity being minimised.
    std::unique_ptr<tt::umd::TlbWindow> clock_tlb_;
    volatile uint32_t* mapped_clock_lo_ = nullptr;
    volatile uint32_t* mapped_clock_hi_ = nullptr;
    // The high word only advances when the low word wraps, which at ~1.35GHz is every 3.2s, so it is tracked rather
    // than re-read: a probe reads it only when the low word goes backwards, or when enough time has passed that a
    // wrap could have been missed. That halves the PCIe reads a probe costs.
    uint32_t cached_clock_hi_ = 0;
    uint32_t last_clock_lo_ = 0;
    std::chrono::steady_clock::time_point last_probe_at_{};

    // Seeded with the commanded AICLK in configure(), refined by calibrate(), re-anchored by resync(). Owned by the
    // sync thread; the receiver never touches it, only the published fields below.
    ClockModel model_;

    // A seqlock: the sync thread writes these, the receiver reads them on every drain. An even sequence means the
    // fields are settled; the payload is atomic and read relaxed so a torn read is not undefined behaviour, and the
    // sequence's acquire/release is what orders it. Too wide for a lock-free atomic, and far too hot for a mutex.
    std::atomic<uint32_t> mapping_seq_{0};
    std::atomic<int64_t> mapping_device_cycle_offset_{0};
    std::atomic<std::chrono::nanoseconds> mapping_sync_error_{};
    std::atomic<double> mapping_frequency_{0.0};
};

}  // namespace tt::tt_metal
