// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "context/context_types.hpp"

namespace tt::umd {
class TlbWindow;
}

namespace tt::tt_metal {

class IDevice;

// Maps device tick timestamps onto host time by interpolating between retained clock probes. Pure arithmetic over
// what it is fed: probes come from whoever reads the device clock -- DeviceClockSync in production, tests
// directly -- and nothing here touches a device.
//
// Single-threaded: retain() and map_record() belong to the thread draining the device.
class DeviceClockMapping {
public:
    // A probe, placed at the midpoint of the bracket its read fell in. Two of them map any device timestamp between
    // them via their secant, whatever the clock did in between.
    struct Anchor {
        std::chrono::steady_clock::time_point host;
        uint64_t ticks = 0;
        std::chrono::nanoseconds bracket{};
    };

    // What one record publishes with, covering both of its timestamps: an affine device-to-host mapping
    // host_ns = (ticks - device_cycle_offset) / frequency, with sync_error the estimated error of a host time
    // derived from it.
    //
    // The rate is the secant over the widest measured window the record itself spans: its probe pair, its own two
    // placements, or -- for a record that outlived the ring -- the whole retained history. Timestamps land exactly
    // where their probes placed them, so sync_error is the anchoring reads' brackets plus the measured departure of
    // the clock from their secant; only a ring-outliving record's start, derived through the ring-wide rate, is
    // additionally charged that slope's noise over the ride. What sync_error cannot cover: the part of a rate step
    // the bow measurement's read-noise deduction swallows, and -- for a ring-outliving start only -- a sustained rate
    // change older than the retained history, which is unobservable in principle. It is an estimate that tracks the
    // measured clock, not a worst-case ceiling; the didt suite bounds what the gap costs in practice.
    struct RecordMapping {
        int64_t device_cycle_offset = 0;
        std::chrono::nanoseconds sync_error{};
        double frequency = 0.0;
    };

    // Probes retained per device: 2 seconds of history at the default sync_interval, covering decode latency and the
    // peek's observation lag with two orders of magnitude to spare (96 KB per device). The pair around an undecoded
    // record's end can never be overwritten: the receiver probes once per drained batch, and on the idle floor only
    // when nothing is in flight, so a record sees a few dozen probes at most between ending and being decoded
    // (asserted against the FIFO geometry in the receiver). A start older than the ring belongs to a long-running
    // program, whose start the receiver pins while its probes are still fresh -- see pin_start.
    static constexpr size_t kProbeHistoryCapacity = std::chrono::seconds(2) / std::chrono::microseconds(500);

    // Retains `probe`. One that does not advance both clocks past the newest retained probe is dropped -- a real
    // counter and steady_clock cannot produce one -- so the ring is strictly monotone in host and ticks and no reader
    // below has to check a pair's orientation.
    void retain(const Anchor& probe);

    // Pins `ticks` to its placement between the probes around it now, so a record starting there can still be mapped
    // exactly after those probes are gone. One slot: a single command queue runs one program at a time, so at most
    // one start is ever in flight, and it is consumed (or superseded) by the next record mapped. Idempotent per
    // ticks value; a value with no retained probe before it is ignored.
    void pin_start(uint64_t ticks);

    // What to publish a record with. Nullopt only when no retained probe precedes either timestamp, which after
    // warm-up cannot happen to a real record; the caller treats it as corruption and rejects the page.
    //
    // Records arrive in tick order, so the probe pair one is placed between also covers the next several; that pair
    // is held here and refreshed when a record's start passes it.
    [[nodiscard]] std::optional<RecordMapping> map_record(uint64_t start_ticks, uint64_t end_ticks);

    // sync_error of the last record mapped, which is the bound currently standing for this device.
    [[nodiscard]] std::chrono::nanoseconds last_sync_error() const { return last_sync_error_; }

private:
    // The probe pair around a run of records, resolved once and reused until a record's start passes `close_ticks`.
    // The slope is carried both ways round because the alternative is a division per record on the drain thread.
    struct Chord {
        std::chrono::nanoseconds sync_error{};
        double frequency = 0.0;
        uint64_t open_ticks = 0;
        double open_host_ns = 0.0;
        double inv_chord_rate = 0.0;
        uint64_t close_ticks = 0;
    };

    // The tightest pair of retained probes around `ticks`, as a chord. Nullopt only when the ring holds nothing
    // before `ticks`. Total otherwise, so a caller that has just probed always gets an answer and nothing is held
    // back for a later pass -- a refusal here once stalled a device's whole data path, being re-asked about the same
    // record with inputs that never changed.
    [[nodiscard]] std::optional<Chord> chord_around(uint64_t ticks) const;

    [[nodiscard]] static double host_ns_on(const Chord& chord, uint64_t ticks) {
        return chord.open_host_ns +
               (static_cast<double>(ticks) - static_cast<double>(chord.open_ticks)) * chord.inv_chord_rate;
    }

    // The mapping that restates one timestamp's placement in terms of `frequency`, so that timestamp lands exactly
    // where it was placed whatever rate is published with it.
    [[nodiscard]] static RecordMapping anchored(
        double frequency, uint64_t anchor_ticks, double anchor_host_ns, std::chrono::nanoseconds error);

    // The endpoint term of an interpolated timestamp's error: it lands on the secant through two measured points, so
    // it inherits how well those points are placed. A clock that moved within the interval adds to this; see
    // measured_bow.
    [[nodiscard]] static std::chrono::nanoseconds interpolation_error(const Anchor& open, const Anchor& close) {
        return std::max(open.bracket, close.bracket) / 2;
    }

    // How far `interior` lies off the chord through `open` and `close`, less its own read noise. A probe inside a
    // chord was not fitted to it, so this is the clock's departure at that point -- measured, where the alternative is
    // to infer it from how much two chords' slopes differ and hope the difference was the clock rather than the reads.
    [[nodiscard]] static std::chrono::nanoseconds departure_from_chord(
        const Anchor& open, const Anchor& close, const Anchor& interior);

    // What the retained probes say the clock did between `close_index - 1` and `close_index` that the line through
    // them does not capture. Zero when there is no third probe to read it from, which is the absence of evidence
    // rather than a claim of linearity.
    [[nodiscard]] std::chrono::nanoseconds measured_bow(uint64_t close_index) const;

    // Index of the oldest retained probe whose counter read reached `ticks`, or probes_end_ when none has. Probes are
    // appended in tick order, so this bisects: the retained span grows with the backlog, and scanning it per record
    // is what turns a backlog into a stall.
    [[nodiscard]] uint64_t first_probe_at_or_past(uint64_t ticks) const;

    [[nodiscard]] uint64_t oldest_probe() const {
        return probes_end_ > probe_history_.size() ? probes_end_ - probe_history_.size() : 0;
    }
    [[nodiscard]] const Anchor& probe_at(uint64_t index) const { return probe_history_[index % probe_history_.size()]; }

    // Overwrites its oldest entry when full, so nothing has to be retired. Not a deque: a deque puts a block
    // malloc/free on the drain thread every few probes, and glibc hands large blocks back with munmap, which takes
    // mmap_lock for write and stalls every other thread in the process.
    std::array<Anchor, kProbeHistoryCapacity> probe_history_{};
    uint64_t probes_end_ = 0;

    std::optional<Chord> chord_;

    // A start observed while it was still inside the ring, held for however long its program runs.
    struct Pin {
        uint64_t ticks = 0;
        double host_ns = 0.0;
        std::chrono::nanoseconds error{};
    };
    std::optional<Pin> pinned_start_;
    uint64_t last_pin_ticks_ = 0;

    std::chrono::nanoseconds last_sync_error_{};
};

// Reads a tensix core's free-running cycle counter over PCIe and feeds the probes into the DeviceClockMapping it
// owns. Nothing runs on device for this: the NOC serves the counter directly, so a read cannot be delayed by
// whatever kernels the core is running.
//
// Single-threaded: warm_up() runs before the driving thread starts and every later call belongs to that thread.
class DeviceClockSync {
public:
    using Anchor = DeviceClockMapping::Anchor;
    using RecordMapping = DeviceClockMapping::RecordMapping;

    // Floor on how often each device's clock is read, under the probe every non-empty read already takes. Probe
    // spacing is the width of the pair a record is placed between, and a rate step inside that pair misplaces it by
    // step * width / 4. AICLK only moves on the ARC firmware's 1ms DVFS timer (dvfs.c:DVFSChange in
    // tt-zephyr-platforms), so anything well under a millisecond resolves a clock that cannot have changed within it:
    // p90 sync error is flat from 100us out to 500us and only breaks upward at the tick. Wormhole pins the upper end,
    // failing the didt p99 limit at 1ms where Blackhole is still at 1.8us.
    static constexpr auto kSyncInterval = std::chrono::microseconds(500);
    static constexpr std::chrono::nanoseconds sync_interval() { return kSyncInterval; }

    // Extra reads taken only while the bracket is still wider than reads have recently been coming back at. Fires
    // rarely -- reads per resync reads 1.00 to two decimals -- but the ones it does take are the wide reads, and the
    // widest bracket is what the error bound is made of: removing it moved stress sync error p99 from 0.63us to
    // 5.05us and max from 1.28us to 25.06us.
    static constexpr int kResyncProbes = 4;

    // `clock_core` is the tensix whose wall-clock register is read; every tensix serves the same counter, so any
    // core the caller already owns will do.
    DeviceClockSync(ContextId context_id, IDevice* device, CoreCoord clock_core);
    ~DeviceClockSync();

    // False when no UC window could be mapped onto the clock register. There is no slower path to fall back to: the
    // generic register read holds a chip-wide mutex and rewrites the window's configuration over PCIe on every call,
    // which lands inside the bracket that is the whole error bound, so a device without a window is not profiled.
    [[nodiscard]] bool has_direct_clock_read() const { return mapped_clock_lo_ != nullptr; }

    // Takes a few spaced probes so the first record already has a pair to be placed between, and a third to read the
    // clock's departure from. Runs before the receiver thread starts.
    void warm_up();

    // Takes one probe and retains it. Cannot fail: the read is a load through an already-mapped window, and a
    // device without one is refused at construction, so the probe history only grows and a caller that has just
    // resynced can take a usable pair for granted.
    void resync();

    // Whether this device is due a probe on the interval floor, having drained nothing to trigger one.
    [[nodiscard]] bool due_for_probe(std::chrono::steady_clock::time_point now) const {
        return now - last_probe_at_ >= sync_interval();
    }

    // Points the peek at dispatch_s's kernel_start_a/b fields, through UMD's statically-mapped L1 window on that
    // core. Without a window (or without dispatch_s) the peek stays disarmed and long-program starts fall back to
    // the ring-wide rate.
    void configure_program_start_peek(CoreCoord dispatch_s_virtual, uint32_t start_a_addr, uint32_t start_b_addr);

    // Reads the start timestamp of whatever program dispatch_s is currently waiting on and pins its placement while
    // the probes around it are fresh. dispatch_s holds the field stable for the program's whole run, and the record
    // it eventually pushes carries these exact bits, so a torn or stale read simply never matches anything. Called
    // from the idle-floor path: a device running a program drains nothing, which is exactly when this matters.
    void peek_running_program_start();

    [[nodiscard]] std::optional<RecordMapping> map_record(uint64_t start_ticks, uint64_t end_ticks) {
        return mapping_.map_record(start_ticks, end_ticks);
    }

    // sync_error of the last record mapped, which is the bound currently standing for this device.
    [[nodiscard]] std::chrono::nanoseconds last_published_sync_error() const { return mapping_.last_sync_error(); }

private:
    void configure_clock_read_path();
    // Placed at the midpoint of its read's bracket, which is where the counter could have been read.
    Anchor probe();
    // Ranked, not thresholded: under record load the whole bracket distribution shifts.
    Anchor best_of(int probes);

    ContextId context_id_;
    uint32_t chip_id_ = 0;
    // Resolved once so the resolve does not sit inside the bracket.
    CoreCoord clock_core_virtual_;
    uint32_t wall_clock_addr_lo_ = 0;
    uint32_t wall_clock_addr_hi_ = 0;
    // A UC window makes a probe a plain load. Required, not preferred: see has_direct_clock_read.
    std::unique_ptr<tt::umd::TlbWindow> clock_tlb_;
    volatile uint32_t* mapped_clock_lo_ = nullptr;
    volatile uint32_t* mapped_clock_hi_ = nullptr;

    // dispatch_s's kernel_start_{a,b} {hi, lo} words
    volatile uint32_t* peek_start_a_ = nullptr;
    volatile uint32_t* peek_start_b_ = nullptr;
    // Recent tightest-read bracket, as an EMA. best_of stops early against this to reduce probes.
    std::chrono::nanoseconds typical_bracket_{};

    // The high word only advances when the low word wraps; every 3.2s at ~1.35GHz.
    uint32_t cached_clock_hi_ = 0;
    uint32_t last_clock_lo_ = 0;
    std::chrono::steady_clock::time_point last_probe_at_;

    DeviceClockMapping mapping_;
};

}  // namespace tt::tt_metal
