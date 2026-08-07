// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/realtime_profiler_clock_sync.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <optional>
#include <thread>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <tt-metalium/device.hpp>
#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <umd/device/pcie/tlb_handle.hpp>
#include <umd/device/pcie/tlb_window.hpp>
#include <umd/device/types/tlb.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/common/env_lib.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

namespace tt::tt_metal {

namespace {

// The counter could have been read anywhere inside the bracket.
constexpr std::chrono::nanoseconds placement_error(std::chrono::nanoseconds bracket) { return bracket / 2; }

// How long a gap between probes forces the high word to be read from the device rather than derived by counting wraps.
// Two wraps inside one gap are indistinguishable from none, and 2^32 ticks is 3.2s at 1.35GHz -- 2.1s if AICLK ever
// reaches 2GHz -- so this has to stay well under the shortest wrap any part might have. A device is probed after every
// non-empty read and on a millisecond idle cadence besides, so reaching this at all means the receiver thread was
// stalled for a second, and then one extra PCIe read is exactly what is wanted.
constexpr auto kMaxProbeGapBeforeRereadingHi = std::chrono::seconds(1);

}  // namespace

void RealtimeProfilerClockMapping::retain(const Anchor& probe) {
    if (probes_end_ > oldest_probe()) {
        const Anchor& newest = probe_at(probes_end_ - 1);
        if (probe.ticks <= newest.ticks || probe.host <= newest.host) {
            return;
        }
    }
    probe_history_[probes_end_ % probe_history_.size()] = probe;
    ++probes_end_;
}

uint64_t RealtimeProfilerClockMapping::first_probe_at_or_past(uint64_t ticks) const {
    uint64_t lo = oldest_probe();
    uint64_t hi = probes_end_;
    while (lo < hi) {
        const uint64_t mid = lo + (hi - lo) / 2;
        if (probe_at(mid).ticks < ticks) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// A probe inside the chord is a point the secant was not fitted to, so its distance from the secant is the clock's
// departure there. Reads zero on a plateau, which is nearly always; zero with no interior probe means no evidence
// rather than no bow.
std::chrono::nanoseconds RealtimeProfilerClockMapping::departure_from_chord(
    const Anchor& open, const Anchor& close, const Anchor& interior) {
    const double span_ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(close.host - open.host).count());
    const double inv_rate = span_ns / static_cast<double>(close.ticks - open.ticks);
    const double on_chord_ns = static_cast<double>(open.host.time_since_epoch().count()) +
                               (static_cast<double>(interior.ticks) - static_cast<double>(open.ticks)) * inv_rate;
    const double measured_ns = static_cast<double>(interior.host.time_since_epoch().count());
    const auto departure = std::chrono::nanoseconds(static_cast<int64_t>(std::abs(measured_ns - on_chord_ns)));
    // Three brackets explain part of any distance before the clock does: the interior probe's own read, and the two
    // the line was drawn through, since a line through points known to +/-b/2 is itself only that well placed between
    // them.
    const auto explained_by_reads = placement_error(interior.bracket) + interpolation_error(open, close);
    return departure - std::min(departure, explained_by_reads);
}

std::chrono::nanoseconds RealtimeProfilerClockMapping::measured_bow(uint64_t close_index) const {
    // The pair placing a record is adjacent, so nothing lies inside it and the departure has to be read one probe out.
    // Scaled by the span ratio, because a rate step bows the trajectory in proportion to the span it acts over.
    if (close_index < oldest_probe() + 2) {
        return {};
    }
    const Anchor& outer = probe_at(close_index - 2);
    const Anchor& open = probe_at(close_index - 1);
    const Anchor& close = probe_at(close_index);
    const auto departure = departure_from_chord(outer, close, open);
    if (departure == std::chrono::nanoseconds::zero()) {
        return {};
    }
    const auto pair_span = close.host - open.host;
    const auto triple_span = close.host - outer.host;
    return departure * pair_span.count() / triple_span.count();
}

std::optional<RealtimeProfilerClockMapping::Chord> RealtimeProfilerClockMapping::chord_around(uint64_t ticks) const {
    const uint64_t begin = oldest_probe();
    if (probes_end_ - begin < 2) {
        return std::nullopt;
    }
    // The caller probes after reading a batch, so a record's timestamps are always behind the newest probe and this
    // clamp is a no-op; it exists so the pair is a pair whatever it is handed.
    const uint64_t close_index = std::min(first_probe_at_or_past(ticks), probes_end_ - 1);
    if (close_index == begin) {
        return std::nullopt;
    }
    const Anchor& open = probe_at(close_index - 1);
    const Anchor& closing = probe_at(close_index);
    // There is no minimum span: a timestamp between two probes cannot be further off than the worse of them however
    // short the span, because a straight line's largest departure from another is at their endpoints, while the
    // clock's own departure from the line grows with span.
    const double span_ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(closing.host - open.host).count());
    return Chord{
        .sync_error = interpolation_error(open, closing) + measured_bow(close_index),
        .frequency = static_cast<double>(closing.ticks - open.ticks) / span_ns,
        .open_ticks = open.ticks,
        .open_host_ns = static_cast<double>(open.host.time_since_epoch().count()),
        .inv_chord_rate = span_ns / static_cast<double>(closing.ticks - open.ticks),
        .close_ticks = closing.ticks,
    };
}

RealtimeProfilerClockMapping::RecordMapping RealtimeProfilerClockMapping::anchored(
    double frequency, uint64_t anchor_ticks, double anchor_host_ns, std::chrono::nanoseconds error) {
    return RecordMapping{
        experimental::ProgramRealtimeClockSync{
            .device_cycle_offset = std::llround(static_cast<double>(anchor_ticks) - frequency * anchor_host_ns),
            .sync_error = error,
        },
        frequency};
}

void RealtimeProfilerClockMapping::pin_start(uint64_t ticks) {
    if (ticks == 0 || ticks == last_pin_ticks_) {
        return;
    }
    last_pin_ticks_ = ticks;
    const std::optional<Chord> chord = chord_around(ticks);
    if (!chord.has_value()) {
        return;
    }
    pinned_start_ = Pin{ticks, host_ns_on(*chord, ticks), chord->sync_error};
}

std::optional<RealtimeProfilerClockMapping::RecordMapping> RealtimeProfilerClockMapping::map_record(
    uint64_t start_ticks, uint64_t end_ticks) {
    if (!chord_.has_value() || start_ticks > chord_->close_ticks) {
        chord_ = chord_around(start_ticks);
    }

    RecordMapping mapping;
    if (!chord_.has_value()) {
        // The start predates every retained probe: the program ran longer than the ring spans. Its end always has a
        // pair -- see kProbeHistoryCapacity -- so the record stands on measured ground either way.
        const std::optional<Chord> end_chord = chord_around(end_ticks);
        if (!end_chord.has_value()) {
            return std::nullopt;
        }
        const double end_host_ns = host_ns_on(*end_chord, end_ticks);
        if (pinned_start_.has_value() && pinned_start_->ticks == start_ticks) {
            // The peek placed this start while its probes were fresh, so both endpoints are measured and the record
            // takes its own secant, exactly as if it had never outlived the ring.
            const double frequency =
                static_cast<double>(end_ticks - start_ticks) / (end_host_ns - pinned_start_->host_ns);
            mapping = anchored(
                frequency, start_ticks, pinned_start_->host_ns, std::max(pinned_start_->error, end_chord->sync_error));
        } else {
            // No pin, so the start rides the widest rate window there is -- the whole, by definition full, ring --
            // charged that slope's own noise. An estimate, not a bound: rate history older than the ring is gone.
            const Anchor& ring_oldest = probe_at(oldest_probe());
            const Anchor& ring_newest = probe_at(probes_end_ - 1);
            const double ring_span_ns = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(ring_newest.host - ring_oldest.host).count());
            const double frequency = static_cast<double>(ring_newest.ticks - ring_oldest.ticks) / ring_span_ns;
            const double ride_ns = static_cast<double>(end_ticks - start_ticks) / frequency;
            const auto ride_noise = std::chrono::nanoseconds(std::llround(
                ride_ns * static_cast<double>((ring_oldest.bracket + ring_newest.bracket).count()) / ring_span_ns));
            mapping = anchored(frequency, end_ticks, end_host_ns, end_chord->sync_error + ride_noise);
        }
    } else if (const Chord& chord = *chord_; end_ticks <= chord.close_ticks) {
        // Both timestamps sit on this chord and the published rate is its own slope, so both land exactly where the
        // pair places them and the pair's bound is the whole story.
        mapping = anchored(chord.frequency, start_ticks, host_ns_on(chord, start_ticks), chord.sync_error);
    } else {
        // A record that outlives its chord takes the secant through its own two placements: any other rate puts the
        // difference, times the record's length, onto its end -- tens of microseconds on a millisecond program,
        // enough to place ends after the read that had already carried them to the host.
        const std::optional<Chord> end_chord = chord_around(end_ticks);
        TT_ASSERT(end_chord.has_value());  // end > chord.close_ticks, so a retained probe precedes it
        const double start_host_ns = host_ns_on(chord, start_ticks);
        const double end_host_ns = host_ns_on(*end_chord, end_ticks);
        // The ring is strictly monotone, so the two placements are ordered and the secant is well defined.
        const double frequency = static_cast<double>(end_ticks - start_ticks) / (end_host_ns - start_host_ns);
        mapping = anchored(frequency, start_ticks, start_host_ns, std::max(chord.sync_error, end_chord->sync_error));
    }
    // Consumed on match, superseded otherwise: records arrive in tick order, so a start at or past the pin means the
    // pinned program's record has now been mapped (this one) or will never arrive.
    if (pinned_start_.has_value() && start_ticks >= pinned_start_->ticks) {
        pinned_start_.reset();
    }
    last_sync_error_ = mapping.clock_sync.sync_error;
    return mapping;
}

std::chrono::nanoseconds RealtimeProfilerClockSync::sync_interval() {
    static const std::chrono::nanoseconds interval =
        std::chrono::microseconds(tt::parse_env<uint32_t>("TT_RT_PROFILER_SYNC_INTERVAL_US", 500));
    return interval;
}

RealtimeProfilerClockSync::RealtimeProfilerClockSync(ContextId context_id, IDevice* device, CoreCoord profiler_core) :
    context_id_(context_id),
    chip_id_(device->id()),
    profiler_core_virtual_(device->virtual_core_from_logical_core(profiler_core, CoreType::WORKER)) {
    TTZoneScopedDN(RT_PROFILER, "ClockSyncConfigure");
    const auto& hal = MetalContext::instance(context_id_).hal();
    wall_clock_addr_lo_ = hal.get_tensix_wall_clock_reg_addr_lo();
    wall_clock_addr_hi_ = hal.get_tensix_wall_clock_reg_addr_hi();
    configure_clock_read_path();
}

RealtimeProfilerClockSync::~RealtimeProfilerClockSync() = default;

void RealtimeProfilerClockSync::configure_clock_read_path() {
    try {
        auto* tlb_manager =
            MetalContext::instance(context_id_).get_cluster().get_driver()->get_chip(chip_id_)->get_tlb_manager();
        if (tlb_manager == nullptr) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {}: no TLB manager, so the clock register cannot be mapped",
                chip_id_);
            return;
        }
        tt::umd::tlb_data cfg{};
        cfg.local_offset = wall_clock_addr_lo_;
        cfg.x_end = profiler_core_virtual_.x;
        cfg.y_end = profiler_core_virtual_.y;
        cfg.ordering = tt::umd::tlb_data::Strict;
        clock_tlb_ = tlb_manager->allocate_tlb_window(cfg, tt::umd::TlbMapping::UC);
        if (clock_tlb_ == nullptr) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {}: no UC TLB window available for the clock register",
                chip_id_);
            return;
        }
        const uint64_t local = clock_tlb_->handle_ref().get_config().local_offset;
        auto* base = clock_tlb_->handle_ref().get_base();
        mapped_clock_lo_ = reinterpret_cast<volatile uint32_t*>(base + (wall_clock_addr_lo_ - local));
        mapped_clock_hi_ = reinterpret_cast<volatile uint32_t*>(base + (wall_clock_addr_hi_ - local));
    } catch (const std::exception& e) {
        log_warning(
            tt::LogMetal, "[Real-time profiler] Device {}: could not map the clock register ({})", chip_id_, e.what());
    }
}

void RealtimeProfilerClockSync::configure_program_start_peek(
    CoreCoord dispatch_s_virtual, uint32_t start_a_addr, uint32_t start_b_addr) {
    try {
        const tt_xy_pair tlb_core(dispatch_s_virtual.x, dispatch_s_virtual.y);
        auto* tlb_manager =
            MetalContext::instance(context_id_).get_cluster().get_driver()->get_chip(chip_id_)->get_tlb_manager();
        const uint32_t span = start_b_addr + 2 * sizeof(uint32_t) - start_a_addr;
        if (tlb_manager == nullptr || !tlb_manager->is_tlb_mapped(tlb_core, start_a_addr, span)) {
            return;
        }
        auto* window = tlb_manager->get_tlb_window(tlb_core);
        if (window == nullptr) {
            return;
        }
        const uint64_t window_offset = window->get_base_address() - window->handle_ref().get_config().local_offset;
        auto* base = window->handle_ref().get_base();
        peek_start_a_ = reinterpret_cast<volatile uint32_t*>(base + window_offset + start_a_addr);
        peek_start_b_ = reinterpret_cast<volatile uint32_t*>(base + window_offset + start_b_addr);
    } catch (const std::exception& e) {
        log_debug(
            tt::LogMetal,
            "[Real-time profiler] Device {}: program-start peek disarmed ({}); long-program starts fall back to the "
            "ring-wide rate",
            chip_id_,
            e.what());
    }
}

void RealtimeProfilerClockSync::peek_running_program_start() {
    if (peek_start_a_ == nullptr) {
        return;
    }
    // {time_hi, time_lo}, written by dispatch_s. The banks ping-pong per program, so the running program's start is
    // the newer of the two; the retired bank's value is stale and pin_start's exact-match consumers ignore it.
    const uint64_t a = (static_cast<uint64_t>(peek_start_a_[0]) << 32) | peek_start_a_[1];
    const uint64_t b = (static_cast<uint64_t>(peek_start_b_[0]) << 32) | peek_start_b_[1];
    mapping_.pin_start(std::max(a, b));
}

RealtimeProfilerClockSync::Anchor RealtimeProfilerClockSync::probe() {
    TTZoneScopedDN(RT_PROFILER, "Probe");
    // The device is only asked for the high word when its value cannot be derived: on the first probe, and after a gap
    // long enough that a wrap could have gone unseen. Otherwise a wrap is counted rather than read -- the low word
    // wrapping advances the high word by exactly one -- which is both exact and one fewer PCIe access on the one path
    // where an access is least welcome.
    const bool must_read_hi = last_probe_at_ == std::chrono::steady_clock::time_point{} ||
                              std::chrono::steady_clock::now() - last_probe_at_ > kMaxProbeGapBeforeRereadingHi;

    std::chrono::steady_clock::time_point host_before;
    std::chrono::steady_clock::time_point host_after;
    uint32_t lo = 0;
    {  // latency-critical
        host_before = std::chrono::steady_clock::now();
        lo = *mapped_clock_lo_;
        host_after = std::chrono::steady_clock::now();
    }
    // Reading the low word latches the high one, so this read is of the value that goes with `lo`, and it stays
    // outside the bracket.
    if (must_read_hi) {
        cached_clock_hi_ = *mapped_clock_hi_;
    } else if (lo < last_clock_lo_) {
        ++cached_clock_hi_;
    }
    last_clock_lo_ = lo;
    last_probe_at_ = host_after;
    clock_reads_.fetch_add(1, std::memory_order_relaxed);
    const auto bracket = host_after - host_before;
    TTZoneValueD(RT_PROFILER, static_cast<uint64_t>(bracket.count()));
    const auto bracket_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(bracket);
    return Anchor{
        host_before + placement_error(bracket_ns), (static_cast<uint64_t>(cached_clock_hi_) << 32) | lo, bracket_ns};
}

RealtimeProfilerClockSync::Anchor RealtimeProfilerClockSync::best_of(int probes) {
    Anchor best = probe();
    for (int i = 1; i < probes; i++) {
        // Each read blocks the calling thread on PCIe, so the remaining ones are only worth taking while they might
        // still tighten the bracket. A read already at the recent typical width leaves them nothing to improve; the
        // full count is spent only when the link is making reads late.
        if (typical_bracket_ > std::chrono::nanoseconds::zero() &&
            best.bracket <= typical_bracket_ + typical_bracket_ / 2) {
            break;
        }
        const Anchor p = probe();
        if (p.bracket < best.bracket) {
            best = p;
        }
    }
    typical_bracket_ =
        typical_bracket_ == std::chrono::nanoseconds::zero() ? best.bracket : (typical_bracket_ * 7 + best.bracket) / 8;
    return best;
}

// Enough probes for the first record to have a pair to be placed between and a third to read the departure from,
// spaced a sync interval apart so the first chords are the same width as every later one. This replaced a 100-probe
// half-second fit whose result never reached a consumer.
void RealtimeProfilerClockSync::warm_up() {
    TTZoneScopedDN(RT_PROFILER, "ClockWarmUp");
    constexpr int kWarmUpProbes = 4;
    // Warms the cold PCIe path; its bracket is not representative and it is not retained.
    (void)probe();
    for (int i = 0; i < kWarmUpProbes; i++) {
        if (i != 0) {
            std::this_thread::sleep_for(sync_interval());
        }
        resync();
    }
}

std::chrono::nanoseconds RealtimeProfilerClockSync::resync() {
    TTZoneScopedDN(RT_PROFILER, "Resync");
    const auto started_at = std::chrono::steady_clock::now();
    resyncs_.fetch_add(1, std::memory_order_relaxed);
    mapping_.retain(best_of(kResyncProbes));
    const auto blocked_for = std::chrono::steady_clock::now() - started_at;
    busy_ns_.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(blocked_for).count(), std::memory_order_relaxed);
    return blocked_for;
}

}  // namespace tt::tt_metal
