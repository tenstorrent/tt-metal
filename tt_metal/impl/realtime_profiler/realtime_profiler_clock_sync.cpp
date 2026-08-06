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
        // Resolved once for sync-latency purposes
        const uint64_t local = clock_tlb_->handle_ref().get_config().local_offset;
        auto* base = clock_tlb_->handle_ref().get_base();
        mapped_clock_lo_ = reinterpret_cast<volatile uint32_t*>(base + (wall_clock_addr_lo_ - local));
        mapped_clock_hi_ = reinterpret_cast<volatile uint32_t*>(base + (wall_clock_addr_hi_ - local));
    } catch (const std::exception& e) {
        log_warning(
            tt::LogMetal, "[Real-time profiler] Device {}: could not map the clock register ({})", chip_id_, e.what());
    }
}

std::optional<RealtimeProfilerClockSync::BaselineRate> RealtimeProfilerClockSync::baseline_rate() const {
    const uint64_t begin = oldest_probe();
    if (probes_end_ - begin < 2) {
        return std::nullopt;
    }
    const Anchor& newest = probe_at(probes_end_ - 1);
    // Walk back to the newest probe that is still at least kRateBaseline older than the newest one. That, not the
    // ring's oldest entry, is the near end of the baseline: how far back the rate is measured has to be a property of
    // the rate, not of how much history the ring happens to be holding.
    const auto cutoff = newest.host - kRateBaseline;
    uint64_t near = probes_end_ - 1;
    while (near > begin && probe_at(near).host > cutoff) {
        --near;
    }
    const Anchor& oldest = probe_at(near);
    if (newest.ticks <= oldest.ticks || newest.host <= oldest.host) {
        return std::nullopt;
    }
    const double span_ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(newest.host - oldest.host).count());
    // A baseline this narrow is no tighter than the chord it exists to improve on, so the chord's own slope stands.
    if (newest.host - oldest.host < kRateBaseline / 4) {
        return std::nullopt;
    }
    return BaselineRate{.rate = static_cast<double>(newest.ticks - oldest.ticks) / span_ns};
}

uint64_t RealtimeProfilerClockSync::first_probe_at_or_past(uint64_t ticks) const {
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

// How far the clock departed from the chord, measured rather than inferred. Every probe strictly inside the chord is a
// point the secant was not fitted to, so its distance from the secant is the departure at that point; a probe's own
// read noise accounts for bracket/2 of any distance, so that comes off first. Reads zero on a plateau, which is nearly
// always.
//
// One interior probe locates the departure only where that probe happens to sit. A rate step at fraction L of the span
// bows the trajectory as L(1-L), so a sample away from the midpoint understates the peak -- every interior probe is
// checked and the largest kept, and with none at all this reports zero, which is the absence of evidence rather than
// the absence of a bow. The span floor is two probe gaps at the rate probes actually arrive, so there is normally one.
std::chrono::nanoseconds RealtimeProfilerClockSync::departure_from_chord(
    const Anchor& open, const Anchor& close, const Anchor& interior) {
    if (close.ticks <= open.ticks || close.host <= open.host) {
        return {};
    }
    const double span_ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(close.host - open.host).count());
    const double inv_rate = span_ns / static_cast<double>(close.ticks - open.ticks);
    const double on_chord_ns = static_cast<double>(open.host.time_since_epoch().count()) +
                               (static_cast<double>(interior.ticks) - static_cast<double>(open.ticks)) * inv_rate;
    const double measured_ns = static_cast<double>(interior.host.time_since_epoch().count());
    const auto departure = std::chrono::nanoseconds(static_cast<int64_t>(std::abs(measured_ns - on_chord_ns)));
    // Three brackets explain part of any distance before the clock does: the interior probe's own read, and the two the
    // line was drawn through, since a line through points known to +/-b/2 is itself only that well placed anywhere
    // between them. Only what survives all three is the clock's.
    const auto explained_by_reads = placement_error(interior.bracket) + interpolation_error(open, close);
    return departure - std::min(departure, explained_by_reads);
}

std::chrono::nanoseconds RealtimeProfilerClockSync::measured_bow(uint64_t close_index) const {
    // The pair placing a record is adjacent, so nothing lies inside it and the clock's departure has to be read one
    // probe out: the probe between `close_index - 2` and `close_index` was not fitted to the line through them, so its
    // distance from that line is the departure there. Scaled by the span ratio, because a rate step bows the trajectory
    // in proportion to the span it acts over and the triple spans roughly twice the pair.
    if (close_index < oldest_probe() + 2) {
        return {};
    }
    const Anchor& outer = probe_at(close_index - 2);
    const Anchor& open = probe_at(close_index - 1);
    const Anchor& close = probe_at(close_index);
    const auto departure = departure_from_chord(outer, close, open);
    if (departure == std::chrono::nanoseconds::zero() || close.host <= outer.host) {
        return {};
    }
    const auto pair_span = close.host - open.host;
    const auto triple_span = close.host - outer.host;
    return departure * pair_span.count() / triple_span.count();
}

std::optional<RealtimeProfilerClockSync::ChordMapping> RealtimeProfilerClockSync::place(uint64_t ticks) {
    const uint64_t probes_begin = oldest_probe();
    // The caller probes after reading a batch, so a record's timestamps are always behind the newest probe and this
    // clamp is a no-op. It exists so the pair is a pair whatever it is handed: past the far anchor place_on_chord
    // charges the distance.
    const uint64_t close_index = std::min(first_probe_at_or_past(ticks), probes_end_ - 1);
    if (close_index == probes_begin) {
        // Nothing retained before it, so there is no pair. After warm_up() this means the ring lapped past `ticks`, and
        // no later pass would help; publish_pages rejects the record.
        return std::nullopt;
    }

    auto chord = plan_chord_mapping(
        probe_at(close_index - 1), probe_at(close_index), baseline_rate(), measured_bow(close_index));
    if (!chord.has_value()) {
        return std::nullopt;
    }
    ++cost_.chords_placed;
    // Counts whether a third probe existed to read the departure from, not whether it read nonzero. A settled clock
    // reads zero, and that has to stay distinguishable from having had nothing to read.
    cost_.chords_with_bow_evidence += close_index >= probes_begin + 2;
    last_published_sync_error_ = chord->mapping.sync_error;
    return chord;
}

std::optional<RealtimeProfilerClockSync::ChordMapping> RealtimeProfilerClockSync::plan_chord_mapping(
    const Anchor& open,
    const Anchor& closing,
    const std::optional<BaselineRate>& baseline,
    std::chrono::nanoseconds measured_bow) {
    // Monotonicity is the whole of what a pair is checked for. There is no minimum span: a timestamp between two probes
    // cannot be further off than the worse of them however short the span, because a straight line's largest departure
    // from another is at their endpoints -- while the clock's own departure from the line grows with span. Narrower is
    // strictly better, and the slope's own noise, which does grow as span shrinks, reaches nothing: it is not what a
    // record is published with, and inside the chord it is pinned at both ends.
    if (closing.host <= open.host || closing.ticks <= open.ticks) {
        return std::nullopt;
    }
    const double span_ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(closing.host - open.host).count());
    const double rate = static_cast<double>(closing.ticks - open.ticks) / span_ns;
    const double rate_noise = static_cast<double>((open.bracket + closing.bracket).count()) / span_ns;

    // The rate published to consumers is the baseline's, not this chord's: a chord this narrow measures its slope to
    // only a few thousand ppm, and every duration a consumer computes divides by it. Placement is unaffected because
    // each record is anchored to where this chord puts it -- see place_on_chord.
    const double published_rate = baseline.has_value() && baseline->rate > 0.0 ? baseline->rate : rate;

    return ChordMapping{
        .mapping =
            experimental::ProgramRealtimeClockSync{
                .device_cycle_offset = 0,  // per record
                .sync_error = interpolation_error(open, closing) + measured_bow,
            },
        .frequency = published_rate,
        .chord_rate = rate,
        .chord_rate_noise = rate_noise,
        .open_ticks = open.ticks,
        .open_host_ns = static_cast<double>(open.host.time_since_epoch().count()),
        .inv_chord_rate = 1.0 / rate,
        .close_ticks = closing.ticks,
        .batch_through_ticks = closing.ticks,
    };
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
    // Reading the low word latches the high one, so this read is of the value that goes with `lo`, and it stays outside
    // the bracket.
    if (must_read_hi) {
        cached_clock_hi_ = *mapped_clock_hi_;
    } else if (lo < last_clock_lo_) {
        ++cached_clock_hi_;
    }
    last_clock_lo_ = lo;
    last_probe_at_ = host_after;
    ++cost_.clock_reads;
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

// Spaced so the pair spans kRateBaseline, which is the whole point: baseline_rate() is what a record's frequency comes
// from, and until it exists a record would be published at the commanded AICLK, which is off by a percent or so. Four
// probes over ~4ms replaces a 100-probe half-second fit whose result never reached a consumer -- the published rate is
// the baseline's, measured continuously, not a fit taken once at bring-up.
void RealtimeProfilerClockSync::warm_up() {
    TTZoneScopedDN(RT_PROFILER, "ClockWarmUp");
    constexpr int kWarmUpProbes = 4;
    // Warms the cold PCIe path; its bracket is not representative and it is not retained.
    (void)probe();
    for (int i = 0; i < kWarmUpProbes; i++) {
        if (i != 0) {
            std::this_thread::sleep_for(kRateBaseline / (kWarmUpProbes - 1));
        }
        resync();
    }
}

void RealtimeProfilerClockSync::resync() {
    TTZoneScopedDN(RT_PROFILER, "Resync");
    const auto started_at = std::chrono::steady_clock::now();
    ++cost_.resyncs;
    probe_history_[probes_end_ % kProbeHistoryCapacity] = best_of(kResyncProbes);
    ++probes_end_;
    cost_.busy += std::chrono::steady_clock::now() - started_at;
}

}  // namespace tt::tt_metal
