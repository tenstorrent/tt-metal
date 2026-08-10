// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/device_clock_sync.hpp"

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
#include <umd/device/driver_atomics.hpp>
#include <umd/device/pcie/tlb_handle.hpp>
#include <umd/device/pcie/tlb_window.hpp>
#include <umd/device/types/tlb.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

namespace tt::tt_metal {

constexpr auto kMaxProbeGapBeforeRereadingHi = std::chrono::seconds(1);

void DeviceClockSync::Mapping::add_probe(const Anchor& probe) {
    const uint64_t close_index = probes_end_;
    const size_t slot = close_index & (kProbeHistoryCapacity - 1);
    probe_history_[slot] = probe;
    ++probes_end_;

    if (probes_end_ - oldest_probe() < 2) {
        return;
    }
    const Anchor& open = probe_at(close_index - 1);
    const Anchor& closing = probe;
    const double span_ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(closing.host_timestamp - open.host_timestamp).count());
    const auto probe_error = std::max(open.error, closing.error);
    const auto nonlinearity = chord_nonlinearity(close_index);
    chords_[slot] = Chord{
        .error = probe_error + nonlinearity,
        .probe_error = probe_error,
        .nonlinearity = nonlinearity,
        .frequency = static_cast<double>(closing.device_timestamp - open.device_timestamp) / span_ns,
        .open_device_timestamp = open.device_timestamp,
        .open_host_ns = static_cast<double>(open.host_timestamp.time_since_epoch().count()),
        .close_device_timestamp = closing.device_timestamp,
    };
}

uint64_t DeviceClockSync::Mapping::first_probe_at_or_past(uint64_t device_timestamp) const {
    const uint64_t begin = oldest_probe();
    const uint64_t end = probes_end_;

    uint64_t i = last_probe_index_;
    if (i < begin || i >= end) {
        i = begin;
    }
    if (probe_at(i).device_timestamp < device_timestamp) {
        while (i < end && probe_at(i).device_timestamp < device_timestamp) {
            ++i;
        }
    } else {
        while (i > begin && probe_at(i - 1).device_timestamp >= device_timestamp) {
            --i;
        }
    }
    last_probe_index_ = i < end ? i : end - 1;
    return i;
}

std::chrono::nanoseconds DeviceClockSync::Mapping::chord_nonlinearity(uint64_t close_index) const {
    const uint64_t begin = oldest_probe();
    if (close_index < begin + 2) {
        return {};
    }
    const uint64_t open_index = close_index - 1;
    const Anchor& open = probe_at(open_index);
    const Anchor& close = probe_at(close_index);
    const auto chord_span = close.host_timestamp - open.host_timestamp;
    const auto min_outer_lead =
        std::min(chord_span, std::chrono::duration_cast<std::chrono::steady_clock::duration>(kDeviceClockSyncInterval));
    const auto outer_deadline = open.host_timestamp - min_outer_lead;
    spaced_outer_index_ = std::max(spaced_outer_index_, begin);
    while (spaced_outer_index_ + 1 < open_index && probe_at(spaced_outer_index_ + 1).host_timestamp <= outer_deadline) {
        ++spaced_outer_index_;
    }
    const Anchor& outer = probe_at(spaced_outer_index_);
    const double span_ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(close.host_timestamp - outer.host_timestamp).count());
    const double ns_per_cycle = span_ns / static_cast<double>(close.device_timestamp - outer.device_timestamp);
    const double on_chord_ns =
        static_cast<double>(outer.host_timestamp.time_since_epoch().count()) +
        (static_cast<double>(open.device_timestamp) - static_cast<double>(outer.device_timestamp)) * ns_per_cycle;
    const double measured_ns = static_cast<double>(open.host_timestamp.time_since_epoch().count());
    const auto departure = std::chrono::nanoseconds(static_cast<int64_t>(std::abs(measured_ns - on_chord_ns)));
    const auto explained_by_reads = open.error + std::max(outer.error, close.error);
    return std::max(departure - explained_by_reads, std::chrono::nanoseconds::zero());
}

std::optional<DeviceClockSync::Mapping::Chord> DeviceClockSync::Mapping::chord_around(uint64_t device_timestamp) const {
    const uint64_t begin = oldest_probe();
    if (probes_end_ - begin < 2) {
        return std::nullopt;
    }
    const uint64_t close_index = std::min(first_probe_at_or_past(device_timestamp), probes_end_ - 1);
    if (close_index == begin) {
        return std::nullopt;
    }
    return chord_at(close_index);
}

void DeviceClockSync::Mapping::pin_start(uint64_t device_timestamp) {
    if (device_timestamp == 0 || device_timestamp == last_pin_device_timestamp_) {
        return;
    }
    last_pin_device_timestamp_ = device_timestamp;
    const std::optional<Chord> chord = chord_around(device_timestamp);
    if (!chord.has_value()) {
        return;
    }
    pinned_start_ = Anchor{
        .host_timestamp = std::chrono::steady_clock::time_point(
            std::chrono::nanoseconds(std::llround(host_ns_on(*chord, device_timestamp)))),
        .device_timestamp = device_timestamp,
        .error = chord->error,
        .probe_error = chord->probe_error,
        .nonlinearity = chord->nonlinearity,
    };
}

std::optional<DeviceClockSync::RecordMapping> DeviceClockSync::Mapping::map_record(
    uint64_t start_device_timestamp, uint64_t end_device_timestamp) {
    if (!active_chord_.has_value() || start_device_timestamp > active_chord_->close_device_timestamp) {
        active_chord_ = chord_around(start_device_timestamp);
    }

    // Prefer the worse chord's claim split so probe_error/nonlinearity sum to the published error.
    const auto claim_from = [](const Chord& chord) {
        return RecordMapping{
            .error = chord.error,
            .probe_error = chord.probe_error,
            .nonlinearity = chord.nonlinearity,
        };
    };
    const auto worse_claim = [&](const Chord& a, const Chord& b) { return claim_from(a.error >= b.error ? a : b); };

    RecordMapping mapping;
    if (!active_chord_.has_value()) {
        // Start is older than our probe history.
        const std::optional<Chord> end_chord = chord_around(end_device_timestamp);
        if (!end_chord.has_value()) {
            return std::nullopt;
        }
        const double end_host_ns = host_ns_on(*end_chord, end_device_timestamp);
        if (pinned_start_.has_value() && pinned_start_->device_timestamp == start_device_timestamp) {
            // Long program: reuse the host time we pinned while it was still running.
            const double start_host_ns = static_cast<double>(pinned_start_->host_timestamp.time_since_epoch().count());
            const double frequency =
                static_cast<double>(end_device_timestamp - start_device_timestamp) / (end_host_ns - start_host_ns);
            if (pinned_start_->error >= end_chord->error) {
                mapping = RecordMapping{
                    .device_cycle_offset =
                        std::llround(static_cast<double>(start_device_timestamp) - frequency * start_host_ns),
                    .error = pinned_start_->error,
                    .probe_error = pinned_start_->probe_error,
                    .nonlinearity = pinned_start_->nonlinearity,
                    .frequency = frequency};
            } else {
                mapping = RecordMapping{
                    .device_cycle_offset =
                        std::llround(static_cast<double>(start_device_timestamp) - frequency * start_host_ns),
                    .error = end_chord->error,
                    .probe_error = end_chord->probe_error,
                    .nonlinearity = end_chord->nonlinearity,
                    .frequency = frequency};
            }
        } else {
            // No pinned start: estimate frequency from the whole probe ring.
            const Anchor& ring_oldest = probe_at(oldest_probe());
            const Anchor& ring_newest = probe_at(probes_end_ - 1);
            const double ring_span_ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                ring_newest.host_timestamp - ring_oldest.host_timestamp)
                                                                .count());
            const double frequency =
                static_cast<double>(ring_newest.device_timestamp - ring_oldest.device_timestamp) / ring_span_ns;
            const double ride_ns = static_cast<double>(end_device_timestamp - start_device_timestamp) / frequency;
            const auto ride_noise = std::chrono::nanoseconds(std::llround(
                ride_ns * static_cast<double>((ring_oldest.error + ring_newest.error).count()) / ring_span_ns));
            mapping = RecordMapping{
                .device_cycle_offset =
                    std::llround(static_cast<double>(end_device_timestamp) - frequency * end_host_ns),
                .error = end_chord->error + ride_noise,
                .probe_error = end_chord->probe_error + ride_noise,
                .nonlinearity = end_chord->nonlinearity,
                .frequency = frequency};
        }
    } else if (const Chord& chord = *active_chord_; end_device_timestamp <= chord.close_device_timestamp) {
        // Usual case: whole record fits between two adjacent probes.
        const double start_host_ns = host_ns_on(chord, start_device_timestamp);
        mapping = RecordMapping{
            .device_cycle_offset =
                std::llround(static_cast<double>(start_device_timestamp) - chord.frequency * start_host_ns),
            .error = chord.error,
            .probe_error = chord.probe_error,
            .nonlinearity = chord.nonlinearity,
            .frequency = chord.frequency};
    } else {
        // Record spans more than one probe gap.
        const std::optional<Chord> end_chord = chord_around(end_device_timestamp);
        TT_ASSERT(end_chord.has_value());
        const double start_host_ns = host_ns_on(chord, start_device_timestamp);
        const double end_host_ns = host_ns_on(*end_chord, end_device_timestamp);
        const double frequency =
            static_cast<double>(end_device_timestamp - start_device_timestamp) / (end_host_ns - start_host_ns);
        const RecordMapping claim = worse_claim(chord, *end_chord);
        mapping = RecordMapping{
            .device_cycle_offset =
                std::llround(static_cast<double>(start_device_timestamp) - frequency * start_host_ns),
            .error = claim.error,
            .probe_error = claim.probe_error,
            .nonlinearity = claim.nonlinearity,
            .frequency = frequency};
    }
    if (pinned_start_.has_value() && start_device_timestamp >= pinned_start_->device_timestamp) {
        pinned_start_.reset();
    }
    return mapping;
}

DeviceClockSync::DeviceClockSync(ContextId context_id, IDevice* device, CoreCoord clock_core) :
    context_id_(context_id),
    chip_id_(device->id()),
    clock_core_virtual_(device->virtual_core_from_logical_core(clock_core, CoreType::WORKER)) {
    TTZoneScopedDN(RT_PROFILER, "ClockSyncConfigure");
    const auto& hal = MetalContext::instance(context_id_).hal();
    wall_clock_addr_lo_ = hal.get_tensix_wall_clock_reg_addr_lo();
    wall_clock_addr_hi_ = hal.get_tensix_wall_clock_reg_addr_hi();
    configure_clock_read_path();
    if (mapped_clock_lo_ != nullptr) {
        // Throwaway cold read, then spaced probes so map_record already has a chord (and a third for nonlinearity).
        constexpr int kWarmUpProbes = 4;
        (void)probe();
        for (int i = 0; i < kWarmUpProbes; i++) {
            if (i != 0) {
                std::this_thread::sleep_for(kDeviceClockSyncInterval);
            }
            resync();
        }
    }
}

DeviceClockSync::~DeviceClockSync() = default;

void DeviceClockSync::configure_clock_read_path() {
    try {
        auto* tlb_manager =
            MetalContext::instance(context_id_).get_cluster().get_driver()->get_chip(chip_id_)->get_tlb_manager();
        if (tlb_manager == nullptr) {
            log_warning(
                tt::LogMetal,
                "[DeviceClockSync] Device {}: no TLB manager, so the clock register cannot be mapped",
                chip_id_);
            return;
        }
        tt::umd::tlb_data cfg{};
        cfg.local_offset = wall_clock_addr_lo_;
        cfg.x_end = clock_core_virtual_.x;
        cfg.y_end = clock_core_virtual_.y;
        cfg.ordering = tt::umd::tlb_data::Strict;
        clock_tlb_ = tlb_manager->allocate_tlb_window(cfg, tt::umd::TlbMapping::UC);
        if (clock_tlb_ == nullptr) {
            log_warning(
                tt::LogMetal,
                "[DeviceClockSync] Device {}: no UC TLB window available for the clock register",
                chip_id_);
            return;
        }
        const uint64_t local = clock_tlb_->handle_ref().get_config().local_offset;
        auto* base = clock_tlb_->handle_ref().get_base();
        mapped_clock_lo_ = reinterpret_cast<volatile uint32_t*>(base + (wall_clock_addr_lo_ - local));
        mapped_clock_hi_ = reinterpret_cast<volatile uint32_t*>(base + (wall_clock_addr_hi_ - local));
    } catch (const std::exception& e) {
        log_warning(
            tt::LogMetal, "[DeviceClockSync] Device {}: could not map the clock register ({})", chip_id_, e.what());
    }
}

DeviceClockSync::Anchor DeviceClockSync::probe() {
    TTZoneScopedDN(RT_PROFILER, "Probe");
    const bool must_read_hi = last_probe_at_ == std::chrono::steady_clock::time_point{} ||
                              std::chrono::steady_clock::now() - last_probe_at_ > kMaxProbeGapBeforeRereadingHi;

    std::chrono::steady_clock::time_point host_before;
    std::chrono::steady_clock::time_point host_after;
    uint32_t lo = 0;

    {  // latency-critical
        host_before = std::chrono::steady_clock::now();
        tt_driver_atomics::lfence();
        lo = *mapped_clock_lo_;
        tt_driver_atomics::lfence();
        host_after = std::chrono::steady_clock::now();
    }

    if (must_read_hi) {
        cached_clock_hi_ = *mapped_clock_hi_;
    } else if (lo < last_clock_lo_) {
        ++cached_clock_hi_;
    }
    last_clock_lo_ = lo;
    last_probe_at_ = host_after;
    const auto bracket = host_after - host_before;
    TTZoneValueD(RT_PROFILER, static_cast<uint64_t>(bracket.count()));
    const auto bracket_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(bracket);
    const auto error = bracket_ns / 2;
    return Anchor{
        .host_timestamp = host_before + error,
        .device_timestamp = (static_cast<uint64_t>(cached_clock_hi_) << 32) | lo,
        .error = error};
}

void DeviceClockSync::resync() {
    TTZoneScopedDN(RT_PROFILER, "Resync");
    Anchor best = probe();
    for (int i = 1; i < kResyncProbes; i++) {
        if (typical_error_ > std::chrono::nanoseconds::zero() && best.error <= typical_error_ + typical_error_ / 2) {
            break;
        }
        const Anchor p = probe();
        if (p.error < best.error) {
            best = p;
        }
    }
    typical_error_ =
        typical_error_ == std::chrono::nanoseconds::zero() ? best.error : (typical_error_ * 7 + best.error) / 8;
    mapping_.add_probe(best);
}

}  // namespace tt::tt_metal
