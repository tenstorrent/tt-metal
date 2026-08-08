// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <bit>
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

// Probe spacing floor.
inline constexpr auto kDeviceClockSyncInterval = std::chrono::microseconds(500);

// Reads a tensix free-running cycle counter over PCIe and maps device timestamps onto host time via retained probes.
class DeviceClockSync {
public:
    // A (host_timestamp, device_timestamp) sample and the uncertainty of that host time.
    struct Anchor {
        std::chrono::steady_clock::time_point host_timestamp;
        uint64_t device_timestamp = 0;
        std::chrono::nanoseconds error{};
    };

    // Affine map: host_ns = (device_timestamp - device_cycle_offset) / frequency.
    struct RecordMapping {
        int64_t device_cycle_offset = 0;
        std::chrono::nanoseconds error{};
        double frequency = 0.0;
    };

    static constexpr size_t kProbeHistoryCapacity =
        std::bit_ceil(static_cast<size_t>(std::chrono::seconds(2) / kDeviceClockSyncInterval));

    static constexpr int kResyncProbes = 4;

    DeviceClockSync(ContextId context_id, IDevice* device, CoreCoord clock_core);
    ~DeviceClockSync();

    [[nodiscard]] bool has_direct_clock_read() const { return mapped_clock_lo_ != nullptr; }

    void resync();

    [[nodiscard]] bool due_for_probe(std::chrono::steady_clock::time_point now) const {
        return now - last_probe_at_ >= kDeviceClockSyncInterval;
    }

    // Snapshot a start timestamp onto the host timeline while probes still cover it (e.g. long-running programs).
    void pin_start(uint64_t device_timestamp) { mapping_.pin_start(device_timestamp); }

    // Nullopt only when no retained probe precedes either timestamp.
    [[nodiscard]] std::optional<RecordMapping> map_record(
        uint64_t start_device_timestamp, uint64_t end_device_timestamp) {
        return mapping_.map_record(start_device_timestamp, end_device_timestamp);
    }

private:
    class Mapping {
    public:
        // Requires probe host/device times strictly after the previous retained probe.
        void add_probe(const Anchor& probe);

        void pin_start(uint64_t device_timestamp);

        [[nodiscard]] std::optional<RecordMapping> map_record(
            uint64_t start_device_timestamp, uint64_t end_device_timestamp);

    private:
        // Secant between adjacent probes.
        struct Chord {
            std::chrono::nanoseconds error{};
            double frequency = 0.0;  // device cycles / host ns
            uint64_t open_device_timestamp = 0;
            double open_host_ns = 0.0;
            uint64_t close_device_timestamp = 0;
        };

        [[nodiscard]] std::optional<Chord> chord_around(uint64_t device_timestamp) const;

        [[nodiscard]] static double host_ns_on(const Chord& chord, uint64_t device_timestamp) {
            return chord.open_host_ns +
                   (static_cast<double>(device_timestamp) - static_cast<double>(chord.open_device_timestamp)) /
                       chord.frequency;
        }

        // Leftover mid-probe departure from the outer->close secant after subtracting probe-read noise.
        // Captures a frequency step (DVFS) inside the neighborhood of this chord.
        [[nodiscard]] std::chrono::nanoseconds chord_nonlinearity(uint64_t close_index) const;

        [[nodiscard]] uint64_t first_probe_at_or_past(uint64_t device_timestamp) const;

        [[nodiscard]] uint64_t oldest_probe() const {
            return probes_end_ > kProbeHistoryCapacity ? probes_end_ - kProbeHistoryCapacity : 0;
        }
        [[nodiscard]] const Anchor& probe_at(uint64_t index) const {
            return probe_history_[index & (kProbeHistoryCapacity - 1)];
        }
        [[nodiscard]] const Chord& chord_at(uint64_t close_index) const {
            return chords_[close_index & (kProbeHistoryCapacity - 1)];
        }

        std::array<Anchor, kProbeHistoryCapacity> probe_history_{};
        std::array<Chord, kProbeHistoryCapacity> chords_{};  // chord ending at this probe index
        uint64_t probes_end_ = 0;
        mutable uint64_t last_probe_index_ = 0;

        // Latest probe still far enough before the current open for nonlinearity leverage.
        mutable uint64_t spaced_outer_index_ = 0;

        // chord_around(start) cache.
        std::optional<Chord> active_chord_;

        // From pin_start; cleared once map_record uses it.
        std::optional<Anchor> pinned_start_;
        uint64_t last_pin_device_timestamp_ = 0;
    };

    void configure_clock_read_path();

    Anchor probe();

    ContextId context_id_;
    uint32_t chip_id_ = 0;
    CoreCoord clock_core_virtual_;
    uint32_t wall_clock_addr_lo_ = 0;
    uint32_t wall_clock_addr_hi_ = 0;

    std::unique_ptr<tt::umd::TlbWindow> clock_tlb_;
    volatile uint32_t* mapped_clock_lo_ = nullptr;
    volatile uint32_t* mapped_clock_hi_ = nullptr;

    // EMA of probe errors; resync stops early against this.
    std::chrono::nanoseconds typical_error_{};

    // High word only advances on low wrap (~3.2s at 1.35GHz).
    uint32_t cached_clock_hi_ = 0;
    uint32_t last_clock_lo_ = 0;
    std::chrono::steady_clock::time_point last_probe_at_;

    Mapping mapping_;
};

}  // namespace tt::tt_metal
