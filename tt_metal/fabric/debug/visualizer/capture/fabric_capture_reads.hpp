// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_device.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_targets.hpp"

namespace tt::tt_fabric::capture {

// Where capture's time comes from: when each read starts, and the wait between rounds.
class CaptureClock {
public:
    virtual ~CaptureClock() = default;

    // Nanoseconds on a steady clock. Only the differences between them mean anything.
    virtual uint64_t get_steady_ns() = 0;
    virtual void sleep_for(std::chrono::milliseconds duration) = 0;
};

// The process's steady clock, and a real sleep.
class SteadyCaptureClock final : public CaptureClock {
public:
    uint64_t get_steady_ns() override;
    void sleep_for(std::chrono::milliseconds duration) override;
};

// One read of a router core: size_bytes bytes at address, in L1 or a register. Which pass it belongs to is where it
// sits: in a round, or in the core's full read.
struct CapturedRead {
    uint64_t address = 0;
    uint32_t size_bytes = 0;
    // When the read started, on the capture's clock.
    uint64_t t_ns = 0;
    // The bytes read, little endian, as the device holds them. Empty when the read failed.
    std::vector<std::byte> bytes;
    // Why the read failed. A failed read is the last of its round, or of its core's full read.
    std::optional<std::string> error;

    bool operator==(const CapturedRead&) const = default;
};

// One round of a router core: the core registers, then each multi-sampled region, the arch areas' and then the
// router's.
struct CapturedRound {
    uint32_t round = 0;
    // When the round started, on the capture's clock.
    uint64_t t_ns = 0;
    std::vector<CapturedRead> reads;

    // Whether a read failed, which ends the round and the router's rounds.
    bool has_failed_read() const;

    bool operator==(const CapturedRound&) const = default;
};

// A router core's full read: the core registers, the whole L1, then each stream register once.
struct CapturedCore {
    std::vector<CapturedRead> reads;

    bool operator==(const CapturedCore&) const = default;
};

// Reads one round of a router core. Stops at the first read that fails. The router must be one capture reads, and
// `chip` the reader's id for its chip.
CapturedRound sample_router(
    const CapturePlan& plan,
    int32_t chip,
    const RouterTarget& target,
    DeviceReader& reader,
    CaptureClock& clock,
    uint32_t round);

// Reads a router core in full. Stops at the first read that fails. The router must be one capture reads, and `chip`
// the reader's id for its chip.
CapturedCore read_router_core(
    const CapturePlan& plan, int32_t chip, const RouterTarget& target, DeviceReader& reader, CaptureClock& clock);

// How many rounds to take, and how far apart. Zero rounds is the full read alone.
struct RoundOptions {
    uint32_t num_rounds = 3;
    std::chrono::milliseconds interval{1000};

    bool operator==(const RoundOptions&) const = default;
};

// Takes the rounds: each round samples every router core capture reads (sample_router), and the rounds are
// options.interval apart. They come before any core's full read, so they show what was running when capture started. A
// router whose round failed isn't sampled again. Returns each router's rounds, indexed by chip and then router, as the
// targets are. A router capture doesn't read has none.
std::vector<std::vector<std::vector<CapturedRound>>> collect_rounds(
    const CapturePlan& plan,
    const std::vector<ChipTarget>& targets,
    DeviceReader& reader,
    CaptureClock& clock,
    const RoundOptions& options);

}  // namespace tt::tt_fabric::capture
