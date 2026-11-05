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

// Where capture's time comes from: when each read starts, and the wait between liveness rounds.
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

// The steps of reading one router core, in order. The registers are read on both sides of the L1 read, so a register
// that differs shows the router moved while its L1 was being read.
enum class ReadStep {
    REGISTERS_BEFORE,
    L1,
    REGISTERS_AFTER,
};

// One read of a router core.
struct CapturedRead {
    ReadStep step = ReadStep::L1;
    uint64_t address = 0;
    uint32_t size_bytes = 0;
    // When the read started, on the capture's clock.
    uint64_t t_ns = 0;
    // The bytes read, little endian, as the device holds them. Empty when the read failed.
    std::vector<std::byte> bytes;
    // Why the read failed. A failed read is the core's last: nothing after it is read.
    std::optional<std::string> error;

    bool operator==(const CapturedRead&) const = default;
};

// One router core's reads, in the order they were made.
struct CapturedCore {
    std::vector<CapturedRead> reads;

    bool operator==(const CapturedCore&) const = default;
};

// Reads one router core, going through all steps in ReadStep order. Stops at the first read that fails.
CapturedCore read_router_core(
    const CapturePlan& plan, const CaptureTarget& target, DeviceReader& reader, CaptureClock& clock);

// How many liveness rounds to take, and how far apart.
struct LivenessOptions {
    uint32_t num_rounds = 3;
    std::chrono::milliseconds interval{1000};
};

// One liveness sample of a router core: its heartbeat word, and its core registers.
struct LivenessSample {
    uint32_t round = 0;
    uint64_t t_ns = 0;
    std::optional<uint32_t> heartbeat;
    // The core registers read before any failure, in the plan's order.
    std::vector<uint32_t> core_registers;
    // Why a read failed. The core isn't sampled again.
    std::optional<std::string> error;

    bool operator==(const LivenessSample&) const = default;
};

// Takes the liveness rounds: each round samples every core capture reads, a few words each, and the rounds are
// options.interval apart. These come before any core's full read, so they show which routers are still running.
// Returns each target's samples, in the targets' order.
std::vector<std::vector<LivenessSample>> collect_liveness_rounds(
    const CapturePlan& plan,
    const std::vector<CaptureTarget>& targets,
    DeviceReader& reader,
    CaptureClock& clock,
    const LivenessOptions& options);

}  // namespace tt::tt_fabric::capture
