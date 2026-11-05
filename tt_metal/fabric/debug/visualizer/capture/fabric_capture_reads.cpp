// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_reads.hpp"

#include <array>
#include <bit>
#include <exception>
#include <functional>
#include <thread>
#include <utility>

#include <tt_stl/assert.hpp>

namespace tt::tt_fabric::capture {

namespace {

std::vector<std::byte> make_register_bytes(uint32_t value) {
    const auto bytes = std::bit_cast<std::array<std::byte, sizeof(uint32_t)>>(value);
    return {bytes.begin(), bytes.end()};
}

// The registers read on each side of a core's L1 read: the core registers, in the plan's order, then the router's
// stream registers.
std::vector<uint64_t> collect_register_addresses(const CapturePlan& plan, const CaptureTarget& target) {
    std::vector<uint64_t> addresses;
    addresses.reserve(plan.core_registers.size() + target.router.stream_register_addresses.size());
    for (const CoreRegister& core_register : plan.core_registers) {
        addresses.push_back(core_register.address);
    }
    addresses.insert(
        addresses.end(),
        target.router.stream_register_addresses.begin(),
        target.router.stream_register_addresses.end());
    return addresses;
}

// Makes one read with `read_bytes`, and records it on `core`, with its error if it threw. Returns whether it worked.
bool record_read(
    CapturedCore& core,
    ReadStep step,
    uint64_t address,
    uint32_t size_bytes,
    CaptureClock& clock,
    // Function to read the bytes from the device
    const std::function<std::vector<std::byte>()>& read_bytes) {
    CapturedRead read{.step = step, .address = address, .size_bytes = size_bytes, .t_ns = clock.get_steady_ns()};
    try {
        read.bytes = read_bytes();
    } catch (const std::exception& error) {
        read.error = error.what();
    }
    const bool succeeded = !read.error.has_value();
    core.reads.push_back(std::move(read));
    return succeeded;
}

LivenessSample sample_liveness(
    const CapturePlan& plan, const CaptureTarget& target, DeviceReader& reader, CaptureClock& clock, uint32_t round) {
    LivenessSample sample{.round = round, .t_ns = clock.get_steady_ns()};
    const int32_t chip = *target.chip;
    const CoreXY core = target.router.virtual_core;
    try {
        std::array<std::byte, sizeof(uint32_t)> heartbeat{};
        reader.read_l1(chip, core, plan.heartbeat_address, heartbeat);
        sample.heartbeat = std::bit_cast<uint32_t>(heartbeat);
        for (const CoreRegister& core_register : plan.core_registers) {
            sample.core_registers.push_back(reader.read_register(chip, core, core_register.address));
        }
    } catch (const std::exception& error) {
        sample.error = error.what();
    }
    return sample;
}

}  // namespace

uint64_t SteadyCaptureClock::get_steady_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

void SteadyCaptureClock::sleep_for(std::chrono::milliseconds duration) { std::this_thread::sleep_for(duration); }

CapturedCore read_router_core(
    const CapturePlan& plan, const CaptureTarget& target, DeviceReader& reader, CaptureClock& clock) {
    TT_FATAL(
        target.is_read() && target.chip.has_value(),
        "Fabric capture: {} isn't a core capture reads",
        target.router.path);
    const int32_t chip = *target.chip;
    const CoreXY core = target.router.virtual_core;
    const std::vector<uint64_t> register_addresses = collect_register_addresses(plan, target);

    CapturedCore captured;

    // Generic lambda to read registers for a given step
    const auto read_registers = [&](ReadStep step) {
        for (const uint64_t address : register_addresses) {
            const bool succeeded = record_read(captured, step, address, sizeof(uint32_t), clock, [&] {
                return make_register_bytes(reader.read_register(chip, core, address));
            });
            if (!succeeded) {
                return false;
            }
        }
        return true;
    };

    // First register reads
    if (!read_registers(ReadStep::REGISTERS_BEFORE)) {
        return captured;
    }

    // L1 read
    const bool l1_read = record_read(captured, ReadStep::L1, plan.l1_address, plan.l1_size_bytes, clock, [&] {
        std::vector<std::byte> l1(plan.l1_size_bytes);
        reader.read_l1(chip, core, plan.l1_address, l1);
        return l1;
    });
    if (!l1_read) {
        return captured;
    }

    // Second register reads
    read_registers(ReadStep::REGISTERS_AFTER);

    return captured;
}

std::vector<std::vector<LivenessSample>> collect_liveness_rounds(
    const CapturePlan& plan,
    const std::vector<CaptureTarget>& targets,
    DeviceReader& reader,
    CaptureClock& clock,
    const LivenessOptions& options) {
    std::vector<std::vector<LivenessSample>> samples_by_target(targets.size());
    for (uint32_t round = 0; round < options.num_rounds; ++round) {
        if (round > 0) {
            clock.sleep_for(options.interval);
        }
        for (size_t i = 0; i < targets.size(); ++i) {
            std::vector<LivenessSample>& samples = samples_by_target[i];
            const bool stopped = !samples.empty() && samples.back().error.has_value();
            if (!targets[i].is_read() || stopped) {
                continue;
            }
            samples.push_back(sample_liveness(plan, targets[i], reader, clock, round));
        }
    }
    return samples_by_target;
}

}  // namespace tt::tt_fabric::capture
