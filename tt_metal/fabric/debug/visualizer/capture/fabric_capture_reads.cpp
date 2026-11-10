// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_reads.hpp"

#include <algorithm>
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

// Makes one read with `read_bytes`, and adds it to `reads`, with its error if it threw. Returns whether it worked.
bool record_read(
    std::vector<CapturedRead>& reads,
    uint64_t address,
    uint32_t size_bytes,
    CaptureClock& clock,
    // Function to read the bytes from the device
    const std::function<std::vector<std::byte>()>& read_bytes) {
    CapturedRead read{.address = address, .size_bytes = size_bytes, .t_ns = clock.get_steady_ns()};
    try {
        read.bytes = read_bytes();
    } catch (const std::exception& error) {
        read.error = error.what();
    }
    const bool succeeded = !read.error.has_value();
    reads.push_back(std::move(read));
    return succeeded;
}

// Reads one register into `reads`. Returns whether it worked.
bool record_register_read(
    std::vector<CapturedRead>& reads,
    DeviceReader& reader,
    int32_t chip,
    CoreXY core,
    uint64_t address,
    CaptureClock& clock) {
    return record_read(reads, address, sizeof(uint32_t), clock, [&] {
        return make_register_bytes(reader.read_register(chip, core, address));
    });
}

// Reads size_bytes bytes of L1 into `reads`. Returns whether it worked.
bool record_l1_read(
    std::vector<CapturedRead>& reads,
    DeviceReader& reader,
    int32_t chip,
    CoreXY core,
    uint64_t address,
    uint32_t size_bytes,
    CaptureClock& clock) {
    return record_read(reads, address, size_bytes, clock, [&] {
        std::vector<std::byte> bytes(size_bytes);
        reader.read_l1(chip, core, address, bytes);
        return bytes;
    });
}

// Reads the core registers into `reads`, in the plan's order. Returns whether every read worked.
bool record_core_register_reads(
    std::vector<CapturedRead>& reads,
    const CapturePlan& plan,
    DeviceReader& reader,
    int32_t chip,
    CoreXY core,
    CaptureClock& clock) {
    return std::ranges::all_of(plan.core_registers, [&](const CoreRegister& core_register) {
        return record_register_read(reads, reader, chip, core, core_register.address, clock);
    });
}

// Reads each of `regions` into `reads`, in order. Returns whether every read worked.
bool record_multi_sampled_reads(
    std::vector<CapturedRead>& reads,
    const std::vector<MultiSampledRegion>& regions,
    DeviceReader& reader,
    int32_t chip,
    CoreXY core,
    CaptureClock& clock) {
    return std::ranges::all_of(regions, [&](const MultiSampledRegion& region) {
        switch (region.kind) {
            case RegionKind::L1:
                return record_l1_read(reads, reader, chip, core, region.address, region.size_bytes, clock);
            case RegionKind::REGISTER: return record_register_read(reads, reader, chip, core, region.address, clock);
        }
        TT_THROW("Fabric capture: unknown region kind {}", static_cast<int>(region.kind));
    });
}

}  // namespace

bool CapturedRound::has_failed_read() const {
    return std::ranges::any_of(reads, [](const CapturedRead& read) { return read.error.has_value(); });
}

uint64_t SteadyCaptureClock::get_steady_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

void SteadyCaptureClock::sleep_for(std::chrono::milliseconds duration) { std::this_thread::sleep_for(duration); }

CapturedRound sample_router(
    const CapturePlan& plan,
    int32_t chip,
    const RouterTarget& target,
    DeviceReader& reader,
    CaptureClock& clock,
    uint32_t round) {
    TT_FATAL(target.is_read(), "Fabric capture: {} isn't a core capture reads", target.router.path);
    const CoreXY core = target.router.virtual_core;
    CapturedRound sampled{.round = round, .t_ns = clock.get_steady_ns()};
    // A failed read ends the round: nothing after it is read.

    // 1. The core registers, such as the wall clock.
    if (!record_core_register_reads(sampled.reads, plan, reader, chip, core, clock)) {
        return sampled;
    }
    // 2. The multi-sampled arch areas, the same on every router core: the heartbeat word and the telemetry area.
    if (!record_multi_sampled_reads(sampled.reads, plan.multi_sampled_arch_areas, reader, chip, core, clock)) {
        return sampled;
    }
    // 3. The router's own multi-sampled regions.
    record_multi_sampled_reads(sampled.reads, target.router.multi_sampled_regions, reader, chip, core, clock);
    return sampled;
}

CapturedCore read_router_core(
    const CapturePlan& plan, int32_t chip, const RouterTarget& target, DeviceReader& reader, CaptureClock& clock) {
    TT_FATAL(target.is_read(), "Fabric capture: {} isn't a core capture reads", target.router.path);
    const CoreXY core = target.router.virtual_core;
    CapturedCore captured;
    // A failed read ends the full read: nothing after it is read.

    // 1. The core registers, such as the wall clock.
    if (!record_core_register_reads(captured.reads, plan, reader, chip, core, clock)) {
        return captured;
    }
    // 2. The whole L1, which holds every L1 region the manifest names.
    if (!record_l1_read(captured.reads, reader, chip, core, plan.l1_address, plan.l1_size_bytes, clock)) {
        return captured;
    }
    // 3. Each stream register the router's fields name, once.
    for (const uint64_t address : target.router.stream_register_addresses) {
        if (!record_register_read(captured.reads, reader, chip, core, address, clock)) {
            return captured;
        }
    }
    return captured;
}

std::vector<std::vector<std::vector<CapturedRound>>> collect_rounds(
    const CapturePlan& plan,
    const std::vector<ChipTarget>& targets,
    DeviceReader& reader,
    CaptureClock& clock,
    const RoundOptions& options) {
    std::vector<std::vector<std::vector<CapturedRound>>> rounds_by_router(targets.size());
    for (size_t chip_index = 0; chip_index < targets.size(); ++chip_index) {
        rounds_by_router[chip_index].resize(targets[chip_index].routers.size());
    }
    for (uint32_t round = 0; round < options.num_rounds; ++round) {
        if (round > 0) {
            clock.sleep_for(options.interval);
        }
        for (size_t chip_index = 0; chip_index < targets.size(); ++chip_index) {
            const ChipTarget& chip = targets[chip_index];
            for (size_t router_index = 0; router_index < chip.routers.size(); ++router_index) {
                std::vector<CapturedRound>& rounds = rounds_by_router[chip_index][router_index];
                const bool stopped = !rounds.empty() && rounds.back().has_failed_read();
                if (!chip.routers[router_index].is_read() || stopped) {
                    continue;
                }
                rounds.push_back(sample_router(plan, *chip.chip, chip.routers[router_index], reader, clock, round));
            }
        }
    }
    return rounds_by_router;
}

}  // namespace tt::tt_fabric::capture
