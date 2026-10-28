// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the MM-8 data-movement tests on Mimir CCEs (test_mimir_dm_*.cpp).

#pragma once

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"

namespace tt::tt_metal::mimir_test {

// Rdcycle cycles per kernel wait or poll. The emulated clock runs at about 85,000 cycles/s while the host polls, so
// 2^22 is about kPollBudgetHostTime.
inline constexpr uint64_t kPollBudgetCycles = 1ull << 22;
inline constexpr std::chrono::seconds kPollBudgetHostTime{50};
inline constexpr std::chrono::milliseconds kRecheckDelay{500};
inline constexpr const char* kFreshSessionNote =
    ". The kernel may still be running; start a fresh emulator session before the next run (MIM-601).";

// The descriptor this run is configured with, e.g. "mimir_2x_package", with every character that is not a letter or
// digit replaced by '_'.
inline std::string configured_descriptor_name() {
    const char* descriptor_path = mimir_contract::selected_soc_desc_path();
    if (descriptor_path == nullptr) {
        return "Unconfigured";
    }
    std::string name = std::filesystem::path(descriptor_path).stem().string();
    for (char& character : name) {
        if (std::isalnum(static_cast<unsigned char>(character)) == 0) {
            character = '_';
        }
    }
    return name.empty() ? "Unconfigured" : name;
}

inline void record_descriptor_properties(const metal_SocDescriptor& soc) {
    const MimirTopology topology = topology_from(soc);
    ::testing::Test::RecordProperty("descriptor", configured_descriptor_name());
    ::testing::Test::RecordProperty(
        "topology", fmt::format("{} Mimirs, {} CCEs", topology.num_mimirs, topology.num_cces));
}

// One plan section 2.3 field of a leg, as the gtest property `<leg>_<field>`.
inline void record_leg(const std::string& leg, const std::string& field, const std::string& value) {
    ::testing::Test::RecordProperty(leg + "_" + field, value);
}

// A kernel writes its status line once at exit at tl1_done_block(hart) of its CCE, with the done magic in word 0.
inline uint32_t read_status_magic(IDevice* device, uint32_t cce_index, uint32_t hart) {
    return read_cce_tl1(device, cce_index, tl1_done_block(hart), sizeof(uint32_t)).front();
}

// Reads the status magic of every launch with no sleep, so the emulated clock keeps running, until launches[i] shows
// done_magics[i] for every i or `host_deadline` passes.
inline bool wait_for_status_magics(
    IDevice* device,
    const std::vector<CceKernelLaunch>& launches,
    const std::vector<uint32_t>& done_magics,
    std::chrono::seconds host_deadline) {
    TT_FATAL(launches.size() == done_magics.size(), "{} launches, {} done magics", launches.size(), done_magics.size());
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + host_deadline;
    std::vector<bool> done(launches.size(), false);
    while (std::chrono::steady_clock::now() < deadline) {
        for (size_t index = 0; index < launches.size(); ++index) {
            done[index] = done[index] || read_status_magic(device, launches[index].cce_index, launches[index].hart) ==
                                             done_magics[index];
        }
        if (std::all_of(done.begin(), done.end(), [](bool line_done) { return line_done; })) {
            return true;
        }
    }
    return false;
}

// Launches `program`, which already holds the kernels of `launches`, with launch_on_idle_cces, waits for every status
// magic, ends the Program and returns each status line (`status_words` words) in launch order. nullopt when
// `host_deadline` passed: a kernel may still be running, and nothing else may run in this session (MIM-601).
inline std::optional<std::vector<std::vector<uint32_t>>> launch_and_wait_for_statuses(
    IDevice* device,
    Program& program,
    const std::vector<CceKernelLaunch>& launches,
    const std::vector<uint32_t>& done_magics,
    uint32_t status_words,
    std::chrono::seconds host_deadline) {
    launch_on_idle_cces(device, program, launches);
    if (!wait_for_status_magics(device, launches, done_magics, host_deadline)) {
        return std::nullopt;
    }
    detail::WaitProgramDone(device, program, /*read_device_profiler_results=*/false);
    std::vector<std::vector<uint32_t>> statuses;
    for (const CceKernelLaunch& launch : launches) {
        statuses.push_back(
            read_cce_tl1(device, launch.cce_index, tl1_done_block(launch.hart), status_words * sizeof(uint32_t)));
    }
    return statuses;
}

// Adds every kernel to `program`, then runs launch_and_wait_for_statuses.
inline std::optional<std::vector<std::vector<uint32_t>>> run_cce_kernels_with_deadline(
    IDevice* device,
    Program& program,
    const std::vector<CceKernelLaunch>& launches,
    const std::vector<uint32_t>& done_magics,
    uint32_t status_words,
    std::chrono::seconds host_deadline) {
    add_cce_kernels(device, program, launches);
    return launch_and_wait_for_statuses(device, program, launches, done_magics, status_words, host_deadline);
}

// Reads the data with read_words(byte_offset, bytes) and compares it with `expected`. On a mismatch, records the first
// one as `<leg>_first_mismatch`, waits kRecheckDelay and re-reads that word; if it changed, re-reads all the data
// before comparing.
inline ::testing::AssertionResult compare_with_recheck(
    const std::string& label,
    const std::string& leg,
    const std::vector<uint32_t>& expected,
    const std::function<std::vector<uint32_t>(uint64_t, uint32_t)>& read_words) {
    const uint32_t num_bytes = static_cast<uint32_t>(expected.size() * sizeof(uint32_t));
    std::vector<uint32_t> actual = read_words(0, num_bytes);
    const size_t mismatch =
        static_cast<size_t>(std::mismatch(expected.begin(), expected.end(), actual.begin()).first - expected.begin());
    if (mismatch < expected.size()) {
        record_leg(
            leg,
            "first_mismatch",
            fmt::format("word {} expected 0x{:08x} actual 0x{:08x}", mismatch, expected[mismatch], actual[mismatch]));
        std::this_thread::sleep_for(kRecheckDelay);
        const uint32_t reread = read_words(mismatch * sizeof(uint32_t), sizeof(uint32_t)).front();
        log_info(
            tt::LogTest,
            "MM8 {}: word {} read 0x{:08x}, then 0x{:08x} after {} ms",
            label,
            mismatch,
            actual[mismatch],
            reread,
            kRecheckDelay.count());
        if (reread != actual[mismatch]) {
            actual = read_words(0, num_bytes);
        }
    }
    return compare_words(label, expected, actual);
}

}  // namespace tt::tt_metal::mimir_test
