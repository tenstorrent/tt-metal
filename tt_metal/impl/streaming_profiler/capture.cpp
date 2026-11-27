// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/capture.hpp"

#include <algorithm>
#include <filesystem>
#include <map>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>

#include <tt-logger/tt-logger.hpp>
#include <tt/insight/host/capture.hpp>
#include <tt/insight/host/capture_plan.hpp>
#include <tt_stl/indestructible.hpp>

#include "impl/streaming_profiler/service.hpp"
#include "llrt/llrt.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

struct RunningCapture {
    std::vector<ChipId> chip_ids;
    tt::insight::Capture capture;
};

// Each device's lease, from lease_device() to end_leases(), and the running captures.
struct Leases {
    std::mutex mutex;
    std::map<ChipId, tt::insight::DeviceLease> by_chip;
    // Metal's 2-ERISC mode, as the devices were lent.
    bool two_eriscs = true;
    std::vector<RunningCapture> captures;
};

Leases& leases() {
    static ttsl::Indestructible<Leases> instance;
    return instance.get();
}

// Metal reads each ELF once for the process and loads the image it makes of it on every device. Every device lends
// the same L1 with the same publishers on its tiles of one kind, so tt-insight patches one copy of the ELF, for the
// leases there are when the ELF is read.
void patch_image(const std::string& elf_path, std::span<std::byte> elf) {
    using tt::insight::Processor;
    Leases& held = leases();
    std::lock_guard lock(held.mutex);
    // Metal names an ELF after the processor it runs on. The main Ethernet program runs on ERISC0, and on ERISC1 in
    // single-ERISC mode.
    const std::map<std::string, Processor::Type> processors{
        {"brisc", Processor::BRISC},
        {"ncrisc", Processor::NCRISC},
        {"trisc0", Processor::TRISC0},
        {"trisc1", Processor::TRISC1},
        {"trisc2", Processor::TRISC2},
        {"active_erisc", held.two_eriscs ? Processor::ERISC0 : Processor::ERISC1},
        {"subordinate_active_erisc", Processor::ERISC1}};
    const auto processor = processors.find(std::filesystem::path(elf_path).stem().string());
    if (processor == processors.end()) {
        return;
    }
    // A program tt-insight refuses runs unpatched and publishes nothing.
    try {
        std::vector<tt::insight::DeviceLease> all;
        for (const auto& [chip_id, lease] : held.by_chip) {
            all.push_back(lease);
        }
        const auto images = tt::insight::patched_program_images(all, elf, processor->second);
        // Metal keeps one image for each ELF and its reader owns the bytes it read, so the one patched image is
        // copied over them.
        if (images.size() != 1) {
            throw std::runtime_error(fmt::format("the leases take {} images, and Metal loads one", images.size()));
        }
        std::ranges::copy(images.front().image, elf.begin());
    } catch (const std::exception& error) {
        log_error(tt::LogMetal, "[streaming profiler] {} publishes nothing: {}", elf_path, error.what());
    }
}

}  // namespace

void lease_device(ChipId chip_id, const tt::insight::DeviceLease::Terms& terms, bool two_eriscs) {
    tt::insight::DeviceLease lease(terms);
    Leases& held = leases();
    std::lock_guard lock(held.mutex);
    held.by_chip.insert_or_assign(chip_id, std::move(lease));
    held.two_eriscs = two_eriscs;
    llrt::set_image_patcher(patch_image);
}

bool is_lent_tile(ChipId chip_id, const CoreCoord& translated) {
    Leases& held = leases();
    std::lock_guard lock(held.mutex);
    const auto lease = held.by_chip.find(chip_id);
    if (lease == held.by_chip.end()) {
        return false;
    }
    for (const tt::insight::TranslatedCoordinate& tile : lease->second.terms().exclusive_tiles) {
        if (tile.x == translated.x && tile.y == translated.y) {
            return true;
        }
    }
    return false;
}

void start_capture(
    const std::vector<ChipId>& chip_ids, const tt::Cluster& cluster, const llrt::RunTimeOptions& rtoptions) {
    if (cluster.get_target_device_type() == tt::TargetDevice::Mock) {
        log_info(tt::LogMetal, "[streaming profiler] mock cluster: kernels are built with the profiler, nothing runs");
        return;
    }
    Leases& held = leases();
    std::vector<ChipId> leased;
    std::vector<tt::insight::DeviceLease> device_leases;
    {
        std::lock_guard lock(held.mutex);
        for (const ChipId chip_id : chip_ids) {
            if (const auto lease = held.by_chip.find(chip_id); lease != held.by_chip.end()) {
                leased.push_back(chip_id);
                device_leases.push_back(lease->second);
            }
        }
    }
    if (device_leases.empty()) {
        log_warning(tt::LogMetal, "[streaming profiler] not capturing: it needs Blackhole devices of this host");
        return;
    }
    // The profiler's own callbacks are registered before the capture starts, so that they miss nothing.
    service().attach_capture(leased, cluster, rtoptions);
    const bool sync_check = rtoptions.get_streaming_profiler_sync_check_enabled();
    if (sync_check) {
        // What the capture will run on each lent tile and what its rings and drainers keep up with.
        log_info(
            tt::LogMetal,
            "[streaming profiler] tt-insight's plan of a capture on what Metal lends:\n{}",
            tt::insight::plan_capture(device_leases, {.clock_sync_self_test = true}).report);
    }
    log_info(tt::LogMetal, "[streaming profiler] active on {} device(s)", device_leases.size());
    tt::insight::Capture capture = tt::insight::Capture::start(device_leases, {.clock_sync_self_test = sync_check});
    std::lock_guard lock(held.mutex);
    held.captures.push_back({.chip_ids = std::move(leased), .capture = std::move(capture)});
}

// tt-insight says on standard error what a capture lost and what failed in its stop.
void stop_captures(const std::vector<ChipId>& chip_ids) {
    Leases& held = leases();
    std::lock_guard lock(held.mutex);
    for (auto running = held.captures.begin(); running != held.captures.end();) {
        if (std::ranges::find_first_of(running->chip_ids, chip_ids) == running->chip_ids.end()) {
            ++running;
            continue;
        }
        try {
            running->capture.stop();
        } catch (const std::exception& error) {
            log_error(tt::LogMetal, "[streaming profiler] the capture's stop failed: {}", error.what());
        }
        service().detach_capture(running->chip_ids.size(), running->capture.clock_sync_self_test());
        running = held.captures.erase(running);
    }
}

void end_leases(const std::vector<ChipId>& chip_ids) {
    stop_captures(chip_ids);
    Leases& held = leases();
    std::lock_guard lock(held.mutex);
    for (const ChipId chip_id : chip_ids) {
        held.by_chip.erase(chip_id);
    }
    if (held.by_chip.empty()) {
        llrt::set_image_patcher(nullptr);
    }
}

}  // namespace tt::tt_metal::streaming_profiler
