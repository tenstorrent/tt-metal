// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <ctime>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_targets.hpp"

namespace tt::tt_fabric::capture {

namespace {

// The time now in UTC, as ISO 8601.
std::string get_utc_now_iso8601() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&now, &utc);
    std::array<char, 32> text{};
    std::strftime(text.data(), text.size(), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return text.data();
}

// Whether the record says anything the manifest names on this rank wasn't read in full.
bool has_recorded_problems(const CaptureRecord& record) {
    const auto has_failed_read = [](const RouterRecord& router) {
        return std::ranges::any_of(router.reads, [](const ReadRecord& read) { return read.error.has_value(); }) ||
               std::ranges::any_of(
                   router.liveness, [](const LivenessSample& sample) { return sample.error.has_value(); });
    };
    return std::ranges::any_of(record.chips, [&](const ChipRecord& chip) {
        return chip.location != ChipLocation::FOUND ||
               std::ranges::any_of(chip.routers, [&](const RouterRecord& router) {
                   return !router.target.is_read() || has_failed_read(router);
               });
    });
}

}  // namespace

CaptureResult capture_rank(
    const CaptureRequest& request, DeviceReader& reader, const ClusterFacts& facts, CaptureClock& clock) {
    const std::string started_at = get_utc_now_iso8601();

    // 1. The manifest: what to read, and which run it's from.
    CapturePlan plan;
    ManifestIdentity identity;
    {
        const ManifestFile manifest = read_manifest_file(request.manifest_path);
        plan = collect_capture_plan(manifest.json);
        identity = collect_manifest_identity(manifest);
    } // Let go of the manifest file as its large and we have taken what we need from it.
    const std::filesystem::path run_directory = request.manifest_path.parent_path();
    TT_FATAL(
        run_directory.filename() == identity.run_id,
        "Fabric capture: {} is from run {}, but isn't in that run's folder; capture writes into the run's folder, so "
        "give "
        "it the manifest where fabric init wrote it",
        request.manifest_path.string(),
        identity.run_id);

    // 2. Where each planned chip and router is on the actual cluster.
    const std::vector<ChipTarget> targets = resolve_capture_targets(plan, facts);

    // 3. The capture's files, refused before any device is read if either is already there.
    const CaptureFilePaths paths =
        make_capture_file_paths(run_directory, request.capture_id, identity.mpi_rank, identity.world_size);

    // 4. Liveness first: every router read, a few words each, a few rounds apart, before any core's full read.
    const auto liveness = collect_liveness_rounds(plan, targets, reader, clock, request.liveness_options);

    // 5. Each router in turn: read its core, write the bytes to the .bin, and keep only its record.
    CaptureBinWriter bin(paths.bin_path);
    std::vector<ChipRecord> chips;
    chips.reserve(targets.size());
    for (size_t chip_index = 0; chip_index < targets.size(); ++chip_index) {
        const ChipTarget& chip = targets[chip_index];
        ChipRecord chip_record{
            .path = chip.path, .asic_id = chip.asic_id, .chip = chip.chip, .location = chip.location};
        chip_record.routers.reserve(chip.routers.size());
        for (size_t router_index = 0; router_index < chip.routers.size(); ++router_index) {
            const RouterTarget& router = chip.routers[router_index];
            RouterRecord router_record{.target = router};
            if (router.is_read()) {
                // The core's bytes are only in memory until append_core has written them.
                const CapturedCore core = read_router_core(plan, *chip.chip, router, reader, clock);
                router_record = bin.append_core(router, core);
            }
            router_record.liveness = liveness[chip_index][router_index];
            chip_record.routers.push_back(std::move(router_record));
        }
        chips.push_back(std::move(chip_record));
    }

    // 6. The .bin is complete: move it into place, then write the record that describes it, last.
    const FinishedBin finished_bin = bin.finish();
    identity.path = identity.path.filename();
    const CaptureRecord record{
        .manifest = std::move(identity),
        .capture_id = request.capture_id,
        .capture_git_sha = request.capture_git_sha,
        .cluster_id = facts.cluster_id,
        .started_at = started_at,
        .finished_at = get_utc_now_iso8601(),
        .liveness_options = request.liveness_options,
        .bin_file = paths.bin_path.filename().string(),
        .bin_size_bytes = finished_bin.size_bytes,
        .bin_blake3 = finished_bin.blake3,
        .discovery_skipped_cores = facts.skipped_cores,
        .chips = std::move(chips),
    };
    write_capture_record(paths.record_path, record);

    return CaptureResult{
        .paths = paths,
        .outcome = has_recorded_problems(record) ? CaptureOutcome::PROBLEMS_RECORDED : CaptureOutcome::COMPLETE,
    };
}

}  // namespace tt::tt_fabric::capture
