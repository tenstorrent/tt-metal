// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_device.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_files.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_reads.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_record.hpp"

namespace tt::tt_fabric::capture {

// One rank's capture: which manifest to base a capture on, and which capture of its run this is (a capture can be taken
// multiple times).
struct CaptureRequest {
    // This rank's manifest, in its run's folder. The capture's files go in that folder's captures/<capture_id>/.
    std::filesystem::path manifest_path;
    // Every rank captured together has the same one.
    std::string capture_id;
    LivenessOptions liveness_options;
    // The commit capture was built from, which the record keeps beside the manifest's.
    std::string capture_git_sha;
};

// How a capture that wrote its files went: whether everything the manifest names on this rank was read.
enum class CaptureOutcome {
    // Every chip was found, and every router's core read in full, with every liveness sample read.
    COMPLETE,
    // The files were written, and the record says what's missing.
    PROBLEMS_RECORDED,
};

// What capture_rank wrote, and how it went.
struct CaptureResult {
    CaptureFilePaths paths;
    CaptureOutcome outcome = CaptureOutcome::COMPLETE;
};

// Captures one rank: reads its manifest, matches the manifest's chips and routers to what the reader found, and reads
// router state for each core.
CaptureResult capture_rank(
    const CaptureRequest& request, DeviceReader& reader, const ClusterFacts& facts, CaptureClock& clock);

}  // namespace tt::tt_fabric::capture
