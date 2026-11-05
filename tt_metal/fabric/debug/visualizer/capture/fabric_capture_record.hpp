// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_device.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_reads.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_targets.hpp"

namespace tt::tt_fabric::capture {

// One read of a router core, as the record keeps it: what was read, and where its bytes start in the .bin. The bytes
// themselves are only in the .bin.
struct ReadRecord {
    ReadStep step = ReadStep::L1;
    uint64_t address = 0;
    uint32_t size_bytes = 0;
    uint64_t t_ns = 0;
    // Where the read's size_bytes bytes start in the .bin. Null when the read failed.
    std::optional<uint64_t> bin_offset;
    std::optional<std::string> error;

    bool operator==(const ReadRecord&) const = default;
};

// One router in the record: where its core was found, its liveness samples, and its reads in the order they were made.
// A router capture doesn't read has no samples and no reads.
struct RouterRecord {
    RouterTarget target;
    std::vector<LivenessSample> liveness;
    std::vector<ReadRecord> reads;

    bool operator==(const RouterRecord&) const = default;
};

// One chip in the record: where the reader found it, and its routers.
struct ChipRecord {
    // The chip's path, such as "M0/C7".
    std::string path;
    uint64_t asic_id = 0;
    // The reader's id for the chip, when it found the chip.
    std::optional<int32_t> chip;
    ChipLocation location = ChipLocation::FOUND;
    // In the manifest's order.
    std::vector<RouterRecord> routers;

    bool operator==(const ChipRecord&) const = default;
};

// A record of a capture, written as the .json beside its .bin: what the capture was taken from, where it found each
// chip and router, what it read from each and where those bytes are in the .bin, and what failed. It's written after
// the .bin, so a record on disk means its .bin is complete.
struct CaptureRecord {
    // The manifest the capture read, including its run id.
    ManifestIdentity manifest;
    // The capture's id, which names its folder in the run's folder. Every rank captured together has the same one,
    // including those across hosts.
    std::string capture_id;
    // The commit capture was built from.
    std::string capture_git_sha;
    // The reader's cluster id, which names the host it ran on.
    std::optional<std::string> cluster_id;
    // When the capture started and finished, in UTC, as ISO 8601.
    std::string started_at;
    std::string finished_at;
    LivenessOptions liveness_options;
    // The .bin beside the record: its file name, size and hash.
    std::string bin_file;
    uint64_t bin_size_bytes = 0;
    std::string bin_blake3;
    // The Ethernet cores the reader's topology discovery skipped.
    std::vector<DiscoverySkippedCore> discovery_skipped_cores;
    // One per chip the manifest has on this rank, in the manifest's order.
    std::vector<ChipRecord> chips;

    bool operator==(const CaptureRecord&) const = default;
};

nlohmann::ordered_json make_capture_record_json(const CaptureRecord& record);

// Reads a record back. Fails on another file kind or version, or a key that's missing or of the wrong type, naming the
// key's JSON pointer.
CaptureRecord parse_capture_record(const nlohmann::ordered_json& document);

}  // namespace tt::tt_fabric::capture
