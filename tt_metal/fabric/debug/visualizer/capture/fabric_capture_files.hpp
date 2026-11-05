// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_reads.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_record.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_targets.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_utils.hpp"

namespace tt::tt_fabric::capture {

// Where one capture's two files go: its record and its .bin, side by side.
struct CaptureFilePaths {
    std::filesystem::path record_path;
    std::filesystem::path bin_path;
};

// The files for one rank's capture in `directory`, named by visualizer::make_capture_file_stem. Fails if either is
// already there: a capture never replaces another, so another capture of the same manifest needs another capture_id.
CaptureFilePaths make_capture_file_paths(
    const std::filesystem::path& directory, uint32_t mpi_rank, uint32_t world_size, std::string_view capture_id);

// The .bin a CaptureBinWriter finished: its size and hash, which the record names it by.
struct FinishedBin {
    uint64_t size_bytes = 0;
    std::string blake3;
};

// Writes a capture's .bin one router core at a time. A core's bytes go to disk as soon as it's read, and only where
// they went is kept, so memory holds one core's L1 at most. The bytes go to a temporary file that finish() moves to the
// .bin's name, so a .bin under that name is always a complete one. A writer that isn't finished removes its temporary
// file.
class CaptureBinWriter {
public:
    explicit CaptureBinWriter(std::filesystem::path bin_path);
    ~CaptureBinWriter();
    CaptureBinWriter(const CaptureBinWriter&) = delete;
    CaptureBinWriter& operator=(const CaptureBinWriter&) = delete;

    // Appends the bytes of each of a router core's reads that worked, in read order, and returns the router's record:
    // its target, and each read with where its bytes start in the .bin. A read that failed has no offset.
    RouterRecord append_core(const RouterTarget& target, const CapturedCore& core);

    // Closes the .bin and moves it to its name, failing rather than replacing a file already there.
    FinishedBin finish();

private:
    std::filesystem::path bin_path_;
    std::filesystem::path temporary_path_;
    std::ofstream out_;
    Blake3Hasher hasher_;
    uint64_t size_bytes_ = 0;
    bool finished_ = false;
};

// Writes the record to `record_path` the same way: to a temporary file, then moved to its name without replacing a file
// there. Write it after the .bin is finished, so a record on disk means its .bin is complete.
void write_capture_record(const std::filesystem::path& record_path, const CaptureRecord& record);

}  // namespace tt::tt_fabric::capture
