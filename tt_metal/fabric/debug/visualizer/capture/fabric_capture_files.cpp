// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_files.hpp"

#include <array>
#include <chrono>
#include <ctime>
#include <ios>
#include <string>
#include <system_error>
#include <utility>

#include <unistd.h>

#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_files.hpp"

namespace tt::tt_fabric::capture {

namespace {

// Where a file is written before it's moved to its name: beside it, so the move stays on one filesystem.
std::filesystem::path make_temporary_path(const std::filesystem::path& path) {
    return path.string() + ".tmp." + std::to_string(::getpid());
}

// Moves `from` to `to`, failing if `to` already exists. A hard link refuses an existing name, where a rename would
// replace it.
void move_without_replacing(const std::filesystem::path& from, const std::filesystem::path& to) {
    std::error_code error;
    std::filesystem::create_hard_link(from, to, error);
    TT_FATAL(!error, "Fabric capture: can't move {} to {}: {}", from.string(), to.string(), error.message());
    std::filesystem::remove(from);
}

}  // namespace

CaptureFilePaths make_capture_file_paths(
    const std::filesystem::path& run_directory, std::string_view capture_id, uint32_t mpi_rank, uint32_t world_size) {
    const std::filesystem::path directory = visualizer::make_capture_directory(run_directory, capture_id);
    const std::string stem = visualizer::make_capture_file_stem(mpi_rank, world_size);
    CaptureFilePaths paths{
        .record_path = directory / (stem + ".json"),
        .bin_path = directory / (stem + ".bin"),
    };
    for (const std::filesystem::path& path : {paths.record_path, paths.bin_path}) {
        TT_FATAL(
            !std::filesystem::exists(path),
            "Fabric capture: {} already exists, and a capture never replaces another; give this one another capture id",
            path.string());
    }
    return paths;
}

std::string make_capture_id() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&now, &utc);
    std::array<char, 32> text{};
    std::strftime(text.data(), text.size(), "%Y%m%dT%H%M%SZ", &utc);
    return text.data();
}

CaptureBinWriter::CaptureBinWriter(std::filesystem::path bin_path) :
    bin_path_(std::move(bin_path)), temporary_path_(make_temporary_path(bin_path_)) {
    std::filesystem::create_directories(bin_path_.parent_path());
    out_.open(temporary_path_, std::ios::binary | std::ios::trunc);
    TT_FATAL(out_.is_open(), "Fabric capture: can't open {} to write", temporary_path_.string());
    out_.exceptions(std::ios::badbit | std::ios::failbit);
}

CaptureBinWriter::~CaptureBinWriter() {
    if (finished_) {
        return;
    }
    // Closing mustn't throw from a destructor.
    out_.exceptions(std::ios::goodbit);
    out_.close();
    std::error_code error;
    std::filesystem::remove(temporary_path_, error);
}

RouterRecord CaptureBinWriter::append_core(const RouterTarget& target, const CapturedCore& core) {
    TT_FATAL(!finished_, "Fabric capture: {} is already finished", bin_path_.string());
    RouterRecord record{.target = target};
    record.reads.reserve(core.reads.size());
    for (const CapturedRead& read : core.reads) {
        ReadRecord read_record{
            .step = read.step,
            .address = read.address,
            .size_bytes = read.size_bytes,
            .t_ns = read.t_ns,
            .error = read.error,
        };
        if (!read.error.has_value()) {
            read_record.bin_offset = size_bytes_;
            out_.write(
                reinterpret_cast<const char*>(read.bytes.data()), static_cast<std::streamsize>(read.bytes.size()));
            hasher_.update(read.bytes);
            size_bytes_ += read.bytes.size();
        }
        record.reads.push_back(std::move(read_record));
    }
    return record;
}

FinishedBin CaptureBinWriter::finish() {
    TT_FATAL(!finished_, "Fabric capture: {} is already finished", bin_path_.string());
    out_.close();
    move_without_replacing(temporary_path_, bin_path_);
    finished_ = true;
    return FinishedBin{.size_bytes = size_bytes_, .blake3 = hasher_.get_hex()};
}

void write_capture_record(const std::filesystem::path& record_path, const CaptureRecord& record) {
    const std::filesystem::path temporary_path = make_temporary_path(record_path);
    try {
        std::ofstream out;
        out.exceptions(std::ios::badbit | std::ios::failbit);
        std::filesystem::create_directories(record_path.parent_path());
        out.open(temporary_path);
        out << make_capture_record_json(record).dump(2) << '\n';
        out.close();
        move_without_replacing(temporary_path, record_path);
    } catch (...) {
        std::error_code error;
        std::filesystem::remove(temporary_path, error);
        throw;
    }
}

}  // namespace tt::tt_fabric::capture
