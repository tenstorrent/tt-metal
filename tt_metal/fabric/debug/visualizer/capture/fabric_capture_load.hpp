// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <span>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_record.hpp"

namespace tt::tt_fabric::capture {

// A capture read back from its files, for decode: its record, and the bytes of each of its reads, straight from the
// .bin. The .bin is mapped rather than read into memory, so a router core's bytes are only paged in when decode looks
// at them.
class LoadedCapture {
public:
    // Maps `bin_path` read-only. Fails unless it's the .bin `record` names, by size and hash, and every read's bytes
    // are inside it.
    LoadedCapture(CaptureRecord record, const std::filesystem::path& bin_path);
    ~LoadedCapture();
    LoadedCapture(LoadedCapture&& other) noexcept;
    LoadedCapture& operator=(LoadedCapture&& other) noexcept;
    LoadedCapture(const LoadedCapture&) = delete;
    LoadedCapture& operator=(const LoadedCapture&) = delete;

    const CaptureRecord& get_record() const { return record_; }

    // The bytes of one of the record's reads, size_bytes of them, valid while the capture is loaded. Fails for a read
    // that failed, which has no bytes.
    std::span<const std::byte> get_read_bytes(const ReadRecord& read) const;

private:
    CaptureRecord record_;
    // The mapped .bin. Null when the .bin is empty, as it is when capture read nothing.
    const std::byte* bin_bytes_ = nullptr;
    size_t bin_size_bytes_ = 0;
};

// Loads the capture whose record is at `record_path`, with its .bin beside it.
LoadedCapture load_capture(const std::filesystem::path& record_path);

}  // namespace tt::tt_fabric::capture
