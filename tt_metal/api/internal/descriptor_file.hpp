// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// File helpers for descriptors that one process writes and another reads (e.g. under /dev/shm).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <tt_stl/span.hpp>

namespace tt::tt_metal::internal {

// Writes `bytes` to `path + ".tmp"`, then renames it to `path`, so a reader never sees a partial file.
void write_file_atomically(const std::string& path, ttsl::Span<const uint8_t> bytes);

// Returns the contents of `path`. Fails if it can't be opened or is empty.
std::vector<uint8_t> read_file(const std::string& path);

// Waits up to `timeout_ms` for `path` to exist, then returns its contents. Fails on timeout.
std::vector<uint8_t> wait_and_read_file(const std::string& path, uint32_t timeout_ms);

}  // namespace tt::tt_metal::internal
