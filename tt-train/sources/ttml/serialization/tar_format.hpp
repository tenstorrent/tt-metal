// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace ttml::serialization::tar_format {

// Tar format constants based on POSIX.1-1988 (ustar) format
// Block size and offsets/sizes from POSIX tar specification
inline constexpr size_t BLOCK_SIZE = 512;
inline constexpr size_t END_BLOCKS = 2;  // Two empty blocks mark end of archive


// Tar header field offsets and sizes (in bytes)
// Note: tar.h documents these offsets in comments but does not define them as constants.
// Note: tar.h documents these offsets in comments but does not define them as constants.
// We define them here based on the POSIX tar.h specification documentation.
namespace header {
// Field offsets (documented in tar.h comments, but not defined as constants)
constexpr size_t FILENAME_OFFSET = 0;
constexpr size_t MODE_OFFSET = 100;
constexpr size_t UID_OFFSET = 108;
constexpr size_t GID_OFFSET = 116;
constexpr size_t SIZE_OFFSET = 124;
constexpr size_t MTIME_OFFSET = 136;
constexpr size_t CHECKSUM_OFFSET = 148;
constexpr size_t TYPE_FLAG_OFFSET = 156;
constexpr size_t LINKNAME_OFFSET = 157;
constexpr size_t MAGIC_OFFSET = 257;
constexpr size_t VERSION_OFFSET = 263;
constexpr size_t UNAME_OFFSET = 265;
constexpr size_t GNAME_OFFSET = 297;
constexpr size_t DEVMAJOR_OFFSET = 329;
constexpr size_t DEVMINOR_OFFSET = 337;
constexpr size_t PREFIX_OFFSET = 345;

// Field sizes (documented in tar.h comments, but not defined as constants)
constexpr size_t FILENAME_SIZE = 100;
constexpr size_t MODE_SIZE = 8;
constexpr size_t UID_SIZE = 8;
constexpr size_t GID_SIZE = 8;
constexpr size_t SIZE_SIZE = 12;
constexpr size_t MTIME_SIZE = 12;
constexpr size_t CHECKSUM_SIZE = 8;
constexpr size_t TYPE_FLAG_SIZE = 1;
constexpr size_t LINKNAME_SIZE = 100;
constexpr size_t UNAME_SIZE = 32;
constexpr size_t GNAME_SIZE = 32;
constexpr size_t DEVMAJOR_SIZE = 8;
constexpr size_t DEVMINOR_SIZE = 8;
constexpr size_t PREFIX_SIZE = 155;

// Total header size
constexpr size_t HEADER_SIZE = BLOCK_SIZE;
}  // namespace header

// Default values (not defined in tar.h, these are application-specific)
namespace defaults {
constexpr const char* MODE = "0000644";  // rw-r--r--
constexpr const char* UID = "0000000";
constexpr const char* GID = "0000000";
constexpr uint64_t MTIME = 0;  // Use 0 for deterministic output
}  // namespace defaults

}  // namespace ttml::serialization::tar_format
