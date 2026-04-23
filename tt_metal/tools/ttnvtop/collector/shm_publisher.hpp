// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../common/shm_schema.hpp"

namespace ttnvtop {

// Owns one /dev/shm/tt_device_<asic_id>_util file, mapped read-write.
// Non-copyable, moveable. Unlinks on destruction so stale files don't accumulate.
class ShmPublisher {
public:
    ShmPublisher() = default;
    ShmPublisher(const ShmPublisher&) = delete;
    ShmPublisher& operator=(const ShmPublisher&) = delete;
    ShmPublisher(ShmPublisher&& other) noexcept;
    ShmPublisher& operator=(ShmPublisher&& other) noexcept;
    ~ShmPublisher();

    // Create (or replace) the SHM file for this chip and populate its header.
    // Returns false on failure (errno preserved for caller logging).
    bool open(uint64_t asic_id, uint32_t arch_id, uint32_t signal_sources, uint32_t num_cores);

    bool is_open() const { return header_ != nullptr; }

    UtilShmHeader* header() { return header_; }
    PerCoreView* cores() { return cores_; }
    uint32_t num_cores() const { return num_cores_; }

    // Stamp last_update_us on the header. Call once per tick after updating cores.
    void mark_updated();

    void close();

private:
    int fd_ = -1;
    void* map_ = nullptr;
    size_t map_size_ = 0;
    UtilShmHeader* header_ = nullptr;
    PerCoreView* cores_ = nullptr;
    uint32_t num_cores_ = 0;
    std::string shm_name_;  // "/tt_device_<asic>_util" (leading slash, no /dev/shm prefix)
};

// Enumerate existing /dev/shm/tt_device_*_util files. Used by the viewer.
struct ShmFileEntry {
    std::string path;  // full path including /dev/shm/
    uint64_t asic_id;
};
std::vector<ShmFileEntry> list_shm_files();

}  // namespace ttnvtop
