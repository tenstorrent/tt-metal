// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <sys/types.h>
#include <vector>

// The two halves of ShmResourceTracker's stale scan, split so the hand-over
// between them can be exercised: deciding which manifests belong to dead
// owners, and reaping what those manifests list. Runtime-internal.

namespace tt::tt_metal::distributed {

// Contents of a manifest file as read from one open descriptor.
struct ManifestContents {
    uint64_t start_time = 0;  // "start <ticks>" line; 0 when absent (older image) or unparsable
    std::vector<std::string> shm_names;
    std::vector<std::string> file_paths;
};

// Opens `path`, records the identity of the file it opened and parses it.
// False when the file cannot be opened (typically it vanished meanwhile).
bool read_manifest(const std::string& path, dev_t& dev, ino_t& ino, ManifestContents& contents);

// A manifest judged stale, together with the entries read from the very file
// that was judged and that file's identity.
struct StaleManifest {
    std::string path;
    dev_t dev = 0;
    ino_t ino = 0;
    bool pid_reused = false;  // the pid is alive but belongs to another process
    std::vector<std::string> shm_names;
    std::vector<std::string> file_paths;
};

struct StaleScan {
    std::vector<StaleManifest> manifests;
    std::vector<std::string> orphan_shm_names;  // bare objects whose embedded pid is dead
};

// First half: which manifests under /dev/shm belong to a dead owner (or to a
// predecessor that held this process's pid), and which bare objects are orphaned.
StaleScan collect_stale_shm_resources();

// Second half: unlink the listed objects and files, then remove each manifest
// file only if it is still the file that was judged (same device and inode).
// A manifest the live holder of that pid republished in between is left alone,
// and so are the resources it lists.
void reap_stale_shm_resources(const StaleScan& scan);

}  // namespace tt::tt_metal::distributed
