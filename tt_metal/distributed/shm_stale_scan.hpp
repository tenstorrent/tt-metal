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

// A file by path plus the identity of the file that was at that path when judged.
struct JudgedFile {
    std::string path;
    dev_t dev = 0;
    ino_t ino = 0;
};

// A manifest judged stale, together with the entries read from the very file
// that was judged and the identity of that file and of each listed file.
// Descriptor files live at deterministic paths that a replacement owner
// republishes, so they are guarded by identity like the manifest; shm object
// names carry a per-process random part and cannot be reproduced.
struct StaleManifest {
    JudgedFile manifest;
    bool pid_reused = false;  // the pid is alive in another process (or is ours, handed on from the predecessor)
    std::vector<std::string> shm_names;
    std::vector<JudgedFile> files;
};

struct StaleScan {
    std::vector<StaleManifest> manifests;
    std::vector<std::string> orphan_shm_names;  // bare objects whose embedded pid is dead
};

// First half: which manifests under /dev/shm belong to a dead owner (or to a
// predecessor that held this process's pid), and which bare objects are orphaned.
StaleScan collect_stale_shm_resources();

// Second half. For each judged manifest: if the manifest file is gone or is no
// longer the file that was judged (device and inode differ), skip it entirely,
// since every writer removes its resources before its manifest; otherwise
// unlink the listed objects, remove each listed file only while it is still
// the file that was judged, and finally remove the manifest, again only by
// identity. A manifest republished by the live holder of a reused pid, and the
// cards it republished at the same paths, therefore survive.
void reap_stale_shm_resources(const StaleScan& scan);

}  // namespace tt::tt_metal::distributed
