// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <set>
#include <string>

namespace tt::tt_metal::distributed {

/**
 * @brief Process-global tracker for POSIX shared memory objects and descriptor files.
 *
 * Ensures that /dev/shm resources created by H2D/D2H sockets are cleaned up even
 * when the process exits abnormally. Two complementary mechanisms:
 *
 *  1. destructor / signal handler – cleans up resources on normal exit,
 *     std::exit(), SIGINT, SIGTERM.
 *
 *  2. Lease cleanup – on first use, scans /dev/shm for manifests whose lifetime
 *     flock can be acquired (handles SIGKILL and hard crashes across PID namespaces).
 *
 * A unique /dev/shm/tt_socket_manifest_v2_<token> records resources, including
 * descriptor files. A separate .lock file remains locked while that manifest is
 * atomically replaced. Cleanup never infers ownership from PIDs in resource names.
 * Legacy manifests and resources without a lease require manual cleanup after
 * confirming that their owners have exited. All processes sharing /dev/shm must
 * use the lease-aware tracker to prevent older binaries' unsafe PID-based sweep.
 */
class ShmResourceTracker {
public:
    static ShmResourceTracker& instance();

    ~ShmResourceTracker();

    ShmResourceTracker(const ShmResourceTracker&) = delete;
    ShmResourceTracker& operator=(const ShmResourceTracker&) = delete;

    void track_shm(const std::string& shm_name);
    void track_file(const std::string& file_path);

    void untrack_shm(const std::string& shm_name);
    void untrack_file(const std::string& file_path);

    void cleanup_all();
    void cleanup_from_signal();

    static void cleanup_stale_resources();

private:
    ShmResourceTracker();

    void flush_manifest();

    std::mutex mutex_;
    std::set<std::string> shm_names_;
    std::set<std::string> file_paths_;
    std::string manifest_path_;
};

}  // namespace tt::tt_metal::distributed
