// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <set>
#include <string>
#include <sys/types.h>

namespace tt::tt_metal::distributed {

/**
 * @brief Process-global tracker for POSIX shared memory objects and descriptor files.
 *
 * Ensures that /dev/shm resources created by H2D/D2H sockets are cleaned up even
 * when the process exits abnormally. Two complementary mechanisms:
 *
 *  1. atexit handler  – cleans up resources on normal exit, uncaught exceptions,
 *     std::exit(), and signals that cause exit() (SIGTERM, SIGINT default handlers).
 *
 *  2. Stale-PID cleanup – on first use, scans /dev/shm for resources whose owning
 *     PID is no longer alive (handles SIGKILL, hard crashes, power loss).
 *
 * A manifest file /dev/shm/tt_socket_manifest_<pid> is maintained so that the
 * stale cleanup can discover descriptor files (whose names don't embed a PID).
 */
class ShmResourceTracker {
public:
    static ShmResourceTracker& instance();

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
    static std::string manifest_path_for_pid(pid_t pid);
    static bool is_pid_alive(pid_t pid);

    std::mutex mutex_;
    std::set<std::string> shm_names_;
    std::set<std::string> file_paths_;
    std::string manifest_path_;
};

}  // namespace tt::tt_metal::distributed
