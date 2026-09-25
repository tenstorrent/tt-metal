// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
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
 *  1. destructor / signal handler – cleans up resources on normal exit,
 *     std::exit(), SIGINT, SIGTERM.
 *
 *  2. Stale-PID cleanup – on first use, scans /dev/shm for resources whose owning
 *     PID is no longer alive (handles SIGKILL, hard crashes, power loss).
 *
 * A manifest file /dev/shm/tt_socket_manifest_<pid> is maintained so that the
 * stale cleanup can discover descriptor files (whose names don't embed a PID).
 *
 * The owner-identity helpers below are the same liveness rule the stale scan
 * uses, exposed so connectors can apply it to a descriptor or segment they are
 * about to attach to: a file left behind by a dead owner is "not published yet",
 * not a socket. They never construct the tracker instance.
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

    // PID embedded in a NamedShm name ("/tt_{prefix}_{pid}_{random}_{counter}", with or
    // without the leading '/'). 0 when the name does not follow that pattern.
    static pid_t pid_from_shm_name(const std::string& shm_name);

    // kill(pid, 0): true if the process exists (or we lack permission to signal it).
    static bool is_pid_alive(pid_t pid);

    // Process start time in clock ticks since boot (/proc/<pid>/stat field 22), which
    // together with the pid identifies one process instance: a reused pid gets a new
    // start time. 0 when it cannot be read (not Linux, process gone, procfs missing).
    static uint64_t process_start_time(pid_t pid);

    // is_pid_alive() refined by start time. `start_time == 0` (unknown, e.g. a descriptor
    // written before the stamp existed) falls back to the pid check alone.
    static bool is_process_alive(pid_t pid, uint64_t start_time);

private:
    ShmResourceTracker();

    void flush_manifest();
    static std::string manifest_path_for_pid(pid_t pid);

    std::mutex mutex_;
    std::set<std::string> shm_names_;
    std::set<std::string> file_paths_;
    std::string manifest_path_;
};

}  // namespace tt::tt_metal::distributed
