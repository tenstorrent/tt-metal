// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/shm_resource_tracker.hpp"

#include <mutex>
#include <tt-logger/tt-logger.hpp>
#include <fmt/format.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace tt::tt_metal::distributed {

namespace {

pid_t extract_pid_from_manifest_name(const std::string& filename) {
    // Expected format: tt_socket_manifest_<pid>
    const std::string prefix = "tt_socket_manifest_";
    if (!filename.starts_with(prefix)) {
        return 0;
    }
    try {
        return static_cast<pid_t>(std::stol(filename.substr(prefix.size())));
    } catch (...) {
        return 0;
    }
}

struct sigaction prev_sigint, prev_sigterm;

void invoke_previous_handler(int sig, const struct sigaction& prev) {
    if (prev.sa_handler == SIG_IGN) {
        return;
    }
    if (prev.sa_flags & SA_SIGINFO) {
        // Restore the original SA_SIGINFO handler and re-raise so the kernel
        // delivers the signal with a real siginfo_t and ucontext_t*.
        struct sigaction restore = prev;
        sigaction(sig, &restore, nullptr);
        raise(sig);
        return;
    }
    if (prev.sa_handler != SIG_DFL) {
        prev.sa_handler(sig);
        return;
    }
    // Previous handler was SIG_DFL or SIG_IGN (or null): restore default and re-raise
    // so the process terminates with the correct signal exit status.
    signal(sig, SIG_DFL);
    raise(sig);
}

void signal_handler(int sig) {
    // Use try_lock to avoid deadlock if the signal interrupted a thread
    // holding mutex_. If we can't acquire the lock, the manifest file
    // ensures the next process will clean up via stale-PID scan.
    ShmResourceTracker::instance().cleanup_from_signal();

    const struct sigaction& prev = (sig == SIGINT) ? prev_sigint : prev_sigterm;
    invoke_previous_handler(sig, prev);
}

}  // namespace

std::string ShmResourceTracker::manifest_path_for_pid(pid_t pid) {
    return fmt::format("/dev/shm/tt_socket_manifest_{}", pid);
}

pid_t ShmResourceTracker::pid_from_shm_name(const std::string& shm_name) {
    // Expected format: [/]tt_{prefix}_{pid}_{random}_{counter} (NamedShm::make_unique_name).
    const std::string filename = (!shm_name.empty() && shm_name[0] == '/') ? shm_name.substr(1) : shm_name;
    if (!filename.starts_with("tt_")) {
        return 0;
    }
    auto first = filename.find('_', 3);
    if (first == std::string::npos) {
        return 0;
    }
    auto second = filename.find('_', first + 1);
    if (second == std::string::npos) {
        return 0;
    }
    try {
        return static_cast<pid_t>(std::stol(filename.substr(first + 1, second - first - 1)));
    } catch (...) {
        return 0;
    }
}

bool ShmResourceTracker::is_pid_alive(pid_t pid) {
    if (pid <= 0) {
        return false;
    }
    return kill(pid, 0) == 0 || errno == EPERM;
}

uint64_t ShmResourceTracker::process_start_time(pid_t pid) {
    if (pid <= 0) {
        return 0;
    }
    std::ifstream stat_file(fmt::format("/proc/{}/stat", pid));
    std::string line;
    if (!stat_file.is_open() || !std::getline(stat_file, line)) {
        return 0;
    }
    // "pid (comm) state ppid ... starttime ..." — comm may contain spaces, so
    // tokenize after the last ')'. starttime is field 22, i.e. the 20th token
    // after the closing parenthesis (state is the first).
    const auto comm_end = line.rfind(')');
    if (comm_end == std::string::npos) {
        return 0;
    }
    std::istringstream fields(line.substr(comm_end + 1));
    std::string token;
    for (int i = 0; i < 20; ++i) {
        if (!(fields >> token)) {
            return 0;
        }
    }
    try {
        return static_cast<uint64_t>(std::stoull(token));
    } catch (...) {
        return 0;
    }
}

bool ShmResourceTracker::is_process_alive(pid_t pid, uint64_t start_time) {
    if (!is_pid_alive(pid)) {
        return false;
    }
    if (start_time == 0) {
        return true;
    }
    const uint64_t current = process_start_time(pid);
    return current == 0 || current == start_time;
}

ShmResourceTracker::ShmResourceTracker() : manifest_path_(manifest_path_for_pid(getpid())) {
    cleanup_stale_resources();

    struct sigaction sa{};
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, &prev_sigint);
    sigaction(SIGTERM, &sa, &prev_sigterm);
}

ShmResourceTracker::~ShmResourceTracker() {
    try {
        cleanup_all();
    } catch (const std::exception& e) {
        log_warning(LogMetal, "ShmResourceTracker cleanup failed: {}", e.what());
    } catch (...) {
        log_warning(LogMetal, "ShmResourceTracker cleanup failed with unknown exception");
    }
}

ShmResourceTracker& ShmResourceTracker::instance() {
    static ShmResourceTracker tracker;
    return tracker;
}

void ShmResourceTracker::track_shm(const std::string& shm_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    shm_names_.insert(shm_name);
    flush_manifest();
}

void ShmResourceTracker::track_file(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    file_paths_.insert(file_path);
    flush_manifest();
}

void ShmResourceTracker::untrack_shm(const std::string& shm_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    shm_names_.erase(shm_name);
    flush_manifest();
}

void ShmResourceTracker::untrack_file(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    file_paths_.erase(file_path);
    flush_manifest();
}

void ShmResourceTracker::flush_manifest() {
    if (shm_names_.empty() && file_paths_.empty()) {
        std::remove(manifest_path_.c_str());
        return;
    }
    // Write to a temp file and atomically rename to avoid leaving a
    // truncated manifest if the process is killed mid-write.
    const std::string tmp_path = manifest_path_ + ".tmp";
    std::ofstream ofs(tmp_path, std::ios::trunc);
    if (!ofs) {
        return;
    }
    for (const auto& name : shm_names_) {
        ofs << "shm " << name << "\n";
    }
    for (const auto& path : file_paths_) {
        ofs << "file " << path << "\n";
    }
    ofs.flush();
    if (!ofs) {
        std::remove(tmp_path.c_str());
        return;
    }
    if (::rename(tmp_path.c_str(), manifest_path_.c_str()) != 0) {
        std::remove(tmp_path.c_str());
    }
}

void ShmResourceTracker::cleanup_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& name : shm_names_) {
        if (shm_unlink(name.c_str()) == 0) {
            log_debug(LogMetal, "ShmResourceTracker: cleaned up shm '{}'", name);
        }
    }
    shm_names_.clear();

    for (const auto& path : file_paths_) {
        if (std::remove(path.c_str()) == 0) {
            log_debug(LogMetal, "ShmResourceTracker: cleaned up file '{}'", path);
        }
    }
    file_paths_.clear();

    std::remove(manifest_path_.c_str());
}

void ShmResourceTracker::cleanup_from_signal() {
    // try_lock avoids deadlock if the signal interrupted a thread holding mutex_.
    // If we can't lock, leave the manifest intact so the next process can
    // discover and clean up all resources via stale-PID scan.
    if (!mutex_.try_lock()) {
        return;
    }

    // Lock acquired. shm_unlink and unlink are async-signal-safe.
    // Avoid logging here (not async-signal-safe).
    for (const auto& name : shm_names_) {
        shm_unlink(name.c_str());
    }
    shm_names_.clear();

    for (const auto& path : file_paths_) {
        ::unlink(path.c_str());
    }
    file_paths_.clear();

    ::unlink(manifest_path_.c_str());
    mutex_.unlock();
}

void ShmResourceTracker::cleanup_stale_resources() {
    DIR* dir = opendir("/dev/shm");
    if (!dir) {
        return;
    }

    std::vector<std::string> stale_manifests;
    std::vector<std::string> stale_shm_names;

    pid_t my_pid = getpid();
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name(entry->d_name);

        // Check for manifest files from dead processes
        pid_t manifest_pid = extract_pid_from_manifest_name(name);
        if (manifest_pid > 0 && manifest_pid != my_pid && !is_pid_alive(manifest_pid)) {
            stale_manifests.push_back("/dev/shm/" + name);
            continue;
        }

        // Check for orphaned shm objects from dead processes
        // Pattern: tt_{h2d|d2h}_{pid}_{random}_{counter}
        pid_t shm_pid = pid_from_shm_name(name);
        if (shm_pid > 0 && shm_pid != my_pid && !is_pid_alive(shm_pid)) {
            stale_shm_names.push_back(name);
        }
    }
    closedir(dir);

    // Clean up resources listed in stale manifests
    for (const auto& manifest : stale_manifests) {
        std::ifstream ifs(manifest);
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.starts_with("shm ")) {
                std::string shm_name = line.substr(4);
                if (shm_unlink(shm_name.c_str()) == 0) {
                    log_info(LogMetal, "ShmResourceTracker: removed stale shm '{}'", shm_name);
                }
            } else if (line.starts_with("file ")) {
                std::string file_path = line.substr(5);
                if (std::remove(file_path.c_str()) == 0) {
                    log_info(LogMetal, "ShmResourceTracker: removed stale file '{}'", file_path);
                }
            }
        }
        std::remove(manifest.c_str());
        log_info(LogMetal, "ShmResourceTracker: removed stale manifest '{}'", manifest);
    }

    // Clean up orphaned shm objects not covered by any manifest
    for (const auto& name : stale_shm_names) {
        std::string shm_name = "/" + name;
        if (shm_unlink(shm_name.c_str()) == 0) {
            log_info(LogMetal, "ShmResourceTracker: removed orphaned shm '{}'", shm_name);
        }
    }
}

}  // namespace tt::tt_metal::distributed
