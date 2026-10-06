// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/shm_resource_tracker.hpp>
#include "shm_owner_liveness.hpp"
#include "shm_stale_scan.hpp"

#include <mutex>
#include <tt-logger/tt-logger.hpp>
#include <fmt/format.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <dirent.h>
#include <fcntl.h>
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

// Reads /proc/<pid>/stat: `state` is field 3, `num_threads` field 20 and
// `start_time` field 22. comm may contain spaces and parentheses, so fields are
// tokenised after the last ')'. False when the entry cannot be read.
bool read_proc_stat(pid_t pid, char& state, long& num_threads, uint64_t& start_time) {
    if (pid <= 0) {
        return false;
    }
    std::ifstream stat_file(fmt::format("/proc/{}/stat", pid));
    std::string line;
    if (!stat_file.is_open() || !std::getline(stat_file, line)) {
        return false;
    }
    const auto comm_end = line.rfind(')');
    if (comm_end == std::string::npos) {
        return false;
    }
    std::istringstream fields(line.substr(comm_end + 1));
    std::string token;
    if (!(fields >> token) || token.empty()) {
        return false;
    }
    state = token[0];
    // num_threads is the 17th token after state, start_time the 19th.
    try {
        for (int i = 0; i < 19; ++i) {
            if (!(fields >> token)) {
                return false;
            }
            if (i == 16) {
                num_threads = std::stol(token);
            }
        }
        start_time = static_cast<uint64_t>(std::stoull(token));
    } catch (...) {
        return false;
    }
    return true;
}

// What /proc says about a pid that kill(2) still accepts. A zombie (exited, not
// yet reaped) is dead. The state belongs to the thread-group leader, though: a
// live process whose main thread left through pthread_exit() also shows 'Z'
// while its other threads run, so 'Z' only counts with no other threads.
bool proc_says_dead(char state, long num_threads) { return state == 'X' || (state == 'Z' && num_threads <= 1); }

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

pid_t pid_from_shm_name(const std::string& shm_name) {
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
    if (kill(pid, 0) != 0 && errno != EPERM) {
        return false;
    }
    // kill(2) also succeeds for a zombie: a process that exited and has not
    // been reaped yet. Its resources are as orphaned as a reaped one's.
    char state = 0;
    long num_threads = 0;
    uint64_t start_time = 0;
    return !(read_proc_stat(pid, state, num_threads, start_time) && proc_says_dead(state, num_threads));
}

uint64_t process_start_time(pid_t pid) {
    char state = 0;
    long num_threads = 0;
    uint64_t start_time = 0;
    return read_proc_stat(pid, state, num_threads, start_time) ? start_time : 0;
}

bool is_process_alive(pid_t pid, uint64_t start_time) {
    if (pid <= 0) {
        return false;
    }
    if (kill(pid, 0) != 0 && errno != EPERM) {
        return false;
    }
    // One /proc read answers both questions (this runs inside 1 ms poll loops).
    char state = 0;
    long num_threads = 0;
    uint64_t current = 0;
    if (!read_proc_stat(pid, state, num_threads, current)) {
        return true;  // /proc not readable: nothing contradicts kill(2)
    }
    if (proc_says_dead(state, num_threads)) {
        return false;
    }
    return start_time == 0 || current == 0 || current == start_time;
}

ShmResourceTracker::ShmResourceTracker() :
    manifest_path_(manifest_path_for_pid(getpid())), start_time_(process_start_time(getpid())) {
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
    ofs << "start " << start_time_ << "\n";
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

bool read_manifest(const std::string& path, dev_t& dev, ino_t& ino, ManifestContents& contents) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd == -1) {
        return false;
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        return false;
    }
    dev = st.st_dev;
    ino = st.st_ino;
    std::string text;
    char buf[4096];
    for (;;) {
        const ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            ::close(fd);
            return false;  // a partial read must not be judged: entries would be missed
        }
        if (n == 0) {
            break;
        }
        text.append(buf, static_cast<std::size_t>(n));
    }
    ::close(fd);

    contents = ManifestContents{};
    std::istringstream lines(text);
    std::string line;
    bool first = true;
    while (std::getline(lines, line)) {
        // The first line is "start <ticks>" (see flush_manifest); older manifests have none.
        if (first && line.starts_with("start ")) {
            try {
                contents.start_time = std::stoull(line.substr(6));
            } catch (...) {
                contents.start_time = 0;
            }
        } else if (line.starts_with("shm ")) {
            contents.shm_names.push_back(line.substr(4));
        } else if (line.starts_with("file ")) {
            contents.file_paths.push_back(line.substr(5));
        }
        first = false;
    }
    return true;
}

StaleScan collect_stale_shm_resources() {
    StaleScan scan;
    DIR* dir = opendir("/dev/shm");
    if (!dir) {
        return scan;
    }

    const pid_t my_pid = getpid();
    const uint64_t my_start = process_start_time(my_pid);
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name(entry->d_name);

        // Manifests of dead owners. A live pid is not enough: the owner may
        // have died and its pid been handed to an unrelated process, which the
        // start time recorded in the manifest tells apart. The manifest is read
        // once here, from the file being judged, so the reap step works on the
        // entries that were judged even if the path is republished meanwhile.
        pid_t manifest_pid = extract_pid_from_manifest_name(name);
        if (manifest_pid > 0) {
            StaleManifest stale;
            stale.manifest.path = "/dev/shm/" + name;
            ManifestContents contents;
            if (!read_manifest(stale.manifest.path, stale.manifest.dev, stale.manifest.ino, contents)) {
                continue;  // gone already; nothing to judge
            }
            bool is_stale = false;
            if (manifest_pid == my_pid) {
                // A manifest at our own path is a predecessor's if it was
                // written by a process with a different start time: we were
                // handed its pid. Without a start line it cannot be told
                // apart from our own and is left alone.
                is_stale = contents.start_time != 0 && my_start != 0 && contents.start_time != my_start;
                stale.pid_reused = true;
            } else {
                const bool pid_alive = ShmResourceTracker::is_pid_alive(manifest_pid);
                is_stale = !pid_alive || !is_process_alive(manifest_pid, contents.start_time);
                stale.pid_reused = pid_alive;
            }
            if (is_stale) {
                stale.shm_names = std::move(contents.shm_names);
                for (auto& file_path : contents.file_paths) {
                    struct stat st{};
                    if (::stat(file_path.c_str(), &st) != 0) {
                        continue;  // already gone; nothing to remove
                    }
                    stale.files.push_back({std::move(file_path), st.st_dev, st.st_ino});
                }
                scan.manifests.push_back(std::move(stale));
            }
            continue;
        }

        // Orphaned shm objects not covered by any manifest.
        // Pattern: tt_{h2d|d2h}_{pid}_{random}_{counter}
        pid_t shm_pid = pid_from_shm_name(name);
        if (shm_pid > 0 && shm_pid != my_pid && !ShmResourceTracker::is_pid_alive(shm_pid)) {
            scan.orphan_shm_names.push_back(name);
        }
    }
    closedir(dir);
    return scan;
}

namespace {

// True while `path` still names the file that was judged (same device and inode).
bool still_the_judged_file(const JudgedFile& file) {
    struct stat st{};
    return ::stat(file.path.c_str(), &st) == 0 && st.st_dev == file.dev && st.st_ino == file.ino;
}

}  // namespace

void reap_stale_shm_resources(const StaleScan& scan) {
    for (const auto& manifest : scan.manifests) {
        // Every writer removes its resources before its manifest (shutdown,
        // signal cleanup, this scan), so a manifest that is gone or has been
        // replaced since it was judged means its resources are gone too; the
        // replacement belongs to a live owner and is left alone. The window
        // between these checks and the removals below is a few instructions
        // and is as small as the filesystem API allows.
        if (!still_the_judged_file(manifest.manifest)) {
            log_info(
                LogMetal,
                "ShmResourceTracker: stale manifest '{}' changed since it was judged; left for the next scan",
                manifest.manifest.path);
            continue;
        }
        for (const auto& shm_name : manifest.shm_names) {
            if (shm_unlink(shm_name.c_str()) == 0) {
                log_info(LogMetal, "ShmResourceTracker: removed stale shm '{}'", shm_name);
            }
        }
        for (const auto& file : manifest.files) {
            // Descriptor files are republished at the same path by a replacement
            // owner; only the file that was judged may go.
            if (still_the_judged_file(file) && std::remove(file.path.c_str()) == 0) {
                log_info(LogMetal, "ShmResourceTracker: removed stale file '{}'", file.path);
            }
        }
        if (!still_the_judged_file(manifest.manifest)) {
            continue;
        }
        std::remove(manifest.manifest.path.c_str());
        if (manifest.pid_reused) {
            log_info(
                LogMetal,
                "ShmResourceTracker: removed stale manifest '{}' (its pid now belongs to another process)",
                manifest.manifest.path);
        } else {
            log_info(LogMetal, "ShmResourceTracker: removed stale manifest '{}'", manifest.manifest.path);
        }
    }

    for (const auto& name : scan.orphan_shm_names) {
        std::string shm_name = "/" + name;
        if (shm_unlink(shm_name.c_str()) == 0) {
            log_info(LogMetal, "ShmResourceTracker: removed orphaned shm '{}'", shm_name);
        }
    }
}

void ShmResourceTracker::cleanup_stale_resources() { reap_stale_shm_resources(collect_stale_shm_resources()); }

}  // namespace tt::tt_metal::distributed
