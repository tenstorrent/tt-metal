// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/shm_resource_tracker.hpp>

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
#include <string_view>
#include <system_error>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tt::tt_metal::distributed {

namespace {

constexpr std::string_view lease_prefix = "tt_socket_manifest_v2_";
constexpr std::string_view lease_suffix = ".lock";

struct FileDescriptor {
    int fd = -1;
    explicit FileDescriptor(int fd = -1) : fd(fd) {}
    ~FileDescriptor() {
        if (fd >= 0) {
            ::close(fd);
        }
    }
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
};

bool same_file(int fd, const std::string& path) {
    struct stat opened{}, current{};
    return ::fstat(fd, &opened) == 0 && ::lstat(path.c_str(), &current) == 0 && S_ISREG(current.st_mode) &&
           opened.st_dev == current.st_dev && opened.st_ino == current.st_ino;
}

// Keep the lease inode stable while flush_manifest atomically replaces the data
// file. flock is independent of PID namespaces and is released even on SIGKILL.
struct ManifestLease {
    FileDescriptor descriptor;
    std::string lock_path;
    std::string manifest_path;

    ManifestLease() {
        for (;;) {
            lock_path = fmt::format("/dev/shm/{}XXXXXX{}", lease_prefix, lease_suffix);
            descriptor.fd = ::mkostemps(lock_path.data(), lease_suffix.size(), O_CLOEXEC);
            if (descriptor.fd < 0) {
                throw std::system_error(errno, std::generic_category(), "Create socket manifest lease");
            }
            int rc;
            do {
                rc = ::flock(descriptor.fd, LOCK_EX);
            } while (rc < 0 && errno == EINTR);
            if (rc < 0) {
                const int error = errno;
                ::unlink(lock_path.c_str());
                throw std::system_error(error, std::generic_category(), "Lock socket manifest lease");
            }
            // A sweeper may have acquired the new file before our flock. In
            // that case it has retired this inode; create a fresh lease.
            if (same_file(descriptor.fd, lock_path)) {
                break;
            }
            ::close(descriptor.fd);
            descriptor.fd = -1;
        }
        manifest_path = lock_path.substr(0, lock_path.size() - lease_suffix.size());
    }

    ~ManifestLease() {
        // If tracker cleanup threw, keep the unlocked lease discoverable so a
        // later process can finish cleaning the remaining manifest.
        struct stat st{};
        if (::lstat(manifest_path.c_str(), &st) != 0 && errno == ENOENT) {
            ::unlink(lock_path.c_str());
        }
    }
};

ManifestLease& manifest_lease() {
    // Constructed before the tracker, so its lock outlives tracker cleanup.
    static ManifestLease lease;
    return lease;
}

void cleanup_stale_manifest(const std::string& lock_path) {
    FileDescriptor descriptor(::open(lock_path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW));
    if (descriptor.fd < 0 || ::flock(descriptor.fd, LOCK_EX | LOCK_NB) != 0 || !same_file(descriptor.fd, lock_path)) {
        return;
    }

    // Only an unlocked lease proves that its owner is gone. Keep the lock
    // through all unlinks so concurrent sweepers cannot act on the same owner.
    const std::string manifest = lock_path.substr(0, lock_path.size() - lease_suffix.size());
    std::ifstream ifs(manifest);
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.starts_with("shm ")) {
            const auto shm_name = line.substr(4);
            if (::shm_unlink(shm_name.c_str()) == 0) {
                log_info(LogMetal, "ShmResourceTracker: removed stale shm '{}'", shm_name);
            }
        } else if (line.starts_with("file ")) {
            const auto file_path = line.substr(5);
            if (::unlink(file_path.c_str()) == 0) {
                log_info(LogMetal, "ShmResourceTracker: removed stale file '{}'", file_path);
            }
        }
    }
    ::unlink(manifest.c_str());
    ::unlink((manifest + ".tmp").c_str());
    ::unlink(lock_path.c_str());
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
    // ensures the next process will clean up via the released lease.
    ShmResourceTracker::instance().cleanup_from_signal();

    const struct sigaction& prev = (sig == SIGINT) ? prev_sigint : prev_sigterm;
    invoke_previous_handler(sig, prev);
}

}  // namespace

ShmResourceTracker::ShmResourceTracker() : manifest_path_(manifest_lease().manifest_path) {
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
    // discover and clean up all resources via the released lease.
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

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        const std::string_view name(entry->d_name);
        if (name.starts_with(lease_prefix) && name.ends_with(lease_suffix) &&
            name.size() == lease_prefix.size() + 6 + lease_suffix.size()) {
            cleanup_stale_manifest("/dev/shm/" + std::string(name));
        }
    }
    closedir(dir);
    // Legacy manifests and untracked shm names have no namespace-independent
    // ownership proof. Never infer that they are stale from kill(pid, 0).
}

}  // namespace tt::tt_metal::distributed
