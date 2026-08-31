// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shm_publisher.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ttnvtop {

namespace {

std::string shm_name_for(uint64_t asic_id) {
    // shm_open expects a leading slash and no /dev/shm prefix.
    char buf[64];
    std::snprintf(buf, sizeof(buf), "/tt_device_%lu_util", static_cast<unsigned long>(asic_id));
    return std::string(buf);
}

uint64_t monotonic_us() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

}  // namespace

ShmPublisher::ShmPublisher(ShmPublisher&& other) noexcept { *this = std::move(other); }

ShmPublisher& ShmPublisher::operator=(ShmPublisher&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_;
        map_ = other.map_;
        map_size_ = other.map_size_;
        header_ = other.header_;
        cores_ = other.cores_;
        num_cores_ = other.num_cores_;
        shm_name_ = std::move(other.shm_name_);
        other.fd_ = -1;
        other.map_ = nullptr;
        other.map_size_ = 0;
        other.header_ = nullptr;
        other.cores_ = nullptr;
        other.num_cores_ = 0;
    }
    return *this;
}

ShmPublisher::~ShmPublisher() { close(); }

bool ShmPublisher::open(uint64_t asic_id, uint32_t arch_id, uint32_t signal_sources, uint32_t num_cores) {
    close();
    shm_name_ = shm_name_for(asic_id);
    // EXCLUSIVE WRITER, enforced by flock -- not by unlinking.
    //
    // This used to `shm_unlink` first and then create with O_CREAT, on the theory that it
    // was clearing a stale file from a crashed collector. What it actually did was let a
    // SECOND collector silently steal the name: unlink detaches the inode the first one is
    // still writing to, O_CREAT makes a fresh one, and from then on the first collector
    // publishes into an inode no reader can open. The viewer sees whichever collector
    // created each chip's file last, so with two running the grid shows a mix of live and
    // permanently frozen chips -- diagnosed as "the TUI is hung" before it was understood.
    //
    // flock has exactly the lifetime we need and no heuristics: it is released when the fd
    // closes OR when the process dies, so a crashed collector's file is takeable while a
    // live one's is not. No pid liveness guessing, no unlink race.
    fd_ = ::shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd_ < 0) {
        return false;
    }
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
        // Someone live owns this chip. Name them: the pid is right there in the header.
        uint32_t other = 0;
        UtilShmHeader peek{};
        if (::pread(fd_, &peek, sizeof(peek), 0) == static_cast<ssize_t>(sizeof(peek)) &&
            std::memcmp(peek.magic, kShmMagic, 4) == 0) {
            other = peek.collector_pid;
        }
        std::fprintf(
            stderr,
            "ttnvtop-collector: another collector is already publishing %s%s%u%s.\n"
            "  Two collectors cannot share one chip's shared memory -- the second would take\n"
            "  the name and the first would publish where no viewer can see it. Stop that one\n"
            "  first, or use --device to give each collector a disjoint set of chips.\n",
            shm_name_.c_str(),
            other ? " (pid " : "",
            other,
            other ? ")" : "");
        std::fflush(stderr);
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    map_size_ = shm_file_size(num_cores);
    if (::ftruncate(fd_, static_cast<off_t>(map_size_)) != 0) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    map_ = ::mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (map_ == MAP_FAILED) {
        map_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    std::memset(map_, 0, map_size_);
    header_ = static_cast<UtilShmHeader*>(map_);
    cores_ = reinterpret_cast<PerCoreView*>(static_cast<char*>(map_) + sizeof(UtilShmHeader));
    num_cores_ = num_cores;

    std::memcpy(header_->magic, kShmMagic, 4);
    header_->version = kShmVersion;
    header_->struct_size = sizeof(PerCoreView);
    header_->asic_id = asic_id;
    header_->arch_id = arch_id;
    header_->signal_sources = signal_sources;
    header_->epoch_us = monotonic_us();
    header_->last_update_us = header_->epoch_us;
    header_->num_cores = num_cores;
    header_->host_assigned_id = 0;
    header_->collector_pid = static_cast<uint32_t>(::getpid());
    return true;
}

void ShmPublisher::mark_updated() {
    if (header_ != nullptr) {
        // Ensure prior core writes are visible before the timestamp update.
        __atomic_thread_fence(__ATOMIC_RELEASE);
        header_->last_update_us = monotonic_us();
    }
}

void ShmPublisher::close() {
    if (map_ != nullptr) {
        ::munmap(map_, map_size_);
        map_ = nullptr;
        map_size_ = 0;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    if (!shm_name_.empty()) {
        ::shm_unlink(shm_name_.c_str());
        shm_name_.clear();
    }
    header_ = nullptr;
    cores_ = nullptr;
    num_cores_ = 0;
}

std::vector<ShmFileEntry> list_shm_files() {
    std::vector<ShmFileEntry> out;
    DIR* d = ::opendir("/dev/shm");
    if (d == nullptr) {
        return out;
    }
    while (struct dirent* e = ::readdir(d)) {
        const std::string name = e->d_name;
        // Match tt_device_<digits>_util exactly.
        if (name.rfind("tt_device_", 0) != 0) {
            continue;
        }
        const std::string suffix = "_util";
        if (name.size() < suffix.size() + std::string("tt_device_").size() + 1) {
            continue;
        }
        if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
            continue;
        }
        const std::string digits = name.substr(
            std::string("tt_device_").size(), name.size() - std::string("tt_device_").size() - suffix.size());
        if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos) {
            continue;
        }
        ShmFileEntry f;
        f.path = "/dev/shm/" + name;
        f.asic_id = std::strtoull(digits.c_str(), nullptr, 10);
        out.push_back(std::move(f));
    }
    ::closedir(d);
    return out;
}

}  // namespace ttnvtop
