// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/named_shm.hpp>
#include <tt-metalium/experimental/sockets/shm_resource_tracker.hpp>

#include <tt_stl/assert.hpp>
#include <fmt/format.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <linux/magic.h>
#include <sys/vfs.h>
#include <random>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tt::tt_metal::distributed {

namespace {
bool is_hugepage_path(const std::string& name) { return name.find('/', 1) != std::string::npos; }

int unlink_backing(const std::string& name) {
    return is_hugepage_path(name) ? ::unlink(name.c_str()) : shm_unlink(name.c_str());
}

size_t hugepage_size(int fd, const std::string& name) {
    struct statfs fs{};
    TT_FATAL(fstatfs(fd, &fs) == 0, "fstatfs failed for '{}': {}", name, std::strerror(errno));
    TT_FATAL(fs.f_type == HUGETLBFS_MAGIC, "Socket backing '{}' must be on hugetlbfs", name);
    return static_cast<size_t>(fs.f_bsize);
}
}  // namespace

NamedShm::NamedShm(const std::string& name, void* ptr, size_t size) : name_(name), ptr_(ptr), size_(size) {}

NamedShm::~NamedShm() noexcept { close(); }

NamedShm::NamedShm(NamedShm&& other) noexcept : name_(std::move(other.name_)), ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

NamedShm& NamedShm::operator=(NamedShm&& other) noexcept {
    if (this != &other) {
        close();
        name_ = std::move(other.name_);
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

NamedShm NamedShm::create(const std::string& name, size_t size) {
    TT_FATAL(!name.empty() && name[0] == '/' && !is_hugepage_path(name), "Invalid POSIX shm name: {}", name);
    TT_FATAL(size > 0, "Shared memory size must be > 0");

    auto& tracker = ShmResourceTracker::instance();
    std::string backing = name;
    if (const char* dir = std::getenv("TT_METAL_SOCKET_HUGEPAGE_DIR");
        dir && *dir && size > static_cast<size_t>(sysconf(_SC_PAGESIZE))) {
        TT_FATAL(std::filesystem::path(dir).is_absolute(), "TT_METAL_SOCKET_HUGEPAGE_DIR must be absolute");
        backing = (std::filesystem::path(dir) / name.substr(1)).string();
        TT_FATAL(is_hugepage_path(backing), "TT_METAL_SOCKET_HUGEPAGE_DIR cannot be the root directory");
    }
    const bool huge = is_hugepage_path(backing);
    int fd = huge ? ::open(backing.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600)
                  : shm_open(backing.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    TT_FATAL(fd != -1, "Creating socket shared memory '{}' failed: {}", backing, std::strerror(errno));

    void* ptr = MAP_FAILED;
    try {
        if (huge) {
            const size_t page = hugepage_size(fd, backing);
            TT_FATAL(
                size <= page,
                "Socket buffer {} B exceeds one {} B hugepage; use a larger-page hugetlbfs mount",
                size,
                page);
            size = page;
        }
        TT_FATAL(
            ftruncate(fd, static_cast<off_t>(size)) == 0,
            "ftruncate failed for '{}': {}",
            backing,
            std::strerror(errno));
        ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        TT_FATAL(ptr != MAP_FAILED, "mmap failed for '{}': {}", backing, std::strerror(errno));
        std::memset(ptr, 0, size);
        if (huge) {
            tracker.track_file(backing);
        } else {
            tracker.track_shm(backing);
        }
    } catch (...) {
        if (ptr != MAP_FAILED) {
            munmap(ptr, size);
        }
        ::close(fd);
        unlink_backing(backing);
        throw;
    }
    ::close(fd);
    return NamedShm(backing, ptr, size);
}

NamedShm NamedShm::open(const std::string& name, size_t size) {
    TT_FATAL(!name.empty() && name[0] == '/', "Shared memory name must start with '/': {}", name);
    TT_FATAL(size > 0, "Shared memory size must be > 0");

    const bool huge = is_hugepage_path(name);
    int fd = huge ? ::open(name.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW) : shm_open(name.c_str(), O_RDWR, 0600);
    TT_FATAL(fd != -1, "Opening socket shared memory '{}' failed: {}", name, std::strerror(errno));
    void* ptr = MAP_FAILED;
    try {
        if (huge) {
            const size_t page = hugepage_size(fd, name);
            TT_FATAL(size == page, "Hugepage socket mapping must span exactly one {} B page, got {}", page, size);
        }
        struct stat st{};
        TT_FATAL(fstat(fd, &st) == 0, "fstat failed for '{}': {}", name, std::strerror(errno));
        TT_FATAL(
            static_cast<size_t>(st.st_size) >= size,
            "Shared memory '{}' backing size ({}) is smaller than requested size ({})",
            name,
            st.st_size,
            size);
        ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        TT_FATAL(ptr != MAP_FAILED, "mmap failed for '{}': {}", name, std::strerror(errno));
    } catch (...) {
        ::close(fd);
        throw;
    }
    ::close(fd);
    return NamedShm(name, ptr, size);
}

void NamedShm::close() {
    if (ptr_ != nullptr) {
        munmap(ptr_, size_);
        ptr_ = nullptr;
        size_ = 0;
    }
}

void NamedShm::unlink() {
    close();
    if (!name_.empty()) {
        int rc = unlink_backing(name_);
        if (rc == 0 || errno == ENOENT) {
            if (is_hugepage_path(name_)) {
                ShmResourceTracker::instance().untrack_file(name_);
            } else {
                ShmResourceTracker::instance().untrack_shm(name_);
            }
            name_.clear();
        }
    }
}

std::string generate_shm_name(const std::string& prefix) {
    static std::atomic<uint32_t> counter{0};
    static const uint32_t random_number = []() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dist;
        return dist(gen);
    }();
    return fmt::format("/tt_{}_{}_{}_{}", prefix, getpid(), random_number, counter.fetch_add(1));
}

}  // namespace tt::tt_metal::distributed
