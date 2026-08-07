// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/named_shm.hpp"
#include "tt_metal/distributed/shm_resource_tracker.hpp"

#include <tt_stl/assert.hpp>
#include <fmt/format.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <random>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tt::tt_metal::distributed {

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
    TT_FATAL(!name.empty() && name[0] == '/', "POSIX shm name must start with '/': {}", name);
    TT_FATAL(size > 0, "Shared memory size must be > 0");

    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    TT_FATAL(
        fd != -1,
        "shm_open(create) failed for '{}': {}. If a stale shm object exists, remove it with shm_unlink or delete "
        "/dev/shm{}.",
        name,
        std::strerror(errno),
        name);

    int rc = ftruncate(fd, static_cast<off_t>(size));
    if (rc == -1) {
        int saved_errno = errno;
        ::close(fd);
        shm_unlink(name.c_str());
        TT_THROW("ftruncate failed for '{}': {}", name, std::strerror(saved_errno));
    }

    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    int mmap_errno = errno;
    ::close(fd);
    if (ptr == MAP_FAILED) {
        shm_unlink(name.c_str());
        TT_THROW("mmap failed for '{}': {}", name, std::strerror(mmap_errno));
    }

    std::memset(ptr, 0, size);
    ShmResourceTracker::instance().track_shm(name);
    return NamedShm(name, ptr, size);
}

NamedShm NamedShm::create_anonymous(size_t size) {
    TT_FATAL(size > 0, "Anonymous shared memory size must be > 0");
    // Without IOMMU, UMD pins with TENSTORRENT_PIN_PAGES_CONTIGUOUS. Ordinary
    // MAP_ANONYMOUS pages are not file-backed (so they clear the tmpfs EINVAL
    // from #47616) but multi-page regions are still usually not physically
    // contiguous. Prefer a 1G hugetlb page when the request spans >1 host page
    // so CONTIGUOUS pin succeeds (needed for Gemma4 ~11 KiB D2H fifos under
    // slow-dispatch, where the sysmem hugepage fallback cannot claim dispatch
    // cores). Fall back to ordinary anonymous mmap otherwise.
    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
#if defined(MAP_HUGETLB) && defined(MAP_HUGE_SHIFT)
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif
    constexpr size_t k_huge_1g = 1ULL << 30;
    if (size > page_size) {
        TT_FATAL(size <= k_huge_1g, "Anonymous DMA buffer size {} exceeds 1G hugepage", size);
        void* huge_ptr = mmap(
            nullptr, k_huge_1g, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED | MAP_HUGETLB | MAP_HUGE_1GB, -1, 0);
        if (huge_ptr != MAP_FAILED) {
            // Kernel zero-initializes hugetlb pages.
            return NamedShm("", huge_ptr, k_huge_1g);
        }
    }
#endif
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, -1, 0);
    if (ptr == MAP_FAILED) {
        TT_THROW("mmap(MAP_ANONYMOUS|MAP_SHARED) failed for size {}: {}", size, std::strerror(errno));
    }
    return NamedShm("", ptr, size);
}

NamedShm NamedShm::open(const std::string& name, size_t size) {
    TT_FATAL(!name.empty() && name[0] == '/', "POSIX shm name must start with '/': {}", name);
    TT_FATAL(size > 0, "Shared memory size must be > 0");

    int fd = shm_open(name.c_str(), O_RDWR, 0600);
    TT_FATAL(fd != -1, "shm_open(open) failed for '{}': {}", name, std::strerror(errno));

    struct stat st;
    TT_FATAL(fstat(fd, &st) == 0, "fstat failed for '{}': {}", name, std::strerror(errno));
    TT_FATAL(
        static_cast<size_t>(st.st_size) >= size,
        "Shared memory '{}' backing size ({}) is smaller than requested size ({})",
        name,
        st.st_size,
        size);

    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    int mmap_errno = errno;
    ::close(fd);
    if (ptr == MAP_FAILED) {
        TT_THROW("mmap failed for '{}': {}", name, std::strerror(mmap_errno));
    }

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
        int rc = shm_unlink(name_.c_str());
        if (rc == 0 || errno == ENOENT) {
            ShmResourceTracker::instance().untrack_shm(name_);
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
