// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/named_shm.hpp"

#include <tt_stl/assert.hpp>

#include <cerrno>
#include <cstring>
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

    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
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
    return NamedShm(name, ptr, size);
}

NamedShm NamedShm::open(const std::string& name, size_t size) {
    TT_FATAL(!name.empty() && name[0] == '/', "POSIX shm name must start with '/': {}", name);
    TT_FATAL(size > 0, "Shared memory size must be > 0");

    int fd = shm_open(name.c_str(), O_RDWR, 0666);
    TT_FATAL(fd != -1, "shm_open(open) failed for '{}': {}", name, std::strerror(errno));

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
        shm_unlink(name_.c_str());
        name_.clear();
    }
}

}  // namespace tt::tt_metal::distributed
