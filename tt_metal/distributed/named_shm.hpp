// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>

namespace tt::tt_metal::distributed {

/**
 * @brief RAII wrapper around a POSIX named shared memory region.
 *
 * Provides create/open/close/unlink semantics for inter-process shared memory.
 * The underlying object lives in /dev/shm/ on Linux.
 */
class NamedShm {
public:
    NamedShm() = default;
    ~NamedShm() noexcept;

    NamedShm(NamedShm&& other) noexcept;
    NamedShm& operator=(NamedShm&& other) noexcept;

    NamedShm(const NamedShm&) = delete;
    NamedShm& operator=(const NamedShm&) = delete;

    /**
     * @brief Create a new named shared memory region.
     *
     * Creates the shm object via shm_open(O_CREAT|O_EXCL|O_RDWR), sets its size via ftruncate,
     * and maps it with mmap(MAP_SHARED). The region is zero-initialized.
     *
     * @param name POSIX shm name (e.g. "/tt_h2d_abc123"). Must start with '/'.
     * @param size Size of the shared memory region in bytes.
     * @return NamedShm owning the new mapping.
     */
    static NamedShm create(const std::string& name, size_t size);

    /**
     * @brief Open and map an existing named shared memory region.
     *
     * Opens the shm object via shm_open(O_RDWR) and maps it with mmap(MAP_SHARED).
     *
     * @param name POSIX shm name matching a previously created region.
     * @param size Size of the region to map (must match the created size).
     * @return NamedShm with a mapping to the existing region.
     */
    static NamedShm open(const std::string& name, size_t size);

    /**
     * @brief Unmap the shared memory region from this process.
     *
     */
    void close();

    /**
     * @brief Remove the named shared memory object from the filesystem.
     *
     */
    void unlink();

    void* ptr() const { return ptr_; }
    size_t size() const { return size_; }
    const std::string& name() const { return name_; }
    bool is_open() const { return ptr_ != nullptr; }

private:
    NamedShm(const std::string& name, void* ptr, size_t size);

    std::string name_;
    void* ptr_ = nullptr;
    size_t size_ = 0;
};

/**
 * @brief Generate a unique POSIX shm name: /tt_{prefix}_{pid}_{counter}.
 */
std::string generate_shm_name(const std::string& prefix);

}  // namespace tt::tt_metal::distributed
