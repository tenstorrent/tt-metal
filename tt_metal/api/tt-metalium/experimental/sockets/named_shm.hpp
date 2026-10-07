// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
 * Host-only objects live in /dev/shm/ on Linux. With TT_METAL_SOCKET_HUGEPAGE_DIR,
 * device-accessible regions larger than a system page use one page on that hugetlbfs mount.
 * Their exported name is an absolute file path; connectors need the same library support.
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
     * @brief Allocate shared memory that will be pinned for device DMA.
     * Uses TT_METAL_SOCKET_HUGEPAGE_DIR for regions larger than one system page.
     * The requested region must fit in one hugepage. name() and size() describe
     * the actual backing. Pinning remains the caller's responsibility.
     */
    static NamedShm create_for_device(const std::string& name, size_t size);

    /**
     * @brief Open and map an existing named shared memory region.
     *
     * Opens a POSIX shm name or an absolute hugetlbfs path and maps it with mmap(MAP_SHARED).
     * Hugepage paths are self-describing; connectors do not need TT_METAL_SOCKET_HUGEPAGE_DIR.
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
    static NamedShm create_impl(const std::string& name, size_t size, bool device_access);
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
