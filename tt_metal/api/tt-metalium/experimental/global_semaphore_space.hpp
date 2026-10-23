// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Experimental: no API-stability guarantee. Everything in this header may change or be removed.

#pragma once

#include <cstdint>
#include <memory>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

class GlobalSemaphoreSpaceImpl;

/**
 * Space for up to `capacity()` live GlobalSemaphores on a fixed set of worker cores.
 *
 * Create the space before any program places L1 (circular buffers, dataflow buffers, scratchpads)
 * on those cores, or before loading a sub-device manager with local L1 that covers them. Semaphores
 * can then be created from it at any time, including after programs have run and while they stay
 * cached.
 *
 * Every core of the space holds capacity() * (L1 alignment) bytes, used or not. Each semaphore keeps
 * its space alive, so that memory is freed once the space and all of its semaphores are destroyed.
 */
class GlobalSemaphoreSpace {
public:
    // Use CreateGlobalSemaphoreSpace() instead of calling this directly.
    explicit GlobalSemaphoreSpace(std::shared_ptr<GlobalSemaphoreSpaceImpl> impl);

    GlobalSemaphoreSpace(const GlobalSemaphoreSpace&) = delete;
    GlobalSemaphoreSpace& operator=(const GlobalSemaphoreSpace&) = delete;
    GlobalSemaphoreSpace(GlobalSemaphoreSpace&&) noexcept;
    GlobalSemaphoreSpace& operator=(GlobalSemaphoreSpace&&) noexcept;
    ~GlobalSemaphoreSpace();

    const CoreRangeSet& cores() const;
    uint32_t capacity() const;
    uint32_t num_live_semaphores() const;

    /**
     * Creates a GlobalSemaphore on `cores`, which must be a subset of cores(), and sets it to 0.
     * Throws if capacity() semaphores are already live.
     *
     * The semaphore has the same L1 address on every one of its cores, on every device of the mesh.
     * Copies refer to the same semaphore, and its slot goes back to the space when the last copy is
     * destroyed. Programs and traces only get the address, so keep a copy alive until they finish.
     *
     * reset_semaphore_value() waits until the write lands. It only waits for programs on command
     * queue 0, so make sure no kernel (on this chip or another) is still signalling the semaphore.
     */
    GlobalSemaphore create_semaphore(const CoreRangeSet& cores);

    // Internal (host runtime, tests).
    const GlobalSemaphoreSpaceImpl& impl() const { return *impl_; }

private:
    std::shared_ptr<GlobalSemaphoreSpaceImpl> impl_;
};

/**
 * Reserves space for `capacity` GlobalSemaphores on `cores` (logical worker cores) of every device in
 * `device`. See GlobalSemaphoreSpace for when this must be called.
 */
GlobalSemaphoreSpace CreateGlobalSemaphoreSpace(
    distributed::MeshDevice& device, const CoreRangeSet& cores, uint32_t capacity);

}  // namespace experimental
}  // namespace tt::tt_metal
