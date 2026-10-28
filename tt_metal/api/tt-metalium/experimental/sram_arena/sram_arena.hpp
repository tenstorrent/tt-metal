// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <tt-metalium/experimental/sram_arena/sram_arena_properties.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_metal::experimental {

struct SRAMArenaImpl;

/**
 * An opaque handle to a device-side memory reservation with a user-managed lifetime.
 *
 * An SRAMArena represents either a lockstep or range-lockstep allocation.
 * It provides a minimal interface for passing memory reservations between
 * the user and the runtime. It is the sole owner of the memory reservation it
 * represents.
 *
 * **Currently SRAM only**.
 */
struct SRAMArena {
    // No default constructor to be consistent with MeshTensor & MeshBuffer.
    SRAMArena() = delete;
    // Releases the associated reservation
    ~SRAMArena();

    // General move-only RAII resource semantics
    // Disable copy
    SRAMArena(const SRAMArena&) = delete;
    SRAMArena& operator=(const SRAMArena&) = delete;
    // Allow move
    SRAMArena(SRAMArena&&) noexcept;
    SRAMArena& operator=(SRAMArena&&) noexcept;

    /**
     * Reserve SRAM memory at a common address across the entire MeshDevice consistent with given properties (e.g.
     * size).
     *
     * See: LockstepSRAMArenaProperties.
     *
     * Precondition:
     * - properties satisfies its documented invariants.
     *
     * Postconditions:
     * - Returns a valid SRAMArena that owns the memory reservation described by properties.
     * - No other allocation can occupy the reserved memory region until the returned allocation is destroyed.
     */
    static SRAMArena allocate(distributed::MeshDevice& device, LockstepSRAMArenaProperties properties);

    /**
     * Reserve SRAM memory at a common address across the banks specified by the properties
     * across the entire MeshDevice.
     *
     * Precondition:
     * - properties satisfies its documented invariants.
     *
     * Postconditions:
     * - Returns a valid SRAMArena that owns the memory reservation described by properties.
     * - No other allocation can occupy the reserved memory region on the specified banks until the returned
     *   allocation is destroyed.
     */
    static SRAMArena allocate(distributed::MeshDevice& device, RangeLockstepSRAMArenaProperties properties);

    // Opaque handle.

    // Implementation details
    explicit SRAMArena(SRAMArenaImpl&& impl);
    SRAMArenaImpl& impl();
    const SRAMArenaImpl& impl() const;

private:
    std::unique_ptr<SRAMArenaImpl> pimpl_;
};

}  // namespace tt::tt_metal::experimental
