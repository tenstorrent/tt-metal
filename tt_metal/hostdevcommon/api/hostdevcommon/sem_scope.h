// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Shared host/device home for the Quasar semaphore-scope enums, so the host
// (program_spec / genfiles) and the device Semaphore class name the SAME enum.
// Leaf header: no device- or host-only dependencies.

/**
 * @brief Physical path a semaphore's accesses take.
 *
 * On Quasar the three RMW sources (TRISC / DM / NoC) reach L1 through different
 * physical tiers with no single hardware arbiter, so a semaphore's scope picks
 * the mechanism that keeps it atomic:
 *
 *  - LOCAL_NONATOMIC (default): legacy plain L1 read-modify-write (uncached alias
 *      on Quasar). NOT atomic across concurrent writers.
 *
 *  - DM_LOCAL_CACHED: touched only by DM cores on this node. Increments use a
 *      32-bit RISC-V AMO (amoadd.w) on the *cached* alias — atomic among the
 *      node's mutually-coherent DM cores, no NoC round-trip. NoC / remote access
 *      is a compile error in this scope.
 *
 *  - EXTERNAL: touched externally (NoC / another node / chip). up() and down()
 *      go through a self-targeted NoC atomic (INCR_GET; decrement = INCR_GET of a
 *      negative value, wrap=31), serializing local and remote writers at one NIU.
 *      up() is fully atomic; down()'s decrement step is atomic but its
 *      check-then-decrement is single-consumer-only. wait()/wait_min()/value()
 *      and set() use the plain uncached alias — set() is a non-atomic destructive
 *      store, so use it init/reset-only, never concurrently with up()/down().
 */
enum class SemScope : uint8_t {
    LOCAL_NONATOMIC = 0,
    DM_LOCAL_CACHED = 1,
    EXTERNAL = 2,
};

/**
 * @brief Host-side per-semaphore scope INTENT, baked into a SemScope by the host.
 *
 * AUTO = the host derives the effective SemScope from the semaphore's reach (who
 * binds it, on how many nodes). Forcing DM_LOCAL_CACHED is validated at build time
 * (a contradiction is a host FATAL); forcing EXTERNAL or LOCAL_NONATOMIC is a
 * pass-through that also skips AUTO's hazard FATALs. Host-only; not used on the device.
 */
enum class SemaphoreScope : uint8_t {
    AUTO = 0,
    LOCAL_NONATOMIC = 1,
    DM_LOCAL_CACHED = 2,
    EXTERNAL = 3,
};
