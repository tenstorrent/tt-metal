// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Shared host/device home for the semaphore-scope and semaphore-access enums. (The host-side
// intent enum, SemaphoreScope, lives in experimental/metal2_host_api/semaphore_spec.hpp.)
// The host (program_spec / genfiles) and the device Semaphore class name the SAME enum.
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
 *      32-bit RISC-V AMO (amoadd.w) and down() an LR/SC conditional-decrement loop
 *      (multi-consumer-safe), both on the *cached* alias — atomic among the node's
 *      mutually-coherent DM cores, no NoC round-trip. NoC / remote access is a
 *      compile error in this scope.
 *
 *  - EXTERNAL: touched externally (NoC / another node / chip). up() and down()
 *      go through a self-targeted NoC atomic (INCR_GET; decrement = INCR_GET of a
 *      negative value, wrap=31), serializing local and remote writers at one NIU.
 *      up() is fully atomic. down() is multi-consumer-safe on Quasar DM (a NoC-CAS
 *      lock serializes consumers; producers' increments commute); on Gen1 it remains
 *      single-consumer and the caller owns that invariant (the emulator also compiles
 *      the Gen1 arm, so emule rejects multi-consumer EXTERNAL at config time). Quasar
 *      EXTERNAL reserves the value 0xFFFFFFFF (CAS-return sentinel).
 *      wait()/wait_min()/value() and set() use the plain uncached alias — set() is a
 *      non-atomic destructive store, so use it init/reset-only, never concurrently
 *      with up()/down().
 */
enum class SemScope : uint8_t {
    LOCAL_NONATOMIC = 0,
    DM_LOCAL_CACHED = 1,
    EXTERNAL = 2,
};

/**
 * @brief What a binding is allowed to DO to its semaphore, compile-time enforced.
 *
 * The first four mirror the host's KernelSpec::SemaphoreBinding::AccessType 1:1 and are baked
 * into the emitted sem:: token; every Semaphore mutator static_asserts against them (zero
 * runtime cost). RAW is device-only: the raw-id Semaphore(uint32_t) ctor's default, keeping
 * every legacy raw-id call site legal (those semaphores are not census-managed).
 *
 *  - INCREMENT: up() / inc_multicast()
 *  - CONSUME:   INCREMENT + down() (the off-node-consumer rejection keys on this label)
 *  - SET:       INCREMENT + set() / set_multicast() / relay destination
 *  - OBSERVE:   wait() / wait_min() / value() only -- pure reader
 *  - RAW:       everything (raw-id construction; never emitted by the host)
 */
enum class SemAccess : uint8_t {
    INCREMENT = 0,
    CONSUME = 1,
    SET = 2,
    OBSERVE = 3,
    RAW = 4,
};
