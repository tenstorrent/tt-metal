// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Shared host/device home for the semaphore-scope and semaphore-access enums: the host
// (program_spec / genfiles) and the device Semaphore class name the SAME enum. (The host-side
// intent enum, SemaphoreScope, lives in experimental/metal2_host_api/semaphore_spec.hpp.)
// Leaf header: no device- or host-only dependencies.

/**
 * @brief Physical path a semaphore's accesses take.
 *
 * On Quasar the three RMW sources (TRISC / DM / NoC) reach L1 through different physical
 * tiers with no single hardware arbiter, so a semaphore's scope picks the mechanism that
 * keeps it atomic:
 *
 *  - LOCAL_NONATOMIC (default): legacy plain L1 read-modify-write. NOT atomic across
 *      concurrent writers.
 *
 *  - DM_LOCAL_CACHED: touched only by DM cores on this node. RISC-V AMO / LR-SC on the
 *      cached alias — atomic among the node's mutually-coherent DM cores, no NoC
 *      round-trip. NoC / remote access is a compile error in this scope.
 *
 *  - EXTERNAL: touched externally (NoC / another node / chip). up() and down() go through
 *      a self-targeted NoC atomic, serializing local and remote writers at one NIU.
 *      down() is multi-consumer-safe on Quasar DM; on Gen1 (and emule) it remains
 *      single-consumer and the caller owns that invariant. Quasar EXTERNAL reserves the
 *      value 0xFFFFFFFF (CAS-return sentinel). set() is a non-atomic destructive store —
 *      init/reset only, never concurrent with up()/down().
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
 * into the emitted sem:: token; Semaphore mutators static_assert against them. RAW is the
 * raw-id Semaphore(uint32_t) ctor's device-only default (legacy raw-id semaphores are not
 * census-managed).
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
