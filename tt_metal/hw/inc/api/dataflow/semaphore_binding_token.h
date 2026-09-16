// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

/**
 * @brief Physical path a semaphore's accesses take. The host picks it automatically while
 *        building the program, from where the semaphore's binder kernels run; there is no user
 *        intervention needed. The kernel gets the pick inside its binding token. The pick is
 *        the fastest path that keeps the semaphore's operations atomic.
 *        The cached and NoC tiers are Quasar (tt-2xx) only. Gen1 (Wormhole, Blackhole) DM
 *        bindings always resolve to LOCAL_NONATOMIC; a Blackhole compute binding resolves to
 *        COMPUTE_ATOMIC.
 *
 *  - LOCAL_NONATOMIC: Stored in L1 and accessed by read-modify-write. Picked only when at most
 *                     one binder instance exists.
 *  - DM_LOCAL_CACHED: Stored in a dedicated L1 pool and accessed through the DM cache via
 *                     RISC-V AMO. Picked only when all binders are DMs on the same node where
 *                     the semaphore exists. The pool is separate so a cached AMO's cache-line
 *                     write-back cannot clobber NoC-written data: see MEM_DM_CACHED_SEM_BASE
 *                     in dev_mem_map.h.
 *  - EXTERNAL:        Stored in L1 and accessed through atomic operations via the NOC. Picked
 *                     whenever the semaphore is reachable beyond a single node.
 *  - COMPUTE_ATOMIC:  Blackhole only. The Tensix hardware (Sync Unit) semaphore, updated
 *                     atomically by SEMPOST/SEMGET, so UNPACK and PACK cannot lose each other's
 *                     updates. Picked when every binder is a compute kernel; a semaphore bound by
 *                     both compute and DM kernels is rejected, and so is a second compute semaphore in
 *                     one program (only one hardware semaphore is free). Core-local, not a NoC atomic.
 *                     Always starts at 0 (seeded on the device by compute_kernel_hw_startup); a nonzero
 *                     initial_value is rejected. Its capacity is SemaphoreAdvancedOptions::max_value
 *                     (1..15, default 15), baked into the compute kernel as COMPUTE_SEMAPHORE_MAX for
 *                     Semaphore::wait_not_full().
 *
 * @note Never access a bound semaphore's word directly (get_semaphore(), the noc_semaphore_*
 *       free functions, raw pointers), always go through the Semaphore class. A raw access is
 *       invisible to the host's mechanism choice and can silently race it.
 */
enum class SemScope : uint8_t {
    LOCAL_NONATOMIC = 0,
    DM_LOCAL_CACHED = 1,
    EXTERNAL = 2,
    COMPUTE_ATOMIC = 3,
};

/**
 * @brief Per-binding token for a semaphore, emitted into the generated kernel header.
 *
 * Carries everything the host resolved for this binding: the semaphore id and the mechanism
 * its accesses must use. A Semaphore is constructed from the token, which is how the
 * mechanism reaches the kernel as a compile-time property.
 */
template <std::uint32_t SEM_ID, SemScope SEM_SCOPE>
struct SemaphoreBindingToken {
    static constexpr std::uint32_t id = SEM_ID;
    static constexpr SemScope scope = SEM_SCOPE;
};
