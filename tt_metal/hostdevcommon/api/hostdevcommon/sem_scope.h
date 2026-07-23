// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Shared host/device home for the Quasar semaphore-scope enums (Phase-2 baking).
// Moved out of api/dataflow/noc_semaphore.h so the host (program_spec / genfiles)
// and the device Semaphore class name the SAME enum. Leaf header: no device- or
// host-only dependencies.

/**
 * @brief Physical path a semaphore's accesses take (Quasar auto-path design).
 *
 * On Quasar the three RMW sources (TRISC / DM / NoC) reach L1 through different
 * physical tiers and no single hardware arbiter serializes them, so a semaphore's
 * *scope* determines which mechanism keeps it correct + atomic:
 *
 *  - LOCAL_NONATOMIC (default): legacy behavior. A plain L1 read-modify-write via
 *      the uncached alias on Quasar. NOT atomic across concurrent writers. Kept as
 *      the default so existing callers are byte-for-byte unchanged. New code should
 *      pick one of the atomic scopes below.
 *
 *  - DM_LOCAL_CACHED: the semaphore is only ever touched by DM cores on this node.
 *      Local increments use a 32-bit RISC-V AMO (amoadd.w) on the *cached* alias,
 *      atomic among the mutually-coherent DM cores and cheapest (no NoC round-trip).
 *      Must never be touched via the NoC / another node — remote ops are a compile
 *      error in this scope. (HW validated: TestDmCachedAmo32.)
 *
 *  - EXTERNAL: the semaphore is touched externally (NoC / another node / chip).
 *      EVERY access — including a local increment — goes through a self-targeted NoC
 *      atomic (NOC_AT_INS_INCR_GET) so local and remote writers serialize at one NIU
 *      atomicity point; local reads use the uncached alias. Correct + atomic; pays a
 *      NoC round-trip. (HW validated: TestSelfTargetedNocAtomicIncrement /
 *      TestSelfVsRemoteNodeNocAtomic.) Cross-domain atomic is increment-first today;
 *      atomic cross-domain decrement/CAS is reachable via the NoC RISCV_AMO/CAS
 *      opcodes (defined in noc_parameters.h, pending emu verification on Quasar).
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
 * binds it, on how many nodes). The explicit values force a specific mechanism and
 * are validated at build time (a contradiction is a host FATAL). Consumed by the
 * Phase-2 host baking pipeline; not used on the device.
 */
enum class SemaphoreScope : uint8_t {
    AUTO = 0,
    LOCAL_NONATOMIC = 1,
    DM_LOCAL_CACHED = 2,
    EXTERNAL = 3,
};
