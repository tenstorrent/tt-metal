// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * The core_sync probe kernel's compile-time core_family argument
 * (tests/device/kernels/core_sync_probe.cpp, compile-time arg 4): the core
 * family a worker runs on, which selects the address space its release
 * semaphore resolves in. The host passes one of these and the kernel decodes
 * it, so both read them from here.
 */
#ifndef X280_CORE_SYNC_CORE_FAMILY_H
#define X280_CORE_SYNC_CORE_FAMILY_H

#define CORE_SYNC_CORE_FAMILY_TENSIX 0u
#define CORE_SYNC_CORE_FAMILY_IDLE_ETH 1u
#define CORE_SYNC_CORE_FAMILY_DRAM 2u
#define CORE_SYNC_CORE_FAMILY_ACTIVE_ETH 3u

#endif /* X280_CORE_SYNC_CORE_FAMILY_H */
