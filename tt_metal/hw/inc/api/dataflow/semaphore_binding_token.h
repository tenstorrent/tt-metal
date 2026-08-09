// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <hostdevcommon/sem_scope.h>

// Typed, NON-convertible accessor for a host-baked semaphore.
//
// genfiles emits, per binding:
//     namespace sem { constexpr SemaphoreBindingToken<<id>, <baked SemScope>, <baked SemAccess>> <name>{}; }
// so `Semaphore s(sem::<name>);` deduces the baked scope and access rights via the CTAD guide in
// noc_semaphore.h, with no kernel-source change.
//
// Deliberately NO `operator uint32_t()` (unlike the DFB accessor): the token cannot decay to a
// raw address, so `get_semaphore(sem::x)` and `noc_semaphore_inc(get_noc_addr(x, y, sem::x))`
// fail to compile. An explicit `Semaphore<...>` pin fails the token-ctor static_asserts on any
// scope/access mismatch; a pin that happens to match compiles and is caught by the hygiene
// sweep (test_semaphore_binding_hygiene) instead.
//
// Access is the host's KernelSpec::SemaphoreBinding::access_type, baked whole: every Semaphore
// mutator static_asserts against it (down() needs CONSUME, set() needs SET, up() anything but
// OBSERVE), so the labels the census trusts are the labels the kernel can actually exercise.
// Defaults to INCREMENT so a two-argument token means "plain writer".
template <uint32_t Id, SemScope S, SemAccess Access = SemAccess::INCREMENT>
struct SemaphoreBindingToken {
    static constexpr uint32_t id = Id;
    static constexpr SemScope scope = S;
    // Exposed as members (like id) so a probe kernel can read back what the host actually baked.
    static constexpr SemAccess access = Access;
    static constexpr bool read_only = (Access == SemAccess::OBSERVE);
};
