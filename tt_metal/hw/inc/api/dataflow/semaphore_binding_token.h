// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <hostdevcommon/sem_scope.h>

// Typed, NON-convertible accessor for a host-baked semaphore.
//
// genfiles emits, per binding:
//     namespace sem { constexpr SemaphoreBindingToken<<id>, <baked SemScope>, <read-only>> <name>{}; }
// so `Semaphore s(sem::<name>);` deduces the baked scope and access rights via the CTAD guide in
// noc_semaphore.h, with no kernel-source change.
//
// Deliberately NO `operator uint32_t()` (unlike the DFB accessor): the token cannot decay to a
// raw address, so `get_semaphore(sem::x)` and `noc_semaphore_inc(get_noc_addr(x, y, sem::x))`
// fail to compile. An explicit `Semaphore<...>` pin fails the token-ctor static_asserts on any
// scope/read-only mismatch; a pin that happens to match compiles and is caught by the hygiene
// sweep (test_semaphore_binding_hygiene) instead.
//
// ReadOnly is the host's KernelSpec::SemaphoreBinding::access_type collapsed to one bit
// (AccessType::OBSERVE => true). OBSERVE bindings are excluded from ResolveSemaphoreScope's
// writer census, so a write through one could leave a contended semaphore on the non-atomic
// path -- hence every Semaphore mutator static_asserts on it. Defaults to false so a
// two-argument token means "mutable".
template <uint32_t Id, SemScope S, bool ReadOnly = false>
struct SemaphoreBindingToken {
    static constexpr uint32_t id = Id;
    static constexpr SemScope scope = S;
    // True iff the host declared this binding AccessType::OBSERVE. Exposed as a member (like id
    // and scope) so a probe kernel can read back what the host actually baked.
    static constexpr bool read_only = ReadOnly;
};
