// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <hostdevcommon/sem_scope.h>

// Typed, NON-convertible accessor for a host-baked semaphore (Phase-2 baking).
//
// From Phase-2 S2 on, genfiles emits, per binding:
//     namespace sem { constexpr SemAccessor<<id>, <baked SemScope>> <name>{}; }
// so a kernel writes `Semaphore s(sem::<name>);` and CTAD (the deduction guide in
// noc_semaphore.h) deduces Semaphore<TENSIX, <baked scope>> — the host's chosen
// mechanism, with ZERO kernel-source change.
//
// It deliberately has NO `operator uint32_t()` (unlike the DFB accessor): a baked
// semaphore symbol therefore cannot be turned back into a raw address, so
// `get_semaphore(sem::x)`, `noc_semaphore_inc(get_noc_addr(x, y, sem::x))`, and
// `Semaphore<>(sem::x)` all fail to compile — closing the raw-coordinate back door
// for the managed symbol (the residual runtime-arg-id path is handled host-side).
template <uint32_t Id, SemScope S>
struct SemAccessor {
    static constexpr uint32_t id = Id;
    static constexpr SemScope scope = S;
};
