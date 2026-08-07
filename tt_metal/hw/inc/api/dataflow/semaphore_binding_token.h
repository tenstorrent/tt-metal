// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <hostdevcommon/sem_scope.h>

// Typed, NON-convertible accessor for a host-baked semaphore (Phase-2 baking).
//
// From Phase-2 S2 on, genfiles emits, per binding:
//     namespace sem { constexpr SemaphoreBindingToken<<id>, <baked SemScope>> <name>{}; }
// so a kernel writes `Semaphore s(sem::<name>);` and CTAD (the deduction guide in
// noc_semaphore.h) deduces Semaphore<TENSIX, <baked scope>> — the host's chosen
// mechanism, with ZERO kernel-source change.
//
// It deliberately has NO `operator uint32_t()` (unlike the DFB accessor): a baked
// semaphore symbol therefore cannot be turned back into a raw address, so
// `get_semaphore(sem::x)`, `noc_semaphore_inc(get_noc_addr(x, y, sem::x))`, and
// `Semaphore<>(sem::x)` all fail to compile — closing the raw-coordinate back door
// for the managed symbol (the residual runtime-arg-id path is handled host-side).
// ReadOnly mirrors the host's KernelSpec::SemaphoreBinding::access_type collapsed to ONE bit:
// AccessType::OBSERVE => true; INCREMENT / CONSUME / SET => false. A single bool rather than the
// full four-value AccessType is deliberate -- "a binding declared OBSERVE must not mutate" is the
// one unambiguous rule, and it is the rule with teeth: an OBSERVE binding is EXCLUDED from
// ResolveSemaphoreScope's writer census, so a dishonest OBSERVE can downgrade a genuinely contended
// semaphore to the non-atomic LOCAL_NONATOMIC mechanism. Distinguishing INCREMENT from SET from
// CONSUME would over-constrain legitimate kernels while protecting no host decision.
//
// Defaulted to false so a two-argument token -- what the emitters wrote before ReadOnly existed --
// still means "mutable", i.e. exactly the previous behaviour.
template <uint32_t Id, SemScope S, bool ReadOnly = false>
struct SemaphoreBindingToken {
    static constexpr uint32_t id = Id;
    static constexpr SemScope scope = S;
    // True iff the host declared this binding AccessType::OBSERVE. Every Semaphore MUTATOR
    // static_asserts on it; wait()/wait_min()/value() are unaffected. Exposed as a member (like id
    // and scope) so a probe kernel can read the bit back and prove the host actually baked it --
    // without that, forgetting to emit it would leave the enforcement silently inert.
    static constexpr bool read_only = ReadOnly;
};
