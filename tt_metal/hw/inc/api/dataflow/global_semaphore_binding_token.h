// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Compile-time handle for a GlobalSemaphore binding (a KernelSpec::SemaphoreBinding that names a
// GlobalSemaphoreParameter), emitted into kernel_bindings_generated.h in the `sem::` namespace.
// Carries only the program's GlobalSemaphore slot id; the semaphore's L1 address is read at run time
// from the launch's kernel-config table, so binding a different GlobalSemaphore never recompiles.
//
// Usage example:
//   // (Host code binds the GlobalSemaphoreParameter "ready" to this kernel under accessor name "ready".)
//   GlobalSemaphore ready(sem::ready);
//
// This header holds only the token, with no dependency beyond <cstdint>, so the generated bindings
// header does not have to pull in api/dataflow/global_semaphore.h.
template <uint32_t SLOT_ID>
struct GlobalSemaphoreBindingToken {
    static constexpr uint32_t slot_id = SLOT_ID;
};
