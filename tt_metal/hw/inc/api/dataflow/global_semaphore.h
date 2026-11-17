// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore_binding_token.h"
#include "api/dataflow/semaphore_dm_impl.h"

namespace experimental {

/**
 * @brief Kernel-side view of a GlobalSemaphore bound to a data-movement kernel.
 *
 * A GlobalSemaphore has the same L1 address on every node it spans, on every device of its MeshDevice.
 * Kernels on peer devices signal it remotely (for example, with a fabric atomic increment to address()).
 *
 * Construct one from the binding token your host code declared on the kernel's GlobalSemaphore binding:
 * @code
 *   experimental::GlobalSemaphore signal(global_sem::signal);
 *   signal.wait(1);
 *   signal.set(0);
 * @endcode
 *
 * The local operations read and write this node's copy of the GlobalSemaphore.
 */
class GlobalSemaphore {
public:
    // Create a GlobalSemaphore from its binding token.
    explicit __attribute__((always_inline)) GlobalSemaphore(GlobalSemaphoreBindingToken token) :
        address_(get_common_arg_val<uint32_t>(token.crta_word())) {}

    // Block until this node's copy equals `value`. Does not modify it.
    __attribute__((always_inline)) void wait(std::uint32_t value) const {
        semaphore_detail::wait<ProgrammableCoreType::TENSIX>(address_, kScope, value);
    }

    // Block until this node's copy is at least `value`. Does not modify it.
    __attribute__((always_inline)) void wait_min(std::uint32_t value) const {
        semaphore_detail::wait_min<ProgrammableCoreType::TENSIX>(address_, kScope, value);
    }

    // Set this node's copy to `value`.
    // A non-atomic store: the caller must make sure that no peer signals the GlobalSemaphore at the same time.
    __attribute__((always_inline)) void set(std::uint32_t value) {
        semaphore_detail::set<ProgrammableCoreType::TENSIX>(address_, kScope, value);
    }

    // The current value of this node's copy.
    __attribute__((always_inline)) std::uint32_t value() const {
        return semaphore_detail::current<ProgrammableCoreType::TENSIX>(address_, kScope);
    }

    // The L1 address of the GlobalSemaphore: the same on every node it spans, on every device.
    __attribute__((always_inline)) std::uint32_t address() const { return static_cast<std::uint32_t>(address_); }

private:
    // Peer devices write the word through the NoC, so local accesses use the plain (uncached) L1 word.
    static constexpr SemScope kScope = SemScope::LOCAL_NONATOMIC;
    std::uintptr_t address_;
};

}  // namespace experimental
