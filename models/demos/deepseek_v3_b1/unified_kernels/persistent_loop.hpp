// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

namespace deepseek_b1_ops {

// ============================================================================
// PersistentLoop — reusable loop controller for persistent-mode fused ops.
//
// Persistent mode: checks a host-written termination semaphore each iteration.
// On TRISC, drains the PACK coprocessor (tensix_sync) before checking so the
// coprocessor pipeline is clean if the kernel exits.
//
// Non-persistent mode: runs up to max_iterations then exits.
//
// Usage:
//   PersistentLoop<persistent_mode> loop(termination_semaphore_addr);
//   while (loop.next()) {
//       // ... iteration body ...
//   }
// ============================================================================
template <bool persistent_mode>
class PersistentLoop {
public:
    PersistentLoop(uint32_t termination_semaphore_addr, uint32_t max_iterations = 1) :
        termination_semaphore_(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr)),
        max_iterations_(max_iterations),
        iteration_(0) {}

    bool next() {
        if constexpr (persistent_mode) {
#if defined(COMPILE_FOR_TRISC)
            tensix_sync();
#endif
            invalidate_l1_cache();
            if (termination_semaphore_[0] == 1) {
                return false;
            }
        } else {
            if (iteration_ >= max_iterations_) {
                return false;
            }
        }
        iteration_++;
        return true;
    }

    uint32_t iteration() const { return iteration_; }

private:
    volatile tt_l1_ptr uint32_t* termination_semaphore_;
    uint32_t max_iterations_;
    uint32_t iteration_;
};

}  // namespace deepseek_b1_ops
