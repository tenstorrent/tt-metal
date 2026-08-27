// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Exercises every bound DFB, each constructed from its generated dfb::dfb_<n> accessor so the
// interface it touches is the device slot the host assigned rather than the binding's position.
// Explicit sync (WH/BH): one-entry credit handshake with the consumer. Implicit sync (Quasar
// default): probe the config only.
//
// TEST_NUM_DFBS is a host-provided define rather than a compile-time arg: it guards references to
// dfb::dfb_<n> names, which only exist in the generated bindings header for accessors this
// kernel actually binds. Prefixed to avoid colliding with dfb::NUM_DFBS in
// dataflow_buffer_config.h.

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifndef TEST_NUM_DFBS
#error "TEST_NUM_DFBS must be defined by the host (KernelSpec compiler_options.defines)"
#endif

template <bool ImplicitSync>
static inline void touch_one(DFBBindingToken token) {
    DataflowBuffer dfb(token);
    (void)dfb.get_entry_size();
    if constexpr (ImplicitSync) {
        // Probe-only: no credits and no finish(). finish() can spin waiting for ISR/TC drain
        // even when this kernel never posted traffic (common with host-disabled implicit sync).
        (void)dfb.get_id();
    } else {
        dfb.reserve_back(1);
        dfb.push_back(1);
        dfb.finish();
    }
}

void kernel_main() {
    constexpr bool implicit_sync = get_arg(args::implicit_sync);

#if TEST_NUM_DFBS > 0
    touch_one<implicit_sync>(dfb::dfb_0);
#endif
#if TEST_NUM_DFBS > 1
    touch_one<implicit_sync>(dfb::dfb_1);
#endif
#if TEST_NUM_DFBS > 2
    touch_one<implicit_sync>(dfb::dfb_2);
#endif
#if TEST_NUM_DFBS > 3
    touch_one<implicit_sync>(dfb::dfb_3);
#endif
#if TEST_NUM_DFBS > 4
    touch_one<implicit_sync>(dfb::dfb_4);
#endif
#if TEST_NUM_DFBS > 5
    touch_one<implicit_sync>(dfb::dfb_5);
#endif
#if TEST_NUM_DFBS > 6
    touch_one<implicit_sync>(dfb::dfb_6);
#endif
#if TEST_NUM_DFBS > 7
    touch_one<implicit_sync>(dfb::dfb_7);
#endif
#if TEST_NUM_DFBS > 8
    touch_one<implicit_sync>(dfb::dfb_8);
#endif
#if TEST_NUM_DFBS > 9
    touch_one<implicit_sync>(dfb::dfb_9);
#endif
#if TEST_NUM_DFBS > 10
    touch_one<implicit_sync>(dfb::dfb_10);
#endif
#if TEST_NUM_DFBS > 11
    touch_one<implicit_sync>(dfb::dfb_11);
#endif
#if TEST_NUM_DFBS > 12
    touch_one<implicit_sync>(dfb::dfb_12);
#endif
#if TEST_NUM_DFBS > 13
    touch_one<implicit_sync>(dfb::dfb_13);
#endif
#if TEST_NUM_DFBS > 14
    touch_one<implicit_sync>(dfb::dfb_14);
#endif
#if TEST_NUM_DFBS > 15
    touch_one<implicit_sync>(dfb::dfb_15);
#endif
}
