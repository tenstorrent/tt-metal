// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch-engine side of the Quasar FDS go/done handshake: write a payload to L1, send an FDS go
// signal to the worker NEOs selected by worker_mask, then wait for the worker's done signal.
// The worker side lives in quasar_fds_worker_signal.cpp.
//
// The FDS wire index of any given worker core is not established, so the host aims the go at every
// NEO wire at once rather than at a chosen one.

#include "risc_attribs.h"
#include "risc_common.h"
#include "api/compile_time_args.h"
#include "overlay/fds_functions.hpp"

#include "quasar_fds_signal_status.h"

namespace {

// One TENSIX_TO_DISPATCH inbox register per NEO wire on the dispatch side.
constexpr uint32_t kNumNeoWires = 32;

// A done must be stable for a single cycle to be accepted. This is the register's reset value,
// written explicitly so the handshake does not depend on the state a previous program left behind.
constexpr uint32_t kNoDeglitchFilter = 0;

}  // namespace

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t worker_mask = get_named_compile_time_arg_val("worker_mask");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");

    // One worker kernel runs in this test, so exactly one NEO can ever drive a done and the group
    // count tops out at one. Any larger threshold makes the wait unsatisfiable.
    constexpr uint32_t done_threshold = 1;

    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_address);
    status[quasar_fds_test::kSlotStarted] = quasar_fds_test::kStarted;
    // Commit the payload to node memory before signalling, so the go signal cannot be observed
    // ahead of the data it is announcing.
    flush_l2_cache_range(l1_address, quasar_fds_test::kNumSlots * sizeof(uint32_t));

    overlay::FdsDispatch::fds_config_filter_length(kNoDeglitchFilter);
    overlay::FdsDispatch::fds_config_groupid(group_id, worker_mask, done_threshold);

    // A done is a held level rather than a pulse, so drop any value a worker was still driving from
    // an earlier epoch. Every wire is cleared because the worker's wire index is not established.
    for (uint32_t neo = 0; neo < kNumNeoWires; neo++) {
        overlay::FdsDispatch::fds_clear_neo_status(neo);
    }

    overlay::FdsDispatch::fds_go(/*ad_enable=*/false, group_id);

    // Bounded instead of fds_poll(): a done that never arrives must fail the test with a readable
    // status word rather than hang it.
    uint32_t done_count = 0;
    for (uint32_t i = 0; i < poll_iterations; i++) {
        done_count = overlay::FdsDispatch::fds_read_group_count(group_id);
        if (done_count >= done_threshold) {
            break;
        }
    }

    status[quasar_fds_test::kSlotObserved] = done_count;
    status[quasar_fds_test::kSlotResult] =
        (done_count >= done_threshold) ? quasar_fds_test::kComplete : quasar_fds_test::kTimeout;
    flush_l2_cache_range(l1_address, quasar_fds_test::kNumSlots * sizeof(uint32_t));
}
