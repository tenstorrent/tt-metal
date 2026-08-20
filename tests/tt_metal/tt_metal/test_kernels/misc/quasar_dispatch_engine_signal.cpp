// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch-engine side of the Quasar FDS go/done handshake: write a payload to L1, send an FDS go
// signal to the worker NEOs selected by worker_mask, then wait until done_threshold workers have
// signalled done. The worker side lives in quasar_fds_worker_signal.cpp.
//
// The FDS wire index of any given worker core is not established, so the host aims the go at every
// NEO wire at once rather than at a chosen one.
//
// quiet_group_id, when non-zero, names a second group configured exactly like the signalled one but
// never sent a go. Its done count is reported at the end: a group nobody signalled must not
// accumulate credit from one that was.

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
    // One per worker kernel the host launched. A threshold above that count makes the wait
    // unsatisfiable, since only that many NEOs can ever drive a done.
    constexpr uint32_t done_threshold = get_named_compile_time_arg_val("done_threshold");
    // Zero means no second group, group 0 being the reserved idle value on the wire.
    constexpr uint32_t quiet_group_id = get_named_compile_time_arg_val("quiet_group_id");

    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_address);
    status[quasar_fds_test::kSlotStarted] = quasar_fds_test::kStarted;
    // Make the status word visible to the host, which reads it after the program completes. This is
    // not a demonstration of ordering against the go signal: the write is local to this core and
    // nothing here reads it on observing a signal. See quasar_fds_worker_ordered_write.cpp for the
    // test that does exercise that.
    flush_l2_cache_range(l1_address, quasar_fds_test::kNumSlots * sizeof(uint32_t));

    overlay::FdsDispatch::fds_config_filter_length(kNoDeglitchFilter);
    overlay::FdsDispatch::fds_config_groupid(group_id, worker_mask, done_threshold);
    // The quiet group watches the same lanes with a threshold of one, so a single stray done would
    // satisfy it. Nothing polls it; its count is read at the end as evidence it stayed at zero.
    if (quiet_group_id != 0) {
        overlay::FdsDispatch::fds_config_groupid(quiet_group_id, worker_mask, 1);
    }

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
    status[quasar_fds_test::kSlotQuietGroupCount] =
        (quiet_group_id != 0) ? overlay::FdsDispatch::fds_read_group_count(quiet_group_id) : 0;
    status[quasar_fds_test::kSlotResult] =
        (done_count >= done_threshold) ? quasar_fds_test::kComplete : quasar_fds_test::kTimeout;
    flush_l2_cache_range(l1_address, quasar_fds_test::kNumSlots * sizeof(uint32_t));
}
