// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The FDS wire index of any given worker core is not established, so the host aims the go at every
// NEO wire at once rather than at a chosen one.
//
// The opening ready wait, and what it buys, is described in quasar_fds_epoch.h.
//
// quiet_group_mask names the groups configured exactly like the signalled one but never sent a go,
// one bit per group id. A group nobody signalled must not accumulate credit from one that was, so
// the first such group found holding a done count is reported at the end.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

// Mirrored by test_quasar_fds.cpp.
constexpr uint32_t kSlotDoneCount = 1;
// The lowest-numbered quiet group that was credited a done anyway, so a leak names the group it
// landed in. Zero means none was, group 0 being the idle value on the wire and so never a group a
// worker can belong to.
constexpr uint32_t kSlotCreditedQuietGroup = 2;
constexpr uint32_t kNumSlots = 3;

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t worker_mask = get_named_compile_time_arg_val("worker_mask");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");
    // One per worker kernel the host launched. A threshold above that count makes the wait
    // unsatisfiable, since only that many NEOs can ever drive a done.
    constexpr uint32_t done_threshold = get_named_compile_time_arg_val("done_threshold");
    // Every worker kernel in the epoch, whatever its group: all of them signal ready, so all of
    // them gate the go.
    constexpr uint32_t num_workers = get_named_compile_time_arg_val("num_workers");
    // Zero means no quiet groups. Bit 0 is never set, group 0 being the reserved idle value on the
    // wire.
    constexpr uint32_t quiet_group_mask = get_named_compile_time_arg_val("quiet_group_mask");
    static_assert(group_id < kReadyTokenA, "payload group ids must stay below the ready tokens");
    static_assert(done_threshold > 0, "a zero threshold would satisfy the done wait with no dones at all");

    fds_kernel::status_ptr status = fds_kernel::begin_dispatch(l1_address, kNumSlots);
    overlay::FdsDispatch::fds_config_groupid(group_id, worker_mask, done_threshold);
    // Each quiet group watches the same lanes with a threshold of one, so a single stray done would
    // satisfy it. Nothing polls them; their counts are read at the end as evidence they stayed at
    // zero.
    for (uint32_t mask = quiet_group_mask, group = 0; mask != 0; mask >>= 1, group++) {
        if (mask & 1u) {
            overlay::FdsDispatch::fds_config_groupid(group, worker_mask, 1);
        }
    }

    if (!fds_kernel::workers_are_ready(status, l1_address, kNumSlots, worker_mask, num_workers, poll_iterations)) {
        return;
    }

    // Clearing first makes the go a guaranteed change on the wire even when a previous epoch of
    // this engine sent the same group id.
    overlay::FdsDispatch::fds_clear_go();
    overlay::FdsDispatch::fds_go(/*ad_enable=*/false, group_id);

    uint32_t done_count = 0;
    fds_kernel::wait_group_count(group_id, done_threshold, poll_iterations, done_count);

    uint32_t credited_quiet_group = 0;
    for (uint32_t mask = quiet_group_mask, group = 0; mask != 0; mask >>= 1, group++) {
        if ((mask & 1u) != 0 && overlay::FdsDispatch::fds_read_group_count(group) != 0) {
            credited_quiet_group = group;
            break;
        }
    }

    overlay::FdsDispatch::fds_clear_go();

    status[kSlotDoneCount] = done_count;
    status[kSlotCreditedQuietGroup] = credited_quiet_group;
    fds_kernel::finish(status, l1_address, kNumSlots, (done_count >= done_threshold) ? kComplete : kTimeout);
}
