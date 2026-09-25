// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/firmware_common.h"
#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "hostdev/dev_msgs.h"

#ifdef FDS_SIGNALLING
#include "overlay/fds_signalling.hpp"
#include "quasar/plic.hpp"

constexpr uint32_t fds_num_go_groups = go_message_num_entries - 1;
constexpr uint32_t fds_go_interrupt_mask = overlay::fds_signalling::go_interrupt_mask(fds_num_go_groups);
static_assert(fds_go_interrupt_mask == 0x1FE, "FDS interrupt mask must cover exactly groups 1..8");

__attribute__((interrupt)) inline void fds_go_interrupt_handler() {
    tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(UNCACHED_MEM_MAILBOX_BASE);
    const uint32_t claimed_source = overlay::quasar::plic_claim();
    if (claimed_source == 0) {
        return;
    }

    const uint32_t group_id = overlay::fds_signalling::go_group_from_plic_source(claimed_source);
    if (overlay::fds_signalling::sub_device_from_go_group(group_id) >= fds_num_go_groups) {
        ASSERT(0);
        // Not completing the claim stops the PLIC from delivering this source again.
        return;
    }
    uint32_t dispatch_lanes = overlay::fds_signalling::worker_read_group_status(group_id);
    while (dispatch_lanes != 0) {
        const uint32_t dispatch_lane = __builtin_ctz(dispatch_lanes);
        overlay::fds_signalling::worker_clear_dispatch_status(dispatch_lane);
        dispatch_lanes &= ~(uint32_t{1} << dispatch_lane);
    }

    // Clear the FDS lanes and read them back before completing the claim so the live level cannot re-pend it.
    (void)overlay::fds_signalling::worker_read_group_status(group_id);
    overlay::quasar::plic_complete(claimed_source);

    // The host rewrites go_message_index only after quiesce with no go in flight, so no locking is needed.
    if (group_id == overlay::fds_signalling::go_group_for_sub_device(mailboxes->go_message_index)) {
        mailboxes->go_messages[overlay::fds_signalling::sub_device_from_go_group(group_id)].signal = RUN_MSG_GO;
    }
}

// Brings up the NOC and whichever go-signal transport this build uses, then publishes the
// firmware-ready RUN_MSG_DONE. The host's wait on that DONE is the only barrier before the dispatch
// cores launch, so the FDS body must publish it last: after the go interrupt is armed.
inline void init_go_signalling(tt_l1_ptr mailboxes_t* const mailboxes) {
    noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
    register_handler_for_interrupt(MACHINE_EXTERNAL_INTERRUPT_OFFSET, fds_go_interrupt_handler);
    // Registering a handler stores a jump instruction into the trap vector table, so the icache has
    // to be invalidated before the first go interrupt fetches through that entry.
    invalidate_l1_icache();
    const uint32_t previous_auto_dispatch_cycle_count =
        overlay::fds_signalling::worker_read_auto_dispatch_cycle_count();
    const uint32_t previous_auto_dispatch_enabled = overlay::fds_signalling::worker_read_auto_dispatch_enable();
    overlay::fds_signalling::worker_disable_auto_dispatch();
    overlay::fds_signalling::worker_config_filter_length(overlay::fds_signalling::filter_length_cycles);
    overlay::fds_signalling::worker_config_interrupt_enable(overlay::fds_signalling::interrupts_disabled);
    overlay::fds_signalling::worker_clear_done_direct();
    for (uint32_t dispatch_lane = 0; dispatch_lane < overlay::fds_signalling::num_dispatch_lanes; ++dispatch_lane) {
        overlay::fds_signalling::worker_clear_dispatch_status(dispatch_lane);
    }
    // A previous run that left the pacing count at 0 with auto dispatch enabled releases queued entries only every
    // 2^32 cycles, so draining its queue at init would take up to one more than the number of queued entries,
    // multiplied by 2^32 cycles. We always write a nonzero pacing count before enabling auto dispatch. This assert
    // guards against the case where the pacing count was left at 0 with auto dispatch enabled.
    ASSERT(previous_auto_dispatch_cycle_count != 0 || previous_auto_dispatch_enabled == 0);
    overlay::fds_signalling::wait_cycles(overlay::auto_dispatch_drain_cycles(
        overlay::worker_auto_dispatch_queue_depth, previous_auto_dispatch_cycle_count));
    WAYPOINT("FACW");
    overlay::fds_signalling::worker_config_auto_dispatch_pacing(
        overlay::fds_signalling::worker_auto_dispatch_pacing_cycle_count);
    overlay::fds_signalling::worker_config_auto_dispatch_outbox(TT_FDS_TENSIXNEO_TENSIX_TO_DISPATCH_REG_ADDR);
    overlay::fds_signalling::worker_enable_auto_dispatch();
    WAYPOINT("FACD");
    for (uint32_t go_group_id = overlay::fds_signalling::idle_group_id + 1; go_group_id <= fds_num_go_groups;
         ++go_group_id) {
        overlay::fds_signalling::worker_config_group(
            go_group_id, overlay::fds_signalling::dispatch_lane_mask, overlay::fds_signalling::worker_go_threshold);
    }
    overlay::quasar::plic_set_threshold(overlay::quasar::plic_threshold_allow_all);
    for (uint32_t go_group_id = overlay::fds_signalling::idle_group_id + 1; go_group_id <= fds_num_go_groups;
         ++go_group_id) {
        const uint32_t plic_source = overlay::fds_signalling::plic_source_for_go_group(go_group_id);
        overlay::quasar::plic_set_priority(plic_source, overlay::fds_signalling::plic_fds_priority);
    }
    overlay::quasar::plic_enable_only_sources(
        overlay::fds_signalling::plic_source_for_go_group(overlay::fds_signalling::idle_group_id + 1),
        overlay::fds_signalling::plic_source_for_go_group(overlay::fds_signalling::idle_group_id + fds_num_go_groups));
    overlay::quasar::plic_drain_pendings();
    // Thresholds and PLIC enables must be set before arming the FDS interrupt level at reset.
    overlay::fds_signalling::worker_config_interrupt_enable(fds_go_interrupt_mask);
    asm volatile("csrrs zero, mie, %0" : : "r"(uint32_t{1} << MACHINE_EXTERNAL_INTERRUPT_OFFSET));
    asm volatile("csrrs zero, mstatus, %0" : : "r"(uint32_t{1} << 3));
    mailboxes->go_messages[0].signal = RUN_MSG_DONE;
}

// Readies this launch's FDS done: waits for the go if asked, then queues idle on the done wire so the
// previous launch's done is not taken for this one. Returns the done group (sub-device index + 1), or 0
// when the launch completes over the NOC instead.
inline uint32_t prepare_worker_completion_signal(
    tt_l1_ptr mailboxes_t* const mailboxes, launch_msg_t* launch_message, bool wait_for_go) {
    if (launch_message->kernel_config.mode != DISPATCH_MODE_DEV) {
        return 0;
    }

    const uint32_t go_message_index = mailboxes->go_message_index;
    if (wait_for_go) {
        WAYPOINT("FGW");
        while (mailboxes->go_messages[go_message_index].signal != RUN_MSG_GO);
        WAYPOINT("FGD");
    }

    overlay::fds_signalling::worker_clear_done();
    return go_message_index + 1;
}

// Sends this launch's done to dispatch by queueing its group on the FDS done wire, where it stays until
// the next launch clears it.
inline void signal_worker_completion(uint32_t worker_completion_group) {
    overlay::fds_signalling::worker_signal_done(worker_completion_group);
}
#else
inline void init_go_signalling(tt_l1_ptr mailboxes_t* const mailboxes) {
    mailboxes->go_messages[0].signal = RUN_MSG_DONE;
    noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
}

inline uint32_t prepare_worker_completion_signal(tt_l1_ptr mailboxes_t* const, launch_msg_t*, bool) { return 0; }

inline void signal_worker_completion(uint32_t) { ASSERT(0); }
#endif
