// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch-engine side of the outbox address-convention test. The auto dispatch trigger compares
// the full untruncated write address against the outbox register, while ordinary register decode
// uses only the low nine bits — so the OFFSET-form macro and the ADDR-form macro alias for every
// register access except this one. fds_go writes through the ADDR form; an outbox programmed with
// the OFFSET form therefore never matches, the write falls through into the (deselected) output
// register, and nothing reaches the wire. This test pins that silent failure mode and then shows
// the matching ADDR form delivering. The assertions live in quasar_fds_outbox_mismatch_worker.cpp.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

using fds_outbox::kMatchedGo;
using fds_outbox::kMismatchedGo;
using fds_outbox::kTokenArmed;
using fds_outbox::kTokenDelivered;
using fds_outbox::kTokenSilenceChecked;

constexpr uint32_t kNumSlots = 1;
constexpr uint32_t kTimeoutArmed = 0x5A5A0006;
constexpr uint32_t kTimeoutSilence = 0x5A5A0007;
constexpr uint32_t kTimeoutDelivered = 0x5A5A0008;

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t worker_mask = get_named_compile_time_arg_val("worker_mask");
    constexpr uint32_t auto_dispatch_cycles = get_named_compile_time_arg_val("auto_dispatch_cycles");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");
    static_assert(kTokenDelivered < kReadyTokenA, "step tokens must stay below the ready tokens");

    fds_kernel::status_ptr status = fds_kernel::begin_dispatch(l1_address, kNumSlots);
    overlay::FdsDispatch::fds_config_groupid(kTokenArmed, worker_mask, 1);
    overlay::FdsDispatch::fds_config_groupid(kTokenSilenceChecked, worker_mask, 1);
    overlay::FdsDispatch::fds_config_groupid(kTokenDelivered, worker_mask, 1);

    if (!fds_kernel::workers_are_ready(status, l1_address, kNumSlots, worker_mask, kNumWorkers, poll_iterations)) {
        return;
    }

    overlay::FdsDispatch::fds_clear_go();
    overlay::FdsDispatch::fds_go(/*ad_enable=*/false, kSessionGo);

    uint32_t result = kComplete;
    if (!fds_kernel::wait_group_count_nonzero(kTokenArmed, poll_iterations)) {
        result = kTimeoutArmed;
    }

    if (result == kComplete) {
        // The outbox holds the OFFSET-form address; the write below goes to the ADDR form. The
        // trigger comparison must miss, so nothing may reach the wire.
        overlay::FdsDispatch::fds_config_auto_dispatch(
            /*enable=*/true, auto_dispatch_cycles, TT_FDS_DISPATCH_DISPATCH_TO_TENSIX_REG_OFFSET);
        overlay::FdsDispatch::fds_go(/*ad_enable=*/true, kMismatchedGo);

        if (!fds_kernel::wait_group_count_nonzero(kTokenSilenceChecked, poll_iterations)) {
            result = kTimeoutSilence;
        }
    }

    if (result == kComplete) {
        // The same write with the outbox holding the ADDR form: now the trigger matches and the
        // value must be delivered.
        overlay::FdsDispatch::fds_config_auto_dispatch(
            /*enable=*/true, auto_dispatch_cycles, TT_FDS_DISPATCH_DISPATCH_TO_TENSIX_REG_ADDR);
        overlay::FdsDispatch::fds_go(/*ad_enable=*/true, kMatchedGo);

        if (!fds_kernel::wait_group_count_nonzero(kTokenDelivered, poll_iterations)) {
            result = kTimeoutDelivered;
        }
    }

    // Back to the direct path; the output register collected the mismatched write, so clear it
    // once the direct path is active again.
    overlay::FdsDispatch::fds_config_auto_dispatch(/*enable=*/false, 0, 0);
    overlay::FdsDispatch::fds_clear_go();

    fds_kernel::finish(status, l1_address, kNumSlots, result);
}
