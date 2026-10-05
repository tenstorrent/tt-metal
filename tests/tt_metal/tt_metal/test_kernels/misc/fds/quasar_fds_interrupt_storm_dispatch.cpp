// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The reset state is a firing state. Out of reset every threshold and every count is zero, so a
// group left at its reset threshold has count 0 equal to threshold 0, and arming it raises the
// level with no traffic on any wire — no workers run in this epoch at all. That is the hazard for
// anything that arms before it configures.
//
// The count reached that way has to be built rather than assumed: input registers keep their
// captures across launches, and the workers of every other test drive this same group id and never
// clear it at exit. So this kernel sheds the inherited captures first, exactly as the kernels with
// a ready wait do, and only then is the count zero because nothing has signalled.
//
// The second half is what makes it more than a curiosity: the reflex that quiets every other
// group, clearing the input registers, cannot quiet this one. The count is already zero, so
// lowering it is not available, and the handler's first entry proves it by clearing everything,
// reading the count back, and being re-entered anyway. Only dropping the interrupt enable bit
// takes this level down, which the second entry does.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"
#include "quasar_fds_interrupt.h"

using fds_interrupt_status::kSlotInterruptCount;

// Mirrored by test_quasar_fds.cpp.
constexpr uint32_t kNumSlots = 2;

// Arming a group whose count and threshold are both zero delivered nothing.
constexpr uint32_t kTimeoutInterrupt = fds_interrupt_status::kTimeoutInterrupt;
// Clearing the input registers took the level down, which it cannot do to a count already at zero.
constexpr uint32_t kQuietedByClear = 0x5A5A0077;
// The enable bit was dropped and deliveries kept coming.
constexpr uint32_t kDisarmDidNotQuiet = 0x5A5A0078;

constexpr uint32_t kL1Address = get_named_compile_time_arg_val("l1_address");
constexpr uint32_t kGroupId = get_named_compile_time_arg_val("group_id");
constexpr uint32_t kWorkerMask = get_named_compile_time_arg_val("worker_mask");
constexpr uint32_t kPollIterations = get_named_compile_time_arg_val("poll_iterations");
constexpr uint32_t kSilenceIterations = get_named_compile_time_arg_val("silence_iterations");

// The reset value, written out rather than inherited so the firing condition is stated where the
// test rests on it.
constexpr uint32_t kResetThreshold = 0;

static_assert(kGroupId < kReadyTokenA, "payload group ids must stay below the ready tokens");

__attribute__((interrupt)) void fds_storm_interrupt_handler() {
    fds_kernel::status_ptr status = reinterpret_cast<fds_kernel::status_ptr>(kL1Address);
    const uint32_t context = fds_interrupt::read_mhartid();
    const uint32_t claimed = fds_interrupt::plic_claim(context);
    const uint32_t entry = status[kSlotInterruptCount];

    if (entry == 0) {
        // Every input register, not just the lanes carrying this group: none of them carries it,
        // which is the point. The count read back is what makes the failure below a measurement
        // rather than an assumption.
        fds_epoch::clear_dispatch_inputs(kWorkerMask);
        overlay::FdsDispatch::fds_read_group_count(kGroupId);
    } else {
        // The one write that does take this level down, followed by the same read back the other
        // handlers make after their clears: the completion below must not outrun it, or the source
        // re-pends and the disarm looks like it failed.
        overlay::FdsDispatch::fds_config_interrupt_en(0);
        overlay::FdsDispatch::fds_read_group_count(kGroupId);
    }
    fds_interrupt::order_fds_before_status();

    fds_interrupt::stop_interrupt_storm(context, claimed, entry);
    fds_interrupt::plic_complete(context, claimed);
    status[kSlotInterruptCount] = entry + 1;
}

void kernel_main() {
    fds_kernel::status_ptr status = fds_kernel::begin_dispatch(kL1Address, kNumSlots);
    fds_interrupt::clear_handler_slots(status, kSlotInterruptCount, kNumSlots);
    // A done captured from an earlier launch carries this group id and would hold the count above
    // the threshold, so arming would deliver nothing and the reset state would look quiet.
    fds_epoch::clear_dispatch_inputs(kWorkerMask);
    overlay::FdsDispatch::fds_read_group_count(kGroupId);
    // A full enable mask and a threshold of zero: every lane is watched and none of them has to do
    // anything for the comparison to hold.
    overlay::FdsDispatch::fds_config_groupid(kGroupId, kWorkerMask, kResetThreshold);

    fds_interrupt::arming_state arming;
    if (!fds_interrupt::arm_external_interrupt(uint32_t{1} << kGroupId, fds_storm_interrupt_handler, arming)) {
        fds_kernel::finish(status, kL1Address, kNumSlots, fds_interrupt_status::kBadHartContext);
        return;
    }

    // No ready handshake and no go: there are no workers in this epoch. The enable bit alone is
    // the whole stimulus.
    overlay::FdsDispatch::fds_config_interrupt_en(uint32_t{1} << kGroupId);

    uint32_t result = kComplete;
    if (!fds_interrupt::wait_for_interrupt_count(status, 1, kPollIterations)) {
        result = kTimeoutInterrupt;
    }

    if (result == kComplete && !fds_interrupt::wait_for_interrupt_count(status, 2, kPollIterations)) {
        result = kQuietedByClear;
    }

    if (result == kComplete && !fds_interrupt::interrupt_count_steady(status, 2, kSilenceIterations)) {
        result = kDisarmDidNotQuiet;
    }

    fds_interrupt::dispatch::disarm_external_interrupt(arming);

    fds_kernel::finish(status, kL1Address, kNumSlots, result);
}
