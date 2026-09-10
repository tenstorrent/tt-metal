// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The worker side of interrupt delivery, and the only test in this suite on the NEO register map's
// interrupt path. A worker that polls for its go can miss one, so the specification has workers run
// with interrupts armed; here the go is collected by a handler and the lane is never polled.
//
// Arming happens before the first ready pulse. The initial handshake is the one moment the engine
// can raise a go, so a worker that armed after pulsing would have been a poller for exactly the
// window in which the go can arrive. The epoch protocol already orders it that way — clear inputs,
// then pulse — and this kernel arms inside that order rather than alongside it.
//
// fds_epoch::wait_for_go cannot serve as the wait: it polls the lane, which is the mechanism under
// test. The loop below keeps the protocol's ready pulsing and takes its completion condition from
// the handler's count instead.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"
#include "quasar_fds_interrupt.h"

using fds_interrupt_status::kSlotClaimedSource;
using fds_interrupt_status::kSlotInterruptCount;

// Mirrored by test_quasar_fds.cpp. Which dispatch instance drove this NEO, as the handler found it.
constexpr uint32_t kSlotGoLane = 3;
constexpr uint32_t kNumSlots = 4;

constexpr uint32_t kL1Address = get_named_compile_time_arg_val("l1_address");
constexpr uint32_t kGroupId = get_named_compile_time_arg_val("group_id");
constexpr uint32_t kDispatchMask = get_named_compile_time_arg_val("dispatch_mask");
constexpr uint32_t kPollIterations = get_named_compile_time_arg_val("poll_iterations");

// One engine drives this worker, so one go on any lane is the whole count.
constexpr uint32_t kGoThreshold = 1;

static_assert(kGroupId < kReadyTokenA, "payload group ids must stay below the ready tokens");

__attribute__((interrupt)) void fds_go_interrupt_handler() {
    fds_kernel::status_ptr status = reinterpret_cast<fds_kernel::status_ptr>(kL1Address);
    const uint32_t context = fds_interrupt::read_mhartid();
    const uint32_t claimed = fds_interrupt::plic_claim(context);
    const uint32_t entry = status[kSlotInterruptCount];

    status[kSlotClaimedSource] = claimed;

    // The engine holds its go, so the level stays up until the input register carrying it is
    // cleared. Which dispatch instance drives this NEO is not established, so every lane the host
    // named is scanned rather than a chosen one.
    for (uint32_t mask = kDispatchMask, inst = 0; mask != 0; mask >>= 1, inst++) {
        if ((mask & 1u) != 0 && overlay::FdsNeo::fds_read_de_status(inst) == kGroupId) {
            status[kSlotGoLane] = inst;
            overlay::FdsNeo::fds_clear_de_status(inst);
        }
    }
    fds_interrupt::neo::wait_status_cleared(kGroupId);
    fds_interrupt::order_fds_before_status();

    fds_interrupt::stop_interrupt_storm(context, claimed, entry);
    fds_interrupt::plic_complete(context, claimed);
    status[kSlotInterruptCount] = entry + 1;
}

void kernel_main() {
    fds_kernel::status_ptr status = fds_kernel::begin_worker(kL1Address, kNumSlots);
    fds_interrupt::clear_handler_slots(status, kSlotInterruptCount, kNumSlots);
    overlay::FdsNeo::fds_config_groupid(kGroupId, kDispatchMask, kGoThreshold);

    fds_interrupt::arming_state arming;
    if (!fds_interrupt::arm_external_interrupt(uint32_t{1} << kGroupId, fds_go_interrupt_handler, arming)) {
        fds_kernel::finish(status, kL1Address, kNumSlots, fds_interrupt_status::kBadHartContext);
        return;
    }

    // Inputs are shed before the enable bit goes on: a go inherited from a previous epoch would
    // otherwise stand at the threshold the moment this group is armed, and the interrupt would
    // report the last epoch's signal.
    fds_epoch::clear_worker_inputs(kDispatchMask);
    overlay::FdsNeo::fds_config_interrupt_en(uint32_t{1} << kGroupId);

    uint32_t ready_token = kReadyTokenA;
    bool go_received = false;
    for (uint32_t i = 0; i < kPollIterations && !go_received; i++) {
        fds_epoch::pulse_ready(ready_token, i);
        go_received = status[kSlotInterruptCount] != 0;
    }

    uint32_t result = go_received ? kComplete : fds_interrupt_status::kTimeoutInterrupt;
    if (result == kComplete && status[kSlotClaimedSource] != fds_interrupt::fds_plic_source(kGroupId)) {
        result = fds_interrupt_status::kWrongSource;
    }

    fds_interrupt::neo::disarm_external_interrupt(arming);

    // As in the polling worker, the store before the done is a local one no reader consumes on
    // seeing the done, so it needs no ordering of its own.
    fds_kernel::finish(status, kL1Address, kNumSlots, result);

    // The last ready token is still on the done wire, so this is a change the engine will capture.
    if (result == kComplete) {
        overlay::FdsNeo::fds_done(/*ad_enable=*/false, kGroupId);
    }
}
