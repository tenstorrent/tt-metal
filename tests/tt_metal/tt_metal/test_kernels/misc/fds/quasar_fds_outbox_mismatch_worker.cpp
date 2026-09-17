// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

using fds_outbox::kMatchedGo;
using fds_outbox::kMismatchedGo;
using fds_outbox::kTokenArmed;
using fds_outbox::kTokenDelivered;
using fds_outbox::kTokenSilenceChecked;

constexpr uint32_t kNumSlots = 2;
constexpr uint32_t kMismatchedDelivered = 0x5A5A0060;
constexpr uint32_t kTimeoutMatched = 0x5A5A0061;

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t dispatch_mask = get_named_compile_time_arg_val("dispatch_mask");
    constexpr uint32_t silence_iterations = get_named_compile_time_arg_val("silence_iterations");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");
    static_assert(kTokenDelivered < kReadyTokenA, "step tokens must stay below the ready tokens");

    fds_kernel::status_ptr status = fds_kernel::begin_worker(l1_address, kNumSlots);

    uint32_t go_inst = 0;
    if (!fds_kernel::received_go(status, l1_address, kNumSlots, dispatch_mask, kSessionGo, poll_iterations, go_inst)) {
        return;
    }

    overlay::FdsNeo::fds_clear_de_status(go_inst);
    overlay::FdsNeo::fds_done(/*ad_enable=*/false, kTokenArmed);

    uint32_t result = kComplete;
    uint32_t observed = fds_kernel::lane_nonzero(go_inst, silence_iterations);
    if (observed != 0) {
        result = kMismatchedDelivered;
    }

    if (result == kComplete) {
        overlay::FdsNeo::fds_done(/*ad_enable=*/false, kTokenSilenceChecked);
        bool matched = false;
        for (uint32_t i = 0; i < poll_iterations && !matched; i++) {
            observed = overlay::FdsNeo::fds_read_de_status(go_inst);
            // The mismatched value surfacing late is the same failure as surfacing early.
            if (observed == kMismatchedGo) {
                result = kMismatchedDelivered;
                break;
            }
            matched = observed == kMatchedGo;
        }
        if (result == kComplete && !matched) {
            result = kTimeoutMatched;
        }
    }

    status[kSlotObservedValue] = observed;
    fds_kernel::finish(status, l1_address, kNumSlots, result);

    if (result == kComplete) {
        overlay::FdsNeo::fds_done(/*ad_enable=*/false, kTokenDelivered);
    }
}
