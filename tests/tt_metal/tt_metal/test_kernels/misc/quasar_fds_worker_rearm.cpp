// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Worker side of the Quasar FDS two-epoch re-arm experiment. The dispatch-engine side lives in
// quasar_dispatch_engine_rearm.cpp and describes the protocol.
//
// This side answers the engine-side half of the same question. A go is a held level, so a second go
// for the same group is invisible unless the first one de-asserts. The worker therefore waits for
// the go, answers, then waits for the go to *drop* before clearing its own done and waiting for the
// go to return. Whether the group status register clears when the go goes idle is recorded, since
// that is what a receiver would key an epoch boundary on.

#include "risc_attribs.h"
#include "risc_common.h"
#include "api/compile_time_args.h"
#include "overlay/fds_functions.hpp"

#include "quasar_fds_signal_status.h"

namespace {

// One DISPATCH_TO_TENSIX inbox register per dispatch instance on the NEO side.
constexpr uint32_t kNumDispatchInstances = 3;

// A go must be stable for a single cycle to be accepted. This is the register's reset value,
// written explicitly so the handshake does not depend on the state a previous program left behind.
constexpr uint32_t kNoDeglitchFilter = 0;

// Bounded wait for any instance to carry the given group, returning the instance or the sentinel.
uint32_t wait_for_go(uint32_t group_id, uint32_t iterations) {
    for (uint32_t i = 0; i < iterations; i++) {
        for (uint32_t inst = 0; inst < kNumDispatchInstances; inst++) {
            if (overlay::FdsNeo::fds_read_de_status(inst) == group_id) {
                return inst;
            }
        }
    }
    return kNumDispatchInstances;
}

// Bounded wait for one instance to stop carrying the given group.
bool wait_for_deassert(uint32_t instance, uint32_t group_id, uint32_t iterations) {
    for (uint32_t i = 0; i < iterations; i++) {
        if (overlay::FdsNeo::fds_read_de_status(instance) != group_id) {
            return true;
        }
    }
    return false;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t dispatch_mask = get_named_compile_time_arg_val("dispatch_mask");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");

    // One dispatch engine takes part, so one instance can drive a go.
    constexpr uint32_t go_threshold = 1;

    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_address);
    status[quasar_fds_test::rearm::kSlotStarted] = quasar_fds_test::kStarted;
    status[quasar_fds_test::rearm::kSlotResult] = quasar_fds_test::kTimeout;

    overlay::FdsNeo::fds_config_filter_length(kNoDeglitchFilter);
    // Worker-side GROUPID_ENABLE selects dispatch instances (3 bits), not workers.
    overlay::FdsNeo::fds_config_groupid(group_id, dispatch_mask, go_threshold);
    overlay::FdsNeo::fds_clear_done();

    // Epoch one.
    const uint32_t instance = wait_for_go(group_id, poll_iterations);
    const bool round1_go = (instance < kNumDispatchInstances);
    status[quasar_fds_test::rearm::kSlotRound1Go] = round1_go ? 1 : 0;

    if (round1_go) {
        overlay::FdsNeo::fds_done(/*ad_enable=*/false, group_id);

        // The dispatch engine drops the go between epochs. Without that edge there is nothing to
        // distinguish a second go for this group from the first one still being held.
        const bool deassert_seen = wait_for_deassert(instance, group_id, poll_iterations);
        status[quasar_fds_test::rearm::kSlotDeassertSeen] = deassert_seen ? 1 : 0;
        // Whether the group latch follows the wire back to idle, which is what a receiver would key
        // an epoch boundary on if it watched status rather than the raw inbox.
        status[quasar_fds_test::rearm::kSlotStatusAfterDeassert] = overlay::FdsNeo::fds_read_group_status(group_id);

        if (deassert_seen) {
            // Required between two dones of the same group, per the shim's own contract.
            overlay::FdsNeo::fds_clear_done();

            const uint32_t second_instance = wait_for_go(group_id, poll_iterations);
            const bool round2_go = (second_instance < kNumDispatchInstances);
            status[quasar_fds_test::rearm::kSlotRound2Go] = round2_go ? 1 : 0;
            if (round2_go) {
                status[quasar_fds_test::rearm::kSlotResult] = quasar_fds_test::kComplete;
                flush_l2_cache_range(l1_address, quasar_fds_test::rearm::kNumSlots * sizeof(uint32_t));
                overlay::FdsNeo::fds_done(/*ad_enable=*/false, group_id);
                return;
            }
        }
    }

    flush_l2_cache_range(l1_address, quasar_fds_test::rearm::kNumSlots * sizeof(uint32_t));
}
