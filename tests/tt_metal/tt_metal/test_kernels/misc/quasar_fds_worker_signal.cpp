// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Worker side of the Quasar FDS go/done handshake: wait for an FDS go signal from a dispatch
// engine, then drive the matching done signal back.
// The dispatch-engine side lives in quasar_dispatch_engine_signal.cpp.
//
// Which dispatch instance drives this NEO is not established, so every inbox register is watched
// rather than a chosen one.

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

}  // namespace

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t dispatch_mask = get_named_compile_time_arg_val("dispatch_mask");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");

    // One dispatch engine runs in this test, so exactly one instance can ever drive a go and the
    // group count tops out at one. Any larger threshold makes the group unable to latch.
    constexpr uint32_t go_threshold = 1;

    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_address);
    status[quasar_fds_test::kSlotStarted] = quasar_fds_test::kStarted;

    overlay::FdsNeo::fds_config_filter_length(kNoDeglitchFilter);
    // Worker-side GROUPID_ENABLE selects dispatch instances (3 bits), not workers.
    overlay::FdsNeo::fds_config_groupid(group_id, dispatch_mask, go_threshold);

    // A done is a held level rather than a pulse, so clear this NEO's output before the epoch to
    // make this epoch's done a fresh assertion rather than a leftover one.
    overlay::FdsNeo::fds_clear_done();

    // Bounded instead of fds_poll(), for the same reason as the dispatch-engine side. The go value
    // is held, so it is still observable if the dispatch engine signalled first.
    uint32_t go_value = 0;
    uint32_t go_instance = kNumDispatchInstances;  // sentinel: no instance delivered a go
    for (uint32_t i = 0; i < poll_iterations && go_instance == kNumDispatchInstances; i++) {
        for (uint32_t inst = 0; inst < kNumDispatchInstances; inst++) {
            const uint32_t value = overlay::FdsNeo::fds_read_de_status(inst);
            if (value == group_id) {
                go_value = value;
                go_instance = inst;
                break;
            }
        }
    }

    const bool go_received = (go_instance < kNumDispatchInstances);
    status[quasar_fds_test::kSlotObserved] = go_value;
    status[quasar_fds_test::kSlotResult] = go_received ? quasar_fds_test::kComplete : quasar_fds_test::kTimeout;
    // Commit this core's work before announcing it, which is the ordering the completion path this
    // handshake is meant to serve requires.
    flush_l2_cache_range(l1_address, quasar_fds_test::kNumSlots * sizeof(uint32_t));

    if (go_received) {
        overlay::FdsNeo::fds_done(/*ad_enable=*/false, group_id);
    }
}
