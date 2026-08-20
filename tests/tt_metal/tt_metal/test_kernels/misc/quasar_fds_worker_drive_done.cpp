// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Worker side of the Quasar FDS lane-mapping experiment. The dispatch-engine side lives in
// quasar_dispatch_engine_lane_map.cpp and describes what is being measured.
//
// This kernel waits for nothing. Each tile is given its own group id and drives a done carrying it,
// so the value arriving at the dispatch engine identifies the tile that sent it. No group
// configuration is needed on this side, because nothing here receives: the go direction is not
// involved in the experiment at all.
//
// The done is a held level, so it stays asserted after this kernel returns. It spins briefly anyway
// so that the tile is still running while the dispatch engine scans, which keeps the measurement
// independent of whether firmware teardown disturbs the register.

#include "risc_attribs.h"
#include "risc_common.h"
#include "api/compile_time_args.h"
#include "overlay/fds_functions.hpp"

#include "quasar_fds_signal_status.h"

namespace {

// A done must be stable for a single cycle to be accepted. This is the register's reset value,
// written explicitly so the result does not depend on the state a previous program left behind.
constexpr uint32_t kNoDeglitchFilter = 0;

}  // namespace

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t hold_iterations = get_named_compile_time_arg_val("hold_iterations");

    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_address);
    status[quasar_fds_test::kSlotStarted] = quasar_fds_test::kStarted;
    status[quasar_fds_test::kSlotObserved] = group_id;
    status[quasar_fds_test::kSlotResult] = quasar_fds_test::kComplete;
    // Commit before signalling, which is the ordering a real completion path requires.
    flush_l2_cache_range(l1_address, quasar_fds_test::kNumSlots * sizeof(uint32_t));

    overlay::FdsNeo::fds_config_filter_length(kNoDeglitchFilter);
    overlay::FdsNeo::fds_clear_done();
    overlay::FdsNeo::fds_done(/*ad_enable=*/false, group_id);

    for (volatile uint32_t i = 0; i < hold_iterations; i++);
}
