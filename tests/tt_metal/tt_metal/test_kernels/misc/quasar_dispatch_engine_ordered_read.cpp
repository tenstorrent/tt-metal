// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch-engine side of the Quasar FDS write-ordering experiment. The worker side lives in
// quasar_fds_worker_ordered_write.cpp and describes what is being tested.
//
// Send a go, wait for the worker's completion signal, and read the payload the worker wrote into this
// core's L1 the instant that signal appears. signal_via_fds selects whether completion arrives over
// the FDS sideband or as a NOC atomic increment of a counter in this core's L1 — the latter being the
// mechanism in use today, and the control arm for whether it is safe by accident. Any word still holding the host's
// pre-fill value is a word that had not landed when the done was observed, which is the corruption a completion fence
// prevents.
//
// The payload region is invalidated from L2 **before** the go is sent, not after the done arrives.
// Without any invalidation a cached copy of the pre-fill would read as a missing write and every run
// would report the hazard whether or not it happened. But doing it after the done put hundreds of
// fenced cache-line operations between the signal and the read, which is ample time for an in-flight
// write to land — it masked the race. Invalidating first leaves the lines cold, so the post-done read
// fetches from node memory with nothing in front of it.

#include "risc_attribs.h"
#include "risc_common.h"
#include "api/compile_time_args.h"
#include "dev_mem_map.h"
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
    constexpr uint32_t payload_address = get_named_compile_time_arg_val("payload_address");
    constexpr uint32_t counter_address = get_named_compile_time_arg_val("counter_address");
    constexpr uint32_t signal_via_fds = get_named_compile_time_arg_val("signal_via_fds");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t worker_mask = get_named_compile_time_arg_val("worker_mask");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");

    // One worker takes part, so one lane can drive a done.
    constexpr uint32_t done_threshold = 1;

    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_address);
    status[quasar_fds_test::ordering::kSlotStarted] = quasar_fds_test::kStarted;
    status[quasar_fds_test::ordering::kSlotResult] = quasar_fds_test::kTimeout;
    flush_l2_cache_range(l1_address, quasar_fds_test::ordering::kNumSlots * sizeof(uint32_t));

    overlay::FdsDispatch::fds_config_filter_length(kNoDeglitchFilter);
    overlay::FdsDispatch::fds_config_groupid(group_id, worker_mask, done_threshold);
    for (uint32_t neo = 0; neo < kNumNeoWires; neo++) {
        overlay::FdsDispatch::fds_clear_neo_status(neo);
    }

    // Cold before the signal, so the read after it is immediate. Nothing reads this region in
    // between, so nothing can re-cache a stale copy.
    invalidate_l2_cache_range(payload_address, quasar_fds_test::ordering::kPayloadBytes);

    overlay::FdsDispatch::fds_go(/*ad_enable=*/false, group_id);

    // Read through the uncached alias, which is how a locally polled word that a remote core writes
    // is seen without invalidating a cache line on every iteration. The alias is core-local, so the
    // worker's atomic targets the plain address and only this read uses the alias.
    volatile tt_l1_ptr uint32_t* counter =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_L1_UNCACHED_BASE + counter_address);
    *counter = 0;

    uint32_t done_count = 0;
    for (uint32_t i = 0; i < poll_iterations; i++) {
        done_count = (signal_via_fds != 0) ? overlay::FdsDispatch::fds_read_group_count(group_id) : *counter;
        if (done_count >= done_threshold) {
            break;
        }
    }

    if (done_count >= done_threshold) {
        volatile tt_l1_ptr uint32_t* payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload_address);

        // The tail word, first and alone. The full scan below takes long enough that its later words
        // have time to arrive while it runs, so this single read is the sharpest evidence the test
        // can offer about the state of the payload at the moment the done was observed.
        status[quasar_fds_test::ordering::kSlotTailWord] = payload[quasar_fds_test::ordering::kPayloadWords - 1];

        uint32_t mismatches = 0;
        uint32_t first_index = quasar_fds_test::ordering::kPayloadWords;
        uint32_t first_value = 0;
        for (uint32_t word = 0; word < quasar_fds_test::ordering::kPayloadWords; word++) {
            const uint32_t observed = payload[word];
            if (observed != quasar_fds_test::ordering::kPayloadSeed + word) {
                if (mismatches == 0) {
                    first_index = word;
                    first_value = observed;
                }
                mismatches++;
            }
        }

        status[quasar_fds_test::ordering::kSlotMismatches] = mismatches;
        status[quasar_fds_test::ordering::kSlotFirstMismatchIndex] = first_index;
        status[quasar_fds_test::ordering::kSlotFirstMismatchValue] = first_value;
        status[quasar_fds_test::ordering::kSlotResult] = quasar_fds_test::kComplete;
    }

    flush_l2_cache_range(l1_address, quasar_fds_test::ordering::kNumSlots * sizeof(uint32_t));
}
