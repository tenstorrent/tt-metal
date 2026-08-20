// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Worker side of the Quasar FDS write-ordering experiment. The dispatch-engine side lives in
// quasar_dispatch_engine_ordered_read.cpp.
//
// This is the only test here that exercises what a completion fence protects. The other FDS tests
// have the worker write status to its own L1, which is a local store with nothing in flight, and
// nobody reads that status until the whole program has finished — so no reader ever depends on data
// being visible at the moment a done is observed. Here the worker writes a payload to the dispatch
// core's L1 *over the NOC* and then drives its done, and the dispatch engine reads that payload
// immediately on seeing the done.
//
// When barrier_before_done is zero the write is deliberately left un-drained before the done is
// driven. That violates the kernel-level contract on purpose: the point is to find out whether
// anything catches it. On the current NOC-atomic completion path the same omission is partly masked,
// because an undrained write and the atomic are both NOC traffic and the atomic may still arrive
// behind the write. FDS is a sideband on dedicated wires, so there is no such accident available and
// the race should be reliable rather than occasional.
//
// A pass with barrier_before_done zero is therefore not evidence of safety. It means this platform
// did not expose the hazard.
//
// signal_via_fds selects how completion is announced. With it set, the done goes over the FDS
// sideband, which is the mechanism under evaluation. With it clear, the worker instead increments a
// counter in the dispatch core's L1 with a NOC atomic — the mechanism in use today, on the same NOC
// and the same virtual channel as the payload write. That control arm is what says whether the
// current completion path is safe because kernels drain, or safe by accident because NOC ordering
// holds the atomic behind the data.

#include "risc_attribs.h"
#include "risc_common.h"
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "overlay/fds_functions.hpp"

#include "quasar_fds_signal_status.h"

namespace {

// One DISPATCH_TO_TENSIX inbox register per dispatch instance on the NEO side.
constexpr uint32_t kNumDispatchInstances = 3;

// A go must be stable for a single cycle to be accepted. This is the register's reset value,
// written explicitly so the handshake does not depend on the state a previous program left behind.
constexpr uint32_t kNoDeglitchFilter = 0;

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

}  // namespace

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t payload_src_address = get_named_compile_time_arg_val("payload_src_address");
    constexpr uint32_t dest_noc_x = get_named_compile_time_arg_val("dest_noc_x");
    constexpr uint32_t dest_noc_y = get_named_compile_time_arg_val("dest_noc_y");
    constexpr uint32_t dest_address = get_named_compile_time_arg_val("dest_address");
    constexpr uint32_t counter_address = get_named_compile_time_arg_val("counter_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t dispatch_mask = get_named_compile_time_arg_val("dispatch_mask");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");
    constexpr uint32_t barrier_before_done = get_named_compile_time_arg_val("barrier_before_done");
    constexpr uint32_t signal_via_fds = get_named_compile_time_arg_val("signal_via_fds");

    constexpr uint32_t go_threshold = 1;

    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_address);
    status[quasar_fds_test::ordering::kSlotStarted] = quasar_fds_test::kStarted;
    status[quasar_fds_test::ordering::kSlotResult] = quasar_fds_test::kTimeout;
    status[quasar_fds_test::ordering::kSlotBarrierUsed] = barrier_before_done;
    status[quasar_fds_test::ordering::kSlotSignalledByFds] = signal_via_fds;

    // Build the payload in this core's L1, then flush it so the NOC reads the values from node
    // memory rather than whatever the cache has not written back yet. This flush is about the source
    // of the transfer and is required either way; it is not the ordering under test.
    volatile tt_l1_ptr uint32_t* payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload_src_address);
    for (uint32_t word = 0; word < quasar_fds_test::ordering::kPayloadWords; word++) {
        payload[word] = quasar_fds_test::ordering::kPayloadSeed + word;
    }
    flush_l2_cache_range(payload_src_address, quasar_fds_test::ordering::kPayloadBytes);

    overlay::FdsNeo::fds_config_filter_length(kNoDeglitchFilter);
    // Worker-side GROUPID_ENABLE selects dispatch instances (3 bits), not workers.
    overlay::FdsNeo::fds_config_groupid(group_id, dispatch_mask, go_threshold);
    overlay::FdsNeo::fds_clear_done();

    const uint32_t instance = wait_for_go(group_id, poll_iterations);
    const bool go_seen = (instance < kNumDispatchInstances);
    status[quasar_fds_test::ordering::kSlotGoSeen] = go_seen ? 1 : 0;
    if (!go_seen) {
        flush_l2_cache_range(l1_address, quasar_fds_test::ordering::kNumSlots * sizeof(uint32_t));
        return;
    }

    Noc noc;
    UnicastEndpoint destination;
    CoreLocalMem<uint32_t> source(payload_src_address);
    noc.async_write(
        source,
        destination,
        quasar_fds_test::ordering::kPayloadBytes,
        {},
        {.noc_x = dest_noc_x, .noc_y = dest_noc_y, .addr = dest_address});

    // The whole experiment is in this branch.
    if constexpr (barrier_before_done != 0) {
        noc.async_write_barrier();
    }

    // Signal with nothing in between. An earlier version wrote and flushed this core's status block
    // here, before the done — the flush is a fenced operation, and that delay was enough for the
    // write to land, which masked the very race the unbarriered arm exists to find. Bookkeeping goes
    // after the signal now.
    if constexpr (signal_via_fds != 0) {
        overlay::FdsNeo::fds_done(/*ad_enable=*/false, group_id);
    } else {
        // The same virtual channel the real completion path uses, so this is a faithful stand-in for
        // notify_dispatch_core_done rather than merely another way to write a word. The plain
        // address, not the uncached alias, because the alias is core-local and not NOC-addressable.
        noc_semaphore_inc(
            get_noc_addr(dest_noc_x, dest_noc_y, counter_address, noc.get_noc_id()),
            1,
            noc.get_noc_id(),
            NOC_UNICAST_WRITE_VC);
    }

    status[quasar_fds_test::ordering::kSlotResult] = quasar_fds_test::kComplete;
    flush_l2_cache_range(l1_address, quasar_fds_test::ordering::kNumSlots * sizeof(uint32_t));

    // Drain before returning regardless, so an un-drained write cannot outlive the kernel and land
    // during a later test's measurements.
    noc.async_write_barrier();
}
