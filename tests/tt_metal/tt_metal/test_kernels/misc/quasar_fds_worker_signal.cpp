// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Worker side of the Quasar FDS go/done bring-up handshake: wait for an FDS go signal from a
// dispatch engine, then drive the matching done signal back.
// The dispatch-engine side lives in quasar_dispatch_engine_signal.cpp.
//
// The two directions are deliberately not chained. Done is driven part way through the wait
// whether or not a go ever arrives, so one run reports on each direction independently instead of
// a missing go leaving the done direction untested.
//
// Which dispatch instance drives this NEO is not yet established, so all three inbox registers are
// watched rather than just one, and the group status register is reported alongside them to
// separate "no go arrived" from "a go arrived but did not latch into the group".
//
// This runs on every user data-movement core at once, each writing to its own status block. That
// started as a search for which processor the sideband reaches and answered a different question
// instead: the whole tile shares one register block, so every processor here is driving and
// reading the same registers. Each one stamps its index into a spare register and reads it back at
// the end, which is the check that established that and keeps it visible.

#include "risc_attribs.h"
#include "risc_common.h"
#include "api/compile_time_args.h"
#include "api/debug/dprint.h"
#include "internal/hw_thread.h"
#include "overlay/fds_functions.hpp"

#include "quasar_fds_probes.h"
#include "quasar_fds_signal_status.h"

namespace {

// One DISPATCH_TO_TENSIX inbox register per dispatch instance on the NEO side.
constexpr uint32_t kNumDispatchInstances = 3;

// DM0 and DM1 are reserved on worker clusters, so user kernels start here. Only this one runs the
// register probes, which describe the block rather than the wiring and would be identical on all
// six otherwise.
constexpr uint32_t kFirstUserProcessor = 2;

// Group ids the handshake does not use, so the probes can write to real register addresses
// without disturbing the group under test.
constexpr uint32_t kScratchGroupA = 14;
constexpr uint32_t kScratchGroupB = 15;

// Establish what the FDS register interface on this processor actually is before reading anything
// from it for real. Nothing here touches DISPATCH_TO_TENSIX: that is where the go arrives, the
// dispatch engine sends it exactly once, and writing to it could erase a go that has already
// landed. The dispatch-engine kernel probes its own hardware-driven inbox instead, where the
// ordering makes it safe.
void probe_register_interface() {
    quasar_fds_probe::address_map();
    quasar_fds_probe::field_truncation(
        TT_FDS_TENSIXNEO_GROUPID_ENABLE_0__REG_ADDR + (kScratchGroupB * sizeof(uint32_t)),
        TT_FDS_TENSIXNEO_GROUPID_ENABLE_DATA_MASK);
    quasar_fds_probe::field_truncation(
        TT_FDS_TENSIXNEO_GROUPID_COUNT_THRESHOLD_0__REG_ADDR + (kScratchGroupB * sizeof(uint32_t)),
        TT_FDS_TENSIXNEO_GROUPID_COUNT_THRESHOLD_DATA_MASK);
    quasar_fds_probe::cross_address(
        TT_FDS_TENSIXNEO_GROUPID_COUNT_THRESHOLD_0__REG_ADDR + (kScratchGroupA * sizeof(uint32_t)),
        TT_FDS_TENSIXNEO_GROUPID_COUNT_THRESHOLD_0__REG_ADDR + (kScratchGroupB * sizeof(uint32_t)));
}

}  // namespace

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t dispatch_mask = get_named_compile_time_arg_val("dispatch_mask");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");

    // One dispatch engine runs in this test, so exactly one instance can ever drive a go and the
    // group count tops out at one. Any larger threshold makes the group unable to latch.
    constexpr uint32_t go_threshold = 1;
    // The wait is divided into one phase per deglitcher setting, and each phase also reports.
    constexpr uint32_t filter_phase_length = poll_iterations / quasar_fds_probe::kNumFilterPhases;
    // This kernel gets through its loop at roughly a third of the dispatch engine's rate, so an
    // eighth of the budget here falls around a third of the way through the dispatch engine's own
    // wait, leaving it plenty of room to observe the done.
    constexpr uint32_t kUnpromptedDoneIteration = poll_iterations / 8;

    // Every user data-movement core runs this kernel, so each takes the status block matching its
    // own hardware thread index and the host can see which of them, if any, the sideband reaches.
    const uint32_t processor_index = internal_::get_hw_thread_idx();
    const uint32_t status_address =
        l1_address + (processor_index * quasar_fds_test::kSlotsPerProcessor * sizeof(uint32_t));
    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(status_address);

    DPRINT(
        "[FDS worker] started: processor={} group={} dispatch_mask={:#x} l1={:#x} poll_limit={}\n",
        processor_index,
        group_id,
        dispatch_mask,
        status_address,
        poll_iterations);
    status[quasar_fds_test::kSlotStarted] = quasar_fds_test::kStarted;

    overlay::FdsNeo::fds_config_filter_length(quasar_fds_probe::kFilterSweep[0]);
    // Worker-side GROUPID_ENABLE selects dispatch instances (3 bits), not workers.
    overlay::FdsNeo::fds_config_groupid(group_id, dispatch_mask, go_threshold);
    // Read the config back before waiting. If these do not return what was just written, the FDS
    // register file is not responding on this processor and no amount of signalling will work,
    // which is a different failure from a go signal that never arrives.
    DPRINT(
        "[FDS worker] config readback: filter={} groupid_enable={:#x} count_threshold={}\n",
        static_cast<uint32_t>(FDS_INTF_READ(TT_FDS_TENSIXNEO_FILTER_COUNT_THRESHOLD_REG_ADDR)),
        static_cast<uint32_t>(
            FDS_INTF_READ(TT_FDS_TENSIXNEO_GROUPID_ENABLE_0__REG_ADDR + (group_id * sizeof(uint32_t)))),
        static_cast<uint32_t>(
            FDS_INTF_READ(TT_FDS_TENSIXNEO_GROUPID_COUNT_THRESHOLD_0__REG_ADDR + (group_id * sizeof(uint32_t)))));

    // Never written by this test and never read until now. Both reset to zero, but stale state is
    // one of the things an initialisation is meant to survive, and an enabled auto-dispatch could
    // hold an outbox write in a queue rather than drive it onto the wire.
    DPRINT(
        "[FDS worker] auto_dispatch_en={} interrupt_enable={:#x}\n",
        static_cast<uint32_t>(FDS_INTF_READ(TT_FDS_TENSIXNEO_AUTO_DISPATCH_EN_REG_ADDR)),
        static_cast<uint32_t>(FDS_INTF_READ(TT_FDS_TENSIXNEO_INTERRUPT_ENABLE_REG_ADDR)));
    FDS_INTF_WRITE(TT_FDS_TENSIXNEO_AUTO_DISPATCH_EN_REG_ADDR, 0);
    FDS_INTF_WRITE(TT_FDS_TENSIXNEO_INTERRUPT_ENABLE_REG_ADDR, 0);

    // Stamp this processor's index into a register nothing else touches. It is read back at the
    // very end, once every processor has certainly written, to find out whether the processors on
    // this tile share one register block or have one each.
    const uint32_t sharedness_address =
        TT_FDS_TENSIXNEO_GROUPID_COUNT_THRESHOLD_0__REG_ADDR + (quasar_fds_probe::kSharednessGroup * sizeof(uint32_t));
    const uint32_t own_stamp = quasar_fds_probe::kStampBase + processor_index;
    FDS_INTF_WRITE(sharedness_address, own_stamp);

    // An idle lane presents group 0, so the group 0 status is the mask of dispatch instances this
    // processor's block is actually observing. Zero here says no lane reaches this processor at
    // all, which is a different finding from a go that never arrives on a lane that does.
    DPRINT(
        "[FDS worker] idle lanes seen by this processor: group0_status={:#x}\n",
        overlay::FdsNeo::fds_read_group_status(0));

    if (processor_index == kFirstUserProcessor) {
        probe_register_interface();
    }

    // The done output holds its value, so clear it before the epoch to make this epoch's
    // done a fresh assertion rather than a leftover one.
    overlay::FdsNeo::fds_clear_done();
    DPRINT("[FDS worker] cleared stale done output\n");

    // Bounded instead of fds_poll(), for the same reason as the dispatch-engine side. The go
    // value is held, so it is still observable if the dispatch engine signalled first.
    uint32_t go_value = 0;
    uint32_t go_instance = kNumDispatchInstances;  // sentinel: no instance delivered a go
    bool done_driven = false;
    for (uint32_t i = 0; i < poll_iterations; i++) {
        for (uint32_t inst = 0; inst < kNumDispatchInstances; inst++) {
            const uint32_t value = overlay::FdsNeo::fds_read_de_status(inst);
            if (value == group_id) {
                go_value = value;
                go_instance = inst;
                break;
            }
        }
        if (go_instance < kNumDispatchInstances) {
            DPRINT("[FDS worker] received go at iteration {}: instance={} value={}\n", i, go_instance, go_value);
            break;
        }
        // Drive done part way through the wait whether or not a go ever arrived. The two
        // directions are otherwise serialised, so a go that never lands hides everything about
        // whether done travels. This point is far enough in that the dispatch engine has long
        // finished clearing its inboxes, and early enough that it is still watching them.
        if (i == kUnpromptedDoneIteration) {
            overlay::FdsNeo::fds_done(/*ad_enable=*/false, group_id);
            done_driven = true;
            DPRINT(
                "[FDS worker] drove done with no go received, to exercise that direction alone: "
                "group={} own_output={}\n",
                group_id,
                static_cast<uint32_t>(FDS_INTF_READ(TT_FDS_TENSIXNEO_TENSIX_TO_DISPATCH_REG_ADDR)));
        }
        if ((i % filter_phase_length) == 0) {
            // Re-arm the deglitcher with the next setting. The go is held, so each setting gets a
            // fresh chance at a signal that is still being driven. The value is read back rather
            // than assumed, so a write that did not take is visible.
            const uint32_t phase = i / filter_phase_length;
            if (phase < quasar_fds_probe::kNumFilterPhases) {
                overlay::FdsNeo::fds_config_filter_length(quasar_fds_probe::kFilterSweep[phase]);
            }
            DPRINT(
                "[FDS worker] waiting for go: iteration={} deglitcher={} group_status={} inbox=[{}, {}, {}]\n",
                i,
                static_cast<uint32_t>(FDS_INTF_READ(TT_FDS_TENSIXNEO_FILTER_COUNT_THRESHOLD_REG_ADDR)),
                overlay::FdsNeo::fds_read_group_status(group_id),
                overlay::FdsNeo::fds_read_de_status(0),
                overlay::FdsNeo::fds_read_de_status(1),
                overlay::FdsNeo::fds_read_de_status(2));
        }
    }

    const bool go_received = (go_instance < kNumDispatchInstances);
    if (!go_received) {
        DPRINT(
            "[FDS worker] timed out waiting for go: group_status={} inbox=[{}, {}, {}]\n",
            overlay::FdsNeo::fds_read_group_status(group_id),
            overlay::FdsNeo::fds_read_de_status(0),
            overlay::FdsNeo::fds_read_de_status(1),
            overlay::FdsNeo::fds_read_de_status(2));
    }

    status[quasar_fds_test::kSlotObserved] = go_value;
    status[quasar_fds_test::kSlotResult] = go_received ? quasar_fds_test::kComplete : quasar_fds_test::kTimeout;
    flush_l2_cache_range(status_address, quasar_fds_test::kNumSlots * sizeof(uint32_t));

    if (go_received && !done_driven) {
        overlay::FdsNeo::fds_done(/*ad_enable=*/false, group_id);
        DPRINT(
            "[FDS worker] sent done: group={} own_output={}\n",
            group_id,
            static_cast<uint32_t>(FDS_INTF_READ(TT_FDS_TENSIXNEO_TENSIX_TO_DISPATCH_REG_ADDR)));
    }
    // Every processor has written its stamp long before now, so a processor reading anything other
    // than its own is reading a register a neighbour wrote, and the block is shared by the tile.
    // Reading one's own stamp proves nothing on its own: it is equally what a private block gives
    // and what a shared block gives to whichever processor wrote last.
    const uint32_t stamp_observed = static_cast<uint32_t>(FDS_INTF_READ(sharedness_address));
    DPRINT(
        "[FDS shared] processor={} stamped {:#x}, register now reads {:#x}\n",
        processor_index,
        own_stamp,
        stamp_observed);
    if (stamp_observed != own_stamp) {
        DPRINT("[FDS shared]   that is another processor's stamp - one block is shared by the tile\n");
    }

    DPRINT("[FDS worker] finished: result={:#x} go_value={}\n", status[quasar_fds_test::kSlotResult], go_value);
}
