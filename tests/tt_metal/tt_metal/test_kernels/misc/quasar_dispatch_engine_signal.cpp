// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch-engine side of the Quasar FDS go/done bring-up handshake: write a payload to L1,
// send an FDS go signal to the worker NEOs selected by worker_mask, then wait for a done signal.
// The worker side lives in quasar_fds_worker_signal.cpp.
//
// The worker drives its done whether or not the go reached it, so the wait below reports on the
// done direction on its own rather than only on a complete round trip.
//
// The FDS wire index of any given worker core is not yet established, so this kernel is written to
// probe rather than assume: the host can aim the go at all 32 NEO wires at once, and every done
// inbox is dumped on both success and timeout so the responding wire can be identified.
//
// This runs on every data-movement core of the tile, each sending a go and watching the inboxes.
// That started as a search for which processor the sideband reaches and answered a different
// question instead: the whole tile shares one register block, so every processor here is driving
// and reading the same registers. Each one stamps its index into a spare register and reads it
// back at the end, which is the check that established that and keeps it visible.

#include "risc_attribs.h"
#include "risc_common.h"
#include "api/compile_time_args.h"
#include "api/debug/dprint.h"
#include "internal/hw_thread.h"
#include "overlay/fds_functions.hpp"

#include "quasar_fds_probes.h"
#include "quasar_fds_signal_status.h"

namespace {

// One TENSIX_TO_DISPATCH inbox register per NEO wire on the dispatch side.
constexpr uint32_t kNumNeoWires = 32;

// Group ids the handshake does not use, so the probes can write to real register addresses
// without disturbing the group under test.
constexpr uint32_t kScratchGroupA = 14;
constexpr uint32_t kScratchGroupB = 15;

// Dispatch-engine tiles reserve no data-movement cores, so the sweep starts at the first one. Only
// that one runs the register probes, which describe the block rather than the wiring and would be
// identical on all eight otherwise.
constexpr uint32_t kFirstProcessor = 0;

// Index of the first NEO driving a non-zero done, or kNumNeoWires if none is.
uint32_t first_driving_neo() {
    for (uint32_t neo = 0; neo < kNumNeoWires; neo++) {
        if (overlay::FdsDispatch::fds_read_neo_status(neo) != 0) {
            return neo;
        }
    }
    return kNumNeoWires;
}

// Dump every NEO done inbox that is driving a non-zero value, so the wire index behind a done
// (or the absence of one) is visible.
void dump_neo_inboxes() {
    uint32_t non_zero = 0;
    for (uint32_t neo = 0; neo < kNumNeoWires; neo++) {
        const uint32_t value = overlay::FdsDispatch::fds_read_neo_status(neo);
        if (value != 0) {
            non_zero++;
            DPRINT("[FDS dispatch]   neo_inbox[{}] = {}\n", neo, value);
        }
    }
    if (non_zero == 0) {
        DPRINT("[FDS dispatch]   all {} neo inboxes are zero\n", kNumNeoWires);
    }
}

// Establish what the FDS register interface on this processor actually is before reading anything
// from it for real. TENSIX_TO_DISPATCH is a hardware-driven input, so it is the most telling
// target: probing it here is safe because the inboxes are cleared immediately afterwards and a
// worker can only drive a done once the go below has been sent.
void probe_register_interface() {
    quasar_fds_probe::address_map();
    quasar_fds_probe::field_truncation(
        TT_FDS_DISPATCH_TENSIX_TO_DISPATCH_0__REG_ADDR, TT_FDS_DISPATCH_TENSIX_TO_DISPATCH_DATA_MASK);
    quasar_fds_probe::field_truncation(
        TT_FDS_DISPATCH_GROUPID_COUNT_THRESHOLD_0__REG_ADDR + (kScratchGroupB * sizeof(uint32_t)),
        TT_FDS_DISPATCH_GROUPID_COUNT_THRESHOLD_DATA_MASK);
    quasar_fds_probe::cross_address(
        TT_FDS_DISPATCH_GROUPID_COUNT_THRESHOLD_0__REG_ADDR + (kScratchGroupA * sizeof(uint32_t)),
        TT_FDS_DISPATCH_GROUPID_COUNT_THRESHOLD_0__REG_ADDR + (kScratchGroupB * sizeof(uint32_t)));
}

}  // namespace

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t worker_mask = get_named_compile_time_arg_val("worker_mask");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");

    // One worker kernel instance runs in this test, so exactly one NEO can ever drive a done and
    // the group count tops out at one. Any larger threshold makes the wait unsatisfiable.
    constexpr uint32_t done_threshold = 1;
    // The wait is divided into one phase per deglitcher setting, and each phase also reports.
    constexpr uint32_t filter_phase_length = poll_iterations / quasar_fds_probe::kNumFilterPhases;

    // Every data-movement core on this tile runs the kernel, so each takes the status block
    // matching its own hardware thread index and the host can see which of them, if any, the
    // sideband reaches.
    const uint32_t processor_index = internal_::get_hw_thread_idx();
    const uint32_t status_address =
        l1_address + (processor_index * quasar_fds_test::kSlotsPerProcessor * sizeof(uint32_t));
    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(status_address);

    DPRINT(
        "[FDS dispatch] started: processor={} group={} worker_mask={:#x} l1={:#x} poll_limit={}\n",
        processor_index,
        group_id,
        worker_mask,
        status_address,
        poll_iterations);

    status[quasar_fds_test::kSlotStarted] = quasar_fds_test::kStarted;
    // Commit the payload to node memory before signalling, so the go signal cannot be
    // observed ahead of the data it is announcing.
    flush_l2_cache_range(status_address, quasar_fds_test::kNumSlots * sizeof(uint32_t));

    overlay::FdsDispatch::fds_config_filter_length(quasar_fds_probe::kFilterSweep[0]);
    overlay::FdsDispatch::fds_config_groupid(group_id, worker_mask, done_threshold);
    // Read the config back before signalling. If these do not return what was just written, the FDS
    // register file is not responding on this processor and no amount of signalling will work,
    // which is a different failure from a done signal that never arrives.
    DPRINT(
        "[FDS dispatch] config readback: filter={} groupid_enable={:#x} count_threshold={}\n",
        static_cast<uint32_t>(FDS_INTF_READ(TT_FDS_DISPATCH_FILTER_COUNT_THRESHOLD_REG_ADDR)),
        static_cast<uint32_t>(
            FDS_INTF_READ(TT_FDS_DISPATCH_GROUPID_ENABLE_0__REG_ADDR + (group_id * sizeof(uint32_t)))),
        static_cast<uint32_t>(
            FDS_INTF_READ(TT_FDS_DISPATCH_GROUPID_COUNT_THRESHOLD_0__REG_ADDR + (group_id * sizeof(uint32_t)))));

    // Never written by this test and never read until now. Both reset to zero, but stale state is
    // one of the things an initialisation is meant to survive, and an enabled auto-dispatch could
    // hold an outbox write in a queue rather than drive it onto the wire.
    DPRINT(
        "[FDS dispatch] auto_dispatch_en={} interrupt_enable={:#x}\n",
        static_cast<uint32_t>(FDS_INTF_READ(TT_FDS_DISPATCH_AUTO_DISPATCH_EN_REG_ADDR)),
        static_cast<uint32_t>(FDS_INTF_READ(TT_FDS_DISPATCH_INTERRUPT_ENABLE_REG_ADDR)));
    FDS_INTF_WRITE(TT_FDS_DISPATCH_AUTO_DISPATCH_EN_REG_ADDR, 0);
    FDS_INTF_WRITE(TT_FDS_DISPATCH_INTERRUPT_ENABLE_REG_ADDR, 0);

    // Stamp this processor's index into a register nothing else touches. It is read back at the
    // very end, once every processor has certainly written, to find out whether the eight
    // processors on this tile share one register block or have one each.
    const uint32_t sharedness_address =
        TT_FDS_DISPATCH_GROUPID_COUNT_THRESHOLD_0__REG_ADDR + (quasar_fds_probe::kSharednessGroup * sizeof(uint32_t));
    const uint32_t own_stamp = quasar_fds_probe::kStampBase + processor_index;
    FDS_INTF_WRITE(sharedness_address, own_stamp);

    // An idle lane presents group 0, so the group 0 status is the mask of done lanes this
    // processor's block is observing. Zero here would say no lane reaches this processor at all,
    // which is a different finding from lanes that are reached but never carry anything.
    DPRINT(
        "[FDS dispatch] idle lanes seen by this processor: group0_status={:#x}\n",
        static_cast<uint32_t>(FDS_INTF_READ(TT_FDS_DISPATCH_GROUPID_STATUS_0__REG_ADDR)));

    if (processor_index == kFirstProcessor) {
        probe_register_interface();
    }

    // Drop any done value a worker was still driving from an earlier epoch. Every wire is cleared
    // because which one the worker sits on is exactly what is under investigation.
    for (uint32_t neo = 0; neo < kNumNeoWires; neo++) {
        overlay::FdsDispatch::fds_clear_neo_status(neo);
    }
    DPRINT("[FDS dispatch] cleared stale done on all {} neo wires\n", kNumNeoWires);

    overlay::FdsDispatch::fds_go(/*ad_enable=*/false, group_id);
    // Reading the go output back shows whether the write reached the outbound register at all,
    // separating a dropped write from a value that is held but never transported.
    DPRINT(
        "[FDS dispatch] sent go: group={} outbox_readback={}\n",
        group_id,
        static_cast<uint32_t>(FDS_INTF_READ(TT_FDS_DISPATCH_DISPATCH_TO_TENSIX_REG_ADDR)));

    // Bounded instead of fds_poll(): the FDS handshake is unproven on Quasar, and an unbounded
    // spin would hang the test rather than report which side failed to signal.
    uint32_t done_count = 0;
    for (uint32_t i = 0; i < poll_iterations; i++) {
        done_count = overlay::FdsDispatch::fds_read_group_count(group_id);
        if (done_count >= done_threshold) {
            DPRINT("[FDS dispatch] received done at iteration {}: count={}\n", i, done_count);
            break;
        }
        if ((i % filter_phase_length) == 0) {
            // Re-arm the deglitcher with the next setting. A done is held once driven, so each
            // setting gets a fresh chance at a signal that is still being driven. The value is
            // read back rather than assumed, so a write that did not take is visible.
            const uint32_t phase = i / filter_phase_length;
            if (phase < quasar_fds_probe::kNumFilterPhases) {
                overlay::FdsDispatch::fds_config_filter_length(quasar_fds_probe::kFilterSweep[phase]);
            }
            // An idle lane presents group 0, so the group 0 status is the mask of done lanes that
            // are currently quiet. A lane that starts driving a real done drops out of that mask,
            // which is how the worker's wire index becomes visible.
            DPRINT(
                "[FDS dispatch] waiting for done: iteration={} deglitcher={} count={} group_status={} "
                "idle_lanes={:#x}\n",
                i,
                static_cast<uint32_t>(FDS_INTF_READ(TT_FDS_DISPATCH_FILTER_COUNT_THRESHOLD_REG_ADDR)),
                done_count,
                static_cast<uint32_t>(
                    FDS_INTF_READ(TT_FDS_DISPATCH_GROUPID_STATUS_0__REG_ADDR + (group_id * sizeof(uint32_t)))),
                static_cast<uint32_t>(FDS_INTF_READ(TT_FDS_DISPATCH_GROUPID_STATUS_0__REG_ADDR)));
            // The worker drives its done part way through its own wait whether or not a go
            // reached it, so a done can show up in an inbox while the group count stays at zero.
            // That is a different defect from no done at all, so watch the raw inboxes too. A
            // done is held once asserted, so scanning at intervals cannot miss one.
            const uint32_t driving_neo = first_driving_neo();
            if (driving_neo < kNumNeoWires) {
                DPRINT(
                    "[FDS dispatch] a neo is driving done at iteration {} while the group count is {}: "
                    "neo={} value={}\n",
                    i,
                    done_count,
                    driving_neo,
                    overlay::FdsDispatch::fds_read_neo_status(driving_neo));
                break;
            }
        }
    }
    // The count may have crossed the threshold between the last read and a break on a raw inbox.
    done_count = overlay::FdsDispatch::fds_read_group_count(group_id);

    if (done_count < done_threshold) {
        DPRINT("[FDS dispatch] timed out waiting for done: final_count={}\n", done_count);
    }
    // Dumped on both paths: a non-zero inbox with a zero group count would mean the done arrived
    // but was not aggregated, which is a different defect from no done at all. Only the first
    // processor dumps, since eight identical listings of 32 quiet lanes would bury everything else.
    if (processor_index == kFirstProcessor) {
        dump_neo_inboxes();
    }

    status[quasar_fds_test::kSlotObserved] = done_count;
    status[quasar_fds_test::kSlotResult] =
        (done_count >= done_threshold) ? quasar_fds_test::kComplete : quasar_fds_test::kTimeout;
    flush_l2_cache_range(status_address, quasar_fds_test::kNumSlots * sizeof(uint32_t));

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

    DPRINT(
        "[FDS dispatch] finished: processor={} result={:#x} done_count={}\n",
        processor_index,
        status[quasar_fds_test::kSlotResult],
        done_count);
}
