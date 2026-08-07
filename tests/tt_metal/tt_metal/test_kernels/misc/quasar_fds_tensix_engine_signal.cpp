// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tensix-engine side of the Quasar FDS bring-up handshake, for the TRISCs rather than the
// data-movement cores. The data-movement version lives in quasar_fds_worker_signal.cpp and the
// dispatch-engine side in quasar_dispatch_engine_signal.cpp.
//
// The register block is named for the Tensix engine and the dispatch side counts 32 done lanes,
// one per engine across eight tiles, so the engine's own processors are the last candidate
// endpoint after every data-movement core came back identical and idle. This kernel drives done
// unconditionally and holds it, which turns the dispatch engine's group 0 status into a chip-wide
// monitor: any processor that reaches a lane clears a bit there.
//
// Reporting is deliberately sparse. Many of these run at once, and the register that matters is on
// the dispatch engine rather than here.

// Compute kernels take the guarded include and the print header directly rather than dprint.h,
// which is the shape the other Quasar compute kernels use.
#if defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK)
#include "api/compute/common.h"
#endif
#include "api/compile_time_args.h"
#include "api/debug/device_print.h"
#include "dev_mem_map.h"
#include "internal/hw_thread.h"
#include "overlay/fds_functions.hpp"
#include "risc_attribs.h"

#include "quasar_fds_signal_status.h"

namespace {

// One DISPATCH_TO_TENSIX inbox register per dispatch instance on the engine side.
constexpr uint32_t kNumDispatchInstances = 3;

// The deglitcher's reset value, which the sweep on the data-movement side showed makes no
// difference either way. Nothing here needs it to be anything else.
constexpr uint32_t kFilterLength = 0;

// A group the handshake does not use, so the probe below writes to real register addresses without
// disturbing the group under test.
constexpr uint32_t kScratchGroup = 15;

// Indices 8 to 11 are the four TRISCs of the first Tensix engine, which is one of each TRISC role.
// The probe describes the processor rather than the wiring, so one engine's worth answers the
// question and repeating it on all sixteen would only add noise.
constexpr uint32_t kLastProbedProcessor = 11;

// Whether the FDS registers answer on this processor at all. The data-movement cores are built as
// ROCC parts and the Tensix engines are not, so the custom instruction these accessors use may
// belong to an entirely different unit here.
//
// A real register drops the bits outside its field, and the two fields chosen have different
// widths, so a processor holding the block returns 0x7 and 0xff while one that does not returns
// something else. The distinction matters because a custom instruction that does nothing leaves
// the read's destination register untouched, so the answer is stale data rather than zero, which
// is easy to mistake for a register reading as idle.
void probe_registers_answer() {
    const uint32_t enable_address = TT_FDS_TENSIXNEO_GROUPID_ENABLE_0__REG_ADDR + (kScratchGroup * sizeof(uint32_t));
    const uint32_t threshold_address =
        TT_FDS_TENSIXNEO_GROUPID_COUNT_THRESHOLD_0__REG_ADDR + (kScratchGroup * sizeof(uint32_t));

    FDS_INTF_WRITE(enable_address, 0xFFFFFFFF);
    const uint32_t enable = static_cast<uint32_t>(FDS_INTF_READ(enable_address));
    FDS_INTF_WRITE(threshold_address, 0xFFFFFFFF);
    const uint32_t threshold = static_cast<uint32_t>(FDS_INTF_READ(threshold_address));
    FDS_INTF_WRITE(enable_address, 0);
    FDS_INTF_WRITE(threshold_address, 0);

    DEVICE_PRINT(
        "[FDS engine] truncation: enable reads {:#x} where a real one gives {:#x}, threshold reads "
        "{:#x} where a real one gives {:#x}\n",
        enable,
        TT_FDS_TENSIXNEO_GROUPID_ENABLE_DATA_MASK,
        threshold,
        TT_FDS_TENSIXNEO_GROUPID_COUNT_THRESHOLD_DATA_MASK);
    if (enable == TT_FDS_TENSIXNEO_GROUPID_ENABLE_DATA_MASK &&
        threshold == TT_FDS_TENSIXNEO_GROUPID_COUNT_THRESHOLD_DATA_MASK) {
        DEVICE_PRINT("[FDS engine]   both truncated to their fields - this processor has the FDS block\n");
    } else {
        DEVICE_PRINT("[FDS engine]   no truncation - these instructions reach no FDS block here\n");
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t dispatch_mask = get_named_compile_time_arg_val("dispatch_mask");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");

    // One dispatch engine runs in this test, so exactly one instance can ever drive a go.
    constexpr uint32_t go_threshold = 1;
    // Early enough that the dispatch engine is still watching its lanes, and past its inbox clear.
    constexpr uint32_t unprompted_done_iteration = poll_iterations / 8;

    // Indices 8 to 23 are the TRISCs, so these blocks never collide with the data-movement ones.
    // TRISC stores reach L1 through the uncached alias.
    const uint32_t processor_index = internal_::get_hw_thread_idx();
    const uint32_t status_address =
        l1_address + (processor_index * quasar_fds_test::kSlotsPerProcessor * sizeof(uint32_t));
    volatile tt_l1_ptr uint32_t* status =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(status_address + MEM_L1_UNCACHED_BASE);

    status[quasar_fds_test::kSlotStarted] = quasar_fds_test::kStarted;

    overlay::FdsNeo::fds_config_filter_length(kFilterLength);
    overlay::FdsNeo::fds_config_groupid(group_id, dispatch_mask, go_threshold);
    FDS_INTF_WRITE(TT_FDS_TENSIXNEO_AUTO_DISPATCH_EN_REG_ADDR, 0);
    FDS_INTF_WRITE(TT_FDS_TENSIXNEO_INTERRUPT_ENABLE_REG_ADDR, 0);

    // An idle lane presents group 0, so this is the mask of dispatch instances this processor's
    // block is observing. It is the one line worth printing before the wait: if the engine
    // processors differ from the data-movement cores, they differ here.
    DEVICE_PRINT(
        "[FDS engine] started: processor={} group0_status={:#x} enable={:#x}\n",
        processor_index,
        overlay::FdsNeo::fds_read_group_status(0),
        static_cast<uint32_t>(
            FDS_INTF_READ(TT_FDS_TENSIXNEO_GROUPID_ENABLE_0__REG_ADDR + (group_id * sizeof(uint32_t)))));

    if (processor_index <= kLastProbedProcessor) {
        probe_registers_answer();
    }

    overlay::FdsNeo::fds_clear_done();

    uint32_t go_value = 0;
    uint32_t go_instance = kNumDispatchInstances;  // sentinel: no instance delivered a go
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
            break;
        }
        if (i == unprompted_done_iteration) {
            overlay::FdsNeo::fds_done(/*ad_enable=*/false, group_id);
        }
    }

    const bool go_received = (go_instance < kNumDispatchInstances);
    status[quasar_fds_test::kSlotObserved] = go_value;
    status[quasar_fds_test::kSlotResult] = go_received ? quasar_fds_test::kComplete : quasar_fds_test::kTimeout;

    // A lane reaching an engine processor is the finding this run exists to catch, so it gets its
    // own shouted line rather than being one field among many.
    if (go_received) {
        DEVICE_PRINT(
            "[FDS engine] RECEIVED GO: processor={} instance={} value={}\n", processor_index, go_instance, go_value);
    } else {
        DEVICE_PRINT(
            "[FDS engine] finished: processor={} no go, own done output={}\n",
            processor_index,
            static_cast<uint32_t>(FDS_INTF_READ(TT_FDS_TENSIXNEO_TENSIX_TO_DISPATCH_REG_ADDR)));
    }
}
