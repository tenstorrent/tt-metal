// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run (from the tt-metal repo root, after an emule build):
//   build_emule/test/tt_metal/unit_tests_fiber_asan --gtest_filter="EmuleSanitizerFiberState.*"
//
// Structural regression guard for the *per-fiber* ASAN sanitizer state.
//
// The bug this fences: every per-launch sanitizer datum (OOB/padding/semaphore
// range views, CB window counters, Dirty-CB flags, the NoC-read counter, the
// Object-Intent recording pointer, and the diagnostic identity) used to live in
// worker-thread `thread_local`s. Under the cooperative fiber engine many kernels
// multiplex onto one worker thread; a kernel that yields mid-body (e.g. a
// semaphore wait parks the fiber) let a co-scheduled fiber overwrite that state,
// so the first kernel's check read the wrong program's ranges and false-positived
// (the all_reduce / global-semaphore OOB on the 8-chip run). The fix moved the
// state into `EmuleSanitizerState`, a `san` member of `ThreadCommonCtx`, reached
// via `__emule_self` — which the scheduler repoints per fiber on swap, so the
// state travels with the fiber and no fiber can clobber another's.
//
// This test pins that invariant two ways:
//   1. Compile-time: `san` must be a member of the per-fiber context. If someone
//      moves it back to a free `thread_local`, this TU stops compiling.
//   2. Runtime: two contexts hold fully independent `san` state, and switching
//      `__emule_self` exposes each fiber's OWN values untouched by the other —
//      exactly what a co-scheduled-fiber yield relies on.
//
// NOTE: the full end-to-end repro (the deferred multi-chip path where the snapshot
// is also kept alive by g_mesh_oob_keep) needs a real 2-device CCL run — it is
// covered by the loudbox ASAN sweep over tests/blaze/micro_ops/mla/test_all_reduce.py
// (run under TT_METAL_EMULE_ASAN=1). A single-device gtest cannot reproduce it,
// which is precisely why the original gtest suite did not catch the regression.

#include <gtest/gtest.h>
#include <cstdint>

#include "jit_hw/internal/emule_thread_ctx.h"

// Defined in libtt_metal (emulated_program_runner.cpp), the one per-fiber pointer.
extern thread_local ThreadCommonCtx* __emule_self;

namespace {

TEST(EmuleSanitizerFiberState, PerFiberIsolation) {
    // Two independent fiber contexts, as the scheduler owns one per fiber.
    DatamovementThreadCtx fiber_a;
    DatamovementThreadCtx fiber_b;

    static const uint64_t ranges_a[1] = {0x0011002200000000ULL};
    static const uint64_t ranges_b[2] = {0x00aa00bb00000000ULL, 0x00cc00dd00000000ULL};

    ThreadCommonCtx* saved = __emule_self;

    // Arm fiber A's sanitizer state (as set_sanitizer_thread_locals would, in-fiber).
    __emule_self = &fiber_a;
    __emule_self->san.l1_tensor_ranges = ranges_a;
    __emule_self->san.l1_tensor_ranges_count = 1;
    __emule_self->san.sem_l1_range_start = 0xA00;
    __emule_self->san.sem_l1_range_end = 0xA40;
    __emule_self->san.cb_reserved_pages[3] = 7;
    __emule_self->san.cb_reserve_dangling[3] = true;
    __emule_self->san.pending_noc_reads = 5;
    __emule_self->san.kernel_name = "kernel_A";
    __emule_self->san_resolved_active = true;  // Object-Intent recording log (per-fiber)
    __emule_self->san_resolved_count = 3;
    __emule_self->san_resolved_log[0] = 0xdead0000beef0000ULL;

    // A different program arms fiber B — this is the co-scheduled fiber that, under
    // the old thread_local design, would have clobbered A's state on the shared worker.
    __emule_self = &fiber_b;
    __emule_self->san.l1_tensor_ranges = ranges_b;
    __emule_self->san.l1_tensor_ranges_count = 2;
    __emule_self->san.sem_l1_range_start = 0xB00;
    __emule_self->san.sem_l1_range_end = 0xB80;
    __emule_self->san.cb_reserved_pages[3] = 0;
    __emule_self->san.cb_reserve_dangling[3] = false;
    __emule_self->san.pending_noc_reads = 0;
    __emule_self->san.kernel_name = "kernel_B";
    __emule_self->san_resolved_active = false;
    __emule_self->san_resolved_count = 0;

    // Resume fiber A: its state must be exactly what it armed, NOT B's.
    __emule_self = &fiber_a;
    EXPECT_EQ(__emule_self->san.l1_tensor_ranges, ranges_a);
    EXPECT_EQ(__emule_self->san.l1_tensor_ranges_count, 1u);
    EXPECT_EQ(__emule_self->san.sem_l1_range_start, 0xA00u);
    EXPECT_EQ(__emule_self->san.sem_l1_range_end, 0xA40u);
    EXPECT_EQ(__emule_self->san.cb_reserved_pages[3], 7u);
    EXPECT_TRUE(__emule_self->san.cb_reserve_dangling[3]);
    EXPECT_EQ(__emule_self->san.pending_noc_reads, 5u);
    EXPECT_STREQ(__emule_self->san.kernel_name, "kernel_A");
    EXPECT_TRUE(__emule_self->san_resolved_active);
    EXPECT_EQ(__emule_self->san_resolved_count, 3u);
    EXPECT_EQ(__emule_self->san_resolved_log[0], 0xdead0000beef0000ULL);

    // And fiber B keeps its own.
    __emule_self = &fiber_b;
    EXPECT_EQ(__emule_self->san.l1_tensor_ranges, ranges_b);
    EXPECT_EQ(__emule_self->san.l1_tensor_ranges_count, 2u);
    EXPECT_EQ(__emule_self->san.sem_l1_range_start, 0xB00u);
    EXPECT_EQ(__emule_self->san.cb_reserved_pages[3], 0u);
    EXPECT_FALSE(__emule_self->san.cb_reserve_dangling[3]);
    EXPECT_EQ(__emule_self->san.pending_noc_reads, 0u);
    EXPECT_STREQ(__emule_self->san.kernel_name, "kernel_B");
    EXPECT_FALSE(__emule_self->san_resolved_active);
    EXPECT_EQ(__emule_self->san_resolved_count, 0u);

    __emule_self = saved;
}

// A fresh context starts with cleared sanitizer state (the checks early-out on
// null ranges when ASAN never armed them). Guards the default-init contract that
// clear_sanitizer_thread_locals also restores.
TEST(EmuleSanitizerFiberState, DefaultConstructedIsInert) {
    DatamovementThreadCtx fiber;
    EXPECT_EQ(fiber.san.l1_tensor_ranges, nullptr);
    EXPECT_EQ(fiber.san.l1_tensor_ranges_count, 0u);
    EXPECT_EQ(fiber.san.sem_l1_range_end, 0u);
    EXPECT_EQ(fiber.san.pending_noc_reads, 0u);
    EXPECT_EQ(fiber.san.cb_reserved_pages[0], 0u);
    EXPECT_FALSE(fiber.san.cb_boundary_strict);
    EXPECT_EQ(fiber.san.kernel_name, nullptr);
    EXPECT_FALSE(fiber.san_resolved_active);
    EXPECT_EQ(fiber.san_resolved_count, 0u);
}

}  // namespace
