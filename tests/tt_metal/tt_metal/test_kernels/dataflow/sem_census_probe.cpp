// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Census / AUTO-classifier probe.
//
// Reports the SCOPE THE HOST BAKED for sem::counter, so a test can assert the classifier's actual
// DECISION rather than merely that the resulting counts happen to be right (every correct mechanism
// produces the same counts, so a behaviour-only test cannot tell them apart -- it would silently
// pass even if AUTO picked the wrong path). The scope is a compile-time constant carried by the
// emitted SemaphoreBindingToken token, so reading it costs nothing and cannot drift from what the kernel
// actually executes.
//
// Also exercises the chosen mechanism: every thread increments `counter` increment_times, then the
// reporter thread publishes the final value. In multi-KERNEL configurations only one kernel is the
// reporter (is_reporter=1) and the count is not meaningful (the other kernel's threads are not part
// of this kernel's barrier) -- such tests assert the reported scope only.
//
// report[0] = baked SemScope of sem::counter (0=LOCAL_NONATOMIC, 1=DM_LOCAL_CACHED, 2=EXTERNAL)
// report[1] = counter.value() after this kernel's threads have all finished incrementing
// report[2] = the RING slot for this semaphore's id, read via the uncached alias. For a cached
//             semaphore this proves RESIDENCY: the count must live in the pool, so the ring slot must
//             still hold the untouched initial value. Nothing else in the suite can tell pool from
//             ring -- every other assertion is count-only and passes under either.
// report[3] = the baked read_only bit (1 iff the host declared this binding AccessType::OBSERVE).
//             Proves the host actually emitted it; without this readback, forgetting to bake the bit
//             would leave OBSERVE enforcement silently inert with every test still green.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t increment_times = get_arg(args::increment_times);
    const uint32_t is_reporter = get_arg(args::is_reporter);
    // Which kernel-barrier slot THIS kernel rendezvouses on. g_kernel_barrier is node-wide and
    // wait_threads() releases on the arriving hart's own participant count, so two co-resident
    // kernels with DIFFERENT thread counts must not share a slot -- they would mix groups and hang
    // at the barrier. The host hands each kernel its own slot.
    const uint32_t barrier_idx = get_arg(args::barrier_idx);

    // NOTE: a DM_LOCAL_CACHED semaphore's pool slot is seeded by sem::init_dm_cached(), which the
    // build AUTO-INJECTS at kernel entry -- no call is needed here.
    Semaphore counter(sem::counter);  // CTAD deduces the host-baked scope AND access rights

    // Guarded on the host's own declaration, not on a separate knob: if the binding is OBSERVE, the
    // mutator is not merely skipped at runtime, it is not compiled at all -- so this kernel cannot
    // contradict the label it was bound with. Before ReadOnly existed, the up() below was compiled
    // unconditionally and the tests that bind this probe OBSERVE relied on passing increment_times=0
    // at RUNTIME to stay honest. That is exactly the gap this parameter closes, and leaving the call
    // unguarded here would now be a hard compile error in those tests.
    if constexpr (!sem::counter.read_only) {
        for (uint32_t i = 0; i < increment_times; i++) {
            counter.up(1);
        }
    }

    // Barrier across THIS kernel's threads, so the reporter sees all of their increments.
    // (No-op when the kernel is single-threaded.)
    sync_threads(barrier_idx);

    if (is_reporter != 0 && get_my_thread_id() == 0u) {
        volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
        // The baked scope: a static constexpr on the emitted SemaphoreBindingToken token.
        report[0] = static_cast<uint32_t>(sem::counter.scope);
        report[1] = counter.value();
        // POSITIVE CONTROL for the ReadOnly plumbing. The dominant failure mode of that feature is
        // not breakage but INERTNESS: several of the host edits compile fine if forgotten and leave
        // read_only=false everywhere, which would enforce nothing while every test stayed green.
        // Reading the baked bit back off the device is what makes that detectable -- a test binding
        // this probe OBSERVE asserts this word is 1.
        report[3] = static_cast<uint32_t>(sem::counter.read_only);
#if defined(ARCH_QUASAR)
        // Residency check: read this id's RING slot directly (uncached alias, so it is TL1 truth).
        // A cached semaphore keeps its count in the pool, so the ring slot must be untouched.
        report[2] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            ::get_semaphore(sem::counter.id) + MEM_L1_UNCACHED_BASE);
        flush_l2_cache_line(report_addr);
        flush_l2_cache_line(report_addr + sizeof(uint32_t));
        flush_l2_cache_line(report_addr + 2 * sizeof(uint32_t));
        flush_l2_cache_line(report_addr + 3 * sizeof(uint32_t));
#else
        report[2] = 0;
#endif
    }
}
