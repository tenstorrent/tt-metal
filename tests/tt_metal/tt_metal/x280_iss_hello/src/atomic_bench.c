/*
 * atomic_bench.c - X280 cross-hart atomic-increment micro-benchmark.
 *
 * ======================================================================
 * WHAT THIS IS FOR
 * ======================================================================
 *
 * Prototype for the real use case: 4 X280 harts on one L2CPU tile all
 * pushing packets into a *single* downstream queue (e.g. an erisc queue)
 * and each needing to bump a shared write-pointer without colliding.
 * The correctness primitive for that is an atomic read-modify-write:
 * each producer atomically adds its packet count to the shared write
 * pointer and gets back a unique, non-overlapping slot range.
 *
 * Here we boil that down to the simplest measurable kernel: N harts each
 * atomically increment ONE shared memory location 1,000,000 times. From
 * that we learn three things:
 *
 *   1. CORRECTNESS. With true atomics the final value must equal
 *      N * iterations exactly (no lost updates). We also run a
 *      deliberately NON-atomic (racy) mode so you can *see* updates get
 *      lost when the atomic guarantee is removed -- that's the "why we
 *      need atomics" demonstration.
 *
 *   2. UNCONTENDED COST. Phase 1 runs a single hart alone. Its
 *      cycles/increment is the baseline: the raw cost of one atomic op
 *      with zero contention.
 *
 *   3. CONTENTION / STALL OVERHEAD. Phases with 2/3/4 harts hammer the
 *      same address concurrently. The memory subsystem must serialize
 *      the atomics, so each hart's cycles/increment rises above the
 *      baseline. The increase is the contention (stall) overhead, and it
 *      tells you how the write-pointer scheme will scale with producers.
 *
 * ======================================================================
 * TWO HARDWARE FACTS THAT SHAPE THIS BENCHMARK
 * ======================================================================
 *
 * (A) The 4 X280 harts are a *cache-coherent cluster*, but LIM is
 *     *uncached scratchpad* (L3 configured as SRAM). See the Tenstorrent
 *     tt-isa-documentation, BlackholeA0/L2CPUTile/README.md + Caches.md.
 *     Because LIM is uncached, an atomic to a LIM address does NOT bounce
 *     a cache line between harts -- every hart's AMO goes straight to the
 *     LIM/L3 controller, which serializes them. That makes LIM a clean,
 *     predictable place to measure raw atomic serialization (no MESI
 *     line-migration noise). Whether the L2CPU actually implements AMOs
 *     against uncached LIM is silicon-specific, so we PROBE for it at
 *     startup (see amo_probe below) instead of assuming it.
 *
 * (B) TIMING -- CLINT mtime (true wall-clock).
 *     Per hart we bracket the timed loop with the CLINT `mtime` counter, a
 *     true 50 MHz wall-clock (20 ns/tick) that keeps ticking through
 *     stalls:
 *         wall_ticks = mtime_end - mtime_start
 *     This is what a producer actually pays, including time blocked waiting
 *     for the atomic subsystem to serialize it. The host converts ticks to
 *     microseconds (ticks/50e6) and, using the PLL frequency, to
 *     cycles-per-op. We deliberately do NOT use rdcycle: on this X280 it
 *     only counts active-pipeline cycles and freezes during memory stalls
 *     (see x280_mtime in rt/x280_hw.h), so it would hide exactly the
 *     atomic-latency cost we want to see. mtime has no such gap.
 *
 * ======================================================================
 * EXECUTION MODEL (ping-pong active FW, LIM mode)
 * ======================================================================
 *
 * This is an *active* firmware in the resident-idle / active-FW ping-pong
 * scheme (see x280/MIGRATION_WORKER_LAYOUT.md §5.0 and x280/src/lim_idle.c).
 * The host never touches the L2CPU reset bit; the idle FW indirect-jumps
 * here and all 4 harts land in main(hartid) via entry.S.
 *
 *   hart 0  = orchestrator. Talks to the host over LIM mailboxes, runs
 *             each phase, collects results, and (unlike socket_echo)
 *             KEEPS harts 1-3 resident so they can participate.
 *   hart 1-3 = workers. They enter atomic_worker_loop() and wait for
 *             hart 0 to hand them per-phase work via WORK[h]. They only
 *             return to the idle FW when hart 0 posts WORK_SHUTDOWN.
 *
 * Per phase (num = 1..4 participating harts), the barrier is:
 *   hart 0: zero counter; clear GATE; post WORK[h]=iters to helpers;
 *           wait until every helper set ARMED[h]=1 (spinning on GATE);
 *           flip GATE=GO; run its OWN timed loop; wait every DONE[h];
 *           record results; read final counter.
 *   helper: sees WORK[h]=iters; sets ARMED[h]=1; spins on GATE; on GO,
 *           runs its timed loop; publishes (active,wall) to DONE[h].
 * The GATE gives all participants a near-simultaneous start so the
 * contention window overlaps.
 *
 * Linker: ld/x280.ld (active-FW LIM region at 0x08001000).
 * Build:  make -C x280 atomic-bench  ->  build/firmware-atomic-bench.bin
 *
 * NOTE ON RECOVERY: if AMOs against LIM are unsupported, the amo_probe
 * step detects it *without* faulting the core (it arms entry.S probe
 * mode), reports AMO_UNSUPPORTED, and cleanly returns to the idle FW.
 * We never let an unprobed AMO hit the normal trap handler, which would
 * CEASE (permanently halt) the hart and strand the L2CPU until a chip
 * reset.
 */
#include "x280.h"

#include "iss_printf.h"
#include "rt/x280_atomic.h"
#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "rt/x280_pll.h"
/* ======================================================================
 * LIM control / result layout  (WIRE CONTRACT with the host driver)
 * ----------------------------------------------------------------------
 * A dedicated region well clear of everything else in LIM:
 *   - firmware .text/.data/.bss           < 0x120000
 *   - migration control + boot handshake  0x130000..0x131FFF
 *   - bank table / slots / staging / fifo  0x132000..0x153FFF
 * We claim [0x160000, 0x163000). It is below the stack (0x1D8000) and
 * does NOT overlap the idle-FW boot-handshake mailboxes (0x130200..),
 * so priming these lines can never wipe the idle heartbeat.
 *
 * Every field lives on its OWN 64-byte cache line. Two reasons, both the
 * same as the migration worker's mailboxes:
 *   - ECC: LIM SRAM is ECC-protected; a partial-word write to a cold
 *     line needs valid ECC in the rest of the line. The host primes each
 *     line with a full 64 B zero write before any partial write.
 *   - No false sharing: control words written by different harts, and
 *     especially the hot shared COUNTER, must not share a line or the
 *     measurement would pick up incidental line traffic.
 *
 * If you change any offset here, change it in
 * x280/host/experiment_atomic_bench.py in the SAME commit.
 * ====================================================================== */
#define ATOMIC_BASE (LIM_BASE + 0x00160000UL)

/* --- control mailboxes -------------------------------------------- */
#define AB_STATUS_ADDR ((volatile uint32_t*)(ATOMIC_BASE + 0x000UL))       /* fw->host */
#define AB_STOP_ADDR ((volatile uint32_t*)(ATOMIC_BASE + 0x040UL))         /* host->fw */
#define AB_CONFIG_READY_ADDR ((volatile uint32_t*)(ATOMIC_BASE + 0x080UL)) /* host->fw */
#define AB_CUR_PHASE_ADDR ((volatile uint32_t*)(ATOMIC_BASE + 0x0C0UL))    /* fw->host progress */

/* --- config block (host->fw, latched after CONFIG_READY) ---------- */
#define AB_ITERATIONS_ADDR ((volatile uint64_t*)(ATOMIC_BASE + 0x100UL))   /* per-hart iters */
#define AB_OP_MODE_ADDR ((volatile uint32_t*)(ATOMIC_BASE + 0x140UL))      /* AB_OP_* */
#define AB_PHASE_MASK_ADDR ((volatile uint32_t*)(ATOMIC_BASE + 0x180UL))   /* bit p => run (p+1) harts */
#define AB_COUNTER_ADDR_ADDR ((volatile uint64_t*)(ATOMIC_BASE + 0x1C0UL)) /* LIM addr of shared counter */

/* --- per-hart barrier mailboxes (each on its own line) ------------ */
/* WORK[h]: hart0 -> hart h.  0 = idle, N = run N iters, 0xFFFFFFFF = shutdown. */
#define AB_WORK_BASE (ATOMIC_BASE + 0x200UL)
#define AB_WORK_ADDR(h) ((volatile uint32_t*)(AB_WORK_BASE + (uint64_t)(h) * 0x40UL))
/* ARMED[h]: hart h -> hart0.  1 = latched work and spinning on GATE. */
#define AB_ARMED_BASE (ATOMIC_BASE + 0x300UL)
#define AB_ARMED_ADDR(h) ((volatile uint32_t*)(AB_ARMED_BASE + (uint64_t)(h) * 0x40UL))
/* DONE[h] line: hart h -> hart0.  wall_ticks(u64)@+0, slot_sum(u64)@+8,
 * flag(u32)@+16 (written last). */
#define AB_DONE_BASE (ATOMIC_BASE + 0x400UL)
#define AB_DONE_ADDR(h) (AB_DONE_BASE + (uint64_t)(h) * 0x40UL)
#define AB_DONE_WALL(h) ((volatile uint64_t*)(AB_DONE_ADDR(h) + 0x00UL))
#define AB_DONE_SLOTSUM(h) ((volatile uint64_t*)(AB_DONE_ADDR(h) + 0x08UL))
#define AB_DONE_FLAG(h) ((volatile uint32_t*)(AB_DONE_ADDR(h) + 0x10UL))
/* Absolute mtime timestamps (same global 50 MHz counter every hart reads,
 * so they are comparable ACROSS harts). Used to verify the harts actually
 * overlapped -- see the per-phase span table below. */
#define AB_DONE_T0(h) ((volatile uint64_t*)(AB_DONE_ADDR(h) + 0x18UL))
#define AB_DONE_T1(h) ((volatile uint64_t*)(AB_DONE_ADDR(h) + 0x20UL))

/* GATE: hart0 -> all participants. Flipped to AB_GATE_GO to release. */
#define AB_GATE_ADDR ((volatile uint32_t*)(ATOMIC_BASE + 0x600UL))

/* --- results tables (fw->host) ------------------------------------ *
 * The benchmark runs TWO sections, each with its own results table:
 *   Section 0 "constant":  every participating hart does `iters`, so the
 *                          total work grows with hart count (N*iters).
 *   Section 1 "split":     the SAME total `iters` is divided evenly, so
 *                          N harts do ~iters/N each (strong scaling).
 *
 * One 32 B record per (phase p in 0..3, hart h in 0..3):
 *   wall_ticks(u64) @ +0, iters_done(u64) @ +8, slot_sum(u64) @ +16.
 * iters_done lets the host compute per-op costs without re-deriving the
 * (uneven) split; slot_sum (sum of the old values this hart got back)
 * feeds the slot-uniqueness check. Record address =
 * <section_base> + (p*4 + h) * 32. */
#define AB_RESULT_REC_SIZE 32u
#define AB_RESULT_BASE (ATOMIC_BASE + 0x1000UL)  /* section 0 records */
#define AB_RESULT2_BASE (ATOMIC_BASE + 0x1200UL) /* section 1 records */
#define AB_REC_WALL(base, p, h) ((volatile uint64_t*)((base) + ((uint64_t)(p) * 4 + (h)) * AB_RESULT_REC_SIZE + 0))
#define AB_REC_ITERS(base, p, h) ((volatile uint64_t*)((base) + ((uint64_t)(p) * 4 + (h)) * AB_RESULT_REC_SIZE + 8))
#define AB_REC_SLOTSUM(base, p, h) ((volatile uint64_t*)((base) + ((uint64_t)(p) * 4 + (h)) * AB_RESULT_REC_SIZE + 16))
/* Per-phase final counter + expected value: 16 B per phase, per section. */
#define AB_FINAL_BASE (ATOMIC_BASE + 0x1400UL)  /* section 0 finals */
#define AB_FINAL2_BASE (ATOMIC_BASE + 0x1440UL) /* section 1 finals */
#define AB_FGOT(base, p) ((volatile uint64_t*)((base) + (uint64_t)(p) * 16 + 0))
#define AB_FEXP(base, p) ((volatile uint64_t*)((base) + (uint64_t)(p) * 16 + 8))
/* Per-phase concurrency span (fw->host): the earliest loop-start and latest
 * loop-end absolute mtime across the phase's participating harts. The host
 * uses (max_end - min_start) as the TRUE concurrent wall-clock and compares
 * it to the slowest single-hart delta -- if they match, the harts really
 * overlapped and the reported throughput is honest; if max_end-min_start is
 * much larger, the harts were staggered/serialized and throughput was
 * over-stated. 16 B per phase (min_start@0, max_end@8), per section. */
#define AB_SPAN_BASE (ATOMIC_BASE + 0x1480UL)  /* section 0 spans */
#define AB_SPAN2_BASE (ATOMIC_BASE + 0x14C0UL) /* section 1 spans */
#define AB_SPAN_START(base, p) ((volatile uint64_t*)((base) + (uint64_t)(p) * 16 + 0))
#define AB_SPAN_END(base, p) ((volatile uint64_t*)((base) + (uint64_t)(p) * 16 + 8))
/* AMO-to-LIM probe result: 1 = AMOs work, 0 = they fault. */
#define AB_AMO_SUPPORTED_ADDR ((volatile uint32_t*)(ATOMIC_BASE + 0x1500UL))
/* Firmware-measured X280 PLL frequency in kHz (fw->host). Measured once at
 * startup by counting rdcycle ticks against the fixed 50 MHz CLINT mtime
 * reference -- so the host can convert cycle counts and CONFIRM the real
 * clock instead of trusting --mhz. Its own 64 B line. */
#define AB_MEASURED_PLL_KHZ_ADDR ((volatile uint32_t*)(ATOMIC_BASE + 0x1540UL))

/* --- protocol constants ------------------------------------------- */
#define AB_WORK_IDLE 0x00000000u
#define AB_WORK_SHUTDOWN 0xFFFFFFFFu
#define AB_GATE_GO 0x600D600Du /* "GOOD GOOD" */
#define AB_DONE_FLAG_VAL 0xD01E0001u
#define AB_STOP_VALUE 0x0000DEADu
#define AB_CONFIG_READY_VALUE 0x000C0FFEu

/* Op modes: what "increment" means. Host selects via AB_OP_MODE.
 * (Value 2 was LR/SC; removed -- reservations don't work on uncached LIM,
 * so sc.d never succeeds and the retry loop live-locks / strands the core.
 * NONATOMIC keeps value 3 so the wire codes stay stable.) */
#define AB_OP_AMOADD_W 0u    /* amoadd.w  (32-bit atomic)          */
#define AB_OP_AMOADD_D 1u    /* amoadd.d  (64-bit atomic)          */
#define AB_OP_NONATOMIC_D 3u /* ld/add/sd (racy, NOT atomic)       */

/* Status lifecycle (fw->host). Monotonic-ish; host polls to sequence. */
#define AB_STATUS_INIT 0xFFFFFFFFu      /* main() up, before config   */
#define AB_STATUS_READY 0x10000001u     /* config latched, probing    */
#define AB_STATUS_RUNNING 0x10000002u   /* phases executing           */
#define AB_STATUS_DONE 0x10000003u      /* all phases done, clean exit */
#define AB_STATUS_AMO_UNSUP 0xE0000001u /* AMO probe failed; aborting  */

/* NUM_HARTS (== 4) comes from x280.h. */
/* ======================================================================
 * Low-level primitives
 * ====================================================================== */

/* Measure the X280 PLL frequency (kHz) by counting rdcycle ticks over a
 * fixed mtime window. PLL_hz = (dcyc / dtick) * 50e6, so
 * PLL_kHz = dcyc * 50000 / dtick. Window = 100000 mtime ticks (~2 ms at
 * 50 MHz) -- long enough that fixed read overhead is negligible. Same trick
 * idle.c / read_x280_clock.py use. Called once by hart 0 before the phases;
 * cost (~2 ms) is outside every timed region. */

/* ---- the increment kernels ---------------------------------------- *
 * Each does exactly ONE increment-by-1 of *p and RETURNS THE OLD VALUE
 * (the value before the add). This mirrors the real use case: a producer
 * does `my_slot = atomic_add(&write_ptr, 1)` and the returned old pointer
 * IS the slot it just claimed. We deliberately do NOT discard it (no
 * rd=x0) -- the loop below accumulates the returned slots, both to model
 * the real "use the value" cost and to enable the slot-sum correctness
 * check (see run_increment_loop).
 *
 * They take a uintptr_t rather than a typed pointer because the op is
 * selected at run time from AB_OP_MODE, so one address feeds both widths.
 *
 * Each is force-inlined so per-call overhead is identical across harts;
 * the "memory" clobber + volatile stop the compiler reordering/eliding
 * the op or the load of its result.                                    */

static inline uint64_t inc_amoadd_w(uintptr_t p) { return x280_amoadd_w((volatile uint32_t*)p, 1u); }

static inline uint64_t inc_amoadd_d(uintptr_t p) { return x280_amoadd_d((volatile uint64_t*)p, 1u); }

static inline uint64_t inc_nonatomic_d(uintptr_t p) {
    /* Deliberately racy: plain load, add, store. With multiple harts, two
     * can read the same old value and both write old+1, LOSING one update
     * AND handing the SAME slot to two producers. Used to demonstrate why
     * atomics are required -- the final count comes out LESS than expected
     * and the slot-sum check fails (duplicate/again-issued slots). */
    uint64_t old, newv;
    __asm__ volatile(
        "  ld   %0, 0(%2)\n"
        "  addi %1, %0, 1\n"
        "  sd   %1, 0(%2)\n"
        : "=&r"(old), "=&r"(newv)
        : "r"(p)
        : "memory");
    return old;
}

/* Run `iters` increments of the given op against `counter_addr`, and
 * return the SUM of all the old values returned (the slots this hart
 * claimed). Kept in one place so hart 0 and the helpers execute
 * byte-identical loops.
 *
 * Why sum the slots: with a correct atomic increment-by-1, the old values
 * returned across ALL harts over the whole phase are exactly the set
 * {0, 1, ..., total-1}, each handed out once. So the sum of every hart's
 * partial slot-sum must equal total*(total-1)/2. That single number is a
 * strong end-to-end proof that no slot was ever issued twice (the exact
 * property the downstream-queue write pointer needs) -- stronger than just
 * "final counter == total". The non-atomic mode breaks it (duplicated
 * slots), which is the point. Summing also gives the compiler a reason to
 * keep each returned value, faithfully modelling code that USES the slot.*/
static uint64_t run_increment_loop(unsigned op, uintptr_t counter_addr, uint64_t iters) {
#ifdef X280_ISS
    if (iters > 10000ULL) {
        iters = 10000ULL;
    }
    uint64_t slot_sum = 0;
    volatile uint64_t* p = (volatile uint64_t*)counter_addr;
    for (uint64_t i = 0; i < iters; i++) {
        uint64_t old;
        if (op == AB_OP_AMOADD_W) {
            old = x280_amoadd_w((volatile uint32_t*)p, 1u);
        } else if (op == AB_OP_AMOADD_D) {
            __asm__ volatile("amoadd.d %0, %2, (%1)" : "=r"(old) : "r"(p), "r"(1ULL) : "memory");
        } else {
            old = *p;
            *p = old + 1;
        }
        slot_sum += old;
    }
    return slot_sum;
#else
    uint64_t slot_sum = 0;
    switch (op) {
        case AB_OP_AMOADD_W:
            for (uint64_t i = 0; i < iters; i++) {
                slot_sum += inc_amoadd_w(counter_addr);
            }
            break;
        case AB_OP_AMOADD_D:
            for (uint64_t i = 0; i < iters; i++) {
                slot_sum += inc_amoadd_d(counter_addr);
            }
            break;
        case AB_OP_NONATOMIC_D:
        default:
            for (uint64_t i = 0; i < iters; i++) {
                slot_sum += inc_nonatomic_d(counter_addr);
            }
            break;
    }
    return slot_sum;
#endif
}

/* ======================================================================
 * AMO-to-LIM support probe (fault-safe)
 * ----------------------------------------------------------------------
 * We must NOT let an unsupported AMO hit entry.S's normal trap handler:
 * that writes mcause/mepc/mtval and executes CEASE, permanently halting
 * the hart (only a chip reset recovers). entry.S has a "probe mode":
 * while _probe_active != 0, load/store/AMO access faults (cause 5/7) and
 * illegal-instruction (cause 2) are skipped (mepc+=4) and _probe_active
 * is set to 2. So: arm probe mode, attempt one op, and check whether it
 * faulted. Returns 1 if the op executed cleanly, 0 if it faulted.
 * ====================================================================== */
static int amo_probe(unsigned op, uintptr_t counter_addr) {
    _probe_active = 1; /* arm: entry.S will skip faults */
    __asm__ volatile("fence ow, ow");
    switch (op) {
        case AB_OP_AMOADD_W: inc_amoadd_w(counter_addr); break;
        case AB_OP_AMOADD_D: inc_amoadd_d(counter_addr); break;
        default: inc_nonatomic_d(counter_addr); break; /* always OK */
    }
    __asm__ volatile("fence iorw, iorw");
    int faulted = (_probe_active == 2);
    _probe_active = 0; /* disarm */
    __asm__ volatile("fence ow, ow");
    return faulted ? 0 : 1;
}

/* ======================================================================
 * Exit paths (ping-pong: hand control back to the resident idle FW)
 * ====================================================================== */

/* hart 0 clean exit: stamp the legacy sentinel + PHASE=RETURNED_TO_IDLE
 * (so the host's poll_sentinel / wait_active_fw_returned works), then
 * indirect-jump to the idle FW _start. Mirrors socket_echo.cpp. */

/* helper exit: jump straight back to the idle FW's wake-poll loop WITHOUT
 * touching the sentinel/phase (those are hart 0's job). WFI is not an
 * option -- it hard-stalls this core forever (host NOC writes raise no
 * interrupt), stranding the L2CPU. */

static inline void set_status(uint32_t s) {
    *AB_STATUS_ADDR = s;
    __asm__ volatile("fence ow, ow");
}

/* ======================================================================
 * Helper hart (1..3) main loop
 * ----------------------------------------------------------------------
 * Park polling WORK[h]. On a positive iteration count, do the barrier
 * handshake and one timed run; on SHUTDOWN, go home. All mailbox lines
 * were host-primed to 0 before the handoff, so an initial WORK[h]==0
 * (idle) read is well-defined -- no init-order dependency on hart 0.
 * ====================================================================== */
static void atomic_worker_loop(uint32_t hartid) {
    volatile uint32_t* my_work = AB_WORK_ADDR(hartid);
    volatile uint32_t* my_armed = AB_ARMED_ADDR(hartid);
    volatile uint64_t* my_wall = AB_DONE_WALL(hartid);
    volatile uint32_t* my_flag = AB_DONE_FLAG(hartid);
    volatile uint64_t* my_slotsum = AB_DONE_SLOTSUM(hartid);
    volatile uint64_t* my_t0 = AB_DONE_T0(hartid);
    volatile uint64_t* my_t1 = AB_DONE_T1(hartid);

    for (;;) {
#ifndef X280_ISS
        __asm__ volatile("fence ir, ir");
#endif
        uint32_t work = *my_work;

        if (work == AB_WORK_IDLE) {
            x280_pause();
            continue;
        }
        if (work == AB_WORK_SHUTDOWN) {
            x280_helper_to_idle(); /* noreturn */
        }

        /* work is a positive iteration count. Latch the run parameters
         * (op + counter address were published by hart 0 with the config
         * and never change during a run). */
        uint64_t iters = *my_work; /* == work */
        unsigned op = *AB_OP_MODE_ADDR;
        uintptr_t caddr = (uintptr_t)*AB_COUNTER_ADDR_ADDR;

        /* Announce we are armed and about to spin on the gate, so hart 0
         * only releases once every participant is ready (a tight start
         * window => the contention actually overlaps). */
        *my_armed = 1u;
        __asm__ volatile("fence ow, ow");

        /* Spin on the gate. Tight (no pause) so we react to GO with
         * minimum latency, minimizing start skew vs the other harts. */
        for (;;) {
#ifndef X280_ISS
            __asm__ volatile("fence ir, ir");
#endif
            if (*AB_GATE_ADDR == AB_GATE_GO) {
                break;
            }
        }

        /* --- timed region --- */
        uint64_t t0 = x280_mtime();
        uint64_t slot_sum = run_increment_loop(op, caddr, iters);
        uint64_t t1 = x280_mtime();

        /* Publish (wall, slot_sum, absolute t0/t1) BEFORE the done flag so
         * hart 0 sees valid numbers the instant it observes the flag. */
        *my_wall = t1 - t0;
        *my_slotsum = slot_sum;
        *my_t0 = t0;
        *my_t1 = t1;
        __asm__ volatile("fence ow, ow");
        *my_armed = 0u;
        *my_work = AB_WORK_IDLE; /* return to idle-poll for next phase */
        __asm__ volatile("fence ow, ow");
        *my_flag = AB_DONE_FLAG_VAL;
        __asm__ volatile("fence ow, ow");
    }
}

/* ======================================================================
 * hart 0: run one phase with `num` participating harts (0..num-1)
 * ----------------------------------------------------------------------
 * iters_arr[h] is the per-hart iteration count (they may differ, e.g. the
 * "split" section gives hart 0 the remainder). `expected` is the total the
 * counter should reach. `result_base` / `final_base` / `span_base` select
 * which section's tables this phase writes into.
 * ====================================================================== */
static void run_phase(
    unsigned phase_idx,
    unsigned num,
    unsigned op,
    uintptr_t caddr,
    const uint64_t iters_arr[NUM_HARTS],
    uint64_t expected,
    uintptr_t result_base,
    uintptr_t final_base,
    uintptr_t span_base) {
    const unsigned n = num;
    volatile uint64_t* counter = (volatile uint64_t*)caddr;

    *AB_CUR_PHASE_ADDR = phase_idx;
    __asm__ volatile("fence ow, ow");

    /* Reset the shared counter (line already ECC-primed by the host, so a
     * plain 8-byte store is safe). Zero it fully as u64 even for the
     * 32-bit amoadd.w mode so stale high bits can't confuse the check. */
    *counter = 0;
    *AB_GATE_ADDR = 0u;
    __asm__ volatile("fence ow, ow");

    /* Arm helpers 1..num-1 with their individual iteration counts. Clear
     * DONE flag first so a stale flag from a previous phase can't be
     * mistaken for this phase's completion. */
    for (unsigned h = 1; h < n; h++) {
        *AB_DONE_FLAG(h) = 0u;
        *AB_ARMED_ADDR(h) = 0u;
        __asm__ volatile("fence ow, ow");
        *AB_WORK_ADDR(h) = (uint32_t)iters_arr[h];
    }
    __asm__ volatile("fence ow, ow");

    /* Wait until every participating helper is armed & spinning on GATE. */
    for (unsigned h = 1; h < n; h++) {
        for (;;) {
#ifndef X280_ISS
            __asm__ volatile("fence ir, ir");
#endif
            if (*AB_ARMED_ADDR(h) == 1u) {
                break;
            }
        }
    }

    /* Release everyone, then immediately run hart 0's own timed loop. */
    *AB_GATE_ADDR = AB_GATE_GO;
    __asm__ volatile("fence ow, ow");

    uint64_t t0 = x280_mtime();
    uint64_t slot_sum0 = run_increment_loop(op, caddr, iters_arr[0]);
    uint64_t t1 = x280_mtime();

    *AB_REC_WALL(result_base, phase_idx, 0) = t1 - t0;
    *AB_REC_ITERS(result_base, phase_idx, 0) = iters_arr[0];
    *AB_REC_SLOTSUM(result_base, phase_idx, 0) = slot_sum0;

    /* Track the true concurrent window across all participants: earliest
     * start and latest end on the shared mtime timebase. Seed with hart 0. */
    uint64_t span_start = t0;
    uint64_t span_end = t1;

    /* Collect helper results. */
    for (unsigned h = 1; h < n; h++) {
        for (;;) {
#ifndef X280_ISS
            __asm__ volatile("fence ir, ir");
#endif
            if (*AB_DONE_FLAG(h) == AB_DONE_FLAG_VAL) {
                break;
            }
        }
        *AB_REC_WALL(result_base, phase_idx, h) = *AB_DONE_WALL(h);
        *AB_REC_ITERS(result_base, phase_idx, h) = iters_arr[h];
        *AB_REC_SLOTSUM(result_base, phase_idx, h) = *AB_DONE_SLOTSUM(h);
        uint64_t h_t0 = *AB_DONE_T0(h);
        uint64_t h_t1 = *AB_DONE_T1(h);
        if (h_t0 < span_start) {
            span_start = h_t0;
        }
        if (h_t1 > span_end) {
            span_end = h_t1;
        }
    }

    /* Publish the concurrent span so the host can check real overlap. */
    *AB_SPAN_START(span_base, phase_idx) = span_start;
    *AB_SPAN_END(span_base, phase_idx) = span_end;

    /* Correctness: read back the final counter and record expected. For
     * amoadd.w only the low 32 bits are meaningful; mask so the host
     * compares apples to apples. */
    uint64_t got = *counter;
    if (op == AB_OP_AMOADD_W) {
        got &= 0xFFFFFFFFULL;
    }
    *AB_FGOT(final_base, phase_idx) = got;
    *AB_FEXP(final_base, phase_idx) = expected;
    __asm__ volatile("fence ow, ow");
}

/* Fill iters_arr[0..num-1] for a section and return the expected total.
 *   section 0 (constant): every hart does `iters`; total = num*iters.
 *   section 1 (split):    the SAME `iters` total split evenly; hart h does
 *                         iters/num, with the remainder handed to the low
 *                         harts so the shares sum EXACTLY to `iters`. */
static uint64_t fill_iters(unsigned section, unsigned num, uint64_t iters, uint64_t iters_arr[NUM_HARTS]) {
    if (section == 0) {
        for (unsigned h = 0; h < num; h++) {
            iters_arr[h] = iters;
        }
        return iters * (uint64_t)num;
    }
    uint64_t base = iters / num;
    uint64_t rem = iters % num;
    for (unsigned h = 0; h < num; h++) {
        iters_arr[h] = base + (h < rem ? 1u : 0u);
    }
    return iters; /* == base*num + rem */
}

/* ======================================================================
 * main -- entry.S calls this on ALL harts with a0 = mhartid
 * ====================================================================== */
int main(uint64_t hartid) {
    /* Workers peel off immediately into their wait loop. */
    if (hartid != 0) {
        atomic_worker_loop((uint32_t)hartid); /* noreturn */
        return 0;                             /* unreachable */
    }

    /* --- hart 0 --- */

    /* Announce the active FW is running (host handoff waits on this). */
    *(volatile uint64_t*)X280_BOOT_PHASE_ADDR = X280_BOOT_PHASE_RUNNING_ACTIVE_FW;
    __asm__ volatile("fence ow, ow");

    set_status(AB_STATUS_INIT);

    /* Wait for the host to publish config (iterations, op, counter addr,
     * phase mask). STOP-interruptible so an aborted run never strands
     * the core out of reset. */
    for (;;) {
        __asm__ volatile("fence ir, ir");
        if (*AB_CONFIG_READY_ADDR == AB_CONFIG_READY_VALUE) {
            break;
        }
        if (*AB_STOP_ADDR == AB_STOP_VALUE) {
            /* Release helpers, then go home. */
            for (unsigned h = 1; h < NUM_HARTS; h++) {
                *AB_WORK_ADDR(h) = AB_WORK_SHUTDOWN;
            }
            __asm__ volatile("fence ow, ow");
            set_status(AB_STATUS_DONE);
            x280_hart0_to_idle();
        }
    }

    printf("AB_CONFIG_READY: config latched on ISS\n");

    /* Latch config. */
    uint64_t iters = *AB_ITERATIONS_ADDR;
    unsigned op = *AB_OP_MODE_ADDR;
    unsigned phase_mask = *AB_PHASE_MASK_ADDR;
    uintptr_t caddr = (uintptr_t)*AB_COUNTER_ADDR_ADDR;
    if (iters == 0) {
        iters = 1000000ULL; /* default 1M */
    }
    if (phase_mask == 0) {
        phase_mask = 0xF; /* default: 1,2,3,4 harts */
    }
    if (caddr == 0) {
        caddr = ATOMIC_BASE + 0x2000UL; /* default counter slot */
    }

    printf("  iters=%lu op=%u phase_mask=0x%x\n", (unsigned long)iters, op, phase_mask);

#ifdef X280_ISS
    {
        volatile uint64_t* p = (volatile uint64_t*)caddr;
        uint64_t n = iters < 8ULL ? iters : 8ULL;
        *p = 0;
        for (uint64_t i = 0; i < n; i++) {
            uint64_t old;
            __asm__ volatile("amoadd.d %0, %2, (%1)" : "=r"(old) : "r"(p), "r"(1ULL) : "memory");
            (void)old;
        }
        printf("  amoadd.d x%lu -> counter=%lu %s\n", (unsigned long)n, (unsigned long)*p, *p == n ? "OK" : "FAIL");
        set_status(AB_STATUS_DONE);
        x280_hart0_to_idle();
    }
#endif

    set_status(AB_STATUS_READY);

    /* Probe AMO-to-LIM support before we run any untrapped atomic. If the
     * selected op is atomic and faults, report and bail cleanly. */
    int amo_ok = amo_probe(op, caddr);
    *AB_AMO_SUPPORTED_ADDR = (uint32_t)amo_ok;
    __asm__ volatile("fence ow, ow");
    if (!amo_ok && op != AB_OP_NONATOMIC_D) {
        for (unsigned h = 1; h < NUM_HARTS; h++) {
            *AB_WORK_ADDR(h) = AB_WORK_SHUTDOWN;
        }
        __asm__ volatile("fence ow, ow");
        set_status(AB_STATUS_AMO_UNSUP);
        x280_hart0_to_idle();
    }

    /* Measure the actual PLL (against the 50 MHz mtime ref) and publish it,
     * so the host converts cycle counts with the REAL clock and can confirm
     * whether we're at 1000 or 1750 MHz -- no reliance on --mhz. One-time,
     * before any timed region. */
    *AB_MEASURED_PLL_KHZ_ADDR = x280_measure_pll_khz();
    __asm__ volatile("fence ow, ow");

    set_status(AB_STATUS_RUNNING);

    /* Run both sections over the enabled phases (phase index p => p+1
     * participating harts):
     *   section 0 "constant": each hart does `iters`  (total grows N*iters)
     *   section 1 "split":    total `iters` split evenly (each does iters/N)
     * The split section answers strong scaling: with a FIXED amount of work,
     * does spreading it across more harts finish faster, or does atomic
     * contention on the one address eat the parallelism? */
    for (unsigned section = 0; section < 2u; section++) {
        uintptr_t rbase = (section == 0) ? AB_RESULT_BASE : AB_RESULT2_BASE;
        uintptr_t fbase = (section == 0) ? AB_FINAL_BASE : AB_FINAL2_BASE;
        uintptr_t sbase = (section == 0) ? AB_SPAN_BASE : AB_SPAN2_BASE;
        for (unsigned p = 0; p < NUM_HARTS; p++) {
            if (((phase_mask >> p) & 1u) == 0u) {
                continue;
            }
            unsigned num = p + 1u;
            uint64_t iters_arr[NUM_HARTS];
            uint64_t expected = fill_iters(section, num, iters, iters_arr);
            run_phase(p, num, op, caddr, iters_arr, expected, rbase, fbase, sbase);
        }
    }

    printf("atomic_bench phases done\n");
    for (unsigned section = 0; section < 2u; section++) {
        uintptr_t fbase = (section == 0) ? AB_FINAL_BASE : AB_FINAL2_BASE;
        for (unsigned p = 0; p < NUM_HARTS; p++) {
            if (((phase_mask >> p) & 1u) == 0u) {
                continue;
            }
            uint64_t got = *AB_FGOT(fbase, p);
            uint64_t exp = *AB_FEXP(fbase, p);
            printf(
                "  section %u phase %u harts: got=%lu expected=%lu %s\n",
                section,
                p + 1u,
                (unsigned long)got,
                (unsigned long)exp,
                got == exp ? "OK" : "MISMATCH");
        }
    }

    /* Shut the helpers down and hand the L2CPU back to the idle FW. */
    for (unsigned h = 1; h < NUM_HARTS; h++) {
        *AB_WORK_ADDR(h) = AB_WORK_SHUTDOWN;
    }
    __asm__ volatile("fence ow, ow");

    set_status(AB_STATUS_DONE);
    x280_hart0_to_idle();
    return 0; /* unreachable */
}
