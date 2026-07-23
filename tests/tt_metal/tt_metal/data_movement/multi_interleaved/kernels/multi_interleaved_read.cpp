// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tensix_types.h"
#include "api/tensor/tensor_accessor.h"
#include "barrier_sync.hpp"

// Read the low 32 bits of the free-running RISC wall clock (same timebase the profiler stamps with).
// Reading WALL_CLOCK_L latches the high half for readback; we only need the low half here.
FORCE_INLINE uint32_t read_wall_clock_l() {
    return *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
}

// Busy-spin on the wall clock for ~cycles cycles to artificially deflate the issue/injection rate.
// 32-bit low half is sufficient: injected delays are far smaller than 2^32 cycles and the unsigned
// subtraction below absorbs a single wraparound of the counter.
FORCE_INLINE void spin_cycles(uint32_t cycles) {
    if (cycles == 0) {
        return;
    }
    uint32_t start = read_wall_clock_l();
    while ((uint32_t)(read_wall_clock_l() - start) < cycles) {
    }
}

// DRAM to L1 read
void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t l1_write_addr = get_arg_val<uint32_t>(1);
    uint32_t page_offset = get_arg_val<uint32_t>(2);
    // Barrier synchronization args
    uint32_t barrier_sem_id = get_arg_val<uint32_t>(3);
    uint32_t barrier_coord_x = get_arg_val<uint32_t>(4);
    uint32_t barrier_coord_y = get_arg_val<uint32_t>(5);
    uint32_t num_cores = get_arg_val<uint32_t>(6);
    uint32_t local_barrier_addr = get_arg_val<uint32_t>(7);  // Local scratch space for polling
    // Per-core PRNG seed for inject_delay_mode==2 (random injection delay). Always sent by the host so
    // the runtime-arg layout is fixed; ignored unless the random inject mode is compiled in.
    [[maybe_unused]] uint32_t inject_seed = get_arg_val<uint32_t>(8);
    // Per-core staggered-issue argument; its meaning depends on stagger_mode (compile-time arg 14):
    //   mode 1 (event handoff): PREFIX count -- number of cores in preceding groups that must finish
    //                           their issue phase before this core's group (column or row) may start.
    //   mode 2 (fixed delay):   the number of wall-clock cycles this core spins after the global barrier
    //                           before issuing (group_rank * delay_cycles, precomputed on the host),
    //                           producing an open-loop time-based staircase release with no handoff.
    // 0 for the first group in either mode. Always sent by the host; ignored when stagger_mode==0.
    [[maybe_unused]] uint32_t stagger_arg = get_arg_val<uint32_t>(9);

    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(3);
    constexpr uint32_t test_id = get_compile_time_arg_val(4);
    constexpr bool sync = get_compile_time_arg_val(5) == 1;
    // default_noc selects which RISC/NOC the reader runs on so the profiler zone name
    // matches the physical RISC: false -> RISCV0/NOC0 (default reader), true -> RISCV1/NOC1.
    constexpr bool default_noc = get_compile_time_arg_val(6) == 1;
    // enable_phase_counters: compile in the t0..t3 read-phase instrumentation (read-only path).
    // Compiled out when 0 so timing is clean by default (host sets it from TT_DM_PHASE_COUNTERS).
    constexpr bool enable_phase_counters = get_compile_time_arg_val(7) == 1;
    // rand_mode: per-core DRAM-bank access-order randomization (host sets it from TT_DM_RAND_OFFSET).
    //   0 = off: sequential page order, byte-for-byte identical to the default issue loop.
    //   1 = static rotation: each core starts on a random bank but keeps sequential order, same every
    //       transaction iteration (perm[k] = (k + start) % num_pages, precomputed on host).
    //   2 = advancing random permutation: each core visits all its pages/banks in a random order
    //       (a host-generated permutation), and the starting point advances by one every transaction
    //       iteration so the batch-to-batch order also decorrelates (e.g. D3 D7 D1 D4 ...).
    // In all modes the page<->L1-slot binding is preserved (page_offset+q read into slot q), so every
    // page is read exactly once per iteration into its own slot and the equality check still passes.
    constexpr uint32_t rand_mode = get_compile_time_arg_val(8);
    // num_warmup_iters: number of uninstrumented warmup passes over the batched read loop before the
    // single measured pass (host sets it from TT_DM_WARMUP_ITERS). 0 = no warmup (original behavior).
    // Warmup lets the measured pass observe steady-state NoC/DRAM behavior (AICLK ramp, NoC
    // arbitration, DRAM row buffers) instead of cold-start transients. Only applied on the batched
    // (!sync) read path; the sync path drives the CB per page with no free-running consumer, so
    // repeating it would deadlock on cb_reserve_back.
    constexpr uint32_t num_warmup_iters = get_compile_time_arg_val(9);
    // enable_page_counters: emit one timestamp marker right after every noc_async_read (host sets it
    // from TT_DM_PAGE_COUNTERS). The gap between consecutive markers is the per-page issue cadence;
    // the payload is the cumulative read-response count at that point, so a stall that coincides with
    // rising responses is noc_cmd_buf_ready backpressure (issue waiting for an outstanding read to
    // free a command-buffer slot). Independent of enable_phase_counters. WARNING: emits one marker
    // per page, so Q above ~110 overflows the per-RISC profiler L1 buffer and the tail is dropped.
    constexpr bool enable_page_counters = get_compile_time_arg_val(10) == 1;
    // inject_delay_mode: artificially deflate the per-core injection rate by spinning on the RISC wall
    // clock for a number of cycles after every noc_async_read (host sets it from TT_DM_INJECT_DELAY*).
    //   0 = off: compiled out entirely, byte-for-byte the baseline max-injection path.
    //   1 = fixed: spin exactly inject_delay_min cycles after every read (uniform lower injection rate).
    //   2 = random: spin a per-transaction random count in [inject_delay_min, inject_delay_max], drawn
    //       from a per-core xorshift PRNG seeded by inject_seed (a host runtime arg) so cores decorrelate
    //       (models bursty/jittered offered load). Note the PRNG itself costs a few cycles per read, so
    //       random mode with inject_delay_min==0 is not a true zero-delay baseline (use mode 0 for that).
    constexpr uint32_t inject_delay_mode = get_compile_time_arg_val(11);
    constexpr uint32_t inject_delay_min = get_compile_time_arg_val(12);
    constexpr uint32_t inject_delay_max = get_compile_time_arg_val(13);
    // stagger_mode: release core groups (whole columns or whole rows) in a staggered order instead of
    // all at once. The host picks the axis (column/row) and direction and folds it into the per-core
    // stagger_arg, so the kernel is axis-agnostic.
    //   0 = off: all cores released together by the global barrier.
    //   1 = event handoff: after the global barrier each core waits on a coordinator progress semaphore
    //       until all cores in preceding groups have finished issuing (stagger_arg = prefix count), runs
    //       its issue phase, then increments the semaphore to release the next group. Handoff point is
    //       issue-complete, but the release travels the shared read-response path, so the next group can
    //       be drain-gated (its poll reads queue behind the previous group's read responses).
    //   2 = fixed delay: after the global barrier each core spins stagger_arg wall-clock cycles before
    //       issuing (stagger_arg = group_rank * delay_cycles). Purely time-based staircase with no
    //       handoff traffic, so groups are never drain-gated -- but the spacing is open-loop (a fixed
    //       constant, not adaptive to how long a group actually takes to issue).
    constexpr uint32_t stagger_mode = get_compile_time_arg_val(14);
    // Progress-semaphore ID for staggered issuing (distinct from the global barrier semaphore).
    constexpr uint32_t stagger_sem_id = get_compile_time_arg_val(15);
    // read_vc_override: when set, force the reader's read command buffer onto a per-core static request
    // VC (the actual VC value is a runtime arg, index 10, so the host can pick it per core -- either a
    // single constant for the VC-validation sweep or a contiguous row-band split). Off by default so the
    // baseline uses the init VC (=1). On DM_DEDICATED_NOC the fast read path never rewrites NOC_CTRL, so
    // a single reprogram below persists for every subsequent read.
    constexpr uint32_t read_vc_override = get_compile_time_arg_val(16);
    // l1_providers_n: L1-provider mode. 0 = off (read from the interleaved DRAM buffer via TensorAccessor,
    // the default). N>0 = read from the L1 of N provider cores instead of DRAM, round-robin per page
    // (provider = page % N), removing the DRAM controllers from the path. Provider physical NoC coords and
    // the shared resident base address arrive as runtime args (11 = base, 12..12+N-1 = packed x<<16|y).
    // All readers read the same resident pages (shared data does not change NoC return-path contention and
    // keeps the provider L1 footprint tiny). Mutually exclusive with rand_mode (host enforces rand off).
    constexpr uint32_t l1_providers_n = get_compile_time_arg_val(17);

    // Per-core PRNG state for random injection delay (mode 2). Advances every transaction, persisting
    // across warmup and the measured pass so the jitter stream is continuous. Seed forced nonzero on the
    // host so xorshift never gets stuck at 0.
    [[maybe_unused]] uint32_t inject_rng = inject_seed;

    // Per-core page-visit order. Only populated (and only sent by the host) when rand_mode != 0:
    // mode 1 holds a rotated identity, mode 2 holds a random permutation of [0, num_pages).
    [[maybe_unused]] uint32_t page_perm[num_pages];
    if constexpr (rand_mode != 0) {
        for (uint32_t p = 0; p < num_pages; p++) {
            // runtime arg 10 is the per-core read VC; the page permutation follows starting at index 11.
            page_perm[p] = get_arg_val<uint32_t>(11 + p);
        }
    }

    // L1-provider mode state: shared resident base address and the provider cores' packed physical NoC
    // coords (x<<16 | y). Same values on every reader. Only read/used when l1_providers_n > 0.
    [[maybe_unused]] uint32_t provider_base = 0;
    [[maybe_unused]] uint32_t provider_xy[l1_providers_n > 0 ? l1_providers_n : 1];
    if constexpr (l1_providers_n > 0) {
        provider_base = get_arg_val<uint32_t>(11);
        for (uint32_t j = 0; j < l1_providers_n; j++) {
            provider_xy[j] = get_arg_val<uint32_t>(12 + j);
        }
    }

    // Tensor accessor compile time args appended to kernel's compile time args
    // so the index is offset to start at 18. Unused in L1-provider mode.
    [[maybe_unused]] auto args = TensorAccessorArgs<18>();
    [[maybe_unused]] auto s = TensorAccessor(args, src_addr);

    constexpr uint32_t transaction_size_bytes = page_size_bytes;
    // These user timestamped-data markers carry bandwidth metadata for the DM profiler CSV, but they
    // are emitted as TS_DATA markers. When NoC-event tracing is on the profiler treats every TS_DATA
    // marker as serialized NoC-event metadata, so these arbitrary payloads trip the "Invalid NoC
    // transfer type" TT_FATAL. Compile them out while tracing (PROFILE_NOC_EVENTS is auto-defined
    // whenever TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1); the bandwidth CSV is not needed for tt-npe runs.
#if !defined(PROFILE_NOC_EVENTS)
    DeviceTimestampedData("Number of transactions", num_of_transactions * num_pages);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);
#endif

    // Wait for all cores to reach this point before starting data movement
    barrier_sync(barrier_sem_id, barrier_coord_x, barrier_coord_y, num_cores, local_barrier_addr);

    // Optionally force every DRAM read onto a specific static request VC. Done after the barrier so the
    // barrier's own polling reads stay on the default VC (=1); only the measured/warmup DRAM reads move.
    // Reprogramming NOC_CTRL once is sufficient on DM_DEDICATED_NOC (the fast read path leaves NOC_CTRL
    // untouched), and it applies to both the warmup and measured passes below.
    if constexpr (read_vc_override) {
        uint32_t read_vc_val = get_arg_val<uint32_t>(10);  // per-core static read VC (constant or row-band)
        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(read_vc_val);
        NOC_CMD_BUF_WRITE_REG(noc_index, read_cmd_buf, NOC_CTRL, noc_rd_cmd_field);
    }

    // measure gates the t0..t3 phase markers and the first-response spin so warmup passes run the
    // exact same read work without emitting timing markers (only the final pass is instrumented).
    auto do_reads = [&]([[maybe_unused]] bool measure) {
        // Phase instrumentation only applies to the read-only (!sync) batched path, where a
        // single barrier drains the whole batch. In sync mode each page barriers individually,
        // so the t0..t3 split is not meaningful. NIU_MST_RD_RESP_RECEIVED is the HW cumulative
        // read-response counter; each marker carries "responses so far" for host phase splitting.
        // Staggered issuing: delay this core's issue phase so groups (columns/rows) start in sequence.
        // Only the measured pass staggers; warmup runs unstaggered. Placed before the t0 marker so the
        // issue phase excludes the wait. Mode 1 blocks on the coordinator handoff semaphore; mode 2 just
        // spins a fixed per-core cycle count (open-loop staircase, no handoff traffic).
        if constexpr (!sync && stagger_mode == 1) {
            if (measure) {
                stagger_wait(stagger_sem_id, barrier_coord_x, barrier_coord_y, stagger_arg, local_barrier_addr);
            }
        } else if constexpr (!sync && stagger_mode == 2) {
            if (measure) {
                spin_cycles(stagger_arg);
            }
        }
        [[maybe_unused]] uint32_t resp_baseline = 0;
        if constexpr (!sync && (enable_phase_counters || enable_page_counters)) {
            if (measure) {
                resp_baseline = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED);
            }
        }
        if constexpr (!sync && enable_phase_counters) {
            if (measure) {
                DeviceTimestampedData("dm_t0_issue_start", (uint64_t)noc_index);
            }
        }

        for (uint32_t i = 0; i < num_of_transactions; i++) {
            for (uint32_t k = 0; k < num_pages; k++) {
                // q is the page (and matching L1 slot) handled this step:
                //   mode 0 -> k                          (sequential; folds away to the default loop)
                //   mode 1 -> page_perm[k]               (static per-core order, same every iteration)
                //   mode 2 -> page_perm[(k + i) % num_pages]  (random order, start advances each iter)
                uint32_t q;
                if constexpr (rand_mode == 1) {
                    q = page_perm[k];
                } else if constexpr (rand_mode == 2) {
                    q = page_perm[(k + i) % num_pages];
                } else {
                    q = k;
                }
                if constexpr (sync) {
                    cb_reserve_back(cb_id_in0, 1);
                }
                uint64_t noc_addr;
                if constexpr (l1_providers_n > 0) {
                    // Round-robin across the N provider cores' L1: page q -> provider (q % N), and the
                    // (q / N)-th page that provider holds, packed at provider_base.
                    uint32_t prov = q % l1_providers_n;
                    uint32_t within = (q / l1_providers_n) * page_size_bytes;
                    uint32_t pxy = provider_xy[prov];
                    noc_addr = get_noc_addr((pxy >> 16) & 0xFFFF, pxy & 0xFFFF, provider_base + within, noc_index);
                } else {
                    noc_addr = s.get_noc_addr(page_offset + q);
                }
                noc_async_read(noc_addr, l1_write_addr + q * page_size_bytes, page_size_bytes);
                if constexpr (!sync && enable_page_counters) {
                    if (measure) {
                        // NOTE: keep on a single source line (GCC __LINE__ marker-name hash quirk).
                        DeviceTimestampedData(
                            "dm_page_issued",
                            (uint64_t)(NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED) - resp_baseline));
                    }
                }
                // Artificially throttle the injection rate: spin on the wall clock after issuing the read
                // so the next read is delayed. Applied on warmup and measured passes alike so steady state
                // matches. Compiled out entirely when inject_delay_mode==0.
                if constexpr (inject_delay_mode == 1) {
                    spin_cycles(inject_delay_min);
                } else if constexpr (inject_delay_mode == 2) {
                    inject_rng ^= inject_rng << 13;
                    inject_rng ^= inject_rng >> 17;
                    inject_rng ^= inject_rng << 5;
                    spin_cycles(inject_delay_min + (inject_rng % (inject_delay_max - inject_delay_min + 1)));
                }
                if constexpr (sync) {
                    noc_async_read_barrier();
                    cb_push_back(cb_id_in0, 1);
                }
            }
        }
        if constexpr (!sync) {
            if constexpr (enable_phase_counters) {
                if (measure) {
                    // NOTE: each DeviceTimestampedData(...) call MUST stay on a single source line
                    // (GCC __LINE__ expansion quirk in the marker-name hash).
                    DeviceTimestampedData(
                        "dm_t1_issue_end",
                        (uint64_t)(NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED) - resp_baseline));
                }
            }
            // Event-handoff only (mode 1): release the next group at issue-complete (before the
            // first-response spin below, so the handoff timing is independent of whether phase counters
            // are compiled in). The next group begins issuing while this group is still draining. Mode 2
            // is time-based and needs no signal.
            if constexpr (stagger_mode == 1) {
                if (measure) {
                    stagger_signal(stagger_sem_id, barrier_coord_x, barrier_coord_y);
                }
            }
            if constexpr (enable_phase_counters) {
                if (measure) {
                    while (NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED) == resp_baseline) {
                    }
                    DeviceTimestampedData(
                        "dm_t2_first_return",
                        (uint64_t)(NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED) - resp_baseline));
                }
            }
            noc_async_read_barrier();
            if constexpr (enable_phase_counters) {
                if (measure) {
                    DeviceTimestampedData(
                        "dm_t3_barrier_clear",
                        (uint64_t)(NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED) - resp_baseline));
                }
            }
        }
    };

    // Uninstrumented warmup passes (batched path only), then re-align all cores so the measured
    // pass runs under full concurrent load rather than trailing behind stragglers' warmup.
    if constexpr (!sync && num_warmup_iters > 0) {
        for (uint32_t w = 0; w < num_warmup_iters; w++) {
            do_reads(false);
        }
        // Coordinator semaphore is cumulative and was already incremented once by the initial
        // barrier above, so this second barrier waits for 2*num_cores arrivals.
        barrier_sync(barrier_sem_id, barrier_coord_x, barrier_coord_y, num_cores, local_barrier_addr, 2u * num_cores);
    }

    if constexpr (default_noc) {
        DeviceZoneScopedN("RISCV1");
        do_reads(true);
    } else {
        DeviceZoneScopedN("RISCV0");
        do_reads(true);
    }
}
