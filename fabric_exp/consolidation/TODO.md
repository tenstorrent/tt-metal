# Router study TODO (started 2026-11-02)

The main session keeps this file current. Each item carries its source doc. Order within a section is the intended
order. North star: readability first (for placement decisions: readability, then stability, then perf). Every code
change gets identity checks, the register-map guard and its headroom check (loopvars.py), and an A/B wherever codegen
changes.

## In progress

- **Galaxy pairs: done (2026-11-05).** glxq_pairs.tsv (36 ring pairs, 32 single-cable pairs, from the FSD);
  `glxq_run.sh --pair any --form F` submits a first-start-wins group (it counts as one allocation). Trial group
  1105034909 / job 176705 (a10u08 + a10u14), the 16x4 form on 330a2628861: lock + cancel worked, ranks mapped rows 0-7 /
  8-15, the data-checked all-to-all smoke passed on both ranks. Open: a smoke row for the two-tori forms (they need
  --no-smoke today); per-step rank log names are new and not yet run on a job.
- **Allocator + TRID landing (user, 2026-11-05):** allocator tweaks (a) worker depth 18 and (b) receivers to the next
  power of 2 first in the leftover order, (c) the 1D 3072 B +1.7% accepted; the TRID per-channel rule approved to land
  (lb2 176756). Sequence: amend the allocator, short P150 rerun, rebase TRID, cherry-pick both onto 330a2628861, P150 val
  set + Galaxy G-PORT P10, then push. The N/S torus decode spill is a separate diagnostic (compile-time coordinates),
  design to the user first.
  **Pushed at 662331732ce on 2026-11-05 ~09:00** (origin/aagarwal/fabric-router-mainline 57c55f98118 -> 662331732ce:
  allocator, TRID, telemetry switch dropped (ce2dee3cd6e), bring-up tools dropped). P10-mm-2x8 stays open as its own
  investigation, not a push gate (entry below). **The clock-read fix e7fe1899049** (only the sender reads the wall clock
  at round start; ERISC1's read could tear ERISC0's latched 64-bit read: the long-run +2^32 outlier) **passed its offline
  image gate and was pushed: origin at e7fe1899049 (2026-11-05).**
  - Gate: 7 configs × 3 telemetry modes.
  - Telemetry off: byte-identical everywhere.
  - Perf recorder: ERISC0 code-identical apart from a symbol rename; ERISC1 −16 B (the read dropped).
  - Dynamic stats: −16 B, round_start_impl only.
  - The only other differences are address-only `_start` diffs.
- **BH bandwidth golden refresh (user, 2026-11-06):** tests/tt_metal/tt_fabric/test_infra/golden/
  golden_bandwidth_summary_blackhole_p150_x4.csv (268 rows, num_links 1-4) must be re-recorded with the new router, as
  WH's was (18e9f1b9ce9: same rows, order and tolerances; telemetry on as CI runs it; median of three runs). Open: it is
  a P150 x4 (QuietBox) golden, so record on that hardware (CI runner or a P150x4 box), not the LoudBox. Check whether
  golden_latency_summary_blackhole_p150_x4.csv needs the same.
- **Push when tests pass (user OK, 2026-11-05):** cleanup branch rebased onto e8decd2e27e (user's WH Galaxy fix), tip
  539cb2a8434 (7326d59a864 flush, bd6e71c4ce5 CT worker semaphores, 83f62902911 router_single_erisc, 32a6e950990 inline
  write forms, 539cb2a8434 credit_each_slot); pre-rebase chain kept at aagarwal/fabric-router-mainline-cleanup-prerebase.
  Gates: G1 G2 G4 G5 identical/as expected; G3 G6 G7 pass; G8 (de8a2ac17a8 single-ERISC inline on BH vs 539cb2a8434)
  pass as expected: two-ERISC, WH, non-router identical, 700/700 BH one-ERISC changed and smaller
  (fabric_exp/port_tune/gates_20261105/). New push gate (coordinator, 2026-11-05): flush lb2 180530 + G8.
  **Flush validation FAILED to run (open, coordinator to decide):** the unpatched arm (PR base) hung on the first
  data-checked test, FlushA2ARing atomic_inc unicast, on LB (lb2 180530, 1024 B) and Galaxy (180570, 4096 B); patched
  arm never ran. Captures: results/fdaa_unp_val_ring_hangcap, results/glx180570. Cause open (hypotheses: test-infra
  limit of flow-controlled all-to-all on Ring, which stock yamls never run; base-router ring/atomic bug); isolating A/B
  = same yaml on main's router. Not run: awaiting go.
- **DONE 2026-11-05: replies posted on PR #63939** (user OK; discussion_r4486914997, r4486915272, r4486915594; user closes
  #63939, review continues on #64160). Was: **Reply on PR #63939 (user, 2026-11-05), after the fixes land; not before.** SeanNijjar's 3 inline comments on the
  preview branch: (a) rename return_slots -> credit_slots; (b) explain the connection_semaphores array size (named
  constant); (c) start_running and the context switch (end_round still switches every SWITCH_INTERVAL rounds). Fixes
  are with the port agent on the cleanup branch. The replies need the user's OK before posting.
- **Test-side changes stay patches, not in the router PR (user, 2026-11-05):** fabric_exp/streams/port/
  test_infra_6dad.patch, and adapter_vc2_credits_6dad.patch (touches tt_metal/; patch only, never pushed with the PR).
- **2026-11-05: cleanup (e5b75d1fbcd) and the decode (6dad5fec638) pushed.** The a09u02 hang was the node (176236). In
  progress, offline: (1) the TRID rule ported onto 6dad5fec638 + the allocator (agent a6e85079ff596efce, worktree
  tt-metal-port-trid; hypothesis C check on the 4x2 / Galaxy torus N/S receivers; device A/B after the allocator
  A/B, on the user's go); (2) the write-path static diff, port vs study, for the remaining write gap (agent a4dc906ee7c1faf44,
  into RECONCILE_DEVICE.md): done (§8), and the quick test (§8.1, job 798) found the gap is the test sender kernel.
  Idle-pass recheck on the final allocator 9ce0acbeaba: gate, slot probe (2D 17,16,16,16 R16; 1D 26,25 R32; VC1
  forwarding 7) and mock vet OK; staged; lb2 177108 done (RECEIVER_TUNING.md §6): i8 vs i4 is 2D −0.20, VC1 −0.08,
  vc1i ±0.06; i2 is +0.27 / +0.80 / ±0.03. **Closed (user, 2026-11-05): IDLE_PASSES_BEFORE_PARTIAL_CREDIT stays 4,
  no census for 8** (i8 within 0.2 cycles of i4, i2 slower on 2D and VC1, VC1 inter-mesh flat on the allocator's
  rows). The pushed comment (57c55f98118) already has the measured reason: no code change. Receiver tuning closed.
  **2026-11-06, user: no time-based partial-credit rule ("even more fragile"); the NX fix stays open, value not
  chosen.** Sweep 4/8/16/32/64 done (RECEIVER_TUNING.md §8): Galaxy 200G NX gap closes at 16 (4352 B keeps +0.55); LB
  burst N8 G500 16 B +3.3% at 16, +1.2% at 8.
  **Reopened by new evidence, 2026-11-05:** the 200G Galaxy NX gap (port +1.15 /
  +3.56 / +3.57 cycles/packet vs main at 3072 / 4096 / 4352 B, Slurm 179756) is this constant. A/B Slurm 180134
  (port only, one constant per arm; fabric_exp/results/glx180134/nxab_report.txt): IDLE_PASSES 4 -> 64 gives
  −1.14 / −3.56 / −3.03 (main parity at 3072 / 4096, +0.54 left at 4352); receiver depth 16 or 64 and an E0 delay of
  10-60 move nothing; an E1 spin of 60 per packet speeds the rows up (−1.15 / −2.50 / −2.38). Mechanism (extra
  credit frames on a wire both directions share) is a hypothesis: frames not counted. The 400G decision (i8 within
  0.2 of i4) did not test a large value or 200G wire-bound bidirectional rows. To settle: a sweep (e.g. 4/8/16/32/64)
  on 200G NX plus the 400G census suites and latency, or a time-based partial-credit rule; the user decides.
  CI streaming-profiler repro (CI run 41963172504; port 57c55f98118, worktree /home/aagarwal/tt-metal-ci-sp, Tracy
  build; port_tune/cisp/):
  - Job 801 (telemetry off) produced no ratios: the test's bandwidth is the router telemetry readout. Its profiler-on
    run also stalled on the Ring config.
  - Job 810 (telemetry on, the user's one-off CI-fidelity exception) reproduces it: NeighborExchange 4096 B write
    num_links_2 on/off is 0.9873 / 0.9861 (CI 0.9873; bound 0.989), with off-to-off +0.0037%. The median is 0.9986 /
    0.9982. The Ring config completed with the profiler on.
  - Not isolated yet: the router hook, the zone code / placement, or the profiler's runtime activity. Next arms are
    the coordinator's / user's call.
  - Split job (results/cisp_ci_20261105-064341; arms off / on / no-hook, router hook compiled out plus the host patch
    cisp/host_link_solve_warn.patch):
    - The no-hook arm aborted at its first fabric-config switch on a third host check (device_programs.cpp:792,
      "link is dead": fewer than 2 rounds in 1 s). The device was held; the broker reset and recovered it (814-816).
    - Partial, one pass, NX config only. NX 4096 B write 2 links: on 0.9862, no-hook 0.9910. The 1-link row is
      0.9972 / 0.9969. The other NX rows are 0.9992-0.9995 on, 1.0000 no-hook.
    - Images: no-hook E0 = on minus the RouterPort / PortBase / eth_ptp functions (end_round and
      SenderChannelInterfaces::start changed).
    - Waiting on the coordinator: extend the host patch, or accept the partial result.
  - Follow-ups, 2026-11-05 (report results/cisp_followups_20261105.md; TELEM=ci; no device runs for (c)):
    - (a) PC split (job 817, results/cisp_pcs_20261105-074220; port, NX 4096 B write, 80M packets, L1 / L2):
      - ERISC0 on vs off: link_sync 0.49-0.51% and end_round +0.10-0.17 points; telemetry 12.2% in both. ERISC1 is
        unchanged.
      - Steady state: on L1 all 20 routers are within 0.1%. On L2, 39 of 40 routers are at 0.999. One router (chip0/eth7,
        transmitter) took 4.29G more cycles (about 3.2 s); one occurrence, cause unknown.
    - (b) Main off/on (job 818, results/cisp_ci_main_20261105-074351; main d2bd12fcf1f + the same test change):
      - **On is faster on every row.** Median 1.0051 / 1.0049, worst 1.0011; NX write and fused-atomic rows +3.7 to
        +4.6%. The port (job 810) is at 0.9986 / 0.9982, worst 0.9873.
      - Per router: main's off routers sit at several cycle levels, and on collapses them all to the fastest. Cause is a
        hypothesis.
    - (c) Per-router map (port_tune/cisp/cisp_router_map.py):
      - **The port's on/off levels are averages of discrete events.** About 540-570k cycles (about 0.4 ms) on one router
        in 1 or 2 of a test's 3 iterations, which shows as +1.6% / +3.1% in the mean.
      - **The events repeat at the same test-iterations in every on run.** NX 4096 B write L2 iterations 1-2 and L1
        iteration 3, FullRing 4096 B fused-atomic L2. They hit 1-26 routers each, a different set per run, transmitters
        and receivers alike.
      - **They also occur with the router hook compiled out** (job 811's no-hook arm). So the link-sync hook is not
        needed; the earlier "hook per-round work / role split" candidates are marked WRONG in the report.
      - **Main has no events.**
      - Hypothesis, not tested: the profiler's activity outside the hook, e.g. the router's kernel_profiler ring waiting
        for a host drain. The isolating A/B is profiler on with the router's zones off, or with a larger ring.
    - (d) Stall sampling (job 819, results/cisp_pcs2_20261105-083111, report section (d)):
      - The events reproduce (288 router-iterations).
      - During them the routers' PC mix, busy share and idle lines match the quiet routers'.
      - Real-time first and last send are within ±0.15 ms of the quiet peers, while the telemetry counts +0.40 ms.
      - **Hypothesis: counted time in the perf recorder**, either a wall-clock step or rounds waiting at iteration
        start, **not a sending stall.** The kernel_profiler ring wait from (c) is not supported: 1 ring_ensure_room_slow
        sample in 4167.
      - Next is a recorder variant on the device: per iteration, the longest counted round and its position, or the wall
        clock beside the cycle counter. It is a design proposal, for the user's go.
    - (e) Recorder instrumentation (job 820, report section (e)):
      - Every entry aligns (busy == cycles in all 8640 rows).
      - Events have no long round (about 2.9k cycles max, as in quiet ones), no clock step, no torn read and no firmware
        run.
      - Instead they have about 8% more counted rounds with fewer packets each (21.2 vs 22.9). The extra rounds' fixed
        cost (~1.3k cycles each) is the ~0.46M excess.
      - Hypothesis: a batching or supply pattern counted by the round-based recorder, not a stall.
      - The +4.29G long-run outlier is exactly 2^32, a torn wall-clock read; the fix e7fe1899049 is local, in its gate.
    - (f) The test's sender-side bandwidth (jobs 821 port, 822 main; report section (f)):
      - **Port:** worst on/off is 0.9537 / 0.9694 (NX 4096 B write L2), against telemetry 0.9865 / 0.9884; median
        0.9987.
      - **The port's sender rows show the same discrete events:** −4.6% iterations at the same positions. So event
        iterations really are slower. The (d)/(e) "counted time, not a stall" readings are marked WRONG in part.
      - **Main:** sender-side median 1.0013, worst 0.9977; NX write rows +3.5% to +5.2% with the profiler on.
      - Hypothesis: a worker-side profiler cost that shows only once the router outruns the worker. WRONG in part (830):
        the same ~4.7% slow state occurs with the profiler off in another row order (C); not a profiler cost itself.
    - (g) Stall hunt (jobs 823 zone CSV, 824 Tensix PC samples; report section (g)):
      - Slow iterations run every sender kernel ~0.43-0.46 ms longer, on most or all chips in the same iteration (41/288,
        unicast_write rows only). It's not a late start.
      - In those iterations the senders shift from fabric-credit waits (19% -> 13% of kernel samples) to NoC
        command-buffer waits (17% -> 25%) and write-ack waits (0.4% -> 2.4%). No profiler code is sampled.
      - Hypothesis: NoC contention from a periodic profiler transfer (relays or DRAM).
      - Next: relay-core sampling on the same timeline, then the A/B (design to the coordinator first).
      - Steps 3-4 (jobs 825, 826; report (g)):
        - The slow clusters are periodic, ~0.7 s or ~18 iterations, 1-5 iterations long. WRONG as a reading (829
          review): ~18 iterations is the yaml's spacing of the unicast_write blocks, the only rows that ever go slow.
        - The eth relay is flat; DRISC activity isn't visible through drisc_pc.
        - The ship-threshold A/B (kShipMinPct 25 -> 50, kIdlePumpStride x2) leaves the period unchanged, so neither is the
          trigger. WRONG (829 review): the period is the row layout and can't move; the slow counts barely move (weak
          evidence only).
        - Candidate: the relay's kMaxDeferSweeps (2048 sweeps ≈ 0.7 s if a sweep takes ~0.34 ms). A/B arms C/D are
          designed, not run.
      - Defer A/B (job 827):
        - kShipMinPct 50 with kMaxDeferSweeps 4096 or 1024 leaves the period unchanged (~0.68-0.69 s, ~18 iterations).
        - The relay's ship cadence is ruled out as the clock. WRONG (829 review): not established; the period is the row
          layout. Next: host-side cadences, after the coordinator's review.
      - Time vs count (job 828):
        - The period stays ~18 iterations at num_packets ×1/×2/×4 (0.7 / 0.9 / 1.2 s), so it's count-based. WRONG as
          worded (829 review): ~18 is the row layout; what stands is that the slow positions in the blocks don't move
          with iteration length, so the slowdown follows the row sequence, not time.
        - The excess scales with iteration length (~4.8% of the iteration), a rate change.
        - No profiler component has a ~0.7 s period; it doesn't matter, since the clock counts iterations. (Premise WRONG,
          829 review: there's no ~18-iteration clock in evidence.)
        - The sender's load address doesn't separate slow from normal.
        - Count-based candidates (producer word count, per-program host state) await review. (Superseded by 829: the
          ~18 count is the row layout.)
      - Dispatch observation (job 829, results/cisp_dispatch_20261105-100012; report (g)):
        - Prefetch, dispatch and dispatch_s sit in the same local wait loops in slow and normal iterations (within ±6
          points); nothing touches the sender tiles. Dispatch is not active during the slow iterations.
        - The off arm runs the realtime profiler (dispatch_s records, reserved core f16-11); the on arm doesn't.
        - Kernel-config ring 69 KB from 0xa3a0: 5 sender programs per wrap on, 6 off; launch ring 8. No ~18-program
          cycle. Neither kcb nor launch slot separates slow from normal.
        - Only unicast_write rows ever go slow (0/96 fused_atomic_inc, 0/96 scatter in 7 runs). Slow iterations sit at
          positions 1-5 of a 6-iteration block, mostly 2-4, almost never the first. Hypothesis: a state that builds up
          over consecutive unicast_write iterations. WRONG (830): no build-up with write rows only.
        - Proposed, not run: yaml-only reorder A/B (write rows only; write block moved). Words-vs-programs build on
          hold (its ~18-count premise is gone).
      - Row-order checks (job 830, results/cisp_orderab_20261105-101713; report (g)):
        - S stock control 33/96 slow on, 0 off. B (write block last): the slowdown moves with the block, with the same
          in-block positions.
        - A (write rows only): no build-up; slow at positions 2-4 of each 6-iteration size group (around the
          num_links 1 -> 2 switch), never the group's first two.
        - C (one write per row, tiny fused rows between): 28/96 slow on, and 14/96 slow OFF at the same ~0.953 level
          (first L1 writes of a size group). The slow state is not a profiler cost; the profiler changes which
          iterations land in it. The stock order keeps the off arm out of it.
        - Open: which flow or core is slow (per-direction rows or per-sender zones), and what setup differs between
          C's slow and normal off-arm writes.
      - AICLK check (offline, report (g)): slow iterations take longer in host time by about the cycle excess (+0.42 ms
        vs +0.44 predicted at 100k packets; +1.5-1.7 ms vs +1.7 at 400k), so AICLK is unchanged within ~1% and the
        slowdown is real. The zone CSV has no ns; the sync check (AICLK stats) was off and needs a router rebuild.
        Next (step 1, design first if a build is needed): router state at iteration boundaries. Lead: the logs'
        per-router telemetry shows two discrete router rates (~48.45 / ~46.3 GB/s) within one iteration.
      - Per-router tally (offline, 826-830, every arm; report (g)): routers are at baseline or +4.6-4.9% (~+5.5
        cycles/packet), nothing between. Slow iterations have 1-25 low routers, random subsets (every router low
        somewhere; no stable set across arms; N/S = E/W, T = R; P1 slightly more than P0). Not chip-wide (2084 of 2113
        chip groups partial). The CSV flag = a low router sourced on chip 7/3/6, with 0 exceptions, because senders end at
        a nearest-neighbor end sync. 823's "all senders on a chip run long" is that sync, not a chip-wide state. Next:
        design to the coordinator (not a stable subset).
      - X1 + route analysis (job 831, results/cisp_flowab_20261105-162525; report (g)): one flow on one link never enters
        the state (0/640, on and off); with two links of a pair only the link-0 router does, on the on arm for the S/N
        and E pairs from chips 1, 2, 5, 6 only (80/320 on, 10/320 off). Routes (debug-logging mock: sender (15,10) link
        0, (15,9) link 1, receivers (1,2..), NoC0 out / NoC1 in) share no NoC link in X1, and c1->c5 (low) and c0->c4
        (never) have identical geometry. Neither (A) as NoC link sharing nor (B) as a lone router; chip-specific with
        two active links. Open: eth-port / chip-specific cause; design to the coordinator.
        WRONG as a condition (user review): "two links of a pair" is too narrow. The NX 1-link rows slow too (2-3 links
        per chip, one per neighbor); only a lone flow per chip never slows. Condition: two or more concurrent flows on a
        chip, any arrangement, chip-dependent, per-packet cost (2048 B = 4096 B in cycles/packet). Next (after Galaxy):
        the per-packet resource concurrent flows on a chip share (outbound sender writes + acks into routers, inbound
        deliveries / acks / TRID flushes, eth subsystem); yaml discriminator: send on one link while receiving on
        another, vs send on two links with nothing inbound.
      - X3 (job 832, results/cisp_flow3ab_20261105-165255): Lone 0/48; two outbound flows from one chip (SO) 5-6/48 on,
        the E router only (sender core (15,10)); one out + one in (SR) 0/48; two inbound (RR) 0/48; NX control 42/96 on,
        0 off. Outbound concurrency is the trigger; inbound isn't. In SO and X1 the flow from core (15,10) slows while
        (15,9) sends. Next: SO with swapped cores / on chips 0 and 3 (design to the coordinator).
        Correction: benchmark mode splits senders by direction (split_senders_by_direction_for_benchmark): NX runs one
        sender core per (direction, link) (823 zones: chip 0 has 3 senders at num_links 1); my mock-based "one core per
        plane" was WRONG (mock ran benchmark_mode false). X4 (broker 834: SOswap / SOfar / chips 0, 3) queued; X5 (835:
        one-core arms) is moot: the split makes SO1 = SO2 and NX1 = NXs.
      - X4 / X5 (jobs 834, 835): every SO arm 0/48 (X3's 5-6/48 doesn't reproduce: swap / far can't discriminate); NX's 20
        flows written explicitly at 1 link 0/960 router-iterations, while the stock NX write rows in the same job are
        316/2880. The state appears where num_links 2 rows run (NX's L1 it2 / L2 it0-1 positions; X1's L2). X3's
        "outbound concurrency is the trigger" marked WRONG. Next (yaml only, coordinator): NX L1-only vs L2-only vs
        alternating blocks.
      - Start skew (823 zones, cisp_skew.py): neighbour starts line up within ~20 us (median deviation ~8 us,
        low-source chip vs neighbour +0.07 us); the slow chip's sender span is +~435 us: the rate is lower all through,
        not a late start.
      - X6 (job 838, results/cisp_flow6ab_20261105-171231): NX write rows L1-only 0/48, L2-only 0/48, stock L1/L2
        alternation 38/96 (on; all 0 off). Positions: L1 it1-2 after an L2 block, L2 it0-1; both planes, every direction
        and chip. The state needs the switching between link counts (link-1 routers idle after traffic, then active).
        Next (coordinator): read-only look at the idle link-1 routers in an L1 row after L2 (pc_sampler), or an idle gap
        between L2 and L1.
      - X7 (job 839, results/cisp_pcs3_20261105-173030; read-only router PC samples, profiler on): the idle link-1
        routers' ERISC0/1 PC shares are the same in (i) L1-after-L2 (slow and normal iterations) and (ii) L1-only
        (passes ~83.5%, telemetry ~10-11%, end_round ~3%, link_sync ~1%; holds_packet 0.7-1.2%, noise); the active
        routers' low and normal iterations match too. Idle gap after L2: L1 it0 slow in 9/16 (stock: 0/16): the state
        builds with time at num_links 1 after L2, not with traffic. Next (coordinator): eth status / error counters of
        the idle links over time, (i) vs (ii).
      - X8 (job 842, results/cisp_ethpoll_20261105-174703; consolidation/eth_poller.py, read-only, no perturbation seen):
        links clean (PCS up, ERR_STAT 0, no retrain / CRC / uncorrected FEC / resend / drop / recovery); idle link-1 TX
        background (txq0/1 1.347 M seq packets/s, txq2 12.8 k transfers/s, hypothesis: link-sync) and FEC corrected
        (~6.5 k/s) the same in slow / normal / L1-only. No link-level counter tracks the slow state.
      - Worker cores (offline, 823 zone CSV, cisp_cores_check.py): identical per row type in every iteration, with or
        without a preceding L2 block (allocator reset per test iteration). Not a placement effect.
      - **PARKED 2026-11-05 (coordinator).** Summary in report (g) "Parked": established (discrete per-flow state,
        trigger = L1/L2 alternation, builds with time at L1 after L2), ruled out (relay knobs, AICLK, placement, skew,
        concurrency alone, router code paths, eth link errors / background TX, worker cores), remains (profiler
        non-router parts, NoC/NIU state, MAC/PCS, RXQ; profiler-off only in 830 C). Next A/Bs listed there; not queued.
  Galaxy 200G PR refresh (Slurm 179467, bh-glx-110-a10u08, all 24 steps pass): report/data/bandwidth_glx_pr.json
  (build_glx_pr.py, 238 records); suite means main/port: mesh 1.33, linear 1.22, ring 1.30, torus 1.52.
  P150 PR refresh (lb2 179436 / 179445 / 179492): report/data/bandwidth_p150.json pair p150pr-*.
  PR gap runs (2026-11-05; pr_perf/mk_pr_gap_yamls.py): 400G lb2 179740 (gapa: scatter / fused 256-4352, atomic_inc
  E/W + N/S, MeshFwd) and 179741 (gapb: XY-torus wrap E/W 2x4 and N/S 4x2, ring 1-hop, 256-4352) -> bandwidth_p150.json
  (p150pr-{ptype,txw24,txw42,ring1}-*, 88 records); 200G Slurm 179863 (512 / 4096 B rows; 179743 lost pass b to an
  ENOSPC on /data, superseded) and 179756 (NX write 1 and 2 links, 256-4352) -> bandwidth_glx_pr.json (build_glx_pr.py
  179467 179863 179756). Observation, no cause: on 200G wire-bound sizes the port is 0.4% (mesh) to 1.4% (NX) below main.
  PR latency (main d2bd12fcf1f vs port e7fe1899049, same job, passes a/b): 400G lb2 179906 (a08u28, exp_latency.yaml),
  200G Slurm 179907 (a02u14, pr_glx_latency.yaml incl. Lat7HopNS) -> latency_samejob.json latpr-{p150,glx}-* (56 records).
  Completion-proof counts (job 800, RECEIVER_TUNING.md §7): 0 cap hits in 598k group proofs, none above 8 reads;
  only 4352 B write deliverers need 5-8 reads; atomic / fused / scatter always pass at read 1. §5.2's "cap binds on
  FWD rows" marked WRONG. Ring G 1 checks: about 20% deferred on the deliver-only receivers. Open: what c128 changes
  on FWD rows (not the proof); FWD_MixCmd c128 pass b +37 outlier.
  deliver_other inline A/B (job 799, RECONCILE_DEVICE.md §9): inlining slows every common 1-hop row by 0.4 to 5.6
  cycles/packet (atomic +3.7, scatter +5.5, fused +2.5); only the receiver images change; all fit. As built (noinline) is the faster arm.
  Done: the study's test-infra changes on 6dad5fec638 as an uncommitted patch, kept apart from the router series
  (fabric_exp/streams/port/test_infra_6dad.patch, README there; worktree /home/aagarwal/tt-metal-port-testinfra,
  built, mock-vetted; optional adapter_vc2_credits_6dad.patch for VC2 benchmark workers). For the user to handle
  separately.
- **/data inodes at 100% (2026-11-05):** the shared FS has ~0.7-1.1M of 400.6M inodes free. Ours is 4.82M (1.2%):
  - this study ~3.68M: frs-glx/jobs 1.63M (66 jobs; ~0.6M of it kernel caches), regress/trees 1.05M (6 built trees),
    frs-glx/trees 0.84M (38), lb2 0.15M;
  - the user's own work 1.14M (tt-metal clones, /data/aagarwal/slurm; memory: leave alone).
  - Inventory with ages and references: consolidation/data_inodes_20261105.tsv. Nothing deleted; awaiting the user's
    decision.
  - mainline_cb42eb7 is complete: restaged at 2026-11-04 08:00 after a 06:40 stage that hit the inode limit. It's the
    last-pass tree for 2x8 (172825).
- **P10-mm-2x8 router sync failure is not node-specific** (2026-11-05 08:44).
  - **The failure:** Fabric Router Sync timeout on Device 0, master eth chan 3 and sub chan 2 at STARTED, the others at
    REMOTE_HANDSHAKE_COMPLETE.
  - **Where it occurs:**
    - 57c55f98118 on a07u02 (177688);
    - c3c1813c42a on a07u02 (177459, 177689, and the 06:58 rerun);
    - c3c1813c42a on **a08u08** (177969).
  - **Last passes:** e5b75d1fbcd on a10u14 (00:08) and cb42eb72501 on a04u14 (172825).
  - The "node-linked a07u02" claim is marked WRONG in TEST_PLAN §4.5.
  - **Commits since e5b75d1fbcd:** 6dad5fec638 decode, cc63d9b0284 review fixes, 330a2628861 comments, 57c55f98118
    comment, then the allocator and TRID commits.
  - **Open, tree vs node vs order:** G-PORT-ORD A/B 178014 (e5b75d1fbcd on a08u08) and 178007 (57c55f98118 on
    a10u14). Not a push gate since the 662331732ce push.
  - **G-PORT-ORD results (2026-11-05 ~09:27):**
    - 2x8-first (right after the job-start reset): PASS on both (57c55f98118 / a10u14, e5b75d1fbcd / a08u08).
    - 4x4: stability PASS on both.
    - 2x8-again (after 4x4): no reading on either; the stability command died on router JIT compile errors.
    - The sync failure did not recur.
    - The JIT errors appear only in these two jobs: 0 in all 24 earlier P10 steps; 5-52 per step here (2x8-first,
      4x4 then 2x8 order). Mechanism open.
    - Not the tree, not one node.
    - Every sync failure followed a 4x4 step in the same job (two earlier 4x4 -> 2x8 jobs passed). Order, state or
      intermittency remains.
    - Next, the coordinator's call: the same order with the JIT cache cleared between steps, or 4x4 -> 2x8 repeated
      on one node.
- **JIT compile errors in G-PORT-ORD (178007, 178014; 2026-11-05) are harness noise, not router failures** (coordinator).
  - **Seen:** router builds fail with "get_named_compile_time_arg_val not declared" or "kernel_main not declared" (a
    kernel directory's generated header missing or empty), mostly in the pass-through command, once in 2x8-again's
    stability command (178007).
  - **Harness stays CI-like:** one TT_METAL_CACHE / TT_METAL_HOME for all tt-run ranks; no per-rank caches.
  - **Mechanism open:** genfiles.cpp writes the generated headers via FileRenamer (tmp + rename), so it isn't a plain
    write race. Look into it only if it recurs.
  - **Reading:** 2x8 is read from the stability command's fabric init.
- **120-c host mapping stall, bh-glx-120-c02u14** (G-PORT 177902, BATCH c3c1813c42a, REGRESS_GLX_SC36_2 window,
  2026-11-05).
  - **P10-mm-4x4:** all 3 ranks start their devices (UMD 08:26:19.25), pass the seed broadcast and log "Using custom
    mesh graph descriptor ... bh_galaxy_split_4x4_2x4_3_mesh". Then nothing until the 600 s kill (HANG, capture rc 0).
    A healthy run logs "Physical groupings ..." 0.34 s after that line (a07u02, 177689).
  - **P10-mm-2x8:** stops at the same line (08:38:11; still silent 40 s later).
  - **Where:** the host control plane's physical grouping / topology mapping, before any router is built or launched.
  - **Hypothesis:** a mapping stall specific to this 120-c Galaxy's links. Not the router; not probed (the user's rule).
  - **Batch answer:** the 110-a hedge 177969 (bh-glx-110-a08u08).
- **Galaxy multi-mesh HANG on e5b75d1fbcd** (Slurm 176161, bh-glx-110-a09u02, 2026-11-04 23:35): P10-mm-4x4 went silent
  on all 3 ranks right after UMD device start (killed at 600 s); P10-mm-2x8 the same. The cleanup push is held. Tree or
  node not yet known; the port agent triages offline, then reruns mainline_cb42eb7 (PASS in 172825). Note: the pushed
  22cdb1e55c3 (drop5) was validated on the P150 only.
- **Decode A/B on the port: measured** (2026-11-05, agent a6e85079ff596efce; lb2 176226 on a07u28; SLOT_ALLOCATOR.md
  §12.13, report fabric_exp/port_slots/report_pvs_176226.md). Reading both route bytes makes the 2x4 mesh N/S rows
  faster (isolating: atomic -7.6, scatter -7.5, writes -1.0 to -1.7 cycles/packet; 96-97% of the atomic and scatter
  gaps to the study). The 4x2 torus N/S rows are faster by -10.0 to -11.9 (net, confounded by a per-proof stack
  reload, bound about 2), 0.4-0.6% faster than the study. E/W, 1D and NeighborExchange rows do not move. Next, for the
  user: a separate port commit with the both-byte decode; RECONCILE_DEVICE §7 A/B-1's Galaxy N/S rows and core census
  before the upstream PR. Mesh write rows keep +0.9% (4352 B) / +2.8% (16 B) to the study in both directions (not the
  decode). **Resolved 2026-11-05** (a09u32 job 798, RECONCILE_DEVICE.md §8.1): the test sender kernel. With the
  study's stream loop in the port's sender and router images identical, the port is -0.2/-0.3% (4352 B) and -1.7%
  (16 B) against the study. The 16 B lead is a hypothesis, cross-host.
- **Per-channel TRIDs on the port** (2026-11-05, agent a6e85079ff596efce): worktree
  /home/aagarwal/tt-metal-port-trid, branch aagarwal/fabric-router-mainline-trid at f9c258d280b (local, unpushed).
  Offline gate and device A/B in TRID_ALLOCATION.md §7. P150 lb2 176756, rule vs uniform:
  - 4x2 torus N/S +1.0% to +1.6% (the decode-site spill sits there);
  - torus E/W -0.5% to -0.8%;
  - ring 0.0%; mesh and 1D (codegen only) within ±0.1%.
  Hypothesis (not isolated): the spill, not the ID policy, costs the N/S rows. An A/B keeping 16 IDs without the
  reload would isolate it; its design goes to the user first.
- **Staged slot allocator on the port** (2026-11-04/05, agent a6e85079ff596efce): worktree
  /home/aagarwal/tt-metal-port-alloc, branch aagarwal/fabric-router-mainline-alloc, commit c53b5a5cbbd (local,
  unpushed). Per-VC forwarding units and a host guard after the 176405 hang (SLOT_ALLOCATOR.md §12.14).
  - Device A/B against tier (harness: test_infra_6dad + adapter_vc2_credits_6dad, local, router images unchanged):
    P150 lb2 176714 and Galaxy 176753, both clean.
  - Met: VC1 / VC2 (P150 and Galaxy; -19% to -59% medians), 1D 4352 B, ring, NeighborExchange.
  - NOT met: 2D 4352 B (forwarding rows +5.5% to +9.0%: staged 20,15,16 vs tier 17,16,16), 2D 3072 B (1-hop N/S
    +3.5%), 1D 3072 B (1-hop E/W +1.7%). The user decides; the allocator is unchanged until then.
  - 2026-11-05: tweaks (a)/(b) and the 32-slot cap in; then the 1D leftover order (user): 1D 26,25,32 at 4352 B and
    43,42,32 at 3072 B (SLOT_ALLOCATOR.md §12.15). port-alloc 9ce0acbeaba; final pair on 57c55f98118: allocator
    07322400fd6 + TRID c3c1813c42a. Offline gate passed. P150 lb2 176985: all four 1D / ring suites met the 1% bar
    (1D 4352 B median -7.3%, 1D 3072 B -0.3%, rings 0.0%; the old 1D 3072 B +1.7% loss is gone). Galaxy 176994
    ring-jam sanity: clean, no jam level; 8-ring 3584 / 4352 B -8.3% / -5.7% against tier, the rest within 0.1%.
    Shas sent to the coordinator; tuning agent told 9ce0acbeaba. Next: the port agent cherry-picks the pair.
- **Study vs port reconciliation** (user, 2026-11-04): the port (22cdb1e55c3) drifted from the study's v4txa11 and
  was only ever checked against main. Known items: the port kept main's branchy route decode (MAINLINE_PLAN.md:116,
  marked WRONG: LEDGER E1-18/E1-19 measured that shape at +5.7 to +6.7 on v3 N/S rows), and the port runs main's tier
  slot table rather than the study's v4 rows. (Narrowed 2026-11-04 by mock captures of both hosts: 2D mesh/torus at 4352 B get the
  same rows, 17,16,16; 1D line/ring differ, study 26,25,32 vs port 16,16,32; VC1/VC2 and other sizes in RECONCILE_HOST.) Device-side list: agent a4dc906ee7c1faf44 -> RECONCILE_DEVICE.md;
  host-side per-config args diff: agent a6e85079ff596efce -> RECONCILE_HOST.md; both offline. Then: the user decides
  per item; A/Bs for anything landing in the port run on port builds; the TRID per-channel A/B is redone on the port
  (the study-tree result and hypothesis C describe the study tree only).
  - Device list done (2026-11-04, RECONCILE_DEVICE.md; tools and data in reconcile_device/): 44 rows, (a) 12 /
    (b) 22 / (c) 10 / (d) 0. Open for the user: A/B-1 decode both bytes on the port (offline check: the port's N/S
    receivers load the X byte behind the Y branch, the study's load both together); A/B-2 cb42eb72501's register
    allocation shift on 44 Galaxy pass-through receivers (optional). Everything else: keep the port's.
  - Write gap after the decode (2026-11-05, RECONCILE_DEVICE.md §8, offline): on the 2D mesh routers the E1 write and
    forward paths are identical study vs port 6dad5fec638; E0 differs only by aa11a8786df's send placement and
    7cd3c13a0db's credit (both measured faster on the port). Nothing in router code explains the slower 1-hop rows;
    candidate: the arms' different test kernels (study's stream benchmark loop vs main's sender). Proposed A/B (not
    submitted): 6dad5fec638 with and without the study test infra on pvs_ns + pvs_c2d, one lb2 job.

- **Receiver tuning sweep on the port** (user via coordinator, 2026-11-04; RECEIVER_TUNING.md): IDLE_PASSES_BEFORE_
  PARTIAL_CREDIT (1/2/4/8) and COMPLETION_PROOF_MAX_CHECKS (8/32/128), plus a delay-perturbation pass on the idle
  arms. Worktree /home/aagarwal/tt-metal-port-tune (22cdb1e55c3 + local fabric_exp/port_tune/port_tune.patch, never
  committed), staged at /data/aagarwal/lb2/trees/port_tune_22cd. Offline gate passed (default arm = unpatched port on
  1,328 images; every deadlock-avoidance receiver identical across arms). lb2 176267 idle, 176268 proof, 176269 pert
  all done (§5). Proof cap stays 32 (coordinator). Idle passes: 1 / 2 regress 2D and amplify sender timing; 4 vs 8
  held until a rerun on the staged allocator's slot rows (planned, RECEIVER_TUNING §6; waits for the allocator's
  device A/B, i.e. the user's go). 2026-11-05: the rerun on 4b524653490 (lb2 176425) hung at its first data-checked
  run like the allocator's own A/B 176405 (staged arm, no tuning patch): an allocator problem, not the idle arm;
  the rerun waits for the allocator's fix (RECEIVER_TUNING §6). Comment texts proposed in §5.4 for the port agent. 176008 and 176130 stopped at the old link check
  on ports outside the fabric; the harness now gates on fabric links only (LB2_HARNESS.md "Link gate"). A value changes only on a full census with no suite regressing.

- **DeepSeek CCL perf matrix** (user, 2026-11-02; **parked by the user**, local SDPA test edit reverted): ccl_trace_perf_matrix.py on main's router (main_fw, baseline),
  v4rxo0b and the receiver lead candidate v4rxf4. Payloads 2048/4096/4352/8192/15232; SDPA tuned only. ttnn built in
  build_ttnn. Runs via ccl_matrix_runs.sh ROUTER CCL [LINKS], PASS=a then b (reversed router order). Results in
  fabric_exp/results/ccl/<router>/<pass>/; table via ccl_matrix_report.py.
  - Passes a and b done (all_gather, all_reduce 1/2 links, broadcast 1/2 links, SDPA): v4 vs main all_gather
    -3.6..-4.6%, all_reduce -2.7..+0.6% (4096 B flat), broadcast -0.6..-1.5%, SDPA -2.3..-3.8% at 4096-8192 B and
    -13.5% at 15232 B (main 6.32 us there, above its own 8192 B 5.64). v4rxf4 = v4rxo0b within noise on every cell; SDPA
    2048 B is noisy on every router (five passes, job 444: medians main 5.69, v4rxf4 5.40, v4rxo0b 5.38).
    Write-up: CCL_MATRIX.md.
  - **SDPA failed on every router before the op ran:** the test gives the row-major output tensor a custom tile, which
    page_config.cpp rejects (TT_FATAL since 2026-07). The runs used a local test edit,
    reverted 2026-11-02 (re-apply to rerun): test_sdpa_reduce_to_all.py output_l_mesh tile=None.
  - **reduce_to_one fails on every router:** the test's trace region (trace_region_size 1 MiB = 128 KiB per DRAM
    bank) holds 128 traces of 8 KiB; the benchmark captures 4 roots x (15 warmup + 30 perf) = 180. Not router-related.
    Waiting for the user's choice (2 MiB trace region by a local test edit, recommended; or fewer samples through the
    script's flags).
  - Broker jobs run with exit-on-error: chain independent steps with `|| rc=$?`, not `;`.

- **Sender pass on v4rxg3** (2026-11-02, sender agent ab746835cd3ae078a; V4_SENDER_OWNERSHIP.md §12 stages). Status
  2026-11-02 evening:
  - X3b adopted (loop-local VC visited in place); X3f (fence decisions: L286 deleted, L312 order_point) byte-identical
    to X3b on all 2,863 configs.
  - X4 (credit command buffers): host patch variants/host_rebuild_20261102b, in the live build since job 464
    (SENDER_CH_<i>_ACK_CMD_BUF_ID: VC0 worker WR, VC2 WR_REG). Loops move only in VC2 families (-O3 +6, in the
    per-round preset restores).
  - X5 / X5b (one shared worker_connection_event, cold path): X5b loop-identical to X4 on the quick set; full coverage
    running.
  - Validation of X5's connection path: job 473 (variant_val 13/13 set) exit 0; job 474 (inline-write, flushed-atomic
    gtests, UDM) running.
  - Main session's calls (2026-11-02):
    - perf A/B approved: X3f / X4 / X5b, core + census + vc1 + VC2 rows, two passes, ~6 jobs; X5b's perf arm dropped
      only if full coverage shows identical loops;
    - X6 (telemetry from the channel's `sent`, functional check of host totals), X7 (single::run shares start and
      round helpers), X8 (comments), then the final census and validation;
    - before queueing: prove v4rxg3 and main images byte-identical under the job-464 host rule. Report jobs since 464
      ran on it.
      - **Result for v4rxg3: not identical, init-only.** 1,695 of 2,863 images differ: every image that runs the sender.
        The changes are the worker-interface constructor storing the unused sender_sync_noc_cmd_buf, its init call
        site, and crt address constants after the text shift. No sender, receiver or round line changes.
      - Cross-run device check (txos1/2 before 464 vs mvc[ab] after, same pcs3 yamls and slots): 10 rows within
        0.12%, 9 within 0.02%. **Accepted; no reruns.**
      - Main: -O3/-Os check due ~23:30. If main's loops change: stop. "Main as shipped" matters; the fix might gate
        the rule to the study kernel. If init-only: the same cross-run check on main's rows.
    - X7 is identical modulo operand order (beq operand swap; one-ERISC: two loads into a commutative add). X8 is
      comments only, same line count. X6's 32-bit wrap is unreachable; the check states it.
    - Device round ready ~01:00 2026-11-03; ~1 h validation + 2-2.5 h census A/B (5-6 jobs).
    - **Main under the job-464 rule: init-only** (worker-interface ctor/init, crt, one moved jump-table immediate).
      Cross-run check: 36 same-yaml main rows (census Oct 29 vs the sweep Nov 2), median +0.01%, max -0.7% (half-ring
      multicast, within its own pass spread). **Accepted.**
    - X7 correction (marked WRONG): the beq swap is in the pass (visit_worker), not round code. The opcode at every
      address is the same; only registers and operand order differ. Accepted as identity; the census measures it.
    - Offline done:
      - X5b = X4 loops;
      - X6 changes only the telemetry/all-instrument images (sender loops -18..-93, stack loads +2..+7);
      - headroom equal to X3b;
      - size vs v4rxg3 (32 shared cases): -O3 VC1 +60 B, -Os -312 B. E1 least free stack -340 B on bh_vc1: stage
        A's worker_interface_storage (336 B .bss) is in E1 images too. Code-cleanup item.
    - **User review of v4txo8 (2026-11-03): consolidate the sender like the receiver.**
      1. Loose facts/constexprs should live in owners (as reads_free_slots_at_pass_start in SenderVc).
      2. worker_credit_cmd_bufs_distinct duplicates the receiver's forward_cmd_bufs_consistent: one command-buffer
         ownership owner for both ERISCs.
      3. worker_channel_positions belongs in its owner.
      4. forwarding_index is loose.
      5. Don't rely on first_sender_channel_by_vc for the worker channel.
      6. The worker interface code should be wrapped in a struct (a channel kind).
      A fresh agent (user's suggestion) is writing the design note V4_SENDER_CONSOLIDATION.md first: inventory,
      owners, stages with identity targets, and stage 0 merging v4txo8 + v4rxnsf. Code only after the user's review.
      **Design note done** (consolidation/V4_SENDER_CONSOLIDATION.md, 688 lines). Stages:
      - C0: merge v4txo8 + v4rxnsf2;
      - C1: facts into owners (identical); C1b optional: the served-worker pass-start rule;
      - C2: one CommandBuffers table for both ERISCs (identical);
      - C3: WorkerChannel / ForwardingChannel kinds + SenderChannelInterfaces + v4txo8w (loops identical, init moves);
      - C4: comments.
      Findings: forwarding_index is dead code; SenderVc::first_level_ack is unused; end_round's send_yielded is
      misnamed; the worker credit preset lacks the DPRINT/watcher guard.
      **User approved the main session's picks (2026-11-03):**
      1. names SenderChannels / WorkerChannel / ForwardingChannel / SenderChannelInterfaces / CommandBuffers;
      2. command-buffer conflicts are static_asserts;
      3. served-worker pass-start rule (C1b) + trimming validation + a short A/B;
      4. DPRINT/watcher guard on the worker credit preset;
      5. receiver mirroring later (only the shared CommandBuffers now);
      6. wait_for_* into router_sender.hpp;
      7. v4txo8w in C3.
      **Implementation agent started** (stages v4txc0..v4txc4). It reports after C0, C2, C3/C4 and asks the main
      session before queueing device jobs.
    - **Helper-file review** (user asked the main session, 2026-11-03): consolidation/ROUTER_HELPERS_REVIEW.md.
      Part A: cleanups in router_common / init / round / local_delivery / telemetry / single / kernel file. Part B:
      push scope, settled by the user: the preview is code shape only (router/*.hpp + the kernel file), nobody will
      run it, so no host patch, rebase or kernel-slot work. **Part A approved (user, 2026-11-03)** and sent to the
      consolidation agent as stages after C4. The user stressed "no random refs in the code for the team to see"
      (A9: zero study pointers in every router file, grep-verified).
    - optimize("Os") on run_handshakes (measured on v4txc4 probes, -O3):
      - without it, two-ERISC E0 grows +576..+796 B (the function 1.1-1.2 KB -> 2.7-3.1 KB);
      - one-ERISC images +0..+148 B; stack unchanged;
      - VC1+VC2 margin 1,808 -> 1,052 B. On this base the margin is 1,808 B, not 240: nsf2's E1 shrink and C3's
        interface storage leaving E1, at the default allocation.
      **User: keep it** (2026-11-03); optimize("Os") is allowed on once-run init code only, with a one-line why.
      **C1b: no extra A/B** (option (c) stands).
    - C0 validation: 525 13/13 (trimming 63/63 both modes); 526 (inline/atomic + UDM) running. C1 is byte-identical
      to C0.
    - C1b changes only 112 T1/T7 images (VC0 worker trimmed, VC2 served): -O3 loop -2 (two-ERISC), +3 (one-ERISC);
      -Os +16 with spills. No device config produces these shapes, so its device step is option (c), skipped, with
      the gap noted. The user may ask for (b), a mock-vetted VC2 + VC0-worker-trim override A/B.
    - C0 validation passed (525/526). C2 (CommandBuffers) is byte-identical, 9 loose facts replaced. C3 (worker /
      forwarding kinds, interfaces, v4txo8w): loops identical per function, init-only; loop starts move (-60..-160 B
      E0), so it gets a census. VC1 E0+E1 -348 B; bh_vc1 E1 free stack back to 4,236 B. C4 (comments) is
      byte-identical. Lines: router_sender 1,041, v4.cpp 232.
    - Device 04:05: census c3a/c3b on a09u32 jobs 527 (core) and 528 (vc1); validation of v4txc4 on a08u28 Slurm
      169770 (lb2; collect with lb2_collect.sh 169770).
    - A2 decision (main session): owner-based init_streams before the first sync, writing only owned registers; a
      behavior change with full validation. Its build is in progress.
    - **C3 census 527/528:** every row within ±0.3 (2D, one ERISC, VC2 study + v8, VC1 split 158.4 -> 158.3,
      inter-mesh 332.9 = 332.9). **C4 validation** (a08u28 Slurm 169770, collected): 13/13, inline/atomic 1D/2D,
      UDM 8/8 + 68/68. The consolidation C0-C4 is done.
    - A1 is byte-identical to C4. A2 (streams by owner) changes init only, loops identical. The written register set is
      the same on 3,104 configs; bh_neighbor* (20) no longer writes the unowned position-1 free-slots register,
      argued safe. A2 validation: a09u32 529 (variant_val) / 530 (inline/atomic + UDM). Then one final v4txa9
      validation + census C4 vs A9.
    - Part A: A1-A7 and A9 are written. A1 keeps the runtime downstream_compact_index form (a shared helper changed
      receiver::run's schedule on 26 capture/replay images). Full builds are running; A2 validation commands follow.
    - A2 passed (529/530). A3 changes init only. A9 is comments only and has zero study pointers. A4 moves loops
      (the round split and this_chip(): VC2 receivers -22..-56 instructions, VC1 -4..+28, sender a swapped compare),
      and the trimming-recorder hoist was reverted. Kept in A9; the census decides, with no fallback built without
      the user. Final runs on a09u32: 531 val, 532 inline/UDM, 533 core, 534 vc1, 535/536 census set (C4 vs A9).
    - 531/532 all pass (mesh 96/96 both modes, val1d 143/143, ring 19/19, torus Y 12/12, VC2 10/10, mux 19/19,
      nbr 7/7, trim 63/63 both modes, inline 2+2, UDM 8 + 68).
    - **A10 (user go, 2026-11-03): kernel file cleanup** (v4txa9 -> v4txa10).
      - Includes cut, so router headers are self-contained with no file-scope using.
      - The bring-up gate becomes per-feature supported-configuration asserts in router_common.
      - A finish_phase(Phase) helper replaces the interleaved syncs, with main's postcodes kept in one table.
      - An init_router() prologue; the dead data-cache call is dropped; receiver setup owns the downstream
        connections.
      - Loops must equal A9's; then validation only. The review list then covers A10.
    - **A9 census (533/534):** VC1 2-hop rows regress by +0.5..+3.4 (merge 16 B +3.4; VC1 mean 158.3 -> 159.0). The
      probes trace it to this_chip()'s by-value return reallocating registers in receiver::run (a hypothesis); the
      round split is +0 on loops. VC2 torus N/S improved by -1.1 from the same change.
      - **User decision: revert this_chip().** A11 = A10 + the revert, with receive loops back to C4's. One device
        round on A11: val, inline/UDM, vc1 and core vs C4.
      - A10's loops are identical to A9's on all 2,863 images. A11's receive loops differ from C4 only by operand order
        and register names: the round split, l1_routing_table(), and the telemetry helpers. The only count changes are
        on test-only images (-1, +2).
      - **Pushed the preview (user, 2026-11-03):** branch aagarwal/fabric-router-v4-preview at 4ca9c94a252
        (v4txa11's router headers + kernel file on origin/main). The user opens a draft PR. If A11's validation or
        census fails, push a fix commit there.
      - **User: option (c).** A11 is kept as it is, with the census set added. On a09u32: 544 val, 545 inline/UDM,
        546 vc1, 547 core, 548/549 census (C4 vs A11).
      - **544-549 passed** (V4_SENDER_CONSOLIDATION.md §6.10): validation 13/13, inline/atomic 4/4, UDM 8/8 + 68/68;
        census flat (every row within ±0.3 of C4). A9's VC1 rows are back (merge 16 B 137.5, 4352 B 142.9), so the
        revert attributes them to this_chip(). ew4352-1e +0.2: identical loops at moved addresses (hypothesis:
        placement).
    - User plan (2026-11-03): after the consolidation, push just the router files to a branch and share it with the
      artifact; refresh the numbers in the background. The page stays silent about in-flight fixes until the
      refresh.
    - Later (user Q5): mirror the receiver: one owner for the VC -> receiver-channel index (ct receiver_channel_by_vc
      vs the receiver's receiver_channel_of), ReceiverVcs.
      The VC2 default-slot stage bisect and v4txo8w are still open; v4txo8w folds into point 6.
    - **Final build v4txo8: validated; waiting for the user's code review** (agent done 2026-11-03).
      - Bisect 507: v4txo4 = v4txo8 within 0.1, so the VC2 default-slot regression (+17.7, +2.1%) sits in X0..X4.
        **Parked by the user (2026-11-03).** The stage bisect when picked up: one job on the three d- rows (d-vc2ew4352,
        d-vc02ew4352, d-vc02bew4352), arms v4rxg3 / v4txo1 / v4txo3k / v4txo3fn / v4txo4, two passes, ~15 min.
        Hypotheses (untested): the X3/X3k placement, or X4's VC2 credit path (WR_REG preset).
      - VC1 inter 1_1/1_2 4352 (+3.6/+1.7) and 1_3 16 B (+6.5) were already at X3a (jobs 453/456): not new.
      - Review package: diff loopvars_maps/sender_pass/v4rxg3_to_v4txo8.diff; reading order in the agent's report
        (router_sender.hpp sections); v4txo8w diff loopvars_maps/sender_pass/v4txo8_to_v4txo8w.diff (option).
    - **Round results (496-501):**
      - validation 13/13, inline/atomic, UDM 8/68 pass; X6 telemetry totals equal main's (204 readings);
      - census vs v4rxg3: mostly ±0.3, several better (v8-vc02bew VC0 -7.8/-8.1/-2.1);
      - worse: default-slot VC2 rows +2.1% (vc2ew4352 857.3 -> 875.0), VC1 inter 1_3 16 B +6.5 (since X3a),
        1_1/1_2 4352 B +3.6/+1.7;
      - bisect 507 (arms v4rxg3 / v4txo4 / v4txo8) moved to the front 00:55; the receiver jobs were resubmitted
        behind it as 508 (was 503), 509 (506), 510 (504), 511 (505);
      - v4txo8w (worker_interface_storage only on sender-serving ERISCs, 3 lines) goes into the user's review as an
        option, not adopted.
    - **Device round queued 2026-11-02 23:54:** jobs 496 (validation), 497 (inline/atomic + UDM), 498 (X6 telemetry
      totals vs main), 499 (core census, f1/f2), 500 (census set), 501 (VC2 default slots + VC1). ~80 min. Then the
      user reviews the sender code.
    - Sender agent, 23:54: main (mainx5pin, pinned headers) is init-only too: worker-interface constructor and init,
      crt constants, and one jump-table address constant after a 4 B code shift (56 images per level). X7: the beq
      swap is in the pass (visit_worker), not round code (WRONG noted in §12); full coverage: same opcode at every
      address, only registers/operand order differ; loops +0. Offline done (X5b loop +0 everywhere; X6 telemetry-off
      identical; headroom X4..X8 = X3b). Device round on v4txo8 queued as jobs 496-501 (val, inline+UDM, telemetry
      totals, census core / census / vc2d+vc1, passes f1/f2).
    - Round done 2026-11-03 01:10: validation 13/13, inline/atomic 1D/2D, UDM 8/68, telemetry totals = main's (two and
      one ERISC). Census: within ±0.2 on 1D/ring/torus/NS/scatter/one-ERISC; VC0 under VC2 contention faster (to -8);
      **open: VC2 with default slots +17.7 (+2.1%) alone, +17.7 / +5.0 under contention; bisect (job 507) puts it in
      X0..X4 (X5b..X8 +0.0); which stage is open (hypotheses: X3/X3k placement or X4 credit path).** VC1 inter 1_1 /
      1_2 4352 B +3.6 / +1.7: already at X3a (X3, X3k). Option for the user: v4txo8w (worker interface storage only on
      sender-serving ERISCs; not adopted, not validated). Review package sent to the main session.
  - Then breakdown P150 phase 2 on the final build. The VC2 line branch needs a second rebase onto it (small).
  - **P150 priority (user, 2026-11-02): the sender pass goes first; the user wants to share the new router shape
    with the team.**
    - Device work collapsed into one round on the final build (X8): validation including X6's telemetry check, plus a
      census A/B against v4rxg3 with VC2 rows, two passes. The intermediate X3f/X4/X5b A/B is dropped; bisect with
      the X4 arm only if a row regresses.
    - When the round is ready, the main session cancels the queued jobs that haven't started (others' and its own)
      and resubmits them behind it with the same commands. The multi-VC and receiver agents are told to wait for the
      new ids and to ask before queueing more P150 jobs. Held commands, in order: consolidation/queue_after_sender.txt
      (the receiver agent's former 482-485, plus 479-481 if they have not started). Never cancel a running job.
  - **After X8 lands (user, 2026-11-02):**
    - the user reviews the sender code first. No router-shape page; the main session doesn't write one.
    - If the review is OK: a focused push of the new router code as a preview, while the cleanup continues. Scope
      and branch to be settled with the user then; push only when asked.
- **VC2 line rebase onto v4rxg3** (2026-11-02, VC2 line agent ad2a55e9ebda3aa9f, worktree tt-metal-vc2line): 2x2
  re-check, then meshes larger than 2x2 on Galaxy; reports before phase 5. **Galaxy: only bh-glx-110-a* nodes
  (`bh_sc36_6`, 110a pods) until the user says otherwise** (2026-11-02; applies to the slot study's P4 too).
  **Done 2026-11-02:**
  - Single mesh, 2x2 and 4x4 splits all pass data checks (jobs 168774, 168925, 168926), on both ERISC counts at
    -O3/-Os. The 4x4 runs are the first device runs of buffer-2 fix (c).
  - The 200G all-three-VC rerun (169196) was cancelled before it ran; 400G only, per the user.
  - The phase-4 "4352 B all three >= 274" is WRONG: it is <= 274.2 cycles/packet, a lower bound on throughput. Marked
    in VC2_LINE_BRINGUP.md; fixed in report/data/fanin.json and gen_fanin.py.
  - The sender pass will need a second rebase (small: the VC2 forwarding channel and an assert).
  - /data/aagarwal/frs-glx/jobs/168925 and 168926 deleted (checked 23:10).
  - The agent is done; nothing of it is queued or running.
  - **Phase 5 (multi-Galaxy) on hold by the user (2026-11-02: "hold off onto phase 5 for a while").** Open question
    for when it resumes: does a 2-node job count as one allocation or two against the cap of 4 (recommended: two). Plan in VC2_LINE_BRINGUP.md: pod partitions, two-Galaxy mocks first, new multi-node sbatch
    (TCP MPI with an explicit interface), then VC0 connectivity, VC2 line rings across hosts, two meshes across
    hosts, four Galaxies.

- **Route-word preload (N/S fix 2) + strict flush-defer: design note** (user, 2026-11-02; receiver agent
  ae61797a992a87495): stage 1 route word only (header word a later arm), an additive codec entry decoding from loaded
  bytes (variant copy only), flush-defer designed in the same note and built after. Design note with static counts and
  predictions first; the user reviews before any adoption build. **Note done** (V4_ROUTE_PRELOAD.md). **Study A/B done**
  (jobs 457-459, §8): the word-form preload is flat or a cost everywhere, including the multi-downstream proxy
  (3-hop +1.7/+2.4, turn +3.1/+4.6); N/S rows +1.8..+13.2, one ERISC +12..+15. PC samples: the route-word samples
  fell ~4, the admission samples did not (spread over the loop: executed bookkeeping, hypothesis), so V4_NS_FLOOR §2's
  "admission waits on the route load" is marked WRONG. **Parked by the user (2026-11-02)**: draft
  variants/v4rxrp3 and §8 kept. Flush-defer (S3) stays queued on its own; N/S fix 1 is next on the receiver side.

- **Study report page** (user, 2026-11-02): page first, deck later; iterate on layout.
  **Refresh on the final router (user go to prep, 2026-11-03).** Router: v4txa11 (the pushed preview). The queue goes in
  only after A11's 544-549 pass. The ring-jam fix doesn't block it; when it lands, the ring/torus suites and the
  census rerun.
  - Consolidation agent, a09u32 + offline: the 400G size sweep (mesh/line/ring/Y torus), one-ERISC/mux/VC1 split,
    multi-VC (3-VC, VC0+VC1, VC0+VC2 matched), same-job latency, headroom delays, P150 breakdown, sizes.
  - Ring agent (resumed for this; the jam work stays paused): the Galaxy 200G sweep (mesh/line/ring/XY torus, main +
    main ka1000 + A11), Galaxy breakdown points, and the 400G XY torus 2x4/4x2 on lb2.
  - Both send ordered command lists first; Claude queues them.
  - **a09u32 queued (A11 passed 544-549):** 550/551 size sweep, 552/553 one-ERISC/mux/VC1 split, 554/555 multivc,
    556 latency, 557 P150 breakdown, 558 XY torus 2x4/4x2 with validation (moved here from the ring agent's LB list,
    the same host as the current data), 559 census two1 (the stream-mix rows). 550 is clean. J7 (VC0+VC2 matched)
    follows after mk_mvc_matched.
  - 552-555 are clean. More cross-run moves on v4txa11 that the chain census never covered:
    - trimmed 2D rows: MeshMulticast#1 write 136.8 -> 139.1, capture-pass rows +1.2..1.6;
    - VC2 at default slots: VC2 alone +1.3..1.9%, under contention +8..19; VC0 under contention +2.4;
    - VC1 inter-mesh +3.0/+2.1, already known from v4txo8.
    Queued: 561 J7 (VC0+VC2 matched), 562 the trim same-job A/B, 563/564 the VC2 default-slot bisect (vcba/vcbb).
    - 563/564 were cancelled unstarted: job 507 already bisected the VC2 default-slot step to v4rxg3 -> X4, flat after.
      The agent's "never run on the chain" was WRONG. The X0..X4 split stays parked by the user's earlier call.
    - **560 bisect:** the whole ring-multicast write step is in v4rxg3 -> v4txo8 (X0..X8): +2.5 / +2.9 / +1.4 / +0.9 /
      +3.3 at 256-4352 B; o8 -> c4 -> a11 is flat. 565 splits X0..X8 with the stage builds on disk.
    - **565:** the step is in X3 -> X3k (packed remote receiver progress; +1.0 / +1.1 / +0.1 / 0.0 / +2.8) plus X3k -> X3a
      (interfaces at a fixed location, UpstreamCredits gone; +1.5 / +1.7 / +1.2 / +0.7 / +0.9). X3b..X8 are flat;
      scatter is flat. Same-job stage A/Bs, so the stages are established; the mechanism is open. The agent is doing
      an offline static look and proposing isolating A/Bs. A fix design goes to the user.
    - **566:** the trimmed MeshMulticast replay step (+2.5 write, +1.0 scatter) is in o8 -> a11 (the C/A chain); the
      capture-write step (+1.2 / +1.7) is in X0..X4. 567 splits o8 -> c0 -> c4 -> a11.
    - **567:** the trimmed-replay step is entirely at C0 (the v4txo8 + v4rxnsf2 merge): write +2.4, scatter +1.1. C0 ->
      A11 is flat. Interval only.
    - **Offline look at X3k/X3a (ring mcast write):**
      - Only E0 changed. Both stages remove E0 work (loop 327 -> 324 -> 317, stack loads 31 -> 28 -> 22), and every
        ring row runs the same images.
      - So it looks like a timing effect, not added cost (hypothesis). Lead H1: worker/forwarding arbitration of far
        slots under the bubble rule. H2: loop placement.
      - 568 is H1's env-only A/B: X3/X3k/X3a at far-receiver depths 26,25,32 / 16,16,51 / 33,34,16.
      - **568 result: H1 (far-receiver depth) is not supported.**
        - The steps are flat from receiver 32 to 16 (+2.8/+0.9 vs +2.9/+0.9).
        - At 16,16,51 they vanish because X3 slows to ~150 (X3k/X3a stay ~150).
        - So X3's 147 needs the deeper sender windows. H3 (hypothesis): X3k/X3a change E0's timing in a
          producer-side credit loop (worker -> E0, or upstream E1 -> E0 forwarding). H2 (placement) stays open.
        - 569 cuts one window at a time (16,25,32 / 26,16,32).
      - **569 result: H3 is not supported as a single window.**
        - worker 16: X3 147.1, X3k 149.9, X3a 150.8 (steps +2.8 / +0.9);
        - forwarding 16: X3 147.7, X3k 150.2, X3a 151.1 (+2.5 / +0.9).
        - X3's advantage survives either cut alone. Only 16,16,51 (both cut and a 51-slot receiver) removes it.
        - To separate them: arms 16,16,32 and 26,25,51 (env only, ~5 min). H2 (placement) stays open.
        - **PARKED by the user (2026-11-03).** The page shows it as measured. Next if resumed: 16,16,32 vs 16,16,51.
    - Probe runs:
      - v4txa11v: 570 VC1+VC2 correctness clean; 571 3-VC a11a clean (1,376 B left at the default allocation); 572
        3-VC matched a11b.
      - v4txa11d: identity gate passed (2,511 identical; the 352 differing are exactly the delay cases, as v4bdd).
        Delay arms 573-576 (two/one+vc1, x11a and x11b reversed).
      - breakdown.json: main E0 at p2/p4 dropped in the builder (126 records).
    - **User decisions (2026-11-03):** go on v4txa11v + v4txa11d (the consolidation agent builds them; v4txa11d is gated
      on offline identity); drop main E0 at breakdown points 2/4 (the ring agent does it in the builder).
    - **Trimmed C0 step:** the user approved an offline look (consolidation agent): which ForwardCommands mode the
      trimmed replay receivers take, o8 vs c0, and the loop diff. Hypotheses plus isolating A/Bs; the fix design
      goes to the user.
      - **Result (offline, scratchpad/c0trim):** E0 is identical at C0. Trimming doesn't change the mode choice, and
        the serviced E1 images are identical trimmed vs untrimmed. C0 changed the mid-row N/S two-entry receivers
        from Stateless to PartialPreset (loop -46 instructions, +4 ready reads). MeshMulticast lands on exactly those;
        Mesh3HopEW uses E/W receivers only, which are identical.
      - H-C0a (hypothesis): PartialPreset's forward waits on both cmd bufs when either is busy, which hurts two
        back-to-back forwards (E and W fan-out).
      - H-C0b (hypothesis): untrimmed MeshMulticast is E0-bound (job 349: E0 slack 0, E1 26.2). Trimming lightens E0
        and hands the bound to E1. **WRONG for the write row (577: trimmed write #1 E0-bound, TX slack 4.0, RX 20.2);
        holds for the scatter row only (RX slack 7.2).**
      - A/Bs: H-C0b is env only on v4txa11d with a trim suite (after 576); H-C0a needs a variant (wait order restored),
        which needs the user's go.
      - **577 (H-C0b):** trimmed MeshMulticast write #1 is E0-bound (TX slack 4.0, RX 20.2), so **H-C0b is WRONG for the
        write row**; scatter #1 is RX-bound (RX slack 7.2). C0 changed only E1, so the write row's +2.4 reaches E0
        indirectly (hypothesis).
      - **User go:** variant v4txa11w, a decoupled slow path in PartialPresetForwards::forward (fast path unchanged;
        slow path waits data -> data -> waits credit -> credit). The ready reads go in a read group (folded in if
        byte-identical alone, else arm v4txa11g). Joint test stays the fast path. Rows: trim, multi-downstream P150,
        VC1, census core; Galaxy N/S later.
      - **Built:**
        - v4txa11w (shape only): changes only the 378 two-ERISC PartialPreset E1 images; fast path unchanged. Slow
          path +22..+126 loop instructions; E1 +108..+512 B (bh_vc1 E0+E1 22,744 B, 3,880 B left).
        - The read group alone was mixed (it pulls a link-field load between the ready reads; use distance 2 -> 1;
          +1..+9 instructions), so it is a separate arm, v4txa11g.
      - Queued: 578/579 validation (w, g), 580 inline/UDM, 581 trimfwd + mdown rows, 582 vc1, 583/584 core.
      - **Result (587-593, all clean):** w and g buy nothing.
        - The trimmed scatter replay is flat on w, so **H-C0a is WRONG (not supported)**.
        - The write replay is flat.
        - w loses +0.7..+1.0 on VC1 rows, and costs +108..+512 B of E1 code.
        - The agent recommends adopting neither.
        - The C0 trimmed step stays unexplained (H-C0a and H-C0b-for-write both ruled out). Proposed to the user:
          park it. The new router is still about 17% faster than main on that row, and only trimmed routers move.
      - **578 hung:** v4txa11w's first validation run (two-ERISC mesh sanity) gave no exit after 300 s.
        - "last tests started" is empty and the capture kept nothing, so it may have hung before traffic (to check).
        - 580-584 were cancelled; 579 (g validation) runs as a control.
        - Per the rules, no hang probing and no fallback without the user. The mechanism comes first.
    - **Ring-jam fix design (RING_JAM_FIX.md), with the user:**
      - The probe's 400G cost was register pressure: the count took a loop register, the worker send spilled, and the
        torus gate was inlined 3x.
      - Mechanism (hypothesis): per-pass 1:1 channel arbitration when the far ring has room.
      - Candidates: C1 contest gate (owed counter in memory); C2 in-ring burst (up to K forwards per pass; recommended
        by the agent; long passes delay the per-pass connection check). C3 (refusal trigger) rejected.
      - K from the host (ring length - 2). First step proposed: a telemetry-only trigger check.
      - **User decisions (2026-11-03):**
        - C2 primary, C1 as the comparison arm.
        - The connection-check concern is equivalent for both: a worker can't disconnect before injecting; C2 adds at
          most one burst.
        - K is a cap: min(K, backlog) from the pass snapshot, never waiting, stopping on no far slot.
        - The host K argument is OK.
        - Telemetry check FIRST: v4txa11j counters for contests with room >= K+2, contests with < 2 free slots,
          refusals; on Galaxy 8-ring 4352/3072 and P150 ring mcast 3072/4352.
        - C2/C1 get built only if the room test separates the regimes.
      - **v4txa11j counters perturbed the router 1.6-2.5x** (Galaxy 171203: 8-ring 4352 B 374.6 -> 607.4), so its counts
        are unusable for the rule. The agent's "a few increments per pass" estimate was WRONG.
        - lb2 171206 was cancelled before it started.
        - Next: v4txa11k, round-sampled (nothing in the pass; one sample per 32-pass round; fractions), off
          byte-identical, cost measured off vs on, a correlation argument or jitter.
      - **v4txa11k Galaxy (171300): the room test does NOT separate, so stop on C2/C1 as designed.**
        - Jam rows (8-ring 3584/4352 B): room >= 8 in 0% of contests, room < 2 in 95-96%; in-ring ready with no far
          slot 72-78%; backlog 11+. It is a full-ring state.
        - Control 3072 B: in-ring no far slot 29%, backlog 2-3.
        - The k cost on the 8-ring: +3..8% (a confound: the worker loses 1 slot).
        - RING_JAM_FIX §3's mechanism (1:1 arbitration with far room) and §4 are marked WRONG; so is FULLRING §8.13's
          "check sees free slots" detail. The full-ring reading stands.
        - New candidate separators from the data: in-ring backlog depth or in-ring credit starvation (C3-like;
          rejected earlier by reasoning only). The P150 counts (lb2 171279, pending) decide them; the agent is
          pre-registering predictions and a rule. Taken to the user.
        - **v4txa11k built (2026-11-03; RING_JAM_FIX.md §8.2):**
          - One sample per round at a rotating pass index (13 r + 5 mod 32), as jitter. A round-boundary sample
            can't be argued uncorrelated: the boundary is a pause, and a 32-pass round phase-locks to periods that
            divide it.
          - Sample and flush are out of line.
          - Off is byte-identical (408 / 408, 1056 / 1056). On passes the headroom check on all 59 exact pairs
            (ring +796 B, torus +868 B).
          - Tree `trees/v4txa11k_1103`; Galaxy configs a11k / a11kam; P150 `VARIANT=v4txa11k frg_jam_p150_runs.sh`.
          - Commands sent to the coordinator; the decision rule is unchanged.
          - 171203 (v4txa11j) counts are recorded in §8.2 as a perturbed reference. Its 4-ring row was not slowed and
            showed room >= 4 in only 24-26% of contests.
        - **Galaxy 171300 (v4txa11k, 2026-11-03): the room test does NOT separate (RING_JAM_FIX.md §8.3).**
          - Jam rows (8-ring 3584 / 4352 B): room >= 8 in 0% of contests, room < 2 in 95-96%, in-ring backlog 11+.
          - Per the rule: stop, no C2 / C1 build, no fallback without the user. The P150 half (lb2 171279, pending)
            cannot change the verdict.
          - Cost: +8% (3584 B) and +3-5% (4352 B) on the 8-ring, other rows <= 2%. Confounded by the on-arm's worker
            slots 26 -> 25.
          - Marked WRONG: §3's mechanism for these rows, §4's "far ring mostly free", predictions 1, 3 and 4, and
            FULLRING_MCAST_GAP.md §8.13's "the check sees free slots".
          - Open for the user: the next step on the ring jam.
        - **P150 lb2 171410 (a07u28, via the new `lb2_run.sh -n any`; RING_JAM_FIX.md §8.5):**
          - Rule S (in-ring ready with no free far slot) SEPARATES: Galaxy jam 59-78%, against P150 11-32% and the
            Galaxy 3072 B control 29%.
          - Rule B (backlog >= 8) does NOT: P150 Unidir and Ring write rows are at 78-100%.
          - No cost flag: RingMulticast write +4.2% / +1.9%, others <= 0.2%.
          - §5.3's C3 rejection reasoning and the §8.4 backlog predictions are marked WRONG.
          - Next step (a rate-based refusal trigger? its own A/Bs) is the user's; no build.
    - **Bubble-threshold sweep (user go, 2026-11-03; RING_JAM_FIX.md §9; variants/v4txa11t):** injection needs T free
      far slots, T 2..6, one T per arm. Off is byte-identical to v4txa11.
      - Galaxy 171672 (a03u08): 45 / 45 steps.
        - 8-ring 3584 B: T5 -24%, T6 -27% (below main's 261).
        - 8-ring 4352 B: -2 to -4% (main 263.8 stays far).
        - 4-ring 4352 B: +4%.
        - Half ring 4352 B: -1 to -5%.
        - Torus: -0.4%.
      - P150 lb2 171629 (a07u28, dyn-slots host stage full-20261104-001304): every row <= 3%, except 4x2 multicast,
        which is noisy.
      - Validation passed in every arm on both systems. No hangs.
      - Per-axis T and follow-ups are for the user.
    - **Slot depth x injection threshold (user-approved; RING_JAM_FIX.md §9.3; Galaxy 171994, a08u02):** 14 configs,
      99 / 99 steps, validation 10 / 10 in all. Slots logged as asked.
      - 8-ring 4352 B reaches 254.3 (-32.5%, below main's 263.8) with a 16-slot receiver: C 16/51/16, D 8/59/16,
        E 4/63/16 at T 2 or 6; B 42/25/16 in pass b only (pass a 364).
      - Today's slots at T 16 also give 254.2.
      - F 8/43/32 (user rule with today's receiver) stays at 375 at T 2, and 304 at T 6.
      - 4-ring 255 and half ring 254 with the same configs.
      - A 4-slot worker costs ring 1-hop +75% (437).
      - The rule decision is for the user.
    - **Rule costing (RING_JAM_FIX.md §9.4):**
      - Galaxy 172155 (a08u02): A16 and C2 fast in both new passes (8-ring 4352 B 254.2-254.4; four of four passes over
        the two jobs). Torus T 8 vs T 2: +4.0% at 3072 B, +1.0% at 4352 B.
      - P150 on the LBs: a07u28's chip 6 eth (7,1) never trained in 172149, 172157 or 172175 (172175: the new poll
        with 3 resets), and also failed the allocator's 172043. The run moves to the a09u32 broker (commands sent).
      - lb2.sbatch's link check now polls 90 s per reset, with up to 2 more resets.
      - frs_glx.sbatch and frs_glx_torus.sbatch are deprecated (guarded).
      - P150 broker 673 (a09u32, dyn-slots host build):
        - A16 and C2 cost at most 0.6% on every P150 ring row; ring 1-hop scatter +0.0% / +0.2%.
        - Torus T 8 costs +2 to +14% on every 2x4 / 4x2 write row; scatter rows 0%.
        - Rule decision: the user's.
    - **Fraction or cap (RING_JAM_FIX.md §9.5; Galaxy 172449, P150 broker 710):**
      - Galaxy jam rows are fast with R - T = 14-16 at every R (C2, A16, G48).
      - R - T = 8 (C8) is fast but 3-7% slower and +7% at 8-ring 3072 B.
      - R - T >= 30 (A2, G2) is slow; G32 (T/R 0.5 at R 64, R - T 32) is mostly slow.
      - P150: C2, A16, G2, G32 and G48 within 0.9%; C8 +2 to +9% on write rows.
      - v4txa11t: T 2..62 with T < R, ring-only validation yamls, 3584 B max-packet rows for R 64.
      - Rule choice: the user's.
        - **Ring-jam rule decided (user, 2026-11-04):** T = max(2, R - 16) on deadlock-avoiding routers. The worker injects
  only while the far VC0 receiver holds at most 16 packets. R x T sweep §9.5: R-T of 14-16 fixes the jam at R16/32/64;
  T = R/2 fails at R64; R-T = 8 over-throttles. Independent of the slot-need sweeps (the rule takes R as input).
  Implemented by a fresh agent on branch aagarwal/fabric-router-mainline-bfc off 4db4f45790b: host named arg plus the
  router's injection threshold; images identical wherever T stays 2; P150 + Galaxy validation and jam rows.
  - **Built and gated (2026-11-04, RING_JAM_FIX.md §10):** commit ac86bf0f116 (not pushed). 49 configs: 5,345 router
    images identical to 4db4f45790b, 170 differ, all 32-slot rings (T 16), one immediate each, +0 bytes. static_assert
    2 <= T <= R (user).
  - **Device results (RING_JAM_FIX.md §10.3):**
    - Galaxy 172983 (bh-glx-120-c08u08): validation 10 / 10 on both trees. Jam rows -23% to -28% (8-ring 3584 B
      310 -> 228.5, 4352 B 374 -> 269.5, 4-ring 324 / 340 -> 255, half ring 332 -> 253.6). 3072 B control and 1-hop flat.
    - P150 758 / 759: validation all pass; ring rows -0.6% to +0.1%; torus rows unchanged.
  - **Next:** the user reviews the device results; then the coordinator pushes ac86bf0f116 (not before).
  - **Open:** ring + Tensix mux keeps the mux's 2-slot check, so the cap does not apply on mux rings (recorded, not
    changed).
  - **Cleanup after the push:**
    - worktrees /home/aagarwal/tt-metal-mainline-bfc (branch) and -bfcbase (4db4f45790b), builds and .cpmcache;
    - /data/aagarwal/regress/trees/bfc and /bfcbase, about 49k files each.
- **Facts review with the user (2026-11-03):** C2 (in-ring burst) and C1's room test as designed are ruled out
          by the measured state: the jam has 0-1 free far slots in 95-96% of contests, so a burst stops at once and
          room >= R never fires. Only the probe's action (hold injection) is shown to work.
        - **Injection-threshold sweep (user go, 2026-11-03; no predictions, per the user):** `variants/v4txa11t`, the
          bubble rule's injection threshold T in {2 (off), 3, 4, 5, 6} through one env switch (no host change),
          off = v4txa11 byte-identical. Galaxy jam rows + controls (frgap style) and P150 probe cost rows + XY torus
          (lb2 -n any), two passes, validation first. Ring agent. Per-axis T or anything further goes to the user.
          - **Galaxy 171672 (a03u08; RING_JAM_FIX §9.1):** 45/45 steps, validation 10/10 per T. 8-ring 3584 B
            306 -> T5 234 / T6 222-227 (main 261); 8-ring 4352 B only -3.6% at T6 (375 -> 361; main 264, probe 254);
            4-ring 4352 B +3.8% (324 -> 337); half ring 4352 B -0.7..-4.8%; torus -0.4%; 3072 B and 1-hop flat.
            P150 cost run lb2 171629 pending.
    - 562 confirms the trimmed MeshMulticast replay step in the same job: write 136.8 -> 139.2 (+2.4), scatter +1.1,
      Mesh3HopEW flat. 566 splits it at X4/X8. 561 (J7) is clean: main reproduces 476/477, A11 combined +1.2..1.8
      vs v4rxg3 cross-run (fits the parked 507 VC2 X0..X4 step).
    - Page rebuild items: skip the empty v8 multi-VC panel (A11 has default slots only) and fix its caption (:399);
      fix the stale dated captions (:316 status, :430 latency, :380 headroom, still v4rxv2i3 until v4txa11d).
    - 558: A11's first XY-torus validation passed 12/12 in both shapes and both passes, with the same axes and
      deadlock avoidance as main. Its rows vs v4rxg3 are cross-job: 4x2 column multicast -34..-39, 2x4 NS scatter
      +2.0. 559 is clean.
  - **Galaxy submitted:** 170094/170095 (the 200G sweep, passes a/b) and 170096 (breakdown points 2/4/5), on tree
    trees/v4txa11_1103 (identical to the variant on 4368 + 864 + 864 images). The ring agent watches them and ingests
    the results into bandwidth_glx_sweep.json and breakdown.json; no page rebuild.
  - **New: P150 RingMulticast write is +2.4..+3.4 cycles slower on v4txa11 than v4rxg3, at every size** (4352 B 147.5 ->
    150.8; tight passes; main flat; scatter/half/unidir flat). The sender-chain census never ran this row (ring1 is
    1-hop), so A11's "flat census" doesn't cover it. Cause open. A same-job bisect over v4rxg3 / v4txo8 / v4txc4 /
    v4txa11 is queued as 560. Waiting on the user's go:
    v4txa11v (3-VC gate probe) and v4txa11d (delay hooks for the headroom chart).
  **Regression testing for landing (user, 2026-11-03):**
  - **Main baselines (d2bd12f, baseline agent):**
    - a09u32: LB-L0-a 7/7 PASS. LB-L0-b PASS except B0-eth's 2 slow-dispatch cases (TT_FATAL tt_cluster.cpp:1569
      "connects to a remote mmio device"; main-only, to re-run once for flakes). LB-L1 B1-apc PASS (131).
    - Queued: B1-night and the rest of L1, L1P (2 passes), L2, LB-P telemetry-off fabric perf (612/613).
    - Galaxy plan OKed: G-L0-a + G-L0-b now, then G-P + G-L1, then G-L1P. G-L2-a/b held until the env report
      shows the model mounts. At most 2 of the agent's Galaxy jobs at once (the quad 171264 stays pending).
  - **TEST_PLAN.md DONE (559 lines; for user review).**
    - Two modes: the shared commits with main's router (byte-identical expected) and the router commits with the new
      router. CI exercises the new router only via a throwaway branch that makes it the default.
    - Perf: fabric goldens only for P150x4 / T3K / WH Galaxy (10-22% tolerance, both directions); model targets in
      models/model_targets.yaml (15%). The main-vs-new thresholds (1% fabric / 2% model) are proposals.
    - Gaps: exabox is BH-only (WH is CI-only); P300 has no coverage; BH Galaxy has no fabric BW golden; Nightly fabric
      fixtures don't run on BH boxes in CI; the TG perf scripts are in no CI list.
    - **Answered (user):**
      - P300 tests held for now.
      - bh_qb2 and bh_lb_single may be used if available, but aren't needed: QB2 (2x P300) is covered via CI dispatch
        (same commands per SKU; the fabric goldens live there), and bh_lb_single is overflow LB capacity only.
    - **Telemetry answered (user):** OFF for main-vs-new fabric perf. Production has it disabled by default, so CI
      should keep it off too. CI's perf goldens were recorded with it on, so propose re-goldening with it off to the
      team.
  - (was) **TEST_PLAN.md (the mainline-audit agent, read-only).** Relevant tests by level:
    - L0: fabric gtests / test_tt_fabric yamls / stability / ubench / infra;
    - L1: CCL functional and perf;
    - L2: production model prefill/decode.
    Each broken down by arch (BH/WH), system (P150, P300, LoudBox/T3K, Galaxy, multi-Galaxy) and kind
    (coverage/perf), with commands and CI workflow. Plus perf-baseline collection on main, and a batching proposal
    (manifest of Slurm-sized groups; runner against main vs the new tree).
  - **Multi-Galaxy harness READY (dry-validated): galaxy/MULTI_GALAXY_HARNESS.md and glxq_* scripts.**
    - The quads (4 Galaxies in a ring, 8 cables per pair; 32x4 RING x RING MGD) are pods 3/6/7/8/9 plus the SC16 racks
      a07-a10. Ring orders come from tt-cluster-config cabling guides; pod6 and pod8 are not in name order.
    - FSD: exabox-latest-main/factory_system_descriptor.textproto.
    - Launch via tt-run rank-binding mode, mpirun-ulfm under srun, NIC ens5f0np0, the tree on /data, per-rank caches
      and logs.
    - Preflight: glx_reset on all 4, then an ETH link count vs the FSD (312 per host).
    - **Finding:** test_tt_fabric can't take the FSD on the quad MGD ("A RELAXED mesh graph cannot be used with STRICT
      system health setup"; the first control plane is built STRICT before the yaml is read). The FSD is used for the
      preflight only by default.
    - Right now: pod6 drained (MINFRA-2421); pods 3/8/9 have no-time-limit jobs; pod7 and SC16 busy.
    - **Needs the user's go:** the first real quad job (smoke only, ~15-25 min, all 4 study allocations).
  - (was) **Multi-Galaxy Slurm harness (ring agent, prep plus dry runs only).**
    - Cabled quads: the 110-a pod partitions bh_sc36_6_110a_pod3/6/7/8/9; tt-sbatch "--topo quad / sc4".
    - FSD from tt-cluster-config (/data/scaleout_configs/tt-cluster-config-sources: bh_glx_exabox, exabox-latest-*);
      tt-run --factory-system-descriptor.
    - Deliverable: galaxy/MULTI_GALAXY_HARNESS.md, an sbatch template and a wrapper with preflight. A quad uses all
      4 study allocations, so no real run without a go.
  **Mainline decisions (user, 2026-11-03):**
  - The approach is agreed: a fresh branch on the latest origin/main, patched on top (MAINLINE_PLAN §7).
  - Kernel name: reuse main's (fabric_erisc_router.cpp); the new router replaces main's implementation.
  - WH bring-up: the user does it with another agent on a WH machine after the push. The notes are in
    consolidation/WH_BRINGUP.md.
  - Strategy (user go on 1-3, 2026-11-03):
    - Checkpoint 13 committed and pushed: e569c5deb20 on origin aagarwal/fabric-router-study, which also carries the
      unpushed checkpoint 12. Hang captures and .tgz are now gitignored.
    - Mainline worktree /home/aagarwal/tt-metal-mainline, branch aagarwal/fabric-router-mainline at d2bd12fcf1f
      (porting agent).
    - Main baseline worktree /home/aagarwal/tt-metal-main at the same commit (baseline agent).
    - WH may be broken on the branch for now (user).
  - **Parity (PARITY.md):** gaps besides WH:
    - VC1+VC2 on one router;
    - UDM on a torus;
    - UDM with multi-mesh;
    - mux with multi-mesh;
    - the streaming-profiler link-sync hook missing;
    - never-built configs 7-12 (DISABLE_INTERMESH_VC: production SC4 prefill; Z-port multi-mesh; pass-through;
      per-mesh tori; express E/W; >16-hop 1D).
    **User approved (2026-11-03):** lift the VC1+VC2 gate (v4txa11v validated), add the profiler hook (own commit),
    host TT_FATAL for 3-5 (known gaps), must-pass Galaxy/quad runs for 7-10, offline builds for 7-12 after commit 5.
    Relayed to the porting agent.
  - Galaxy scheduling (user): quad 171264 stays queued and single-Galaxy jobs run alongside (Galaxy k telemetry
    171298, P150 k 171279).
  **Mainline prep (user, 2026-11-03): audit DONE, consolidation/MAINLINE_PLAN.md (611 lines), for user review.**
  - Build the mainline branch fresh on origin/main from listed pieces (13 of the study's 33 changed tt_metal/ files,
    plus dataflow_api_raw.h, the router headers and the kernel); don't rebase the study branch. All 28 host
    TT_FABRIC_* switches stay behind.
  - Breakers:
    - ct:971 renames is_receiver_channel_forwarding_disabled, which main's router reads;
    - the 8-template-argument noc_fast_write_dw_inline_with_state breaks Quasar's 7-argument form.
  - Host selects v4 via TT_FABRIC_KERNEL_VERSION=2 in 4 places. Proposed: one router_kernel() owner, plus a host
    NUM_TRANSACTION_IDS arg, which drops the FABRIC_ERISC_ROUTER_V4 define. The kernel name is the user's call.
  - Stragglers: presets per buffer pair from its lowest VC (drops forwards_on_vc0_bufs and the VC == 0 assert), and
    one downstream_name/entry pair (fixes live_downstreams 0x1u). Also forward_data[vc] indexes a per-rx-channel host
    array by VC.
  - WH: the BH assert, sync_ready outside the BH #if, BH TXQ-busy semantics, WH's single TXQ getting stream-reg
    credits, and IRAM size (single-ERISC 16,680 B vs main 12,604 B).
  - origin/main drift to adapt to: decode_action express axis, handshake_info_t header, the connection enum,
    NocCmdBuf ids, multi_txq_enabled(), NOC_SCATTER_CHUNK_ENCODING_MASK, TT_METAL_FABRIC_DISABLE_INTERMESH_VC.
  - Main-side findings to raise separately: the flush without DA waits only on rx ch0's trids; the UDM relay open
    zeroes L1 address 0 (per a v4 comment); postcode suppression on VC1.
  - Cleanup: untracked tt_metal/fabric/erisc_datamover_builder.cpp.orig.
  - **Audit (a new read-only agent; deliverable consolidation/MAINLINE_PLAN.md):**
    - file inventory against origin/main (merge-base 6d8a103) with dispositions;
    - dead and study-only code in the shared headers (e.g. fabric_edm_packet_transmission.hpp);
    - the host-side shape (named/per-VC args, env switches);
    - placement (Ethernet TXQ regs -> Ethernet API, NoC helpers -> NoC API for every arch, stream regs);
    - stragglers (forwards_on_vc0_bufs and the VC0 preset assert; live_downstreams' 0x1u mixing entry- and
      direction-indexed masks);
    - WH gaps (the router asserts Blackhole);
    - a commit/PR sequence, including a minimal first push for BH and a WH trial.
  - **Ring-jam fix design (ring agent, design only, on v4txa11):** why the probe cost +4..7 with the gate open (torus)
    and over-throttled lighter patterns; candidates (jam trigger on far-credit refusal, mask-only once-per-pass gate,
    K from the host); cost estimates and an isolating A/B. Goes to the user before any build.
  **A11 refresh rebuild (2026-11-03), a private review draft:** https://claude.ai/artifact/WLSk9ZHX6rMJiXqu9mLjaK
  (page/router_report_review.html). **The user OKed it; published to the shared page as v23 (2026-11-03).**
  - data/ is rewritten by the agents' builders; the backup is in scratchpad/data_before_refresh/.
  - **Bug fixed:** build_page.classify gave a multicast test's write and scatter rows one key, so scatter won the
    dedupe. Every multicast row on the published page (v21/v22), including the KPI "2D mesh multicast 4352 B" and the
    intro's multicast peaks, showed the scatter-write rate. Multicast rows now carry ", scatter write" / ", fused" in
    the label. KPI: 173.1 (72% of line) -> 141.7 (88.5%); main 215.1 -> 182.0.
  - Trailing rows now come from the chart's own rows (>= 0.5% and >= 1 cycle). gaps.json's Galaxy rows were still
    v4trm (347.7).
  - Template: the status line, headroom/latency dates, multi-VC caption (no v8 panel, finish < 1%, check < 0.5%),
    the empty-panel guard, the latency keepalive sentence guard, the 3-VC fit caption (1,376 B), the breakdown notes
    for p2/p4 (main E0 dropped) and p5, and the to-measure list trimmed.
  - H-C0b trimmed bound A/B queued as 577.
  **v22 (2026-11-03): Ring jam tab** (user). Data: data/ring_jam.json from scripts/build_ringjam.py (Galaxy 169482/3,
  169903, 169871, 170016; lb2 170034).
  - Sections: what we saw (full-ring row, main vs new by size), the receiver-delay sweep chart + write-skip table,
    the mechanism as a hypothesis (the two-buffer hop, the CBS paper), and the throttle on Galaxy and P150.
  - The Gaps sentence now points to the tab. It was not rendered locally (no JS runtime on the host); the user
    checks it in the browser.
  **v21 (2026-11-03): tabs** (user):
  - tabs: Overview / Bandwidth (incl. keepalive) / Multi-VC / Latency / Cycle breakdown / Code and stack / What it
    enables / Gaps and next;
  - sticky tab bar; URL hash and localStorage (try/catch) remember the tab; arrow keys move between tabs;
  - **Cycle breakdown tab:** stacked bars per point x router x ERISC from data/breakdown.json. Category colors are a
    validated palette (polls gray; aqua/orange/violet/green/magenta/yellow), with a table view and per-point notes
    (point 5 caveat);
  - the section numbers were dropped. The page numbers refresh when the new data lands.
  **v19 (2026-11-03):**
  - Section 5 rewritten on the user's points: native fan-in first (mux cores freed, a hop removed, the VC2 line with
    2D headers or up to three 1D line VCs; stack is not the limit, code space is); slot allocation without the
    power-of-two constraint (main 2-6 cycles per non-pow2 channel, the new router's sender channels flat; the model
    within 0.6%); codegen stable and explainable (stated placement, guard and headroom, identity across ~2,900
    configs, no compiler tricks).
  - The 200G ring and XY torus (Galaxy, both axes RING) are in. The intro is computed per topology: on the XY torus
    at 4352 B main is at 94% and the new router at 99%. Data bundle
  fabric_exp/report/ (data/*.json, DATA_INVENTORY.md, CLAIMS.md); page built by report/build_page.py from
  report/page_template.html (edit the template, rerun, republish the same file); artifact
  https://claude.ai/artifact/36ZXA694Bsg6WkFyAKKd8D (draft v1, bandwidth chart A). Before the final version: the
  same-job reruns on the final lead in DATA_INVENTORY's gap list (P150 census, Galaxy, latency, max-feature build).
  The main tree's frs_glx*.sbatch have no 110-a node guard yet (job 167648 ran on 120-c): add it before main-tree
  Galaxy runs.

- **N/S fix 1, partial presets: design note** (user, 2026-11-02; receiver agent ae61797a992a87495):
  V4_NS_PRESETS.md. Data on buffer 2 with invariant fields preset, the credit on its own preset buffer 1, through the
  stateful NoC APIs. Shapes (a) multi-downstream routers only vs (b) every forward: draft (b), take it if its E/W cost
  is at most 1-2 cycles/packet, else **bring the numbers to the user** (readability first). **Note done**
  (multi-downstream forwards -26 instructions, -5 register writes; drafts v4rxnsa / v4rxnsb / v4rxnsbml). User
  decisions 2026-11-02: bundle the VC change and posted credits; draft the two (b) trims (readable only); accept the
  torus spill for the A/B; UDM relay stays stateless; the additive NoC API is fine (upstream review). Running:
  correctness, proxy A/B, decision rows, VC1/torus, validation, then Galaxy N/S rows (fresh base tree from the Galaxy
  report agent). Job 481 queued (2026-11-02 22:38): validation of v4rxnsb and v4rxnsa, the multi-downstream proxy A/B
  v4rxg3ml vs v4rxnsbml (two passes), PC samples of both on 2D hop3.
  **Milestone 1 (job 481):**
  - correctness: mesh sanity 96/96 and the inline/flushed-atomic gtests (1D, 2D) pass on both drafts;
  - proxy (multi-downstream) gains, 1024 / 4352 B: 3-hop E/W -33.5 / -31.0, turn -26.3 / -20.8, multicast
    -39.1 / -43.4, row merge -11.2 / -10.7; 1-hop rows flat;
  - **(b) costs on 1D single-downstream at 4352 B:** Lin3HopEW +6.6 (124.5 -> 131.0, off the wire bound), LinRowMerge
    +2.9; 1024 B +0.7 / 0. Over the agreed 1-2 cycle bound for (b), so the numbers went to the user.
  - Agent's hypothesis (not tested): the joint ready test serializes the data write behind a busy credit buffer
    under 4352 B NoC backpressure.
  - Proposed isolating probe: v4rxnsbr (the lead's ready-test pattern on (b)), plus a serial-poll arm. Not built;
    designs going into V4_NS_PRESETS.md for the user. Jobs 493/494 show whether lin3 reproduces.
  - **User: "lets try and see if the 1D cost can be avoided"** (2026-11-02). Asked whether the forward strategy can be
    a compile-time choice from one struct; designed in §9 as ForwardCommands<VC> = Stateless / FullPreset /
    PartialPreset: ~570-590 lines, three named facts replacing nine predicates, design only.
  - Jobs 493/494, decision rows at 4352 B:
    - lin3: lead 124.4, (a) 124.4, (b) 130.3 (+5.9, reproduces);
    - every other row flat; (a) within ±0.1 everywhere;
    - hop3 PC samples on (b)'s forwarding E1: the joint test's branch 7.4 cycles, each wait loop 1.9. Fits the
      hypothesis but doesn't test it (hop3 is E0-bound).
  - On 1D, (b) differs from the lead only in the per-forward node writes (data RET_ADDR_COORDINATE, credit
    TARG_ADDR_LO/COORDINATE + the coordinate load) and the joint ready test; the rest is codegen (merged counters).
  - Probes built: v4rxnsbr / v4rxnsbs.
    - Executed instructions per forward (1D/E-W/N-S): nsbr 29/30/29, nsbs 28/28/28 ((b) 24/25/24, lead 23/24).
    - Headroom: both pass every production family, bh_torus 12/12 included. The predicted spills were WRONG (marked).
  - Queued 2026-11-03 00:07 after the sender round: 502/503 probe A/B (nfra/nfrb), 504/505 nfka(+lin3 PCs)/nfkb.
    The proxy job with the ml probe copies follows when built.
  - **Probe results (502/508, 510/511, 509):**
    - neither wait form gets (b) within 1-2 cycles on 1D: nsbr 1D 3-hop 4352 +2.2 but 1024 +5.8; nsbs +7.9/+13.9;
      the joint (b) +6.5/+0.7;
    - the joint ready test causes most of (b)'s 4352-B cost (nsb -> nsbr -4.3);
    - multi-downstream proxy: the joint test is best (3-hop -33.5/-31.0, turn -26.4/-20.8, multicast -38.9/-43.1);
    - VC1 gains: 2-hop 239.2 -> 231.7 (16 B), vc1i mean 336.8 -> 331.6;
    - (a) is 0 on single-downstream rows (byte-identical).
  - Agent recommends ForwardCommands (§9): (a)'s behaviour, three forms behind one interface, ~570-590 lines.
    **User approved building ForwardCommands (2026-11-03: "yes, go ahead and build it").**
    - Offline first: single-downstream images byte-identical to v4rxg3, multi-downstream identical to v4rxnsa,
      the VC1-can't-share images classified, guard/headroom/size.
    - Then a report, then validation + decision rows on the device with the main session's OK.
    - Galaxy N/S rows (lead vs v4rxnsa, + main_fw) proceed within the Galaxy cap (1 allocation while the report
      agent holds 3), excluding a02u14.
    - **Built: v4rxnsf.** Byte-identical to v4rxnsa on all 2,863 images at -O3 and -Os; single-downstream images
      identical to v4rxg3. router_downstream.hpp is 618 lines (estimate was 570-590; lead 532, (a) 632, (b) 503).
      can_share_vc0_destination<VC>() is a template; credit_register_varies is private to FullPresetForwards. No
      VC1-can't-share config is in coverage.
    - Validation: 513 (variant_val), 514 (inline/atomic + UDM). Decision rows skipped: the images are identical to
      (a), already measured. Galaxy N/S job 169615 pending (2 KiB and 4352 B slots, g3 / nsa / main_fw).
    - The 618 lines: selection facts 58 (incl. forward_cmd_bufs_consistent 15, can_share_vc0_destination 13),
      struct framing 10-12 per preset form. Trim 1 approved: PartialPresetForwards presets VC0's pair only, plus a
      static_assert that VC1 is absent or on VC0's buffers (~-10 lines), as a new variant (v4rxnsf2) that must be
      byte-identical to nsf. If it is, nsf's validation covers it. Readable floor ~605-608.
    - **Validated (513/514):** 13/13 (both ERISC modes incl. trimming), inline/flushed atomic 1D/2D, UDM 8/8 + 68/68.
    - **v4rxnsf2** (trim 1) is byte-identical to nsf on all images: 621 lines (the shared template helper ate the
      trim). Kept for readability: one preset, stated assumption. This is the receiver-side build for the sender
      consolidation's stage 0 merge.
    - **DONE (2026-11-03). Galaxy 169615** (a10u14, both passes, no failures).
      - 2 KiB slots, 1024 B, lead -> fix: 3-hop N/S 172.9 -> 136.8, 7-hop N/S 173.5 -> 137.4, corner 173.6 -> 140.2,
        turn 170.9 -> 140.0, column merge 159.1 -> 133.2; E/W control 140.3 unchanged; main_fw ~227. **The N/S floor
        is gone at 2 KiB slots.**
      - 4352 B slots: 3-hop N/S 182.7 -> 152.6, turn 178.6 -> 145.1; 7-hop / corner 186 -> 167, window-bound at 16
        forwarding slots (a slot question). All 4352-B payload rows on the wire.
      - Open:
        - Galaxy bound side after the fix (no delay hooks in these trees);
        - the additive NoC API (3 functions) for upstream review;
        - the VC1-can't-share change has no device test;
        - FLUSH_DEFER step B now concerns atomics only (forward doorbells left buffer 3);
        - census FWD_MixCmd is bimodal (172/212) for any router: a noise source.
      - Note: V4_NS_PRESETS.md (§8.9 Galaxy, §9.1-9.3). Results: results/glx169615/.

- **Report: cycle breakdown, main vs new** (user, 2026-11-02; breakdown agent): five points (2D 1-hop write 4352 B
  400G; 2D 1-hop write 1024 B 200G; 2D multicast 4352 B 400G; Galaxy N/S 7 hops; Galaxy full-ring multicast 4352 B),
  both routers, both ERISCs, PC samples mapped to shared categories per source line. Phase 1 offline (taxonomy,
  maps, validation, plan, provisional report/data/breakdown.json, report/BREAKDOWN_PLAN.md); phase 2 sampling after
  the sender pass lands. Includes the user's pending costliest-rows profile (new router only). **Phase 1 done**
  (report/BREAKDOWN_PLAN.md: 8 categories, maps for both routers, 100% mapped on existing samples). User decisions
  2026-11-02: drop "waiting" and mark the new router's A/B spare time instead; multi-router tests show the router that
  sets the rate (A/B) else the busiest, named; point 4 at default slots. P150 phase 2 (bd_p150_runs.sh, ~15 min) on
  the final sender build; Galaxy points 2/4/5 by the Galaxy agent.

- **Report: 400G size sweep** (user, 2026-11-02): main_fw vs v4rxg3, same job, 2D / 1D / ring / torus two-ERISC suites
  at 256-4352 B (report/sweep/*_sizes.yaml, consolidation/report_size_runs.sh), jobs 465-468 (pass a main first, pass
  b reversed); data via report/scripts/build_size_sweep.py -> report/data/bandwidth_sweep.json (the page prefers these
  same-job rows). **Done**: 16/16 runs clean, 260 pairs, median pass spread 0.01% (one row 3.3%), the new router
  slower than main on none. Page v6. One ERISC, mux and VC1 not yet swept.
- **Report: multi-VC throughput, VC0 + VC2 on one link** (user, 2026-11-02): consolidation/multivc_runs.sh, pcs3 rows
  vc02ew / vc02bew / vc2ew at 256-4352 B + VC2-on ew4352, main_fw vs v4rxg3 same job, slots default (each router's
  own) and v8 (17,10,16,8,10 for both), jobs 469-472 (a: main first; b reversed). First main-vs-new VC2 runs ever.
  VC0 + VC1 (two meshes) to follow with the two-mesh harness. Page: neighbor exchange split from the 1D line (v7).
  **Done** (both passes, max spread 0.3%): own slots (VC2 2/4 on both) 4352 B main 25.7 vs new 41.9 GB/s; same slots
  (VC2 8/10) main 30.6 vs new 46.7 of 46.9 GB/s; main splits VC0/VC2 equally, the new router ~2:1. Page v9.
  **Check of the two-run sum** (user 2026-11-02: measure the combined rate directly):
  - from the existing runs: in vc02ew, VC2's measured finish time matches the time predicted from vc02bew's contended
    rate plus vc2ew's alone rate within 0.26% in 22 of 24 cases;
  - main at v8/4352 B already finishes both VCs within 0.4% in one run: 30.6 GB/s measured = the sum;
  - direct runs: matched counts (consolidation/mk_mvc_matched.py -> pcs3/vc02m/,
    consolidation/multivc_matched_runs.sh), jobs 476 (a) / 477 (b), labels mvm<PASS>_*. **Done**:
    - the one-run combined rate matches the two-run sum within 0.53% on every row (mean of passes), within 0.12% on
      10 of 12;
    - finish gaps are 0.0-0.35%, except one pass at 1.17% (new router, own slots, 1024 B), which reads 1.1% low;
    - multivc.json now has combined_matched / vc0_share / finish_gap_pct; the page (v10) shows the one-run numbers
      with the two-run check, and the "estimate" wording is gone.
  **All three VCs (new router only) + VC0 + VC1 main vs new, 400G only** (user 2026-11-02: "lets just focus on the
  400G numbers for now"; the 200G Galaxy allvc rerun is dropped). Multi-VC agent:
  - probe v4rxg3v = v4rxg3 with `!has_vc1` dropped from `vc2_config_supported` (the v4rxc7v change); VC2 runs in
    neighbor-exchange mode, not the line build;
  - offline: other images identical, VC1+VC2 size and margin at -O3/-Os;
  - vc1vc2_runs.sh validation;
  - P150 two-mesh benchmark: one mesh-1 link with VC0 local + VC1 forwarded + VC2 neighbor, each VC alone, pairs and
    all three, at 256/1024/4352 B;
  - VC0 + VC1 main_fw vs v4rxg3, same job, two passes;
  - output report/data/multivc_3vc.json + multivc_vc0vc1.json; then page rows.
  - Queued 2026-11-02:
    - job 478 (vc1vc2_runs.sh g v4rxg3v validation; running since 22:52);
    - 479 (gate + three-VC pass a, default and 8/10 slots);
    - 480 (gate + VC0 + VC1 main vs new pass a).
  - Offline: v4rxg3v is byte-identical to v4rxg3 on all 2,863 shared images (same builder). The 131 VC1+VC2 images
    build: -O3 25,932 B (692 B before runtime args), -Os 24,848 B.
  - Pass b (two jobs) is OK to queue after pass a. Matched three-VC rows (both slot settings, twice per job) and a
    matched VC0+VC1 row (main vs new) are approved.
  - 478 passed everything. 479 (A pass a): A_d clean.
    - A_v8 was refused: "Program size (26640) too large for kernel config buffer (26624)", 16 B over at -O3. E1
      grows 272 B with VC2 8/10 slots.
    - Margin at default slots, -O3: 432 B runtime args + 25,952 B text = 26,384 of 26,624 B, so 240 B left.
    - **User decision (2026-11-02): no -Os.**
      - Three-VC rows at -O3 with default slots only, labeled "default allocation" on the page: A_d plus the matched
        three-VC row. A_v8 and all -Os runs are dropped.
      - Which other allocations fit, and how, is checked in the code-cleanup stage.
      - B is unchanged: main VC1 at -Os (main as shipped), new router at -O3.
    - Job 480 ran nothing (a GROUPS bash-variable bug, fixed). B pass a requeued as 491: clean (main 96/96 compiles
      at -Os, new at -O3; every competitor outlasted its measured flow). A_d pass b at -O3 + matched three-VC row: 492.
      B pass b + matched VC0+VC1 row: 495.
  - **Done (agent's final report 2026-11-02):** data report/data/multivc_3vc.json (82) and multivc_vc0vc1.json (78),
    built by report/scripts/build_multivc3.py. Page v12 adds the "VC0 and VC1 on one link" and "All three VCs" panels;
    VC colors are now VC0 aqua, VC1 violet, VC2 orange (validated; blue stays the new router's).
    - VC0 + VC1, 4352 B (one run): main 21.27 GB/s (45% of line), new 39.99 (85%); 1024 B 5.11 vs 9.54; 256 B 1.29
      vs 2.39.
    - Three VCs (new only, default allocation, -O3), 4352 B: 35.96 GB/s (77%), against VC0 alone on the same build
      35.16. Shares VC0 56-58%, VC1 28-30%, VC2 14%.
    - Open observations (hypotheses):
      - extra VCs add little total next to VC0 (a per-packet router cost?);
      - VC0 alone is 167 cycles on the VC1+VC2 build vs 142 on VC1-only. That is cross-config (slots differ too);
        it fits the idle-VC2 E0 cost (V4_RECEIVER_SHAPE §29).
    - The stored cov_m v4rxg3_O3/_Os predate the ACK_CMD_BUF_ID injection, and size.json was built from them. The
      difference is init-only (+4 B on some VC2 cases); refresh with the final build's size pass.
    - Code space for VC1 + VC2 at -O3 is thin: feeds the "cold init/control paths runtime-parameterized" stage. Add
      there: fit the 8/10 VC2 allocation with VC1 + VC2 at -O3 (16 B over today).
- **Report: 200G Galaxy runs** (user, 2026-11-02; Galaxy report agent): fresh /data base tree + v4rxg3 tree, the node
  guard in the main-tree frs_glx*.sbatch, mesh / linear / ring at 256-4352 B + full-ring multicast + 7-hop N/S, main_fw
  vs main (ka 1000) vs v4rxg3, same job, two passes; then PC sampling for breakdown points 2, 4, 5 (both routers). 110-a
  only, at most 4 concurrent study Galaxy allocations (user, 2026-11-02). Data: report/data/bandwidth_glx_sweep.json.
  Smoke 475 passed on both trees (sampler OK). Galaxy jobs 169263 (pass a), 169264 (pass b) and 169265 (breakdown
  sampling) were pending on bh_sc36_6 at 23:00; ~60 min each once started (16-25 min of that is the tree copy).
  23:35: pass a (169263, a10u02) failed fast before any test. glx_check_2d.yaml was missing from the /data trees, and
  the dry run had used live stubs. Fix: fabric_exp/galaxy synced into both trees; the sbatch now preflights every
  step's yaml and tool before the copy and the reset. Pass a resubmitted as 169343. Pass b 169264 running on a10u02
  since 23:35 (its copy came after the fix). Sampling 169265 pending.
  Then the check row itself failed: glx_check_2d.yaml (explicit senders, no flow control) gave DATA_MISMATCH at the
  7-hop N/S receiver on both main (169265, a09u08) and v4rxg3 (169264, a10u02), so the row is at fault. Agent's
  hypothesis: without flow control the sender laps the receiver's ring. Logs are in /data/.../jobs/169264|169265/
  results/*_check.log. New check row: one parametrization of test_fabric_mesh_sanity_common.yaml (passed 96/96 on a
  110-a Galaxy in job 168925). Resubmitted: pass a 169343, pass b 169403, sampling 169404 (all pending).
  00:45:
  - Pass a (169343, a02u14): tree checks OK, mesh sanity 96/96 on main and v4rxg3; mesh + linear done for all
    three routers.
  - The ring step hit a TT_FATAL: "Graph specified in MGD could not fit in the discovered physical topology". The
    agent's hypothesis: links down on a02u14, the node that also showed one link per chip pair. Pass b 169403 on
    a02u14 runs mesh + linear only.
  - Ring/torus moved to 169482 (a) / 169483 (b); sampling resubmitted as 169481 (mesh points first); all with
    --exclude=a02u14. 4 study allocations.
  - AICLK settled at 1,337-1,343 MHz on some chips (UMD warning). The cycles are device counts, but the page's GB/s
    and wire model assume 1.35 GHz (up to ~1% off on those chips). Agent recording per-chip AICLK; page treatment
    to decide.
  - **Mesh + linear done for both passes** (169343 a, 169403 b, both a02u14): report/data/bandwidth_glx_sweep.json,
    195 records; passes agree within 0.3 cycles.
    - 2D 1-hop E/W 1024 B: main 224.1, v4rxg3 123.1. 7-hop N/S 1024 B: 222.8 / 186.1.
    - At 3072 B v4rxg3 is on the wire (179.6) and main is at 223.7. At 4352 B both are wire-bound (main 251.1, v4rxg3
      252.1, main at keepalive 1000 also 252.1).
    - AICLK: UMD names no chips for the 1,337/1,343 readings; the profiler reads 1,350 on sender devices 0/1/31. No
      rescale.
    - Page v18 prefers these same-job rows at 200G. The ring rows are still from 10-29 until 169482/169483 land;
      sampling is 169481.
  - 01:40: sampling 169481 on a10u08 works.
    - Main point 2 (E/W 1-hop 1024 B): 560 targets, 2.26M samples in 25 s, 0 read errors.
    - The sampled rate 224.1 equals the unsampled 224.1 (no perturbation).
    - Ring/torus pass a 169482 started on a10u20 (main's check passed); pass b 169483 pending.
  - **XY torus on the LoudBoxes, both 2x4 and 4x2** (user, 2026-11-03). Folded into the second-LB agent (user: no
    extra agent duplicating setup; the separate torus agent was stopped before it wrote anything). Plan:
    - a study 4x2 descriptor;
    - per-shape validation + benchmark yamls in fabric_exp/torus/, with wrap-crossing rows;
    - mock-vet both routers: no fallback, the 4-long axis RING, DA on all four directions;
    - consolidation/torus_xy_runs.sh and report/scripts/build_torus_xy.py.
    Device runs go to a07u28 via the second-LB harness after 510/511.
  Base tree for the receiver agent: base_1102. Unblocking (user, 2026-11-02):
  - submit right after 475;
  - later tree checks run on the Galaxy at job start, not on the P150 queue;
  - pass a, pass b and the sampling run as parallel allocations (cap 4, 110-a only), each pass same-job, its node
    recorded.

  **DONE (agent's final report, 2026-11-03):**
  - Mesh/linear (a02u14), ring/torus (a10u20 / a10u02) and sampling (169481, a10u08) all finished. Data:
    report/data/bandwidth_glx_sweep.json, 315 records. New fields pass_jobs / pass_nodes / topology_realized /
    aiclk_passes are not yet in SCHEMA.md / DATA_INVENTORY / finalize.py FILES (housekeeping).
  - **Full-ring multicast 4352 B regression persists:** v4rxg3 372.2 vs main 263.8 (0.71x); 256-3072 B are
    1.41-1.43x in the new router's favour. Cause not established. **Investigation agent started (user, 2026-11-03):**
    offline only (no device or Slurm jobs), using the point-5 PC samples, the size sweep and the P150 analogues. It
    writes consolidation/FULLRING_MCAST_GAP.md: ranked hypotheses, each with its cheapest isolating A/B and a
    prediction.
    **Done (note + fr_gap_samples.py).** Top hypotheses:
    1. The 8-chip rings are limited by the per-hop credit loop (first-level ack after wire time; forward + local write
       2 x 4400 B on one NoC before completion), and v4's loop is longer. A/B: 8-ring-only vs 4-ring-only yamls, then
       one-side delays.
    2. A single-slot completion is checked once per pass (router_receiver.hpp:581). The earlier bounded wait (v4wt vs
       v4uf: 350 -> 310) was never ported. A/B: a one-line variant, predicted 372 -> 327-342.
    3. A size-dependent multicast-hop cost: P150 multicast rows step at 4352 B on v4 only. A/B: a P150 size sweep at
       3584-4352 B, then skip the local write or move it to NoC0.
    Also found:
    - the 4-chip rings show a TX-queue fairness defect on both routers (alternate chips spin ~181 / 118 per packet);
    - **breakdown caveat:** in the v4 pass-b sample set traffic starts ~17.5 s into the 25 s window (window it), and
      the busiest-router pick lands on a 4-chip-ring router, not the 8-ring routers that set the row.
    Variants (A/Bs 2, 4, 7) need the user's design OK.
    **No-build A/B results (Galaxy 169776/169777/169808, P150 lb2 169780):**
    - the 8-chip rings set the row (8-ring-only: v4 374.1 vs main 263.7);
    - the cliff grows 192 -> 295 -> 358 -> 365 -> 368 (3072..4352 B) with no frame step; main flat;
    - slots, keepalive and the 4-ring starvation are ruled out; P150 multicast grows smoothly with size;
    - P1 samples: the extra cycles at 4352 B are all waiting (admission +42, header first use +47, polling).
    - Torus: main 290.5 / 291.5, v4 246.4 / 270.2. The note's and the main session's "no cliff on the torus /
      specific to 1D ring routers" was **WRONG as stated** (user caught it): v4 steps +23.8 on the torus while main
      is flat. 2D is not ruled out; H2 stays open. Corrected in FULLRING_MCAST_GAP.md.
    - **User go (2026-11-03):** the full-ring agent was resumed for:
      1. the Linear saturated-multicast row (no build);
      2. A/B 7 v4rxg3bd one-side delays on the 1D 8-ring and torus rows at 3072/4352 B;
      3. A/B 4 v4rxg3sl local-write skip on the 1D 8-ring, torus and P150 multicast rows.
      A/B 2 (bounded wait, exit on proof or new packet) is held; its design comes back first if the results point at
      completion timing.
    - **Round two done (FULLRING_MCAST_GAP.md §8):**
      - The Linear row runs at the wire, so the 1D multicast hop alone is not the cliff.
      - A/B 7: the 8-ring at 4352 B gets *faster* with an E1 delay (−44.2), so H1b (credit-loop / latency bound) is
        WRONG as stated. The torus at 3072 B is E1-bound.
      - A/B 4: on P150 the local write is the size step (confirmed). On Galaxy, skipping it pushes rows into a slow mode
        (8-ring 3072 B +86.6, torus 4352 B +72.2).
      - The main session's chat idea of "busy passes credit one slot per pass while writes are still in flight" is
        WRONG as stated: a faster write (skipped) made the rows slower.
      - New hypothesis H9: ring/torus routers enter a slow mode at 1.4-1.6x the wire when v4's E1 serves faster than
        the link. Candidates: (a) extra credit frames per packet, (b) completion one pass late, (c) faster
        first-level acks shifting the bubble balance.
      - The main session OKed the no-build RX-delay sweep (v4rxg3bd 0-96 on 8-ring 3584/4352 B and torus 4352 B, plus
        an optional 3072 B control) under the user's standing A/B approval. Any design comes back before code.
      - **Sweep done (Galaxy 169903, §8.9-8.12): H9 confirmed as a regime.** The slow mode holds exactly while E1
        serves faster than the link. The 8-ring at 4352 B runs 373.5 at RX delay 0 and 256.3 at 48 (main 263.8);
        out of the mode, every row sits on its E1-bound line. (a) and (b) are weak by the code read; the
        mechanism ((c) ack timing vs E0 arbitration) is open. No per-sender spread exists in the data.
      - **Pending user decision:** the agent's three separation tests:
        1. a host TXQ-counter snapshot (no build; needs a safety trial);
        2. first-level ack at retire;
        3. forwarding priority over the worker.
        Nothing is built or queued.
      - **Superseded by the main session's read (hypothesis):** a bubble-flow-control jam.
        - The bubble check reads only E1's receiver ring. When E1 outruns the link, the queue moves to the
          forwarding channel, the ring fills, and it moves bubble by bubble.
        - Strict forwarding priority and the bubble-8 arm were withdrawn (both starve the worker; the row reports the
          slowest sender).
        - Proposed instead: the worker waits while the local forwarding channel holds >= T packets (T = 4, 12).
          Awaiting the user.
        - **User go (2026-11-03), final design T + K:**
          - An injection channel (the worker, plus turns on a torus) sends when it has >= 2 far free slots and either
            the straight forwarding channel's backlog is < T or K in-ring sends have passed since the last
            injection.
          - The user's point: traffic can leave the ring early, so a fixed ratio alone isn't general. T adapts to
            the backlog; K is the floor, sized for the longest trips (N-2).
          - Arms: A T1/K2, B T1/K6, C T4/K6. The full-ring agent builds it as a study switch off v4rxg3 (identical
            with the switch off) and runs Galaxy plus lb2 P150, including short-trip rows and both XY torus layouts.
        - **Galaxy results (170016, a06u02, both passes clean; frg_tables.py glx):**
          - Holding back injection removes the slow mode, confirmed by this A/B. K behaves as the fair-share
            argument predicted.
          - 8-ring 4352 B: 371.4 -> A 334.7 / B 254.4 / C 254.4 (main 263.8). 3584 B: 301.3 -> 247.7 / 225.7 /
            228.7. 3072 B: flat.
          - 4-ring 4352 B: 339.3 -> 255.0 in every arm (K=2 is the 4-ring's fair share).
          - Half-ring multicast 4352 B: 332.2 -> 274.9 / 253.7 / 253.7. It was slow at base too, so the
            "short-trip rows unchanged" prediction was WRONG for this row. 1-hop unicast: flat.
          - Torus 4352 B: 269.9 -> 262.8 / 259.6 / 271.5 (C +1.6). 3072 B: flat.
          - The P150 rows (a07u28 170034) are pending.
        - **PAUSED (user, 2026-11-03).** The full-ring agent was stopped with no jobs running; its context is
          resumable. Open threads when resumed:
          - a design for a jam-aware trigger: engage only when the straight channel is blocked for lack of far credits,
            not on any backlog. First, a one-side delay A/B on P150 ring multicast to learn its bound;
          - a leaner torus gate (the +4..7 on 1-hop torus writes with the gate open);
          - on hold: the torus T sweep (1/2/4 at K 6) and a 2-hop ring unicast row (B vs C).
          The variant is variants/v4rxg3it; the Galaxy tree is trees/v4rxg3it_1102. Notes: FULLRING_MCAST_GAP.md §8.14-8.19;
          page tab "Ring jam".
        - **Open, possible v4rxg3 P150 defect: FrgValFullRingMcast hangs on the P150 with the switch off.**
          - Row: Ring, 1 link, enable_flow_control, 4096 B, 2000 packets, write + fused atomic inc, full-ring
            multicast.
          - It was SIGKILLed about 20 s into "Waiting for programs", on a08u28 (170018) and on a07u28 (170023).
            On a08u28, PCI device 2 vanished afterwards. The same yaml passes on Galaxy.
          - **Main hangs too (a09u32 broker 537, user's main-first call):** "Waiting for programs" for over 150 s,
            then killed by run_cap and hang-captured (results/fcval_main_fw_hangcap). The decode found no router
            configs in main's cache. v4rxg3 was not run.
            - So it is not a v4 regression: either the test (flow control plus full-ring multicast on the P150
              ring) or a defect both routers share. Unresolved.
            - The device was reset afterwards with tt_device_reset; health OK.
            - The P150 probe validation was switched to the proven rows. Galaxy runs the same yaml fine.
          - **Built and running (2026-11-03):** variant v4rxg3it (TT_FABRIC_EXP_DRISC_DELAY = 1<<16 | T<<8 | K).
            - With the switch off it is byte-identical to v4rxg3 on every image (Galaxy and P150, both torus shapes;
              §8.15).
            - E0 .text with it on: ring +32/+32/+36 B, torus +88/+96/+108 B (A/B/C); E1 unchanged.
            - Predictions are in §8.14. Galaxy job 170016 (a06u02); P150 lb2 job 170034 (a07u28, after 170018 and
              170023 were killed on a flow-control row, §8.16). Validation ran first in both.
            - **Done (§8.17-8.19):** Galaxy B and C remove the slow mode (8-ring 4352 B 371.4 -> 254.4; main 263.8).
              On P150 the throttle costs up to +22 on multicast and 1-11 on every torus unicast write row. Not a
              generic fix as built; a narrower trigger is a design question for the user. Nothing built or queued.
            - **Fix design (2026-11-03): consolidation/RING_JAM_FIX.md**, on v4txa11, awaiting the user's review.
              **User (2026-11-03):** C2 is primary and C1 the comparison arm; K is a cap; the host K arg is OK.
              Telemetry runs first: `variants/v4txa11j` is built (off byte-identical; §8.1). Galaxy and P150
              commands went to the coordinator. If the room test does not separate, stop and report.
              - Recommended: C2, the in-ring burst. Fallback: C1, the contest gate.
              - First A/B: a telemetry-only trigger check.
              - Decisions for the user: C2's pass-length bound, a host ring-share argument, and running the trigger
                check first.
          - **Open, separate (§8.16):** P150 FrgValFullRingMcast (Ring, flow control, 4096 B, full-ring mcast) hangs
            with main too (a09u32 broker 537). LoudBoxes SIGKILL it ~20 s in. Do not run flow-control full-ring rows
            on a LoudBox; the main_fw vs v4rxg3 A/B is on hold with the user.
        - A known behavior: the CBS paper (IPDPS 2011) §II.C / §V.C, post-saturation throughput drop. Recorded in
          FULLRING_MCAST_GAP.md §8.13. The user asked to note it, not to adapt CBS.
    **User (2026-11-03) approved running the A/Bs on P150 and Galaxy.** The full-ring agent was resumed (it holds
    the context): the no-build A/Bs now (Galaxy 8-ring-only vs 4-ring-only, size sweep, slots, keepalive, 3072 B
    sampling; P150 multicast size sweep 3584-4352 B on the LoudBoxes). The variant designs (bounded completion wait,
    local-write skip / NoC0, one-side delays) come to the main session before any build. The point-5 PC samples (within-ERISC splits) exist
    to analyse; an A/B is needed before any cause claim.
  - Breakdown samples for points 2/4/5 (both routers, two passes) are saved under
    /data/aagarwal/frs-glx/jobs/169481/results (manifest, ELFs with DWARF, src snapshots). Mapping them into
    breakdown.json is still to do (breakdown phase 2), together with the P150 points on the final build.
- **Second BH LB via Slurm** (user 2026-11-03: "if the queue is backed up we can start dispatching to other BH Lbs
  as well"; second-LB agent):
  - both bh-lb-120-a07u28 (bh_lb_C45) and a08u28 (bh_lb_C67) while available (user 2026-11-03), up to two LB
    allocations; not a04u24 or a08u24;
  - be smart about what goes there (user): one /data snapshot with cheap refreshes (variants and scripts only).
    Send only frozen-input jobs, no host-library changes and no a09u32-only caches, ~20 min or more per dispatch,
    both passes of an A/B on one host. The sender round stays on a09u32;
  - read-only host check, then a node-local tree at /home/aagarwal/tt-metal staged via /data;
  - harness: sbatch + lb2_run.sh wrapper (node guard, busy check, reset in own allocation, results back with a
    manifest);
  - calibration on ew4352 / hop3 / lin3 (main_fw, v4rxg3) against a09u32's numbers, then report;
  - first jobs to move: N/S probes 502/503 and 504/505, if they haven't started on a09u32 (cancel only queued ones).
  - **01:10 status:**
    - a07u28 passed the host check (169530: wiring identical, all 48 mesh link ends 400G, FW 19.16.0 / KMD 2.11.0
      as a09u32) and calibration (169531: ew4352/hop3/lin3 equal within 0.6; the 260-row sweep median |diff|
      0.01-0.03%).
    - Calibration outliers: v4rxg3 Mesh3HopEW scatter is 1.1-3.5% faster on a07u28 (main matches); Linear3HopEW
      writes +0.4%. So multi-hop rows are compared within one host only. Cause untested.
    - Then another user (ahossain, 169537) took a07u28. 510/511 stay on a09u32.
    - a08u28 is idle, but /home/aagarwal/tt-metal there is an old Oct 9 tree. Using it needs the user's OK to rename
      that tree aside (not delete).
    - Harness: consolidation/lb2_*.sh, stage on /data/aagarwal/lb2; results in fabric_exp/results/lb2/.
    - XY torus files ready and mock-vetted (fabric_exp/torus/; the 4x2 MGD has dims [4,2] with dim_types
      [RING, LINE]; DA=1 on all four directions; 9/9 groups past setup per shape and router). The axis-derivation
      check runs at device validation.
  - **01:50:** user OKed renaming old trees aside, and a04u24 is added to the harness (lb2_run.sh / lb2.sbatch).
    Stage refreshed (src-20261103-014849). Queued in Slurm, all waiting for nodes held by other users:
    - a08u28: 169605 move-aside + sync, 169606 hostcheck, 169612 XY;
    - a04u24: 169607 move-aside + sync, 169608 hostcheck, 169613 XY;
    - a07u28: 169565 dry, 169614 XY.
    XY runs once, on whichever node starts first: a lock dir /data/aagarwal/lb2/locks/torus_xy; the others exit 0.
    After it, collect with lb2_collect.sh <id>, then report/scripts/build_torus_xy.py, then the page.
    Calibrate a08u28/a04u24 later.
  - **01:51:** the a09u32 broker went idle, so XY torus runs here as broker job 512 (claimed the lock). The LB XY
    copies (169612-169614, dry 169565) were cancelled. The setup chains (169605/169606 a08u28, 169607/169608 a04u24)
    stay for future dispatch. After 512: build_torus_xy.py, then the page.
  - **512 done** (138 s, both passes, both shapes).
    - Axes realized as wanted: 2x4 Y=LINE X=RING, 4x2 Y=RING X=LINE.
    - Bug: jobs_lib.sh's VAL="stream mesh val1d" overrode the script's VAL flag, so the data-checked validation and
      the DA check were skipped. The flag is renamed TXY_VAL; validation reruns as pass c, job 515.
    - Results (cycles/packet, flat across sizes for the new router):
      - 2x4: 1-hop E/W main 268.1 / new 146.9; row multicast 267.1 / 202.5; 2D multicast 159.6 / 126.1.
      - 4x2: 1-hop N/S main 268.9 / new 150.3; column multicast 306.6 / 259.8.
      - Mesh for reference: main ~222, new ~127 at 4352 B. Torus mode costs main ~46 and the new router ~20.
    - **515 (pass c with validation):** 12/12 data-checked per router and shape; axes as wanted; DA=1 in all four
      directions (checks sidecars txyvc_*_checks.txt). Data: report/data/bandwidth_torus_xy.json (320 records).
      Page v20 shows the 400G XY torus rows: 1-hop main ~268, new ~147-150 (line 125.4); full-row multicast
      324.0 / 264.3, column 358.1 / 297.4.
- **Breakdown P150 sampling:**
  - Job 516 failed on an unset EXTRA_ENV under set -u.
  - Job 518 failed because sample()'s subshell inherited jobs_lib.sh's EXIT trap and restored the tree after the new
    router's first sampled row; the next row compiled the stale base v2. Fixed with `trap - EXIT` in the subshell.
  - Rerun as job 522 (EXTRA_ENV= bash bd_p150_runs.sh v4rxg3 g3p).
  - Then bd_phase2_sets.py p150 g3p v4rxg3 + build_breakdown.py, then the page section.
  - **522 done; breakdown built** (report/data/breakdown.json, 140 records, all 5 points, both routers, both
    ERISCs, two passes).
    - Fixes: build_breakdown.py crashed on an ERISC with no pick (now skipped with a message). The Galaxy p2/p4 sets
      matched no build through the log-derived config hashes; they are now matched by launch-layout head ("logs" ->
      "logs_unused" in sets_phase2.json).
    - Point 5 caveats (FULLRING_MCAST_GAP.md): the busiest-router pick lands on a 4-chip-ring router, not the
      8-ring routers that likely set the row; the v4 pass-b set's traffic starts ~17.5 s into its 25 s window.
    - Not yet built: the costliest-rows profile (points_costliest.json -> data/breakdown_costliest.json). Not yet on
      the page.
  - **02:05:** a08u28 (169605/169606) and a04u24 (169607/169608) are set up.
    - Old trees moved to /home/aagarwal/tt-metal.pre-lb2-20261103; synced; host checks rc 0 (12 pairs x 2 links).
    - **a04u24 runs firmware bundle 19.12.0** (a09u32 / a07u28 / a08u28: 19.16.0). Calibrate before comparing
      absolute numbers.
    - Calibrations 169656 (a08u28) and 169657 (a04u24) done and collected.
      - main_fw matches a09u32 everywhere: ew4352 222.5, hop3 221.7-221.8, lin3 187.5.
      - v4rxg3 matches on ew4352 (126.9-127.0) and lin3 (124.4), but **2D 3-hop varies by host**: a04u24 142.1,
        a08u28 142.9, a09u32 146.7, a07u28 147.3. Same pattern as a07u28's scatter outlier: the new router's
        forwarded multi-hop rows are host-dependent and main's aren't. Cause untested (harvest masks / worker
        placement are hypotheses). Compare multi-hop rows within one host only.

- **P150 "torus" rows don't use wrap-around links** (found 2026-11-03 on the user's question):
  - FABRIC_2D_TORUS_Y on the P150x8 logs "TORUS_Y could not be realized ... using MESH" (topology_mapper.cpp:1794)
    and "axis topologies derived as Y=LINE X=LINE" (control_plane.cpp:2606). The routers are built and run in torus
    mode, but traffic is mesh-mapped.
  - The cube wiring closes each 4-chip row (X) into a ring; Y (2 rows) has no wrap. 1D Ring is realized (no
    fallback). Torus X (row wraps) was never tried on the P150; it needs mock vetting first.
  - Page v16 states this under "2D torus at 400G" (55 rows, all 400G).
  - Mesh shape is 2x4: Y = dim 0 (2 rows), X = dim 1 (4 columns). topology_mapper.cpp:1656 even forces 4x2 to 2x4.
  - Deadlock avoidance follows the *requested* fabric type, not the realized axes:
    FabricContext::need_deadlock_avoidance_support (fabric_context.cpp:305-333). Only under express routing does it
    ask per realized axis (mesh_has_protected_ring_in_axis_of).
    - Captured args, cov_m cases: TORUS_Y (bh_torus_vc2*) is mesh-mapped but has DA=1 on N/S routers, 0 on E/W.
      TORUS_XY (bh_torus*) is realized with the X wrap only (Y dropped, extent 2) and has DA=1 on all four
      directions.
    - So on the P150, N/S routers carry DA (and first-level ack) for a wrap that doesn't exist. The dim-size-2 logic
      (topology_mapper.cpp:1770-1790 has_genuine_torus_axis; physical_grouping_descriptor_matching.cpp:1736) only
      governs mapping.
    - User (2026-11-03): "We asked for torus on Y, and we got it, and hence DA on. that settles it." Working as
      requested; not an issue to raise. An X-torus suite for the LoudBox (real row wraps) was offered, not requested.
  - The Galaxy agent was asked to check realization for its 200G torus/ring suites and label any fallbacks.

- **Per-side slack study, main vs the port (user-approved, 2026-11-05; port_tune/slack/NOTES.md).** One-side spin
  A/Bs on P150 (1-hop E/W 2048/4096/4352 B, NeighborExchange 4096 B 1/2 links), local roots only.
  - Status: done, including the settling A/B (mcycle spin, a09u32 jobs 840/841, NOTES §10). slack.json has the mcycle
    set; the wall-clock set (jobs 833/836/837) is kept as `wallclock_*` records, SUPERSEDED. The page's SLACK_HOLD on
    `port@e7fe1899049` is the coordinator's to release.
  - Bound sides: port 2D 1-hop both ERISCs (clear-drop knees 4-8 cycles on both, E0 steeper); port NeighborExchange
    neither; main E0 on every row.
  - **Follow-up 1, spin floor / N+12 reading:** the study's v4txa11 "E1 slack ~25" at Mesh1HopEW 4352 B (slack.json
    v4txa11 records) is an N+12 model reading. On the port the full mcycle ladder shows E1 pays +1.3 at D=1 and +4.9 at
    D=8 (knee 4-8), with c32 matching v4txa11's. Hypothesis: v4txa11 reads the same way. To settle it, run the mcycle
    ladder (D 1..32) on v4txa11's Mesh1HopEW rows; mark its records WRONG only if it reproduces. Also open: the
    wall-clock spin's ~10-cycle step at D>=4 (D>=2 on main's E0); the debug wall-clock read was not timed (NOTES
    §10.2).
  - **Follow-up 2, main's E0 spin cost:** main's E0 pays ~11 cycles/packet for a 1-cycle mcycle spin (the port's
    ~1.3), on the regular 2D path and on the speedy NeighborExchange path. The cause is not isolated. Hypotheses:
    kernel_main code generation around the inlined send copies (+12 instructions at mtx1), or the spin delaying E0
    work after a send. First isolating A/B: the same spin at a different main E0 site (e.g. after the credit
    processing), or a code-only arm.

## Next, in order

1. **Host: correct the worker-role rule in the main tree** (`SENDER_CH_<i>_IS_WORKER_CHANNEL`): the first sender
   channel of each VC whose mode has a worker (FULL_2D, LINE, NEIGHBOUR yes; INTER_MESH no). **Done (2026-11-02, job
   448):** `is_worker_channel` (also the live-check skip) = VC0's worker-connected channel or VC2's first channel;
   patch pinned in variants/host_rebuild_20261102/, script host_rebuild_20261102.sh, backup
   ~/.tt-metal-host-rebuild-backup/erisc_datamover_builder.cpp.pre20261102. Args identical on every current config
   (V4_SENDER_OWNERSHIP.md §12).
2. **Sender glossary and structure, revision 4:** written and reviewed (2026-11-02). §5 decisions settled except
   placement. **P-B / P-C probe done** (jobs 422, 425, 429, 432; V4_SENDER_OWNERSHIP.md §8.3): P-C flat or better on
   every row (keeps the lead's VC2 far-ring spill; +2 carried values per line VC); P-B no hot spill in any family but
   its far rings in memory cost +4.4 combined fan-in at 256/1024 B, +3..+4.7 VC2 alone, +4.2..+8.8 VC1 inter-mesh 16 B.
   **User picked P-B (2026-11-02).** Packed probe done (jobs 445-446, `v4txpbk`, §8.4): every VC's remote receiver
   progress as one word (low 16 bits free slots, high 16 bits next slot address / 16). Against P-B: multi-VC rows back
   to the lead (VC2 alone -3.5..-3.9, VC0 contended -11.5..-13.9, combined fan-in 122.0 -> 118.5 at 256 B; P-C 117.0),
   single-VC +0.1..+0.7, VC0 with VC2 compiled in and idle +2.3/+2.4 at 256/1024 B; loop -16 instructions, 2 fewer
   carried registers, 15 fewer stack accesses. **User adopted packed P-B (2026-11-02)**, as a proper type with operations
   (`RemoteReceiverChannelProgress<VC>`: has_room(n), next_slot_addr(), take_slot(), add_free_slots(n); the encoding
   and its static_asserts in one place; no bit masking at call sites). Optional, before any fairness work: the VC2 visit-timing delay sweep (TXQ share
   hypothesis, §8.3). Probe caches (~/.tt-metal-cache-txp*, -txpvc1-*, -bndp-txp*, cov_m txpq_*) deleted (job 449, refs checked).
3. **Sender ownership pass** (X0-X6 on the receiver lead), after review:
   - host-owned credit command buffers, with a third (RD) for a 1D VC1 worker;
   - per-VC first-level acks on the sender;
   - one shared `worker_connection_event` (cold path; library refactor of the worker interface's connection state);
   - begin/end_read_group for the sender's fences (router_sender.hpp lines 240, 243, 279, 282, 286, 301, 308, 312,
     467, 474);
   - comment trim (the sender still has cycle numbers and variant names);
   - register-budget inventory plus headroom check for the two-worker routers; don't pack hot counters.
4. **VC2 line, 2D** (VC2 line agent, worktree tt-metal-vc2line, branch aagarwal/vc2-line): rebase onto the receiver
   lead **after the receiver g1-g3 stage lands** (user, 2026-11-02) → meshes larger than 2x2 (needs buffer-2 fix (c), which is in the receiver chain) → phase 5 multi-Galaxy rings
   and multi-mesh. In parallel with the sender pass. VC2_LINE_BRINGUP.md. **Rebase and 4x4 done; phase 5 on hold by
   the user (2026-11-02).**
5. **Slot study P4:** VC1 on Galaxy 4x4 / 4x2 splits (prepared; needs the receiver chain's buffer-2 fix).
   SLOT_STUDY.md.
6. **1D multiple line VCs, router side** (after the sender pass; host side pushed as 0d76a021a70):
   - read the new host args;
   - per-VC 1D stream names;
   - drop the kernel's VC gate;
   - third credit buffer;
   - per-VC sender acks;
   - VC1/VC2 1D downstream rt-arg blocks.

   Headline row: N workers fanning in on N VCs, against today's mux path. LINE_VCS_HOST_DESIGN.md,
   VC2_LINE_1D_VCS.md.

## Planned stages (after both passes land)

- **Receiver, N/S floor fix 1:** partial presets on routers with several live downstreams. The forward writes only its
  per-packet destination fields; the credit rides its own preset buffer (buffer 1). V4_NS_FLOOR.md.
- **Receiver, N/S floor fix 2 (probe):** the next packet's route word loaded ahead into channel state, stored after
  the step's tail (precedent v4rx1rg lost on rings). Then the decode's ~8 cycles, only after an isolating A/B. Its
  row set must include the codec's rows (forwarded atomic / fused / scatter, multicast scatter, E/W fused, one-ERISC
  E/W: +1.8 to +4.2 accepted 2026-11-02, V4_RECEIVER_OWNERSHIP §17), as it targets the same route-before-admit path.
  The codec takes a route-buffer pointer; a load-ahead needs a decode from loaded bytes (additive hostdevcommon API).
- **Receiver, single-source rings:** no SlotSources and a constant credit target when only one far sender channel can
  feed a VC's receiver (VC2 neighbour, speedy VC0, mux, trimmed shapes). Needs a host fact.
- **Code space: make the cold init and control paths runtime-parameterized** (one copy instead of per VC/channel):
  receiver per-VC downstream init, sender worker_connection_event, and anything else the size breakdown shows. Init
  and control copies are 11-18% of each image. No per-packet cost.
  Also: worker_interface_storage (336 B .bss) sits in E1 images, which serve no workers (bh_vc1 E1 least free stack
  4,236 -> 3,896 B from sender stage A).
  Target (user, 2026-11-02): fit other slot allocations with VC1 + VC2 at -O3. Today default slots leave 240 B
  including runtime args, and VC2 8/10 is 16 B over. Then rerun the three-VC rows at those allocations.
  Why slot counts change code size (2026-11-02, mvc3 caches A_d vs A_v8):
  - Measured on a pair with the same named args, only the slot args differing: E1 +68 instructions (272 B). Nearly
    all of it is in the receiver's per-VC code; VC2's receiving count (4 -> 10) is the only receiving count that
    changed, so it is attributed to VC2's copy.
  - Index wrap: AND for a power-of-two count, compare + subtract otherwise (index_add, router_receiver.hpp:66-73).
  - The partial last completion group (10 = 4 + 4 + 2): router_receiver.hpp:230-231, 525-560.
  - Lever to test: power-of-two counts that are multiples of the group (e.g. VC2 receiver 8) should add little or
    nothing, or the non-power-of-two path could be made cheaper. Untested.
- **Flush defer, strict mode with the stall removed** (FLUSH_DEFER_DESIGN.md, revision note and user clarification
  2026-11-02):
  - one pending atomic per channel; the channel takes no new packet while it waits;
  - E1 keeps other VCs going, plus the current VC's acks, completions and next-header load-ahead;
  - the proof is acked (REQS_OUTSTANDING_ID), reusing the completion bookkeeping;
  - design it together with N/S fix 2; the headroom check flagged torus + VC2 with a flush-defer record.
  The relaxed flush mode (later writes go through) is separate and more complex; not now.
- **Sender per-packet send cost:** at small packets the sender's per-packet work caps multi-VC totals (~115
  cycles/packet on P150, jobs 389/390).
- **Sender batching** (equal per-channel bound): parked until the slot decisions (2D worker depth); re-test on the
  chosen split. V4_SEND_BATCH.md.

## Parked

- **VC2 default-slot regression (+2.1%, X0..X4)** in the sender pass: stage bisect job designed (Sender pass entry),
  parked by the user 2026-11-03.
- **P150 ring multicast write +0.9..3.3 (X3k, X3a):** parked by the user 2026-11-03; record in
  V4_SENDER_CONSOLIDATION.md §6.11.
- **Channel-trimmed MeshMulticast replay +2.4 write / +1.1 scatter, from C0 (the N/S-fix merge):** parked by the user
  2026-11-03.
  - Hypotheses ruled out: H-C0a (wait order; v4txa11w flat) and H-C0b for the write row (E0-bound).
  - The new router is still ~17% faster than main on the row; only trimmed routers move.
  - Record: V4_SENDER_CONSOLIDATION.md §6.12.
  - Not adopted: v4txa11w, v4txa11g.
- **TRID allocator: the NoC transaction-ID count from the HAL** (parked by the user 2026-11-05). Today
  fabric_trid_allocator.hpp:31 hard-codes `noc_transaction_ids = 16`.
  - Plan:
    - Add a Hal field and getter, e.g. `get_noc_num_transaction_ids()`, next to `get_noc_max_burst_size_bytes()`
      (hal.hpp:640). Set it in bh_hal.cpp / wh_hal.cpp to `NOC_MAX_TRANSACTION_ID + 1` (16).
    - Set it in qa_hal.cpp to the kernel-usable pool `USER_TXN_ID_MAX + 1` (8). On Quasar the dataflow buffers own
      IDs 8-15 (dataflow_buffer_config.h).
    - `assign_transaction_ids(channels, available_ids)` takes the count as an input. The caller
      (erisc_datamover_builder.cpp:1478) passes the HAL value.
    - Check with the standalone old-vs-new harness: identical outputs on WH/BH.
  - Related, same mismatch:
    - the device static_assert in router_receiver.hpp:83 bounds by NOC_MAX_TRANSACTION_ID, which would allow Quasar's
      DFB IDs;
    - FabricMuxV2Config::kTridRingCapacity = 16 (fabric.hpp:399) could use the same getter.

## Dynamic slot allocator (user, 2026-11-03)

- **Rework (user, 2026-11-04):** decent defaults and simple later tuning, not per-case optima; no override
  mechanism or perf notes in the allocator now; golden table CPU-only if needed. Staged design (needs; receivers
  first in order, not amount; recalibrate; senders; leftover list) in ALLOCATOR_NEEDS_FIRST.md (v1 kept as _v1),
  sketch slotalloc/staged_sketch.py. Waiting on: §12.12 on the port (lb2 175938) and the user's D1-D4.

- **Two steps (agreed):**
  1. Make it dynamic, still one allocation for every router on the chip.
  2. Adapt it per router on top of the team's builder changes.
- **Design (agreed):**
  - A pure allocate_slots(router shape) function: step 1 calls it chip-wide, step 2 per router.
  - Inputs: arch, slot size, VCs present, channel kinds, mesh dims, ring/line per dim. No measured perf, no link
    speed, no optimizer.
  - Byte-budget shares (slot-size invariant).
  - Minimums: >= 2 per channel; ring receivers bubble 2 + K+2 (K+2 dropped later, see Redesign).
  - About five weights (per channel kind, VC priority, a long-dim forwarding bump), seeded from measured ratios;
    a byte-share cap; reasons logged; explicit override kept.
- **Step 1 (a fresh agent, study branch, off by default):** SLOT_ALLOCATOR.md, a builder switch, an offline
  allocation table vs tiers and the study's best rows, then a P150 + Galaxy A/B at 4352 and 3072 B slots plus a
  code-size check. It becomes mainline commit 6.
  - 2026-11-03: reworked to the user's step-1 decisions (slots; VC ratios 3:1 / 12:1 / 12:4:1; integer kind weights;
    order a-d; `receiver_round`; min 2, no K+2). SLOT_ALLOCATOR.md, `TT_FABRIC_EXP_DYN_SLOTS=1`, host patch
    variants/host_dynslots; tools in slotalloc/.
  - Fit:
    - worker 2 : forwarding 1, x2 on long dims, receiver 4/5 of its VC's heaviest sender;
    - 3:1 with largest-remainder VC shares;
    - at 4352 B it keeps 17,16,16 / 26,25,32 / 19,9,16,5,4, and VC1 first (2:1) gives the measured 18,7,16,7,5.
  - Rows that move: VC2 on 17,14,16 + 4,2 (was + 2,4); three VCs 17,8,16,5,4 + 3,2.
  - 2026-11-04:
    - host rebuilt (broker 648);
    - LB full stage full-20261104-001304;
    - Galaxy tree v4txa11_dyn;
    - mock vet clean (26 checks; code +168 B max, stack unchanged);
    - submitted the P150 A/B (lb2 171995, a07u28; 171943 stopped on a08u28's /home space) and Galaxy pass a
      (171946, running); pass b waits for the cap.
  - Slot-need sweeps (SLOT_ALLOCATOR §12, knee/): sets A-G3, Q, CG, XO done (§12.1-12.9). §12.10 done 2026-11-04:
    study switch TT_FABRIC_EXP_COMPLETION_GROUP (host rebuild broker 769; Galaxy tree v4txa11id_1104cg; gate
    cg_gate.py, vet trid_vet.sh CG=) and the k-channel merge sweep (sets M24 / M42 / MG; broker 777, 778; Slurm
    175221; knee/merge_report.py). Result: G4's cost on squeezed senders shrinks with k and reverses once the merged
    supply is enough; G4 is fastest for unsqueezed senders; rules A / B (G2 for squeezed depths) do not beat today
    across the suite. §12.8 / §12.9 depth readings marked WRONG. Open: why G4 is faster on a busy merged link (no
    tested hypothesis), which ERISC bounds it (needs a one-side A/B); both wait for the user. §12.11 (broker 787): P150
    1D at 8704 B slots (13,12,16): at 8704 B packets G1 / G2 / G4 all 243.8 cycles (no large-slot exception); at
    256 B G4 costs a lone worker 6.3% against G2 and wins the 2-flow merge by 11.3%.
  - The `dynw24` arm decides `worker_cap_slots`. Galaxy (passes a + b, 171946 / 172172): dynw24 never >1% slower
    than dyn, up to 3.6% faster on the VC1 split at 3072 B. VC1 split 3072 B misses the best row by up to +17% on the
    inter-mesh rows (both passes; hypothesis VC1 7 / 6 against 9 / 7, not isolated). P150 rows: broker parts
    `dyn_lb_ab.sh r2 a1|a2|b1|b2` (a07u28 needs an operator; a04u24 busy).
  - Receiver multiple-of-4 sweep done (lb2 171579, SLOT_ALLOCATOR §9):
    - mesh rows: 0 except 3-hop scatter +1.5 cycles/packet;
    - ring: +8 (+5%);
    - `receiver_round=mult4` on non-ring routers is the user's call.
  - VC0 receiver floor on 2D routers (user, 2026-11-04; SLOT_ALLOCATOR §10), offline done:
    - `vc0_receiver_floor=1:5` (default; `off`), share = max(weighted, N/5), then the pow2 rounding; `two_d` from
      the topology; a missed floor takes what fits, rounded down to a power of 2, and logs both (three VCs at max
      payload: 2);
    - "every channel's minimum first" (approved by the coordinator): VC shares raised to their channels' minimums,
      and senders raised to the minimum take slots back from the largest sender; only max-payload rows that were
      errors change;
    - 4352 B must-keep rows unchanged; the 4x4 VC1 split goes 15,13,8,5,4 -> 13,11,16,5,4; every 2D 3072 B row's VC0
      receiver goes 16 -> 32 (2D 23,20,32; VC1 split 26,10,32,7,6);
    - host: patch 1 updated (one line, `shape.two_d`); goes into the TRID rebuild (host_rebuild_tridalloc.sh checks
      both);
    - the running A/B (lb2 171995, Galaxy 171946, pass b on trees/v4txa11_dyn) measures the pre-floor rows; on a
      floor library the scripts run dyn / dynw24 with `vc0_receiver_floor=off` themselves;
    - next P150 / Galaxy round: the `dynfloor` arm at 3072 B (2D 23,20,32, VC1 2x2 26,10,32,7,6) next to the
      floor-off rows (dyn_lb_ab.sh, dyn_glx_submit.sh add it on a floor library); the vet's dynfloor arm recaptures
      the VC1 split for the TRID compile.
- **Transaction-ID allocation that follows the slot counts** (user-approved 2026-11-04; TRID_ALLOCATION.md):
  - host-owned rule per receiving ERISC;
  - per-channel G / ids / start named args;
  - router variant v4txa11id;
  - switch TT_FABRIC_EXP_TRID_ALLOC=1.
  - 2026-11-04: rule, host patch (variants/host_dynslots/2_trid_alloc.patch, host_rebuild_tridalloc.sh) and variant
    built. Offline table: every expected value holds. Offline compile of captured configs (trid_offline_build.py):
    152 ELFs, E1 +32 B on the VC1 split, stack unchanged.
  - 2026-11-04 correction (coordinator): a shrinking count rounds down to a power of 2 first (20 -> 16), then halves;
    the first version sent 20 to 8. Fixed in fabric_trid_allocator.cpp; unit checks added to the offline table (all
    pass). No captured config or table row changes (host only, no router change).
  - Host rebuild with the slot floor: broker 649 done (HOST_REBUILT, library has the floor and the TRID switch). lb2
    171995 cancelled while pending; LB re-staged (src/full-20261104-020334); P150 dyn A/B resubmitted as lb2 172043
    (a07u28). Galaxy base_1104trid with trees v4txa11id_1104 and v4txa11_1104. TRID mock vet (trid_vet.sh): 728
    configs, 0 mismatches against the rule, switch on and off; code E0+E1 22,268 max. Slot vet on the new host:
    floor-off rows = pass a's, dynfloor rows as the table.
  - Latent harness race in the VC1 suites (coordinator, 2026-11-04; the port's VC1+VC2 "hang", broker 660): the
    data-checked MultiMeshAllToAll runs 100 packets per flow, no flow control, inter-mesh across the two ranks, into a
    256 KiB receiver ring (max_configs_per_core 4): 64 slots at 4096 B, 85 at 3072 B. The start sync is per mesh, so
    one rank's senders can lap the other rank's receivers. Exposed: vc1/{sls_vc1_all,vc1_alloc,vc1_sanity,vc1_split},
    slotstudy/sls_glx_vc1{,_4x2,_4x4}, slotalloc/yamls/sls_vc1_all_3072 (sls_glx_vc1_3k at 2048 B: 128 slots, safe).
    A hang or data mismatch there is the harness, not the router. Fix (coordinator-approved: cap, not flow control),
    applied 2026-11-04 03:48 after 671/672, lb2 172170 and Galaxy 172172 had ended: slotalloc/cap_ring_packets.py
    --apply, num_packets 100 -> 64 (85 in the 3072 B file). Checked by parsing before / after: in every file the only
    change is MultiMeshAllToAll's defaults.num_packets, a data-checked test (benchmark_mode off) with no benchmark rows
    in the CSVs, so no benchmark row's numbers change; every other test and the top level are identical. LB stage
    refreshed (src-20261104-034921); the Galaxy trees' copies (sls_bdd, sls_bddml, sls_bddsl, sls_lead,
    v4txa11_1104, v4txa11_dyn, v4txa11id_1104) capped by replacing the files (new inodes, so running hard-linked job
    copies are untouched). The porting agent uses the same cap for its vc1_sanity.yaml.
  - P150 validation passed (broker 661, 662: 12 runs; every two-ERISC run logged the assigned IDs, none uniform).
    Galaxy ring / torus rows: Slurm 172183 (trid_glx.sbatch).
  - P150 A/B done (broker 671, 672; TRID_ALLOCATION.md §6.2): the ID counts give no gain (ring 0.0%, 2x4 torus -0.5%,
    VC1 0.1%), lose ~+1% on the 4x2 torus N/S rows and +1.0-1.5% on two VC1-with-floor rows; census neutral; the code
    change alone +0.3% on census 1D. Galaxy rows (Slurm 172183) agree: no suite gains; torus N/S rows +1.4%
    (~+2 cycles/packet), as on the P150 4x2. For the user's decision: the rule as built does not pay.
  - Credit-cadence A/B done (broker 704 safety, 705 A/B; TRID_ALLOCATION.md §6.4): the snapshot loop leaves the
    rate unchanged (+0.02 cycles/packet). Credits per data packet are 1.000 with 8 and with 16 IDs on every 4x2-torus
    N/S row (k = 2.00), while 16 IDs still lose 2.0-2.2 cycles/packet: hypothesis A (credit cadence) WRONG. Next, on
    a go: hypothesis B (occupancy), round-sampled far-ring room.
  - Port measurements (2026-11-04, user top priority; SLOT_ALLOCATOR §12.12 to be written): worktree
    /home/aagarwal/tt-metal-port-slots at 22cdb1e55c3 + fabric_exp/port_slots/slot_override.patch (local, never
    committed: the study's explicit-row switches and slot log), staged lean at /data/aagarwal/lb2/trees/port_slots_22cd.
    lb2 175938 (port_slot_runs.sh V F2 F1 S: forwarding knee 2D / 1D at 3072 B, receiver and forwarding at 3072 vs
    4352 B) and 175939 (pvs_runs.sh: port against study, slot rows matched; afterok 175938). All arms mock-vetted.
  - Host reconciliation done (RECONCILE_HOST.md, offline): slot rows differ on 1D (study split) and VC1 / VC2 (study
    design rows); T = 16 on the port's 32-slot rings (ring-jam fix) against 2; ID counts identical (uniform rule);
    study-only args unused by v4txa11; -DL1_ECC_HALT only on the port (main's). Port worktree to clean up after
    §12.12 (the user's call).
  - Offline codegen diff of the N/S loss done (TRID_ALLOCATION.md §6.5, no device runs): E0 byte-identical in both
    arms; only the 4x2 N/S receivers get a per-packet stack reload (a chip coordinate in the route decode) and a net
    +1 instruction per packet with 16 IDs, every other receiver nets -1. Hypothesis C (that spill causes the ~2
    cycles) and B both untested; the isolating A/B (change only that spill) needs a design for the user.
  - P150 validation script: slotalloc/trid_val_p150_runs.sh a|b (part a sanity / ring / tori, part b VC1 split,
    inline + atomic gtests, UDM).
  - Next:
    - the host rebuild (broker, through the coordinator); the LB re-stage must wait for lb2 171995 to start;
    - mock vet;
    - P150 validation;
    - the A/B.
- **Redesign (user, 2026-11-03, sent to the allocator agent):**
  - in slots: N = floor(buffer bytes / slot size); caps in slots or fractions of N;
  - VC split by an integer ratio by VCs built: VC0+VC1 3:1, VC0+VC2 12:1, all three 12:4:1 ("VC1 first" = 2:1);
    replaces vc_step / vc1_first / vc2_priority; kind weights become integer ratios too, fitted to keep today's
    4352 B rows;
  - receivers first (VC0: nearest power of 2, policy key `receiver_round`), senders split the rest, leftover to the
    VC0 worker up to its cap, past the cap to VC0 forwarding;
  - K+2 ring-jam minimum DROPPED (bubble 2 stays);
  - worker cap in slots (24) stays an option, decided by the dynw24 arm. 24 is a pick, not a measurement (3072 B best
    worker ~20; 1D 4352 B flat 23-27);
  - mix rows are judged by the link total (user: yes), with per-flow numbers shown.
- **Transaction-ID allocation from slot counts (user, 2026-11-04; allocator agent, after its vet + A/B submits):**
  per receiver channel, groups = ceil(slots / G) with today's G rule; one ID per group when the total fits the NoC's
  16 (identity map); otherwise shrink the largest count until it fits: round it down to a power of 2 first (20 -> 16),
  then halve (mask); at least 1 ID each; leftover
  IDs unused; consecutive starts. The host owns the rule (per-channel named args; the router asserts it). Switch
  TT_FABRIC_EXP_TRID_ALLOC=1, variant on v4txa11. Replaces main's uniform 8/4 and the study's lone-receiver 16, which
  were never A/B'd as counts. Design and results: TRID_ALLOCATION.md.
- **Allocator reframe (user, 2026-11-04): slots, not bytes.** A slot holds one packet of any size up to the slot
  size, so a channel's need is packets in flight (credit loop / time per packet), and it bottoms out at the router's
  per-packet cost: smaller packets don't need more slots. The byte framing (VC1 "constant bytes", the byte floor, the
  2 KB-slot worker test) is dropped. The 1/5 VC0 receiver floor hurt the VC1 split at 3072 B (VC0 2-hop +24-32%), so it
  stays off by default. Next: sweeps of each channel kind's knee in slots versus the traffic's packet size at a fixed
  slot size (worker, VC0 forwarding, VC0 receiver, VC1; P150 + Galaxy), then a needs-first allocator sketch for the
  user. No allocator rule changes until then.
- **TRID result and next step (2026-11-04):** the group-sized ID rule costs about 2 cycles/packet on torus N/S rows
  (P150 4x2, Galaxy) and 1-1.5% on the floored VC1 split. idoff (new code, old counts) is neutral, so the counts, not the
  code, cost. The user values the rule's consistency, so the mechanism comes first: credit-cadence A/B (credit frames
  per data packet at 8 vs 16 IDs, from read-only TXQ0/TXQ1 transfer counters; safety check first). Hypothesis A: 8-ID
  aliasing batches completions, so fewer credit frames (WRONG, broker 705: credit frames per data packet equal at
  1.000). Hypothesis B: earlier credits raise ring occupancy (untested, next). Allocator agent owns it.
- **Receiver multiple-of-4 A/B (no build, does not block the bring-up):** P150 2D 3072 B receiver 16/20/24/28/32 with
  the senders fixed, plus ring 24/28/32. Hypothesis: mesh routers (groups of 4) pay ≤1 cycle/packet off a power of 2;
  rings (first-level acks, groups of 1, ~4 index wraps/packet) pay ~8 as on v4u (RESULTS §98). If meshes are free,
  the candidate is `receiver_round=mult4` on non-ring routers (user decides). A router change for cheap ring wraps
  needs a design first.

## Mainline port follow-ups (user decisions 2026-11-04)

- Branch origin/aagarwal/fabric-router-mainline at 4db4f45790b (2026-11-04, fast-forward from b2473363016): 6 port
  commits + WH doc + 4 image-identical cleanups + WH bring-up tools + the cleanup batch (5206b9a57da names, d7a9ced282f
  VC1+VC2 ring/torus TT_FATAL, d2343171a56 stock inline write, bd290f3246c SKIP_SRC_CH_ID_UPDATE dropped, 25d65133347
  generic inline with_state write, 4db4f45790b is_vc_configured; no opt-rule commit: -Os only for VC1+VC2).
  Validated on the P150 (a09u32): 659 inline_udm (inline 1D/2D, UDM unit 8, nightly 68: pass); 650 two, its non-trim
  runs (mesh 96, val1d 143, ring 19, Y-torus 12, mux 19, VC2 10, NeighborExchange 7, trimming gtests 63: pass; its
  trim2d ran the earlier random-pairing yaml, which hangs by design; the deterministic yaml has not run on the batch
  yet); 674 vc1 at 4db4f45790b (vc1 and vc1_1e at -O3, vc1_vc2 at -Os, 12/12 on both ranks each).
- Unpushed on aagarwal/fabric-router-mainline-cleanup after 4db4f45790b: 3b4ccab4dd7 cold (control paths; was
  16260ea5bff, message "first of two") and cb42eb72501 multicast landing out of line and cold (was cdceea943bc; message
  states the pass-through receiver::run side effect, cause unknown). Re-gate passed as decided (PARITY.md, CI table).
  Validation tree /home/aagarwal/tt-metal-mainline and glx tree at cb42eb72501 (no host change).
  - P150 validation of cb42eb72501 clean (broker 723 two incl. deterministic trim2d 2/2, 724 inline_udm, 725 vc1:
    vc1 / vc1_1e -O3, vc1_vc2 -Os); push of 3b4ccab4dd7 + cb42eb72501 cleared by the coordinator's rule.
  - Landing A/B (726 / 727, v4txa11land vs v4txa11, passes reversed): VC1 mean 158.3 / 158.4, inter-mesh unicast
    mean 333.1 / 333.0, every row within 0.4: no measurable effect. Multicast landings unmeasured (no traffic).
  - 288afd237a7 (A2, option b): noc_wwrite_with_state `update_addr_mid`; noc_write_to_node_with_state removed; gate
    5,007 / 5,007 images byte-identical to cb42eb72501.
  - **Pushed (coordinator, 2026-11-04): origin/aagarwal/fabric-router-mainline 4db4f45790b..288afd237a7** (3b4ccab4dd7,
    cb42eb72501, 288afd237a7). 288afd237a7 also committed a stray symlink tt_metal/pre-compiled (git add -A); fix
    2fc06852f48 removes it (no history rewrite), `/tt_metal/pre-compiled` added to .git/info/exclude; awaiting push.
    acc4656caaf's message stays; the PR description carries "no new NoC names, only extensions of main's".
  - Validation tree at 2fc06852f48 (rebuilt, CCACHE_DISABLE=1 while /data has no inodes; no host change).
  - Ring-jam rule (user-approved): ac86bf0f116 cherry-picked onto 2fc06852f48 as **df64d0ad677** (no conflict; patch
    identical). Gate (two mock captures, base host 2fc06852f48 = validation tree, new host df64d0ad677 = glx tree, the
    rule agent's 49 configs, bfc_gate.py pairing with the new arg stripped): router 170 differ / 5,362 identical, other
    232 identical; the 170 = bh_ring E0 36, bh_ring_1erisc E1 36, glx_ring E0 60, glx_1x32_ring E0 38, each exactly one
    instruction (srli 0x11 -> 0x14, visit_worker has_room, router_sender.hpp:643), code bytes +0. Unpaired: 80 + 80
    bh_2d_streaming router builds (paired via the JIT cache: 3 lui immediates each at kernel_profiler_streaming.hpp:202,
    the per-run profiler TU ids, as the rule agent found), glx3_mesh 2 + 2 (identical), and non-router test kernels.
    T from the new capture's named args: rings 16 (bh_ring x72, glx_ring x120, glx_1x32_ring x76, ring+mux x36 unused),
    tori / express / per-mesh torus / trim gtests 2. Outputs: bfc/gate/gate_full_on_2fc06852f48.txt,
    insn_summary_on_2fc06852f48.txt; captures pruned. glx tree now at df64d0ad677 (host rebuilt; its /data copy
    mainline_cb42eb7 unchanged).
  - G-PORT P10 (4x4 + 2x8 pass-through) staging of mainline_cb42eb7 (port manifest .scratch/regress/manifest.yaml)
    blocked: /data out of inodes (7 free, 2026-11-04 07:2x); Galaxy 172734 (A1/A2 pass b) failed at mkdir for the
    same reason and needs a resubmit.
    Done after the /data cleanup: staged mainline_cb42eb7 (07:50-08:00, verify OK); **G-PORT P10 Slurm 172825
    (bh-glx-110-a04u14): P10-mm-4x4 PASS and P10-mm-2x8 PASS** (each: multi-mesh stability 104 tests and the mesh
    pass-through yaml 48 tests on 3 ranks, 0 failed, no program-size error). The 2x8 pass-through line, which main
    fails 96 B over, runs on the port (router images of cb42eb72501 = 288afd237a7 = 2fc06852f48). The port manifest
    copy has min_data_inodes_free 200000 (coordinator's one-off; the shared manifest keeps 2M).
    A1/A2 Galaxy pass b: Slurm 172794 (bh-glx-110-a01u08), every step 0 failed; ran 2d2k pass b plus 2dk passes a/b.
  - **Pushed (coordinator, 2026-11-04): origin/aagarwal/fabric-router-mainline at 7cd3c13a0db** (df64d0ad677 ring-jam
    rule, 80ec66916c8, 7cd3c13a0db; unpushed on top: 3cbc209a497 order_point drop, c7a4d0fb29f comment nit).
    80ec66916c8 restores main's router design block at the top of
    fabric_erisc_router.cpp (flow-control section rewritten for this router); gate: every image identical.
    7cd3c13a0db full-preset worker credit (the credit buffer holds the whole posted write; a credit stores the read
    counter and sends): gate 2,466 E0 images differ, only WorkerChannel<0> (plus <4>/<7> VC2 workers) credit 19 -> 13
    and preset_credit 18 -> 24 instructions; non-router kernels identical. P150 validation at 7cd3c13a0db: 764 two
    (mesh 96, val1d 143, ring 19, Y-torus 12, mux 19, VC2 10 pass; nbr / trim / trim2d failed on /home ENOSPC, rerun
    775: nbr 7, trim 63 + 3 T3K skips, trim2d 2 pass), 765 inline_udm (udm_nightly ENOSPC, rerun 776: inline 1D/2D,
    UDM unit 8, nightly 68 pass), 766 vc1 (12/12 x3), 767 credit (worker 256 B 2, stability 960: pass).
    Perf A/B 768 (vs df64d0ad677, passes reversed, credit_perf_p150.yaml, cycles/packet base -> new): 1-hop
    unidirectional 4352 B -0.5 (1D 131.6 -> 131.1) and -0.4 (2D 128.4 -> 128.1), 1D bidir 4352 B -0.2, 2D bidir
    4352 B 0.0, 256 B rows 0.0..-0.1; both passes agree, base pass spread <= 0.1. The A/B changes only the credit, so
    the gain is the credit's; that fewer instructions per credit is why is a hypothesis.
  - Placement-primitive audit (coordinator, 2026-11-04), offline steps 1-2 at 7cd3c13a0db (11 arms, each removing one
    site; router-only PCH-less offline builds of the credit capture, a_base = the capture's JIT images 5,010/5,010):
    order_point (sender 617) is inert (5,010 identical); every read group and keep_in_memory site changes code
    (scheduling, spills, inlining of single::receiver_pass). All MMIO accessors in the read groups are volatile, so no
    site guards correctness at the compiler level; device A/B per arm decides keep / drop (step 3, not run).
    Tables: scratchpad audit_r1..r4.txt (copied to bfc/audit/).
    - order_point dropped (user): **3cbc209a497** on the cleanup branch (use and helper removed); gate: offline
      router-only build of the cleanup tree from the credit capture, 5,010 / 5,010 images identical to 7cd3c13a0db.
    - Per-config classification of each arm's loop changes (code = instruction / spill counts, sched = order only):
      bfc/audit/audit_cases.txt. rg_s537 / rg_s605 change order only on two ERISCs (rg_s537 +2 in single-ERISC
      sender_pass); rg_r755 changes two-ERISC receiver::run order only except bh_neighbor E1 (+17, not run).
    - Step 3 (user: about 1 h of device time): arm farms /home/aagarwal/tt-metal-audit/<arm> (mkfarm.py: symlinks into
      the validation tree, real router headers with the arm applied; mock-checked: the base farm's images equal the
      tree's, each arm's differ where the gate says); plan .scratch/audit/audit_plan.py (rows per arm from the
      classification, rg_s537 + rg_s605 as one arm rg_sched), runner audit_ab2.sh JOB (base, arms..., base on one host),
      report audit_report.py JOB. Jobs A (rg_s583, kim_s755, rg_d414) and B (rg_sched, rg_s673, rg_r755, kim_s816,
      kim_r713), about 14 and 13 min. Confirmation pass (reversed) only for arms past max(spread, 0.5).
    - Job A (broker 779, 389 s; bases agree within 0.1; cycles/packet, d = arm - base, negative = removal faster):
      rg_s583 2D 3-hop -5.8 / -3.0 (256 / 4352 B), every other row within 0.3, one ERISC flat: drop candidate.
      kim_s755 2D 3-hop +8.3 / +6.1, torus Y 3-hop E/W +8.7 / +6.3, every one-ERISC 2D row +7.6..+7.8, 1D 4352 B
      +0.6: keep. rg_d414 two ERISCs within 0.1, one-ERISC 3-hop -2.6 / -2.2, multicast -0.6: drop candidate.
      VC1 inter rows: no data (main defect below). Tables: bfc/audit/auditA_table.txt, auditAB_tests.txt.
    - Job B (780): VC2-worker rows do not compile in benchmark mode and inter-mesh rows abort (both main test-infra
      defects, TEST_PLAN §4.5); after the abort of the torus Y + VC2 run every later fabric init timed out on device 0
      eth core 28-25 (coordinator reset the host; hypothesis: the abort with the torus fabric up). Salvaged rows (per-test
      CSVs, against A's bases + B's first base): rg_sched 2D 3-hop 256 B +2.3, one-ERISC -1.0 (mixed: split into
      rg_s537 / rg_s605); rg_s673 2D 3-hop -1.1 / -0.8, one-ERISC -1.0; rg_r755 one-ERISC -2.9 / -3.0 / multicast -7.7.
      kim_s816 / kim_r713: no rows ran.
    - Job C (reversed confirmation + B rerun, VC0 workers on VC2-enabled fabrics for kim_r713 / kim_s816) and job D
      (two-rank split, VC0 traffic inside mesh 0, the VC1-config images) sent; runner now stops at the first eth-core
      fabric-init timeout. These rows measure the per-pass cost of the VC1 / VC2 routers, not VC1 / VC2 traffic.
    - **Result (C 785, D 786, E 788; every run passed; C reproduced A / B, E reproduced C / D within 0.2):**
      drop: order_point (done, 3cbc209a497), rg 537 (one-ERISC -1.0), rg 583 (2D 3-hop -5.8 / -3.0, VC1 2-hop
      -1.2), rg receiver 755 (one-ERISC -2.9..-3.0), kim 816 (inert on one-ERISC VC2 rows, single pass), kim 713
      (one-ERISC VC2 -1.4..-2.0); keep: rg 605 (2D 3-hop 256 B +2.2), kim 755 (+6..+9 forwarding, +7.7 one-ERISC);
      mixed, lean keep (user decides): rg 673 (-0.8..-3.0 except VC1 2-hop 4352 B +2.0), rg 414 (one-ERISC -2.6,
      VC1 2-hop +4.3). Causes: hypotheses (each A/B isolates its site, not the mechanism). Tables bfc/audit/auditC/D/E.
    - Joint check of the five drops (user: allocation and scheduling aren't additive), arm drop5 = 537 + 583 +
      receiver 755 + 816 + 713. Offline (bfc/audit/drop5_gate.txt): changed functions = the union of the singles' in
      every image; no inlining change beyond the singles'; pass-loop instruction / stack load / store counts = base +
      the singles' deltas in 90 of 91 (config, ERISC, function) groups; bh_vc1_1erisc E1 single::sender_pass 918
      instructions vs 920 predicted (same 6 / 6 stack loads / stores). Device: job F (base, drop5, base, drop5, base;
      the union of the arms' rows plus torus Y 3-hop and the one-ERISC VC1 split) and optional job G (singles on F's
      rows they never ran on); report drop5_report.py; accept only if no row is slower beyond max(noise, 0.5).
      **Job F (789, every run passed): no row slower.** drop5 vs base (3 base runs, 2 drop5 runs, noise <= 0.4):
      2D 3-hop and torus Y 3-hop E/W -5.8 / -3.0, one-ERISC 2D -3.9..-4.0, one-ERISC multicast -5.0,
      one-ERISC VC2fab -5.1 / -3.1..-3.7, one-ERISC VC1 split -7.7..-11.0, VC1 split 2-hop -1.1, VC2fab -0.5..-1.4;
      2D 1-hop / bidir within 0.3. Where every single arm was measured, drop5 = the sum of the singles within 0.1;
      elsewhere drop5 is faster than the measured singles' sum, the difference being unmeasured singles (job G
      not run; the coordinator holds it unless a row is slower). Table: bfc/audit/drop5_F.txt.
    - **User decision: the five drops.** Commits on the cleanup branch after c7a4d0fb29f: 993c99a564b (537),
      aa11a8786df (583), 82ea91f44a4 (receiver 755), 5fb378a7880 (816), bc0108869e2 (713; after_step removed),
      22cdb1e55c3 (comment only: measured reasons at the kept sites 605, 673, 414, sender 755). Gate
      (bfc/audit/drop_commits_gate.txt): each commit changes exactly its single arm's functions in 5,010 / 5,010
      images; the comment commit 0 images; 22cdb1e55c3 = a_drop5 in 5,010 / 5,010. Farms removed; validation
      tree at 22cdb1e55c3 (router headers only since 7cd3c13a0db, no host rebuild). Val set: broker 790 two, 791 one, 792
      inline_udm, 793 vc1 (VAL_TAG=drop5).
  - Cleanup batch: item 1 **3860614b218** (has_vc1 = ACTUAL_VC1_SENDER_CHANNELS > 0; FABRIC_2D_VC1_ACTIVE gone from
    host and kernel). Precondition: define present exactly when ACTUAL_VC1_SENDER_CHANNELS > 0 in 16,543 router
    kernels of 15 captures (bfc/audit/vc1_active_check.txt); the CI captures without caches are covered by the host
    code (both from vc_shape_.sender_counts[1]). Gate: offline build with the define stripped from the captured
    commands, 5,010 / 5,010 identical to 22cdb1e55c3. Host rebuild + val pending (after 790-793).
    463f35e3eb0: WH_BRINGUP.md wording (the receiver pass's first reads). Drop5 val 790-793 clean; coordinator pushed
    22cdb1e55c3. Items 3 / 5 / 6 / 4 (user-approved design; vc_forwards_to_routers renamed any_forwards_on_vc):
    44ad66cba15 (item 3: 262 E1 images differ, the predicted speedy-path and VC0-crossover receivers; VC0 downstream
    init / preset gone, VC1 presets the shared pair, loop changes are memory offsets only), d79db3f2600 (item 5,
    identical), 22ef990288c (item 6; 12 bh_2d_telemetry E1 images reorder 4 functions, layout only; a first
    version failed to build without VC1: ForwardCommands<1> named in a non-template's discarded branch, fixed with
    preset_forward_pair<VC>), e5b75d1fbcd (item 4: validate_router_config in get_compile_time_args, 12 TT_FATALs,
    kernel keeps 5 invariants; identical images). Gates in bfc/audit/item*_gate.txt. Next: host rebuild with a
    temporary force patch for the 12 forced mocks, unforced mocks of every case, P150 val (cl1), Galaxy P10.
    Done 2026-11-04 22:05-23:31: validation tree at e5b75d1fbcd, forced mocks 12/12 PASS (bfc/audit/forced_mocks.txt),
    patch removed, git clean, full build + install; unforced mocks of all 103 cov.py cases: no support-matrix
    TT_FATAL, no -DFABRIC_2D_VC1_ACTIVE in 19,843 router compile lines (glx3_torus_yx_vc2_* stop at the known
    VC1+VC2 ring/torus TT_FATAL). Glx tree at e5b75d1fbcd, built, staged as mainline_e5b75d1 (verify OK; rsync
    58 min, ~1.4M inodes). Galaxy dry-run: allocation failure while every bh-glx-110-a* node is allocated.
    P150 cl1 lines and the Galaxy --go line sent to the coordinator.
    P150 cl1 (broker 794 two, 795 one, 796 inline_udm, 797 vc1): every run 0 failed, no hang capture; router
    compile lines carry no FABRIC_2D_VC1_ACTIVE. Galaxy G-PORT P10-mm-4x4 / 2x8: Slurm 176161 (pending at
    23:35). Pass: per step multi-mesh stability 104 and pass-through 48 tests on 3 ranks, 0 failed, no size error.
    176161 (bh-glx-110-a09u02): both steps HUNG (600 s) right after UMD "Starting devices in cluster completed",
    before main()'s first log ("Relative fpath", CmdlineParser), so before any fabric build or router code; no
    rank stderr / TT_FATAL. Same signature as TEST_PLAN §4.5's ENV tt-run stall (a03u14, a08u20). Hypothesis:
    node a09u02. Node A/B proposed: cb42eb7 P10-mm-4x4 pinned to a09u02 (and optionally e5b75d1 on a04u14).
    Rerun 176235 (e5b75d1fbcd, bh-glx-110-a10u14): P10-mm-4x4 PASS and P10-mm-2x8 PASS (each: 104 + 48 tests on
    3 ranks, 0 failed, no size error); covers the cleanup batch and drop5 on Galaxy multi-mesh. Node A/B 176236
    (cb42eb7, the morning's PASS tree, pinned to a09u02): both steps HUNG with the same signature, so the node.
    a09u02 added to the standing excludes (testplan manifest and the port copy) and to TEST_PLAN §4.5's ENV entry.
  - Coordinator pushed the cleanup batch: origin 22cdb1e55c3..e5b75d1fbcd (2026-11-05).
  - Route decode (user-approved; lb2 176226): **6dad5fec638** on the cleanup branch, decode_action_y_first reads both
    bytes then selects. Gate 1 (credit capture): 1,047 / 5,010 images differ, all E1 2D Y-first receivers (N/S,
    Z on a Y express axis) and intermesh landings; E0, 1D, E/W and X-axis Z identical; functions receiver::run /
    single::receiver_pass (+ _start layout). Gate 2 (the A/B's 616 decode-on builds rebuilt on 6dad5fec638): 508
    byte-identical; the 108 others differ the same way decode off vs e5b75d1fbcd (the cleanup batch's function
    order and VC0-crossover receivers), and the decode change itself is identical at both bases (after normalizing
    the downstream-state offsets item 3 moved, in the VC1-split receivers). Files bfc/audit/decode_*.
    **Pushed (coordinator, 2026-11-05): origin e5b75d1fbcd..6dad5fec638.** Its functional validation rides with the next
    batch's P150 val set (open).
    - c7a4d0fb29f on the cleanup branch: comment-only fix of deliver_other's comment (user nit); gate 5,010 / 5,010
      router images identical to 7cd3c13a0db (bfc/audit/dlvcmt_gate.txt).
  - Gates (2026-11-04, offline full builds of the c5 set, per-function compare with branch/call targets normalized):
    16260ea5bff passes (only teardown / run_router_command / wait_for_connect_request, `_start` immediates, and
    layout). coldwait (fba07604d81, cold on the library's wait_for_notification) FAILED: it changed the single-ERISC
    loops of 136 Tensix-mux / UDM E1 images (receiver_pass -16..-59 instructions) for 160-224 B; **dropped**
    (coordinator). Landing option 1 changes receiver::run's register allocation in 44 glx3_passthrough E1 images
    (frame 240 -> 224 B, same calls); kept as a codegen side effect of a source change (coordinator, per the user's
    rule). IPA-RA as the cause: **WRONG** (refuted: an offline -fno-ipa-ra build shows the same change).
  - Later option (not now): `cold` on the router's own startup wrappers (handshakes, the downstream-mux wait) instead
    of the library's wait_for_notification; would need its own gate (loops unchanged in mux / UDM builds).
- **Code size (user, 2026-11-04): cold attribute on run_router_command / teardown / wait_for_connect_request and on
  the library's wait_for_notification, plus landing option 1 (multicast landing rebuild out of line, cold).** That fits
  every CI multi-mesh config at -O3 (2x8 pass-through 896 B left, a line main fails). The -Os-for-VC1 opt-rule commit
  is dropped; -Os stays only for VC1+VC2. Known gap: VC1+VC2 2x8 pass-through 688 B over even with the init/open
  merge (non-CI), left for later merges. Validation: the P150 set, an inter-mesh multicast perf row, Galaxy P10-2x8.
- **Code-size gates (2026-11-04):** cold x3 (16260ea5bff) passes: loops identical once branch/call targets are
  normalized. Cold on the library's wait_for_notification (fba07604d81) changed single-ERISC loop code in 136 mux/UDM E1
  images to save only 160-224 B, so it is dropped (later option: cold on the router's own startup wrappers). Landing
  option 1 changed receiver::run's register allocation in 44 glx3_passthrough E1 images (fewer instructions and
  spills); hypothesis: IPA-RA clobber sets, checked with -fno-ipa-ra. It is kept as a codegen side effect, not a perf
  claim. CI size re-measured without coldwait (2x8_pt about 736 B left).
- **-Os for VC1+VC2 stays** until the cold-path stage lands (at -O3: P150 1,456 B left, Galaxy Z-ring 608 B left,
  dual-Galaxy / 3-mesh split 400 B over, pass-through 1,424 B over). Cold-path stage: design first (cold attribute on 4
  E0 functions, set_landing_route shape, template merges), offline sizes, then the switch to -O3 everywhere.
- **TT_FATAL for VC1+VC2 on ring/torus routers** (3 receiver channels x 8 IDs > 16): a known gap until the
  per-channel transaction-ID allocation lands.
- **API audit (porting agent):** drop the inline-write cmd_buf / emulated_write_on_cmd_buf API change (stock call +
  local_write_init) and the SKIP_SRC_CH_ID_UPDATE host arg; keep the node-addressed credit write, the one-check flush
  and the ct-arg arrays. Removal A/Bs A1 (skip flags), A2 (node-addressed data write), A3 (header load/unpack), A4
  (merged credit frames), A5 (credit write's share of the partial-preset gain), all approved.
  - A/B runs (2026-11-04): P150 broker 711-714 (a15a / a34a / a15b / a34b); Galaxy N/S Slurm 172435 (trees
    a11{ns,ww,cs}_1104 = v4txa11_1104 + the arm's file). **A5 (v4txa11cs, stateless credit write) HUNG on the Galaxy**
    (2d2k_a_cs at SlsMulticast 2048, node a10u14): the capture shows over-credited E1 copies (45 > 32 slots) and lost
    credits upstream, i.e. credits landing on the wrong stream where a router has several downstreams (hypothesis: the
    stateless write disturbs command-buffer state the stateful writes rely on). A5 as written is ruled out (hang rule;
    decode: scratchpad dec172435.txt, capture /data/aagarwal/frs-glx/jobs/172435/results/2d2k_a_cs_hangcap). The decode
    is also the reason the node-addressed inline credit write stays: the stateless replacement is not functionally
    equivalent on multi-downstream routers, so the stateful form is required, not only faster. P150 jobs re-queued
    without cs / csml: 717 (a15a), 718 (a34a), 719 (a15b), 720 (a34b); Galaxy pass b (ns / ww only) waits for a slot.
  - **P150 results (717-720, 2026-11-04; passes a / b reversed, every run 0 failed, no hang, passes within 0.2;
    cycles/packet, arm vs v4txa11; positive = the arm is slower, i.e. the port's form is faster):**
    - A1 ns (skip_cmdbuf_chk off): 1-hop rows and ring1 0; lin3 +10.6 (+8.5%); VC1 2-hop +4.0 / +2.5 (16 B / 4352 B),
      vc1 mean +0.4%, inter-mesh mean +1.2% (to mesh 1 chip 3, 16 B +11.8); ml proxy 3-hop E/W +19.7 (+13.6%), turn
      +12.1, row merge +6.8, multicast +18.4 / +23.3, 1D lin3 +16.0.
    - A2 ww (main's noc_wwrite_with_state): 1D / ring / VC1 0 (largest +0.6, inter-mesh 16 B); ml proxy +0.1..+0.4
      (3-hop 4352 B +0.4 = +0.3%, multicast +0.4, row merge +0.3). Galaxy pass a: N/S +0.7..+1.0.
    - A3 hdr (header load/unpack): ewscatter +6.8 (+4.8%), ring1 +5.9 (+4.2%), torusns +6.4 (+4.3%), 2D multicast
      +0.2, ring multicast -1.8 (-1.2%).
    - A4 cr (one credit frame per event): ewscatter +2.3 (+1.6%), ring1 +12.6 (+8.9%), torusns +14.3 (+9.6%),
      2D multicast +10.3 (+7.3%), ring multicast +3.0 (+2.0%).
    - Tables: row_ab_report.py a15a|a15b|a34a|a34b|a34am|a34bm; census_report.py --routers=v4txa11,v4txa11ns,v4txa11ww
      --passes=a15a,a15b --rows vc1 vc1i; loopvars_maps/ns_presets/slsab.py a15a,a15b v4txa11ml,v4txa11nsml,v4txa11wwml.
    - Galaxy N/S (Slurm 172435 pass a: a11 / ns / ww 0 failed, cs hung; 172794 pass b: every step 0 failed;
      sls2k_report.py <job>/results 2d):
      2 KiB slots, both passes agree: A1 ns +16..+23 cycles/packet on N/S forwarding rows (3-hop N/S 136.8 -> 159.4 /
      159.6), A2 ww +0.7..+1.0 (3-hop N/S 136.8 -> 137.8). 4352 B slots (172794, passes a / b): A1 3-hop N/S 1 KiB
      152.5 -> 170.1, turn 145.2 -> 165.6; A2 +0.4..+0.8. 4352 B payloads are wire-bound (252.1) for every arm.
- **HANG in the cleanup batch's VC1 validation (job 660, 2026-11-04):** vc1_vc2 (P150 split, VC1+VC2 at -Os) stalled at
  MultiMeshAllToAll_size_4096 unicast_write; rank 0 never finished; the router decode shows idle, empty routers
  (hypothesis: a worker or host-side wait). The pushed 58a9a1e passed the same set (647). A/B: broker 669 (batch tip
  x3) vs 670 (pushed tip x3). The push of b2473363016..4db4f45790b is on hold until this is understood. A fresh porting
  agent took over (the previous one ran out of context).
  - Worker decode (2026-11-04; hang_workers.py ported to .scratch/val with main's Tensix L1 bases; capture dir
    results_660_os/mval_vc1_vc1_vc2_hangcap/decode_workers.txt, slot_map.txt): rank 0 waited on its 16 inter-mesh
    receivers (tensix (1,5)-(1,8) on chips 0-3), each at validated 0/100, polling packet 1. Their rank-1 senders sent
    100/100; router counters agree; every channel empty. All 16 rings hold packets 65-100 in slots 1-36 and 37-64 in
    37-64, every word correct: everything arrived, and packet 65 overwrote slot 1 before the receiver saw packet 1. The
    yaml allows this: 100 x 4096 B into a 64-slot ring, no receiver credits, start sync per mesh only (nothing
    holds rank 1's senders for rank 0's receivers; only the host barrier aligns the ranks).
    The overrun is observed; "rank 0 started late" is a hypothesis (cross-rank launch skew): a harness race in our own
    vc1_sanity.yaml, not a router fault. 669 and 670 both passed 3/3, so no bisect. Sleep-on-one-rank A/B skipped
    (coordinator). Fix applied: MultiMeshAllToAll num_packets 100 -> 64 (slotalloc/cap_ring_packets.py; mock run parses,
    compiles, launches). Next: `val_runs.sh vc1` with the validation worktree at 4db4f45790b (push range, VC1 at -O3);
    clean -> pushed. Done: 674 clean, 4db4f45790b pushed (2026-11-04).
- §5.2-5.4 cleanup commit (5206b9a57da): pushed with 4db4f45790b; its 14 channel-trimming-capture E1 image changes
  appear only when three of its edits combine (stated in the commit message).
- **NoC API: extend main's functions, add none (user, 2026-11-04).** The inline credit write extends
  noc_inline_dw_write_with_state / noc_fast_write_dw_inline_with_state (both address updates, node coordinate as an
  extra argument; the narrow *_to_node_* inline forms go; identity-gated). The data write uses main's
  noc_wwrite_with_state if A2 (one extra RET_ADDR_MID store per forward) shows no cost; otherwise an option inside
  wwrite. noc_write_to_node_with_state is a copy of wwrite minus the MID write and goes either way.
- Porting agent's gate tool (cov.py --alt-tree) could build the wrong worktree: the pushed cleanup gates are valid;
  the §5.2-5.4, (b) and (c) gates were WRONG and are being redone.

## To explore (user ideas, not started)

- **A second 2D routing VC with the mirrored dimension order** (user, 2026-11-03). Today 2D routing is one dimension
  order on one VC. A second routing VC running the flipped order (YX next to XY, or the reverse) would give traffic
  two path families to balance load across 2D meshes: e.g. spread flows between the two orders per packet, flow or
  destination, so hot rows and columns share the load.
  - Questions to work out before any design:
    - deadlock freedom (each VC stays dimension-ordered on its own, and packets don't switch order mid-route, or
      switch only in a safe direction);
    - how the sender picks the VC (host policy, header bit, worker API);
    - ordering guarantees for flows that need them (pin a flow to one order);
    - buffer/slot cost of another forwarding VC (code space and slots; the VC2 line idea and native fan-in compete
      for the same room);
    - interaction with torus deadlock avoidance (bubble rule per ring per VC) and with the ring-jam fix.
  - Related: VC2_LINE_IDEA.md (a second forwarding VC), native fan-in / multi-VC (report section 5), and the
    multi-VC measurements.

## Decisions pending with the user

- **Slots** (SLOT_DECISIONS.md):
  - 200G default slot size (3072 B with measured allocations recommended);
  - VC1 policy on 400G (today's default plus a host setting recommended);
  - when to write the allocator design note (mesh size, packet size, link speed);
  - a burst-aware worker step.
- **Open, not chased (placement audit job D, 786, 2026-11-04):** on the P150x8 split into two 2x2 meshes (VC1-enabled
  routers, two ranks), VC0 traffic inside mesh 0 runs 175.2 / 193.2 cycles/packet for 1 hop and 425.8 / 478.8 for
  2 hops with a turn (256 / 4352 B; the port at 7cd3c13a0db). On the plain 2x4 mesh the turn row runs 143.5 / 144.9
  and 1 hop 120.3 / 128.0. Not compared like for like (different mesh shape, VC1 on, two processes); no cause
  claimed. Rows: .scratch/audit/yamls/D_base_vc1.yaml; results pauD_*_vc1.

## Tooling and housekeeping

- 2026-11-04: root reached 100%. Deleted 228 JIT caches (~148 GB) that no recorded command, note, cmds.json or
  yaml under fabric_exp referenced and that had been untouched for 2+ days; the list is
  consolidation/deleted_caches_20261104.txt. Saved ELFs in consolidation/builds and loopvars_golden still hold debug
  paths into some of them, but the ELFs are self-contained. Root is back to about 200 GB free.


- Bubble flow control's 2 as one shared host-device constant (V4_SENDER_OWNERSHIP §2 #6; decided): deferred from the
  sender pass (2026-11-02). It lives in hostdevcommon/fabric_common.h, which another agent's study jobs swap; doing it
  needs a host change to that file and pinning it in every variant.

- alloc_model.py row map and fabric_exp/vc1/vc1_mix.yaml assume the pre-merge placement (fix; sls_vc1_all.yaml has
  the corrected rows).
- offline_build.py should snapshot or pin tt_metal/hw/inc. Offline builds read it from the live tree, where a
  broker job's swap can change it. Agents pin by hand today.
- V4_SIZE.md: a per-config code breakdown (loop / init once / per-VC copies), to track what grows with the 1D
  line-VC work.
- Test infra bug: a mux and a sync core both connect to one router channel (Galaxy PerfAllFlows hang,
  VC2_LINE_BRINGUP.md phase 4). Report upstream eventually.
- **Test infra, not the router (2026-11-05/06; coordinator):**
  - **Flow-controlled ring all-to-all with atomic_inc hangs on main and on the PR.** FlushA2ARing_size_4096_ntype_
    atomic_inc_ftype_unicast (pr_perf/flushda/hangab_glx_ring.yaml: all-to-all, enable_flow_control, 200 packets) hangs
    on Galaxy:
    - PR base: Slurm 180570, a10u08. Main d2bd12fcf1f: 180726, a08u02.
    - The decoded end state is identical (results/glx180570/a_unp_val_ring_decode.txt, results/glx180726):
      - every router channel empty;
      - 670 sender configs at 128/200 with no receiver credits;
      - 232 "sent all 200, not arrived";
      - receivers' atomic counters sum to 512.
    - The P150 1024 B form hangs the PR base the same way (lb2 180530).
    - Stock yamls run this pattern only on Linear/Mesh, and on Torus in the 2D torus stability yaml.
    - Never run flow-controlled full-ring rows on a LoudBox.
  - **Data-checked explicit 1-hop ring rows mismatch at large sizes.** DATA_MISMATCH at receiver (M0, D1) core 0-0
    for a sender device [0,0] → destination [0,1] on Ring (P150, a09u32 broker):
    - FlushRing1HopEW_size_4096_ntype_fused_atomic_inc without flush: the PR base and main both fail (851, 852).
    - Ring1HopEW_size_3072 unicast_write (benchmark off): the tip (861) and main (874) both fail.
    - The same form at 1024 B passes, and so does atomic_inc at 4096 B.
    - Cause open (hypothesis: the test infra's data check on this pattern).
- Candidate upstream issue (for the user): main's test_fabric_multi_mesh_sanity_common.yaml MultiMeshAllToAll runs
  num_packets 100 at 4096 B with sync on and no flow control: a 64-slot receiver ring, and the start sync is per mesh,
  so one rank's senders can overrun the other rank's receivers (seen in the port's job 660; PARITY.md "Device
  validation notes"). Fix there: cap num_packets at the ring's slots or enable flow control.
- Optional outside the study: `tt_buf_rw::note<>` always_inline would recover about half the -Os gap for every -Os
  kernel (job 367). A suggestion for the API owners.
- Uncommitted since checkpoint 12 (main tree): host changes (builder: per-VC args, worker-role arg, VC1 -O3; worker
  adapters template fix), the profiler aggregation, the microbenchmark, coverage.py, notes, scripts, variants and
  results. Commit only when the user asks. Never commit Fabric_instrumentation_analysis.md.
- Cache cleanups: check references first (memory feedback-cache-cleanup-check-refs; the must-stay list is there).

## Done recently (kept for orientation)

- **Same-job latency, main vs the receiver lead** (2026-11-02, jobs 462-463, lat_ab_runs.sh / lat_ab_report.py):
  1 hop -28 to -82 ns, 3 hops -166 to -212 ns (~55-70 ns per hop), both passes agree; main at keepalive 1000 within
  16 ns of main as shipped (keepalive does not move latency). Report data: report/data/latency_samejob.json.

- **Receiver stage g1-g3 (2026-11-02): v4rxg3 is the receiver lead** (V4_RECEIVER_OWNERSHIP §18). g1 renames and
  asserts: 2,713/2,713 byte-identical; g2 ReceiverChannelStates owns the rule: layout-only (all diffs from the
  downstream field swap, classified); g3 derived loop_local_vc: only the 84 images of four new VC0-trimmed mock captures
  change (build checks; the far senders still send on VC0). 13/13 incl. trimming gtests both ERISC modes (job 447);
  headroom and size equal to v4rxf4. Full coverage is now 3,124 configs (2,863 build).

- **Receiver pass closed (2026-11-02): v4rxf4 is the lead** (ownership pass, inline-write fix (ii'), buffer-2 fix (c),
  S2-explicit placement, naming polish, S4, codec decode with its cost accepted, comment trim). Census, 13/13, UDM,
  inline-write gtests. V4_RECEIVER_OWNERSHIP.md §17.

- Inline-write hang fix (ii'), stage 0, validated (job 385). Validation now includes the inline-write and
  flushed-atomic gtests.
- Host rebuild (job 399): ENABLE_FIRST_LEVEL_ACK_VC2, VC2_CROSSOVER_TO_VC1, SENDER_CH_<i>_IS_WORKER_CHANNEL (rule to
  fix, above), VC1 at -O3 for the study router.
- Slot study P1-P3, 2 KiB and 3-3.5 KiB on Galaxy (SLOT_DECISIONS.md). N/S floor cause found (V4_NS_FLOOR.md).
- VC2 line phases 1-4 on Galaxy; 1D line-VC census and host side; committed and pushed on aagarwal/vc2-line
  (8abd80861f7, 0d76a021a70).
- Register-map guard loopvars.py (self-tested, golden records, --live, --headroom).
- Bound map, command-buffer audit, TX-queue probe (V4_BOUNDS.md); send batching study (V4_SEND_BATCH.md); VC1
  opt-level A/B (job 367).
- 2026-11-05 JIT compile errors in multi-rank regress steps (3 ranks, one shared warm TT_METAL_CACHE per job, as CI on
  one host): "get_named_compile_time_arg_val not declared" (named_ct_arg_map_generated.h) and "kernel_main not
  declared" (kernel source wrapper), in 178007 / 178014 pass-through and 2x8-again commands. Harness noise, not router.
  Known class (user's Glean search): tt-metal #46925 (fix d090a3934eb on blaze-metal branches; the same atomic
  named_args_generated.h write is already in our base, genfiles.cpp:701-768, so not this one), #29164 (per-process
  caches as workaround), tt-xla #4453, #34014, #39858 (old race not reproduced on main in 10 cold-cache runs; ours is
  warm-cache). Mechanism open: generated headers are FileRenamer-atomic in our base. Optional: add repro data to
  #39858 / #29164 after the user's OK.
- 2026-11-05 slot allocator cleanup eebd9e9e320 (pushed to origin/aagarwal/fabric-router-mainline): the allocator returns numbers only (no
  notes / reason strings), enchantum for ChannelKind, RouterShape::is_2d from is_2D_topology(), a per-VC depth table.
  Identical counts/stages over 92,160 shapes (scratchpad alloc_eq/). Study tool slotalloc/slot_alloc_table.cpp still
  uses notes / reason / one_d: update it before its next use. The TRID allocator (fabric_trid_allocator.cpp,
  trid_detail::describe -> reason, logged in erisc_datamover_builder.cpp:1499) has the same string pattern: offer the
  same treatment.
