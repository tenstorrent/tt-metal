# Router study TODO (started 2026-11-02)

The main session keeps this file current. Each item carries its source doc. Order within a section is the intended
order. North star: readability first (for placement decisions: readability, then stability, then perf). Every code
change gets identity checks, the register-map guard and its headroom check (loopvars.py), and an A/B wherever codegen
changes.

## In progress

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
  - **TEST_PLAN.md DONE (559 lines; for user review).**
    - Two modes: the shared commits with main's router (byte-identical expected) and the router commits with the new
      router. CI exercises the new router only via a throwaway branch that makes it the default.
    - Perf: fabric goldens only for P150x4 / T3K / WH Galaxy (10-22% tolerance, both directions); model targets in
      models/model_targets.yaml (15%). The main-vs-new thresholds (1% fabric / 2% model) are proposals.
    - Gaps: exabox is BH-only (WH is CI-only); P300 has no coverage; BH Galaxy has no fabric BW golden; Nightly fabric
      fixtures don't run on BH boxes in CI; the TG perf scripts are in no CI list.
    - **Questions for the user:** may we use bh_qb2 / bh_lb_single; can a 1-card QB2 stand in for P300; telemetry
      on (CI default) vs off for main-vs-new perf.
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
  - Strategy (proposed to the user, pending): checkpoint the study branch (no merge of main); a mainline worktree on
    latest main; baselines on main right away; port commits 1-5 with bytecmp gates; the main-config parity check.
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

## Tooling and housekeeping

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
