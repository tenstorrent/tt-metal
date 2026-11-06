# Dynamic slot allocator, step 1 (2026-11-03)

Status: prototype on the study branch, off by default. It replaces main's tier tables with a few readable rules and
later becomes mainline commit 6.
- The design was agreed with the user on 2026-11-03 (TODO.md "Dynamic slot allocator"). The user's step-1 decisions,
  relayed by the coordinator the same evening, set the rules in section 2.
- Built offline: the pure function and the offline table.
- Not yet in the host library: `host_rebuild_dynslots.sh` must run as a broker job. My own run of it was refused by the
  permission system.
- No device data yet: the A/B is planned (section 8). The receiver multiple-of-4 sweep is ready (section 9).

Data behind the numbers: SLOT_STUDY.md (P1-P3, the 2 KiB and 3 KiB Galaxy sections), SLOT_DECISIONS.md and
ALLOCATION_NOTE.md. Labels used here: measured (job), hypothesis.

## 1. The function and its inputs

The function is `allocate_slots(RouterShape, Policy)`. It returns slots per VC and channel kind, each with its reason.
- It lives in tt_metal/fabric/builder/fabric_slot_allocator.{hpp,cpp}.
- It is pure and uses only the standard library, so the offline tool compiles the same file.

RouterShape holds only facts the host knows without measuring anything:
- the channel buffer bytes and the slot size;
- per VC: its worker, forwarding and receiver channels;
- the mesh dims;
- whether the router is 2D (mesh or torus), for the VC0 receiver floor (section 10).

There is no perf number, no link speed and no optimizer.

Step 1: the host fills one shape for the whole fabric.
- The mesh dims are the largest of its meshes, read from the control plane.
- VC0's worker-connected sender is its worker; its other senders forward.
- VC1's senders all forward.
- VC2's sender is its worker.

## 2. Rules (the user's decisions, 2026-11-03)

1. **Work in slots.** N = floor(channel buffer bytes / slot size) is the only place bytes appear. Caps are slot
   counts.
2. **Split N between VCs by an integer ratio that depends on the VCs built.** Each VC's share is N x its ratio over the
   ratio sum, rounded by largest remainder (ties go to the lower VC).

   | VCs built | ratio |
   | --- | --- |
   | VC0 only | all of N |
   | VC0 + VC1 | 3:1 |
   | VC0 + VC2 | 12:1 |
   | VC0 + VC1 + VC2 | 12:4:1 |

   "VC1 first" is the ratio 2:1. One override key, `vc_ratio=a:b[:c]`, sets the ratio for the VCs built.

   Every channel's minimum comes first (2026-11-04, section 10): a VC whose share is below its channels' minimums is
   raised to them, one slot at a time from the VC with the most above its own. It changes only rows that were an
   error before (max payload with VC2 or a 4x4 VC1 split).
3. **Kind weights inside a VC are integer ratios.**
   - Worker 2, forwarding 1.
   - Forwarding counts x2 when a mesh dim has more than 2 chips (only there can traffic pass straight through).
   - A receiver weighs 4/5 of its VC's heaviest sender kind.
4. **Order of allocation, per VC.**
   1. Receivers first. A VC0 receiver takes the power of 2 nearest its share, by ratio (`receiver_round=pow2`, the
      default), or with `receiver_round=mult4` the nearest multiple of 4. VC1 and VC2 receivers take their share,
      rounded down. On a 2D router the VC0 receiver's share is at least 1/5 of N before the rounding
      (`vc0_receiver_floor=1:5`, user 2026-11-04, section 10).
   2. The senders split what is left by weight, rounding down. Any channel below its minimum is raised; if that takes
      the VC past its share, the slots come back from the sender kind with the most above its minimum.
   3. Caps: the VC0 worker's `worker_cap_slots` (off by default; section 5), and 127 slots per channel.
   4. The leftover from every VC goes to the VC0 worker, up to its cap. Anything past the cap goes to VC0 forwarding,
      one slot per channel at a time.
5. **Minimum 2 per channel.** A ring's VC0 receiver keeps the bubble rule's 2. The K + 2 ring-jam minimum was dropped
   (user).
6. **Log every count with its reason.** An explicit slot switch still wins:
   - `TT_FABRIC_EXP_SLOTS`, `_1D`, `_2D`, `_VC1`, `_VC2`;
   - `_TIER_SLOTS`, `_WORKER_SLOTS`, `_MAX_SENDER_SLOTS`, `_MAX_RECEIVER_SLOTS`, `_LINEAR_SPARE_SLOTS`.

## 3. The fit (offline, `slotalloc/slot_alloc_table.sh`)

Today's measured 4352 B rows all stay the same: 2D 17,16,16, 1D 26,25,32 and the VC1 split 19,9,16,5,4.

How the fit was found: a brute-force search over the integers, with the exact rules above
(scratchpad `fit/search.py` and `fit/alloc_v3.py`).
- **3:1 needs largest-remainder VC shares.** With floored shares, VC0 gets 60, which cannot hold 16 + 18 + 3 x 9; only
  13:4 or 16:5 then keep the VC1 row.
- **No single receiver weight serves all VCs.** VC0 needs its receiver at 1.5x or more of forwarding (to round to 16 in
  2D and to 32 in 1D). VC1 needs it below 1x of forwarding (ring 4 against forwarding 5). The weight "relative to the
  VC's heaviest sender" covers both: 3/4 to 7/8 all fit, and 4/5 is the middle. 1:1 does not fit with 3:1.
- Worker 2 : forwarding 1 with the x2 bump on long dims gives worker ≈ forwarding (2D 17 / 16, 1D 26 / 25) and the VC1
  split's 19 / 9 on 2-chip dims.

Rows that move (default policy):

| config | 4352 B: today -> dynamic | note |
| --- | --- | --- |
| VC1 first (2:1) | 18,7,16,7,5 (measured pick, job 388) -> 18,7,16,7,5 | unchanged |
| VC2 on (12:1) | 17,14,16 + 2,4 -> 17,14,16 + 4,2 | VC0 the same. VC2 worker 4 / receiver 2, against main's and the study row's 2 / 4. |
| three VCs (12:4:1) | 19,8,16,4,4 + 2,4 -> 17,8,16,5,4 + 3,2 | the study row was never measured |

At 3072 B:

| config | main | study router today | measured best | dynamic before the floor | dynamic with the floor |
| --- | --- | --- | --- | --- | --- |
| 2D | 35,16,32 | 35,16,32 | 20,21,32 | 27,24,16 | 23,20,32 |
| 1D | 32,32,32 | 43,42,32 | - | 43,42,32 | 43,42,32 |
| VC1 split | 85,4,8,2,4 | 53,9,16,5,4 | 20,15,16,9,7 | 30,14,16,7,6 | 26,10,32,7,6 |
| VC2 on | 77,7,11 + 2,4 | 51,14,16 + 2,4 | 20,14,16 + 16,16 | 24,22,16 + 5,4 | 20,18,32 + 5,4 |

## 4. `receiver_round=mult4` as a one-line swap

At 4352 B, mult4 changes three rows; the VC1 split is unchanged (19,9,16,5,4):
- 2D: 17,16,16 becomes 18,17,12;
- 1D: 26,25,32 becomes 30,29,24;
- VC2 on: 17,14,16 + 4,2 becomes 18,15,12 + 4,2.

A receiver of 12 sits between the measured 8 (binds every row, ALLOCATION_NOTE §4) and 16 (binds none). So the swap
would need its own receiver fit, and it waits for the section 9 sweep and the user's decision.

## 5. Limits and options (for the user)

1. **The worker's need is per slot.** Small-packet 1-hop rows are router-bound, and the worker saturates near 20 at
   every slot size. Fixed shares of N scale it up with N anyway.
   - Cost: at 3072 B the VC1 split's VC0 forwarding gets 14 where the measured best had 15. Hypothesis: the 2-hop row
     is off the wire by a few %.
   - Option `worker_cap_slots=24`. 24 is a pick, not a measurement; the A/B's `dynw24` arm against no cap decides the
     default. Step 4.4 sends the slots past the cap to VC0 forwarding:

     | config | no cap | `worker_cap_slots=24` |
     | --- | --- | --- |
     | VC1 3072 B | 30,14,16,7,6 | 24,16,16,7,6 |
     | 2D 3072 B | 27,24,16 | 24,25,16 |
     | 1D 4352 B | 26,25,32 | 24,27,32 (a 4352 B row moves) |
     | 1D 3072 B | 43,42,32 | 24,61,32 |

2. **VC2's ratio is a policy choice.** At 4352 B there is no room for VC2 beyond about 6 slots without costing VC0.
   12:1 gives VC2 4 / 2. The study's 3 KiB VC2 pick of 16 / 16 needs about 3:1, which would cost VC0 at 4352 B.
3. **Mixed rows are scored by link total** (user, 2026-11-03): 1 / sum(1 / per-flow cycles), with the per-flow numbers
   shown next to it for reading. A per-flow drop is not a regression unless the total drops (`dyn_report.py`).
   - Earlier draft (same day): it called the Galaxy 3 KiB GlxMixWorker change "a link-share trade". **WRONG by link
     total:** 24,15,16,8,6 against 53,9,16,5,4 is +2.4% at 3072 B (173.1 -> 177.2). The total improves at 256 B
     (-14.0%), and GlxMixForward improves at both sizes (-8.1%, -21.4%).
   - Caveat (hypothesis): flows of different speed that run for equal packet counts do not overlap for the whole run,
     so per-flow rates overstate the shared-link total. Matched-count rows (as mvc3_matched.py) are exact.
4. **VC0 + VC1 on meshes with a dim longer than 2 chips** (multi-mesh splits such as P4's 4x4, not in the current
   A/B): the forwarding bump shrinks VC0's receiver share to 10.2, which rounds to 8. The 4352 B row is
   15,13,8,5,4. A receiver of 8 binds every row (ALLOCATION_NOTE §4), and the fit never covered this shape: VC1 data
   exist only for 2x2 splits. It needs a fit, for example a receiver floor of 16 for a VC with a worker, before such
   splits use the dynamic slots.
   - 2026-11-04: the 2D VC0 receiver floor (section 10) gives 13,11,16,5,4. The VC1 counts on such splits are still
     unmeasured.
5. **Code space.** The three-VC build had 240 B left at -O3 (TODO.md, job 479), and VC2 8 / 10 grew it by 272 B. The
   dynamic three-VC row is a risk the code-size check covers.

## 6. Step 2: per router, the same rules

The builder calls allocate_slots once per router, with that router's own facts:
- its channels: trimmed ones are gone, and the Z / boundary family has 5 VC0 and 4 VC1 senders;
- the axis each forwarding channel serves, so the long-dim bump becomes per channel;
- whether it sits on a mesh edge.

RouterShape grows from per-kind counts to per-channel facts. The ratios and the order a-d do not change.

A link's two ends must agree on the receiver depth. Since the function is pure, the builder can also evaluate the far
router's shape.

Step 1 stays chip-wide because one FabricEriscDatamoverConfig (one allocator) serves every router today.

## 7. Files and switches

- `TT_FABRIC_EXP_DYN_SLOTS=1` turns it on, on any router, for fabrics without the tensix extension. Then
  `fabric_static_sized_channels_allocator.cpp`:
  - reads the mesh dims from the control plane, so no builder struct changes layout and the test binaries stay valid
    after a libtt_metal-only rebuild;
  - logs every count ("dynamic slots: VC0 worker x1: 19 slots: ...").
- `TT_FABRIC_EXP_DYN_SLOTS_POLICY=k=v,...` sets the policy. Keys: vc_ratio, worker, forwarding, long_dim_bump,
  receiver (num:den), receiver_round, vc0_receiver_floor (num:den of N, or off), worker_cap_slots, min.
- `TT_FABRIC_EXP_LOG_SLOTS=1` logs the final depths; the A/B scripts check them against the expected row.
- Host patch: `variants/host_dynslots/1_dyn_slots.patch` (that file and sources.cmake), built by
  `host_rebuild_dynslots.sh`, a broker job that restores on failure. The allocator's own files are already in the
  tree, and nothing builds them before the patch.
- `slotalloc/`:
  - `slot_alloc_table.cpp` / `.sh`: the offline table;
  - `mk_dyn_yamls.py` and `rxm4_yamls.py`: the 3072 B suites, in `yamls/`;
  - `dyn_p150_runs.sh`: the P150 A/B;
  - `dyn_glx.sbatch`: the Galaxy A/B;
  - `rxm4_runs.sh`: the receiver sweep;
  - `dyn_report.py`: per-row arm comparison, mixed rows by link total;
  - `dyn_size.py`: code size and stack per arm.

## 8. A/B plan (v4txa11, -O3, BW telemetry off)

Arms:
- main's tier pick (`TT_FABRIC_EXP_TIER_SLOTS=1`);
- dynamic (`TT_FABRIC_EXP_DYN_SLOTS=1`);
- `dynw24` where the worker cap changes the row;
- where they differ, the study router's default row and the measured best (explicit row).

Every run checks the logged depths against the table's row. Rows where dynamic equals main (2D and tori at 4352 B)
build the same binaries and need no device time.

Order:
1. Host rebuild (broker).
2. Galaxy base and v4txa11 trees.
3. Mock vet with the code-size check (CPU).
4. P150: one LB job with all arms (`lb2_run.sh -n any`, which needs a full re-stage after the rebuild), or broker jobs.
5. Galaxy passes a and b (when allocations are free).

Step 3 done (2026-11-04):
- Mock vet (`dyn_vet.sh`, tree v4txa11_dyn): every configuration and arm (26 checks) logs exactly the table's rows,
  including the VC1 and three-VC tt-run split cases.
- The VC1 tt-run case from the live study tree (coverage.py bh_vc1, two ranks) also logs 19,9,16,5,4.
- Code (2-ERISC, -O3, largest over the configurations):

  | arm | E0+E1 | of the 26,624 B buffer |
  | --- | ---: | --- |
  | tier | 22,056 | |
  | dyn | 22,224 | +168 B |
  | dynw24 | 22,124 | |

  Least free stack: E0 1,684, E1 4,252, unchanged.
- The three-VC build (VC1 + VC2) does not compile on v4txa11 at any slot count: router_common.hpp's static_assert,
  "VC2 is supported as neighbour exchange on 2D single-mesh routers". The 240 B margin note was the v4rxg3v probe's.
- An earlier checker run reported false MISSINGs on the split logs: `grep -q` on a pipe under pipefail. It is fixed:
  the logs are cleaned to a file first.

Submitted (2026-11-04):
- P150: lb2 171943, all arms in one job. It stopped before any device work on bh-lb-120-a08u28, whose /home has
  13 GB free (the job's floor is 25). It was resubmitted as lb2 171995, pinned to a07u28. It ran from stage src-20261104-001304 with `--stale-ok`, so the staged
  dyn_p150_runs.sh ran; the XY-torus suite was swapped for the Y-torus one (12 tests, mock-vetted) to stay inside its
  300 s hang timeout.
- Galaxy pass a: Slurm 171946. Pass b waits for room under the cap of 4.
- 2026-11-04 02:00: host rebuilt with the TRID patch and the VC0 receiver floor (broker 649). lb2 171995 was
  cancelled while pending (a07u28 held by another user's job) by the coordinator. After the full re-stage
  (src/full-20261104-020334) the P150 A/B was resubmitted as lb2 172043 (pinned to a07u28, `dyn_lb_ab.sh r2`: dyn and
  dynw24 run with the floor off, plus the dynfloor arm at 3072 B). Galaxy trees on the new host: base_1104trid,
  v4txa11_1104 (the next dyn round with dynfloor) and v4txa11id_1104 (TRID); both got fabric_exp/slotstudy (the
  split runs' rank bindings and the Galaxy yamls), as v4txa11_dyn has.
- lb2 172043 stopped before the command (after the reset, chip 6 eth (7,1) not up); nothing ran. Resubmitted as lb2
  172170 (`-n a07u28,a04u24`, same command and labels r2). Galaxy pass b: Slurm 172172 on trees/v4txa11_dyn (the
  pre-floor library, so no dynfloor arm, the same rows as pass a).
- Galaxy pass b (Slurm 172172, bh-glx-110-a08u14, 03:21-03:41; tree v4txa11_dyn, the pre-floor library): every
  step's slots as expected. Arms compared within each pass (the passes ran on two nodes); both passes give the same
  deltas to 0.1-0.2 points (out/glx_dyn_ab_171946_172172.md):
  - dyn and dynw24 are never more than 0.7% slower than main's tiers in any suite (1D, VC1 split, VC2 on at 4352 B;
    2D, VC1 split, VC2 on at 3072 B), and up to 23-70% faster.
  - VC2 on: dyn beats today's study default by 21.5% (4352 B) and 55% (3072 B) on the VC2 rows; VC0 rows equal.
  - 2D 3072 B against the measured best (20,21,32): dyn median 0.0%, worst +0.8%; dynw24 worst +1.6% (one turn row,
    pass b). Inside the 1-2% target.
  - VC1 split 3072 B against the measured best (20,15,16,9,7): confirmed by pass b, outside the target. The
    inter-mesh rows lose +6 to +17% (GlxInter 0_x -> 1_1 / 1_3 at 3072 B +15-17%), the mix link totals at 256 B
    +8.5-8.8%; median +1.1%. dynw24 has the same worst rows. Hypothesis (not isolated): VC1's 7 / 6 against 9 / 7.
    The VC0 receiver floor does not touch VC1: its rows keep VC1 at 7 / 6.
  - VC1-only isolating A/B (coordinator, 2026-11-04; explicit rows, no build): 30,14,16,9,7 does not fit (122 of
    115 slots; the host's TT_FATAL, mock-checked), so the one-variable pair keeps the worker at 23:
    23,14,16,7,6 (7 slots unused) against 23,14,16,9,7; 30,14,16,7,6 (dyn's row) anchors it, so A -> B is the
    worker 30 -> 23 alone and B -> C the VC1 counts alone. All three rows mock-vetted on trees/v4txa11_dyn (two
    ranks, launch reached, slots as given). Galaxy: Slurm 172240 (vc13k, two passes reversed in one job, excludes
    a01u14, a02u02, a02u08, a03u14, a08u20). P150: one broker job on the vc13k suite (same three rows, two passes).
    If 9 / 7 closes the gap, the next step is a design question for the user (a VC1 floor or a ratio change).
  - Galaxy result (Slurm 172240, bh-glx-110-a03u08, 04:01-04:06; every step's slots as given; both passes agree to
    0.1 point): the worker 30 -> 23 (A -> B) changes nothing beyond +0.5-0.9% on 256 B 1-hop worker rows. VC1 7 / 6
    -> 9 / 7 (B -> C) speeds the inter-mesh rows by 3-15% (GlxInter 0_x -> 1_1 at 3072 B 297 -> 255-257, the best
    row's 251-256) and the mix link totals by 8% at 256 B and 3% (MixWorker) at 3072 B; MixForward 3072 B link total
    +0.8%. So the VC1 split's gap to the best row at 3072 B is VC1's 7 / 6 (A/B-backed, one variable). The allocator's
    3:1 VC ratio gives VC1 29 of 115 slots, which its weights split 7 x 3 + 6; 9 x 3 + 7 needs 34. Design question
    for the user: a VC1 floor (e.g. forwarding 9 / receiver 7 at this size) or a VC ratio change; no variant.
  - P150 result (broker 679, same three rows, two passes): B -> C VC1Inter rows -6.7% median (to -13.9%), mixes to
    -16.5%; A -> B (worker 30 -> 23) +0.05% median. Same answer as the Galaxy.
- P150 dyn A/B (broker 675-678 = dyn_lb_ab.sh r2 a1 a2 b1 b2; out/p150_dyn_ab_675_678.md): every step's slots as
  expected. dyn, dynw24, dynfloor and the study default are never more than 1% slower than main's tiers in any suite.
  dynw24 is never more than 1% slower than dyn. 2D 3072 B: dyn, dynw24 and dynfloor within 1% of the best row.
  VC1 split 3072 B: dyn and dynw24 +16% on VC1Inter against the best row (VC1 7 / 6, as above).
  - **The VC0 receiver floor costs the VC1 split at 3072 B** (dynfloor 26,10,32,7,6 against dyn 30,14,16,7,6, both
    passes): VC0 2-hop +24% (16 B) and +32% (3072 B), multicast +5.8%, 2-hop merge +3.4-4.7%, mix link totals
    +1.9-3.1%. Hypothesis (not isolated): the floor's receiver 32 takes VC0 forwarding from 14 to 10. The floor is
    in the default policy; this goes to the user with the floor's other rows (2D 3072 B neutral).
  - VC2 on 3072 B against the study row (20,14,16 + 16,16): VC2 rows +41% median, Vc2Alone +142% (the 12:1 ratio's
    known cost, section 5.2).
  - dynw24 against dyn: never more than 1% slower; faster on the VC1 split at 3072 B (GlxSoloForward -3.6%,
    GlxMixForward link total -2.3%, GlxInter 0_0 -> 1_0 -2.2%) and 1D 7-hop (-1.0%). Galaxy evidence for
    worker_cap_slots=24; the P150 rows (broker parts r2 a1-b2) are pending.
- Mock vet on the new host (tree v4txa11_1104; out/dyn_vet_1104_floor.txt), the two suites lb2 172043 runs with the
  floor: P150 2D 3072 B and the VC1 split 3072 B, every arm as expected (tier, dyn and dynw24 with the floor off =
  pass a's rows; dynfloor 23,20,32 and 26,10,32,7,6). Code E0+E1 at most 22,144 (dynfloor = dyn); E1 least free stack
  4,236 with dynfloor (4,252 otherwise).
- dyn_vet.sh's launch check matched "launch reached by 0 process(es)"; it now needs at least one process.
- Galaxy pass a (171946) finished 01:35, every step's slots as expected. Results:
  /data/aagarwal/frs-glx/jobs/171946/results; report slotalloc/out/glx171946_pass_a.md (dyn_report.py, cycles/packet,
  mixed rows by link total). One pass on one node; pass b decides. Readings, not claims:
  - dyn and dynw24 are never more than 1% slower than main's tiers, in every suite (1D, VC1 split, VC2 on, 2D 3072 B,
    VC1 split 3072 B, VC2 on 3072 B).
  - 2D 3072 B against the measured best (20,21,32): dyn within 1.5% either way (3-hop -1.4%, column merge +0.7%).
  - VC1 split 3072 B against the measured best (20,15,16,9,7): dyn is slower on the inter-mesh rows, most on
    GlxInter 0_0 -> 1_1 at 3072 B (296.5 against 255.7, +16%), and on two link totals (MixForward 256 B +8.5%,
    MixWorker 3072 B +3.3%); MixForward 3072 B link total -1.2%. dynw24 is the same as dyn. Outside the 1-2% target.
    Hypothesis (not isolated): VC1's 7 / 6 against the best's 9 / 7; the floor's 26,10,32,7,6 keeps VC1 at 7 / 6.
  - VC2 on 3072 B: dyn's VC2 5 / 4 against the row's 16 / 16 leaves VC2 rows about 2x slower (Vc2Alone 371 against
    180), the 12:1 ratio's known cost (section 5.2); VC0 rows equal.
  - VC2 on 4352 B: dyn's VC2 4 / 2 is 23% faster on the VC2 rows than today's 2 / 4.
- These runs (host 648) predate the VC0 receiver floor (section 10). Their 3072 B `dyn` and `dynw24` arms are the
  floor-off rows. After a rebuild with the floor, the same rows need
  `TT_FABRIC_EXP_DYN_SLOTS_POLICY=vc0_receiver_floor=off`, and the expected rows in dyn_vet.sh, dyn_glx_submit.sh and
  dyn_lb_ab.sh name the floor-off rows.

Targets:
- within ~1-2% of the study's best row;
- no row worse than main's tiers (mixed rows by link total);
- E0 + E1 code within 26,624 B and no negative stack margin.

## 9. Receiver multiple-of-4 sweep (no build; coordinator, 2026-11-03)

Code facts (variants/v4txa11 router_receiver.hpp):
- `index_add` (:73) is a mask only for a power of 2.
- On ring/torus routers, first-level acks make completion groups 1 slot, so it runs about 4 times per packet (:288,
  :293, :541, :588). That fits the measured +7.5..+8.7 cycles/packet for rings of 24 / 40 / 48 (RESULTS §98, v4u).
- On mesh and line routers, groups are 4 slots.

Hypothesis, stated before the data: a multiple-of-4 receiver costs ≤ 1 cycle/packet on a mesh router, and ~8 on a
ring router.

`rxm4_runs.sh`: v4txa11 with explicit rows, 3072 B slots.
- 2D: 20,21,R with R = 16 / 20 / 24 / 28 / 32. Rows: 1-hop write, bidirectional write, 3-hop write and scatter,
  multicast write.
- Ring: 43,42,R with R = 24 / 28 / 32, on val_ring_p150's benchmark row.
- Data-checked validation first: mesh sanity's Mesh test at R = 20 / 24 / 28, and val_ring_p150 at 24 / 28 / 32.
- Two passes with the arms interleaved, in one LB job.

How to read it: a cost from the wrap shows as 20 / 24 / 28 sitting off the 16 -> 32 line; window effects move
monotonically with depth.

Results (lb2 job 171579 on bh-lb-120-a07u28, 2026-11-04 00:33-00:39; two passes, arms interleaved, one host;
cycles/packet, mean of the passes):
- Validation, data-checked, all passed: mesh sanity's Mesh test 48/48 at R = 20, 24 and 28; val_ring_p150 19/19 at
  R = 24, 28 and 32.
- 2D mesh, 3072 B slots, 20,21,R:

  | row | R=16 | R=20 | R=24 | R=28 | R=32 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 1-hop write | 113.2 | 113.0 | 112.9 | 112.9 | 112.9 |
  | bidirectional write | 113.3 | 113.2 | 113.1 | 113.0 | 113.1 |
  | 3-hop write | 142.2 | 141.9 | 142.3 | 142.0 | 142.3 |
  | 3-hop scatter | 140.9 | 142.5 | 142.4 | 142.3 | 140.9 |
  | multicast write (per sender) | 414.4 | 415.9 | 415.7 | 415.5 | 413.3 |

  Pass spread is at most 0.9; on the 3-hop scatter row, 141.1 / 140.6 at R=16 and 140.8 / 140.9 at R=32.
  - Measured: the non-power-of-2 receivers (20 / 24 / 28) are off the 16 -> 32 line by +1.4 to +1.6 on the 3-hop
    scatter row only. Every other mesh row is within ±0.4.
  - Hypothesis (not isolated): the wrap's compare costs where the 3-hop scatter row's receivers bind.
  - Against the stated ≤1 cycle: about 1.5 on one row, ~0 elsewhere.
- Ring (FullRingUnicastDeadlockQuick, 1024 B, 43,42,R): R=32 163.3; R=24 171.4 (+8.1, +5.0%); R=28 171.3 (+8.0). Pass
  spread is 0.01. Matches the ~8 cycles/packet of RESULTS §98 (v4u) on v4txa11.
- Read-out for the user's decision (not mine): `receiver_round=mult4` on non-ring routers would cost 0 to 1.6
  cycles/packet (the 3-hop scatter row the most). Ring routers stay pow2.

Caches (router ELFs per R): /data/aagarwal/lb2/jobs/171579/caches/. Results: fabric_exp/results/rxm4_{a,b,v}_*
(manifest results/lb2/171579_a07u28.tsv).

## 10. VC0 receiver floor on 2D routers (user, 2026-11-04; offline only)

Rule: on a 2D router (mesh or torus) the VC0 receiver's share is max(weighted share, N x 1/5), then the usual rounding
(nearest power of 2 by ratio). 1/5 is 16 of 81 at 4352 B, the smallest power of 2 that binds no 2D row.
- It is a ratio of N, not bytes. Policy key `vc0_receiver_floor=num:den` (default 1:5), `vc0_receiver_floor=off` to
  turn it off.
- It applies before the senders split the rest, and comes out of VC0's share.
- Every channel's minimum of 2 comes first. If the floor does not fit after VC0's senders' minimums, the receiver
  takes what fits rounded down to a power of 2 (a multiple of 4 under `receiver_round=mult4`), never below its weighted
  count. The reason logs both: "FLOOR MISSED: 3 fit ..., rounded down to a power of 2: 2" (coordinator, 2026-11-04).
- 1D routers (line, ring) are unchanged: the host passes `two_d` from the topology.

Applying "every channel's minimum comes first" to the whole allocator needed two more rules (sections 2.2 and 2.4.2),
approved by the coordinator (2026-11-04: what the instruction meant):
- a VC's share is raised to its channels' minimums;
- senders raised to the minimum take their slots back from the largest sender.

Without them the max-payload rows with VC2 or a 4x4 VC1 split were errors before the floor ("the counts need 24 slots,
N is 23"), so no floor miss could show there. They change no row that worked before. With the floor off, every row
and reason at 4352 and 3072 B, for the default policy and `worker_cap_slots=24`, is identical to the saved tables
(out/slot_table_reasons*.txt).

Rows the floor changes (`slot_alloc_table.sh --sizes 2048,3072,4352,15232`, against `--policy vc0_receiver_floor=off`
and the allocator before 2026-11-04). N = 169 / 115 / 81 / 23. 15232 B is the max payload (slot 15,328 B). Rows not
listed are unchanged.

| shape | payload | before | floor off | floor 1:5 | what changed |
| --- | ---: | --- | --- | --- | --- |
| 2D VC0 only (P150 2x4, Galaxy 8x4, both tori) | 3072 | 27,24,16 | 27,24,16 | 23,20,32 | floor: 19.2 -> 23 -> 32 |
| VC0+VC1 2x2 split, VC0 first | 3072 | 30,14,16,7,6 | 30,14,16,7,6 | 26,10,32,7,6 | floor: 23 -> 32 |
| VC0+VC1 2x2 split, VC1 first | 3072 | 25,12,16,10,8 | 25,12,16,10,8 | 18,9,32,10,8 | floor |
| VC0+VC1 4x4 split (and the multi-mesh torus) | 2048 | 31,27,16,11,8 | 31,27,16,11,8 | 27,23,32,11,8 | floor: 33.8 -> 32 |
| VC0+VC1 4x4 split | 3072 | 21,17,16,7,6 | 21,17,16,7,6 | 17,13,32,7,6 | floor |
| VC0+VC1 4x4 split | 4352 | 15,13,8,5,4 | 15,13,8,5,4 | **13,11,16,5,4** | floor: 10.2 -> 16.2 -> 16 (8 -> 16) |
| VC0+VC1 4x4 split | 15232 | error | 4,3,2,2,2 | 5,2,4,2,2 | minimum first, then the floor: 4.6 -> 4 |
| VC2 on | 3072 | 24,22,16 + 5,4 | 24,22,16 + 5,4 | 20,18,32 + 5,4 | floor |
| VC2 on | 15232 | error | 6,3,4 + 2,2 | 6,3,4 + 2,2 | minimum first; floor met (4) |
| three VCs | 3072 | 27,13,16,7,5 + 4,3 | 27,13,16,7,5 + 4,3 | 23,9,32,7,5 + 4,3 | floor |
| three VCs | 15232 | error | 3,2,2,2,2 + 2,2 | 3,2,2,2,2 + 2,2 | FLOOR MISSED: VC0 11 after the minimums, 3 fit, rounded down to 2 |

Checks:
- The coordinator's expected receivers hold: 4352 B 16 (the 4x4 split 8 -> 16), 3072 B 23 -> 32, max payload 4.6 -> 4,
  2048 B 33.8 -> 32.
- The must-keep 4352 B rows are unchanged: 2D 17,16,16, 1D 26,25,32, VC1 split 19,9,16,5,4, VC1 first 18,7,16,7,5,
  VC2 on 17,14,16 + 4,2, three VCs 17,8,16,5,4 + 3,2.
- 2048 B rows are unchanged except the 4x4 split. 1D rows are unchanged at every size.
- With `worker_cap_slots=24` at 3072 B: 2D 23,20,32 (cap not reached), VC1 split 24,10,32,7,6 (2 unused).

To read before any device run (no claims, nothing measured):
- At 3072 B the 2D row moves toward the measured best (20,21,32). The VC1 split moves away from it: forwarding 14 ->
  10 against the best's 15, receiver 32 against 16.
- The rxm4 sweep (section 9) measured R = 16 against 32 at 20,21,R and found no difference beyond 0.4 cycles/packet.
  It did not cover the senders the floor takes (27,24 -> 23,20).
- The three-VC max-payload row misses the floor (receiver 2, the same row as with the floor off). The three-VC build
  does not compile on v4txa11 (section 8), so it is table only.
- `receiver_round=mult4` (not adopted) cannot place that row with or without the floor: its VC0 receiver minimum of 4
  does not fit (an error, as before the floor).
- Transaction IDs (`--trids`): the 3072 B dyn rows now carry VC0 receivers of 32. 2D 4/8 (8 IDs); VC1 split 4/8 +
  1/6 (14); three VCs 4/8 + 1/5 + 1/3 (16, the whole NoC); multi-mesh torus 32a + 6 -> 1/8! + 1/6 (14).

Device plan (coordinator, 2026-11-04): the floor goes into the TRID host rebuild (host_rebuild_tridalloc.sh checks
for it and for patch 1's `shape.two_d`). After that rebuild:
- `dyn` and `dynw24` run with `vc0_receiver_floor=off` on any library that knows the key (dyn_p150_runs.sh,
  dyn_glx.sbatch and dyn_vet.sh detect it), so pass b and any rerun match pass a and the hard-coded rows. Pass b on
  trees/v4txa11_dyn keeps the pre-floor library and needs nothing.
- A separate `dynfloor` arm (default policy) at 3072 B: 2D 23,20,32 and the VC1 2x2 split 26,10,32,7,6, next to the
  floor-off rows in the next P150 round (dyn_lb_ab.sh) and Galaxy round (dyn_glx_submit.sh, on a tree with the floor).
  It is skipped on a library without the floor.
- The vet's `dynfloor` arm recaptures the VC1 split (32 | 6) for the TRID offline compile; it also checks 2D, both
  P150 3072 B suites and VC2 on (20,18,32 + 5,4).

## 11. VC1 window sweep (design, not run; user go 2026-11-04) — SUPERSEDED by section 12

Dropped by the user (2026-11-04): slots, not bytes, are what a channel needs (packets in flight); the "constant bytes"
framing, the VC0 receiver byte floor and the 2 KB-slot worker test go. Kept below as the record of the proposal.


The user rejected a VC1 floor; the rule must be explainable and hold across systems. Hypothesis under test: a channel
needs its credit loop covered (loop latency x link bandwidth, in bytes); if the loop is mostly serialization, the bytes
VC1 needs stay about the same on 400G P150 and 200G Galaxy, so a plain VC0:VC1 byte share explains it. Alternative:
the need grows with the destination mesh's size (VC1 path length), which calls for a mesh-dims term.
Known points: Galaxy 200G 3072 B best 9 / 7 (forwarding 28.5 KB of 3168 B slots), P150 400G 4352 B VC1-first 7 / 5
(31.1 KB of 4448 B slots).

Arms (explicit TT_FABRIC_EXP_SLOTS_VC1 rows; VC0 held, VC1 forwarding F1 and receiver R1 = F1 - 2 swept, as in both
best rows; VC0 receiver 16, a power of 2; VC1 receivers follow the allocator's round-down, not powers of 2):

| size (N, slot) | VC0 held | sweep (F1 / R1), F1 bytes | R probe | VC0 check |
| --- | --- | --- | --- | --- |
| 3072 B (115, 3168 B) | 16,14,16 (74) | 5/3 15.8 KB, 7/5 22.2 KB, 9/7 28.5 KB, 11/8 34.8 KB | 9/9 | 20,15,16,9,7 (the best row) |
| 4352 B (81, 4448 B) | 14,7,16 (51) | 4/2 17.8, 5/3 22.2, 6/4 26.7, 7/5 31.1, 8/6 35.6 KB | 7/7 | 18,7,16,7,5 (VC1-first best) |

- VC0 hold: forwarding 14 at 3072 B (the isolation's 23,14,16,9,7 reached the best row on the inter rows), 7 at
  4352 B (the VC1-first row that ran VC1 to its bound, job 388); worker 16 / 14, the most each size leaves VC1 room
  for. Worker 30 -> 23 changed no VC1 row (172240, 679); the VC0-check arm (the best row, more VC0 at one sweep
  point) shows whether the hold binds. Unused slots in the sweep points are deliberate (VC0 fixed).
- R probe: R1 + 2 at the expected knee; if it gains > 1%, the receiver binds there.
- Shapes: P150 2x2 split (400G; sls_vc1_all.yaml and sls_vc1_all_3072.yaml), Galaxy 2x2 split (200G; vc1, vc13k),
  Galaxy 4x4 split (200G, longer VC1 paths; sls_glx_vc1_4x4.yaml plus a 3072 B copy to make). Dual-Galaxy: not now.
  All with the packet-capped VC1 yamls; two-rank steps exclude a01u14, a02u02, a02u08, a03u14, a08u20.
- Rows: VC1 inter-mesh rows (GlxInter on Galaxy, VC1Inter on P150) and the mix link totals; VC0 2-hop and multicast
  as the check that VC0 is not the binding one.
- Jobs: P150 two broker jobs (one per size; 6 / 7 arms x 2 passes reversed, ~30-40 s each: ~9 min); Galaxy 2x2 one
  allocation (26 steps, ~27 min); Galaxy 4x4 one allocation (~32 min), after prep: the 3072 B 4x4 yaml, the
  vc1x44 / vc1x443k kinds in dyn_glx.sbatch (the launcher has vc1x44, the step table does not), a mock vet of every
  arm on the 4x4 split, and the multi-mesh sanity first in the job (v4txa11 should carry buffer-2 fix (c) from the
  receiver chain; the 4x4 split last ran data-checked on 2026-11-02, 168925 / 168926).
- Read-out per shape and size: the knee = the smallest F1 whose inter rows' median and mix link totals are within 1%
  of the sweep's best, in slots and bytes. Constant bytes across shapes and sizes -> a byte share; growth with mesh
  size -> a mesh-dims term; neither -> keep 3:1 and document.
- Bar for any resulting rule, on P150 and Galaxy: never worse than main's tiers, and within a stated margin (proposed
  2%; mix rows by link total) of each system's best.

## 12. Slot-need sweeps: each channel kind's knee in slots vs packet size (design, not run; user 2026-11-04)

Question: a channel's need is packets in flight, about credit-loop latency / time per packet. Hypothesis to test: the
knee (in slots) is highest and flat for small packets (router-bound) and lower for large, wire-bound packets.
The floor stays off and the worker cap stays undecided in every table here; nothing changes in the allocator.

How the slot size is set: the test yaml's `fabric_setup.max_packet_size` (-> FabricRouterConfig
max_packet_payload_size_bytes; Blackhole allows up to 15,232 B, the host checks it). Explicit rows by the study
switches `TT_FABRIC_EXP_SLOTS_2D` / `_1D` / `_VC1` (W,F,R[,F1,R1]); every run logs and checks its slots.
Checked on the mock (P150 descriptor, trees/v4txa11_1104, 2026-11-04): 8,704 and 15,232 B slots reach launch on 1D
and 2D, default and explicit rows log as set (1D 8704 default 13,12,16; 2D 8704 tier 9,8,8; 2D 15232 tier 7,4,4;
explicit 12,13,16 / 7,6,16 / 8,7,8 / 5,3,8). Router code is unaffected: largest E0+E1 1D 14,776 / 14,764 / 14,672 B
and 2D 16,312 / 16,332 / 16,184 B at 4352 / 8704 / 15232 B slots; stack within 16 B. Yamls: the slot-study suites
(sls_1d_p150, sls_2d_p150 and the VC1 suites) take any payload up to max_packet_size (unicast_write); the mock does
not move data, so each large-slot job starts with a data-checked 1-hop write at 8,704 and 15,232 B.

Slot budgets N: 4352 B 81 (2D, VC1) / 83 (1D); 3072 B 115 / 117; 8704 B 41 / 41; 15232 B 23 / 23.
Packet sizes per arm: 256, 1024, 2048, 4352 B (plus 8704 / 15232 B where the slot allows). Arms vary one channel kind;
the others are held above their need or, when not on the rows' path, at the minimum 2. Two passes, arms reversed.

| set | system, slots | kind (rows) | held | swept |
| --- | --- | --- | --- | --- |
| A1 | P150 4352 B, 2D | worker (1-hop E/W write, bidir) | F 2 (off path), R 32 | W 4, 8, 12, 16, 20, 24, 32 |
| A2 | P150 4352 B, 2D | receiver (1-hop write, bidir) | W 24, F 8 | R 4, 8, 16, 32 |
| A3 | P150 4352 B, 2D | forwarding (3-hop E/W, N/S 2-hop, turn) | W 20, R 16 | F 2, 4, 6, 8, 10, 12, 15; check W 29 at F 12 |
| B1-B3 | P150 4352 B, 1D | worker / receiver / forwarding (1-hop, merge / 3-hop) | as A, F 2 or W 24, R 32 | W as A1; R 4-32; F 2, 4, 8, 12, 16, 24 |
| C | P150 4352 B, VC1 2x2 split | VC1 forwarding / receiver (VC1Inter, mix link totals; VC0 2-hop, multicast as checks) | VC0 14,7,16 | F1 2, 4, 6, 7 at R1 8; R1 2, 4 at F1 6 |
| D | P150 3072 B, 2D | worker, receiver (confirm the knee does not follow the slot size) | as A | 3 points each around the 4352 B knee |
| E | P150 8704 B, 1D then 2D | worker, receiver; 1D forwarding | W 16, R 16, F 2 (1D F <= 9) | W 4-20; R 4, 8, 16; 1D F 2, 4, 6, 9; packets 256, 1024, 4352, 8704 |
| F | P150 15232 B, 2D and 1D (squeeze) | which squeeze costs least | N = 23 | 2D: worker 13,2,4; forwarding 4,5,4; receiver 9,2,8; main's tier 7,4,4. 1D: 13,2,8; 4,11,8; 4,3,16; 8,7,8 |
| G1 | Galaxy 4352 B, 2D | worker, receiver, forwarding (as A1-A3) | as A | as A |
| G2 | Galaxy 4352 B, VC1 2x2 and 4x4 splits | VC1 (GlxInter, mix link totals) | VC0 14,7,16; 4x4 adds a VC0-check arm | as C |

Rows per arm: only the rows on the kind's path (1-hop / bidir for worker and receiver; 3-hop, N/S multi-hop and turn
for forwarding; inter-mesh and mix rows for VC1), each at the four packet sizes. Read-out: per kind x system x slot
size, the smallest swept depth within 1% of the sweep's best on those rows, per packet size (mix rows by link total).

Jobs and time (estimates from today's runs: ~20 s per P150 single-process arm, ~35 s per two-rank arm, ~50-60 s per
Galaxy step, two passes):
- P150 broker (<= ~17 min each): P1 = A1-A3 (18 arms, 36 runs, ~13 min); P2 = B1-B3 (17 arms, ~12 min); P3 = C + D
  + F (6 + 6 + 9 arms, ~17 min); P4 = E (20 arms + the data-checked large-packet validation, ~15 min). 4 jobs, ~1 h.
- Galaxy (excludes a01u14, a02u02, a02u08, a02u14 for rings; a03u14, a08u20 for two-rank steps; cap 4): G1 one
  allocation (36 steps, ~35 min); G2 one allocation (~30 min) after prep: size-swept VC1 yamls (GlxInter at 256,
  1024, 2048, 4352 B, packet-capped), the vc1x44 kind in dyn_glx.sbatch's step table, a mock vet of the 4x4 arms,
  and the data-checked multi-mesh sanity first.
- Prep (CPU): the size-swept yamls per kind, a runner with one label per arm (dyn_p150_runs.sh takes the arm row as
  `row` with a per-arm pass tag, as broker 679 did), mock vets of every arm's slots.

Cut, and why:
- Galaxy 1D (lines): the 1D kinds are covered on P150; Galaxy keeps the 2D and VC1 kinds, which differ by link speed.
- Galaxy at 3072 / 8704 / 15232 B: the slot-size questions are answered on P150; Galaxy answers the 200G question.
- 2D forwarding at 8704 B: with N = 41 the worker and receiver cannot be held above their need while F is swept.
- VC1 at other slot sizes and a 2-D (F1 x R1) VC1 grid: one F1 sweep at R1 8 plus an R1 probe answer the knee.
- Dual-Galaxy: not available within the cap.
- Multicast rows only as VC0 checks, not swept.

### 12.1 First runs (2026-11-04) and the /data inode exhaustion

Prep done: knee/ (mk_knee_yamls.py, 23 suites; knee_arms.sh, 101 arms within budget; knee_p150_runs.sh; the knee kinds
in dyn_glx.sbatch; knee_glx_submit.sh; knee_vet.sh: all 115 arm-runs mock-vetted, launch reached, slots as set;
knee_report.py). /data ran out of inodes from about 06:22 (`df -i`: 3,137 of 400.6M free at 06:58; shared with
other users): Galaxy G1 (172652, 172673) and G2 (172669) failed creating their JIT caches there, lb2 B (172650, 172733)
and E (172651) stopped at the stage check or lost their gather, A + D (172649) ran to ALL_DONE on a07u28 but its
results were not copied back (the start marker could not be written), E (172672) ran to ALL_DONE on a04u32 with 50 of
132 result files copied. Complete so far: C + F (172668, a07u28, both passes).

C (P150 4352 B, VC1 2x2 split, VC0 held 14,7,16): the VC1 inter rows are window-bound across the whole range the
budget allows. VC1Inter 0_0 -> 1_1 at 256 B: forwarding 2 / 4 / 6 / 7 (receiver 8) 834 / 434 / 301 / 265 cycles per
packet (cycles x F1 = 1,670-1,850: nearly proportional), receiver 2 / 4 / 8 (forwarding 6) 620 / 311 / 301; 4352 B
alike (932 / 487 / 336 / 293). Knee: forwarding >= 7 and receiver >= 8 at every packet size; the VC0 rows (2-hop,
multicast) and the solo worker do not move. Reading (hypothesis): the VC1 credit loop holds ~1,800 cycles of packets
at this hold, so the knee lies past 7; the sweep needs more VC1 room (a smaller VC0 hold or larger slots' N) to find it.

F (P150 15232 B, N 23), cycles/packet at 256 / 4352 / 15232 B packets:

| row (2D) | worker 13,2,4 | forwarding 4,5,4 | receiver 9,2,8 | main's tier 7,4,4 |
| --- | --- | --- | --- | --- |
| 1-hop | 271 / 333 / 449 | 372 / 427 / 552 | **188 / 203 / 423** | 271 / 333 / 449 |
| 3-hop | 765 / 887 / 1228 | **377 / 424 / 552** | 819 / 923 / 1232 | 394 / 462 / 616 |

| row (1D) | worker 13,2,8 | forwarding 4,11,8 | receiver 4,3,16 | balanced 8,7,8 |
| --- | --- | --- | --- | --- |
| 1-hop | **150 / 175 / 422** | 400 / 439 / 532 | 420 / 474 / 532 | 197 / 222 / 422 |
| 3-hop | 813 / 921 / 1217 | 403 / 441 / 532 | 547 / 618 / 811 | **248 / 277 / 422** |
| row merge | 483 / 555 / 838 | 186 / 211 / 436 | 321 / 374 / 553 | **170 / 194 / 422** |

Reading: at N 23 a receiver of 4 costs the 2D 1-hop rows 44% at small packets (188 vs 271) and still 6% at
15,232 B; forwarding 2 doubles multi-hop rows; the 1D balanced row reaches the 15,232 B wire floor (~422) on every row.

### 12.2 P150 knees (2026-11-04): A, B, D, E

Runs: A + D and B on the a09u32 broker (728, 729; labels kq*), A + D again on a07u28 (lb2 172649, fetched by 172798;
labels kn*; same knees as a09u32), E on a04u32 (lb2 172672, fetched by 172797); a second E on lb2 172799 (labels kr*)
running. Both passes everywhere; compare within a host. Knee = the smallest swept depth within 1% of the sweep's best, given as
the interval the sweep bounds it to ("17-20": above 16, at most 20); ">16": the largest depth swept still gains more
than 1% over 16, no upper bound. Large-packet data checks (1-hop write at 4352, 8704 and 15232 B packets, 1D and 2D)
all passed.

| kind | system, slots (depths swept) | 256 B | 1024 B | 2048 B | 4352 B | 8704 B |
| --- | --- | --- | --- | --- | --- | --- |
| worker, 2D (1-hop, bidir) | P150 4352 B (4-32) | 17-20 | 17-20 | 17-20 | 17-20 | - |
| worker, 2D | P150 3072 B (8, 16, 24) | >16 | >16 | >16 | - (3072 B: >16) | - |
| worker, 2D | P150 8704 B (4-19) | >16 | >16 | - | >16 | 9-12 |
| worker, 1D | P150 4352 B (4-32) | 21-24 | 21-24 | 21-24 | 17-20 | - |
| worker, 1D | P150 8704 B (4-20) | >16 | >16 | - | >16 | 9-12 |
| receiver, 2D | P150 4352 B (4-32) / 3072 B (8-32) | 9-16 | 9-16 | 9-16 | 9-16 | - |
| receiver, 1D | P150 4352 B (4-32) | >16 (32 is 3% better) | >16 | >16 | 9-16 | - |
| receiver, 1D and 2D | P150 8704 B (4-16) | >8 | >8 | - | >8 | >8 (some rows 5-8) |
| forwarding, 2D (3-hop, turn, merge) | P150 4352 B (2-15) | >12 | >12 | >12 | >12 | - |
| forwarding, 1D (3-hop, merge) | P150 4352 B (2-24) | >16 (24 is 6% better) | >16 | >16 | >16 | - |
| forwarding, 1D | P150 8704 B (2-9) | >6 | >6 | - | >6 | >6 |

Readings (measured unless marked):
- **D2 (lb2 172838, a04u32, both passes, spread <= 0.8 cycles), the 2D worker at 16 / 18 / 20 / 22 / 24, forwarding 2,
  receiver 32, at both slot sizes on one host:** 16 is 120.2-120.8 at 256 B under both; 18 reaches 113.3-113.5
  (3072 B slots) and 114.2-114.4 (4352 B slots); 20 and 24 sit at 112.7-113.5 and 113.8-114.7. Knee against that
  plateau: 17-18 at both slot sizes for 256-1024 B packets; 19-20 for 2048-3072 B packets at 3072 B slots, 17-20 for
  4352 B packets at 4352 B slots. One point does not fit: 22 at 4352 B slots runs 112.5-113.1, 1.1-1.3% below 20 and
  24 in both passes (no such dip at 3072 B slots); the strict 1% rule then reads 21-22 there. Observation only, no
  cause claimed; not chased unless it shows up in other rows. Reading: the worker's knee is the same in slots at 3072 and 4352 B slots (17-18 at small
  packets); this supports the claim below for the 2D worker; the other kinds were not swept this way.
- **Hypothesis (supported for the 2D worker by D2): the knee in slots does not follow the slot size.** For it: the 2D worker at 256 B gives
  the same curve below the knee at every slot size (121 cycles/packet at 16 under 4352, 3072 and 8704 B slots; 120.3
  and 120.9 at 3072 and 4352 B), and 19 under 8704 B slots (114.5) matches 20 under 4352 B (113.8). Open: the 3072 B
  sweep has no point between 16 and 24, so it bounds the knee only to >16; 24 at 3072 B slots runs 112.6-113.4,
  about 1 cycle faster than 20-24 at 4352 B (113.6-114.5), within both passes. Settling it needs 18 / 20 / 22 at
  3072 B next to the same depths at 4352 B on one host. The worker arms hold forwarding 2 and receiver 32 at both slot
  sizes; pass-to-pass spread is at most 0.7 cycles.
- Below the knee, cycles/packet x depth is nearly constant per kind (2D worker ~1,700-2,100 cycles; 1D worker
  ~1,600-1,900; 2D forwarding ~1,600-2,200): a window of packets covering a credit loop. Hypothesis: the need is
  that loop over the per-packet time at the bound, which matches 1D needing more than 2D (floor 95.6 against 113.8
  cycles) and wire-bound large packets needing fewer (8704 B: worker 9-12; 4352 B in 1D: worker 17-20, receiver 9-16).
- Small packets (256-2048 B) share one knee per kind (router-bound, flat); the knee drops only where the packet is
  wire-bound (4352 B in 1D, 8704 B everywhere).
- Forwarding needs more than the 2D budget allows with worker 20 and receiver 16 (3-hop still 13% better at 15 than
  at 12); in 1D it is still 6% better at 24 than at 16.

### 12.3 C2: the VC1 knee past C's range (user, 2026-11-04; lb2 172863, a07u28, both passes)

Slot size 3072 B (N 115) so that VC0 keeps its worker (20) and receiver (16) at their knees: VC0 held 20,11,16; VC1
forwarding 4-12 at receiver 8, receiver 4 / 8 / 16 at forwarding 10, and a VC0-check arm (VC0 forwarding 14) at 8 / 8.
At 4352 B slots the same VC1 ranges would have forced VC0's receiver to 8 and its forwarding to 3-4, below their
knees. VC0's forwarding 11 is below its own need (>12): the VC0 2-hop row measures VC0 forwarding there (187 vs 169
cycles at 256 B against forwarding 14), while the VC1 inter rows do not move with it (within 0.2 cycles), so the hold
does not bind them. All arms mock-vetted first; /data had 2.6M free inodes at submission.

| VC1 kind | 256 B | 1024 B | 2048 B | 3072 B |
| --- | --- | --- | --- | --- |
| forwarding (4, 6, 8, 10, 12 at receiver 8) | >10 | >10 | >10 | >10 |
| receiver (4, 8, 16 at forwarding 10) | 5-8 | 5-8 | 5-8 | 5-8 |

VC1Inter 0_0 -> 1_1 at 256 B: forwarding 4 / 6 / 8 / 10 / 12 give 433 / 300 / 254 / 246 / 243 cycles per packet
(10 -> 12: 1.2%, just past the 1% rule; 8 -> 10: 3%); receiver 4 / 8 / 16 give 310 / 246 / 247. The 0_0 -> 1_2 and
1_3 rows and 3072 B packets follow the same shape. Cycles x depth below the knee: forwarding 1,730-2,070 (4-8),
receiver 1,240-1,390 (4). Hypothesis (credit loop, not tested): the window covers a loop of about 1,700-2,000 cycles
for VC1 forwarding, so the need is that loop over the rows' bound (~243 cycles here), about 8-10 slots; the small
packets' and 3072 B packets' knees agree.

### 12.4 G1: Galaxy 2D at 4352 B slots (Slurm 172820, bh-glx-110-a03u08, both passes, node-local caches)

| kind (rows as A) | 256 B | 1024 B | 2048 B | 4352 B |
| --- | --- | --- | --- | --- |
| worker | 17-20 | 17-20 | 17-20 | 9-12 |
| receiver | 9-16 | 9-16 | 9-16 | 5-8 |
| forwarding | >12 | >12 | >12 | 11-12 |

Against the P150 (§12.2): the same knees for 256-2048 B packets (router-bound on both), and lower knees for 4352 B
packets on the Galaxy's 200G links (worker 9-12 against 17-20, receiver 5-8 against 9-16), where those packets are
wire-bound. This fits the reading that wire-bound packets need fewer slots (hypothesis: the need is the credit loop
over the per-packet time at the bound). The E rerun (lb2 172799, a07u28, labels kr*) repeats a04u32's E knees.

### 12.5 G2: Galaxy VC1 at 4352 B slots (Slurm 172891, bh-glx-110-a03u08, both passes, node-local caches)

The 4x4 split's data-checked multi-mesh sanity passed first (6 tests, both ranks). VC0 held 14,7,16 as in C (P150):
its worker (14) and forwarding (7) are below their own knees, so VC0 rows measure VC0 there; the 4x4 VC0-check arm
(worker 18 at VC1 6 / 8) moves no row by more than 0.4 cycles. Pass spread at most 0.9 cycles.

| split, VC1 kind (depths swept) | 256 B | 1024 B | 2048 B | 4352 B |
| --- | --- | --- | --- | --- |
| 2x2, forwarding (2, 4, 6, 7 at receiver 8) | >6 | >6 | >6 | >6 |
| 2x2, receiver (2, 4, 8 at forwarding 6) | >4 | >4 | >4 | >4 |
| 4x4, forwarding (2, 4, 6, 7 at receiver 8) | >6 | >6 | >6 | >6 |
| 4x4, receiver (2, 4, 8 at forwarding 6) | >4 | >4 | >4 | >4 |

Some inter rows settle earlier (forwarding 5-6, receiver 3-4: e.g. GlxInter 0_0 -> 1_3 holds at ~390 cycles from
forwarding 6, a bound outside VC1's window). Cycles x VC1 forwarding depth below the knee: 2x2 1,800-2,370, 4x4
1,890-2,520 (P150, C: 1,670-1,850); receiver at 2-4: 2x2 1,270-1,740, 4x4 1,360-1,750. Readings (hypotheses): the
VC1 window on the Galaxy covers a loop of about 1,900-2,500 cycles, somewhat longer than the P150's; the 4x4 split's
products are close to the 2x2's at these depths, so this sweep shows no clear growth with mesh size. Locating the
knee itself needs more VC1 room, as C2 did on the P150 (3072 B slots).

### 12.6 Galaxy C2 (G3): the VC1 knee at 3072 B slots (Slurm 172999, bh-glx-110-a10u08, both passes, 2026-11-04)

The first submission (Slurm 172992) landed on bh-glx-120-c08u14 (bh_sc36_2, the user's one-time exception) and
stopped at its first step: the host could not map the 4x4 split's mesh graph onto that Galaxy (topology_mapper, before
any kernel); resubmitted to the 110-a partitions, the same node type as G1 / G2 (not the same node: a10u08 against
a03u08). The 4x4 split's data-checked multi-mesh sanity at 3072 B slots passed first (6 tests, both ranks). VC0 held
20,11,16 as on the P150 C2: worker and receiver at their knees, forwarding 11 below its own need (>12). The VC0-check
arm (forwarding 14 at VC1 8 / 8) leaves the inter rows within 0.2 cycles except GlxInter 0_0 -> 1_3 (269 -> 259 at
256 B, 290 -> 278 at 3072 B), whose path uses VC0 forwarding, so that row reads VC0 there. Pass spread at most 2.1.

| split, VC1 kind (depths swept) | 256 B | 1024 B | 2048 B | 3072 B |
| --- | --- | --- | --- | --- |
| 2x2, forwarding (4, 6, 8, 10, 12 at receiver 8) | >10 | >10 | >10 | >10 |
| 2x2, receiver (4, 8, 16 at forwarding 10) | 5-8 | 5-8 | 5-8 | >8 (one mix row; the rest 5-8) |
| 4x4, forwarding | >10 | >10 | >10 | >10 |
| 4x4, receiver | 5-8 | 5-8 | 5-8 | 5-8 |

">10" rows gain 1.1-1.6% from 10 to 12 (GlxInter 0_1 -> 1_1 256 B: 248 -> 244; X44Inter 0_0 -> 1_0: 283 -> 280); the
other inter rows settle at 9-10. Cycles x depth below the knee (4-8): 2x2 forwarding 1,870-2,320, 4x4 1,950-2,500,
receiver at 4: 1,325-1,680 (P150 C2: 1,730-2,070 and 1,240-1,390). Readings: the VC1 knee in slots is the same on the
P150 and the Galaxy and on both splits (forwarding about 10-12, receiver 5-8 at every packet size); the Galaxy's
products run about 10-20% longer, and the 4x4 split's are close to the 2x2's (no clear growth with mesh size).
Hypothesis (credit loop, not tested): the VC1 window covers a loop of about 1,900-2,500 cycles on the Galaxy.

### 12.7 Composition test: two squeezed channels on one path (user, 2026-11-04; a09u32 broker 760, 761)

Question: when two channels on one path are both squeezed, does the path run at the minimum of their fractions (equal
fractions are right) or at about their product? P150 2D, 4352 B slots, explicit rows (TT_FABRIC_EXP_SLOTS_2D), 256 and
4352 B packets, data-checked 1-hop sanity first (passed), both passes reversed on a09u32. Deviation: the requested
"full" 20,16,16 needs 84 of 81 slots, so every arm whose forwarding is "full" uses 15 (off the 1-hop rows' path; on
the 3-hop and turn rows it is below forwarding's own need, so "full" there is itself forwarding-limited). Broker 761
reran all of 760's arms plus the two singles the 3-hop pairs need (receiver 8 alone, worker 10 alone on 3-hop / turn);
the two runs agree within 0.6 cycles; pass spread at most 1.3. Numbers below: 761 (report_Q_761.md; 760 in q760/).

| arm (W,F,R) | 1-hop 256 / 4352 B | bidir 256 / 4352 B | 3-hop 256 / 4352 B | turn 256 / 4352 B |
| --- | --- | --- | --- | --- |
| full 20,15,16 | 114.7 / 125.8 | 114.8 / 126.6 | 145.2 / 155.2 | 145.8 / 160.1 |
| W½ 10,15,16 | 184.4 / 202.1 (0.62 / 0.62) | 184.4 / 212.8 (0.62 / 0.60) | 192.2 / 206.9 (0.76 / 0.75) | 193.5 / 207.3 (0.75 / 0.77) |
| R½ 20,15,8 | 157.4 / 179.7 (0.73 / 0.70) | 157.3 / 190.3 (0.73 / 0.67) | 167.8 / 185.9 (0.87 / 0.84) | 168.2 / 186.4 (0.87 / 0.86) |
| W½+R½ 10,15,8 | 167.2 / 180.7 (0.69 / 0.70) | 166.9 / 192.4 (0.69 / 0.66) | - | - |
| unequal 14,15,4 | 270.0 / 332.6 (0.43 / 0.38) | 269.7 / 345.6 (0.43 / 0.37) | - | - |
| F½ 20,8,16 | - | - | 239.9 / 266.8 (0.61 / 0.58) | 253.1 / 274.0 (0.58 / 0.58) |
| F½+R½ 20,8,8 | - | - | 216.9 / 245.2 (0.67 / 0.63) | 224.0 / 249.6 (0.65 / 0.64) |
| W½+F½+R½ 10,8,8 | 167.3 / 180.7 (0.69 / 0.70) | 167.0 / 192.3 (0.69 / 0.66) | 216.9 / 245.3 (0.67 / 0.63) | 224.1 / 249.8 (0.65 / 0.64) |

Cycles/packet (rate as a fraction of full). Pairs against min(single fractions) and their product:

| pair, row | singles | both squeezed | min | product |
| --- | --- | --- | --- | --- |
| W½ + R½, 1-hop 256 B | 0.62, 0.73 | **0.69** | 0.62 | 0.45 |
| W½ + R½, 1-hop 4352 B | 0.62, 0.70 | **0.70** | 0.62 | 0.44 |
| W½ + R½, bidir 4352 B | 0.60, 0.67 | **0.66** | 0.60 | 0.40 |
| F½ + R½, 3-hop 256 B | 0.61, 0.87 | **0.67** | 0.61 | 0.52 |
| F½ + R½, turn 4352 B | 0.58, 0.86 | **0.64** | 0.58 | 0.50 |
| (F½+R½) + W½, 3-hop 256 B | 0.67, 0.76 | **0.67** | 0.67 | 0.51 |

Readings (measured):
- Neither model fits the two-channel squeezes: the both-squeezed path runs above the minimum of the singles (by
  0.05-0.07, far outside the pass noise) and far above their product. Squeezing the receiver to 8 makes the
  worker-squeezed path faster (W½ 184.4 -> W½+R½ 167.2 cycles) and the forwarding-squeezed path faster (F½ 239.9 ->
  F½+R½ 216.9).
- Adding a third, non-binding squeeze composes as the minimum: W½ on top of F½+R½ leaves the 3-hop and turn rows
  unchanged (216.9 / 224.1), and the forwarding squeeze is off the 1-hop path (W½+F½+R½ = W½+R½ there).
- At the same worker + receiver total (18 slots), the unequal split 14 / 4 loses 38-46% of the rate against the equal
  10 / 8 (270.0 against 167.2 cycles at 256 B): a receiver of 4 costs far more than the worker gains.
- Hypothesis (not tested): the receiver's completion-group size changes the upstream channel's credit loop. With 16
  slots a receiver frees slots in groups of 4, with 8 in groups of 2 (the router's 1 / 2 / 4 rule for < 8 / 8-15 /
  >= 16 slots), so a smaller receiver returns credits in finer steps and a squeezed upstream window recycles sooner.
  An isolating A/B would hold worker 10 and compare receiver 12 (groups of 2) with 16 (groups of 4): if 12 beats 16,
  the grouping, not the depth, is the lever. Not run.

### 12.8 Completion-group A/B: is the receiver's credit granularity the lever? (user, 2026-11-04; broker 762)

P150 2D, 4352 B slots, explicit rows, a09u32 broker, both passes reversed, every arm mock-vetted, the data-checked
1-hop sanity first (passed); pass spread at most 0.8 cycles. Group sizes confirmed: router_receiver.hpp computes
completion_group_size = first_level_ack ? 1 : (slots < 8 ? 1 : slots < 16 ? 2 : 4); the builds carry
ENABLE_FIRST_LEVEL_ACK_VC0 = 0 (2D mesh, no deadlock avoidance), and the host's mirror logged RX_CH_0 G 2 for
receivers 8 and 12 and G 4 for 16, 20 and 24 on every router.

| row | size | W10: R8 (G2) | R12 (G2) | R16 (G4) | R20 (G4) | R24 (G4) | W20: R12 (G2) | W20: R16 (G4) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop | 256 | 167.3 | 168.8 | 184.3 | 185.5 | 185.3 | 127.3 | 114.8 |
| 1-hop | 4352 | 180.8 | 182.6 | 202.1 | 202.7 | 202.6 | 130.3 | 125.8 |
| bidir | 256 | 167.0 | 168.2 | 184.4 | 185.5 | 185.3 | 127.2 | 114.5 |
| bidir | 4352 | 192.5 | 194.3 | 213.2 | 212.8 | 212.2 | 138.2 | 126.7 |

Forwarding-squeezed 3-hop / turn (worker 20, forwarding 8): receiver 12 (G2) against 16 (G4): 3-hop 218.6 against
240.0 (256 B), 245.7 against 266.7 (4352 B); turn 226.1 against 253.2, 250.7 against 274.1.

Readings:
- With a squeezed upstream (worker 10, or forwarding 8), the rate follows the receiver's group size, not its depth:
  flat within each group (8 ~ 12 at G2: 167-169; 16 ~ 20 ~ 24 at G4: 184-186) and a 9-10% step between them (12
  beats 16). With the worker unsqueezed (20), depth wins (16 beats 12 by 10%, 114.8 against 127.3). [WRONG (§12.10):
  these arms changed depth and group size together; with the depth fixed (R16 and R32, 3072 B slots) worker 20 runs
  122.8 at G2 against 113.0 at G4, so most of the 10% is the group size. The rest (127.3 at R12 / 4352 B against
  122.8 at R16 / 3072 B) is depth or slot size, not isolated.] Both halves of
  the readout hold, and the arms change the group size at a fixed upstream squeeze while depth inside a group has no
  effect, so the grouping is the lever for the squeezed path (A/B-backed). This also explains §12.7's "both squeezed
  beats the minimum": its R½ arm moved the receiver from groups of 4 to groups of 2.
- Why the grouping matters is still a hypothesis: a group's slots are credited back together, so with groups of 4 the
  upstream sender sees credits in steps of 4 and its small window idles longer between steps.
- Caveat: receivers 12, 20 and 24 are not powers of 2 (the ring index compares instead of masking); within G4,
  16 (power of 2) and 20 / 24 (not) run the same, so that path does not show here.

### 12.9 Group-size crossover: where groups of 4 stop costing a squeezed sender (user, 2026-11-04; broker 763)

P150 2D, 4352 B slots, explicit rows, a09u32 broker, both passes reversed, all 21 arms mock-vetted, data-checked 1-hop
sanity first (passed); pass spread at most 0.7 cycles. Receiver 12 = groups of 2 (G2), 16 = groups of 4 (G4).
Budget note: worker 24 with receiver 16 needs 85 of 81 slots at forwarding 15, so that arm runs forwarding 13 (off
the 1-hop path). Cycles/packet G2 / G4 (G4 against G2):

| worker (fwd 15) | 8 | 12 | 16 | 20 | 24 |
| --- | --- | --- | --- | --- | --- |
| 1-hop 256 B | 205.7 / 231.4 (+12.5%) | 144.4 / 155.7 (+7.8%) | 127.3 / 121.0 (-5.0%) | 127.3 / 114.4 (-10.1%) | 127.3 / 114.7 (-9.9%) |
| 1-hop 2048 B | 211.6 / 236.2 (+11.6%) | 148.5 / 160.2 (+7.9%) | 127.7 / 124.3 (-2.6%) | 127.7 / 114.3 (-10.5%) | 127.7 / 114.2 (-10.6%) |
| 1-hop 4352 B | 226.0 / 254.3 (+12.6%) | 155.7 / 173.2 (+11.2%) | 130.3 / 133.1 (+2.1%) | 130.3 / 125.8 (-3.5%) | 130.3 / 125.8 (-3.4%) |
| bidir 256 B | 205.4 / 232.4 (+13.1%) | 144.2 / 155.9 (+8.1%) | 127.2 / 120.7 (-5.1%) | 127.2 / 114.8 (-9.7%) | 127.2 / 114.7 (-9.8%) |
| bidir 2048 B | 215.1 / 237.2 (+10.3%) | 150.5 / 161.9 (+7.6%) | 128.3 / 125.7 (-2.0%) | 128.2 / 114.8 (-10.5%) | 128.2 / 114.4 (-10.8%) |
| bidir 4352 B | 238.9 / 256.9 (+7.5%) | 164.1 / 184.3 (+12.3%) | 138.2 / 142.5 (+3.1%) | 138.2 / 126.6 (-8.4%) | 138.1 / 126.7 (-8.3%) |

| forwarding (worker 20) | 6 | 8 | 10 | 12 | 15 |
| --- | --- | --- | --- | --- | --- |
| 3-hop 256 B | 285.3 / 311.4 (+9.2%) | 218.5 / 239.8 (+9.8%) | 180.3 / 196.0 (+8.8%) | 157.3 / 167.5 (+6.5%) | 152.5 / 145.2 (-4.8%) |
| 3-hop 4352 B | 324.1 / 344.0 (+6.1%) | 245.7 / 266.5 (+8.5%) | 199.4 / 217.1 (+8.9%) | 172.2 / 184.6 (+7.2%) | 152.6 / 155.1 (+1.7%) |
| turn 256 B | 295.2 / 329.0 (+11.5%) | 226.2 / 253.1 (+11.9%) | 186.9 / 205.3 (+9.8%) | 162.4 / 174.3 (+7.3%) | 152.2 / 145.9 (-4.2%) |
| turn 4352 B | 329.6 / 360.9 (+9.5%) | 250.6 / 273.9 (+9.3%) | 205.0 / 225.4 (+9.9%) | 176.4 / 189.3 (+7.3%) | 154.0 / 160.1 (+3.9%) |

Readings:
- Below the crossover groups of 4 cost a squeezed sender 6-13%, roughly flat in the sender's depth (worker 8-12,
  forwarding 6-12), at every packet size.
- Crossover (G4 stops losing beyond the 0.7-cycle noise): worker between 12 and 16 for 256-2048 B packets, between
  16 and 20 for 4352 B; forwarding between 12 and 15 for 256 B, not reached by 15 for 4352 B (+1.7% / +3.9%).
- Fixed slots or a ratio: both kinds cross between 12 and 16 slots for small packets, about 3-4 groups of 4 in the
  sender's window, and later for 4352 B packets. With one group-size pair (2 against 4) these arms cannot tell a fixed
  count from sender depth / G; the shift with packet size fits "the crossover sits near the sender's own need"
  (worker knee 17-20, forwarding > 12 in §12.2) better than a fixed count (hypothesis).
- Confound past the crossover: the G2 arm's receiver is 12 slots, which caps it at 127.3 (1-hop) and about 152
  (3-hop / turn) from worker 16 and forwarding 15 on, the same 127.3 as §12.8's worker-20 control. So "G4 wins" past
  the crossover is the receiver's depth (16 against 12), not its grouping. [WRONG (§12.10): with the switch, at a fixed
  receiver of 16 or 32 slots, worker 20 runs 122.8 at G2 against 113.0 at G4: past the crossover G4 wins mostly by its
  grouping.] Whether groups of 4 still cost an unsqueezed sender is not isolated: under the current rule a receiver
  with groups of 2 cannot have 16 or more slots. [Isolated in §12.10: they do not; G4 is the fastest group size for an
  unsqueezed sender at every k.]
- Mechanism (credit granularity) remains a hypothesis; §12.8 isolated the grouping as the lever, not the reason.


### 12.10 Completion-group switch and k-channel merges (user, 2026-11-04; broker 777, 778; Slurm 175221)

The user's idea: the cost of groups of 4 on a squeezed sender (§12.8, §12.9) shrinks as more far sender channels (k)
feed one receiver. Not an allocator decision; the switch stays study-only.

Setup:
- Study switch `TT_FABRIC_EXP_COMPLETION_GROUP=G` (host; variants/host_dynslots/3_completion_group.patch; library
  rebuilt by broker 769; Galaxy tree v4txa11id_1104cg). It sets the completion group of every receiver channel
  without first-level acks to G (1, 2 or 4; one value, or one per VC); first-level-ack receivers keep groups of 1.
  The host refuses G above a receiver's slots or a value other than 1/2/4. Unset, the host emits today's sizes (G
  from the slot count: 16 and 32 slots -> 4). Router v4txa11id (takes each receiver's G, ID count and first ID from
  the host), uniform TRID rule: the one VC0 receiver gets 16 IDs, so the R32 G1 arms share IDs (32 groups on 16
  IDs, group & 15).
- Default-identical gate (cg_gate.py): with the switch unset, the 769 host builds the same router images as the 649
  host: 728 of 728 builds per TRID rule (P150 2D, 1D, ring, Y torus, XY torus; Galaxy VC1 split, VC2 on, VC1 3072 B
  floor), compared on named args and every loadable segment; the gate tells the two TRID rules apart (80 of 80 P150
  2D builds differ). Switch vet (trid_vet.sh CG=1/2/4, both TRID rules, the same 8 configs): every config as the
  checker expects; the three refusals fire. Largest E1 code over those configs: unset 10160 B, G1 9852, G2 10320, G4
  10456 (E0 12108 throughout).
- Rows (knee/merge/*.yaml; unicast writes of 256 and 3072 B, 50000 packets per flow, 3072 B slots): on one link, k1 =
  the link's worker alone; k1f = one forwarded channel alone; k2 .. k4 add forwarded channels (straight, then the
  turns) into the same far receiver.
  - P150 2x4: link 5 -> 6 (worker 5, straight from 4, turn N from 1); k up to 3.
  - P150 4x2 (mesh descriptor knee/merge/p150_x8_4x2_mesh_mgd.textproto): link 2 -> 3 (worker 2, turns from 0 and
    4); k up to 3.
  - Galaxy 8x4 interior: link 13 -> 14 (worker 13, straight 12, turns from 9 and 17); k up to 4.
- Arms: (W, F) in {8,8; 12,12; 20,16} x R in {16, 32} x G in {1, 2, 4}, explicit rows. All at 3072 B slots (budget
  115; 20,16,32 needs 100) so every G and R comparison shares one slot size; R32 at 3072 B leans on D2's slot-size
  independence of the knee. The data-checked merge rows ran first on each system (passed). Both passes reversed;
  every arm mock-vetted with its row and G logged; each run's log checked for G on every receiver (the runners stop
  otherwise).
- Rates: each sender's own cycle count (the run log's per-sender lines; the CSV row carries only the busiest link's
  local sender). Flows on the merged link finish within 0.02% of each other (fair share), so each flow runs at
  k x the link total; link total = 1 / sum(1 / flow). Noise: the largest pass spread is 0.5 cycles (P150 2x4), 1.3
  (P150 4x2), 1.8 (Galaxy).
- Reports: knee/report_M24_777.md, report_M42_778.md, report_MG_175221.md (per arm, k, size: link total, per flow,
  rules).

Link total, cycles/packet, R16, G1 / G2 / G4 (R32 within -2.2 .. +2.4 cycles of R16 in every cell, below):

P150 2x4 (broker 777):

| 256 B (W,F) | k1 | k2 | k3 | k1f |
| --- | --- | --- | --- | --- |
| 8,8 | 189.5 / 203.8 / 230.2 | 127.8 / 127.0 / 132.4 | 125.9 / 125.5 / 111.3 | 202.7 / 218.7 / 248.3 |
| 12,12 | 137.4 / 142.7 / 155.2 | 126.3 / 125.0 / 113.1 | 125.9 / 125.3 / 111.0 | 160.6 / 155.6 / 168.9 |
| 20,16 | 132.5 / 122.8 / 113.0 | 126.2 / 125.0 / 113.3 | 125.9 / 125.2 / 111.1 | 160.3 / 151.6 / 143.6 |

| 3072 B (W,F) | k1 | k2 | k3 | k1f |
| --- | --- | --- | --- | --- |
| 8,8 | 210.4 / 217.5 / 241.6 | 130.4 / 132.9 / 139.9 | 126.2 / 126.0 / 111.5 | 224.9 / 236.2 / 259.5 |
| 12,12 | 145.2 / 150.4 / 164.0 | 126.5 / 125.3 / 113.1 | 126.1 / 125.5 / 111.2 | 161.5 / 165.7 / 179.5 |
| 20,16 | 132.5 / 123.0 / 113.2 | 126.4 / 125.3 / 113.3 | 126.1 / 125.5 / 111.3 | 155.7 / 151.0 / 142.8 |

P150 4x2 (broker 778):

| 256 B (W,F) | k1 | k2 | k3 | k1f |
| --- | --- | --- | --- | --- |
| 8,8 | 189.3 / 203.7 / 230.0 | 127.6 / 126.6 / 128.8 | 125.9 / 125.3 / 110.3 | 209.6 / 222.8 / 244.3 |
| 12,12 | 137.1 / 142.6 / 155.0 | 126.3 / 125.0 / 112.6 | 125.8 / 125.1 / 110.2 | 170.6 / 164.6 / 171.0 |
| 20,16 | 132.4 / 122.8 / 113.0 | 126.3 / 125.0 / 112.6 | 125.8 / 125.1 / 110.1 | 164.3 / 157.9 / 142.8 |

| 3072 B (W,F) | k1 | k2 | k3 | k1f |
| --- | --- | --- | --- | --- |
| 8,8 | 210.1 / 217.2 / 241.2 | 128.7 / 129.5 / 137.1 | 126.2 / 125.8 / 112.0 | 223.7 / 235.4 / 253.6 |
| 12,12 | 145.1 / 150.3 / 163.8 | 126.5 / 125.3 / 112.6 | 126.1 / 125.4 / 111.2 | 173.9 / 170.4 / 178.4 |
| 20,16 | 132.5 / 122.9 / 113.2 | 126.4 / 125.3 / 112.7 | 126.0 / 125.3 / 110.8 | 165.4 / 159.2 / 144.0 |

Galaxy 8x4 (Slurm 175221, bh-glx-110-a07u08, node-local caches):

| 256 B (W,F) | k1 | k2 | k3 | k4 | k1f |
| --- | --- | --- | --- | --- | --- |
| 8,8 | 200.8 / 214.0 / 238.8 | 129.1 / 128.1 / 131.3 | 126.0 / 125.8 / 111.5 | 126.0 / 125.2 / 111.2 | 209.0 / 219.8 / 244.0 |
| 12,12 | 142.6 / 149.6 / 162.0 | 126.5 / 125.2 / 112.9 | 126.0 / 125.4 / 111.2 | 125.9 / 125.2 / 110.8 | 160.5 / 155.3 / 168.3 |
| 20,16 | 132.5 / 122.9 / 113.5 | 126.3 / 125.2 / 113.0 | 126.0 / 125.3 / 111.2 | 125.9 / 125.2 / 110.8 | 160.3 / 150.7 / 142.9 |

| 3072 B (W,F) | k1 | k2 | k3 | k4 | k1f |
| --- | --- | --- | --- | --- | --- |
| 8,8 | 229.3 / 241.1 / 255.8 | 179.8 / 179.6 / 179.6 | 179.7 / 179.6 / 180.7 | 179.6 / 179.6 / 179.7 | 233.1 / 247.6 / 270.0 |
| 12,12 | 179.8 / 179.9 / 188.6 | 179.6 / 179.6 / 180.0 | 179.6 / 179.6 / 179.8 | 179.6 / 179.6 / 179.6 | 179.8 / 180.1 / 193.0 |
| 20,16 | 179.7 / 179.7 / 179.7 | 179.6 / 179.6 / 179.6 | 179.6 / 179.5 / 179.6 | 179.6 / 179.5 / 179.6 | 179.7 / 179.7 / 179.7 |

Readings:
- One channel alone (k1 = the worker, k1f = one forwarded channel), depth fixed: the best group size grows with the
  channel's depth, the same on all three systems at 256 B.
  - Depth 8: G1 < G2 < G4. G4 costs 21% over G1 on the P150 (230.2 against 189.5) and 19% on the Galaxy; G2
    costs 7%.
  - Depth 12: the worker is fastest at G1 (137.4 / 142.7 / 155.2). The forwarded channel is fastest at G2 (155.6
    against G1 160.6 and G4 168.9).
  - Worker 20 (unsqueezed): the order reverses, G4 < G2 < G1 (113.0 / 122.8 / 132.5). G2 costs 8.7% and G1 17%.
  - Forwarding 16 alone (k1f at 20,16): G4 is fastest (143 against 151-158 at G2 and 160-166 at G1). It still
    runs at 143 against the lone worker's 113, consistent with §12.2's forwarding knee above 16.
  - §12.9's crossover holds with the depth fixed: G4 costs worker 12 8.8% and saves worker 20 8.0%. G1 was not
    tested before; it beats G2 at depth 8 and at worker 12.
- The user's hypothesis holds: G4's cost on squeezed senders shrinks as k grows, then reverses. G4 against G2 at
  256 B (P150 2x4 / P150 4x2 / Galaxy):
  - (8,8): k1 +13.0 / +12.9 / +11.6%; k2 +4.3 / +1.7 / +2.5%; k3 -11.3 / -12.0 / -11.4%; k4 (Galaxy) -11.2%.
  - (12,12): k1 +8.8 / +8.7 / +8.3%; k2 -9.5 / -9.9 / -9.8%; k3 -11.4 / -11.9 / -11.3%; k4 (Galaxy) -11.5%.
  - At 3072 B the P150 shows the same pattern: (8,8) k2 +5.3 / +5.9%, k3 -11.5 / -11.0%.
  - The Galaxy's 3072 B rows sit at a ceiling of 179.5-180.7 cycles at every G and every k. Only the single squeezed
    channels sit above it (W8 k1: 229.3 / 241.1 / 255.8; W12 k1 and k1f at G4: 188.6, 193.0). No arm moves the
    ceiling (likely the link; not tested).
- Merged links settle at a plateau. Once the merged supply is enough, the link total is 125.8-128.6 cycles/packet
  at G1, 124.4-127.3 at G2 and 108.9-113.3 at G4. "Enough" means (12,12) from k2, (8,8) from k3, and (20,16) from
  k2. At G4 that is the lone unsqueezed worker's rate (113) or a little better.
  - It holds on all three systems, at both sizes on the P150 and at 256 B on the Galaxy, at R16 and R32.
  - G1 and G2 run nearly the same there (within 3 cycles), so the gap is not proportional to the number of groups.
  - Why G4 runs faster on a busy merged link, and why G1 and G2 then tie, is open. Which ERISC bounds the merged
    link was not tested (that needs an A/B that changes one side only). Neither question has a tested hypothesis
    here.
- Unsqueezed senders (20,16): G4 is the fastest at every k on all three systems. G2 costs 8.3-8.7% at k1 and
  10-14% at k2-k4; G1 costs 11-17%. A receiver of 16 or more slots could have groups of 2, but G2 would cost these
  senders, not help them.
- R32 against R16: within -2.2 .. +2.4 cycles in every cell on all three systems (pass spread up to 1.8).
  - The R32 G1 arms share IDs (32 groups on 16 IDs). They run 0 to +2.4 cycles slower than R16 G1 on some merged
    rows; the largest is P150 4x2 (8,8) k3 at 3072 B.
  - That is not isolated: the depth and the ID sharing change together, and the gap is within about twice the noise.
- Rules. Today's rule (G from R) gives G4 on every arm here. Rules A and B both give G2 on (8,8) and (12,12) and G4
  on (20,16), so they coincide on every arm (merge_report.py's rules table has every cell).
  - A single squeezed channel (k1, k1f at depth 8 and 12): A and B run 4-12% faster than today. G1 would gain more
    at depth 8 (18%).
  - Merged squeezed channels: A and B run 10-15% slower than today wherever the merge reaches the plateau ((12,12)
    from k2, (8,8) from k3), except on the Galaxy's 3072 B rows, which sit at the ceiling for every G. They run 2-6%
    faster only at (8,8) k2.
  - Unsqueezed senders: A and B equal today.
  - So neither rule beats today across the suite. Here the sign of the group effect follows how many busy channels
    feed the receiver, which no rule built from slot depths alone can see. This is an observation; no allocator
    decision follows from it.
- Corrections to earlier sections, marked WRONG where they were written:
  - §12.8's "with the worker unsqueezed, depth wins".
  - §12.9's "past the crossover G4 wins by the receiver's depth, not its grouping".
  - In both, the arms changed depth and group size together. With the depth fixed, most of the 10% gap is the group
    size.

### 12.11 1D large-slot check: does G4 lose on a lone flow at 8704 B? (user, 2026-11-04; broker 787)

Asked because today's rule (G from the receiver's depth) stays unless a 1D large-slot case shows a real exception.

Setup:
- P150 1D line, 8704 B slots, the default 1D row 13,12,16 (worker 13, forwarding 12, receiver 16; 41 of 41 slots)
  set explicitly.
- v4txa11id with TT_FABRIC_EXP_COMPLETION_GROUP = 1 / 2 / 4 (§12.10's switch; uniform TRID rule).
- Link chip 1 -> chip 2 (knee/merge/merge_p1d_8704_*.yaml): k1 = the worker alone, k1f = a pass-through from chip 0
  alone (chip 1's forwarding channel), k2 = both into chip 2's receiver; 256 and 8704 B packets, 50000 per flow.
- a09u32 broker, both passes reversed, the data-checked rows (8704 B, 30 packets per flow) first (passed); the yamls
  and all four arms mock-vetted, each run's log checked for its G on every receiver. Pass spread 0.1 cycles.
- Report: knee/report_L1_787.md.

Link total, cycles/packet, G1 / G2 / G4:

| packet | k1 (worker 13) | k1f (pass-through, forwarding 12) | k2 (both) |
| --- | --- | --- | --- |
| 8704 B | 243.8 / 243.8 / 243.8 | 243.8 / 243.8 / 243.8 | 243.8 / 243.8 / 243.8 |
| 256 B | 126.6 / 131.6 / 139.9 | 146.2 / 144.7 / 151.6 | 118.3 / 119.4 / 105.9 |

Readings:
- At 8704 B packets, the traffic these slots are sized for, G4 does not lose: every row runs at 243.8 cycles/packet
  at every G (equal to 0.01 cycles in both passes). That is 48.2 GB/s, about the link's line rate, so the rows are
  likely wire-bound (as §12.2's large packets).
- At 256 B packets on these slots, the §12.10 pattern repeats:
  - Lone worker (13 slots): G4 costs 6.3% against G2 and 10.5% against G1.
  - Lone pass-through (forwarding 12): G4 costs 4.8% against G2.
  - Two flows merged: G4 is the fastest, 11.3% ahead of G2 and 10.5% ahead of G1.
- No 1D large-slot exception for large packets. The small-packet cost on a lone flow is the same single-channel
  effect as §12.10, and it reverses on the merge. No allocator decision.

### 12.12 On the port: the forwarding knee, slot-size independence, and port against study (2026-11-04; lb2 175938, 175939)

**Measured on the port**: commit 22cdb1e55c3 plus fabric_exp/port_slots/slot_override.patch, a local patch that is
never committed. The patch adds the study's explicit-row switches TT_FABRIC_EXP_SLOTS_2D / _1D / _VC1 (same semantics)
and the slot log; nothing else.

Setup:
- **Host and tree:** built in /home/aagarwal/tt-metal-port-slots and staged lean at /data/aagarwal/lb2/trees/port_slots_22cd.
- **Node:** both jobs ran on lb2 bh-lb-120-a04u32, both passes reversed, BW telemetry off.
- **Run settings:** the port's own (TT_METAL_CLEAR_L1, profiler off); no study kernel or flag switch.
- **Vetting:** every arm was mock-vetted on the port host (fabric_exp/port_slots/port_mock_vet.sh) with its row logged, and every run checked its logged row.
- **Sanity:** the data-checked 1-hop rows at 3072 and 4352 B (2D, 1D) passed.
- **Noise:** pass spread at most 0.8 cycles (175938).
- **Knee rule:** the smallest swept depth within 1% of the sweep's best, as §12.2 defines it.
- **Report:** fabric_exp/port_slots/report_175938.md (port_slot_report.py).

Measurement 1, VC0 forwarding knee at 3072 B slots:

| kind, rows | swept | 256 B | 1024 B | 2048 B | 3072 B |
| --- | --- | --- | --- | --- | --- |
| 2D forwarding, W 20, R 16: 3-hop E/W, turn, row merge, multicast (2x4 mesh) | 12, 14 .. 26 | 23-24 (3-hop; others 15-18) | 19-20 (turn) | 19-20 (turn) | 17-18 (turn) |
| 2D forwarding, N/S 3-hop (the P150 as a 4x2 mesh) | 12 .. 26 | 21-22 | 21-22 | 21-22 | 23-24 |
| 1D forwarding, W 24, R 32: 3-hop, row merge | 16, 24 .. 56 | 17-24 | 17-24 | 17-24 | 17-24 |

- 2D, 2x4 mesh rows: 12 to 16 is the step (3-hop 156.9 -> 137.6 at 256 B; turn 164.3 -> 137.8). From 16 to 24 the rows
  stay flat within 0.3 cycles, and 26 is about 1 cycle (~1%) faster on every row and size, multicast and merge included.
  The 1% rule then reads 23-24 for the 3-hop at 256 B. The step at 26 is an observation; no cause is claimed, and 26 is
  the largest forwarding depth the 115-slot budget allows with W 20 and R 16.
- N/S multi-hop needs more: the 4x2-mesh N/S 3-hop keeps gaining up to 22-24 (143.0 at 16, 136.9 at 22, 136.1 at 26).
- 1D: 16 -> 24 gains 2-3%; 24 to 56 is flat within 0.4%. The 1D forwarding knee is 17-24, not above 24.

Measurement 2, the same depths at 3072 and 4352 B slots (256-2048 B packets, one host):

| kind | depths | knee at 3072 B | knee at 4352 B | 4352 against 3072, same depth |
| --- | --- | --- | --- | --- |
| 2D receiver, W 24, F 8 (1-hop E/W, N/S, E/W bidir) | 8, 12, 16, 24 | 13-16 | 13-16 | -0.6% .. +0.7% |
| 2D forwarding, W 20, R 16 (3-hop, turn, merge, multicast) | 8, 12, 15 | >12 | >12 | +0.1% .. +0.9% |

The receiver and forwarding needs are the same in slots at both slot sizes, as D2 showed for the 2D worker.
At 4352 B the forwarding arms run 0.4-0.9% slower at every depth (an observation, within the 1% rule).

Against earlier §12 readings: no conflict.
- §12.2's 2D forwarding at 4352 B (">12", 3-hop 13% better at 15 than 12) matches (>12 at both slot sizes here; the
  knee lands at 15-20 at 3072 B, where the budget reaches further).
- §12.2's 1D forwarding (">16, 24 is 6% better") matches 17-24 here, which bounds it above.
- §12.2's 2D receiver (9-16) matches 13-16 here (this sweep also has 12).
- One reading changes: the forwarding need is 16-20 on the 2x4 mesh rows, and the N/S multi-hop row on the 4x2 mesh
  needs 21-24. Neither the study nor this sweep set an upper bound for N/S multi-hop above 26.

Port against study (lb2 175939, the same host as 175938, two passes with the arms reversed, slot rows matched per suite):
- **STUDY:** v4txa11 with the study's run_cap settings (TT_FABRIC_KERNEL_VERSION=2, flags 13129088, keepalive 1000, -O3) and its default rows.
- **PORT:** the patched port with the override set to the study's row for each suite: 17,16,16 on mesh and tori, 26,25,32 on line and ring, NeighborExchange left as is (32/32).
- Both hosts logged the same slot rows on every suite, in both passes.
- **Not matched (no port switch):** the bubble flow control threshold on the half-ring row (port 16, study 2), and `-DL1_ECC_HALT` on the port. RECONCILE_HOST.md lists the rest; none of it is read by either router.
- Report: fabric_exp/port_slots/report_pvs_175939.md. Cycles/packet, mean of the two passes; each pass within 0.3 of the mean:

| suite | rows | port against study |
| --- | --- | --- |
| P150 4x2 XY torus, N/S (1-hop, bidir, 3-hop, wrap) | writes 256-4352 B | +6.8 to +7.6% (147.5-148.2 -> 157.8-158.7) |
| same | scatter writes 256-4352 B | +4.9 to +5.6% (186.7-187.8 -> 196.5-197.5) |
| P150 4x2 XY torus, 1-hop E/W | writes, scatter | -0.4 to -0.6% |
| P150 2x4 mesh, 1-hop N/S | atomic inc | +6.0% (132.3 -> 140.3) |
| same | write 16 B / 4352 B | +3.6% (117.1 -> 121.2) / +2.2% (127.5 -> 130.3) |
| same | scatter 4352 B | +5.4% (145.0 -> 152.8) |
| P150 2x4 mesh, 1-hop E/W | atomic inc | +0.1% |
| same | write 16 B / 4352 B | +3.0% (116.6 -> 120.2) / +0.9% (127.2 -> 128.3) |
| same | scatter 4352 B | +0.1% |
| P150 2x4 mesh, E/W bidir | write / scatter 4352 B | +0.3% / +0.1% |
| P150 2x4 mesh, 3-hop E/W | write / scatter 4352 B | -3.0% / -0.5% |
| P150 2x4 mesh, multicast | write / scatter 4352 B | -1.8% / +0.1% |
| P150 line (1-hop, bidir, N/S, 3-hop, multicasts) | 4352 B | -0.1 to +0.1% |
| P150 ring, half-ring multicast (threshold not matched) | 4352 B | -0.6 / -0.7% |
| P150 NeighborExchange | 4352 B | 0.0% |

Readings (no causal claims):
- With the slot rows matched, the rows that differ are the 2D rows. The N/S rows lose the most, from +2.2% to +7.6%, on
  the mesh and on the 4x2 torus. E/W 1-hop rows are within 1%, except the 16 B write (+3.0%). 3-hop E/W writes and
  multicast writes are faster on the port.
- Line, ring and NeighborExchange rows match within 0.7%.
- The route decode is the per-packet difference RECONCILE_DEVICE.md leaves unmeasured: the port has main's branchy
  decode_action_y_first; the study reads both bytes. These rows do not isolate it.
- The brief's two cases: the port does not match the study on the N/S rows, and the N/S rows are not the only ones
  that differ. The N/S losses (+2.2% to +7.6%) are the largest. The E/W 16 B write also loses (+3.0%), and some 2D E/W
  rows gain. A one-function A/B (port against port + both-byte decode) would isolate the decode's share. Its design goes
  to the user first. (Done: §12.13, lb2 176226. The decode closes the 2x4 N/S atomic and scatter gaps and the 4x2 torus
  N/S gaps.)

### 12.13 The route decode on the port: one-function A/B (user, 2026-11-04/05; lb2 176226)

**Question:** how much of §12.12's N/S gap (port against study) comes from the route decode? The port has main's
branchy `decode_action_y_first`, which loads Y, branches, and loads X only when Y is 0. The study loads both bytes
back to back, then selects (RECONCILE_DEVICE.md P1, A/B-1).

**The change.** A local patch, never committed: fabric_exp/port_slots/slot_override_decode.patch, which is
slot_override.patch plus the following.
- `decode_action_y_first` (hostdevcommon fabric_common.h) gets the study's three lines behind
  `#ifdef TT_FABRIC_EXP_DECODE_BOTH`.
- compute_mesh_router_builder.cpp sets that define on router kernels when the env TT_FABRIC_EXP_DECODE_BOTH is set.
- One host build serves both port arms.

**Offline gate.** Mock captures, decode off and on, on 7 configs: P150 mesh, 4x2 torus, 2x4 torus, line and ring;
Galaxy VC1 split 2x2 and Galaxy torus. Tool: fabric_exp/port_slots/decode_gate.py.
- **Check 1, pass.** With decode off, every router image is identical to the plain port + slot patch.
- **Check 2, pass.** With decode on, the only images that change are E1 receivers facing N/S, plus 2 E1 E-facing
  intermesh-landing builds on the VC1 split. No E0 image changes; nor do E/W receivers, line or ring. 1D routers get
  the define too (their compile lines carry it), but their images are identical.
- **At the decode site:**
  - off: `lbu Y; bnez; lw a5,12(sp); add; lbu X`;
  - on: two `lbu` back to back, then a select.
- **Check 3 (the rest of the loop unchanged): pass on the 2x4 mesh and 2x4 torus N/S receivers.**
- **Check 3 fails on the 4x2 torus and Galaxy torus N/S receivers.** With decode on, the completion proof reloads its
  NoC status base from the stack (`lw a2,8(sp)`, core-local, 1-2 cycles). Decode off keeps it in s10. On tori G is 1, so
  this runs once per packet.
- The user chose (a) + (c): one job with every row, the 2x4 mesh N/S rows read as the isolating result, and the 4x2
  torus N/S rows as the net effect of the source change, codegen included.
- (b) is out (no register pinning).
- **WRONG (2026-11-05), the gate report's line** "the off arm's in-loop stack load in the decode's X path runs only
  when the Y byte is 0, so not on the N/S rows' packets". On these rows the destination's Y byte is 0; see "Which
  decode path" below. The X-path reload `lw a5,12(sp)` runs on every N/S packet at its destination.

**Which decode path the measured packets take.** The 2D unicast traffic of test_tt_fabric sets its route with
`fabric_set_unicast_route` (tt_fabric_test_kernels_utils.hpp:463), which calls `widen_2d_route_to_chip`
(fabric_2d_route_interface.h:146). It does not use the single-hop helper (tt_fabric_api.h:301), which would put
LOCAL_DELIVER in the Y byte.
- **Destination receiver.** The host table builder (routing_2d_table_builder.hpp:88-99) writes STOP (0) on the
  destination's own Y entry. The widen puts LOCAL_DELIVER only in the X map at dst_x. So every N/S unicast packet's
  destination receiver reads Y = 0, then X.
  - Decode off: load, branch, stack reload, then the second load.
  - Decode on: both loads together.
- **Intermediate N/S hops** of a multi-hop route read a nonzero Y byte: decode off loads Y only, decode on loads both.
  The 4x2 3HopNS row (chip 0 to 6, y 0 to 3 on a ring of 4) runs at 1HopNS's rate. Whether the control plane routes it
  over the wrap in 1 hop was not checked. Either way, its destination reads Y = 0.
- **Multicast N/S hops.** `encode_2d_mcast_maps` ORs the teeth and the deliver bit into every target Y row, and
  forwarding rows carry N/S codes, so Y is never 0 there.
- **E/W receivers** decode X only, in both arms.

**Setup.**
- lb2 176226 on bh-lb-120-a07u28, three arms in one job, two passes with suites and arms reversed.
- STUDY and PORT are as in §12.12 (v4txa11 with run_cap settings; the port with the study's rows: 17,16,16 on mesh and
  tori, 26,25,32 on line and ring, NeighborExchange as is). PORTDEC is PORT plus TT_FABRIC_EXP_DECODE_BOTH=1, with its
  own kernel cache.
- Every run passed. All three arms logged the same slot rows on every suite, in both passes.
- Every pass-a port-arm compile was checked for the define: on 68/68 (mesh), 96/96 (4x2 torus), 68/68 (line, ring),
  56/56 (NeighborExchange); off 0 on all.
- STUDY and PORT reproduce 175939 (a04u32) within 0.1 cycles on the rows compared (for example, 2x4 N/S atomic
  132.3 / 140.3 on both hosts).
- Report: fabric_exp/port_slots/report_pvs_176226.md.

**Results.** Cycles/packet; per pass a / b; mean of the passes for the percentages. Gap closed = (PORT - PORTDEC) /
(PORT - STUDY).

| rows | PORTDEC - PORT, pass a / b | PORTDEC vs PORT | PORTDEC vs STUDY | gap closed |
| --- | --- | --- | --- | --- |
| **2x4 mesh, 1-hop N/S (isolating)**: atomic inc 4352 B | -7.6 / -7.5 | -5.4% | +0.3% | 96% |
| same: scatter 4352 B | -7.5 / -7.5 | -4.9% | +0.2% | 97% |
| same: write 4352 B (ns, c2d) | -1.7 / -1.6; -1.6 / -1.6 | -1.2 to -1.3% | +0.9 to +1.0% | 56-58% |
| same: write 16 B | -1.0 / -1.0 | -0.8% | +2.8% | 23% |
| **4x2 torus N/S (net, confounded)**: 1-hop, bidir, 3-hop, wrap; writes 256-4352 B | -10.6 to -11.9 | -6.7 to -7.5% | -0.4 to -0.5% | 106-108% |
| same: scatter 256-4352 B | -10.0 to -11.5 | -5.1 to -5.8% | -0.5 to -0.6% | 109-112% |
| 2x4 mesh E/W (1-hop, bidir, 3-hop) | -0.2 to +0.1 | -0.1 to +0.1% | as §12.12 | - |
| 2x4 mesh multicast (crosses N/S receivers) | -0.3 to +0.8 | -0.1 to +0.1% | write -0.7%, scatter -0.0% | - |
| 4x2 torus 1-hop E/W, writes and scatter | -0.1 to +0.2 | 0.0% | -0.4 to -0.6% | - |
| line, ring, NeighborExchange (identical images: null control) | -0.7 to +0.3 | -0.1 to +0.0% | as §12.12 | - |

**Readings.**
- **The isolating result.** On the 2x4 mesh N/S receivers the decode is the only code that changes (checks 1-3 pass).
  Reading both bytes makes the N/S rows faster, by 7.5-7.6 cycles on the atomic and scatter rows and 1.0-1.7 on the
  writes. The atomic and scatter gaps to the study close (96-97%). Pass to pass the deltas agree within 0.1 cycles.
  The null control (rows whose path keeps identical images: mesh and torus E/W, line, ring, NeighborExchange) moves
  at most 0.7 cycles (0.2%, one pass of a 1D multicast).
- **This is a one-side A/B.** Only E1 (N/S receiver) images change, so the 2x4 N/S atomic and scatter rows run at a
  rate the destination receiver's per-packet cost sets. The writes respond less: 1.0-1.7 cycles.
- **What is left on the mesh is not N/S-specific.** After the change, each 2x4 N/S row's offset from its E/W
  counterpart equals the study's within 0.3 cycles. The write rows keep the same gap to the study in both directions:
  16 B +2.8% / +2.9%, 4352 B +0.9-1.0%. The decode does not explain these.
  - 2026-11-05 (a09u32 job 798, RECONCILE_DEVICE.md §8.1): that gap is the test sender kernel, not the router. The
    study runs the stream loop and the port runs main's `send_packets_stateful`. With the study's loop in the port's
    sender, and router images identical, the port is -0.2 to -0.3% (4352 B) and -1.7% (16 B) against the study.
- **4x2 torus N/S, the net effect of the source change.** -10.0 to -11.9 cycles; the gap to the study closes fully,
  and PORTDEC runs 0.4-0.6% faster than the study, the same offset the E/W rows show.
  - The confound: decode on adds one stack reload per proof (`lw a2,8(sp)`, core-local, 1-2 cycles; G 1, so once per
    packet). It removes a branch and the X path's reload `lw a5,12(sp)`, which also runs once per packet here.
  - The confound's bound is about 2 cycles/packet either way, against the -10.0 to -11.9 cycles measured.
- **Mechanism, a hypothesis (not isolated).** Decode off issues the X load only after the Y load returns and the branch
  resolves: two chained L1 loads, 7-10 cycles each. Decode on overlaps them, which would save about one L1 load
  latency per destination packet. This is consistent with the 7.5-11.9 cycles measured on the receiver-bound rows.
- **Earlier results.** The study's v3 A/B (RESULTS.md §98; LEDGER E1-18 / E1-19) agrees: v3vd, with the X load behind
  the Y test, cost NS_PureAtomic +6.7 and N/S write 16 B +5.7, and v3ve, reading both bytes, removed it. This is the
  same change, now measured on the port.
- **For the port commit (proposed, not built).** Reading both bytes is plain source: no register pinning, and the
  buffer is already volatile.
  - RECONCILE_DEVICE.md §7 A/B-1 also lists the Galaxy N/S rows and the core census. Neither ran here; the upstream PR
    needs them.
  - The Galaxy torus N/S receivers have the same proof-base reload as the 4x2 torus.

### 12.14 The staged allocator on the port: offline gate (user, 2026-11-04/05; no device runs yet)

**The build.**
- Worktree /home/aagarwal/tt-metal-port-alloc, branch aagarwal/fabric-router-mainline-alloc, commit 4b524653490 on
  the cleanup tip e5b75d1fbcd. Local and unpushed.
- New: fabric_slot_allocator.{hpp,cpp}, plus `configure_staged_buffer_slots` in the static allocator. Blackhole takes
  the staged path after the MUX case. Wormhole, the MUX path and the spare-slot block keep the tier table.
- Stages: as the header describes. Receivers get the nearest power of 2, ties up. The default leftover order is
  [[VC0 forwarding], [VC0 worker]]: a group with no channels passes its leftovers on (user, 2026-11-04).
- The tier arm is the same build with a local switch, TT_FABRIC_EXP_TIER_SLOTS=1 (fabric_exp/port_slots/
  alloc_tier_switch.patch, never committed).

**Rows.**
- Tables: fabric_exp/port_slots/alloc_gate_rows_rc_4b52.md (24 reconcile configs) and alloc_gate_rows_cov_4b52.md
  (51 coverage cases).
- Every logged row equals staged_sketch.py's, with N taken from the build's slot total.
- NeighborExchange is the only shape the leftover-order amend changes. At 4352 B it gets W 51, R 32 (83 slots, none
  unused). The previous order gave 24 / 32 with 27 slots unused; the tier table gives 32 / 32.
  - The sketch gives 51,32. The rows tool skips one-sender shapes, so this was checked by hand.
  - All 72 NeighborExchange router builds carry 51 / 32 in their compile args.
  - The five other 32,32 allocator calls in that log appear in both arms, and no router build reads them.

| config | staged | tier |
| --- | --- | --- |
| 2D mesh/torus, 4352 B (N 81) | 19,16/15/15,16 | 17,16,16 |
| 2D, 3072 B (N 115) | 20,27/26/26,16 | 35,16,32 |
| 2D, 8704 B (N 41) | 9,8,8 | 9,8,8 |
| 1D line/ring, 4352 B (N 83) | 24,27,32 | 16,16,32 |
| 1D, 3072 B (N 117) | 24,61,32 | 32,32,32 |
| 1D, 8704 B (N 41) | 12,13,16 | 8,8,16 |
| NeighborExchange, 4352 B | 51,32 | 32,32 |
| VC1 split, 4352 B | 14,12/11/11,8 + 7,4 | 51,4,8 + 2,4 |
| VC1 split, 3072 B | 19,15,16 + 9,8 | 85,4,8 + 2,4 |
| VC1 split, 8704 B | 7,6,4 + 4/3/3,2 | 11,4,8 + 2,4 |
| VC1 split, 5 senders (Galaxy Z / dual-mesh CI) | 11,9,8 + 6/6/5/5,4 | 45,4,8 + 2,4 |
| VC2 on, 4352 B | 17,14/14/13,16 + 3,4 | 43,7,11 + 2,4 |
| VC2 on, 3072 B | 20,24/24/23,16 + 4,4 | 77,7,11 + 2,4 |
| VC2 on, 8704 B | 8,7,8 + 2,2 | 15,4,8 + 2,4 |
| VC1 + VC2, 4352 B | 13,10,8 + 7,4 + 3,2 | 45,4,8 + 2,4 + 2,4 |
| express (4 forwarding), 4352 B | 16,13/12/12/12,16 | 33,8,16 |

**Builds, size, stack (-O3, the port's default).**
- Every router builds: 3,494 router ELFs in each arm and no build errors.
- Coverage: largest E0+E1 is 25,940 B staged against 25,876 B tier (glx2_mesh_vc2_O3), with 684 B left in the 26,624 B
  kernel config buffer. Per case, staged minus tier runs from -8 to +176 B. No config stops fitting.
- Reconcile configs: largest 21,684 B staged (glx_vc1split_3072) against 21,548 B tier. Per case, -184 to +280 B.
- Least free stack is the same in both arms: E0 1,396 B and E1 3,632 B on coverage; E0 1,620 B and E1 4,232 B on the
  reconcile configs.
- Tables: alloc_gate_size_rc_4b52.md and alloc_gate_size_cov_4b52.md.

**Derived values.** The bubble threshold T = max(2, R - 16) and NUM_TRANSACTION_IDS follow the final receiver counts:
0 mismatches. This was checked on the [[VC0 forwarding]] build; the amend changes no receiver count.

**Next, after the user's go:**
- **Device A/B, P150 via lb2:** staged against tier, two passes reversed, same host per comparison. Suites: 2D,
  torus, 1D, VC1 split and VC2 on at 4352 and 3072 B, plus 8704 B.
- **Galaxy:** VC1 split (2x2, 4x4) and VC2 rows.
- **Acceptance, proposed:** never more than 1% slower than the tier table in any suite; mixed rows scored by link
  total.

**Device A/B, first attempt (2026-11-05): HANG. Stopped; the fix needs the user's go.**
- **The runs:**
  - lb2 176405 (bh-lb-120-a07u28; fabric_exp/port_slots/alloc_runs.sh) hung on its first run: paa_staged_pval2d_4352, the data-checked 1-hop E/W write at 4352 B with `sync: true`, on the staged rows (VC0 19,16/15/15, R 16). It stopped at "Waiting for programs" and was killed after 300 s. The hang was captured, and the node reset with links OK.
  - The tier arm never ran.
  - The receiver-tuning agent's lb2 176425 (bh-lb-120-a04u24; 4b524653490 + port_tune.patch, staged rows by default) hung the same way on its first data-checked row (ValMesh1HopEW_4352).
  - Galaxy 176408 never started; I cancelled it while it was still pending.
- **Mechanism, a hypothesis (no isolating A/B yet): a router contract the staged rows break.**
  - The port's router keeps one forwarding-ring size per VC: router_downstream.hpp, `DownstreamTraits<VC>::num_slots = DOWNSTREAM_SENDER_NUM_BUFFERS_ARRAY[VC]`. The local downstream credit copy and the ring wrap of every downstream forwarding channel on that VC come from it.
  - The host emits that size from whichever downstream connection was added last (static_sized_channel_connection_writer_adapter.cpp:70, `downstream_sender_channels_num_buffers[vc]`).
  - Stage 2's largest remainder gives a VC's forwarding channels different counts (VC0 16/15/15 at 4352 B, 27/26/26 at 3072 B; 14/14/13 with VC2 on; 12/11/11 on the VC1 split). A receiver forwarding into a 15-slot channel with a 16-slot ring overruns it, and its credits drift.
  - The tier table and every explicit-row switch run so far give one forwarding count per VC.
  - The 1-hop rows forward too: the test's sync traffic crosses the mesh through forwarding channels.
- **Not confirmed from the capture.** hang_capture.py and hang_decode.py only know the study's `fabric_erisc_router_v2` kernel directory, so they kept no router configs for the port's `fabric_erisc_router`.
- **Proposed (user's call):**
  - (a) One forwarding count per VC in the allocator. Stage 2 treats a VC's forwarding channels as one unit. Stage 3 gives that group leftovers in multiples of its size, with the remainder to the VC0 worker. Receivers do not change.
  - (c) A host TT_FATAL when a VC's forwarding channels differ.
  - Not proposed: (b), per-downstream ring sizes in the router.
  - Rows under (a), from the sketch:
    - 2D: 20,15,16 at 4352 B; 21,26,16 at 3072 B.
    - VC1 split: 15,11,8 + 7,4 at 4352 B.
    - VC2 on: 18,13,16 + 4,4 at 4352 B; 22,23,16 + 4,4 at 3072 B.
    - 1D and NeighborExchange: unchanged.

**Evidence from the hang capture (2026-11-05).**
- **Tool fix:** hang_capture.py, hang_decode.py and vc1_capture_lib.sh now also accept the port's router kernel directory (`fabric_erisc_router`), not only the study's v2 slot. Before this they kept no router configs for port runs.
- **Decode:** I re-decoded 176405's capture with router configs rebuilt on a mock of the node: same build, pval2d_4352, a07u28's cluster descriptor. The decoder matched them by code anchor. Output: fabric_exp/port_slots/hang176405_decode.txt.
- **Finding:** 8 receivers show their local copy of a downstream sender channel's free-slots register OUT OF RANGE. The downstream channel reads 16 while the receiver's ring is 15 slots, so more increments than decrements: over-credit.
- **Reading:** this fits the mechanism above. Those receivers' adapters took a 15-slot ring (the last connection's count) while one downstream channel has 16 slots.
- **Caveat:** the configs come from a mock, not the run's own cache, which the capture did not keep. The decoder also notes 28 channel-number differences between the mock and the node.

**The fix, (a) + (c) (user-approved 2026-11-05), amended into the allocator commit: c53b5a5cbbd (local, unpushed).**
- **Stage 2:** a VC's forwarding channels are one unit with one count. The unit takes the floor of f' x depth on every channel and grows by a slot on each at a time, with largest remainder at the unit level. Each worker is a unit of its own.
- **Stage 3:** leftovers go out in whole rounds (+1 on every channel of a group). A group with no channels passes it on. The default order ends with the VC0 worker, which takes what no whole round fits.
- **Router contract:** the header now states it. The router has one downstream forwarding ring size per VC (DownstreamTraits::num_slots).
- **Guard (c):** static_sized_channel_connection_writer_adapter.cpp now raises a TT_FATAL where it fills that per-VC size, if a receiver's downstream connections on one VC have different slot counts. It covers every slot path: tier, staged and explicit rows.
  - **Forced mock:** a local switch (TT_FABRIC_EXP_SKEW_FWD=1, alloc_tier_switch.patch) takes a slot from the first VC0 forwarding channel. The mock then stops at the guard with "VC0 forwards into downstream sender channels of 14 and 15 slots, but the router keeps one downstream ring size per VC ...".
  - **Note:** taking the slot from the last channel instead did not trip the guard. On the 2x4 mesh no receiver forwards into the last channel together with another one; the guard fires only when a receiver mixes sizes, which is exactly the router's contract.
- **Sketch:** staged_sketch.py follows the same unit rule and matches the C++.
- **Offline gate:**
  - All 24 reconcile configs log the sketch's rows; every VC's forwarding channels now have one count. The changed rows:

    | config | 4352 B | 3072 B | 8704 B |
    | --- | --- | --- | --- |
    | 2D | 20,15,16 | 21,26,16 | 9,8,8 |
    | VC1 split | 15,11,8 + 7,4 | 19,15,16 + 9,8 | 8,6,4 + 3,2 |
    | VC2 on | 18,13,16 + 4,4 | 22,23,16 + 4,4 | 8,7,8 + 2,2 |

  - 1D, ring and NeighborExchange rows are unchanged, and so is the tier arm.
  - Size at -O3 over the reconcile configs: largest E0+E1 21,684 B staged against 21,548 B tier. Least free stack E0 1,620 B and E1 4,232 B in both arms.
  - The P150 suite mock vet (yamls/alloc_expected_rows.json) passes.
  - Coverage (51 cases per arm) is running.
- **P150 stage:** /data ran short of inodes on 2026-11-05, so the tree is staged as one file, /data/aagarwal/lb2/trees/port_alloc_c53b.tar.zst (+ .sha256). alloc_runs.sh unpacks it on the node. The Galaxy sbatch can unpack a tarball on node-local /tmp (TREE_TAR, EXTRA_TAR).
- **Coverage gate on c53b5a5cbbd (done):**
  - All 51 cases in both arms build: 7,544 router ELFs per arm, no build errors.
  - The guard never trips on a real configuration, including mux, trimming, one-ERISC, UDM and the multi-mesh cases.
  - The staged rows equal the sketch. ci_nsew's one "13,9,8 + 0,4" row is the rows tool pairing interleaved multi-rank log lines: the raw lines are 13,9,9,9,9 / 8 and VC1 5,5,5,5 / 4.
  - Largest E0+E1 is 25,932 B staged (glx2_mesh_vc2_O3, 692 B left) against 25,876 B tier.
  - Least free stack is equal in both arms: E0 1,396 B, E1 3,632 B.
- **Device:** P150 lb2 176643 runs the data-checked 2D rows that hung first, then the staged-vs-tier suites, two passes reversed. Galaxy Slurm 176650 is afterok on it; its dry run through the tarball path passed all 20 steps.

**Device A/B on c53b5a5cbbd: P150, lb2 176714 (bh-lb-120-a07u28, 2026-11-05)**
- **Run:** two passes reversed, one host. Clean: all 51 runs passed, slot rows as vetted, no hang. The data-checked rows passed first, including pval2d_4352, which had hung on 4b524653490.
- **Harness, local, both arms:** test_infra_6dad.patch and adapter_vc2_credits_6dad.patch. Router images are unchanged by them (8 configs recaptured, 0 differ).
- **Earlier attempt, lb2 176643:** it stopped on main's bandwidth profiler, which has no hops for VC1 inter-mesh rows (std::bad_optional_access). That is the test harness, not the allocator. Galaxy 176650 was cancelled before it started.
- **Report:** fabric_exp/port_slots/report_alloc_176714.md.
- **Per suite**, staged vs tier, mean of the passes, mixes by link total. The passes agree within 0.5% on every row.

| suite | rows | wins / losses (> 1%) | median | worst | proposed bar (no suite > 1% slower) |
| --- | ---: | --- | ---: | --- | --- |
| 2D 4352 B | 14 | 6 / 3 | +0.0% | +9.0% 3-hop E/W 4352 B | NOT met |
| 1D 4352 B | 6 | 4 / 0 | -7.4% | -0.9% | met |
| ring 4352 B | 14 | 1 / 0 | -0.1% | +0.0% | met |
| VC1 split + inter-mesh + mixes 4352 B | 29 | 17 / 0 | -18.8% | +0.3% | met |
| VC2 on 4352 B | 11 | 11 / 0 | -18.6% | -14.7% | met |
| NeighborExchange 4352 B | 2 | 0 / 0 | -0.1% | -0.0% | met |
| 2D 3072 B | 14 | 1 / 2 | -0.1% | +3.5% 1-hop N/S 3072 B | NOT met |
| 1D 3072 B | 6 | 0 / 2 | -0.3% | +1.7% 1-hop E/W 3072 B | NOT met |
| ring 3072 B | 14 | 0 / 0 | -0.0% | +0.1% | met |
| VC1 3072 B | 29 | 29 / 0 | -38.3% | -9.6% | met |
| VC2 on 3072 B | 11 | 11 / 0 | -19.7% | -15.7% | met |
| NeighborExchange 3072 B | 2 | 0 / 0 | -0.1% | -0.0% | met |

- **The losing rows and their slot rows** (no causal claim beyond the slots):
  - 2D 4352 B, staged 20,15,16 against tier 17,16,16:
    - losses: 3-hop E/W +5.5% (1024 B) and +9.0% (4352 B); Turn4 4352 B +6.7%.
    - wins: every 1-hop row, -1.3% to -6.8%.
  - 2D 3072 B, staged 21,26,16 against tier 35,16,32: 1-hop N/S +2.5% / +3.5%. 1-hop E/W is -0.2% and 3-hop E/W -2.5%.
  - 1D 3072 B, staged 24,61,32 against tier 32,32,32: 1-hop E/W +1.4% / +1.7%.
- **Galaxy:** Slurm 176715 (VC1 split 2x2 / 4x4, VC2) started on its afterok dependency.

**Device A/B on c53b5a5cbbd: Galaxy, Slurm 176753 (bh-glx-110-a01u20, 2026-11-05).**
- **Run:** one allocation, two passes reversed, staged vs tier.
  - Tree: port_alloc_c53bh.tar.zst unpacked on the node's /tmp, with glx_extra_c53b.tar.
  - All 20 steps passed, every rank, slot rows as captured.
  - An earlier submit, 176715, stopped at its first step: my sbatch gave the yaml as a relative path, which the test binary reads under its test_yamls directory. That is fixed, and the dry-run stub now rejects relative paths.
- **Report:** fabric_exp/port_slots/report_alloc_glx_176753.md.

| suite (rows) | slot rows: staged / tier | wins / losses (> 1%) | median | worst | bar |
| --- | --- | --- | ---: | --- | --- |
| VC1 split 2x2, 4352 B (37) | 15,11,8 + 7,4 / 51,4,8 + 2,4 | 25 / 0 | -50.6% | +0.3% | met |
| VC1 split 2x2, 3072 B (37) | 19,15,16 + 9,8 / 85,4,8 + 2,4 | 37 / 0 | -58.2% | -9.6% | met |
| VC1 split 4x4, 4352 B (42) | 15,11,8 + 7,4 / 51,4,8 + 2,4 | 36 / 0 | -58.9% | +0.2% | met |
| VC2 on, 4352 B (4) | 18,13,16 + 4,4 / 43,7,11 + 2,4 | 4 / 0 | -47.9% | -16.2% | met |
| VC2 on, 3072 B (8) | 22,23,16 + 4,4 / 77,7,11 + 2,4 | 7 / 0 | -46.0% | +0.0% | met |

### 12.15 The 1D leftover order: worker and forwarding together (user, 2026-11-05)

**Change (user, 2026-11-05):** 1D routers (line, ring, NeighborExchange) get their own default leftover order,
`[[VC0 receiver], [VC0 worker, VC0 fwd]]`. The worker and the forwarding channel take +1 each in turn, worker first.
2D keeps `[[VC0 receiver], [VC0 fwd], [VC0 worker]]`. The receiver step and the 32-slot cap still come first.
- **Code:** `default_leftover_order(bool one_d)`. `allocate_slots(shape, limits)` uses the order for `shape.one_d`.
  Stage 3's senders now take +1 a unit at a time (stage 2's units: each worker, and a VC's forwarding channels
  together), unit after unit in the group's order. A one-kind group still takes whole rounds, so 2D is unchanged.
- **Commits:**
  - port-alloc: 9ce0acbeaba on e5b75d1fbcd (amends ae77c88b4c6), for the tuning agent's rebase.
  - Final pair on 57c55f98118: allocator 07322400fd6 (amends 6efabd47b4f), then TRID c3c1813c42a (077a0b9b200 re-picked
    without conflicts). The allocator files are identical in both.
- **Rows** (sketch = port, mock gate): 1D 4352 B 24,27,32 -> 26,25,32; 1D 3072 B 24,61,32 -> 43,42,32. The Galaxy 1x32
  line and ring (4416 B, N 82) go from 24,26,32 to 25,25,32. Unchanged: NeighborExchange (51,32 / 85,32), 1D 8704 B
  12,13,16 and 15232 B 7,8,8 (no leftover there), and every 2D, VC1 and VC2 row.
- **Receivers, T and TRIDs do not change** (alloc_derived.py on the port-alloc captures, against the capped build, all 1D
  cases): R stays 32; line T -, ring T 16; uniform ids 16 (line) / 8 (ring); G 4 (line) / 1 (ring); rule ids 8 / 16.

**Offline gate on 9ce0acbeaba (port-alloc), 2026-11-05:**
- Mock rows, staged against tier (rc_capture.sh: p150_line 4352 / 3072 / 8704, p150_ring 4352 / 3072, p150_ne_4352):
  every staged row equals the sketch; tier rows unchanged (16,16,32 / 32,32,32 / 8,8,16).
- P150 suite vet (alloc_mock_vet.sh: 1d, ring, ne at both sizes, 2d_3072 as the control) passes; 2d_3072 still
  20,21,21,21 / 32. yamls/alloc_expected_rows.json updated for those suites.
- Builds: 72 router builds per case, both arms, no errors.
- Size and stack (alloc_gate_size.py, the same cases): 1D 3072 B E0 +8 B against the capped build (7,804 against
  7,796 B), ring 3072 B E0 +8 B; 4352 and 8704 B identical. Largest E0+E1 14,116 B in every arm. Free stack unchanged
  (E0 2,020 B, E1 4,496 B).

**Offline gate on the final pair (07322400fd6 + c3c1813c42a on 57c55f98118), 2026-11-05:**
- Reconcile captures of the same six 1D configs, rule and uniform arms: rows equal the sketch; the TRID ID table is
  identical to the pre-amend pair's (line 32 slots: G 4, 8 groups, rule 8 / uniform 16 IDs; ring: G 1, 32 groups,
  rule 16 / uniform 8, shared by mask); E0 images identical between the arms, E1 differ only by the IDs, as before.
- Coverage, every case with a 1D router (14: bh 1d / ring / one-ERISC / neighbor / mux / trim, the trimming gtest,
  Galaxy 1d / ring / 1x32 line and ring): all build (1,292 router ERISC builds), rows equal the sketch, TRID rows
  identical to the pre-amend coverage. bh_1d_1erisc's mock run once ended in a segfault after launch (in the mock
  dispatch's go-signal write) instead of the usual result-check abort; a rerun ended as usual, with the same 38
  builds. The 37 other coverage cases have no 1D router; their pre-amend run (covfin) stands.
- Size: the largest image and the least free stack do not change (E0+E1 16,444 B, bh_trim_gtest; free stack E0
  1,876 B, E1 4,312 B); 1D images move by -8 to +8 B.

**Device runs (one P150 lb2 job, then one Galaxy allocation afterok on it):**
- **P150, lb2 176985** (staged against tier, two passes reversed, one host). The data-checked rows run first on the
  staged arm (fabric_exp/port_slots/mk_p1d_yamls.py): pval1d 4352 / 3072, pvline 4352 / 3072 (val1d_m1's Linear tests,
  1-2 links), pvring 4352 / 3072 (val_ring_p150, 1-2 links). Then the suites 1d_4352, 1d_3072, ring_4352, ring_3072.
  No flow-control rows.
  - An earlier submit, lb2 176973, stopped after its third data-checked row. The row passed (38 executed, 0 failed),
    but my runner expected 19: my count missed that LinearMulticastMicroLooped's 20 pattern iterations each count as a
    test. mk_p1d_yamls.py now counts them.
  - **Result (bh-lb-120-a07u28, 05:31-05:35 UTC): clean.** All 6 data-checked rows passed on the staged arm (pval1d
    1 / 1 each, pvline 38 / 38 each, pvring 19 / 19 each). All 16 suite runs passed with slot rows as vetted. No hang.
    The passes agree within 0.1% on every row. Report: fabric_exp/port_slots/report_alloc_176985.md.

| suite (rows) | staged / tier | wins / losses (> 1%) | median | worst | bar |
| --- | --- | --- | ---: | --- | --- |
| 1D 4352 B (6) | 26,25,32 / 16,16,32 | 4 / 0 | -7.3% | -0.8% (row merge 1024 B) | met |
| 1D 3072 B (6) | 43,42,32 / 32,32,32 | 0 / 0 | -0.3% | -0.1% (row merge 3072 B) | met |
| ring 4352 B (14) | 26,25,32 / 16,16,32 | 1 / 0 | -0.0% | +0.0% | met |
| ring 3072 B (14) | 43,42,32 / 32,32,32 | 0 / 0 | -0.0% | +0.0% | met |

  - **Against the capped build's rows (24,27,32 / 24,61,32), cross-job on the same host (lb2 176714, also a07u28):**
    tier drifts at most 0.1% between the jobs on every row.
    - 1D 3072 B 1-hop E/W: -1.7% (1024 B) and -2.0% (3072 B). 176714's +1.4% / +1.7% loss against tier is now
      -0.3% / -0.3%.
    - Every other 1D and ring row moves within +-0.2%; 1D 4352 B 1-hop 1024 B -0.6%.
    - These are slot rows only: no claim on why the worker depth moves the 1-hop row.
- **Galaxy, Slurm 176994** (alloc_glx.sbatch JAM=1; port_alloc_9ce0h.tar.zst + glx_extra_jam_9ce0.tar; 110-a only,
  standing excludes). RING_JAM_FIX.md 10.3's 172983 rows, staged against tier: glx_frg_val (data-checked, both arms,
  first), then frg_nsrx (8-ring 3072 / 3584 / 4352 B), frg_ew (4-ring 4352 B) and frg_short4k (ring 1-hop, half ring),
  pass a staged then tier, pass b reversed. Ring rows: staged 26,25,32 against tier 16,16,32 (R 32, T 16 in both); the
  validation's XY torus is 17,16,16 in both arms. The stub dry run passed all 14 steps.
  - **Result (bh-glx-110-a01u02, 05:36-05:43 UTC): clean.** All 14 steps exit 0, slot rows as captured, no hang.
    glx_frg_val 10 / 10 on each arm. Report: fabric_exp/port_slots/report_alloc_glx_176994.md.

| row | tier | staged | staged vs tier | 172983 branch (120-c) | 172983 base, jammed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8-ring 3072 B (control) | 194.2 / 194.3 | 190.8 | -1.8% | 194.7 / 194.8 | 195.6 |
| 8-ring 3584 B | 227.9 / 228.1 | 209.0 / 209.1 | -8.3% | 228.5 / 228.4 | 310.1 / 312.6 |
| 8-ring 4352 B | 269.4 | 254.1 | -5.7% | 269.6 / 269.3 | 375.1 / 372.2 |
| 4-ring 4352 B | 254.9 | 255.1 | +0.1% | 254.9 / 255.0 | 323.7 / 339.9 |
| half ring 4352 B | 253.6 | 253.7 | +0.0% | 253.6 | 331.5 / 332.1 |
| ring 1-hop 4352 B | 249.6 | 249.5 | -0.0% | 249.5 | 249.5 |

  - No row reaches the jam level in either arm or pass; the passes agree within 0.1%.
  - Tier here matches 172983's branch within 0.3% on every row, on a different node class.
  - Staged matches RING_JAM_FIX §9.5's A16 arm (the study, 26,25,32 and T 16, 110-a) within 0.5% on every row
    (209.1 / 254.2 / 255.1 / 253.8 / 249.5; 3072 B 191.6).
  - The 8-ring rows and the 3072 B control are 1.8-8.3% faster on staged. Slot rows only (same node, same job); no
    claim on the mechanism.
