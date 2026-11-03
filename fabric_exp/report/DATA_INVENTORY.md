# Data inventory: the study router (v4) against main's router (2026-11-02)

The bundle compiles existing measurements only. No job, build or Slurm run was made for it. Every record in `data/*.json`
names its source file, section and job. The schema is in `data/SCHEMA.md`. The builder scripts in `scripts/` read the
study's result files and write only into `report/data/`; `scripts/finalize.py` merges and checks the bundle.

**Routers.**
- "main" is main's router as shipped: kernel v1, `TT_FABRIC_EXP_FLAGS=0`, firmware keepalive, the builder's opt
  level and slot tables.
- v4 records name their variant. Leads over time: v4trm (10-29), v4pl6 (10-30), v4rxv2i3 (11-01), v4rxf4 (11-02),
  v4rxg3 (now).
- v4rxg3 has the same per-packet code as v4rxf4 on every production config: g1 is byte-identical, g2 is layout-only,
  g3 changes only mock-captured trimmed shapes (V4_RECEIVER_OWNERSHIP §18). v4rxf4 numbers therefore stand for the lead.
- The sender pass on v4rxg3 is in progress. Jobs 451-456 (`txo_runs.sh`, census sets with v4txo* variants) were queued
  or running at 19:05. **Every v4 number here predates the final lead's E0 code.**

**Comparison rule.** `same_job_with_main` is true only when main and the v4 row ran in the same broker or Slurm job.
Job numbers for the CCL runs were checked against the broker's job list (`tt_device_recent_jobs`).

## What exists, per dataset

| file | records | content | newest data | lead it is from | same job with main |
| --- | ---: | --- | --- | --- | --- |
| bandwidth.json | 1,179 | P150: 13 census suites, 234 rows, plus suite means. Galaxy: mesh / line / ring / multicast sweep | P150 same-job 2026-10-29; P150 v4rxf4 2026-11-02; Galaxy same-job 2026-10-29 | v4trm (same job), v4rxf4 (cross-run); Galaxy v4trm (same job), v4rxv2i3 (cross-run) | P150 jobs 164-171; Galaxy Slurm 163369 (and pre-merge 155833-158562) |
| wire.json | 72 | frame model at 400G / 200G, both header sizes; quoted figures checked; bare-link probe; rows at or near the wire | 2026-11-02 | v4trm / v4rxf4 (P150), v4rxv2i3 (Galaxy line rate) | P150 rows from jobs 164-171 |
| slack.json | 1,028 | one-side delay A/Bs: base rate, E0 and E1 slack, bound side per row | 2026-11-01 | v4trm (jobs 172-176), v4rxv2i3 (jobs 349, 357); main only RESULTS §2 (2026-10-09, 2 KiB, pre-merge) | none |
| latency.json | 48 | exp_latency.yaml: 1-hop / 3-hop E/W, write / fused, 16 B / 4096 B, median round trip | 2026-11-01 | v4rxv2i3 (job 373) | **none**: main as shipped only on a04u24, pre-merge |
| ccl.json | 279 | DeepSeek decode CCL op times per pass, plus pass means | 2026-11-02 | v4rxf4, v4rxo0b | SDPA only (job 440 pass a; job 444 SDPA 2048 B passes c-e) |
| size.json | 623 | code size and least free stack per coverage case, main against v4rxg3 at -O3 and -Os, plus V4_SIZE figures | builds 2026-11-02 | v4rxg3 | n/a (offline builds) |
| slots.json | 141 | default depths per fabric, main's code constraints, measured slot tables | 2026-11-01 | v4u, v4gi (same job as main), v4rxv2i3 | jobs 076-077 (1D), job 112 (VC1) |
| stability.json | 427 | register-map guard headroom cells, spills, fragility, opt-level deltas, jitter band | 2026-11-02 | v4rxf1 ... v4rxg3, sender probes | main: VC1 opt levels and size only |
| fanin.json | 368 | VC2 (neighbour / line mode), VC0+VC2 contention, combined fan-in probes, mux rows, VC2 line on Galaxy | 2026-11-02 | v4rxv2i3, v4txp* probes, vc2l_p3b (VC2 line worktree) | the mux census rows (jobs 164-171); main cannot build VC1+VC2 or VC2 on one ERISC |
| gaps.json | 55 | rows where v4 trails main (P150, Galaxy), plus measurements missing for the final lead | | | |
| galaxy_v4_only.json | 553 | Galaxy v4 rows with no main: line-rate sweep, N/S floor A/Bs, slot-size grid, VC2 rows | 2026-11-02 | v4rxv2i3 (+ diagnostic v4bddsl / v4bddml) | none |

### Bandwidth (P150)

- **Same-job census** (a09u32 jobs 164-171, 2026-10-29): two passes, routers back to back. Per suite: main as shipped,
  main at the study keepalive (1000 cycles), v4trm.
  - v4trm's production images equal v4rs's, except the 22 VC1 senders (V4_PERF_GAPS).
  - No row moved more than 3% between passes, except main's one-ERISC HalfRingMulticast write (flagged in notes).
- **Latest lead census** (jobs 420 / 421, passes f4a / f4b, 2026-11-02): v4rxv2i3 and v4rxf4 only.
  - The v4rxf4 records are **cross-run against main**.
  - The host was rebuilt twice since main's column was taken (jobs 399, 448: study-router args, VC1 at -O3 for the
    study router). Main's v1 path was not re-measured after either rebuild.
- **"As shipped" differs on three settings, not just the router.** Each record says which applies:
  - keepalive: main uses the firmware's 8000, v4 the study's 1000;
  - VC1 slots: main uses its tier table 51,4,8,2,4, v4 uses 19,9,16,5,4;
  - VC1 opt level: main -Os, v4 -O3.
  - The study-keepalive column separates the keepalive from the router (main's 1D 1-hop write: 187.6 at the firmware
    keepalive, 168.4 at 1000).
  - Job 112 (slots.json) separates the slots: main and v4gi ran at the same VC1 depths.
- Suite means (`rate_suite_mean`) reproduce census_report.py exactly: V4_PERF_GAPS's table, and V4_RECEIVER_OWNERSHIP
  §17 for v4rxf4.

### Bandwidth (Galaxy, 200G)

- **Most recent same-job pair:** Slurm 163369 (2026-10-29, bh-glx-120-c01u14, post-merge trees), main against v4trm,
  two passes. Mesh 27 rows, line 10, ring 5, full-ring multicast sweep 6.
- **Older same-job pairs** (pre-merge trees, 2026-10-27): jobs 155833, 156067, 157929, 158477, 158562, with v4l / v4u /
  v4uf / v4wt / v4eg / v4p / v4p2. These are the history of the full-ring multicast gap. v4p / v4p2 hold completions and
  collapse to 17,000-20,500 cycles/packet; that is a rejected design, not a lead.
- **Cross-run pairs:** 1-hop E/W and N/S at 1536-4096 B. Main from 155281 (pre-merge base); v4rxv2i3 from 167648.
- **Galaxy values:** the mean of the passes, as on the P150; `value_pass1` keeps pass 1.
- **The lead has never run on a Galaxy against main.** Neither has v4pl6. v4rxv2i3 ran only in the slot study, without
  main.

### Wire

- Model (RESULTS §29 frame model):
  `wire_cycles = (P + H + 56 * ceil((P + H) / 4114) + 84) / (Gbps * 1e9 / 8 / 1.35e9)`.
  - H = 96 B (2D, VC1, VC2) or 48 B (1D).
  - 37.04 B/cycle at 400G, 18.52 at 200G.
- It reproduces every quoted figure within 0.3 cycles:

  | link, payload | quoted | model |
  | --- | ---: | ---: |
  | 400G, 4096 B | 118.5 | 118.5 |
  | 400G, 4352 B | 125.4 | 125.4 |
  | 200G, 4352 B | 251.0 (measured 252.1) | 250.8 |
  | 200G, 3072 B | 178.9 (measured 179.6) | 178.6 |
  | 200G, 2048 B | 123.5 | 123.3 |

- The slot study's fit, 12.80 + P / 18.50 (+3.0 above ~4 KiB), is the same model: 12.80 cycles x 18.5 B/cycle = 237 B =
  96 + 56 + 84, and +3.0 is the second 56 B frame.

### Slack (router faster than the wire)

- Bound sides come from one-side delay spins, which are isolating A/Bs (CLAUDE.md), so they are recorded as `A/B`.
- For "neither" rows, the notes name what binds only where a doc settled it by A/B:
  - 1D 1-hop: the wire;
  - mux rows: the mux;
  - VC1 inter-mesh 1_1..1_3: the VC1 windows, job 360.
- Slack values are per variant: v4trm (jobs 172-176) and v4rxv2i3 (jobs 349, 357). Neither is the lead. The codec stage
  (v4rxf3) added 2-4 cycles to E/W and forwarded small-packet rows after these runs.
- Main has no slack data on the current tree. RESULTS §2 is main at 453ee7b (2026-10-09), 2 KiB, spins of 30 / 60
  cycles; it is context only.
- Galaxy bound sides (V4_NS_FLOOR §1, job 167648, A/B) are in `galaxy_v4_only.json`.
  - E1 binds every N/S-involving row (~172 cycles/packet at 1-2 KiB).
  - E0 binds E/W rows.
  - At 4352 B every VC0 row is on the 200G wire.

### Latency

- Only `exp_latency.yaml` exists: P150 2D, VC0, two ERISCs, 12 cases.
- **There is no same-job main-vs-v4 latency.** Main as shipped: `dqs_stock_kafw_lat` (a04u24, pre-merge, 2026-10-19).
  Latest v4: `sblat_v4rxv2i3_lat_1` (a09u32, job 373, 2026-11-01), a single run.
- A second pair set (`a8v1` = kernel v1 at the study flags 13129088, and `s1two_v4gi` = v4se) is kept only because the
  docs' latency claim was computed from it (conflict 1 below).
- RESULTS §66: the host-to-host offset is up to ~72 ns on 3-hop rows.

### CCL

- Op times per pass (us), DeepSeek decode CCLs on the P150, payloads 2048-15232 B.
- **Job per pass, from the broker's job list:**

  | pass | main_fw | v4rxo0b | v4rxf4 |
  | --- | --- | --- | --- |
  | a: all_gather | 426 | 431 | 435 |
  | a: all_reduce, broadcast | 430 | 431 | 435 |
  | a: SDPA | 440 | 440 | 440 |
  | b (one router per job) | 443 | 442 | 441 |
  | c / d / e (SDPA 2048 B) | 444 | 444 | 444 |

- So all_gather, all_reduce and broadcast are **cross-job** comparisons: adjacent jobs, alternating order, same day.
  Only SDPA has same-job pairs.
- reduce_to_one never ran.

### Size and stack

- Read from the `size_report.json` files that `size_report.py` left in the coverage builds:
  - main_O3, 32 production cases;
  - v4rxg3_O3 and v4rxg3_Os, 54 cases;
  - main_Os, bh_vc1 only.
- Main's other -Os figures come from the V4_SIZE.md baseline (2026-10-30).
- Main's source is `variants/mainv2cap` over the post-merge base, 6d8a103.
- **VC1 + VC2:**
  - the lead does not build it (static_assert at fabric_erisc_router_v2.cpp:113 on 131 configs);
  - main's coverage set never included it;
  - fanin.json / stability.json (job 320) record that main's router fails its -Werror stack limit (2,176 B) on VC1+VC2.

### Slots

- **Defaults per fabric.** The 2D allocation is the same on both routers (17,16,16). The others differ:

  | fabric | main | v4 |
  | --- | --- | --- |
  | 1D | 16,16,32 | 26,25,32 |
  | VC1 | 51,4,8,2,4 | 19,9,16,5,4 |
  | VC0 + VC2 | 43,7,7,7 rx 11 + 2/4 | 17,14,16 + 2/4 |

- **Main's code constraints** (git 6d8a103): at most 127 slots per channel; **no power-of-2 requirement**; mask fast paths
  for power-of-2 depths.
- **Same-job slot tables:**
  - main at five 1D splits next to v4u (jobs 076-077);
  - main at its VC1 table and at 25,8,16,4,4, next to v4gi at three allocations (job 112).
- v4rxv2i3 at main's VC1 table (SLOT_STUDY P1, jobs 376-377) isolates the slot row on v4 alone.

### Stability

- **Receiver headroom guard** (loopvars.py `--headroom`, offline): 7 families x {+1 value, +2 values, flush-defer record}
  for v4rxs2x, v4rxb2p, v4rxp2p, v4rxe1, v4rxf1, v4rxf2, v4rxf4, v4rxg3. Each record lists the hot values that moved.
- **Sender headroom guard:** v4txo0 and v4txo1, 10 families x 4 probes.
- **Measured fragility** (job 417, v4rxs2x against v4rxs2xh2, two extra loop values): 0 on single-channel rows, +4 to +7
  on two-channel rows. Not rerun on the lead.
- **Placement probe costs** (jobs 403-417); register maps and carried registers (sender lead, P-B, P-C, packed P-B);
  identity checks of the readability stages.
- **Opt level:**
  - job 367: v4 VC1 -O3 beats -Os by 10.7% on the split and 12.7% inter-mesh; the suite means were computed from the
    CSVs and match the note;
  - main's VC1 at -Os against -O2, jobs 089/090: ~1%.
- **Jitter band** (RESULTS §96-§97): v3 predecessors (v3tk, v3v6, v3x, v3z, s79mc) only. **No v4 variant has a jitter
  band.**
- **Main:**
  - no headroom or jitter data;
  - its VC1 router does not fit at -O3 (29,940 B);
  - it cannot build VC1+VC2 (stack 2,176 B, -Werror=stack-usage; V4_RECEIVER_SHAPE §25 / job 320);
  - it cannot build VC2 on one ERISC (RESULTS §98, job 141).

### Fan-in

- **P150 VC0+VC2 on one link** (cross-run, v4 only):
  - the lead v4rxv2i3 (jobs 333 / 340);
  - the sender placement probes v4rxf1 / v4txpc / v4txpb / v4txpbk (jobs 422 / 425 / 432, 445 / 446);
  - the idle-VC2 one-side A/Bs (jobs 322, 326).
- **Galaxy VC2 line mode** (vc2l_p3b = v4rxv2i3 + line mode, worktree tt-metal-vc2line; jobs 167441-167482); Galaxy
  slot-study VC2-on rows.
- **Mux mode, same job as main:** jobs 164-171, 22 rows; suite means main 373.3, v4trm 373.6.
- **1D line VCs:** host-side only (73/73 unit tests, 871/871 identical images). The router does not compile two or more
  line VCs yet.
- **Main has no VC2 traffic rate anywhere** (VC2 runs on main are pass/fail validation only).

## Conflicts between sources

1. **The latency claim against "main" uses another router.**
   - Claim: "1-hop within 30 ns, 3-hop -124 to -203 ns" (RESULTS §98 "Where v4 stands", V4_PERF_GAPS "Where v4 is
     ahead").
   - The baseline is `a8v1`: kernel v1 at study flags 13129088 and keepalive 1000 (dq66b_runs.sh line 21), on a08u24.
     It is not main as shipped, and it is cross-host (RESULTS §66: latency compares within one host only).
   - Against main as shipped (`dqs_stock_kafw`, a04u24) the same v4se run gives -155 to -190 ns at 3 hops and -38 to
     -85 ns at 1 hop, also cross-host.
   - V4_PERF_GAPS "Sources" lists `a8v1` as "main mix / latency".
2. **FLUSH_DEFER_DESIGN.md's "Main's router: flushed fused ~495 vs ~154 (meas, RESULTS §63)".**
   - Those are §62's columns: kernel v1 at the study flags with §62's staging.
   - RESULTS §58 says the flush was "Not measured on main".
   - No main-as-shipped or lead number with flush on exists.
3. **CCL_MATRIX.md's job list (430, 431, 435, 440-444) omits job 426**, which ran main's pass-a all_gather.
   - Its comparisons are cross-job except SDPA.
   - "v4 is faster than main on every CCL" holds for the pass means except all_reduce at 4096 B: v4rxf4 is 0.6% slower
     at 1 link and 0.2% slower at 2 links (CCL_MATRIX says "flat").
   - The memory note's "0.5-4.6% faster" leaves out SDPA 15232 B (-13.3%) and those 4096 B cells.
4. **Wire time at 4 KiB on 400G: 117 vs 118.5.**
   - The early target, 117 (RESULTS TL;DR, §13), counts one Ethernet frame.
   - A 4096 B payload with its 96 B header needs two frames: §29's model gives 118.5. The bare link measured 118.1 for
     4176 B (§16).
5. **The 200G wire at 4352 B: 252 vs 251.0 vs 250.8.** 252 is the measured line rate (252.1), 251.0 the slot study's
   fit, 250.8 the frame model. All three agree within 0.5%.
6. **Galaxy keepalive in job 163369.**
   - V4_PERF_GAPS says "firmware keepalive on both"; the frs_glx_torus.sbatch header says v4 uses the host keepalive.
   - Every wire-bound 4352 B row has v4trm +1.0 over main (252.1 vs 251.1). That is the keepalive-1000 stretch measured
     in 155240.
   - The Galaxy records set v4trm's keepalive to 1000 and flag this.
7. **Galaxy, job 156067.** RESULTS §98's table gives main at 2048 B = 265.7 and 3072 B = 261.2; the CSV gives 275.8 and
   267.4. The bundle uses the CSV.
8. **Full-ring multicast at 1-3 KB on the Galaxy.** V4_PERF_GAPS quotes 1.32-1.37x. Pass 2 at 1024 B is 1.30x.
9. **Galaxy 2D 1-hop at 2048 B with 2 KiB slots.** V4_NS_FLOOR §1 says E0 (A/B); SLOT_STUDY says the wire. The spin data
   shows E0 with ~5 cycles of slack, so both are near-binding.
10. **"main" in slot-study labels** (`vc1_*_main`, SLOT_STUDY P1's "main 51,4,8,2,4" column) is the lead at main's slot
    table, not main's router.
11. **RESULTS §44's "v4" latency row is the DRISC offload v4**, not the study router.
12. **V4_SIZE.md's "largest one-ERISC 16,860 B"** is a v4-only case (bh_vc1_trim_w0, one ERISC). Over the cases main
    also builds, the largest one-ERISC binary is glx_express_1erisc: v4rxg3 13,848 B, main 12,604 B.
13. **"VC1+VC2 26,156 B at -O3, 468 B left"** is a size probe of v4rxv2i3 plus v4rxc7v's gate line. It is not the lead,
    which refuses VC1+VC2. Production builds VC1+VC2 at -Os (get_kernel_opt_level), and runtime args share the
    26,624 B buffer, so the -O3 headroom is below 468 B.
14. **Keepalive in the P150 census.** Main's one-way 1D rows as shipped sit on the firmware-keepalive floor (187.6,
    RESULTS §15). Part of the 1D "router" gain is the host keepalive: main at keepalive 1000 runs 168.4 on 1D 1-hop
    writes.
15. **What binds VC2 across meshes.** VC2_LINE_BRINGUP phase 4 says "not established": doubling VC2's windows moves the
    flow 8.6%. Its later perf §3 says these flows are "bound by VC2's windows first". fanin.json records it as a
    hypothesis.
16. **"`tt_buf_rw::note<>` always_inline recovers about half the -Os gap"** (memory note, TODO). From job 367's CSVs it
    recovers 53% on the VC1 split (10.1 of 19.2 cycles/packet) but only ~27% on inter-mesh (13.2 of 48.9).
17. **V4_SENDER_OWNERSHIP §7.3** still says "no packed send progress or counters". §8.4's packed P-B won after it, and
    the memory note marks the earlier advice WRONG, but the doc line is not marked. §8.3 and §8.4's P-B columns also
    differ by up to 0.5 (e.g. VC2 at 1024 B: 282.1 against 282.6); fanin.json uses §8.4.
18. **The memory note feedback-no-compiler-fragile-tricks** says VC1 builds at -Os in production. Since job 399 the
    host builds v4's VC1 routers at -O3 (get_kernel_opt_level); only main's VC1 and v4's VC1+VC2 stay at -Os.

## Same-job against cross-run

| comparison | same job | where |
| --- | --- | --- |
| P150 census, main vs v4trm, 13 suites | yes | jobs 164-171 |
| P150 census, main vs v4rxf4 (the lead's code) | **no** (main from 164-171, v4rxf4 from 420-421) | bandwidth.json `same_job_with_main` false |
| Galaxy, main vs v4trm | yes | Slurm 163369 (and pre-merge 155833-158562 for older variants) |
| Galaxy, main vs v4rxv2i3 at 1536-4096 B | no | main 155281, v4 167648 |
| Latency | **no pair is same-job** | |
| CCL all_gather / all_reduce / broadcast | no (one router per job) | 426/430, 431, 435, 441-443 |
| CCL SDPA | yes | 440 (pass a), 444 (2048 B, passes c-e) |
| Slots: main at chosen depths with v4 | yes | 076-077 (1D), 112 (VC1) |
| Slack | no main data on the current tree | |
| Size | offline; builds of the same tree state (main from mainv2cap over 6d8a103) | |
| Mux rows (fanin.json) | yes | 164-171 |

## Gaps: what to rerun on the final lead, and why

Ordered by what the report needs most. Each one is also a `to_measure` record in gaps.json.

1. **Same-job P150 census, main as shipped against the final lead, all 13 suites, two passes in reversed router
   order.**
   - The last same-job main column is v4trm (10-29). Since then the placement, receiver redesign and codec stages have
     moved rows by -16 to +4. The sender pass is changing E0 now, and the host was rebuilt.
   - Run: `ROUTERS='main_fw <lead>' census_runs.sh two1|two2|one PASS` and `census_vc1_runs.sh PASS`.
   - Add main at keepalive 1000 (router `main`) to keep the keepalive separate.
2. **Same-job Galaxy run, main against the final lead.**
   - Use frs_glx_torus.sbatch with `CONFIGS="main <lead>"`, steps mesh / linear / ring / mcs twice (the 163369 recipe),
     on 110-a nodes only.
   - It needs a fresh /data base tree from the current study tree: the 10-29 base lacks the lead's host args, and main
     must run on the same base.
   - The 4352 B full-ring multicast gap (0.76x on v4trm) has not been measured on any later lead.
   - Galaxy agent's finding: the sbatch files don't enforce the 110-a rule (sls_glx.sbatch / sls_glx2k.sbatch list
     bh_sc36_2), and job 167648 ran on bh-glx-120-c05u02. Fix this before submitting.
3. **Same-job latency** (`exp_latency.yaml`, 12 cases).
   - Add a latency group to census_runs.sh (`suite lat $(suite_yaml lat) ... "main_fw <lead> <lead> main_fw"`).
   - No main-as-shipped latency exists on a09u32 or after the merge.
4. **Max-feature config: inter-mesh + express + bubble flow control + VC1 + VC2, both routers.**
   - No capture combines them (`~/.tt-metal-v3r-builds/cov_m/cases/`):
     - glx_express (Galaxy 8x4 torus XY + express: bubble flow control and express) has no VC1 / VC2;
     - bh_vc1_vc2 (P150 two 2x2 meshes: inter-mesh VC1 + VC2) has no torus or express.
   - Neither router builds VC1 + VC2 today:
     - the lead refuses it (static_assert);
     - main fails its stack limit (job 320, 2,176 B, -Werror=stack-usage) and doesn't build VC2 on one ERISC.
   - Needed:
     - a mock capture of a multi-mesh torus with express and VC2, if the host's builder accepts the combination (that
       is the open question);
     - the lead brought up on VC1+VC2;
     - coverage builds of both routers at -O3 and -Os;
     - size_report.py.
   - This is the config that decides whether "-O3 with headroom" holds.
5. **CCL matrix with main and the lead in one job per pass.**
   - Follow job 440's form for every CCL.
   - Add reduce_to_one after the trace-region choice.
   - SDPA needs the local test edit.
6. **Flushed atomics** (`streams/flush_mix.yaml`, p3_fused with flush on): main as shipped and the lead in one job.
   - Nobody has measured either router as shipped with flush on.
   - The FLUSH_DEFER design is parked on this cost.
7. **VC2-on rows, main against the lead, same job.**
   - The census has no VC2-on suite.
   - Main runs VC2 at 43,7,7,7 rx 11 + 2/4. The lead runs 17,14,16 + 2/4, and its VC2 alone is window-bound (2 worker
     slots: ~860-870 cycles/packet).
8. **One-side delay A/Bs on the final lead** (variants/v4bdd rebased): every bound side in slack.json is from v4trm or
   v4rxv2i3.
9. **Not measured on either router:**
   - trimmed one-ERISC rows and mix_trim;
   - express benchmark rows on the Galaxy;
   - main at 3072 B slots on the Galaxy (the size SLOT_DECISIONS recommends for 200G);
   - VC1 split and inter-mesh rows against main on the Galaxy;
   - full-ring multicast at 3072 B slots (would show whether 3 KiB slots close the 4352 B gap).
10. **Size and stack of the final lead** after the sender pass: `coverage.py build` + `size_report.py --opt O3|Os main
    <lead>`.
11. **Native fan-in headline row:** N workers on N line VCs against N workers through main's mux on VC0.
    - Not measurable yet: the router does not compile two or more line VCs (TODO "Next" item 6).
    - Main through the test-infra mux with N workers was never run.
    - VC2 line mode on the lead (vc2l_g3a) is queued on the Galaxy (jobs 168748 / 168774).
12. **Codegen fragility on the final lead.**
    - Rerun the two-extra-values device probe (job 417 form) on the lead's receiver.
    - Add a sender probe.
    - No v4 variant has a jitter band, and there is no -O3 / -Os census of the lead.

## One suggested chart per dataset

| file | chart |
| --- | --- |
| bandwidth.json | Dot plot of the speedup main / v4 (cycles per packet) per census suite mean. Two dots per suite: v4trm (same job, filled), v4rxf4 (cross-run, hollow). 1.0 reference line, suites sorted by speedup. Drill-down per row. Galaxy as its own panel from Slurm 163369. |
| wire.json | Cycles per packet against payload size, one panel per link speed: the wire model line, main's and v4's measured rows. At 200G, the Galaxy line-rate sweep at 4352 B slots meets the wire at ~2.3 KiB. |
| slack.json | Per row, a horizontal bar pair of E0 and E1 slack (cycles per packet, v4rxv2i3), coloured by the A/B bound side, grouped by suite. Rows bound by the wire or a window stand out as two long bars. |
| latency.json | v4 minus main (ns) per case, faceted by hop count, with a shaded ±72 ns band for the host-to-host offset and a "cross-host, not same job" label. |
| ccl.json | Ratio v4rxf4 / main per CCL and payload (pass means), with markers for the same-job SDPA passes; 1.0 reference line. |
| size.json | Grouped bars per coverage case: main against v4rxg3 code (E0+E1, or one-ERISC binary) at the production opt level, with the 26,624 B buffer as a line; a second small panel for least free E0 stack. |
| slots.json | Small multiples: rate against slot allocation for main and v4 in the same job (job 112 VC1 rows; jobs 076-077 1D rows). Shows that main's VC1 table costs main too, and that v4's lead holds at equal slots. |
| stability.json | Heatmap of headroom cells: family x probe (+1, +2, flush-defer record), pass / fail, one column per variant (v4rxf1 ... v4rxg3, sender probes). |
| fanin.json | Bar chart of per-flow cycles per packet for VC0 alone, VC2 alone and VC0+VC2 contended, across the placement probes (P-B, P-C, packed P-B), plus the mux rows main against v4trm. |
| gaps.json | Table rather than chart: rows where v4 trails main (same-job and cross-run marked), then the to-measure list. |
| galaxy_v4_only.json | Payload GB/s against payload size at 2048 / 3072 / 3584 / 4352 B slots per path (VC0 1-hop, N/S, VC1, inter-mesh), with the 200G wire line. |
