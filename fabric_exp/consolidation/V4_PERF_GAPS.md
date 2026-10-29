# v4 performance gaps (parked 2026-10-29 morning; resumed the same day with a same-job census)

User (2026-10-29): close out the remaining router features first, chase performance after; keep a note of the gaps.
Features closed (UDM, VC2, speedy path, telemetry, express / Z, trimming); the lead is **v4trm** (`variants/v4trm`), whose
images equal v4rs's (the lead at parking) on every production config but the 22 VC1 senders (a stack-slot
permutation; their VC1 rows equal v4gi's). All numbers in cycles/packet (1.35 GHz), lower is better.

## Census, 2026-10-29 (same jobs; supersedes the cross-run tables in the gap sections below)

a09u32 jobs 164-171 (`census_runs.sh`, `census_vc1_runs.sh`, report `census_report.py`), two passes per suite, every
router back to back in the same job; no v4trm row moved more than 3% between passes. Routers: main as shipped (kernel
v1, flags 0, firmware keepalive); main at the study keepalive (1000); **v3qm** (`variants/v3qm` = v3q with the
post-merge packet-header path, nothing else; builds all 336 post-merge production configs; mesh sanity 96/96 and ring
validation 19/19 here); v4trm at the study settings. Galaxy: Slurm job 163369 (bh-glx-120-c01u14), main and v4trm,
two passes, firmware keepalive on both.

| suite (rows) | main | main ka1000 | v3qm | v4trm | v4 / main | v4 / v3qm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2D 4352 B (10) | 217.2 | 217.2 | 144.5 | 144.4 | 0.665 | 1.000 |
| 1D incl. ring row and NeighborExchange (16) | 200.5 | 196.8 | 162.0 | 152.2 | 0.759 | 0.940 |
| mix (30, MixCmd left out) | 222.2 | 219.9 | 127.0 | 128.7 | 0.579 | 1.014 |
| bursts (44) | 555.0 | 555.0 | 522.5 | 523.3 | 0.943 | 1.001 |
| ring (14) | 239.9 | 238.6 | 227.6 | 188.6 | 0.786 | 0.829 |
| Y-torus (12) | 233.7 | 233.8 | 164.7 | 157.4 | 0.674 | 0.956 |
| one ERISC: 2D (10) | 331.3 | 331.3 | 245.5 | 276.6 | 0.835 | **1.127** |
| one ERISC: 1D (16) | 318.3 | 318.5 | 284.5 | 305.7 | 0.960 | **1.075** |
| one ERISC: ring (16) | 375.7 | 374.2 | 360.6 | 375.1 | 0.998 | 1.040 |
| mux mode (22) | 373.3 | 373.3 | 368.7 | 373.6 | 1.001 | 1.013 |
| trimmed 2D (20) | 215.9 | 211.9 | - | 154.4 | 0.715 | - |
| VC1 split (13) | 332.6 | - | 155.2 | 171.5 | 0.516 | **1.105** |
| VC1 inter-mesh (8) | 841.0 | - | 336.4 | 342.5 | 0.407 | 1.018 |

Galaxy (200G links; v4trm / main as rate): mesh and line rows below 4352 B 1.18-1.85x; ring 1-hop 1.27-1.29x; full-ring
multicast 1024-3072 B 1.32-1.37x, 3584 B 1.07x, **4352 B 0.76x** (main 263.7, v4trm 346-349, both passes); 4352 B
unicast rows are wire-bound on both (~250, equal).

Rows where v4trm is slower than main as shipped (13, all one ERISC): NeighborExchange write 251.3 vs 234.1 (+7.4%) and
scatter +1.7%; half-ring multicast +4% (1D and ring suites); ring 1-hop bidirectional +4%; full-ring multicast 1 KB
+2.5%; mux full-ring multicast +4%, mux ring multicast +1.7%.

Gaps by weight now (each needs its isolating A/B before any attribution):
1. **One ERISC against v3q** (new): 2D +12.7% (1-hop write 207.0 vs 179.2, multicast 326.8 vs 275.5, 3-hop 262.4 vs
   222.1), 1D +7.5%, ring +4%; and the 13 rows above where one-ERISC v4 trails main. One ERISC is Wormhole's only mode.
   Gap 2 below is the ring and mux part of this.
2. **VC1 split against v3q** (new; the old standings had v3q at 180-198, from before the VC1 slot defaults changed):
   1-hop 158 vs 137 at both 16 B and 4352 B (+15%, size-independent), 2-hop +6%; v4trm equals v4gi (158) here, so not
   a regression from the features.
3. **Galaxy full-ring multicast at 4352 B** (gap 1 below): 0.76x main on the lead, as v4uf predicted.
4. **Two-ERISC rows behind v3q** (gap 4 below): 3-hop write 151.7 vs 142.9, multicast write 145.9 vs 141.2, pure
   atomics N/S +5%, mix +1.4% overall; elsewhere v4 is ahead of v3q (1D -6%, ring -17%, Y-torus -4%).
5. Gap 3 (two-ERISC ring receiver headroom) is unchanged: v4 leads both other routers there.

Rules for working these (memory `feedback-attribution-needs-ab`, `tt-metal/.claude/CLAUDE.md`): which ERISC bounds a row
and what caused a gain are decided by an isolating A/B, never by PC-sample shares or line-cost estimates; until then
they are hypotheses (marked below), and no variant is built on them before the A/B.

## Gap 1. Galaxy saturated full-ring multicast at 4352 B: below main

The one production-topology row where v4 is slower than main.

| full-ring multicast (200G Galaxy) | main | v4u | v4wt | v4uf | v4eg |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024-3072 B | 261-282 | 216-217 | 210-212 | 207-209 | 205-207 |
| 3584 B | 261.1 | 218.6 | 216.0 | 237.5-238.2 | 235.0 |
| 4352 B | 263.4-263.8 | 302.7-305.2 (0.87x) | 309.7-310.3 (0.85x) | 345.5-351.9 (0.75x) | 356.0-356.9 (0.74x) |

Slurm jobs 156067, 158477, 158562 (RESULTS §98 "Galaxy check of the lead v4uf", "Single-slot completion timing on
Galaxy"). 4352 B is the one size whose wire time (~248 at 200G) exceeds v4's router time; the P150's rings never
saturate, so this row exists only on Galaxy.
- Established (A/B): a bounded completion wait on passes without a packet is what the saturated ring needs (v4wt vs
  v4uf, 350 -> 310); crediting the newest slot early does not help (v4eg); holding any completion collapses the ring
  (v4p / v4p2, 17,000-20,000 cycles/packet: bubble flow control, "a completion must not wait").
- Not measured: the lead. v4gi, and so v4tm / v4rs, check a single slot once, as v4uf did; expected near v4uf's 0.75x
  (hypothesis until run).
- Hypotheses (untested): how the sender shares a saturated link between injection and forwarded traffic; the
  single-slot completion timing (worth ~40 of the gap per the v4wt A/B, not all of it). The frame count per packet is
  a weak suspect: main sends up to two frames per packet too and keeps the link ~94% busy (argument, not an A/B).
- First steps when resumed: the lead and main on Galaxy (`glx_ring_mcast_sweep.yaml`, galaxy-slurm-harness memory);
  then the lead with v4wt's bounded wait on idle passes, as an A/B.

## Gap 2. One-ERISC rings with multi-flow traffic, and mux-mode rings: below main

Mux mode forces one ERISC, so its ring rows belong here. User: one ERISC must be functional on Blackhole and tuned on
Wormhole; this gap is a Wormhole-tuning item unless mux-mode rings matter first.

| one ERISC (jobs 127-128, 138) | main | v4tm | v4rs |
| --- | ---: | ---: | ---: |
| ring 1-hop / 3-hop (4352 B) | 259.8 / 316.6 | 255.9 / 314.9 | 255.9 / 314.9 |
| ring 1-hop bidirectional | 364.5 | 383.6 | 379.5 |
| half-ring multicast (1 KB / 4352 B) | 446-457 / 433-451 | 467.0 / 467.2 | 461.8 / 461.9 |
| full-ring multicast (1 KB / 4352 B) | 471-492 / 487-488 | 500.6 / 500.3 | 493.5 / 493.1 |
| mux full ring (1 KB / 4352 B) | 419.6 / 419.8 | 440.9 / 441.5 | 436.0 / 436.6 |
| mux ring multicast (1 KB / 4352 B) | 326.1 / 345.3 | 332.5 / 353.3 | 331.2 / 351.5 |

Main's half / full ring rows move up to ~20 between passes (rsea vs rseb); compare within a job.
- Established: credit bookkeeping costs both routers ~89 per packet by source line (job 131); stream-register credits
  are slower than v4's counter frames everywhere (v4sc, jobs 129-130); ring-position credits' receiver half helps where
  the receiver binds (v4rr: mux full ring -38, one-ERISC 1-hop -26, bidirectional -34), and their sender attribution
  loses on mixed traffic (v4rr one-ERISC full ring +77: a run per packet, out-of-line calls).
- Hypothesis (profile only, job 131): the rest is v4's single-ERISC pass structure (loop ~15 against main's ~4; two
  out-of-line passes over state in memory).
- Candidate: ring-position credits with a sender whose per-packet attribution costs nothing extra on mixed traffic (the
  register bit-history idea; user's option 2, not chosen yet).

## Gap 3. Two-ERISC ring and torus headroom (not a regression)

v4 already leads main by ~51 and v3q by ~42 on the ring suite mean. Established by A/B (job 138, v4rs / v4rq / v4rr):
light two-ERISC ring rows are receiver-bound, and the receiver's acks by count are worth -10 to -38 there (v4rr: ring
1-hop write 150.7 -> 140.7, scatter 189.5 -> 151.7; Y-torus N/S scatter 200.8 -> 167.2). v4rr lost on all-to-all ring
multicast (+38 / +26) and half-ring write (+24). Same candidate as gap 2.

## Gap 4. Two-ERISC rows behind v3q

| row | v3q | v4 | |
| --- | ---: | ---: | ---: |
| 2D 3-hop write; Y-torus 3-hop E/W | 142.9; 142.9 | 151.8; 151.7 | +9 |
| 2D multicast write; Y-torus multicast | 141.1; 141.3 | 145.9; 146.0 | +5 |
| worker writes 16-1024 B (mix, E/W / N/S) | 116.3-118.7 | 119.3-120.4 | +2 to +3 |
| forwarded writes 16-1024 B, MixSize (FWD) | 112.7-112.9 | 118.1-118.7 | +5 to +6 |
| pure atomics (E/W, N/S, FWD) | 126.9-127.9 | 130.3-133.9 | +2 to +6 |
| bursts of 16 B (N1 G0, N8 G500, N32) | 116.5-170.3 | 119.4-173.9 | +1 to +6 |
| MixCmd (E/W, N/S, FWD) | 160.9 / 174.6 / 154.0 | 197.2 / 198.1 / 168.7 | +15 to +36 |

v3q's mix and ring rows: a09u32 (`v3q_13129088_*`); v3q's bursts: v3zy on a08u24 (`p34_v3zy`); v4: job 121
(`s1two_v4gi_*`).
- MixCmd: strict atomics wait for every earlier write's acknowledgement; v3q defers flushed atomics. Deferred atomics
  are off in v4 by decision (RESULTS §98 v4j), so this stays unless that decision changes.
- Pure atomics: v4gi's group status read for groups no command read (A/B against v4vc: +2.0 / +3.3).
- 16 B bursts: tied to the worker's 17-slot credit window (A/B: with 32 worker slots v4gi equals v4vc on every burst
  row; RESULTS §98 "v4gi's burst losses diagnosed").
- 3-hop, multicast, small writes: profile-based explanations only (the middle sender's send +4-6, the per-pass
  free-slots mask; RESULTS §98 v4i, v4j); hypotheses.
- User guidance: don't chase 1-2 cycle deltas; robust over peak (memory `feedback-robust-over-peak`).

## Not yet measured on v4 (added 2026-10-29)

- Trimmed rows: measured in the census (2D 4352 B, two ERISCs: v4trm 154.4 vs main 215.9). Not measured: trimmed
  one-ERISC rows (a router left with one role runs the single-ERISC loop with an empty out-of-line pass for the other)
  and `mix_trim`.
- Express / Z rows on Galaxy: validated only (V4_EXPRESS.md); no benchmark rows.

## Where v4 is ahead (for context)

Two ERISCs against main: 2D -73, 1D -51, ring -51, Y-torus -25, VC1 split 1-hop ~-160, 3-hop latency -124 to -203 ns.
One ERISC against main: 2D -55, 1D -22, VC1 ~-144. Galaxy, router-bound sizes (1-3.5 KiB full ring): 1.2-1.3x main.

## Sources (`results/`, labels `LABEL_13129088_<suite>.csv` unless noted)

| column | label | notes |
| --- | --- | --- |
| main 2D / 1D | `mpost_v1` | as shipped (firmware keepalive), after the merge |
| main ring | `main1d_tier_ring` | firmware keepalive, tier slots; `v1_13129088_ring` = keepalive 1000 |
| main Y-torus | `v1_13129088_torusy` | keepalive 1000 (no firmware-keepalive run) |
| main mix / latency | `a8v1` | a08u24 |
| main one ERISC | `se_fab1_v1` (2D / 1D), `rsea_v1` / `rseb_v1` (rings), `mxa_v1` / `mxb_v1` (mux), `mpostvc1_v1_*` (VC1) | |
| v3q | `v3q2d` (2D), `v3qw24` / `v3qw16` (1D), `v3q` (mix, ring, Y-torus), `p34_v3zy` (bursts, a08u24) | keepalive 1000 |
| v4 two ERISCs | `s1two_v4gi` (job 121) | = v4tm / v4rs on these configs within 0.4 |
| v4 rings, torus, one ERISC, mux | `rsa_tm`, `rsa_rs`, `rsa_rr` (job 138) | `rsb_*` the second pass |
| v4 one ERISC 2D / 1D | `s1fab1_v4se` | |
| v4 VC1 | `mpostvc1_v4gi_*` | |
