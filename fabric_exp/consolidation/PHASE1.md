# Phase 1: v3 designs

v3 is a new kernel file (`fabric_erisc_router_v3.cpp`). During Phase 1 every candidate is built and run in v2's
place (selector `TT_FABRIC_KERNEL_VERSION=2`, same host args); the `=3` selector comes with the winner.

## Decisions (user, 2026-10-24)
- New v3 file.
- No branch-layout hints (`[[likely]]`/`[[unlikely]]`) in v3. They may be revisited once a v3 shape exists, and kept
  only if they win at -O3, -O2 and -Os and under the probes.
- -Os with VC1 exists because everything is inlined and VC1 adds 4–5 sender channels: code size grows with channel
  count. A v3 design should not make code size scale with the number of channels.
- Forwarding channels with no upstream router (every 2D E0 on the 2x4 mesh polls 1–2) are handled by a separate
  team track. v3 designs must not rely on trimming them: generic perf targets assume every serviced channel may be
  live.
- ERISC-only; DRISC offload out of scope.

## Candidates (RESULTS §83–§89)
| Name | ERISC | What | Status |
| --- | --- | --- | --- |
| v3a | both | Faithful consolidation of s79mc (no flags, DRISC, profiler hooks or study hints) | Reference. Same code as s79mc-without-hints; VC1 does not fit at -O3 |
| v3u | E0 | One runtime table for every sender channel | §85: ~30 slower on VC0 worker rows; flat across levels; hook +18 |
| v3u2 | E0 | Specialized worker + runtime forwarding table with a per-VC working copy | §89: best -O3 worker rows (116.8) but not at -O2 (121.2); multicast +29; VC1 fits -O3. v3u3 in progress |
| v3u3 | E0 | v3u2 + scalar group-read values + forwarding state in place (no working copy) | Blocked. §90: worker 116.1 / 119.0 at -O3/-O2, multicast +21, forwarded +9; hung on VC1 at -Os with v3a's E1 (admission bug + single-bit idle test) |
| v3l | E0 | Specialized worker + one combined forwarding test + forwarding via a noinline body | §86: worker 117.2; forwarding +19 to +41 |
| v3l2 | E0 | v3l's worker + inline compile-time forwarding visits; E0 init sender-side only | §88: ≈ v3a on VC0 (means +0.2 to +1.4; multicast +5); VC1 −30 at -O2/-O3; fits -O3 |
| v3e1 | E1 | Own E1 program: cached downstream credits, E1-owned forwarding, exact completion check, VC1 admission fix | §87: forwarding/multicast −4 to −27; local delivery +2 to +12; VC1 4352 slower. v3e1b in progress (also -Os) |

| v3e1b | E1 | v3e1 + cheaper exact completion check, uniform-batch credit, direct local write, inline VC1 channel 0 | §91: -Os penalty halved (means +8 to +17 vs v3a's +19 to +32); ≈ v3a or better at -O3/-O2; VC1 better than v3a; scatter +3 |
| v3l2e1b | both | Exact composition of v3l2 (E0) and v3e1b (E1) | Blocked (superseded by v3l2be1b). §92: VC0 -Os penalty +4.5 to +8.8 (v3a +19 to +32); E/W 16 B 120.8 / 119.3 / 122.6; HUNG on VC1 at -Os like v3u3 → root cause in progress |

| v3l2b | E0 | v3l2 with v3a's exact idle predicates everywhere (no single-bit test on 2-slot VC1 channels) | Offline; VC0 2D/1D E0 byte-identical to v3l2 |
| v3l2be1b | both | Exact composition of v3l2b (E0) and v3e1b (E1) | §93: passes everything incl. VC1 at all levels and 10/10 VC1 repeats; ≈ v3a at -O3/-O2; -Os penalty −70%; VC1 −25 to −40; gaps: forwarded +7.8, multicast +3.5, scatter +2.7. Superseded by v3l3be1c |
| v3l3 | E0 | v3l2b + no calls on the pass path, yield-only context-switch flag, no VC0 combined test | Unblocked (§98: the hang was the test race). §94: forwarded 16 B 112.3, worker 115.6; HUNG on VC1 at -O3 (third VC1 hang) |
| v3l3b | E0 | v3l3 with VC0's combined test kept | §94: ≈ v3l3; passes everything incl. VC1 at all levels (once) |
| v3e1c | E1 | v3e1b + direct scatter/atomic/fused local delivery (scatter counters once per packet) | §94: scatter −7 to −12, atomic −4 to −11 vs v3a; forwarded fused still +5.6 |
| v3l3be1c | both | Exact composition of v3l3b (E0) and v3e1c (E1); cleaned form `v3sb` (on v3r, PT_LOAD-identical) | §94: lead until v3t (§95). Means vs v3a -O3: -O3 −4.2/−3.6/−4.2/−0.1/−1.5, -O2 −2.3/−2.1/−4.1/−0.6/−1.6, -Os −0.3/+4.7/+0.9/+3.9/+1.9; hook no cost; VC1 passes all levels (once). Open: forwarded fused, E1 opt-level sensitivity, VC1 hang cause |
| v3r / v3r2 | both | Mainline cleanup of v3l2be1b (PT_LOAD-identical) / plus six null cleanups | §94: v3r2 costs +1 to +2 on means and +4 to +6 on E1 scatter at -O3/-O2, 0 at -Os: E1's inlined main loop is layout-sensitive |
| v3t | E1 | v3sb + E1 main loop in its own `noinline` function with no non-power-of-two constant multiplies; full-batch completion fast path; 1D decides once; VC2 guard | §95: lead until v3v4 (§98). Means vs v3a -O3: -O3 −8.4/−4.8/−3.8/−0.5/−6.8, -O2 −6.6/−4.5/−3.9/−1.0/−6.3, -Os −5.4/−2.1/−1.3/+0.5/−1.4. Forwarded rows level-independent; VC1 passes all levels. Open: N/S rows and half-ring multicast at -Os, forwarded fused +4.6, hook +2.5 |
| v3t2 / v3tb | E1 | v3t + v3r2's six null cleanups / v3t with the loop inlined | §95: v3t2 = v3t within 0.4 on every mean and scatter row at all levels (the stability test passes); v3tb ≈ v3t on 2D, 1D +2.4, hook +3 |
| **v3z** | both | Exact composition of v3yb (E0) and v3x3 (E1) | **§98: lead (adopted 2026-10-26).** Each ERISC PT_LOAD-identical to its parent; 133 logs clean, VC1 34/34 at all levels. VC1 1-hop −20 to −32 vs v3v6; bursts at v3tk's level or better; jitter band −0.18..+0.06 (2D), edits ≈ run-to-run noise. Means vs v3tk (4352/mix/flushmix/1D): -O3 −2.0/−1.3/−1.1/−3.0, -O2 −2.6/−0.9/−1.0/−3.2, -Os −2.6/−2.2/−1.1/−1.7; vs v3v6 +0.4 to +2.1. Open: VC1 multicast vs v3x (one run each), -Os E0 cost (~1), half-ring -Os |
| v3x3 | E1 | v3x without the record-word batch cut (= v3x_nocut, mainline comments; PT_LOAD-identical) | §98: every bursty loss back to v3tk's; steady means +1.0 / +0.4 / +0.5 / +0.8 vs v3x at -O3 (the cut's worth); EW_MixSize4 −3.8; NS_Chunk4Fused_Flush +10 (v3tk's level) |
| v3x2 | E1 | v3x with the cut only on passes that handle a packet | §98: recovers about half of each bursty loss. Out |
| v3x | E1 | v3v6 + credit holds bounded by the ring (T = 1 below 8 slots, else min(4, slots / 4); partial batch after 4 idle passes) | §98: VC1 −12 to −83 vs v3v6; 2D +0.2 to +0.5; jitter band ±0.03. Bursty losses (16 B N8 G500 +4..5) were the cut's → v3x3 |
| v3yb | E0 | v3y (helpers, per-round work out of line) + the worker visit takes completions before its send decision | §98: passes everything; VC1 1-hop −4 to −13 vs v3v6; -O3/-O2 means ±0.4; -Os +0.2 to +1.2. Chosen E0 for v3z |
| v3yc | E0 | helpers + the worker change only | §98: VC1 as v3yb; -O3 1D +0.9 (half-ring); -Os +0.2 to +1.0 |
| v3v6l3 | E0 | v3v6 with round 11's v3l3 E0 change (no VC0 combined forwarding test) | §98: VC1 worse on every row (up to +14). Out |
| v3w / v3w4 | E1 | Continuous completion (rate K) / the same rule in batch code with `uncompleted > T` | §98: out (E1-bound rows +7 to +16 / VC1 +30 to +45) |
| v3v6 | E1 | v3v4 + wait for a full batch's newest write to leave the NIU before crediting (NoC API, capped at 32 reads) | §98: lead until v3z. Means vs v3t (4352/mix/flushmix/1D): -O3 −3.4/−1.7/−1.6/−4.1, -O2 −4.1/−1.9/−2.3/−3.6, -Os −4.8/−3.7/−2.6/−2.5. One-hop 4352 B writes now better than v3t (126.8 vs 128.1). Left: MixSize4 +3 (batch cut), Linear3HopEW -Os +3.5 (1D cut), half-ring -Os |
| v3v4 | E1 | v3tk + local-write preset via the NoC API (`local_write_init_state` / `local_write_with_state`), fixed 2D route decode, completion batches cut at record-word boundaries, 2D partial batch held 8 idle passes | §98: lead until v3v6. Means vs v3t (4352/mix/flushmix/1D): -O3 −2.1/−1.4/−0.2/−3.7, -O2 −2.6/−1.4/−1.0/−3.8, -Os −3.4/−3.3/−1.4/−1.9; N/S -Os gap mostly closed; FWD fused 154.7 (beats v3a -O3). Known costs: one-hop 4352 B writes +2.4..3 (cause open), MixSize4 +3 (batch cut); half-ring -Os open |
| v3tk | both | v3t reading the keepalive host setting `ETH_TXQ_SEQ_UPDATE_TIMEOUT` (builder: 1000 on Blackhole, 0 = firmware elsewhere) | §95: PT_LOAD-identical to v3t; on hardware without the study override it reproduces v3t at all three levels (means within 0.3, VC1 34/34) |

Key findings so far: the whole -Os penalty is E1's (§88); v2/v3a's VC1 receiver admits on VC0's streams (§87).

E0 and E1 are separate programs (`MY_ERISC_ID`); the best E0 and the best E1 combine. Shared design rules:
`DESIGN_BRIEF.md`.

## How a candidate is judged
Offline first (no device): builds all 440 logged configs at -O3, -O2, -Os; `.text` and frames; probes apply.
Then on hardware (`prep_round.sh`, `p1rN_runs.sh`, `compare.py`):
1. Correct: validation (stream, mesh, 1D sanity) with RELAXED and with STRICT flushes.
2. Performance at -O3: the benchmark suites against s79m (and v3a).
3. Stability: the same at -O2 and -Os, and at -O3 with the `hook` probe. The yardsticks from s79m are -O2 +2.5 to
   +3.9, -Os +19 to +28 (RESULTS §82), hook up to +4.4 (§83). A design wins on stability if those shrink.
4. Code size, and (once VC1 configs are captured) whether VC1 fits at -O2/-O3.
Suite means decide, not single rows; a design within about a cycle of the best and clearly more stable wins.

## Next (user, 2026-10-25): v3l2be1b is the v3 base; two tracks in parallel
1. **Close the gaps** (-O3, vs v3a; attribution from v3l2 = lean E0 + v3a E1 and v3e1b = v3a E0 + v3e1b E1):
   - E0 (`v3l3`, V3L_NOTES §11; E1 must stay v3e1b's): forwarded small writes +7.5, 3-hop +3.5 and multicast +6 vs
     what v3e1b's E1 allows (143.4 / 137.5).
   - E1 (`v3e1c`, V3E1_NOTES §10; E0 must stay v3l2b's): scatter +2 to +7 (worst 1D and forwarded scatter/fused),
     atomics +3 to +4, chunk-4/8 fused flushed +1 to +2.
   - The two are merged three-way over v3l2be1b, then run through the full protocol.
2. **Mainline review** (`V3_REVIEW.md`): findings, plus `variants/v3r` = v3l2be1b cleaned with PT_LOAD-identical
   ELFs at all three levels. Codegen-changing cleanups go in `v3r2`, measured on hardware. Renames are applied by
   `v3r_renames.py`, so the track-1 diffs port mechanically.

### Round 11 status (2026-10-25, jobs 252-257, 263-266)
- v3l3 (E0 forwarding fix; E1 v3e1b): -O3 FWD 16 B 112.3, worker E/W 115.6, validation passes, -O2/-Os pass.
  **HUNG on VC1 at -O3** (job 254: rank 1 never finished MultiMeshAllToAll_size_4096 unicast_scatter_write; -Os, -O2
  passed). Third VC1 hang, all at MultiMeshAllToAll 4 KiB. Blocked pending cause.
- v3l3b (v3l3 + VC0 combined test kept): passes everything incl. VC1 at -Os/-O2/-O3 (once each); ≈ v3l3 at -O3.
- v3e1c, v3e1d (E1 scatter/atomic/fused): round a queued (256, 257). Merges v3l3e1c, v3l3be1c are exact compositions
  (all levels); v3l3be1c full protocol queued (263, 264). v3l3e1c not run (v3l3 blocked).
- Review: V3_REVIEW.md; v3r PT_LOAD-identical to v3l2be1b; v3r2 (6 null cleanups, all ELFs change) queued (265, 266).
  Latent bug confirmed: generic loop's deferral bypass lacks `!intermesh_egress` (E1 copy has it); fixed in v3r.

### Round 12 status (2026-10-25, jobs 277-284)
- v3t (E1 insulation + level-independent paths) is the lead; v3tk is the same code reading the keepalive host setting.
- v3t2 (six null cleanups) did not move it on hardware: goal 1's failure mode from §94 is fixed for E1.
- Keepalive (user: include it in v3): host setting in the tree's builder (`set_eth_txq_seq_update_timeout`, 1000 in
  the Blackhole block of `compute_mesh_router_builder.cpp`); `TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT` remains as a study
  override only. Job scripts: `KA_ENV=` drops the override.
- Trimming replay hangs with the tree's v1 too (job 282): not the v3 router's.
- Jitter band (§96, jobs 285-286; user: no team rules or gates): 8 ordinary edits move v3tk's suite means by ±0.4
  at -O3 and -Os (noise); a few rows up to 6 (alignment). -Os is not more stable. The multiply rule is dropped: the
  loop's own function is the insulation; synth_mult is a documented gotcha.
- 2x2 (§97, offline): constant multiplies have no effect; per-round work (termination, heartbeat, context switch)
  out of the loop function is what insulates E1's loop. The multiply rule is withdrawn.

### Round 13 plan (2026-10-25): close v3t's remaining gaps, all E1 (E0 = v3sb's, -Os-robust per §94's injection)
- N/S rows at -Os (+6 to +12 vs -O2; E/W rows are level-independent; N/S paths +6 to +10 instructions at -Os).
- Half-ring multicast at -Os (+14) is in every E1 design, v3a's too: shared 1D multicast code.
- Forwarded fused: v3t 159.8 at every level. v3a's 155.2 exists only at -O3 (-O2 159.5, -Os 189.9), so the target is
  getting by structure what -O3 does for v3a.
- Minor: E/W atomic +2 vs v3sb at -O3/-O2.
- E1 design agent working offline on `variants/v3v` (V3E1_NOTES §12); then a hardware round with VC1 and the jitter
  edits.

### VC1 hang cause (2026-10-26, RESULTS §98)
- The VC1 hangs since §94 are a test_tt_fabric race, not router faults: receivers are not synced before senders start,
  and a late receiver's buffer is overrun (4 KiB all-to-all, 100 packets into 64 slots).
- Confirmed by the worker capture (job 319). User decision: v3l3, v3vc and v3w are unblocked.
- Test fix agreed with the user: a receiver readiness sync for all tests; flow control on by default for validation
  mode is under evaluation; benchmark mode keeps no flow control.
- Fix in the experiment tree: the receiver start gate v2 (`streams_rxready.patch`; each receiver counts itself ready on
  its chip's sync core, the host reads one word per chip). Job 337: a forced 200 ms launch skew, 24/24 on both ranks;
  job 338: 25 plain repeats, all clean, no gate warning. Flow control by default is still the user's decision.

### Lead (2026-10-26): v3z
- v3z = v3yb's E0 + v3x3's E1, adopted by the user as the robust-over-peak trade (RESULTS §98): bursts at v3tk's level,
  the best VC1 1-hop rows, codegen stable under ordinary edits on both ERISCs; 0.4 to 2.1 behind v3v6 on steady means.
- LEDGER.md is classified against v3z.
- Open: VC1 multicast vs v3x (one run each), the ~1 cycle the record-word cut saved on steady streams (a cheaper
  four-record read that keeps the batch shape), v3yc's E0 at -Os (~1), half-ring multicast at -Os.

## Known defects in the tree's v2 (`fabric_erisc_router_v2.cpp`), left as is (user, 2026-10-25)
Both come from the study's own changes; main's router has neither. v3 (v3e1b's E1 onward, v3r's generic loop) fixes
both. Do not use v2 for VC1 or intermesh-edge measurements without keeping them in mind.
1. **VC1 admission on VC0's credits** (RESULTS §87, §92). The study's compile-time `admit_2d_dispatch` (§53) reads
   `vc0_local_free_slots_stream_id` for the VC1 receiver too: the hardware binary reads downstream streams 4/5/6 and
   never 10–13. A forward can then overwrite an unsent packet in a 2-slot VC1 channel. The v3u3 hang (§90) is
   consistent with it.
2. **Deferred flushed atomic vs intermesh egress** (V3_REVIEW.md a1.1). A deferred flushed atomic stays staged in
   `write_at_cmd_buf` until it fires. The generic receiver step's bypass test lets an intermesh-exit packet through,
   and that packet's stateless egress credit write (`noc_inline_dw_write`, same command buffer) overwrites the staged
   atomic. Reachable only on a 2D intermesh edge router with deferral on (no logged config). The fix is
   `!intermesh_egress` in the bypass test, as v3's E1 has.
