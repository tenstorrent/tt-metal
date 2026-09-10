# VSA lever log (started 2026-09-09)

Baseline (v19, commit 259c14821bc): real dev-5 indices 23.0 ms standalone; synthetic median-model 19.7 ms
(20.0% listed-math util). Probe 9: compute TRISCs wait 43-47% for windows; delivery floor (probe 1) 10.05 ms;
compute busy ~12.7 ms -> serialized. Phases (MATH): qk 13%, max 17%, pv-drain 15%, exp 7%, flush 3%.
No-sums 21.3 ms (sum traffic 1.7 ms).

## Lever 1: three-stage ring (fill / compute / pending-PV)
Implemented: VSA_STAGES define (factory kStages=3, TT_VSA_STAGES env; default depth 6*stages=18), worker
reader pending/pendq/half_outstanding/trid groups generalized, leader-as-worker kOwnWin=depth/stages,
writer pull_idx/ASSERT. Unit suite (22), determinism, trace replay, precision: all pass at 3 stages.
Results (real dev-5 shard): 2 stages/depth16 23.08 ms -> 3 stages/depth18 22.60 ms (-2%).
Synthetic median-model: 19.7 -> 20.1 ms (smaller windows: +18% visits). Depth 24 overflows L1 (1.63 MB).
Probe 9 at 3 stages: wait 37-42% (was 44-47%), visits 1567 -> 1851, total ticks -1.5%.
=> Ring stages were NOT the main coupling. Keep 3 stages only if later data favors it.

New hypothesis (from code): the LEADER's own resident rows gate its fetch (own_commit at distance
depth - fetch lag), and every worker's window completes no faster than the leader publishes, so the
group is effectively a one-window-lookahead pipeline: per window = compute (~14 us) + fetch (~11 us)
= 25 us x 904 windows = 22.6 ms (matches). Test: TT_VSA_LEADER_ROWS=0 (pure streaming leader).
Also queued: depth 21, no-sums at 3 stages, rmax 8/12, phase-2 knobs TT_VSA_EXP=1..4.

chain2 results (real dev-5 shard, 3 stages unless noted):
- pure streaming leader (TT_VSA_LEADER_ROWS=0): 24.8 ms (2 stages: 25.1; depth 21: 24.5) -> WORSE. Leader
  credits are not the coupling. Knob kept (experiment only).
- depth 21: 22.4 (-0.2). no-sums: 21.3 (sum traffic 1.3). rmax 8: 23.1; rmax 12: 22.9 (no gain).
- phase-2 knobs: no anchor copy 22.7 (nothing); NO DECISION 18.9 (-3.7 ms!); no decision+no RISC sync 18.8;
  no reduce (decision on stale data) 21.5 (reduce+pack ~1.2 ms). => the UNPACK-RISC compare loop (64
  strided L1 reads x2 per visit) is the phase-2 cost, MATH/PACK idle at the mailbox meanwhile.
- cb_ctrl 8 -> 64 pages: 22.56 / 22.96 (3 / 2 stages) -> no change; reader was not blocking there. Kept (3.5 KB).

## Lever 2: cheap decision (implemented, testing)
Threshold keys cached contiguously per row slot in the unused cb_sum_res tile, refreshed only for rows whose
threshold changed (per-thread dirty bits); visits split even/odd between UNPACK and MATH RISCs (MATH waits
on the packer via PACK_DONE, unused by LLKs), masks exchanged by mailbox. Same decision, same numerics.
Expected: 3.7 ms -> ~1 ms.

Ideas queue:
- Instrument the stream reader/writer/leader under probe 9 (spin-on-log / credit wait / emit-blocked /
  leader gate / kack wait) to find the remaining ~40% compute wait -- three theories failed, measure.

Lever 2 RESULT: real shard 22.56 -> 20.09 ms (3 stages) / 19.68 ms (2 stages, depth 16); synthetic median
20.1 -> 17.4 ms (22.6% util). All 22 unit cases + determinism + precision + trace pass. Probe: max phase
5.76M -> 3.78M ticks; total 27.7M -> 24.9M; compute WAIT unchanged at ~10.4M ticks (an absolute block).
With the cheap decision, 2 stages beat 3 (wider windows win) -> revisit default after lever 3.

## Lever 3: dataflow-side probe (new timers VSAL/VSAW/VSAS under probe 9)
Leader reader: own-credit gate 31%, worker gate 18%, kack 2%. Worker readers: spin-on-log 62-87%,
credit wait 5-29%. Worker writers: sum service 43%, kreq/K pulls 34%.
Root cause candidate: while the leader is gated, its kFetchLag(4) prefetch queue stays UNPUBLISHED
(publish only happens on the next fetch), so the tail of the open window is withheld for the whole
stall and no consumer can close it / return credits -> latency-bound lock-step.
Fix: publish the whole prefetch queue before blocking on a closed gate (non-blocking gate check first).
First attempt HUNG: the leader writer acks K lazily (kAckLag 2, on the next kreq or sentinel), so
publishing the newest blocks waited for acks that only the post-gate kreq would trigger. Fixed by having
the leader writer ack landed blocks (non-blocking trid check) whenever idle. Unit suite passes.
Lesson: every lazy handshake in this protocol (acks, publishes, credits) must have an idle-time drain,
or any change to who-waits-first deadlocks.

Lever 3 continued:
- gate drain + eager leader-writer acks: 20.15 (3st) / 19.74 (2st d16) / 19.19 (2st d20) -> neutral vs
  before (20.09 / 19.68); kept (needed for any wider-bin variant; harmless).
- worker emission latency probe: close->emit small (0.7M ticks over 338 windows); not the coupling.
- KEY OBSERVATION: worker (1,0) closed only 338 windows over 5406 arrivals (bins of 6): real selections are
  spatially clustered -> a core's rows want long empty stretches then bursts. Per-window compute is bursty
  while the leader's fetch per window is constant; an 18-arrival ring cannot absorb it, so the leader
  stalls on the slowest consumer (own 28%, workers 17%) and everyone else's compute idles.
- wide worker bins (18 arrivals, close on 6 slots): 20.30 / 19.12 ms -> neutral; and it breaks the
  raw-vs-assembled bit-equality (partition now depends on co-resident rows). REVERTED.
- Best config so far: 2 stages, depth 20, cheap decision: 19.19 ms (from 23.0 at session start, -17%).
Next: TT_VSA_LEADER_SHARE sweep (leader takes fewer rows); strided stream order to de-cluster bursts.

## Lever 4/5 and wrap-up (2026-09-09 late)
- leader share sweep (2 stages, depth 20): 100% 19.09 | 70% 21.64 | 50% 21.71 | 30% 21.31 -> WORSE. Knob kept as experiment.
- stream orders (2 stages, depth 20): identity 19.01 | stride (de-clustering) 19.94 | zorder 20.58 -> identity stays.
  The permuted leader loop no longer deadlocks (the publish-before-gate drain removed the latent race).
- leader active-loop breakdown (probe 9): V-read issue 20% (~940 ticks/block for 8 accessor reads), publish 13%,
  own-consume 9%; stalls: own-credit gate 30%, worker gate 17%.
- DEFAULTS now: 2 stages, depth 20, cheap decision. Real shard 19.24 ms (was 23.0, -16%); synthetic median-model
  15.96 ms @ 24.7% util (was 19.7); worst-shard model 22.8 ms; 10 s 7.86 ms; 5 s 2.84 ms.
- Gates at defaults: unit suite 47 passed (incl. dist), determinism/precision/trace 17 passed; galaxy oracle and
  15 s block: see VSA_STREAM_DESIGN.md 10 / commit message.
- Lever 4 (exact row sums, 1.3 ms) not attempted this pass.
- Open: the ~40% compute idle. Six hypotheses measured and rejected (ring stages, control ring depth, leader
  credits, withheld prefetch publishes, wide bins, de-clustering). Next tool: an event timeline in an L1 trace
  buffer (leader publish / gate enter-exit vs compute window start-end) to see which consumer gates which window.

Depth 20 does NOT fit the model: traced 15 s block and the attention tests clash with the live L1 buffers
(CB region ends 1528640 / 1518528 vs L1 buffers at 1504000). Default set to 2 stages, depth 18 (windows of 9);
TT_VSA_DEPTH=20 stays available standalone. Final numbers at depth 18: see the end of VSA_STREAM_DESIGN.md 10.

FINAL (2 stages, depth 18): real shard 19.56 ms; synthetic median-model 16.51 ms @ 23.9%; block profile
vsa_sdpa 17.47 ms slowest device (was 21.52), block period ~59.8 ms (was ~64.0). All gates green.

## Event timeline (probe 10, 2026-09-10)
Tooling: kernels/vsa_trace.hpp; per-RISC 4 KB event regions in a probe-only CB, dumped by the WRITER of two
host-selected physical cores (TT_VSA_TRACE_CORES="1-2,2-2" = logical (0,0),(1,0)); scratch timeline.py.
Lessons: (1) a DPRINT on any core the print server is not draining blocks forever -> "compute DPRINT hangs"
was this all along; (2) compute get_tile_address() is a cross-thread handshake, never call it under MATH((...));
(3) writes during a disk-quota failure truncated five source files to 0 bytes -- verify sizes after edits.
Findings (last launch, real dev-5 shard, 2 stages depth 18):
- worker (1,0), 2.26 ms window: compute busy 42%, idle 58%; 82% of the idle time is while the reader spins
  for the leader's NEXT publish (331 spins of ~4.2k ticks = one publish period), 3% during credit waits.
  Windows closed = emitted = 44; 80 chunks.
- leader (0,0): publishes a pair every ~4870 ticks (2435 ticks/arrival) while its gates are open; in the
  captured stretch the leader's own compute is idle 83% and the leader is active (not gated) 92% of that.
=> In sparse stretches the LEADER'S ISSUE RATE is the bottleneck (workers and the leader's own compute wait
   for publishes); in bursts the compute is. The ring cannot average them, so total ~ sum. The leader's rate
   is dominated by the V-read issue (~940 ticks/block: the generic TensorAccessor decomposes each page id
   into 4-D coordinates). Fix under test: InterleavedAddrGen<true> for the leader's V (reader) and K
   (writer) fetches.
Follow-ups (2026-09-10):
- InterleavedAddrGen for the leader's V/K reads: V-issue ticks 5.09M -> 4.61M only; kernel 19.56 -> 19.50 (noise).
  => per-read cost is NoC command issue (~100 ticks each), not address arithmetic.
- V fetch moved to the leader's writer RISC (TT_VSA_V_ON_WRITER=1): 20.6 ms, WORSE -- reader viss -> 0.37M but its
  kack wait +5M: one NoC then carries all 16 reads per block instead of 8+8. Kept as a knob.
- probe 3 (workers skip all K/V pulls): 18.87 vs 19.48 -> the leader's L1 egress to 7 pulling workers is NOT the
  limit either (0.6 ms).
=> The leader reader is instruction/issue-bound: per pair ~16 read issues + 7 unicast publishes + gate check
   (7 progress words) + own-consume scans ~= 4.9k ticks. Levers: NoC state reuse per DRAM bank per pair
   (8 set-state + 16 light issues instead of 16 full), cached gate slack (skip the 7-word check while slack
   remains), multicast publish (1 command instead of 7; the old deadlock may be gone with the gate drain).
- NoC state reuse per bank (8 set-state + 16 light issues per pair) + cached gate slack: 19.57 ms, and the
  leader's V-issue ticks did NOT move (4.85M). Prefetch depth 2/6/8: 19.55 / 19.61 / 19.72 (flat).
=> The leader's V issue time is invariant to the API, to the instruction count and to the in-flight depth,
   and independent of worker pulls. What fits all of it: CHIP-LEVEL DRAM BANDWIDTH. 14 leaders x 32 KB per
   2.4k-tick arrival = ~240 GB/s of 2 KB reads, and the probe-1 delivery floor (10 ms) is exactly the
   2.42 GB of K/V (3 passes x 1808 blocks x 32 KB x 14 heads) at that rate. The kernel is DRAM-streaming
   (10 ms) + compute (11 ms), poorly overlapped because per-window compute is bursty.
   LEVER: fewer passes (each pass re-reads all K/V). 226 rows / 9 consumers = 25.1 rows per core: rmax 10
   -> 3 passes, rmax 13 -> 2 passes (-33% DRAM traffic). Testing rmax 13/14 at depths 14-18 (L1 trade).
- rmax 13 (2 passes): depth 18 overflows L1 by 8 KB; depth 16: 20.25 ms -> WORSE than 3 passes at 19.5.
  DRAM bandwidth is not the limiter either.
CONCLUSION of the timeline work: the ~40% compute idle is a CONVOY. The leader may run only its ring depth
(18 arrivals, ~2 windows) past the slowest of its 9 consumers; per-window compute varies 2-3x between cores
(clustered selections), so half the time the leader is gated on the slowest consumer while the others idle,
and the other half it publishes at its maximum cadence while everyone waits for it. Nothing that changes the
leader's cadence (RISC work, NoC API, in-flight depth, egress, DRAM traffic) moves the total, and the ring
cannot grow on cores that also hold resident rows.
NEXT DESIGN (not built): role-specific L1. A non-computing leader needs no Q/O/qk/max/corr residency
(~600 KB), so its K/V ring can be ~36-40 arrivals (4-5 windows of slack for the workers) instead of 18.
Needs per-role CB sizes (same CB index, different pages on leader vs worker ranges; stream rings first so
their L1 base is shared), the log ring sized from the leader depth, and the leader's compute share (-11%
capacity) given back to workers. Expected: convoy losses shrink toward the compute bound (~12.5 ms x 1.13).

## Deep leader ring (building, 2026-09-10)
Per-role CB sizes: cb_roles() pushes a worker-grid and a leader-grid descriptor for EVERY CB (a CB spanning both
roles gets one address = the max region end over its cores, which pushed all later worker CBs past the leader's
deep ring: worker L1 1.75 MB). Cross-core L1 addresses must come from CBs that precede the first role-dependent
one: cb_log (leader -> worker entries) and cb_ackbox (worker -> leader READY/progress) are now CB 0 and 1, then
cb_k_stream (shared base), cb_v_stream; a worker pulls V from the leader at K base + leader_depth x K block.
First attempt hung in the READY handshake for exactly this reason. Leader: kLeaderDepth 40, TT_VSA_LEADER_DEPTH,
TT_VSA_LEADER_ROWS=1 = old leader-as-worker layout. Compute already returns early on row_count 0.
Small shapes pass; measuring.
RESULT deep leader (depth 40, no leader rows): real shard 20.92 ms (baseline 19.5) -> WORSE; depth 56 overflows the
leader's L1 (1.99 MB; ~42 is the ceiling). Probe: arrivals 7208 (a 4th pass: 8 consumers, chunk-cyclic dealing gives
one worker 32 rows), leader gated on worker progress 47%, own 0; worker (1,0) reader waits for its OWN compute's
credits 45%, compute busy 64%. The traced 15 s block HANGS with it (timeout). Made opt-in (TT_VSA_DEEP_LEADER=1).
Reading: the workers' two-window rings cap how far each worker can run ahead, and workers' work is concentrated in
different stretches of the stream (spatially clustered rows), so their lag grows far beyond any leader ring depth.
The remaining lever is the stream ORDER: interleave runs of consecutive blocks from several spatial segments so one
window carries work for several workers while multi-block visits survive (bstrideR.S in the real perf test).

## Blocked-stride stream order (2026-09-10) -- the convoy lever that works
Real dev-5 shard, default kernel config: identity 19.53 | bstride2.8 18.84 | bstride4.8 18.92 | bstride8.8 18.27 |
bstride16.8 19.27 | bstride4.16 17.68 | bstride8.16 18.57 | bstride2.32 18.35 | bstride4.32 17.72 | bstride8.32 18.75 |
bstride4.64 18.96 | bstride8.64 18.94. Best: runs of 4 blocks over 16 segments, -9.5%. (PCC vs identity 0.9995:
a different but equally exact rounding order; each order is deterministic.) The permuted leader loop no longer
deadlocks (publish-before-gate drain). Model default set to "bstride4.16" (MiniMaxH3VSAGeometry.stream_order).
Model gates with bstride4.16 default: geometry 17 passed, stages oracle 11, attention oracle 99.51-99.57 %, sparsity-0
99.97 %, traced 15 s block bit-exact. Block profile: vsa_sdpa slowest device 17.47 -> 16.70 ms, block ~59.8 -> 59.2 ms.

## E2E (2026-09-10 afternoon)
- e2e t2va 50 steps with the bstride4.16 default HUNG in the warmup denoise (device 0 head 13: leader in
  wait_all_workers_at, all 8 workers spinning on the log, writers/computes idle). Identity order hung the same way
  (device 23 head 2). Neither of this session's kernel commits had been run e2e (block test only).
- Bisect: kernels @73bd7a38847 (lever pass) PASS e2e: denoise 150.2 s / 50 steps = 3.005 s/step (checkpoint 09-09:
  160.7 s, 3.28 s/step); total warm ~180 s (was 193.2). HEAD factory+compute with 73bd reader+writer PASS (4 steps)
  -> the deadlock is in this session's reader/writer protocol changes (gate drain/cache, eager acks), i.e. the
  documented timing-sensitive NOC_0 ack cycle re-exposed by the changed leader-loop timing.
- Fix under test: READY and progress posts as POSTED inline dword writes (no ack dependency).
- Posted inline dword writes for READY and progress posts: HEAD kernels pass the 4-step e2e (denoise 9.0 s, was hanging).
  Root cause matches the design doc's "latent timing-sensitive deadlock": non-posted progress posts wait for acks on
  NOC_0 alongside the workers' V pulls and the leader's publishes; a timing change in the leader loop (the gate drain)
  closed the cycle. Posted writes remove the ack dependency. Final 50-step e2e with bstride4.16 running.
- Posted posts did NOT fix it: identity 50-step run hung in its first step too (4-step pass was luck); bstride 50-step
  hung identically (device 26 head 10). REVERTED the leader protocol changes to the pre-session loop (no drain, no
  gate cache, no eager acks, non-posted posts); all were perf-neutral. Consequence: the bstride order needs the drain
  (its own permuted-order deadlock), so the model default is back to identity until the protocol race is understood.
  Stuck-state probe (probe 11: leader gate target + every worker's posted progress; worker consumed/posted/post_limit/
  pending) is staged in scratch (apply_stuck_probe.py) for the next investigation.
- FINAL e2e (restored protocol, identity order, 50 steps, real weights): denoise 151.3 s = 3.026 s/step, total warm
  181.4 s (dense 325.9 s -> 1.80x; VSA checkpoint 09-09 was 193.2 s / 3.28 s per step).

## Cleanup (2026-09-10, end of day)
Kept: decision rewrite, 2x9 ring (depth 18), cb_ctrl 64, e2e perf test, bstride order in the geometry (parked, default
identity), docs. Removed: dataflow probe-9 timers, probe 10 (vsa_trace.hpp, vsa_timeline.py), probe 3, TT_VSA_STAGES,
TT_VSA_LEADER_ROWS/SHARE/DEPTH, TT_VSA_DEEP_LEADER + cb_roles + CB reorder, TT_VSA_FETCH_LAG, TT_VSA_V_ON_WRITER,
generator DRAM reads, stuck probe. Reader and writer are the checkpoint (259c14821bc) files again.
