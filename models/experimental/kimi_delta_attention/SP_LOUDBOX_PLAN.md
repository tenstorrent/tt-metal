# KDA LoudBox sequence-parallel prototype

## Objective

Turn the existing `TP=8` whole-head KDA implementation into a correctness-first
sequence-parallel prototype on the available eight Blackhole devices, without
regressing the existing path.  The production scale-out shape remains
`SP=8 x TP=4` on 32 Galaxy devices; LoudBox is the place to validate the
state-transfer protocol and measure its components.

## Baseline (verified on this branch)

* Branch: `pjosipovic/kda-loudbox-sp-prototype`, based on Momcilo's
  `mvasilijevic/codex/kimi-linear-kda` at `671b0c9acb6`.
* Hardware: one eight-device Blackhole LoudBox.
* `test_ttnn_layer.py`: 8 passed (T=1, 4, 32, state continuity and external
  state tests).
* `test_tp_weights.py`: 2 passed; TP=8 output PCC 0.999965, recurrent-state
  PCC 0.999910, convolution-state PCC 0.999997.
* The traced LB TP=8 control at T=5120 completed in 3.114 ms on the slowest
  device (three warm trace replays; report
  `2026_07_25_16_17_42`).  Historical branch reports measured 619.594 us at
  T=640 and 3183.263 us at T=5120.  At T=5120 the output projection's fused
  matmul/reduce-scatter is about 1.04 ms, so its CCL is a primary reason that
  merely tuning local KDA will not meet the next goal.

## Current implementation and measurements (2026-07-25)

* SP=2/TP=4 is functional at the production-rank-equivalent shape: global
  T=5120, H=2304, 32 heads, 8 heads/rank.  Output, recurrent cache, short-conv
  cache, and the first post-boundary token pass PCC >= 0.98.
* TP=4 needs a 96-tile causal-convolution input while the original direct
  kernel holds all channel tiles in each worker's L1.  The kernel now splits
  that into two 48-tile channel blocks; the target PCC test validates the
  split convolution through the full layer.
* The TP=4 long path uses explicit post-scan KDA RMS normalization.  The
  TP=8-only fused scan/RMS variant produced incorrect TP=4 long outputs even
  though the recurrent cache was correct.
* The original parent-mesh warmed three-replay trace at global T=5120 is
  **5.707 ms/forward**
  (report `2026_07_25_18_16_20`), versus the <= **2.958 ms** primary gate.
  This prototype therefore misses the gate by 2.749 ms (93%).  The comparable
  eager signpost is 5.961 ms/forward.  The experimental local
  `QWEN_KDA_GROUP_PREFIX=1` route is correct but slower, 13.197 ms/forward;
  it is not an optimisation to enable.
* The remaining critical-path cause is architectural: first-span recurrence,
  cache transfer, and second-span recurrence are ordered.  Transport tuning
  cannot recover the 2.749 ms deficit.  The next implementation target is
  Milestone 4's inter-span affine scan, which makes the large span scans
  concurrent after a small log-depth prefix over `(A, B)` summaries.
* An opt-in `KDA_SP_AFFINE=1` SP=2 prototype validates the affine state
  transition and overlaps prepared final scans.  Its direct state-only
  span-summary kernel was correct but measured **6.821 ms/forward** (three
  repetitions; report `2026_07_25_18_36_33`).  Replacing it with eight-chunk
  summaries plus the device `KdaAffinePrefixOperation` preserves the full
  T=5120 PCC gate, but is **7.572 ms/forward** (report
  `2026_07_25_18_44_53`): it still computes a summary scan and a final output
  scan.  Both remain disabled by default.
* An opt-in `KDA_SP_PIPELINED=1` control overlaps span two's convolution/input
  preparation with span one's normal recurrence, then transfers the ordinary
  final recurrent state.  It passes the same PCC gate, but measures **7.050
  ms/forward** (report `2026_07_25_18_46_43`), so host scheduling alone is
  also not the lever.
* The concrete next target is a split-phase grouped KDA interface: retain the
  existing `chunk_gdn_prep` results, emit group `(A, B)` summaries and their
  affine prefix, send the prefix-derived final state immediately, then run
  both spans' group output scans from those already-computed entry states.  It
  must eliminate the duplicate summary prep/scan before this topology can
  plausibly reach the 2.958 ms LB gate.
* That split-phase interface is now implemented behind
  `KDA_SP_SPLIT_AFFINE=1`.  Its reusable preparation, summary, prefix, and
  seeded grouped-output scan agree with the regular KDA path (primitive PCC
  >= 0.999; end-to-end SP PCC >= 0.98).  The first three-repetition eager
  measurement is **7.771 ms/forward** (report `2026_07_25_18_55_46`), still
  above the serial control.  Thus removing duplicate preparation alone is not
  sufficient: the next investigation must make the two TP=4 final scans
  genuinely concurrent at the queue/fabric level, rather than only enqueueing
  their work from the host in that order.
* Capturing one trace per TP=4 child mesh fixes that scheduling artifact: the
  parent 1x8 mesh owns no KDA queues, whereas each child trace owns one side
  of the socket handoff.  With `KDA_SP_SPLIT_AFFINE=1`, the warmed T=5120
  replay was **4.364 ms/forward** at the old 10 KiB socket FIFO (report
  `2026_07_25_19_02_45`).  The recurrent transfer was then the critical
  bottleneck: its sender took roughly 1.3--1.5 ms.
* Sweeping the direct fabric FIFO shows that this is pure backpressure, not
  an unavoidable link latency: 64 KiB gives **4.050 ms** (report
  `2026_07_25_19_04_31`), 256 KiB gives **3.967 ms** (report
  `2026_07_25_19_05_19`), and one full-state **512 KiB** FIFO gives
  **3.297 ms/forward** (report `2026_07_25_19_06_19`).  At 512 KiB, the
  recurrent send/receive is about 0.10--0.12 ms and the full target-shape
  PCC gate passes.  The 512 KiB depth is consequently the default, with
  `KDA_SP_SOCKET_FIFO_BYTES` retained as an explicit sweep override.
* A fresh 512 KiB replay measured **3.317 ms/forward** (report
  `2026_07_25_19_24_11`).  This still misses the primary head-to-head gate by
  0.359 ms (12.1%).  Transport is now only about 0.10--0.12 ms, so the next
  optimization must remove split-affine scan/materialization work on the
  child-trace critical path rather than further tuning the FIFO.
* The planned SP=8/TP=1 seven-boundary protocol probe is now implemented as
  `SP8TP1KimiDeltaAttention`. At global T=5120 it executes eight real
  640-token KDA spans with all 32 heads/chip, passes output/recurrent/
  convolution PCC >= 0.98, and validates the first output token after every
  one of the seven fabric handoffs. The 640-token span previously selected
  the generic convolution path and exceeded L1; the channel-blocked direct
  KDA convolution now applies at `T >= 640`.
* The TP=4 performance control must open the physical 1x8 LoudBox with 2D
  fabric and profile its first logical 1x4 submesh. Opening only a physical
  1x4 mesh makes fabric-router initialization time out; the logical submesh is
  the same TP group used by SP=2/TP=4 child traces.
* With that apples-to-apples trace harness, the initial T=1280 controls are:
  TP=4 T=640 **1.071 ms** (report `2026_07_25_19_32_51`), TP=4 T=1280
  **1.590 ms** (report `2026_07_25_19_34_47`), and SP=2/TP=4 global T=1280
  **1.505 ms** (report `2026_07_25_19_33_50`). The last result is a
  **serialized fallback**, not a split-affine result: each 640-token span is
  not divisible by the current 256-token group. It is useful as a legacy
  control (1.056x equal-global speedup and 1.405x local-work latency), but it
  is not a valid measurement of the intended production-rank affine path.
* Group preparation now selects 128-token groups when a span is not divisible
  by 256, which makes the T=640 span five groups while retaining eight-chunk
  groups for the T=2560 production-equivalent path. The primitive has PCC >=
  0.995 at T=640 and the full T=1280 SP test passes the output, recurrent,
  convolution, and boundary-token PCC >= 0.98 gate. The valid warmed
  split-affine child-trace result was **1.438 ms** (report
  `2026_07_25_19_42_27`). Fusing the terminal affine application into the
  prefix program removes the DRAM slices and standalone matmul/add, reducing
  it to **1.396 ms** (report `2026_07_25_19_49_41`): 1.139x the equal-global
  TP=4 control and 1.303x the local-work control. It now misses the actionable
  1.339 ms production-rank budget by only **0.057 ms**; the next target is the
  remaining final-scan/epilogue critical path.
* Skipping the now-redundant first grouped scan's final-state write is correct
  but exposes no measurable critical-path gain: T=1280 measured **1.400 ms**
  (report `2026_07_25_20_00_07`).  The write was already hidden by scan and
  epilogue work.  The API retains this mode to avoid consuming the unwritten
  state buffer.
* Socket page striping over two worker/link pairs is correct at both T=1280
  and T=5120.  With `KDA_SP_SOCKET_LANES=2` and 512 KiB FIFO *per lane*, the
  initial T=1280 trace improved to **1.362 ms** (report
  `2026_07_25_20_05_26`), and avoiding a subsequent unused slice/read of the
  intentionally unwritten first scan state improves it again to **1.343 ms**
  (report `2026_07_25_20_16_29`). T=5120 improves from 3.317 ms to **3.219 ms** (report
  `2026_07_25_20_13_42`).  This is an opt-in experiment because the socket
  runtime warns that multiple sender cores may have fabric limitations.
  It remains 0.004 ms over the T=1280 budget and 0.261 ms over the T=5120
  head-to-head gate.
* A TP=4 output all-reduce Ring-topology A/B deadlocked under the guarded LB
  test and was removed.  Keep the validated Linear collective; do not retry
  the ring variant without a CCL protocol investigation.
* Two subsequent output-fusion probes confirm that this is not just a choice
  of all-reduce topology.  Replacing the TP=4 linear-plus-Linear-all-reduce
  output path with the existing `matmul_reduce_scatter_async` primitive was
  correct to compile but timed out on the two TP=4 child meshes with both its
  Ring and Linear reduce-scatter modes (guarded T=1280 PCC tests on
  2026-07-25).  Both probes were removed and the normal path subsequently
  passed after the safe device reset.  Treat child-mesh fused MRS support as
  a separate CCL/runtime investigation, not a KDA-layer toggle.
* A terminal-only affine-prefix attempt made the boundary carry available
  before materializing the first span's entries, but needed a second prefix
  to obtain those entries.  It remained correct at T=1280 but regressed the
  three-replay boundary to **1.400 ms** (report `2026_07_25_20_28_29`) from
  1.343 ms, so it was removed.  Future scan work must eliminate, rather than
  duplicate, prefix or final-scan work.
* Grouped scan-to-gated-RMS fusion was made correct by waiting until all
  independently-produced group-head slices were present before the RMS reader
  drained them.  It narrowly met the T=1280 gate at **1.338 ms** (report
  `2026_07_25_20_47_15`), but regressed the representative T=5120 trace to
  **3.311 ms** (report `2026_07_25_20_49_36`) from 3.219 ms.  The experiment
  was removed: its cross-group synchronization serializes enough work to lose
  at long sequence length.  A viable fused epilogue needs ordered per-output
  publication or a staging design that avoids that global consumer wait.
* Splitting chunk preparation at the causal-convolution boundary lets the
  producer queue its convolution-cache send before independent decay/gate
  work.  This preserves the normal layer path and passes direct single-device
  PCC plus SP2xTP4 PCC at T=1280 and T=5120.  The child-trace boundary improves
  to **1.332 ms** at T=1280 (report `2026_07_25_20_56_27`) and **3.206 ms** at
  T=5120 (report `2026_07_25_20_57_59`), versus 1.343 ms and 3.219 ms.  It is
  the new baseline, but still misses the 2.958 ms production-rank target by
  0.248 ms; the next optimization must address the final scan/output path.
* Grouped-recurrence granularity was probed at the T=5120 production-rank
  shape.  Four chunks per group creates 160 group-heads on TP=4 and violates
  the 120-core grouped-scan mapping.  Sixteen chunks per group is correct but
  measured **3.249 ms** (report `2026_07_25_21_04_31`), slower than the
  eight-chunk 3.206 ms baseline.  Keep the default eight-chunk grouping; it is
  the best viable layout tested on LoudBox.
* TP=4 output reduce-scatter benefits from a length-aware worker schedule.
  Four workers per link is PCC-correct and improves T=5120 to **3.196 ms**
  (report `2026_07_25_21_07_47`) from 3.206 ms, but regresses T=1280 to
  1.456 ms (report `2026_07_25_21_09_28`).  The implementation therefore uses
  two workers below the 1024-token local-span crossover and four at or above
  it; the environment can still override the choice for future CCL tuning.
  Other safe knobs were negative: 20 sync chunks measured 3.215 ms (report
  `2026_07_25_21_11_53`), one link 3.581 ms (report
  `2026_07_25_21_13_35`), and three workers 3.207 ms (report
  `2026_07_25_21_15_17`).
* Extending that safe output-CCL sweep did not find a fifth-worker or deeper
  channel-pipeline win.  Five workers completed ten traced T=5120 replays
  (report `2026_07_25_21_23_07`) but its per-device replay-span median was
  2.681 ms versus 2.677 ms for the four-worker report.  Three buffers per
  channel also passed the full SP2×TP4 PCC suite, but increased the output-RS
  device median from 442.8 to 449.2 us and the same replay span to 2.683 ms
  (report `2026_07_25_21_26_52`).  The implementation remains at four workers
  and two buffers.  Five chunks per sync was also correct but regressed the
  RS median to 451.4 us and the replay span to 2.687 ms (report
  `2026_07_25_21_30_20`).  Do not add further direct-RS scheduling knobs
  without a measured long-span gain.
* The TP=4 fused `matmul_reduce_scatter_async` path initially exposed a real
  CCL factory error: a request for `Linear` constructed the Ring
  reduce-scatter program.  The factory now selects the Line builder and its
  runtime override; focused 1x4 child-mesh tests cover both LoudBox TP=4
  placements, T=1280 local spans, FP32 activations, and the first output tile.
* The apparent remaining KDA boundary corruption was a host-layout bug in the
  diagnostic, not a live-schedule CCL defect.  Fused MRS returns
  `[B, 1, T, H/TP]`; the test concatenated its flattened values correctly but
  indexed the singleton dimension as though it were `[B, T, H/TP]`, yielding
  an empty boundary slice and PCC 0.0.  Normalize the fused result back to
  `[B, T, H/TP]` before the SP layer consumes it.  The full target-shape
  SP2xTP4 PCC suite now passes with the TP=4 fused Line MRS enabled.
* Line MRS needs two input-sized intermediate regions for its forward and
  backward partial reductions, unlike Ring MRS.  The shared prefill buffer is
  therefore cached by topology and allocates the required Line intermediate.
  The next decision is performance, not another CCL correctness investigation.
  The ten-replay T=5120 child-trace measurement with the corrected fused path
  is **2.854 ms/forward** median on the slowest device (sessions 2--11;
  report `2026_07_26_09_22_22`).  This beats the 3.114 ms TP=8 control by
  8.4% and clears the <=2.958 ms primary LoudBox gate by 104 us (but not the
  2.803 ms stretch goal).  The first measured replay was a 3.922 ms outlier;
  the remaining nine span 2.842--2.858 ms.  This supersedes the signposted
  host-interval figure from `2026_07_26_09_19_57`, which is not the
  slowest-device firmware metric used by this plan.
* A fresh ten-replay post-barrier rerun of the unchanged SP2×TP4 child-trace
  control confirms the result: sessions 3--11 have a **2.855 ms**
  slowest-device median (range 2.848--2.860 ms; report
  `2026_07_26_15_59_04`).  The first two sessions are 3.925 ms cold/warm
  outliers and are excluded by the same steady-state convention as the prior
  report.  This is statistically indistinguishable from 2.854 ms, remains
  8.3% below the 3.114 ms TP8 control, and clears the <= 2.958 ms gate by
  103 us.
* The prioritized path to the best measurable LB performance is documented in
  [SP_PERFORMANCE_OPTIMIZATION_PLAN.md](SP_PERFORMANCE_OPTIMIZATION_PLAN.md).
  It starts with the fused TP4 output MRS/clone tail, whose dependent union is
  about 916 us, before considering further SP transport tuning.
* The principal Galaxy-scale risk now has a LoudBox probe: an SP=8,
  three-stage Hillis--Steele affine prefix transfers the exact per-TP4-rank
  payload, eight FP32 `[128,128]` A matrices plus eight `[128,128]` B matrices
  (512 KiB + 512 KiB = 1 MiB/rank).  All eight inclusive transforms and their
  nonzero-state entry states pass PCC >= 0.999 against the host composition;
  there is no host state handoff.  The traced standalone prefix is stable with
  one socket lane at 2.333 ms (three measured replays; report
  `2026_07_26_10_29_27`) and two lanes at **1.274 ms** median (ten sessions
  2--11, range 1.272--1.275 ms; report `2026_07_26_10_31_24`).  Two lanes are
  correct and 45% faster, but emit the runtime's experimental multi-sender
  warning; retain the lane selection as an explicit environment-controlled
  candidate.  This measures prefix transport plus general FP32 composition in
  isolation, not its overlap with real TP=4 KDA scans or a 32-device mesh.
  An intentionally optimistic overlap-capacity capture then queued eight
  independent TP=4-rank-shaped local KDA final scans behind stage-zero prefix
  traffic.  It timed out twice under the guarded runner, including after all
  socket and KDA programs were warmed, and both failures recovered only after
  the runner reset the device.  Treat concurrent prefix sockets plus KDA child
  traces as a current LB runtime/queue limitation, not a measured performance
  opportunity; it must be resolved with a trace/fabric runtime investigation
  before attempting an end-to-end SP=8 overlap implementation.
  Triage identifies the missing global stage ordering as the immediate cause:
  fabric routers were blocked on receiver credits and buffer space while
  independently replayed ranks had advanced to different prefix distances.
  An eager stage-barrier harness resolves that condition.  It queues real
  eight-head, 2,560-token local KDA grouped scans while stage-zero traffic is
  in flight, then synchronizes every device before composing that stage or
  issuing the next one.  This passed on LB with both one and two socket lanes.
  It is a stability proof only: its all-rank scan release is optimistic and
  the host stage barriers make it unsuitable as a performance measurement.
  The same ordering is now exercised as three separately captured child-trace
  distances, with a host synchronization only between trace replays.  Ten
  two-lane replays take **2.692 ms/replay** in the signposted host interval
  (report `2026_07_26_13_25_08`) for prefix plus the optimistic grouped-scan
  workload.  It is a safe scheduling baseline, not a layer benchmark: it
  omits the KDA epilogue and TP=4 CCL, and production must replace the host
  stage fences with supported device events or a fused fabric protocol.
* The first end-to-end implementation now exists as
  `SP8AffineTP1KimiDeltaAttention`, a deliberately correctness-first LoudBox
  protocol.  It derives each rank's terminal `(A, B)` map entirely on device
  from the grouped KDA transforms, performs the three fabric prefix stages,
  and uses fabric sends for both the common incoming carry and each rank's
  exclusive recurrent entry state.  It passes output, recurrent-state, and
  short-convolution-cache PCC >= 0.98 at the production-rank payload and
  global T=5120 with two prefix lanes.  The default test additionally passes
  two sequential 4K calls, covering cache reuse.  This is not a performance
  result: convolution handoff and incoming-carry broadcast are serial, and
  every prefix distance uses a host synchronization.  It intentionally has
  no TP=4 output CCL.  Its value is a device-resident correctness contract for
  the next TP=4 rank-release scheduler.
* An opt-in `KDA_SP8_RANK_RELEASE=1` variant now pipelines the incoming
  recurrent and short-convolution relays (one fence per chain rather than one
  per hop).  It keeps the first two prefix distances globally ordered, queues
  every final-distance fabric send, then releases ranks 0--3 to their real
  grouped KDA scans while that last distance drains.  This passes the same
  production-rank T=5120 PCC gate and the two-call cache-reuse gate.  The
  first three-call eager host interval is **56.189 ms/call**, versus
  **57.473 ms/call** for the matching non-release control (reports
  `2026_07_26_14_10_32` and `2026_07_26_14_09_40`), a 2.2% improvement.
  Treat this as directional only: a prior cold-profiler sample was 66.149
  ms/call, the interval includes host fences and TP=1 epilogues, and no
  slowest-device replay metric exists for this eager protocol.  It is a
  stable scheduling experiment, not a new LB or Galaxy performance result.
  The scheduler now uses fixed-address per-rank shadow buffers instead of
  allocating eight `ttnn.clone` tensors per forward; PCC and two-call reuse
  remain correct, but its first 3-call eager interval was 60.197 ms/call, so
  allocation removal is not a measurable win at this granularity.  It is
  retained as a trace-stability prerequisite.  Mesh events cannot replace the
  host stage fence: they order command queues at the *same mesh coordinate*,
  not completion across all eight coordinates; event synchronization is also
  unsupported during trace capture.  The next real fence-removal target is
  therefore a custom fused fabric protocol with an explicit cross-rank stage
  barrier, rather than an event-only Python scheduler.
* The first device-queued version of that barrier now exists behind
  `KDA_SP_DEVICE_BARRIER=1`.  It is intentionally a socket-level prototype,
  not a new C++ fused operation: after each of the first two affine prefix
  distances, a reversed binary tree gathers a one-tile completion token at
  rank 0, then the existing forward trees release it to all ranks.  Queue
  order places the gather after the local `(A, B)` compositions and the next
  prefix distance after the release, providing the missing global ordering
  without `ttnn.synchronize_device` between stages.  The exact 1 MiB/rank
  affine-prefix PCC test, the two-call SP8 KDA/cache-reuse test, and the
  production-rank T=5120 KDA PCC test all pass on LB (2026-07-26).  The
  three-repetition eager scheduler probe also completes without a timeout.
  This is a correctness/stability result only: it has not yet been profiled
  as a slowest-device interval, is not yet trace-captured, and two prefix
  lanes still emit the runtime's multiple-sender-core warning.  The next
  optimisation step is to replace the 14 control-token transfers per barrier
  with a compact UDM/custom-kernel gather-release that reuses fabric state.
* The direct TP=4 output matmul has no safe grid-only win.  An explicit 10x8
  prefill grid passed the full target-shape SP2xTP4 PCC suite, but raised its
  output matmul from about **205 us** to **223 us** and the three-replay
  T=5120 child-trace span to about **2.70 ms** (report
  `2026_07_25_22_01_05`), versus the retained four-worker baseline's
  2.677 ms.  The probe was removed.  The attempted configuration also exposed
  that `create_prefill_matmul_program_config` requires an inner-K block that
  divides the 32 K tiles exactly; do not use a 10-column grid through that
  helper without correcting that generic constraint separately.

## End-to-end execution plan (next slice)

There are two distinct finish lines.  The LoudBox end-to-end proxy is already
green: SP=2/TP=4 at global T=5120 is **2.854 ms** on the slowest device, below
the **2.958 ms** primary gate.  The remaining objective is the Galaxy-ready
layer protocol: SP=8/TP=4, with all four TP ranks performing the same
device-ordered affine prefix and with the normal TP=4 output CCL intact.

1. **Measure the socket-token barrier before changing it.** Profile its
   two inter-stage boundaries inside the standalone SP8 affine probe and the
   SP8 TP1 KDA proof, using a warm eager run and a device timeline.  Success:
   the device-barrier variant stays PCC-correct, has no runner recovery, and
   identifies the control-token cost separately from the 1 MiB `(A,B)` payload.
   This is a measurement step, not an expected layer-speedup: SP=2 has only
   one sequence boundary, so it cannot demonstrate the SP=8 overlap benefit.

2. **Replace socket control tokens with a compact trace-safe UDM barrier.**
   Implement a single mesh operation with a gather/release semaphore tree on
   dedicated fabric workers.  It must be queued after a prefix stage's local
   composition and before the next stage; it must not use host fences or
   mesh events.  Success: a single captured eight-rank affine prefix replays
   at least ten times without fabric credit deadlock.  This removes the
   current 14 tiny send/receive transfers per boundary and is the prerequisite
   for a meaningful device-side SP=8 schedule.

   The first comparable eager measurement is positive but deliberately not a
   layer claim: ten fully drained 1 MiB/rank prefixes took **11.467 ms/prefix**
   with host fences (report `2026_07_26_14_50_43`) and **8.434 ms/prefix** with
   the socket-token barrier (report `2026_07_26_14_51_33`), a **26.5%** host
   interval reduction. Inputs were prepared before the signpost and each
   iteration drained all eight queues. This proves that the host fences are
   material; it does *not* prove a slowest-device or traced speedup, since the
   barrier tokens themselves are ordinary socket operations.

   The current candidate is `KDA_SP_FABRIC_TREE_BARRIER=1`: six cached
   `generic_op` programs per boundary, each a Mesh-API fabric atomic increment
   plus a local fixed-address semaphore wait. They are deliberately enqueued
   on the existing 1x1 span submeshes, not on the parent 1x8 mesh, because
   parent-mesh work conflicts with the child meshes that own KDA's command
   queues. The first parent-mesh prototype stalled and left the board PCIe-hung;
   it was replaced with this child-mesh design. The replacement is now
   validated on LoudBox (2026-07-26): ten isolated barrier rounds pass, the
   exact 1 MiB/rank prefix has PCC >= 0.999, and the production-rank SP8×TP1
   KDA gate passes output/recurrent/convolution PCC >= 0.98. A separately
   captured prefix also completed ten one-lane replays. This establishes
   correctness and trace stability, not an end-to-end performance claim:

   ```bash
   # Isolate the six-level fabric-atomic barrier; requires ten clean rounds.
   KDA_SP_FABRIC_TREE_BARRIER_TEST=1 KDA_SP_FABRIC_TREE_REPS=10 \
     scripts/run_safe_pytest.sh -q -s \
     models/experimental/kimi_delta_attention/tests/test_sp8_affine_prefix.py::test_sp8_fabric_tree_barrier_stability

   # Then re-run the 1 MiB/rank affine prefix PCC gate.
   KDA_SP_FABRIC_TREE_BARRIER=1 KDA_SP_PREFIX_LANES=1 \
     scripts/run_safe_pytest.sh -q -s \
     models/experimental/kimi_delta_attention/tests/test_sp8_affine_prefix.py::test_sp8_affine_prefix_production_tp4_payload

   # Finally use the same device-side ordering in the full SP8 TP1 KDA proof.
   KDA_SP_FABRIC_TREE_BARRIER=1 KDA_SP8_AFFINE_TARGET_SHAPE=1 \
     KDA_SP8_PIPELINED_HANDOFFS=1 KDA_SP_PREFIX_LANES=1 \
     scripts/run_safe_pytest.sh -q -s \
     models/experimental/kimi_delta_attention/tests/test_sp8_tp1.py::test_sp8_tp1_affine_layer_pcc

   # Complete no-host-fence SP8 TP1 layer capture, including two replayed
   # calls and both persistent KDA caches, at production-rank work.
   KDA_SP8_AFFINE_TRACE_TEST=1 KDA_SP8_AFFINE_TRACE_TARGET_SHAPE=1 \
     scripts/run_safe_pytest.sh -q -s \
     models/experimental/kimi_delta_attention/tests/test_sp8_tp1.py::test_sp8_tp1_affine_trace_layer_pcc
   ```

3. **Factor the prefix scheduler by `(SP rank, TP rank)`.**  Move the current
   TP1 affine proof's preparation, terminal-map construction, entry-state
   installation, and rank-release ordering behind a per-TP-rank interface.
   Build an `SP8TP4` scheduler that launches four independent eight-rank
   prefix trees and retains the regular TP=4 tensor layout.  On LoudBox, test
   the same code in its SP=2/TP=4 projection: it must preserve output,
   recurrent-state, and convolution-state PCC >= 0.98 at global T=5120 and
   retain the existing fused Line MRS output path.

   The common `SPTPTopology` mapping now accepts both LoudBox's flattened
   `1 x (SP*TP)` layout and Galaxy's native `SP x TP` layout.  A serial
   `SP8TP4KimiDeltaAttention` baseline now constructs eight real TP=4 layers
   and seven rank-aligned cache handoffs on an `8 x 4` mesh.  It is deliberately
   not an affine-performance path; it is the Galaxy correctness reference that
   the prefix scheduler must replace.  Its first hardware gate is:

   ```bash
   scripts/run_safe_pytest.sh -q -s \
     models/experimental/kimi_delta_attention/tests/test_sp8_tp4.py::test_sp8_tp4_serial_layer_pcc
   ```

   `SP8AffineTP4KimiDeltaAttention` now applies the same three-stage affine
   prefix to those eight TP=4 groups, with four rank-aligned socket streams at
   each SP edge and the normal TP4 CCL retained in every layer. The
   fabric-atomic candidate now creates four independent trees (one per TP
   rank) and dispatches each program through its owning TP4 span mesh. It is
   still unvalidated on hardware; use host fences first, then qualify the
   atomic-tree mode before profiling. The initial Galaxy affine correctness
   gate is:

   ```bash
   # Qualify all four independent TP-rank atomic trees first.
   KDA_SP8TP4_FABRIC_TREE_BARRIER_TEST=1 KDA_SP_FABRIC_TREE_REPS=10 \
     scripts/run_safe_pytest.sh -q -s \
     models/experimental/kimi_delta_attention/tests/test_sp8_tp4.py::test_sp8_tp4_fabric_tree_barrier_stability

   # Verify the serial SP8xTP4 reference, then the affine scheduler.
   scripts/run_safe_pytest.sh -q -s \
     models/experimental/kimi_delta_attention/tests/test_sp8_tp4.py::test_sp8_tp4_serial_layer_pcc
   scripts/run_safe_pytest.sh -q -s \
     models/experimental/kimi_delta_attention/tests/test_sp8_tp4.py::test_sp8_tp4_affine_layer_pcc

   # Capture/replay the complete affine layer and check its output before a
   # profiler report is accepted.
   KDA_SP8TP4_AFFINE_TRACE_TEST=1 scripts/run_safe_pytest.sh -q -s \
     models/experimental/kimi_delta_attention/tests/test_sp8_tp4.py::test_sp8_tp4_affine_trace_layer_pcc
   ```

4. **Capture the actual E2E SP2×TP4 layer on LB.**  Keep persistent state,
   sockets, barrier resources, and MRS intermediate buffers at fixed
   addresses.  Capture/replay the whole layer rather than separate prefix and
   scan traces, and profile the slowest device over ten warm replays.  The
   acceptance gate is <= **2.958 ms** (current 2.854 ms); the near-term
   stretch is <= **2.803 ms**, requiring about **51 us** from the current
   result.  Any change that misses the primary gate is reverted even if its
   isolated prefix number improves.

5. **Run the 32-device Galaxy bring-up as a protocol validation, then tune.**
   First validate one SP8×TP4 layer at global T=5120: PCC >= 0.98, no host
   state copies, no host stage fences, stable ten-replay trace, and a 1 MiB
   FP32 `(A,B)` payload per TP rank per prefix distance.  Only after that is
   stable should we tune overlap of final grouped scans, the final epilogue,
   and TP=4 MRS.  Galaxy performance targets must be set from its first
   slowest-device trace rather than extrapolated from the TP1 eager interval.

## E2E closure plan

The current SP2×TP4 layer is already an end-to-end performance control.  The
remaining work is to make the *same device-ordered protocol* trace-safe,
prove it on the only available eight-device system, and then qualify the
native 8×4 Galaxy topology.  No new implementation is accepted merely because
an isolated prefix becomes faster.

| Step | Deliverable and gate | LoudBox target / expected value |
| --- | --- | --- |
| 0. Recovery checkpoint | Resume hardware work only once a minimal guarded device test completes. Use `scripts/run_safe_pytest.sh`; do not issue another manual reset. | Prevents treating a PCIe-hung board as a protocol failure. |
| 1. Atomic-barrier qualification | Run 10 barrier-only rounds, then the 1 MiB/rank prefix PCC test, then the full SP8×TP1 two-call cache-reuse test. Start with one prefix lane. | PCC >= 0.98 end-to-end; zero runner recovery/timeouts. This is a stability gate, not a latency claim. |
| 2. Captured SP2×TP4 integration | Allocate the barrier, socket, recurrent-state and Line-MRS buffers once; capture the complete SP2×TP4 forward, including its normal TP=4 output CCL; replay it 10 times. | Must retain the current <= 2.958 ms slowest-device gate. Target is to preserve or beat the 2.854 ms control; <= 2.803 ms is the 51 us stretch. |
| 3. Trace diagnosis and one bounded optimization | Profile only a gate-passing trace. Attribute time to affine-prefix/barrier, carry sockets, grouped final scan, epilogue, and Line MRS; optimize the largest *unhidden* span only. | The plausible LB gain is 0--51 us: the new barrier mainly enables Galaxy ordering, so a larger LB claim would be unjustified. Revert any change that exceeds 2.958 ms. |
| 4. Galaxy functional bring-up | Run the serial SP8×TP4 reference, then the affine version with host fences, then the four TP-rank atomic trees. Verify output, recurrent state, convolution state and boundary tokens. | PCC >= 0.98; no host state copy/fence; 10 clean trace replays. This is the first native-topology proof. |
| 5. Galaxy e2e performance closure | Capture the whole SP8×TP4 layer, establish a slowest-device baseline, and tune prefix/scan overlap only after the trace is stable. | Record the baseline first. The deployment stretch goal is <= 1.339 ms, 1.25× the 1.071 ms TP4/T640 local-work control; failure means prefix traffic is not sufficiently hidden, not that correctness is incomplete. |
| 6. Operational hand-off | Add the measured command lines/report IDs, keep the serial path as a debug oracle, and document the topology/resource lifetime. | Reproducible correctness and performance evidence for every accepted mode. |

The immediate critical path is therefore **0 → 1 → 2**.  Steps 4--5 require
a healthy 32-device Galaxy; LoudBox can validate the exact resource-lifetime,
barrier, TP=4 CCL, and regression behavior but cannot claim native SP8×TP4
latency.

The native profiler lives at
`tests/perf/test_kda_sp8_tp4_layer_perf.py`.  Its trace path is deliberately
strict: it requires all SP ranks to use the fully device-queued affine
schedule, rather than quietly retaining host fences during capture.

```bash
# First establish the serial Galaxy control.
PERF_SEQ=5120 PERF_REPS=10 PERF_TRACE=1 PERF_SP8TP4_AFFINE=0 \
  scripts/run_safe_pytest.sh --profile -q -s \
  models/experimental/kimi_delta_attention/tests/perf/test_kda_sp8_tp4_layer_perf.py

# Then capture the complete affine SP8xTP4 layer.  Run only after the atomic
# barrier stability and PCC gates above are green.
PERF_SEQ=5120 PERF_REPS=10 PERF_TRACE=1 PERF_SP8TP4_AFFINE=1 \
  KDA_SP8_TRACE_SCHEDULE=1 KDA_SP_FABRIC_TREE_BARRIER=1 \
  KDA_SP8_PIPELINED_HANDOFFS=1 KDA_SP_PREFIX_LANES=1 \
  scripts/run_safe_pytest.sh --profile -q -s \
  models/experimental/kimi_delta_attention/tests/perf/test_kda_sp8_tp4_layer_perf.py
```

## Milestones

1. **Protocol reference and tests.** Add a partitioned PyTorch reference that
   exactly composes KDA spans.  It must show that each sequence boundary needs
   the incoming recurrent state plus the preceding three projected Q/K/V
   samples for the causal short convolution.
2. **LoudBox SP=2, TP=4 topology.** This is the primary end-to-end benchmark.
   Refactor logical TP from physical mesh size, preserve TP=4 output
   reduce-scatter inside each group, and transfer the per-TP-rank carry at the
   single SP boundary.  Use global T=1280, so every rank owns 8 heads x 640
   tokens: the same KDA work per device as Galaxy SP=8, TP=4 at global T=5120.
   Validate output and state PCC >= 0.98.
3. **LoudBox SP=8, TP=1 protocol probe.** Each chip owns one contiguous T/8
   span and all 32 heads.  It validates seven ordered boundaries at T=5120 but
   has four times the target head work and no TP collective, so it is a
   carry/scan correctness and latency probe rather than an end-to-end
   production-performance comparison.
4. **Galaxy-ready affine scan.** Replace serialized state relay by a
   log-depth scan over each span's affine summary `(A, B)`.  For K=V=128 and
   TP=4, a rank owns eight heads: `A` is 512 KiB and `B` is 512 KiB, so one
   uncompressed summary is 1 MiB.  Keep the actual recurrent state FP32; do
   not silently quantize it.
5. **Profiler gate.** Add/profile harnesses for TP=8, SP=8/TP=1 and
   SP=2/TP=4.  Record operation timing and transfer payloads before proceeding
   to Galaxy integration.

## LoudBox performance goals and decision gates

The real LB topology is `SP=2 x TP=4`.  `SP=8 x TP=4` requires 32 physical
devices; LB cannot execute its CCLs as a 32-rank mesh.  A TP=4 group at T=640
does, however, execute precisely the work of one production rank.  This gives
us a direct local-work control and SP=2 supplies the real cross-device
boundary.

* **Controls to record:** trace TP=4/SP=1 at T=1280 and TP=4/SP=1 at T=640,
  then trace SP=2/TP=4 at global T=1280.  All measurements use the same
  warmed trace harness and report the slowest device's layer critical path.
* **Primary LB speed gate:** SP=2/TP=4 at global T=1280 must be at least
  **1.5x faster** than TP=4/SP=1 at T=1280.  The stretch target is **1.7x**.
  This is an end-to-end layer comparison with the same model, global sequence,
  and output semantics; it is the closest full-system experiment LB can run.
  The first controls show that 1.5x requires <= **1.060 ms**, below the
  measured 1.071 ms cost of mandatory TP=4/T=640 local work before any
  cross-SP boundary. Treat it as an aspirational scaling comparison, not a
  presently attainable acceptance condition; the production-rank budget below
  is the actionable near-term gate.
* **Head-to-head LB topology gate:** at global T=5120, SP=2/TP=4 must beat
  the traced TP=8 control of 3.114 ms by at least **5%**, i.e. <= **2.958 ms**
  on the slowest device; the stretch target is 10%, <= **2.803 ms**.  Both
  layouts do 20,480 head-tokens per chip, so this is deliberately a stringent
  topology-and-CCL efficiency goal, not a claimed work-reduction speedup.
* **Production-rank budget:** SP=2/TP=4 at T=1280 must be no more than
  **1.25x** the latency of the TP=4/SP=1 T=640 local-work control.  The
  difference is the measurable cost of one real SP boundary plus scheduling;
  it prevents a nominal 2-way speedup hiding an excessive handoff tax.
* **Carry budget:** the 512 KiB FP32 recurrent carry plus 18 KiB BF16 short
  convolution carry, sent per TP rank, must consume <= **10%** of the local
  T=640 layer critical path.  The trace must show no host tensor conversion or
  host-mediated copy.
* **SP=8/TP=1 boundary probe:** seven-hop relay at T=5120 must preserve PCC
  >= 0.98 and expose per-hop latency.  It has no speedup target because its
  per-device head count differs from production.
* **SP=8 affine production-rank correctness gate:** with eight local heads,
  T=5120, and two prefix lanes, output, recurrent carry, and short-conv carry
  must all reach PCC >= 0.98.  This is explicitly a protocol gate, not a
  latency target: the current TP=1 implementation serializes several sends
  and omits the TP=4 output CCL.
* **Stop/adjust condition:** if the boundary cannot meet the 10% carry budget,
  do not proceed with a full 32-device integration; first fuse summary
  generation/consumption or reduce the transport representation with an
  accuracy study.

## Reproducible LoudBox gate commands

Run these only after the eight-device board is healthy.  Each profiler command
uses three warmed trace replays by default and brackets the measured region
with a distinct Tracy signpost, so the slowest-device critical path can be
read from the resulting report.

```bash
# Functional target gate: output, recurrent carry, short-conv carry, and the
# first output token after the SP boundary must all reach PCC >= 0.98.
KDA_SP_SPLIT_AFFINE=1 KDA_SP_SOCKET_LANES=2 KDA_SP_SOCKET_FIFO_BYTES=524288 KDA_SP_TARGET_SHAPE=1 scripts/run_safe_pytest.sh -svv models/experimental/kimi_delta_attention/tests/test_sp2_tp4.py

# SP=8/TP=1 protocol probe: seven real KDA boundary handoffs at global T=5120.
KDA_SP8_TARGET_SHAPE=1 KDA_SP8_TEST_SEQ=5120 scripts/run_safe_pytest.sh -svv models/experimental/kimi_delta_attention/tests/test_sp8_tp1.py

# SP=8 affine protocol: production TP4-rank state payload (8 heads) with the
# real 3-stage fabric prefix. This is a PCC gate, not a perf benchmark.
KDA_SP_PREFIX_LANES=2 KDA_SP8_AFFINE_TARGET_SHAPE=1 scripts/run_safe_pytest.sh -svv models/experimental/kimi_delta_attention/tests/test_sp8_tp1.py::test_sp8_tp1_affine_layer_pcc

# Device-queued SP8 affine prefix barrier: removes the host fences between
# the first two prefix distances. This remains a PCC/stability gate.
KDA_SP_DEVICE_BARRIER=1 KDA_SP_PREFIX_LANES=2 scripts/run_safe_pytest.sh -svv models/experimental/kimi_delta_attention/tests/test_sp8_affine_prefix.py::test_sp8_affine_prefix_production_tp4_payload
KDA_SP_DEVICE_BARRIER=1 KDA_SP_PREFIX_LANES=2 KDA_SP8_AFFINE_TARGET_SHAPE=1 scripts/run_safe_pytest.sh -svv models/experimental/kimi_delta_attention/tests/test_sp8_tp1.py::test_sp8_tp1_affine_layer_pcc

# Safe eager rank-release scheduler control and candidate. Compare the Tracy
# signpost intervals only; neither path is a Galaxy latency estimate.
PERF_SEQ=5120 PERF_REPS=3 PERF_RANK_RELEASE=0 scripts/run_safe_pytest.sh --profile -svv models/experimental/kimi_delta_attention/tests/perf/test_kda_sp8_rank_release_perf.py
PERF_SEQ=5120 PERF_REPS=3 PERF_RANK_RELEASE=1 scripts/run_safe_pytest.sh --profile -svv models/experimental/kimi_delta_attention/tests/perf/test_kda_sp8_rank_release_perf.py

# Local TP=4 controls: run both lengths with the same warmed trace procedure.
PERF_SEQ=640  PERF_TRACE=1 scripts/run_safe_pytest.sh --profile -svv models/experimental/kimi_delta_attention/tests/perf/test_kda_tp4_layer_perf.py
PERF_SEQ=1280 PERF_TRACE=1 scripts/run_safe_pytest.sh --profile -svv models/experimental/kimi_delta_attention/tests/perf/test_kda_tp4_layer_perf.py

# Primary SP experiment and the direct TP=8 topology comparison at global T=5120.
KDA_SP_SPLIT_AFFINE=1 KDA_SP_SOCKET_LANES=2 KDA_SP_SOCKET_FIFO_BYTES=524288 PERF_SEQ=1280 PERF_CHILD_TRACE=1 scripts/run_safe_pytest.sh --profile -svv models/experimental/kimi_delta_attention/tests/perf/test_kda_sp2_tp4_layer_perf.py
KDA_SP_SPLIT_AFFINE=1 KDA_SP_SOCKET_LANES=2 KDA_SP_SOCKET_FIFO_BYTES=524288 PERF_SEQ=5120 PERF_CHILD_TRACE=1 scripts/run_safe_pytest.sh --profile -svv models/experimental/kimi_delta_attention/tests/perf/test_kda_sp2_tp4_layer_perf.py
PERF_SEQ=5120 PERF_TRACE=1 scripts/run_safe_pytest.sh --profile -svv models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py
```

The final two reports are the head-to-head decision: SP=2/TP=4 must be no
slower than 2.958 ms on the slowest device to pass, against the existing
3.114 ms TP=8 control.

## Non-goals for this first slice

* No FP32 recurrent-state downgrade.
* No all-reduce used in place of the ordered causal carry.
* No unrelated TP=8 micro-tuning before the cross-device SP bottleneck is
  measured.
