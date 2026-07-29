# KDA optimization backlog

This document ranks the remaining Kimi Linear KDA optimization ideas after the
mixed-storage sweep. It distinguishes measured evidence from hypotheses and
keeps failed experiments visible so they are not repeated unchanged.

## Current measured endpoint

- Hardware: LoudBox, 8 Blackhole chips, TP=8, `FABRIC_1D`.
- Selected storage policy: mask `0x26`; `kd`, `q_decay`, and `dl` cross the
  prep-to-scan boundary in BF16. Arithmetic, `v_beta`, `k_dec_t`, `intra`,
  `t_inv`, recurrent state, and public outputs remain FP32.
- The transformed decay gate is stored in BF16 for tile-aligned chunk prefill;
  recurrent/decode gate storage remains FP32. Mixed-format prep arithmetic and
  recurrent state remain FP32.
- Prepared tensors now remain interleaved in distributed L1 between prep and
  scan. `QWEN_KDA_PREP_DRAM=1` preserves the measured DRAM control.
- T=640 wall: `643.623 -> 619.594 us` (`-3.73%`).
- T=5,120 retained wall: `3183.263 us` (10-replay median, slowest-device
  span). The recurrence is now a one-pass affine-summary builder, a
  receiver-owned distributed prefix, and an eight-chunk grouped final scan.
- T=5,120 selected recurrence times: affine summaries `134.994 us`, affine
  prefix `115.753 us`, and grouped final scan `121.325 us`.
- Estimated T=5,120 compute utilization: `14.15%` (including grouped-summary and prefix work).
- Estimated T=5,120 CCL utilization: `39.67%` against the 40% aspiration.
- Final correctness gate: 27/27 tests passed; focused TP output/recurrent
  state/convolution PCC was `0.999965/0.999910/0.999997`.

The complete evidence is in `docs/bringup_log.md`, `ROOFLINE.md`, and
`perf_report/codex-kda_perf_report.html`.

## Top three by maximum plausible reward

This is a gross-upside ranking, not the implementation order. Percentages use
the retained T=5,120 wall of `3385.003 us`; ceilings are not forecasts and
overlap, so they must not be added.

1. **Distributed-L1 two-level affine prefix scan.** The serial scan costs
   `500.685 us`, giving a `14.79%` whole-layer gross ceiling. This is the
   largest remaining single dependency-chain target, but also the highest-risk
   idea: a DRAM-backed version is already ruled out by its traffic floor.
2. **Bounded persistent prep-to-scan producer/consumer pipeline.** Prep costs
   `306.496 us` and scan costs `500.685 us`. Perfect overlap could hide at most
   the smaller phase, a `9.05%` wall ceiling. The retained distributed-L1
   endpoint proves residency is useful; it does not yet prove that ordered
   consumers can avoid starvation and backpressure.
3. **Direct tiled projection-to-convolution-to-Q/K/V program.** The former
   route spends `86.093 us` cropping QKV, `95.591 us` untilizing it, and about
   `94.533 us` on three post-convolution Q/K/V slices. Consuming projected
   tiles by offset and writing Q/K/V plus the three-row carry directly can
   remove up to `276.217 us`, an `8.16%` traffic/launch ceiling before custom
   kernel overhead. The row-major-input prototype proved correctness but not
   speed. The T=672 row oracle localized every error to rows 0-2 of each
   32-token block. Correcting padded stride was necessary but insufficient;
   alternate face ordering remained wrong, and full previous-tile staging with
   either separate or reused untilize CBs deadlocked. Do not retry partial-row
   tiled-prefix gathers. The viable lower-ceiling follow-up is the correct
   row-major custom kernel writing Q/K/V directly. That lower-ceiling path is
   now retained: it reduced matched wall `3380.644 -> 3334.239 us` (`-1.37%`)
   and calls `24 -> 21`. The remaining tiled-input ceiling is the pre-conv
   crop/untilize boundary only.

## Current execution queue

The receiver-owned affine prefix, L1 group entries, and owner-sharded summaries
are now the T>=5,120 default. The execution queue is exhausted at the `0.1%`
whole-layer stopping threshold.

- An empirical 83-prep/27-summary elastic service-rate experiment passed the
  non-profiled T=5,120 correctness gate, but reproducibly hung in two profiled
  warm replays, including a retry after kernel caching and device reset. The
  static work model bounds ideal overlap at about `37 us`, while the attempted
  multi-head summary CB protocol is not trace-safe. No candidate is retained.
- The remaining summary-to-prefix fusion can remove only the dispatch gap
  because A/B summaries are already owner-sharded in L1. Across 88
  device/replay observations, that gap is `0.539 us` median and `0.577 us`
  maximum, only `0.0169%` and `0.0181%` of the retained `3183.263 us` wall.
  This is below the `3.183 us` continuation threshold.

Direct tiled convolution is exhausted for this campaign. The corrected padded
stride plus canonical TL/TR/BL/BR bottom-row addresses reproduced output PCC
`0.951854` at T=672 while both states remained correct. A future retry requires
a standalone prefix-import primitive or a projection producer that exports the
three boundary rows; do not retry raw partial-row gathers.

The affine-prefix experiment proved the algebra and FP32 numerics, but rejected
a DRAM-backed implementation: its estimated memory floor exceeds the current
serial scan.

## Initial top three by potential reward

### 1. Associative affine prefix scan across chunks

Represent each chunk as an affine recurrent-state transform and compose those
transforms with a parallel prefix scan.

- Potential: attacks the fundamental serial chunk axis; could expose parallelism
  across cores and eventually across sequence-sharded chips.
- Evidence: the retained scan uses 16 workers that each walk 160 chunks at
  T=5,120, consuming `606.168 us`.
- Risk: very high. The transform must reproduce token-ordered KDA numerics,
  control intermediate growth, and preserve state PCC.
- First proof: implement a host/PyTorch affine-composition reference and prove
  equivalence against the existing recurrence before writing a device kernel.
- Experiment result, 2026-07-24: the proof passed. At K=V=128, C=32, and 160
  chunks, the FP32 balanced prefix matched the serial recurrence with PCC
  `1.000000000`, maximum absolute error `3.874302e-7`, and RMSE
  `5.075655e-8`.
- Cost verdict: reject a DRAM-backed prefix. Padding 160 chunks to 256 requires
  510 affine compositions, `4.278 GFLOP/head` (`17.11 GFLOP/chip`) and
  `80 MiB/chip` merely to store A+B for four heads. Approximately `780 MiB`
  of level traffic implies a DRAM floor of at least `1.5 ms`, already above
  the measured `606.168 us` serial scan before output reconstruction.
- Retained research path: distributed-L1 storage and execution only. Do not
  start a general DRAM implementation.
- Distributed-L1 model, 2026-07-24: reject storing every chunk prefix, but
  accept a three-phase grouped prototype. Use 27 groups/head (26 groups of six
  chunks and one group of four), hence 108 whole-V workers/device. Phase 1
  builds one affine summary per group while keeping its local accumulator in
  CBs; phase 2 scans only the 108 summaries; phase 3 runs the existing ordered
  recurrence for at most six chunks/core from the corrected group state.
- Work model: chunk-transform construction is 48 tile matmuls/chunk
  (`2.013 GFLOP/chip`); 532 local and at most 248 padded-tree summary
  compositions cost `6.543 GFLOP/chip`; the final grouped recurrence retains
  `2.349 GFLOP/chip`. Total is at most `10.905 GFLOP/chip`, with a
  `71.7 us` compute-peak floor. Beating the current `500.685 us` scan requires
  `21.78 TFLOP/s`, or `14.32%` of chip peak. That is only 67% of the retained
  scan active-core efficiency, so the compute model admits a win.
- Capacity model: selected prepared tensors occupy exactly `64 KiB/chunk`,
  or `40 MiB/chip`; 108 FP32 `(A,B)` summaries add `13.5 MiB/chip`. Their
  average distributed-L1 occupancy is about `498 KiB/core`. The full-V scan CB
  plan is `576 KiB/core`, leaving about `327 KiB/core` under the measured
  `1,434,496 B` budget. Full-prefix storage would add `80 MiB/chip` and exceed
  this budget once scan CBs are live, which is why only summaries are viable.
- Traffic model: local group compositions stay in CBs. Two reads of the
  `40 MiB` prepared set, summary writes, a padded summary scan bounded by
  `248 * 384 KiB`, and final output are roughly `200 MiB/chip` of
  distributed-L1/NOC traffic, requiring about `400 GB/s` to beat the serial
  scan. This is a hypothesis to validate with the device prototype, not a
  measured NOC result.
- Device prototype, 2026-07-24: reuse the validated recurrence itself to
  construct summaries. At target K=V, an eight-chunk scan from zero yields B;
  a second scan from block identity yields A+B; one subtraction yields A. This
  changes the map to 20 groups/head and 80 whole-V workers, avoiding a new
  transform-construction kernel.
- Correctness: a one-group T=256 test consumed the summaries as `A @ S0 + B`
  with nonzero S0 and passed output/state PCC `0.999992/0.999993`. All six
  direct KDA regression cases pass with the experiment disabled.
- Matched T=5,120 result: control wall `3380.644 us`; summary wall
  `3557.576 us`; overhead `176.932 us`. Call medians are `70.869 us` for the
  zero scan, `91.912 us` for the identity scan, and `21.706 us` for subtract,
  totaling `184.487 us`. The original serial scan remains `500.140 us`.
- Revised decision: retain the opt-in summary prototype and implement the
  summary-prefix phase. Reusing a `70.869 us` grouped scan after prefix leaves
  `244.784 us` for prefix at break-even, `241.403 us` for a gain above the
  `0.1%` noise floor, and `210.978 us` for the established 1% wall gate.

### 2. Direct prep-to-scan producer/consumer pipeline

Keep the 110-core chunk-parallel prep phase and 16-core ordered scan phase, but
send prepared chunks through L1/NOC queues rather than materializing every
intermediate in DRAM and reading it back.

- Potential: targets both sides of the `920.831 us` recurrence boundary and
  removes traffic that the successful BF16 experiment proved is expensive.
- Risk: high. Backpressure and synchronization must not serialize prep or
  starve scan.
- First proof: stream one prepared tensor for a bounded chunk window while the
  remaining tensors retain the current DRAM path.
- Experiment result, 2026-07-24: the lower-risk storage form of this idea won.
  Keeping every prepared tensor in interleaved distributed L1 across the two
  existing programs reduced T=640 wall by `3.27%`, T=5,120 wall by `5.64%`,
  and T=5,120 recurrence by `20.88%`. It passed the full 27-test gate.
- Fused-program result, 2026-07-24: reject the static 94-prep/16-scan
  partition. Exact per-chunk readiness bits preserved PCC at T=32 and T=672,
  but T=5,120 wall regressed `3380.644 -> 3421.029 us` (`+1.19%`). The
  fused recurrence measured `842.027 us`, worse than the control sequential
  prep plus scan medians of `306.028 + 500.140 = 806.168 us`. Static core
  partitioning lost more prep throughput than overlap recovered.
- Remaining path: revisit only with elastic core reassignment or a bounded
  rolling window whose producer cores can join scan after draining their work;
  do not repeat the disjoint static partition.
- Elastic follow-up, 2026-07-25: an 83-prep/27-summary implementation passed
  T=5,120 PCC `0.999958/0.999890/0.999997` without profiling, but hung in two
  profiled warm replays. The second run had 100% JIT cache hits, ruling out
  first-build latency. Reject the timing-sensitive multi-head CB protocol.

### 3. Remove duplicated scan reads per head

Each of four V workers for a head needs the same V-independent `kd`,
`q_decay`, `intra`, `k_dec_t`, `t_inv`, and `dl`. Replace independent DRAM
reads with static multicast or pre-positioned shared L1 data. Combine this with
dependency-ordered NOC barrier batching and selective double buffering.

- Potential: reduces repeated traffic in the `606.168 us` scan without changing
  KDA mathematics, TP distribution, or the ordered chunk loop.
- Risk: medium. A previous semaphore-based sharing design regressed
  `97.387 -> 145.942 us`; the new design must avoid runtime fan-out barriers.
- First proof: multicast one immutable tensor per head with compile-time
  destinations and no per-chunk semaphore handshake.
- Experiment result, 2026-07-24: a static `k_dec_t` sender with one batched
  ready/valid handshake per chunk was correct but regressed T=640 wall
  `622.564 -> 630.371 us` (`+1.25%`) and recurrence
  `144.945 -> 154.716 us` (`+6.74%`). Do not retain one-tensor sharing.
- Retained hypothesis: amortize the same one handshake across all six common
  tensors before rejecting static multicast as a class.
- Diagnostic result: row-aligned placement alone measured `621.223 us` wall
  and `145.140 us` recurrence versus `622.564/144.945 us`; it is neutral and
  cannot explain the one-tensor multicast regression.
- Batched result: sharing all six common inputs behind one handshake regressed
  T=640 wall `622.564 -> 657.290 us` (`+5.58%`) and recurrence
  `144.945 -> 181.216 us` (`+25.02%`). Reject runtime fan-out/multicast; do
  not repeat without a synchronization-free producer-consumer protocol.

## Execution order

No experiment remains above the `0.1%` whole-layer continuation threshold.
Future work requires a new architectural hypothesis or a materially different
producer/consumer protocol; the measured rejected designs must not be repeated
unchanged.

For local scan changes, retain an experiment only when:

- T=5,120 scan improves by at least 5%.
- T=5,120 wall improves by at least 1%.
- T=640 wall does not regress by more than 0.5%.
- Focused TP PCC is no worse than the selected endpoint.
- The 27-test final gate passes.

## Complete idea inventory

### A. Scan data movement and scheduling

1. Batch dependency-compatible scan reads behind fewer NOC barriers.
   - Experiment result, 2026-07-24: globally collapsing seven barriers into
     one regressed T=640 wall `622.564 -> 638.924 us` (`+2.63%`) and
     recurrence `144.945 -> 163.298 us` (`+12.66%`). It withholds all CBs
     until the slowest read and destroys reader/compute streaming. Reject.
2. Read tensors in compute-consumption order rather than declaration order.
   - Experiment result, 2026-07-24: T=640 improved wall `-0.71%` and
     recurrence `-2.04%`, but T=5,120 wall regressed `+0.28%` while recurrence
     improved only `-0.29%`. Reject because it does not generalize and misses
     both long-context gates.
3. Double-buffer BF16 `kd`, `q_decay`, and `dl`.
4. Extend double buffering to other inputs only after measuring L1 headroom.
   - Experiment result, 2026-07-24: doubling all seven streamed scan-input
     CBs passed target correctness, but T=640 improved only `0.35%` wall /
     `0.64%` recurrence and T=5,120 regressed `0.46%` wall / `0.38%`
     recurrence. The reader does not exploit the extra one-chunk capacity;
     added L1 occupancy and scheduling perturbation are net-negative. Reject
     both the narrow and extended variants: the all-input experiment is the
     upper bound on overlap opportunity.
5. Pack prepared inputs into one or two contiguous DRAM records per chunk.
6. Pipeline prep producers directly into scan consumers through L1/NOC.
7. Use a bounded rolling window of prepared chunks in L1.
8. Multicast V-independent inputs to the four V workers for each head.
9. Pre-position common head data in shared L1 with static addresses.
10. Tune V splitting specifically at T=5,120.
11. Split both K and V across scan workers and reduce partial results.
12. Fuse scan arithmetic stages to reduce intermediate CB traffic.
13. Retain intermediate tiles in DST registers across adjacent scan operations.
14. Overlap final-state writes with the last output tiles.

### B. Additional precision and storage

Retained experiment, 2026-07-24: store the transformed decay gate in BF16
for tile-aligned chunk prefill. Focused TP output/state/conv PCC was
`0.999965/0.999910/0.999997`; the full suite passed 27/27. T=640 wall
improved `622.564 -> 619.594 us` (`-0.48%`). T=5,120 wall improved
`3474.029 -> 3385.003 us` (`-2.56%`), while the decay block fell
`200.638 -> 108.106 us` (`-46.1%`) and recurrence improved
`809.704 -> 807.181 us` (`-0.31%`). Recurrent/decode remains FP32.

15. Store `k_dec_t` as scaled BF16.
16. Store `k_dec_t` as BF16 plus a compressed residual.
17. Use block-floating storage for state-sensitive prepared tensors.
18. Quantize only `k_dec_t` tiles whose measured dynamic range is safe.
19. Enable additional mixed storage only after an FP32 warm-up prefix.
20. Re-evaluate BF16 `v_beta` only for long-sequence execution.
21. Evaluate per-channel scaling for BF16 prepared tensors.
22. Evaluate stochastic rounding only if deterministic alternatives fail.

### C. Prep kernel

23. Split one 32-token prep item across two cores at T=640.
24. Use sequence-length-specific prep mappings.
25. Coalesce prep output writes.
26. Fuse BF16 packing into the final producing operations.
27. Reduce inverse scratch-CB traffic.
28. Build a specialized 32x32 unit-lower-triangular inverse/solve kernel.
29. Retune the existing exact-doubling inverse schedule.
30. Evaluate chunk size 64.
    - Experiment result, 2026-07-25: rejected before profiling. Ct=2 first
      exposed two latent producer-count defects: `k_dec_t` published Kt tiles
      while the writer consumed Kt*Ct, and the 32x32 inverse published one
      tile while the scan consumed Ct*Ct. Reusing the existing Ct=2 blocked
      solver eliminated the hangs, but a T=64 TP gate reached only
      output/state PCC `0.996686/0.929095` with external Q/K normalization.
      Generalized 64x64 exact doubling produced the identical result; restoring
      in-prep normalization regressed PCC to `0.949638/0.907320`. The retained
      Ct=1 T=5,120 path was fully restored and passed at
      `0.999958/0.999890/0.999997`. Do not profile or retain C=64 until a
      standalone Ct=2 numerical oracle localizes the state-update error.
31. Evaluate chunk size 16 as a control.
32. Pipeline prep reads, compute, and output writes across consecutive items.

### D. Layer-level memory and launch reduction

33. Remove or alias the post-MMRS clone.
    - Experiment result, 2026-07-24: an unsafe borrowed `rs` alias measured
      T=640 wall `622.564 -> 616.505 us` (`-0.97%`) and T=5,120
      `3474.029 -> 3446.590 us` (`-0.79%`). Returning cached `out_buf` under
      normal caller deallocation hung the trace and forced an 8-device reset.
      Do not remove the clone without an explicit pipeline-owned destination or
      borrowed-output lifetime contract; default Tensor refcounting is not safe.
34. Replace recurrent-state copies with buffer swapping.
35. Replace convolution-state copies with buffer swapping.
36. Fuse state updates into producing writers.
37. Eliminate Q/K/V and auxiliary slice programs with addressable projection
    groups or offset-aware consumers.
    - Fused crop/untilize control, 2026-07-24: `untilize_with_unpadding`
      removes the QKV slice and preserves exact TP PCC at T=32, but T=5,120
      trace replay deadlocks and forces an eight-device reset. Reject the
      existing composite for target trace execution; retain an offset-aware
      custom reader as the safe implementation path.
    - Native equal-width split control, 2026-07-24: target Kimi Q/K/V can use
      the single-pass tiled `ttnn.split` kernel, reducing three slice programs
      (`30.390 + 30.067 + 30.291 us`) to one `87.567 us` split. T=5,120 wall
      improves only `3385.003 -> 3381.006 us` (`-0.118%`) and the split block
      improves about `2.9%`; both miss the `1%` wall / `5%` block gates.
      Reject launch-only consolidation; data movement must also be eliminated.
38. Consume beta sigmoid/typecast inside prep without a separate launch.
39. Fuse decay projection, bias, Softplus, scaling, and FP32 conversion.
40. Feed decay output directly into prep.
    - Experiment result, 2026-07-24: full-sequence distributed-L1 gate
      residency preserved TP PCC and improved T=640 wall `0.41%` and recurrence
      `1.54%`, but both T=5,120 attempts produced no trace replay sessions
      (488 eager/setup rows, empty trace IDs). Reject the unbounded form; retain
      only a bounded rolling-window producer/consumer design.
41. Feed recurrence output directly into gated RMS.
    - Retained result, 2026-07-24: a single device program keeps the 16 scan
      producers and assigns the other 94 Tensix cores as gated-RMS consumers.
      Each `(head, chunk)` consumer waits for all four V-block producers, then
      reads the completed FP32 scan row and writes the token-major gated output.
      Both T=672 and T=5,120 preserve output/state/convolution PCC at
      `0.999958/0.999890+/0.999997`. Ten T=5,120 replay sessions improve median
      wall `3330.328 -> 3265.240 us` (`-65.088 us`, `-1.954%`). The combined
      scan program measures `514.395 us` and hides about 78% of the former
      `83.557 us` standalone RMS stage. Retain the bounded pipeline.
    - Direct-L1 follow-up, 2026-07-24: CB0 spans all 110 cores and gives each
      RMS consumer up to seven fixed full-V staging slots (`112 KiB/core` at
      T=5,120). Producers write their unique slot, flush, then signal; consumers
      publish the same slots into the local CB in cyclic order. T=672 and
      T=5,120 preserve PCC. Two 10-replay candidate medians were `3259.237` and
      `3259.763 us`; a freshly rebuilt DRAM-handoff control was `3264.154 us`.
      The replicated candidate center is `3259.500 us`, a matched `4.655 us`
      (`0.143%`) wall gain; scan improves `514.549 -> 512.951 us`. Retain, but
      classify this as a threshold-level micro-optimization rather than a new
      large roofline shift.
42. Produce the exact output-projection input layout from gated RMS.
43. Remove remaining reshape/view programs that materialize data.

### E. Convolution

44. Replace the long-sequence untilize/tilize path with a tiled depthwise
    convolution reader.
    - Reuse control, 2026-07-24: the existing generic tiled four-tap FIR is
      correct (`0.999959/0.999913/0.999997` PCC) but regresses T=5,120 wall
      `3385.003 -> 4837.313 us` (`+42.9%`). It expands convolution from 15 to
      23 calls and repeats untilize/slice/tilize/ternary work per tap. Retain
      the custom single-program reader/compute/SiLU design; do not compose it
      from generic TTNN operations.
    - Custom-program result, 2026-07-24: a 110-core 32-token-block reader,
      four-tap FPU depthwise compute, row-broadcast tap weights, and fused SiLU
      passed direct identity PCC `0.999998` and full TP output/state/conv PCC
      `0.999955/0.999905/0.999997`. At T=5,120 it reduced calls `35 -> 24`,
      but convolution active time regressed `606.271 -> 609.761 us` and wall
      regressed `3385.003 -> 3390.084 us` (`+0.150%`). The custom kernel costs
      `426.786 us`; retained QKV crop plus external untilize still cost
      `181.952 us`. Reject this boundary. The next viable version must consume
      tiled projection output directly and eliminate those two prep programs.
    - Tiled-input correction, 2026-07-24: withdraw the apparent T=5,120
      `3385.003 -> 3204.486 us` speedup. The only PCC control used T=32 and
      bypassed the `T > 640` route. A new T=672 gate measured output/state/conv
      PCC `0.048428/-0.001323/-0.005458`. Native control passed
      `0.999967/0.999920/0.999997`. The first proven bug was floor-dividing the
      non-tile-aligned projection width (`609/32=19` instead of 20 tiles; target
      `2180/32=68` instead of 69). Ceiling division repaired state PCC but left
      output PCC `0.951851`, localizing a second error to multi-block prefix
      handling. Reject and revert the custom path; do not cite its latency as a
      valid endpoint.
45. Reduce T=5,120 DRAM slicing overhead.
46. Implement numerically exact SiLU in the convolution output kernel.
    - Experiment result, 2026-07-24: `Conv1dConfig.activation=SILU` is ignored
      by the native depthwise Conv1d path. Removing the standalone unary reduced
      output PCC to `0.884282`; restoring it while leaving the epilogue configured
      restored the exact retained `0.999965/0.999910/0.999997` PCC tuple. A
      custom writer/compute epilogue remains viable, but the public config knob
      cannot eliminate the `106 us` unary pass.
47. Update convolution carry inside the convolution writer.
48. Tune DRAM slice width and count explicitly.
    - Experiment result, 2026-07-24: reject finer slicing. Auto selects two
      slices and measures `3385.003 us` wall with 35 calls per replay. Four
      slices measured `3467.099 us` (`+2.43%`, 45 calls); eight measured
      `3712.531 us` (`+9.68%`, 65 calls). Five extra device ops per added
      slice make orchestration cost dominate; retain auto/two-slice routing.
49. Pipeline convolution slice reads, compute, and writes.
    - Activation-reuse control, 2026-07-24: not applicable to this geometry.
      The kernel requires activation block height greater than output width;
      KDA has `1 vs 1` tiles at T=32 and `2 vs 160` at T=5,120. Both fail
      validation before execution. Retain ordinary slice streaming.
    - Forced split-reader result, 2026-07-24: convolution active time improved
      only `606.272 -> 603.237 us` (`-0.50%`) while wall regressed
      `3385.003 -> 3387.903 us` (`+0.09%`). Reject the override; the native
      viability model correctly prefers overlapping the single activation
      reader with weight transfer.
    - Activation double-buffer result, 2026-07-24: reject for capacity. Even
      160 width slices cannot fit doubled activation CBs in the available
      `1,434,496 B` L1 budget, so auto slicing fails before execution.
50. Produce tiled Q/K/V directly when that avoids a later layout conversion.
    - Retained result, 2026-07-24: the correct row-major custom convolution now
      writes three tiled Q/K/V outputs directly. At T=5,120 it reduced
      calls/device/replay `24 -> 21` and matched slowest-device wall
      `3380.644 -> 3334.239 us` (`-1.37%`). The custom op median was
      `413.165 us`; replay spans were `3330.362-3340.451 us`. Full TP PCC at
      T=5,120 was output/state/conv
      `0.999958/0.999890/0.999997`. Retain this boundary.

### F. Projection and epilogue

51. Retune the fused input projection at T=5,120.
52. Evaluate BF8 projection weights with BF16 activations and FP32 accumulation.
    - Input-projection result, 2026-07-24: established `bfloat8_b` weight
      storage passed the test suite threshold but failed the selected endpoint
      guard: output PCC `0.999964 -> 0.999867`, recurrent state
      `0.999903 -> 0.999878`. Reject BF8 for the fused input projection; the
      output projection remains an independent experiment.
    - Output-projection result, 2026-07-24: recurrent PCC stayed unchanged
      and output PCC remained above `0.999900` at `0.999938`, but T=5,120
      wall regressed `0.17%` and output/CCL regressed `0.11%`. Reject: the
      fused MMRS is CCL-bound, so compressed weights do not improve it.
53. Fold additional auxiliary projection work into the grouped projection.
54. Tune gated-RMS core mapping separately for short and long sequences.
    - Retained result, 2026-07-24: choose the fewest cores that preserve the
      all-core maximum work items per worker. At T=5,120 this maps 640 items to
      107 rather than 110 cores while retaining six items/worker. Gated-RMS
      kernel time improved `86.339 -> 83.557 us`; matched wall improved
      `3334.239 -> 3330.328 us` (`-0.117%`).
55. Evaluate lower-precision gated-RMS output only at the projection boundary.
    - Experiment result, 2026-07-24: reject the available mixed boundary. A
      configurable BF16 gated-RMS output paired with FP32 persistent MMRS
      buffers failed T=5,120 output PCC at `-0.000049`; recurrent and
      convolution states remained `0.999890/0.999997`, localizing the failure
      after recurrence. The fused MMRS op derives its matmul output format from
      the BF16 input but writes into caller-provided FP32 persistent buffers, so
      the formats alias rather than provide BF16-input/FP32-partial arithmetic.
      A typecast restores compatibility but also restores the eliminated pass.
      Reject until MMRS exposes an independent accumulation/buffer dtype.
56. Fuse gated-RMS output packing with output-projection input staging.

### G. Output projection and CCL

CCL is lower priority because its estimated utilization is already `39.79%`
against the 40% aspiration.

57. Remove the output clone. See item 33: measured ceiling `0.79%` at
    T=5,120, but both implicit ownership strategies are unsafe.
58. Overlap one layer's fused MMRS with independent work from another layer.
59. Reserve CCL cores across layer boundaries rather than within one layer.
60. Stream output tiles into reduce-scatter earlier.
61. Evaluate scaled BF16 or block-floating communication partials.
62. Use FP32 local accumulation with compensated lower-precision communication.
63. Compress only tiles whose dynamic range satisfies an explicit error bound.

### H. Distribution and architectural changes

64. Compose chunk transforms with an associative parallel-prefix scan.
65. Use affine prefix composition to enable sequence parallelism across chips.
66. Implement a two-level scan: local chunk groups plus prefix correction.
67. Explore hierarchical head/V/K distribution.
68. Keep scan state resident in a persistent kernel across invocations.
69. Fuse prep and scan into a persistent program with separate core groups.
70. Pipeline recurrence work across adjacent model layers.
71. Pipeline different sequence chunks across convolution, prep, scan, and
    output stages.

## Measured ideas already retained

- Trace capture and persistent constants.
- Native depthwise Conv1d.
- Head-major recurrence output and flat output gate.
- Exact doubling inverse.
- Adaptive four-way V splitting at local head count four.
- Fused input projection.
- Row-major convolution flow and prepared-weight caching.
- Offline `g_b @ g_a` precomposition.
- Fused gated RMS epilogue.
- Fused Softplus-scale multiply.
- DRAM width-sliced long-sequence Conv1d.
- BF16 storage for `kd`, `q_decay`, and `dl`.

## Do not repeat unchanged

- Plain BF16 `v_beta`: wall regressed `0.18%`.
- Plain BF16 `k_dec_t`: faster, but recurrent-state PCC fell to `0.999899`,
  below the selected `0.999900` guard.
- Semaphore-based common-input sharing: `97.387 -> 145.942 us`.
- Smaller output matmul grids: compute loss exceeded CCL benefit.
- Extra CCL workers that consume matmul rows: slower end to end.
- `1x3` output subblock: `0.63%` slower.
- Plain BF16 MMRS partials: PCC `0.004862`.
- HiFi2/LoFi recurrence: slower or below the required PCC.
- Fast TF32 row reducer: `3.17%` slower.
- Removing the prep-reader startup burst: `9.37%` slower.
- Generic convolution-packer SiLU: PCC `0.884267`.
- Existing beta sigmoid/typecast fusion: introduced an approximately `31 us`
  serialized scheduling gap.
- Ordinary sequence partitioning with ordered state handoff: architecturally
  serializes the recurrence boundary.

Rejected ideas may be revisited only when the new mechanism directly addresses
the measured failure mode.

- Device evidence for the prefix step: the generic five-stage Hillis-Steele oracle is correct at two groups/head (`0.999991/0.999993` output/state PCC), but its 62 prefix/correction ops cost `3120.528 us` and produce `6275.149 us` T=5120 wall. General matmul/slice/concat is rejected. The next implementation is one persistent five-stage program with core-local A/B ping-pong; its unchanged acceptance bound is `<210.978 us` for a 1% whole-layer win.
