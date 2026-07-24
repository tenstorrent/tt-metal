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
- T=5,120 wall: `3681.529 -> 3385.003 us` (`-8.05%` versus the matched FP32
  DRAM control).
- T=5,120 recurrence: `1023.411 -> 807.181 us` (`-21.13%` versus control).
- T=5,120 selected block times: projection `524.798 us`, convolution
  `601.588 us`, decay transform `108.106 us`, recurrence `807.181 us`, and
  output/CCL `1069.975 us`.
- Estimated T=5,120 compute utilization: `13.14%`.
- Estimated T=5,120 CCL utilization: `39.79%` against the 40% aspiration.
- Final correctness gate: 27/27 tests passed; focused TP output/recurrent
  state/convolution PCC was `0.999965/0.999910/0.999997`.

The complete evidence is in `bringup_log.md`, `ROOFLINE.md`, and
`perf_report/codex-kda_perf_report.html`.

## Current ranked queue

1. Selectively double-buffer early scan inputs.
2. Fuse prep and scan into a bounded producer/consumer pipeline.
3. Distributed-L1 affine prefix scan.

The affine-prefix experiment proved the algebra and FP32 numerics, but rejected
a DRAM-backed implementation: its estimated memory floor exceeds the current
serial scan. It remains third only if transforms and tree levels stay in
distributed L1.

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
- Remaining path: fuse the programs only if a bounded rolling window can save
  additional launch/synchronization time beyond the retained L1 endpoint.

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

Potential reward and implementation order are different. The recommended
experimental sequence is:

1. Selective double buffering.
3. Bounded prep-to-scan producer/consumer fusion.
4. Distributed-L1 affine-prefix device prototype only if its storage and
   level-traffic model beats the measured serial scan.

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
38. Consume beta sigmoid/typecast inside prep without a separate launch.
39. Fuse decay projection, bias, Softplus, scaling, and FP32 conversion.
40. Feed decay output directly into prep.
    - Experiment result, 2026-07-24: full-sequence distributed-L1 gate
      residency preserved TP PCC and improved T=640 wall `0.41%` and recurrence
      `1.54%`, but both T=5,120 attempts produced no trace replay sessions
      (488 eager/setup rows, empty trace IDs). Reject the unbounded form; retain
      only a bounded rolling-window producer/consumer design.
41. Feed recurrence output directly into gated RMS.
42. Produce the exact output-projection input layout from gated RMS.
43. Remove remaining reshape/view programs that materialize data.

### E. Convolution

44. Replace the long-sequence untilize/tilize path with a tiled depthwise
    convolution reader.
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
49. Pipeline convolution slice reads, compute, and writes.
50. Produce tiled Q/K/V directly when that avoids a later layout conversion.

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
55. Evaluate lower-precision gated-RMS output only at the projection boundary.
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
