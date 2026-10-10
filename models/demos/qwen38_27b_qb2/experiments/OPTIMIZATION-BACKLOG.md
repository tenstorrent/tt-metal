# Remaining optimization opportunities, Oct 9 2026

**Historical snapshot, superseded October 10.** Use the
[operator roadmap](OPERATOR-ROADMAP.md) for current priorities and results.
B16/32K/TP4 BFP8 compact decode has since reached **20.035 TSU**, with
matched full-model controls and full GPQA **177/198 (89.39%)**.
The current target is **30 native TSU**, without speculative decoding. The
unqualified/queued statuses and performance values below describe October 9,
not the present deployment or experiment queue.

Priority: 32K ISL, then 16K, with active 128K/256K checks and explicit tradeoffs.
Optimize total committed output throughput. Historical optimized BFP4 32K/B32 point is
373.70 output tok/s per TP4 after shared Q/K (+6.64% over the previous single-step
path), with identical generated tokens in the full-model comparison. Its 2989.6
output tok/s Galaxy projection is not a physical measurement.

Corrected native BFP8 GPQA finished **176/198 (88.89%), zero truncations** and
was accepted by the user. The original 177/198 gate remains missed in its raw
receipt. Current native BFP8 32K B16/B32 is 11.749/7.354 TSU; reaching 20 TSU
requires 50-ms steps versus 85.11/135.99 ms measured. A persistent matched
BFP8 native/shared-QK/native queue now follows container qualification, then
profiles B16/B32 and runs fresh G0/full GPQA. No optimized BFP8 speed or accuracy
result is claimed yet. [Launch and bounds](../galaxy-evidence/bfp8-gdn-followup-v1/README.md).

The optional 64K prefill-budget sweep hit a clean DRAM allocator limit at B32.
Container and BFP8 followers have been restored in fresh v3/v2 units after an
audited release; the working budget stays 32K.
[Recovery](../galaxy-evidence/capacity-recovery-v1/README.md).
The standalone fused GDN output-layout/norm/z epilogue now matches native output
bit-for-bit in all nine B1/B16/B32 simulator cases. Hardware timing and model
integration remain untested; no speedup is credited.
[Numerical evidence](../galaxy-evidence/gdn-epilogue-simulator-v1/README.md).

The completed physical HTTP sweep exposes a separate serving bottleneck: long
full-prefill steps repeatedly interrupt decode. At 32K/C128, median client
stream speed is 2.705 tok/s/user and whole-burst output is 172.26 tok/s including
prefill. A plugin continuation-slot prerequisite is fixed with a failing-before
reproduction. Full-model state/position testing passed; actual scheduler and
device-sampling qualification must follow before enabling shorter chunks. This
is scheduling between separate programs, not the mixed-row compute/memory
overlap described later. Any throughput/TTFT tradeoff must be measured.

| Priority | Work still open | Evidence and next useful test | Tradeoff or limit |
|---|---|---|---|
| 1 | Bank-local bulk KV reading with compute-worker delivery | Isolated read probe reaches 499-508 GB/s/chip; actual attention delivers roughly 70-75% of assumed peak as useful KV bytes/time. All 54 delivery/backpressure variants passed, and larger packets with opposite receiver placement improved delivery to 275-280 GB/s/chip. Depth four/eight are nearly equal. Production page-table traversal and attention integration remain open. | Raw-read bandwidth excludes redistribution and attention math. Delivered bandwidth is currently below useful attention bandwidth; do not integrate this mover as a performance improvement yet. Gains should matter more at 128K/256K. Placement alone did not help tile-at-a-time reads. |
| 2 | GDN output epilogue and layout fusion | Shared Q/K is complete at component and full-model boundaries; gated RMSNorm and SiLU(z), output layouts and preparation remain separate. Fuse while preserving FP32 recurrence and existing numerical gates. | Extra compute, L1 pressure and changed reduction order require long-horizon/real-weight checks. The public/native op and P1 latency gate are still open. |
| 2 | Convolution, RoPE and decoder graph cleanup | Pre-shared profile at B32: packed convolution about 150 us per GDN layer, RoPE about 109 us per attention layer; tilize/reshape traffic is material. Keep compatible layouts in L1, remove redundant conversions and handle small/irregular convolution batches. | These stage sums are not all removable latency. Must count launches/programs properly; <=15 programs/layer is not achieved. Stronger relative benefit at 16K/32K. |
| 3 | Expose tested B32 efficiently in serving; implement B64 projections | Fixed-shape B32 native model runs; resident serving buckets still 1/8/16. B64 is blocked by the fast DRAM-sharded projection's one-tile-row limit. Extend token-row handling, projection kernels and trace/state buckets. | Larger batches trade per-user latency and KV/workspace capacity for aggregate throughput. No measured B64 full-model speedup or capacity claim. |
| 4 | Weight matmul placement, L1 retention and collectives | MLP/head measurements still leave headroom; head is about 0.5 ms/step and the overall model includes repeated collectives/conversions. Profile representative projection widths, bank/core placement and communication schedules. | Lower priority than GDN/layout/KV at long context; do not apply a peak-bandwidth multiplier to the whole model. TP8 costs remain uncalibrated. |
| 5 | Prefill efficiency | Current 32K/B32 prefill is about 5.2K input tok/s/TP4. Tune chunk geometry, batched matmuls and collective overlap; P4 target 20K input tok/s/TP4 and 40% MFU remains open. | Primarily improves TTFT and mixed-workload capacity, not isolated steady-state decode output tok/s. Larger chunks previously caused workspace OOM. |
| Later | Mixed prefill/decode execution | Requires mixed-row matmuls and scheduling; current stages run separately. | Potential aggregate-serving benefit, with substantial scheduling/kernel work; cannot count modeled overlap as achieved. |
| Conditional | Fused Blaze decoder layer | P3 remains unimplemented. Decide after graph cleanup and measured residual overhead. | Larger implementation/qualification cost; the >30% overhead trigger is not established. |

## Small wins and deferred experiments

- Partial-query attention passed physical output equality. At 32K/B32 its
  attention-call gain was only about 0.85%, so it is a small model-level gain.
  At some longer-context geometries it regressed slightly; use shape-specific
  selection only after a matched full-model check.
- Larger attention reader barriers were slower and are rejected. Increasing
  bank-local read buffering from four to eight slots gave no useful gain.
- Speculative decoding stays opt-in. Promotion requires higher committed output
  throughput at matched offered concurrency, including every draft/verify/state
  commit/sampling cost. Existing 32-row projections otherwise reduce active
  users per verify pass; a low-concurrency latency win is not sufficient.
- BFP4 KV is unqualified: synthetic K4/V4 attention-output error was about 16%
  versus 1-2% for BFP8. BF16 recurrent state is also unqualified. Bandwidth and
  fusion work preserve precision first.

No remaining item has a proven end-to-end uplift until its own matched test.
Do not add projected gains together. Reference evals, long-context eight-replica
scaling and a calibrated stage roofline remain gates, not optimization credit.

Evidence: [full-model comparisons](../galaxy-evidence/shared-qk-full-model-v1/README.md),
[bandwidth and profiles](../galaxy-evidence/shared-qk-and-bandwidth-v1/README.md),
[speculative scope](SPECULATIVE-DECODE-SCOPE.md), [plan status](PLAN-STATUS.md).
