# Fused-decoder work log

## Hardware and baseline

- Base commit: `c3cc345a10b`.
- `timeout 60 tt-smi -ls --local`: four Blackhole p300c boards visible.
- Bounded 1x1 mesh open/close: pass; no reset required.
- Functional ten-replay traced host medians (full b1/b32, linear b1/b32):
  2.450535 / 2.653672 / 3.140631 / 21.474161 ms.
- Fused ten-replay medians: 2.444631 / 2.650596 / 3.130992 /
  21.475945 ms. Because the last host median was noise-sized, the device
  profiler was decisive: 21.470382 → 21.457416 ms.

## Topology audit

| Region | Functional sequence and movement | Assessment |
|---|---|---|
| common norms/residuals | RMSNorm; mixer; add; RMSNorm; MLP; add, DRAM | RMSNorm already dedicated. Residual-add+RMSNorm cannot remove the add because the summed residual is also required by the following residual edge; rejected structurally. |
| MLP | two linears; SiLU; multiply; down linear, DRAM | SiLU-on-multiply retained; PCC unchanged. Two peers do not meet the ≥3 shared-LHS pattern. |
| full prefill QKV | three linears; reshape/permute; per-head RMSNorm; partial MRoPE; paged fill; SDPA; concat heads; sigmoid gate; output linear | Q+gate/K/V packed despite unequal widths; one projection plus exact slices is faster. SDPA, paged fill, and head concat already dedicated. Dedicated rotary cannot express partial rotary plus passthrough without slice/concat. |
| full decode | linears; fused create-QKV-heads; norms; partial MRoPE; paged updates; paged SDPA decode; sharded concat heads; gate; output linear | Existing dedicated decode ops retained. Two cache writes target distinct cache tensors and cannot merge. Required DRAM↔height-sharded conversions are exactly the contracts of create-heads/cache/SDPA/concat; no redundant reshard pair exists. |
| linear prefill | four projections; vector causal conv; dedicated activations; affine scan matmuls; RMSNorm; gated output; projection | QKV/Z/beta/decay packed into one projection plus exact slices and retained after PCC/perf wins. `prefix_scan` cannot express dense gated-delta affine state. Available conv ops do not expose the mutable depthwise causal state contract. |
| linear decode | four projections; stateful conv; normalize/update matmuls; RMSNorm; gated output; projection | Recurrent update is stateful dense rank-1 math with persistent FP32 cache; no matching dedicated op. L2 normalize has no equivalent TTNN dedicated last-dim op for these shapes. |

All graph-fusing categories were assessed: dedicated ops, structural rewrites,
and adjacent-op merging. Convolution, batchnorm, pooling, spatial mean, TopK,
MoE, and collectives are absent. Multi-chip CCL and cross-layer/full-model
patterns are intentionally outside Stage 02.

## Commands

Correctness:

```text
python .../full_attention_synthetic_pcc.py --mode decode
python .../full_attention_synthetic_pcc.py --mode prefill --sequence 33 --batch {1,32}
python .../linear_attention_synthetic_pcc.py --mode decode
python .../linear_attention_real_pcc.py
python .../linear_attention_synthetic_pcc.py --mode prefill --sequence 65
python .../linear_attention_synthetic_pcc.py --mode prefill --sequence 5 --batch 32
python .../full_attention_cache_pcc.py
```

Trace A/B:

```text
python .../traced_synthetic_pcc.py --decoder {functional,fused} \
  --kind {full,linear} --batch {1,32} --perf-iterations 10
```

Profiler and reports:

```text
python -m tracy -r -p -v -o <artifact-dir> <script> <args>
tt-perf-report <ops.csv> --start-signpost PERF_{DECODE,PREFILL} \
  --end-signpost PERF_{DECODE,PREFILL}_END --no-color \
  --csv <artifact-dir>/perf_report.csv --summary-file <artifact-dir>/perf_summary
```

Watcher was run separately from profiling:

```text
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_LOGS_PATH=<unique-dir> \
python .../traced_synthetic_pcc.py --decoder fused --kind {full,linear} \
  --batch 32 --perf-iterations 10
```

## Final gates

- Context contract unchanged: no fused layout/dtype/cache-capacity change.
- Runtime fallback audit: fused source has no Torch/from_torch/to_torch path.
- Non-aligned logical lengths: full seq33 and linear seq65 pass.
- Stress/determinism: ten trace replays per kind and batch; two numerical
  sequential steps validate mutable cache/state and changing positions.
- Watcher: both batch-32 fused runs clean.
- First independent review: `more-work-needed` on unmeasured shared-LHS packing.
  Remediation implemented both packing families; all final profiler rows improve.
- Fresh independent rereview: `clean-pass`, no required work; recorded in
  `stage_review.md`.
- Scoped local commit SHA: pending.

## 2026-07-30 — fresh review and AutoFix

A fresh independent review returned `more-work-needed` because the original
topology audit rejected two-way MLP gate/up packing from the illustrative
three-peer threshold without an experiment. `AUTODEBUG.md` confirmed this was
an exhaustion-evidence gap and specified the exact setup-time packed RHS,
17,408-wide slices, and SiLU input ordering.

The candidate replaced the two gate/up linears with one 34,816-wide linear and
two exact slices. Static fused-path tests passed, and full-attention traced PCC
was unchanged at batch 1 and 32. It was rejected on performance:

| Full-attention traced decode, batch 32 | us/replay |
|---|---:|
| retained fused winner | 2386.759 |
| packed MLP gate/up | 2388.755 |

The candidate profiler artifact is
`tracy/candidate_mlp_pack_full_b32/perf_report.csv`; the host median also
regressed from 2.559 to 2.572 ms. The candidate implementation was reverted,
so the final graph remains the faster prior packed-QKV/packed-linear-input/
SiLU-on-multiply path.

The trace regression now snapshots mutable KV or linear-attention state,
executes once, restores the identical state, executes again, and requires
bit-exact output equality. Final fused batch-32 checks pass for both layer
kinds (`FULL_TRACED_DETERMINISM exact=True` and
`LINEAR_TRACED_DETERMINISM exact=True`) while preserving sequential PCC,
row-distinction, fallback-hard-failure, and ten-replay stress coverage.

A new independent `$stage-review` then returned `clean-pass` with no required
work. It independently recomputed all eight final before/after profiler totals,
confirmed the rejected candidate CSV contains the adapted 34,816-wide MLP
matmul plus exact slices, verified the candidate was absent from final source,
and inspected the determinism restore/replay logic. The verdict is recorded in
`stage_review.md`.

Local checkpoints (never pushed):

- repo: `/home/mvasiljevic/tt-metal`
- branch: `skillexp-cell/fuse-noadvise/qwen`
- fused implementation and primary evidence:
  `b881fb0d60a8072097dc9b8df4cae34b4e1da077`
- AutoFix, rejected-candidate evidence, determinism regression, and clean
  rereview: `bae7875dc22b88ade5cab049170cb63ee9079f06`
