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
- Scoped local stage implementation/evidence commit created.

## Local checkpoint

- Repository: `/home/mvasiljevic/tt-metal`
- Branch: `skillexp-cell/fuse-advise/qwen`
- Stage implementation/evidence commit: `03d31d7bbc6`
- Push: not performed.
