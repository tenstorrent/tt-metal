# North-Mini optimized decoder

This directory records the single-device optimization of
`CohereLabs/North-Mini-Code-1.0` at revision
`d11e61a842617a22dc328552fa5bb86231ee4f37`. The implementation is
`tt/optimized_decoder.py`; it preserves the public prefill, decode, paged
KV-cache, trace, batch, and logical sequence-length contracts of the completed
functional decoder. This stage does not contain multichip, full-model, or vLLM
work.

## Selected implementation

- Decode keeps the residual stream width-sharded in L1 through sharded RMS
  norm, attention projections, residual additions, and dense MLP boundaries.
- Q/K/V is one packed same-input projection. All decode projections use
  DRAM-width-sharded weights and
  `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`.
- Batch 1 and batch 32 have separately materialized layout/program contracts.
  Both currently select 16-core QKV/output/dense projections, but cache update
  differs: fused K+V update wins at batch 1 and separate updates win at 32.
- Dense decode uses BFP8/LoFi attention weights and BFP4/LoFi gate, up, and
  down weights. Prefill retains BFP8/HiFi2 dense weights because BFP4 prefill
  missed the PCC bar at valid non-aligned lengths.
- MoE routing remains BF16/HiFi2, expert weights are BFP8/LoFi, and expert
  activations are BF16. Batch-1 expert projections use `ttnn.sparse_matmul`
  with separately tuned gate/up and down grids.
- Prefill uses DRAM-interleaved activations and explicit 8x8 matmul programs
  for small token matrices, switching to 10x10 programs at 1024 or more
  tokens. Public non-aligned `seq_len` values are padded/chunked internally;
  there is no divisibility restriction.
- Multi-user non-aligned prefill packs logical users onto one token axis for
  QKV and O matmuls, then restores independent user axes before RoPE, paged
  cache fill, and SDPA. This avoids the invalid batch-32/sequence-1 fused
  geometry and the Blackhole one-row tilize stall recorded in
  `AUTOTRIAGE.md`.
- Paged fill/update and paged SDPA remain TTNN composite operations. The
  framework-selected SDPA decode program is faster than the explicit
  candidate and is therefore the selected contract.

The implementation subclasses `FunctionalDecoder` only to reuse its public
validation and helper contract. It overrides weight materialization, both
forward methods, attention, normalization, dense MLP, and routed MoE. The
optimized-path audit fails if any measured method falls back to the functional
math implementation.

## Correctness and contracts

The final fast suite command is:

```bash
pytest -q -s \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py
```

Result: `20 passed in 125.08s`.

| Check | Final evidence |
|---|---:|
| real layer-0 selected decode / prefill policy | PCC 0.998001 / 0.998580 |
| real layer-1 / layer-4 selected decode policy | PCC 0.999311 / 0.999742 |
| active-expert layer 1 / layer 4 non-aligned prefill | PCC 0.998653 / 0.998560 |
| dense-expert serving prefill b32/s1 | PCC 0.998695 |
| dense-expert multi-user non-aligned prefill b2/s33 | PCC 0.998729 |
| traced dense batch 1 / 32 | PCC 0.998574 / 0.998507 |
| traced sliding-MoE batch 1 / 32 | PCC 0.999271 / 0.998374 |
| traced no-RoPE MoE batch 1 | PCC 0.999268 |
| repeated trace replay, 10 runs | deterministic; PCC 0.999398 |
| permuted paged cache, positions 5/17/31/63 | K/V PCC 0.999573–0.999702 |
| dense logical lengths 1/31/33/65 | PCC 0.999131–0.999138 |

The acceptance bar remains PCC `>= 0.995`. Reduced precision explains the
small delta from the functional real-layer-1 PCC 0.999798; the selected policy
remains comfortably above the bar for real weights and every representative
layer kind.

The cache remains BF16 with the functional page size and layout. An explicit
optimized batch-32 probe allocated the 32,768,000,000-byte cache at the
advertised 500,000-token context and replayed position 499,999 in 130.87 ms
with finite output:

```bash
python models/autoports/coherelabs_north_mini_code_1_0/tests/optimized_decoder_capacity.py \
  --mode decode --context 500000 --batch 32 --layer 0
```

`doc/context_contract.json` therefore remains unchanged: supported context is
500,000, serving batch is 32, and there is no capability reduction.

## Performance

Wall times below are warmed means over 10 samples at sequence 128. Decode is
complete-forward trace replay. Prefill and decode are measured at both the
primary batch 1 and serving batch 32.

| Layer kind | Phase | Functional b1 | Optimized b1 | Functional b32 | Optimized b32 |
|---|---|---:|---:|---:|---:|
| dense/full/forced-RoPE | prefill | 0.636 ms | 0.508 ms | 13.758 ms | 4.724 ms |
| dense/full/forced-RoPE | decode | 0.356 ms | 0.187 ms | 6.652 ms | 0.252 ms |
| sliding/RoPE/MoE | prefill | 14.908 ms | 4.979 ms | 147.182 ms | 141.145 ms |
| sliding/RoPE/MoE | decode | 9.528 ms | 0.791 ms | 11.122 ms | 3.400 ms |
| full/no-RoPE/MoE | prefill | 14.655 ms | 4.668 ms | 146.699 ms | 140.767 ms |
| full/no-RoPE/MoE | decode | 9.524 ms | 0.795 ms | 11.129 ms | 3.400 ms |

The final rerun reproduces primary batch-1 decode improvements of 1.90x for
dense and about 12x for both MoE kinds. Batch 32 improves by 26.37x for dense
and about 3.27x for MoE. No serving-batch regression is accepted. The exact
rerun distributions and cumulative policies are in
`candidates/final_verified_layer{0,1,4}_{prefill,decode}_b{1,32}.json`.

Batch 1 uses active-expert sparse matmuls. Batch 32 retains the device-resident
dense expert family after AutoFix exhausted three legal model-local
active-expert formulations; see `AUTOFIX.md`. This is an explicit TTNN
single-device output limitation, not a functional fallback.

## Operation-topology audit

| Area | Functional/current topology | Candidates | Selected action and evidence |
|---|---|---|---|
| Q/K/V | three same-input linears in functional path | packed QKV; split Q/K/V | packed QKV; avoids two weight-stream dispatches and is trace-safe |
| dense gate/up | repeated same-input projections | packed one-matmul + split; tuned separate projections | separate wins 0.2535/0.3188 ms vs packed 0.2628/0.3401 ms at b1/b32 |
| output projection | attention output required a layout restore under 32 cores | 16- and 32-core width shards | 16 cores removes the avoidable reshard and improves 0.1967→0.1919 ms b1 and 0.2598→0.2545 ms b32 |
| cache update | independent K/V writes | fused and separate paged update | fused b1 (0.2628 vs 0.2654 ms); separate b32 (0.3253 vs fused-family 0.3303 ms) |
| attention | paged SDPA composite with explicit program | explicit and TTNN-selected program | TTNN-selected: 0.1865/0.2521 ms vs explicit 0.1919/0.2545 ms |
| residual/norm | interleaved boundaries and reshards | coherent 8/12/16/32-core residual/norm/matmul chains | 16-core selected: 0.1885/0.2536 ms; coherent 8 is 0.1871/0.2801, output-12 is 0.1915/0.2559, and 32 is 0.2159/0.2813. Four required width-shard conversions remain at QKV-head, SDPA-concat, MLP activation, and residual-family boundaries; they total about 6.4 us and removing them regressed the whole layer. |
| dense MLP precision | BF16/BFP8 family | BFP4/LoFi gate/up/down crossed with final geometry | selected for decode after authentic layer-0 PCC 0.998001; all-BFP4 authentic decode/prefill fails at 0.990750/0.991947 |
| attention precision | BF16/BFP8/BFP4 and LoFi/HiFi2 | all-attention BFP4/LoFi | BFP8/LoFi retained; authentic all-BFP4 layer-0 decode PCC 0.990750 fails |
| dense prefill precision | reused BFP4 decode weights | phase-specific BFP8/HiFi2 weights | selected; BFP4 failed seq 31/33/65 PCC, BFP8 passes all |
| prefill programs | default/small fixed grid | 8x8 and 10x10 2-D programs; special fused b32/s1; per-user and token-packed non-aligned paths | 10x10 for M>=1024; token-packed non-aligned multi-user path selected. The special fused path watcher-asserted and the per-user one-row tilize stalled; both were adapted rather than rejected on their first error. |
| MoE router | reduced fidelity router | BF16/HiFi2 with BF16 or FP32 destination accumulation | BF16/HiFi2 with FP32 destination accumulation; the default accumulator changed authentic layer-4 top-k routing and failed active-expert correctness |
| MoE experts | BFP4 and BFP8, LoFi/HiFi2 | sparse gate/up/down grid and block sweeps, then authentic layers 1/4 | BFP8/LoFi retained. BFP4 is faster (b32 decode 2.214 vs 3.391 ms; prefill 139.965 vs 141.142 ms) and authentic PCC is 0.998206/0.999607, but the required layer-4 stress route fails at 0.982697; gate-only, down-only, and HiFi2 adaptations also fail the complete gate. |
| sparse grids | common grid for all expert projections | gate 8x3/6x4; down 8x8 and larger adapted grids | gate 8x3, down 8x8, blocks 16/12; 0.7966 vs 0.8557 ms b1 |
| MoE routing composites | top-k, sigmoid, scatter, row-major sparsity | scatter/sparse and dense expert families | all device-resident; intrinsic scatter tilize/untilize costs are recorded below |
| collectives | none; single-device stage | CCL/fused CCL candidates | not applicable in this stage |

## Profiler and accounting

Tracy collection and watcher collection were separate. `tt-perf-report` 1.2.8
was run with advice enabled and `--active-experts 8` for MoE rows.

| Profile | Ops | Device time | Wall time | Wall-device gap | Modeled DRAM roofline |
|---|---:|---:|---:|---:|---:|
| dense decode b1 | 27 | 166.697 us | 186.481 us | 19.784 us | 33.2% / 170 GB/s |
| dense decode b32 | 28 | 236.639 us | 252.084 us | 15.445 us | 23.4% / 120 GB/s |
| MoE decode b1 | 47 | 766.271 us | 792.036 us | 25.765 us | 14.6% / 75 GB/s |
| MoE decode b32 | 48 | 3295.267 us | 3329.682 us | 34.415 us | 39.1% / 200 GB/s |
| no-RoPE MoE decode b1 | 43 | 764.737 us | 794.651 us | 29.914 us | 14.6% / 75 GB/s |
| dense prefill b1 / b32 | 17 / 143 | 467.425 / 4117.478 us | 509.421 / 4660.260 us | 41.996 / 542.782 us | 18.8% / 12.9% |
| MoE prefill b1 / b32 | 33 / 227 | 4610.162 / 139808.997 us | 4731.353 / 140962.160 us | 121.191 / 1153.163 us | 38.3% / 16.0% |

The dense batch-1 compulsory stream is approximately 30.7 MB of selected
projection weights plus 0.26 MB of BF16 KV reads at sequence 128, giving a
rough 60 us 512-GB/s roofline. The measured 166.7 us device window is
consistent with the report's 33.2% modeled utilization and the cost of the
small composite operations. The MoE batch-1 compulsory stream is about
60.7 MB when eight experts are active, or roughly 119 us at peak bandwidth;
the many sparse/routing operations explain the lower 14.6% modeled result.

At batch 32, the dense profile reports 120 GB/s effective DRAM bandwidth
(23.4% of the 512-GB/s model) over 236.6 us. The selected dense-all-expert MoE
path streams approximately 604 MB of BFP8 expert projections plus attention,
router, and activation traffic; the measured 3295.3-us device window reports
200 GB/s (39.1%). Its weight-only lower bound is about 1.18 ms at peak, so the
remaining gap is consistent with batched expert matmuls and routing/composite
overhead rather than host dispatch. The wall/device gaps are only 15.4 and
34.4 us, respectively.

Advice identifies the three expert matmuls as the batch-1 MoE limit and the
batched expert matmuls as the batch-32 limit. Gate/up and down were therefore
tuned separately; larger legal `in0_block_w` values improved batch 1.
Batch-32 100-core expert matmuls beat 64 cores. Remaining report advice for a
larger output subblock is inapplicable to the sparse rows because
`per_core_M=per_core_N=1`.

There are no torch/from-torch/to-torch operations or host fallbacks in the
measured aligned prefill/decode methods. TTNN's `scatter` composite internally
emits device untilize/scatter/tilize operations: about 13.5 us in b1 decode,
24.9 us in b1 prefill, and about 120 us across the four b32 prefill chunks.
The sparse API requires a row-major sparsity mask, so this is intrinsic
composite work. The non-aligned multi-user compatibility branch has explicit
row-major/tile conversions solely to pack and restore user axes; normal
sequence-128 benchmarks do not enter that branch.

## Watcher and artifacts

The final watcher-only run used:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_LOGS_PATH=models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/watcher/final_rereview \
pytest -q -s \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py
```

Result: `20 passed in 134.01s`. The 2,170-line watcher log has no
fatal/assert/invalid-NOC/CB-bounds/overflow/sanitizer/timeout/hang/error
signature. All four p300c boards were healthy afterward, with DRAM status
true, temperatures 48.2–54.4 C, advancing heartbeats, and zero corrected or
uncorrected GDDR errors.

- `candidates/`: cumulative b1/b32 policy, layout, precision, and program
  sweeps.
- `tracy/selected/`: final raw ops CSV, advice-enabled filtered CSV, and
  summary CSV/PNG for representative prefill/decode paths.
- `watcher/final_rereview/generated/watcher/watcher.log`: final watcher
  evidence.
- `artifacts/optimized_correctness.{log,xml}` and
  `artifacts/optimized_watcher.{log,xml}`: machine-readable and full-text test
  evidence.
- `triage/` and `AUTOTRIAGE.md`: failed-candidate hang capture, root-cause
  ledger, reset evidence, and the token-packed model-side fix.
- `context500000_decode_b32.json`: optimized advertised-context evidence.
- `AUTODEBUG.md` / `AUTOFIX.md`: source diagnosis and isolated active-expert
  hypothesis experiments.
- `work_log.md`: command ledger, failures/adaptations, checklist, AutoFix, and
  independent-review record.
