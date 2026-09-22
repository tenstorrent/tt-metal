# Qwen3.8-27B fused decoder

Stage 2 implementation for `Qwen/Qwen3.8-27B`, checkpoint revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Completed: final correctness, context, profiler and watcher checks pass.
Independent [stage review](stage_review_2.md) returned **clean-pass**.

## Implementation and contract

`../../tt/fused_decoder.py` independently implements `FusedDecoder`; its runtime
does not call the functional decoder. The delivered tests patch functional
construction to raise if dispatch accidentally falls back to it.

The setup API is `from_state_dict(..., hf_config, layer_idx, mesh_device)` and
`allocate_state(batch_size, num_pages)`. `prefill_forward` accepts BF16 tiled
`[B,T,5120]` input and arbitrary valid logical T, including continuation.
`decode_forward` accepts one token per user. State, page table, absolute
positions and RoPE tensors belong to the caller; refresh the captured tensors
before replay and restore state after warmup/capture. See the runner for exact
allocation, replay and page-routing examples.

The supported context remains **262144**, with maximum prefill and last valid
decode tested for both linear attention (layer 0) and full attention (layer 3).
Full KV is BF16 with 32-token pages. Linear recurrence remains FP32. Only the
three-token BF16 convolution history changes to row-major, reducing its physical
storage from 655360 to 61440 bytes per user. No advertised capability was reduced.
The context and batch coverage points are recorded in `../context_contract.json`;
maximum context is tested at B1, while B32 is tested at length257.

Selected graph changes:

- Full attention: pack/reorder Q/K/V/gate weights; native prefill/decode head
  creation, partial RoPE and prefill head concatenation; keep decode head order;
  fuse K/V updates on disjoint core sets; fuse the sigmoid gate into multiply.
- Linear attention: pack QKV/Z/B/A, native causal convolution plus SiLU, flat
  delta scan with in-kernel Q/K normalization and head mapping, head-major scan
  output, and native gated RMSNorm. Neutral padded transitions preserve logical
  tails; row-major history avoids repeated conversions.
- Common MLP: separate gate/up projections with a real SiLU matmul epilogue on
  the gate projection. Packing both projections was measured and rejected.

Weights, activations and KV remain BF16; projections use HiFi4, FP32 destination
accumulation, exact math mode and L1 accumulation. Linear recurrence/native norm
intermediates are FP32. Common linear-attention RMSNorm calls explicitly use
the existing exact/FP32 compute config; the native default failed an added
changed-token case. Full attention retains its faster passing native default. The selected native convolution uses channel chunk256,
the scan uses chunk32, and public prefill internally uses chunks of at most128.
Batch groups of two fit 48 value heads on the 11x10 worker grid. Full B1 decode
uses block-sharded Q/K normalization and L1 RoPE intermediates; larger batches
use interleaved normalization to satisfy its native contract.

## Correctness and performance

Measured on one Blackhole device (device 3, 11x10 grid) through a 1x1 mesh.
The original environment, existing `python_env`, real weights and frozen
functional implementation are retained. No native source, dependency or build
was changed. Hardware jobs ran serially; watcher and profiler ran separately.

Batch1, prefill128; medians of five warmed prefill and 30 traced decode samples:

| Layer kind | Prefill before → after, ms | Decode before → after, ms | Latency reduction prefill / decode |
|---|---:|---:|---:|
| Linear attention | 4.84831 → 3.07446 | 3.35083 → 2.49408 | 36.6% / 25.6% |
| Full attention | 3.55431 → 2.75436 | 2.44381 → 2.26043 | 22.5% / 7.5% |

These are synchronized wall latencies, with state restoration and readback
outside timing. Every replay sample is checked bitwise. Final default-source
artifacts are `corrected_{linear,full}_benchmark.{json,log,source.sha256}`.
Historical `final_*`, `delivered_*` and `fixed_*` files predate the final scoped
normalization correction and are retained as candidate evidence.

| Layer kind | HF prefill PCC before → after | HF decode PCC before → after | Fused/functional PCC prefill / decode |
|---|---:|---:|---:|
| Linear | 0.99844807 → 0.99996030 | 0.99994612 → 0.99999452 | 0.99846750 / 0.99994248 |
| Full | 0.99892026 → 0.99893528 | 0.99824780 → 0.99824804 | 0.99996179 / 0.99997485 |

All listed PCCs exceed 0.995; no acceptance bar was lowered. AutoFix found that
the inherited default linear RMSNorm configuration fails an added B3,
length257 changed-token case. Its corrected stock-HF PCC is0.99988180 versus
0.99362123 before the fix and0.99300992 for the functional control. The corrected
output has PCC0.99181628 with that erroneous functional output: this material
delta is an accuracy repair, not universal bitwise/numerical equivalence.
`AUTOFIX.md` and `probe_batch3_equivalence.json` isolate and explain it.

Earlier faster linear candidates using the failed normalization policy are
excluded by the expanded gate. Under the corrected policy, three paired runs
compare the selected packed graph with the closest valid split-projection
alternate. Pooled90-sample medians are2.492811 versus2.492334 ms; the winner
reverses between runs, a0.019% difference without a consistent advantage.
The packed graph is retained. Full attention retains the best correct native
RoPE/head layout candidate; differences between its final repeats are timing
variation. The substantive improvements are the measured baseline reductions.

## Profiler conclusions

`tracy_final/{baseline,fused}_l{0,3}/` contains raw op CSVs and both prefill/decode
`*_perf_report.csv` plus readable `*.txt` tables from `tt-perf-report`.
`performance_final.json` preserves actual dtype/fidelity, movement and gap rows.

| Kind / mode | Device ops before → after | Sum of device kernels before → after, µs |
|---|---:|---:|
| Linear prefill | 88 → 32 | 4600.646 → 2865.165 |
| Linear decode | 100 → 38 | 3011.400 → 2466.776 |
| Full prefill | 52 → 29 | 3400.790 → 2590.584 |
| Full decode | 57 → 32 | 2375.970 → 2223.064 |

Kernel sums exclude gaps and are distinct from wall medians. All measured rows
are device operations; decode rows carry the captured trace ID. Runtime guards
reject Torch dispatch and host conversion inside measured forward/capture.
There is no measured host fallback, setup constant creation or dead projection.
The remaining conversions have concrete consumers:

- Linear projection must become row-major for the native convolution. Its
  untilize costs 40.116 µs in prefill and 4.791 µs in decode. Matmul-untilize
  fusion was adapted and tested; available executable variants fail PCC. A
  same-layout tiled-output control passes. See `work_log.md` for exact results.
- Two linear head-vector transposes (about2 µs each) and reshapes are required
  by the native scan adapter to map token-major beta/decay to per-head vectors
  (reshape: about10 µs each prefill, 6 µs each decode).
- Full decode retains two necessary reshard operations (0.371/0.366 µs) and
  a 4.114 µs head flatten. Native concat requires extra output movement and
  was slower after adapting its supported layout and padded batch contract.

`patterns.md` assesses every graph-fusing pattern, including inapplicable
patterns with architecture reasons. `candidate_metrics.csv`, snapshots under
`candidates/`, and experiment logs retain the successful, slower and invalid
options. No untested applicable rewrite remains in the assessed native graph.
New kernels, precision sweeps and subsequent pipeline stages are not part of
this stage.

## Reproduction and evidence

Run from the checkout root, on an exclusively available device:

```bash
bash models/autoports/qwen_qwen3_8_27b/tests/run_fusion_experiment.sh check_linear \
 --layer 0 --length 128 --benchmark
bash models/autoports/qwen_qwen3_8_27b/tests/run_fusion_experiment.sh check_full \
 --layer 3 --length 128 --benchmark
PYTHONPATH=. python_env/bin/python -m pytest -q \
 models/autoports/qwen_qwen3_8_27b/tests/test_fused_decoder.py
bash models/autoports/qwen_qwen3_8_27b/tests/profile_fused_decoder.sh
```

Add `--baseline` for the functional control, `--batch` for batch coverage, and
`--lengths 1,2,3,31,32,33,127,128,129,257,31 --continuation` for boundaries.
The wrappers pin checkpoint/environment paths to this run; adjust those setup
paths when reproducing elsewhere. Exact historical commands are in
`commands.log` and `work_log.md`, including watcher and full-context invocations.

- `verified_synthetic.{log,xml}` and `verified_synthetic_l*_b*.json`: three real-shape
  cases, 11 lengths each, continuation, deterministic replay, changed inputs and
  no functional fallback. Real-weight boundary/routing runs are also retained.
- `watcher_final_l{0,3}_b{1,32}.*` and corresponding `generated/watcher/watcher.log`:
  four clean real-weight 257-token continuation runs. `watcher_final_summary.json`
  records all paths/results. No hang, reset or recovery occurred.
- `verified_{linear,full}_context.{json,log}`: whole public requests at 4097,262143,262144,
  then31, with full-output fused/functional correlation and repeated trace.
  Conservative HF bounds reuse frozen functional-stage HF correlations and
  their artifact hashes; these are explicitly bounds, not freshly rerun HF.
  At maximum prefill the bounds are linear0.99623798/full0.99606463; the last
  valid full-attention decode bound is0.99743386. Direct HF tests cover shorter
  lengths separately.
- `stage_review_1.md`: initial independent review and required corrections.
  `AUTODEBUG.md` / `AUTOFIX.md`: additional batch-3 numerical investigation.

Raw Tracy captures and Torch control tensors remain local (`raw_artifacts.json`).
Raw op CSVs are losslessly included as `ops.csv.gz`; candidate sources are
archived in `candidates.zip` with per-file hashes in `candidate_sources.json`.
Extract the archive in this directory to restore historical candidate paths.
Raw execution/watcher logs are in `logs_execution.zip` / `logs_watcher.zip`;
`log_sources.json` records original hashes. Extract either archive in this
directory to restore the exact log paths. Compact report CSVs, tables and
metrics accompany the stage commit.
This evidence covers representative single-device decoder layers, not full
model accumulated accuracy, generation, serving, or every batch/context pair.
