# Optimized multichip decoder work log

Repo revision: `31b45719e2ca21b695a8e7f15b5e8895bc1fb3bb`
Date: 2026-06-09 UTC
Target: `google/gemma-4-12B`

Scope: optimize the completed `MultichipDecoder` TP8 path in
`tt/multichip_decoder.py`. Full-model and vLLM work were intentionally not
started.

## Starting point

The completed multichip decoder already used the target mesh path:

- `MeshShape(1, 8)` on Wormhole T3K.
- TP8 on mesh axis 1 with `FABRIC_1D_RING` and ring topology.
- Replicated residual stream between decoder layers.
- Decode activations width-sharded in L1.
- Prefill activations DRAM interleaved.
- Column-parallel QKV/gate/up, local SDPA/GeGLU, row-parallel O/down, ring
  all-reduce after each row-parallel projection.
- Paged per-device KV cache, with full-attention KV replicated because the
  model has only one full-attention KV head.
- Dense model, no MoE path.

## Baseline artifacts

Baseline multichip performance and PCC came from
`models/autoports/google/gemma-4-12B/doc/multichip_decoder/`. The accepted
optimized artifacts copied the baseline profiles to:

```bash
models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/tracy/sliding
models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/tracy/full
models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/tracy/baseline_multichip_perf_summary.json
```

Final accepted PCC evidence:

```bash
tail -n 16 models/autoports/google/gemma-4-12B/doc/multichip_decoder/pcc_results.jsonl | head -n 8 > \
  models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/pcc_results.jsonl
```

## Correctness commands

Syntax:

```bash
python -m py_compile models/autoports/google/gemma-4-12B/tt/multichip_decoder.py
```

Result: passed.

Full correctness, cache, layout, and trace replay:

```bash
pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_runtime_fallback_audit_source_clean \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_paged_prefill_then_decode_pcc_vs_optimized \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_long_context_paged_prefill_decode_vs_optimized \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_cache_and_stacked_layout_contract \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_decode_trace_replay_pcc_and_determinism_vs_optimized \
  --tb=short --timeout=900
```

Result: `9 passed, 3 warnings in 87.47s`.

Repeated short PCC coverage:

```bash
for i in 1 2 3; do
  pytest -q \
    models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_paged_prefill_then_decode_pcc_vs_optimized \
    --tb=short --timeout=600
done
```

Results:

| Run | Result |
| --- | --- |
| 1 | `2 passed, 3 warnings in 25.99s` |
| 2 | `2 passed, 3 warnings in 25.07s` |
| 3 | `2 passed, 3 warnings in 24.31s` |

Runtime fallback source audit:

```bash
grep -n -E "ttnn\.from_torch|ttnn\.to_torch|FunctionalDecoder|prefill_l1_inputs" \
  models/autoports/google/gemma-4-12B/tt/multichip_decoder.py
```

Result: no matches.

## PCC table

| Layer | Seq | Prefill PCC | Prefill bar | Decode PCC | Decode bar | Replica PCCs |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Sliding | 128 | 0.9997969662 | 0.995 | 0.9964325808 | 0.993 | all 1.0 |
| Full | 128 | 0.9996992886 | 0.995 | 0.9983275303 | 0.995 | all 1.0 |
| Sliding | 1024 | 0.9994995075 | 0.995 | 0.9993016740 | 0.992 | all 1.0 |
| Full | 1024 | 0.9994667300 | 0.995 | 0.9924891310 | 0.992 | all 1.0 |

Trace replay:

| Layer | Replay PCC | Replay bar | Determinism PCC | Determinism bar |
| --- | ---: | ---: | ---: | ---: |
| Sliding | 0.9964325808 | 0.993 | 1.0 | 0.9999 |
| Full | 0.9983275303 | 0.995 | 1.0 | 0.9999 |

## Performance commands

Accepted profiles are the completed multichip decoder profile, copied into this
stage after all optimization trials were rejected. The `tt-perf-report` files
were generated with advice enabled. Human-readable tables:

```bash
tt-perf-report models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/tracy/sliding/ops.csv \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --no-summary \
  > models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/tracy/sliding/prefill_perf_report.txt

tt-perf-report models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/tracy/sliding/ops.csv \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-summary \
  > models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/tracy/sliding/decode_perf_report.txt

tt-perf-report models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/tracy/full/ops.csv \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --no-summary \
  > models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/tracy/full/prefill_perf_report.txt

tt-perf-report models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/tracy/full/ops.csv \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-summary \
  > models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/tracy/full/decode_perf_report.txt
```

CSV/provenance and stacked summaries used the same inputs with `--csv` and
`--summary-file`; files are under `tracy/{sliding,full}/`.

Final accepted latency:

| Layer | Mode | Before us | After us | CCL us | Matmul us | Movement us | Op gap us | Ops | Host ops |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sliding | Prefill | 1487.611 | 1487.611 | 421.360 | 563.834 | 0.000 | 1111.227 | 26 | 0 |
| Sliding | Traced decode | 679.406 | 679.406 | 216.587 | 197.993 | 107.574 | 631.727 | 47 | 0 |
| Full | Prefill | 1713.979 | 1713.979 | 420.338 | 682.993 | 0.000 | 906.137 | 26 | 0 |
| Full | Traced decode | 881.606 | 881.606 | 225.801 | 284.531 | 164.049 | 75.282 | 47 | 0 |

## Optimization trials

### Async CCL

Change tested: replace `_all_reduce_tp` with
`ttnn.experimental.all_reduce_async` using `mesh_device`, `cluster_axis`,
`num_links=1`, `math_op=ttnn.ReduceType.Sum`, `memory_config`, and ring
topology.

Focused correctness passed. Profile artifacts:
`tracy/async_ccl/{sliding,full}/`.

| Layer | Mode | Baseline us | Async CCL us | Decision |
| --- | --- | ---: | ---: | --- |
| Sliding | Prefill | 1487.611 | 1490.959 | Reject |
| Sliding | Traced decode | 679.406 | 681.769 | Reject |
| Full | Prefill | 1713.979 | 1709.462 | Mixed |
| Full | Traced decode | 881.606 | 883.397 | Reject |

Decision: reverted to `ttnn.all_reduce` because the decode target regressed and
CCL time did not improve.

### CCL `num_links=2`

Change tested: `RingCCLManager` default `num_links=2`.

Focused correctness hit:

```text
TT_FATAL Unexpected values for event in completion queue, got cmd id
CQDispatchCmdId::CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST, is event 0,
length 8208, pad1 0 expected 1024
```

The stuck pytest process was killed and devices were reset with `tt-smi -r all`.
Decision: rejected and reverted to `num_links=1`.

### Prefill L1 inputs

Change tested: temporary `prefill_l1_inputs` conversions for attention and MLP
prefill inputs and pre-O/down activations.

Short correctness passed and the profile was mixed:

| Layer | Mode | Baseline us | L1 trial us |
| --- | --- | ---: | ---: |
| Sliding | Prefill | 1487.611 | 1481.033 |
| Sliding | Traced decode | 679.406 | 677.761 |
| Full | Prefill | 1713.979 | 1716.737 |
| Full | Traced decode | 881.606 | 882.515 |

The sliding-only variant then failed long-context PCC:

```text
sliding_attention seq=1024 multichip prefill PCC 0.9915908094485228 < 0.995
```

Decision: rejected and all `prefill_l1_inputs` code was removed.

### Precision and fidelity reductions

Artifact: `precision_trials.jsonl`.

| Trial | Layer | Prefill PCC | Decode PCC | Bar | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| MLP BFP4 all decode weights | Sliding | 0.9997969662 | 0.9898308853 | 0.993 | Reject |
| MLP BFP4 all decode weights | Full | 0.9996992886 | 0.9917303167 | 0.995 | Reject |
| Full-attention BFP8 QKV/O | Full | 0.9962399233 | 0.9838415619 | 0.995 | Reject |

Decision: keep the completed-path selective mix: BF16 activations and norms,
BF16 KV cache, sliding attention decode BFP8 QKV with BF16 O, full attention
BF16 with HiFi3/fp32 accumulation where needed, and MLP BFP8 weights.

### Fused matmul-CCL

Investigated TTNN/local examples:

- `models/tt_transformers/tt/attention.py` uses `all_gather_matmul_async` for
  gather-then-matmul contracts.
- `ttnn.experimental.llama_rs_matmul` is a reduce-scatter+matmul style API.

This decoder's row-parallel O/down projections produce local partial hidden
chunks that must be summed to the replicated residual stream. The available
fused APIs do not express matmul-then-sum-all-reduce-to-replicated-hidden for
this contract. Using them would add gather/scatter around layer boundaries.
Decision: ruled out for this stage.

### Residual layout, activation sharding, and DRAM-sharded matmuls

The completed path already implements the accepted layout:

- Replicated residual stream between layers.
- Decode activations stay local L1 width-sharded across norm, attention,
  residual, MLP, and output boundaries.
- Prefill activations stay DRAM interleaved.
- Decode matmuls use DRAM-sharded weights/configs on the local device path.

Trials that moved prefill activations to L1 were rejected on PCC. Trials that
would move the inter-layer contract to a scattered form were rejected by
contract/API evidence because they require extra gather before the next layer.

### Semaphore and preallocated CCL buffer reuse

No public per-call preallocated-buffer handle was available for the accepted
`ttnn.all_reduce` call site. The closest supported path, the async CCL API,
passed correctness but regressed traced decode. The two-link collective trial
was unstable. Decision: no accepted semaphore/buffer reuse change.

## Watcher

Fully enabled watcher with dispatch watcher active was tried first:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_LOGS_PATH=models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/watcher \
pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_paged_prefill_then_decode_pcc_vs_optimized \
  --tb=short --timeout=1200
```

Result: failed before completion with an idle-ETH dispatch watcher assertion in
`tt_metal/impl/dispatch/kernels/cq_prefetch.cpp`:

```text
ierisc detected invalid NOC command buffer state before starting the next kernel
Current kernel: tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
Watcher detected tripped assert and stopped device.
```

This was isolated to dispatch watcher instrumentation because the repo watcher
docs explicitly provide `TT_METAL_WATCHER_DISABLE_DISPATCH=1` for dispatch
kernel watcher trouble, and the same multichip path passed with ETH watcher
still active:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DISABLE_DISPATCH=1 \
TT_METAL_LOGS_PATH=models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/watcher_eth_no_dispatch \
pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_paged_prefill_then_decode_pcc_vs_optimized \
  --tb=short --timeout=1200
```

Result: `2 passed, 3 warnings in 556.97s`.

Clean log audit:

```bash
grep -n -i -E "assert|error|fault|hang|tripped|critical|watcher stopped|invalid NOC" \
  models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/watcher_eth_no_dispatch/generated/watcher/watcher.log
```

Result: no matches.

Tensix-only watcher was also clean:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
TT_METAL_LOGS_PATH=models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/watcher_tensix \
pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_paged_prefill_then_decode_pcc_vs_optimized \
  --tb=short --timeout=1200
```

Result: `2 passed, 3 warnings in 25.48s`; direct watcher log grep returned no
assert/error matches.

## AutoFix note

`$autofix` was not launched for the final watcher issue because the smallest
repro isolated it to repo-documented dispatch watcher instrumentation rather
than a decoder, CCL, cache, or PCC bug. A serial fix/refutation loop was still
applied: fully enabled watcher failed in dispatch `cq_prefetch.cpp`; the
documented dispatch watcher workaround kept ETH watcher active and passed both
representative layer kinds with a clean watcher log.

## Checklist evidence

| Optimize checklist item | Status | Evidence |
| --- | --- | --- |
| Functional checks pass | Done | `9 passed, 3 warnings in 87.47s` |
| PCC remains at acceptance bar | Done | PCC table above and `pcc_results.jsonl` |
| Paged KV-cache and trace replay correct | Done | Cache/layout and trace tests included in full suite |
| Runtime fallback audit clean | Done | Source grep returned no matches |
| Stress/repeated coverage | Done | Three repeated short PCC runs passed |
| Warmed prefill and traced decode before/after | Done | Performance table and `perf_summary.json` |
| Advice-backed `tt-perf-report` output | Done | `tracy/{sliding,full}/{prefill,decode}_perf_report.txt` and CSV files |
| Watcher clean | Done | ETH watcher active with dispatch watcher disabled: `2 passed` and clean log grep |
| Decoder path traced with no host fallbacks | Done | Trace replay tests passed; perf reports show zero host ops |
| Decode activations width-sharded in L1 | Done | Layout test reports `WIDTH_SHARDED`, L1, shard `[32, 128]` |
| Prefill activations DRAM interleaved | Done | Completed path retained; L1 prefill trial rejected on PCC |
| Optimized composite ops | Done | Local SDPA and optimized RMSNorm/matmul paths retained |
| Explicit configs | Done | Completed path uses explicit memory/program/compute configs inherited from optimized decoder where applicable |
| Clean shard specs/core grids | Done | Hidden/intermediate/local QKV widths are tile aligned under TP8 |
| DRAM-sharded decode matmuls | Done | Completed path retained; reports show decode matmuls on DRAM-sharded path |
| Fused matmul-CCL | Done | Investigated and ruled out by API/contract evidence |
| MoE routed active experts | Not applicable | Dense model |
| Reduced precision/fidelity trials | Done | `precision_trials.jsonl`; all lower-precision candidates rejected on decode PCC |

## Final status

The optimized multichip decoder state is complete for the current hardware and
TTNN API surface. No candidate change was left deferred. The final code path is
the completed TP8 multichip decoder because it is the fastest path that also
preserves the accepted PCC, cache, trace, fallback, and watcher requirements.
