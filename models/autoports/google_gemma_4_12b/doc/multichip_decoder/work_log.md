# Multichip decoder work log

Repo revision: `31b45719e2ca21b695a8e7f15b5e8895bc1fb3bb`
Date: 2026-06-09 UTC
Target: `google/gemma-4-12B`

Scope: only `tt/multichip_decoder.py`, `tests/test_multichip_decoder.py`, and
`doc/multichip_decoder/*`. Full-model and vLLM work were not started.

## Implementation summary

- Added `MultichipDecoder`, using `OptimizedDecoder` as the TP1 baseline and
  targeting only T3K `MeshShape(1, 8)`.
- Implemented TP8 with replicated residual activations, local decode
  width-sharded L1 tensors, column-parallel QKV and MLP gate/up, local SDPA,
  row-parallel attention O and MLP down, and ring all-reduce after each
  row-parallel projection.
- Packed fused QKV weights per TP rank. Sliding layers shard 8 KV heads over 8
  devices; full layers replicate their single KV head on every device.
- Reused optimized single-chip matmul, RMSNorm, SDPA, paged-cache, dtype, and
  memory-config choices where they remain valid locally.
- Added tests for source fallback audit, short and long PCC versus optimized,
  paged KV/page-position/layout contracts, decode trace replay and determinism,
  and signposted perf windows.
- No MoE routing is needed because this model is dense.

## Mesh plan

| Item | Value |
| --- | --- |
| Hardware | Wormhole T3K |
| Devices | 8 |
| Mesh | 1x8 |
| TP | 8 on axis 1 |
| Fabric | `FABRIC_1D_RING` |
| CCL topology | `ttnn.Topology.Ring` |
| Residual contract | Replicated `[1, 1, S, 3840]` |
| Decode output layout | Per-device L1 width-sharded, shard shape `[32, 128]`, 30 cores |
| Padding | None required |

Per-device dimensions:

| Tensor | Sliding | Full |
| --- | ---: | ---: |
| Q heads | 2 | 2 |
| KV heads | 1 | 1 replicated |
| Local Q width | 512 | 1024 |
| Local K width | 256 | 512 |
| Local V width | 256 | 512 |
| Local fused QKV width | 1024 | 2048 |
| Local MLP intermediate | 1920 | 1920 |
| Hidden if sharded | 480 | 480 |

Rejected alternatives:

- Replicated all weights: no mesh speedup.
- Hidden-sharded residual stream: distributed RMSNorm and stack-boundary gathers
  made it a worse layer-stack baseline.
- Sequence-parallel prefill: conflicts with the simple replicated residual
  contract and decode-focused target.
- 2D mesh: not available on the target T3K.
- Reduce-scatter output contract: would require a gather before the next layer.
- Full-attention KV sharding: impossible with one KV head.
- HiFi4/fp32 decode QKV: marginal PCC improvement over HiFi3/fp32, with
  Wormhole warning risk.
- SDPA decode grid sweeps: no PCC improvement in long full decode.

## Correctness commands

Syntax:

```bash
python -m py_compile \
  models/autoports/google/gemma-4-12B/tt/multichip_decoder.py \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py
```

Result: passed.

Full correctness, cache, and trace coverage:

```bash
pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_runtime_fallback_audit_source_clean \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_paged_prefill_then_decode_pcc_vs_optimized \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_long_context_paged_prefill_decode_vs_optimized \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_cache_and_stacked_layout_contract \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_decode_trace_replay_pcc_and_determinism_vs_optimized \
  --tb=short --timeout=900
```

Result: `9 passed, 3 warnings in 87.70s`.

Source fallback audit:

```bash
grep -nE 'ttnn\.from_torch|ttnn\.to_torch|FunctionalDecoder' \
  models/autoports/google/gemma-4-12B/tt/multichip_decoder.py || true
```

Result: no matches.

## PCC results

Latest accepted PCCs from `pcc_results.jsonl`:

| Layer | Seq | Prefill PCC | Decode PCC | Decode bar | Replica PCC |
| --- | ---: | ---: | ---: | ---: | --- |
| Sliding | 128 | 0.9997969662 | 0.9964325808 | 0.993 | all 1.0 |
| Full | 128 | 0.9996992886 | 0.9983275303 | 0.995 | all 1.0 |
| Sliding | 1024 | 0.9994995075 | 0.9993016740 | 0.992 | all 1.0 |
| Full | 1024 | 0.9994667300 | 0.9924891310 | 0.992 | all 1.0 |

Trace replay:

| Layer | Replay PCC | Determinism PCC | Replica PCC |
| --- | ---: | ---: | --- |
| Sliding | 0.9964325808 | 1.0 | all 1.0 |
| Full | 0.9983275303 | 1.0 | all 1.0 |

Cache and stacked-decoder contracts:

| Layer | Seq 128 KV shape | Seq 1024 KV shape | Page table shapes | Position shapes |
| --- | --- | --- | --- | --- |
| Sliding | `[7, 1, 64, 256]` | `[21, 1, 64, 256]` | `[1, 7]`, `[1, 21]` | `[1, 1]` |
| Full | `[7, 1, 64, 512]` | `[21, 1, 64, 512]` | `[1, 7]`, `[1, 21]` | `[1, 1]` |

The decode output contract is `[1, 1, 1, 3840]` with L1 width-sharded memory.

## Debugging and AutoFix notes

The subtle bug was long-context full-attention decode PCC, initially around
0.9893 against the optimized baseline. I followed the AutoFix isolation loop
locally. Forked subagents were not launched because the available multi-agent
tool is restricted to requests that explicitly ask for delegated agents.

Smallest reproducible evidence:

- Full long decode failed while sliding long decode passed.
- QKV prefill K/V chunks matched the optimized baseline exactly: PCC 1.0 and max
  error 0.0.
- Paged cache fill from TP K/V matched the expected page-table scatter: PCC 1.0
  on every device.
- Decode Q/K/V and cache update were high quality, but local SDPA chunks showed
  large PCC drops on a few ranks, showing softmax sensitivity to small decode
  QKV projection error.

Concrete fix/refutation loop:

| Trial | Result | Decision |
| --- | --- | --- |
| Full attention O HiFi3/fp32 | PCC stayed near 0.9893 | Reject |
| Full prefill QKV HiFi3/fp32 | PCC worsened to about 0.9870 | Reject |
| Interleaved BF16 full prefill weights | Prefill fell to about 0.9876 | Reject |
| Separate BF16 K projection cache fill | Decode about 0.9788 | Reject |
| Separate K projection with HiFi4/fp32 | Decode about 0.9871 | Reject |
| SDPA grid sweep | No material PCC movement | Reject |
| Full decode QKV HiFi3/fp32 | Long decode 0.992489, passes 0.992 bar | Keep |
| Full decode QKV HiFi4/fp32 | 0.992540, marginally higher with warning risk | Reject |

The final path uses HiFi3 with fp32 destination accumulation for full decode
QKV. Sliding decode keeps the optimized HiFi2 path, with the inherited optimized
long-position precision behavior.

## Watcher

Clean Tensix watcher run:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
TT_METAL_LOGS_PATH=models/autoports/google/gemma-4-12B/doc/multichip_decoder/watcher_tensix \
pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_paged_prefill_then_decode_pcc_vs_optimized \
  --tb=short --timeout=600
```

Result: `2 passed, 3 warnings in 637.58s`.

Watcher artifact:
`models/autoports/google/gemma-4-12B/doc/multichip_decoder/watcher_tensix/generated/watcher/watcher.log`.

I also tried ETH-enabled watcher runs while debugging. Those were not used as
the clean evidence path because one run overflowed watcher firmware size without
`TT_METAL_WATCHER_NOINLINE=1`, and a later ETH-enabled run tripped a dispatch
prefetch watcher assertion outside the decoder kernels. Devices were reset
before continuing, and the final Tensix-only watcher run completed cleanly.

## Performance commands

Sliding Tracy collection:

```bash
python -m tracy -r -p -v \
  -o models/autoports/google/gemma-4-12B/doc/multichip_decoder/tracy/sliding/raw \
  -m pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_perf_warmed_prefill_and_traced_decode \
  --tb=short -k sliding --timeout=900
```

Result: `1 passed, 1 deselected, 3 warnings in 105.09s`.
Raw CSV: `tracy/sliding/raw/reports/2026_06_09_00_15_16/ops_perf_results_2026_06_09_00_15_16.csv`.

Full Tracy collection:

```bash
python -m tracy -r -p -v \
  -o models/autoports/google/gemma-4-12B/doc/multichip_decoder/tracy/full/raw \
  -m pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_perf_warmed_prefill_and_traced_decode \
  --tb=short -k full --timeout=900
```

Result: `1 passed, 1 deselected, 3 warnings in 113.18s`.
Raw CSV: `tracy/full/raw/reports/2026_06_09_00_17_32/ops_perf_results_2026_06_09_00_17_32.csv`.

Report extraction:

```bash
for layer in sliding full; do
  raw_dir="models/autoports/google/gemma-4-12B/doc/multichip_decoder/tracy/${layer}/raw/reports"
  latest_csv=$(find "$raw_dir" -name 'ops_perf_results_*.csv' -type f | sort | tail -1)
  cp "$latest_csv" "models/autoports/google/gemma-4-12B/doc/multichip_decoder/tracy/${layer}/ops.csv"
  for mode in prefill decode; do
    if [ "$mode" = prefill ]; then
      start=PERF_PREFILL
      end=PERF_PREFILL_END
    else
      start=PERF_DECODE
      end=PERF_DECODE_END
    fi
    base="models/autoports/google/gemma-4-12B/doc/multichip_decoder/tracy/${layer}/${mode}_perf_report"
    tt-perf-report "models/autoports/google/gemma-4-12B/doc/multichip_decoder/tracy/${layer}/ops.csv" \
      --start-signpost "$start" \
      --end-signpost "$end" \
      --no-summary > "${base}.txt"
    tt-perf-report "models/autoports/google/gemma-4-12B/doc/multichip_decoder/tracy/${layer}/ops.csv" \
      --start-signpost "$start" \
      --end-signpost "$end" \
      --csv "${base}.csv" \
      --summary-file "${base}_stacked" > "${base}.console.log"
  done
done
```

## Final performance

Metric: `tt-perf-report` `Device Time`, compared against
`doc/optimized_decoder/tracy/perf_summary.json`.

| Layer | Mode | Optimized us | Multichip us | Speedup | Efficiency | CCL ops | CCL us | Host ops |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sliding | Prefill | 3294.000 | 1487.611 | 2.214x | 27.7% | 4 | 421.360 | 0 |
| Sliding | Traced decode | 1374.000 | 679.406 | 2.022x | 25.3% | 4 | 216.587 | 0 |
| Full | Prefill | 4404.000 | 1713.979 | 2.569x | 32.1% | 4 | 420.338 | 0 |
| Full | Traced decode | 1740.000 | 881.606 | 1.974x | 24.7% | 4 | 225.801 | 0 |

Communication, DRAM, compute, and movement findings:

- Each row-parallel projection contributes one ring all-reduce. The report
  decomposes two all-reduces into four CCL ops per layer.
- CCL time is a major limiter: 216.587 us in sliding decode and 225.801 us in
  full decode, or roughly one quarter to one third of decode `Device Time`.
- Decode local matmuls are smaller after TP8 but are still marked `SLOW` by
  `tt-perf-report`, mostly DRAM limited at about 118-162 GB/s.
- Full decode has additional BF16/HiFi3 QKV cost to preserve long-context PCC.
- Data movement remains material in decode: 160.301 us sliding and 248.068 us
  full across sharding, tilize/untilize, RoPE, QKV head split, cache update, and
  concat operations.
- `tt-perf-report` reports zero host ops in every signposted window.

Performance artifacts:

- Summary: `tracy/perf_summary.json`
- Stable raw OP CSVs: `tracy/sliding/ops.csv`, `tracy/full/ops.csv`
- Human-readable reports:
  `tracy/sliding/prefill_perf_report.txt`,
  `tracy/sliding/decode_perf_report.txt`,
  `tracy/full/prefill_perf_report.txt`,
  `tracy/full/decode_perf_report.txt`
- CSV and stacked reports:
  `tracy/{sliding,full}/{prefill,decode}_perf_report.csv`,
  `tracy/{sliding,full}/{prefill,decode}_perf_report_stacked.csv`,
  `tracy/{sliding,full}/{prefill,decode}_perf_report_stacked.png`

## Final checklist

- [x] `tt/multichip_decoder.py` exists and uses `OptimizedDecoder` as TP1
  baseline.
- [x] TP8 T3K strategy chosen with calculated per-device shapes and no required
  padding.
- [x] Activation, tensor, KV-cache, collective, and MoE strategies documented.
- [x] Prefill and decode PCC validated against optimized single-chip TTNN for
  sliding and full attention.
- [x] Paged KV-cache, page tables, current positions, local KV layout, and
  stacked input/output layout contracts validated.
- [x] Warmed decode trace replay and determinism validated.
- [x] Warmed optimized baseline and multichip latency, speedup, and efficiency
  reported.
- [x] `tt-perf-report` human-readable tables, CSVs, stacked reports, and
  provenance logs exist.
- [x] Runtime fallback source audit and Tracy host-op audit are clean.
- [x] Watcher-clean evidence exists for the Tensix decoder path.
- [x] Dense model has no MoE active-expert requirement.
- [x] Full-model and vLLM work remain untouched.
