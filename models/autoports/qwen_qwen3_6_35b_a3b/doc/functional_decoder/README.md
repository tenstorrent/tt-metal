# Qwen3.6-35B-A3B Functional Decoder

This directory records the functional-decoder state for `Qwen/Qwen3.6-35B-A3B` under `models/autoports/qwen_qwen3_6_35b_a3b`. The implementation is single-device TTNN decoder-layer code only; no optimized decoder, multichip decoder, full model, or vLLM work is included.

## Implementation

`tt/functional_decoder.py` implements `FunctionalDecoder(LightweightModule)` for the two text decoder layer kinds advertised by the HF config:

| Kind | Representative layer | Count | State |
| --- | ---: | ---: | --- |
| `linear_attention` | 0 | 30 | four-tap causal conv state plus gated-delta recurrent state |
| `full_attention` | 3 | 10 | paged K/V cache with caller-owned page table |

Public contract:

| Forward | Hidden shape | Required state/arguments | Return |
| --- | --- | --- | --- |
| `prefill_forward` | `[1, batch, seq, 2048]` | full attention: `position_embeddings`, `page_table`, optional `kv_cache`; linear attention: `linear_state` | `FunctionalDecoderResult` |
| `decode_forward` | `[1, 1, batch, 2048]` | `current_pos`; full attention also needs `position_embeddings`, `page_table`, `kv_cache`; linear attention needs `linear_state` | `FunctionalDecoderResult` |

Full attention implements Qwen's per-head Q/gate split, Q/K RMSNorm with unit offset, partial RoPE, paged prefill/update cache, paged decode attention, and output gate. Linear attention implements causal conv plus gated-delta state. Prefill streams 64-token chunks through the TTNN chunked gated-delta rule with carried conv and recurrent state; decode uses the single-token recurrent path. Decode has warmed TTNN trace capture/replay coverage for both layer kinds.

The MoE path uses real router/top-k/scatter semantics. Single-token decode uses active-expert sparse matmul. Prefill uses 32-token chunks with sparse matmul and post-routing reduction, avoiding dense all-expert routed-down allocation.

Setup helpers and tests use Torch at explicit boundaries; runtime prefill/decode methods are audited for no Torch conversion or host fallback calls.

## Correctness

Acceptance bar: PCC >= 0.995.

Command:

```bash
set -o pipefail
timeout 900 env RUN_QWEN36_REAL_WEIGHTS=1 RUN_QWEN36_CONTEXT_PROBE=1 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py \
  -k 'not perf' -s \
  | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/logs/correctness_full.log
```

Result artifact: `logs/correctness_full.log` (`18 passed, 4 deselected`). A focused real-weight rerun that isolates the multi-token MoE prefill path is recorded in `logs/real_weight_multitoken_moe.log`.

| Case | Prefill PCC | Traced decode PCC |
| --- | ---: | ---: |
| synthetic linear layer 0, batch 1, seq 5 | 0.9994461003286241 | 0.9994787951042461 |
| synthetic full layer 3, batch 1, seq 33 | 0.9996404835641908 | 0.9994230880969464 |
| synthetic linear layer 0, batch 2, seq 5 | 0.9995132077531952 | 0.9995844476837497 |
| synthetic full layer 3, batch 2, seq 33 | 0.9996329431454074 | 0.9994690230596567 |
| real-weight linear layer 0, batch 1, seq 1 | 0.9996229995741831 | 0.9988370795673545 |
| real-weight full layer 3, batch 1, seq 1 | 0.9998681212753325 | 0.9995886582000745 |
| real-weight linear layer 0, batch 1, seq 5 | 0.9993761564364843 | 0.9997758895133866 |
| real-weight full layer 3, batch 1, seq 5 | 0.9997494253841673 | 0.9996446766105499 |

Additional controls:

| Case | PCC |
| --- | ---: |
| synthetic linear traced-decode eager-vs-trace | 1.0 |
| synthetic full traced-decode eager-vs-trace | 1.0 |
| synthetic linear repeated decode determinism | 1.0 |
| synthetic full repeated decode determinism | 1.0 |
| full-attention traced advertised-context decode control | 1.0 |

`logs/py_compile.log` records a successful syntax check for `tt/functional_decoder.py`, `tests/test_functional_decoder.py`, and `tests/conftest.py`.

## Context

The HF text config advertises `max_position_embeddings = 262144`. No functional-decoder context capability reduction is recorded.

| Claim | Status | Artifact |
| --- | --- | --- |
| full-attention prefill seq 262144 | passed, 356.49s call | `logs/context_probe_full_prefill_262144_sparse.log` |
| full-attention prefill seq 262143 | passed, 359.25s call | `logs/context_probe_full_prefill_262143_sparse.log` |
| full-attention traced decode context 262144, `current_pos=262143` | passed, control PCC 1.0 | `logs/correctness_full.log`, `logs/context_probe_traced_decode_advertised.log` |
| linear-attention prefill/decode seq 262144, `current_pos=262144` | passed, 315.48s call | `logs/context_probe_linear_prefill_262144_chunked.log` |
| linear-attention prefill/decode seq 262143, `current_pos=262143` | passed, 320.42s call | `logs/context_probe_linear_prefill_262143_chunked.log` |

Additional current-implementation context artifacts cover non-aligned lengths around tile/page/chunk boundaries and near the supported context: synthetic parity at `5` and `33`, linear boundary prefill/decode at `65`, full-attention prefill at `1025`, and near-max non-divisible prefill/decode at `262143`. The small probes are in `logs/correctness_full.log`; linear chunked probes at `65` and `1025` are also recorded in `logs/autofix_chunked_linear_context_65.log` and `logs/autofix_chunked_linear_context_1025.log`. The structured contract is in `../context_contract.json`.

Historical note: the earlier token-stepped linear prefill path completed through `131073` and timed out at `262144` after 5400 seconds without hardware failure evidence. `AUTODEBUG_linear_context.md` identified dispatch scaling in the token-stepped gated-delta recurrence, and `AUTOFIX_linear_chunked_design.md` records the source-only design used for the 64-token TTNN chunked implementation that now passes advertised context.

## Performance

Performance was captured in a separate Tracy run with warmed prefill and warmed traced decode. `tt-perf-report` CSV `Device Time` is in microseconds.

Command:

```bash
set -o pipefail
timeout 1200 env RUN_QWEN36_PERF=1 RUN_QWEN36_PERF_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m tracy -r -p -v \
  --no-runtime-analysis --op-support-count=5000 --check-exit-code \
  --output-folder models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/tracy/raw \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=1200 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py \
  -k test_perf_qwen36 -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/logs/tracy_perf_summary.log
```

Result artifact: `logs/tracy_perf_summary.log.gz` (`4 passed, 16 deselected`). Raw ops CSV archive: `tracy/raw/reports/2026_08_18_22_31_32/ops_perf_results_2026_08_18_22_31_32.csv.gz`. `tt-perf-report` tables were generated from `ops_perf_results_2026_08_18_22_31_32_blackhole.csv.gz`, a metadata-normalized copy that sets the legacy CSV's blank device architecture fields to Blackhole and worker-core count to 110. Oversized profiler artifacts are committed as gzip archives to satisfy the repository per-file size hook; each expands to the same path without `.gz`.

| Case | Wall ms | tt-perf-report device ms | Table | Filtered ops CSV |
| --- | ---: | ---: | --- | --- |
| linear prefill, seq 5 | 45.456 | 37.162 | `tracy/linear_attention/prefill_perf_report.txt` | `tracy/linear_attention/prefill_ops.csv.gz` |
| full prefill, seq 33 | 35.810 | 34.158 | `tracy/full_attention/prefill_perf_report.txt` | `tracy/full_attention/prefill_ops.csv` |
| linear traced decode after seq 5 | 3.023 | 2.923 | `tracy/linear_attention/decode_perf_report.txt` | `tracy/linear_attention/decode_ops.csv` |
| full traced decode after seq 33 | 2.714 | 2.621 | `tracy/full_attention/decode_perf_report.txt` | `tracy/full_attention/decode_ops.csv` |

Each `*_ops.csv` is filtered to the corresponding signposted measured window; `*_perf_report.csv`, `*_perf_report_stacked.csv`, `*_perf_report_stacked.png`, and `*_perf_report.console.log` sit beside the table.

## Fallback And Watcher

Runtime fallback audit:

```bash
set -o pipefail
timeout 600 env TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=600 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py \
  -k 'test_synthetic_functional_decoder_prefill_decode_against_hf or runtime_fallback_audit_source' -s \
  | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/logs/runtime_fallback_audit.log
```

Artifact: `logs/runtime_fallback_audit.log` (`3 passed, 19 deselected`). The source audit covers the runtime decoder methods and helpers used inside a measured pass, rejecting `torch`, `ttnn.from_torch`, `ttnn.to_torch`, and `get_fallback_function`.

Watcher command:

```bash
set -o pipefail
timeout 900 env \
  TT_METAL_LOGS_PATH=/localdev/vkovacevic/tt-metal/models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/watcher/final \
  TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
  RUN_QWEN36_CONTEXT_PROBE=1 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py \
  -k 'test_synthetic_functional_decoder_prefill_decode_against_hf or test_synthetic_functional_decoder_traced_decode or test_context_probe_full_attention_decode_advertised_context_traced_control' -s \
  | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/logs/watcher_correctness.log
```

Artifact: `logs/watcher_correctness.log` (`5 passed, 15 deselected`) and `watcher/final/generated/watcher/watcher.log`. A scan for fatal/assert/NOC/L1/CB/sanitize/timeout/deadlock/hang/error patterns found only pytest timeout configuration lines and no watcher failures.

## Notes

The repo root `conftest.py` imports `models.tt_transformers.demo.trace_region_config`, which is unavailable in this checkout. The tests are run with `--confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests` so this autoport's local `conftest.py` supplies the TTNN device fixture without editing unrelated repo files.

An unwarmed trace-PCC harness attempt failed with TTNN's expected "Writes are not supported during trace capture" guard. The helper now performs an eager warmup before capture, matching the existing trace smoke. Triage and recovery evidence is under `triage/trace_capture_unwarmed/` and `logs/tt_smi_*after_trace_failure*.log`.
