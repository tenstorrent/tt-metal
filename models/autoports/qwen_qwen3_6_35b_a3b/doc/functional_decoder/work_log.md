# Functional Decoder Work Log

Model: `Qwen/Qwen3.6-35B-A3B`
Autoport: `models/autoports/qwen_qwen3_6_35b_a3b`
Branch: `vkovacevic/agentic-research/qb2-qwen36-35b-a3b`
Base SHA at evidence time: `9e4baaa71435fc344b21912c9e30366a23c34a5d`
Stage commit SHA: pending until final local commit

## Architecture

- HF text config is nested under `model.language_model`.
- Text decoder has 40 layers: 30 `linear_attention` layers and 10 `full_attention` layers.
- Pattern starts as `linear_attention, linear_attention, linear_attention, full_attention`.
- Key dimensions: hidden 2048, full-attention heads 16, KV heads 2, head dim 256, partial RoPE dim 64.
- Linear-attention dimensions: 16 key heads x 128, 32 value heads x 128, conv kernel 4.
- MoE dimensions: 256 experts, top-8 routing, expert intermediate 512, shared expert intermediate 512.

## Implementation

- Added `tt/functional_decoder.py`.
- Added `FunctionalDecoder.from_state_dict`, `prefill_forward`, `decode_forward`, full-attention cache allocation, and linear-attention state allocation.
- Full attention implements Qwen per-head Q/gate split, Q/K RMSNorm with Qwen unit offset, partial RoPE, paged fill/update cache, paged decode attention, and output gate.
- Linear attention implements causal depthwise conv state, gated-delta recurrence, L2 Q/K normalization, query scale, recurrent state update, and output projection.
- Linear prefill now uses a 64-token TTNN chunked gated-delta rule. This replaced the earlier token-stepped prefill path for long contexts while keeping single-token decode unchanged.
- MoE implements real router/top-k/scatter semantics, active-expert sparse single-token decode, and 32-token chunked sparse prefill with post-routing reduction.

## Correctness Commands

Syntax:

```bash
set -o pipefail
./python_env/bin/python -m py_compile \
  models/autoports/qwen_qwen3_6_35b_a3b/tt/functional_decoder.py \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/conftest.py \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/logs/py_compile.log
```

Full non-perf suite:

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

Result: `18 passed, 4 deselected` in `logs/correctness_full.log`.

PCC highlights:

- Synthetic linear layer 0, seq 5: prefill `0.9994461003286241`, traced decode `0.9994787951042461`.
- Synthetic full layer 3, seq 33: prefill `0.9996404835641908`, traced decode `0.9994230880969464`.
- Synthetic batch-2 linear layer 0, seq 5: prefill `0.9995132077531952`, traced decode `0.9995844476837497`.
- Synthetic batch-2 full layer 3, seq 33: prefill `0.9996329431454074`, traced decode `0.9994690230596567`.
- Real-weight linear layer 0, seq 1: prefill `0.9996229995741831`, traced decode `0.9988370795673545`.
- Real-weight full layer 3, seq 1: prefill `0.9998681212753325`, traced decode `0.9995886582000745`.
- Real-weight linear layer 0, seq 5: prefill `0.9993761564364843`, traced decode `0.9997758895133866`.
- Real-weight full layer 3, seq 5: prefill `0.9997494253841673`, traced decode `0.9996446766105499`.
- Trace eager-vs-trace controls: `1.0` for both layer kinds.
- Repeated decode determinism: `1.0` for both layer kinds.
- Full-attention advertised-context traced decode control: `1.0`.

Focused real-weight multi-token MoE prefill check:

```bash
set -o pipefail
timeout 600 env RUN_QWEN36_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=600 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py \
  -k 'test_real_weight_functional_decoder_prefill_decode_against_hf' -s \
  | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/logs/real_weight_multitoken_moe.log
```

Result: `4 passed, 18 deselected` in `logs/real_weight_multitoken_moe.log`.

## Context Commands

Default context probes included in the full suite:

- Full-attention advertised decode: context `262144`, `current_pos=262143`.
- Full-attention advertised traced decode: context `262144`, `current_pos=262143`, PCC `1.0`.
- Full-attention prefill non-aligned: seq `1025`.
- Linear-attention prefill/decode non-aligned: seq `65`.

Full-attention advertised prefill refresh on current code:

```bash
set -o pipefail
timeout 1800 env RUN_QWEN36_CONTEXT_PROBE=1 QWEN36_CONTEXT_PREFILL_SEQ=262144 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=1800 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py \
  -k 'test_context_probe_full_attention_prefill_non_aligned' -s \
  | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/logs/context_probe_full_prefill_262144_sparse.log
```

Result: `1 passed, 19 deselected` in `359.85s`; call time `356.49s`.

Full-attention near-max non-divisible prefill on current code:

```bash
set -o pipefail
timeout 1800 env RUN_QWEN36_CONTEXT_PROBE=1 QWEN36_CONTEXT_PREFILL_SEQ=262143 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=1800 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py \
  -k 'test_context_probe_full_attention_prefill_non_aligned' -s \
  | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/logs/context_probe_full_prefill_262143_sparse.log
```

Result: `1 passed, 21 deselected` in `362.66s`; call time `359.25s`.

Linear-attention advertised prefill/decode after the chunked fix:

```bash
set -o pipefail
timeout 2400 env RUN_QWEN36_CONTEXT_PROBE=1 QWEN36_CONTEXT_LINEAR_SEQ=262144 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=2400 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py \
  -k 'test_context_probe_linear_attention_prefill_decode_non_aligned' -s \
  | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/logs/context_probe_linear_prefill_262144_chunked.log
```

Result: `1 passed, 19 deselected` in `319.40s`; call time `315.48s`.

Linear-attention near-max non-divisible prefill/decode after the chunked fix:

```bash
set -o pipefail
timeout 2400 env RUN_QWEN36_CONTEXT_PROBE=1 QWEN36_CONTEXT_LINEAR_SEQ=262143 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=2400 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py \
  -k 'test_context_probe_linear_attention_prefill_decode_non_aligned' -s \
  | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/logs/context_probe_linear_prefill_262143_chunked.log
```

Result: `1 passed, 21 deselected` in `324.38s`; call time `320.42s`.

Linear non-aligned chunk-boundary probes:

- `QWEN36_CONTEXT_LINEAR_SEQ=65`: `1 passed, 19 deselected` in `logs/autofix_chunked_linear_context_65.log`.
- `QWEN36_CONTEXT_LINEAR_SEQ=1025`: `1 passed, 19 deselected` in `logs/autofix_chunked_linear_context_1025.log`.

Historical AutoFix evidence:

- The original token-stepped linear path passed `131073` tokens in `logs/context_probe_linear_prefill_131073_sparse.log`.
- The original token-stepped linear path timed out at `262144` after 5400 seconds in `logs/context_probe_linear_prefill_262144_long.log` with no TT_FATAL, traceback, FAIL, PASS, or device-health failure line.
- `AUTODEBUG_linear_context.md` records the diagnosis: algorithmic dispatch scaling in token-stepped gated-delta prefill, not a physical DRAM/L1/NoC/watcher limit.
- `AUTOFIX_linear_chunked_design.md` records the source-only chunked design that was implemented.

## Performance Commands

Tracy run:

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

Result: `4 passed, 16 deselected` in `logs/tracy_perf_summary.log.gz`. The original command wrote `logs/tracy_perf_summary.log`; the committed artifact is gzip-compressed for the repository per-file size hook and expands to the same path without `.gz`.

Raw ops CSVs:

- Original legacy Tracy output: `tracy/raw/reports/2026_08_18_22_31_32/ops_perf_results_2026_08_18_22_31_32.csv.gz`
- Blackhole-normalized report input: `tracy/raw/reports/2026_08_18_22_31_32/ops_perf_results_2026_08_18_22_31_32_blackhole.csv.gz`

`tt-perf-report` tables and CSVs:

- `tracy/linear_attention/prefill_ops.csv.gz`
- `tracy/linear_attention/prefill_perf_report.txt`
- `tracy/linear_attention/prefill_perf_report.csv`
- `tracy/linear_attention/decode_ops.csv`
- `tracy/linear_attention/decode_perf_report.txt`
- `tracy/linear_attention/decode_perf_report.csv`
- `tracy/full_attention/prefill_ops.csv`
- `tracy/full_attention/prefill_perf_report.txt`
- `tracy/full_attention/prefill_perf_report.csv`
- `tracy/full_attention/decode_ops.csv`
- `tracy/full_attention/decode_perf_report.txt`
- `tracy/full_attention/decode_perf_report.csv`

Perf summary:

- Linear prefill seq 5: wall `45.456 ms`, report device time `37.162 ms`.
- Full prefill seq 33: wall `35.810 ms`, report device time `34.158 ms`.
- Linear traced decode after seq 5: wall `3.023 ms`, report device time `2.923 ms`.
- Full traced decode after seq 33: wall `2.714 ms`, report device time `2.621 ms`.

## Fallback And Watcher

Fallback audit:

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

Result: `3 passed, 19 deselected` in `logs/runtime_fallback_audit.log`.

Watcher:

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

Result: `5 passed, 15 deselected` in `logs/watcher_correctness.log`.

Watcher scan:

```bash
rg -n -i 'fatal|assert|watcher exception|noc|l1|cb|stack|sanitize|timeout|deadlock|hang|error' \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/watcher/final/generated/watcher \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/logs/watcher_correctness.log
```

Result: no watcher error patterns; only pytest timeout configuration lines in `logs/watcher_correctness.log`.

## Hardware Recovery Note

An initial traced-PCC helper attempted capture without a warmup and hit TTNN's trace guard:

```text
TT_FATAL: Writes are not supported during trace capture
TT_FATAL: Reads are not supported during trace capture
```

Triage was captured under `triage/trace_capture_unwarmed/`. I stopped only the stale pytest from that run, then ran `tt-smi` list/reset/list and a mesh-open smoke.

Artifacts:

- `logs/tt_smi_list_after_trace_failure.log`
- `logs/tt_smi_reset_after_trace_failure.log`
- `logs/tt_smi_list_after_reset.log`
- `logs/mesh_smoke_after_reset.log`

The test helper now warms decode before trace capture.

## Limitations

- Functional-decoder context capability is not reduced from the HF-advertised 262144 tokens for either target layer kind.
- This stage is correctness-oriented single-device decoder-layer work. Optimized decoder, multichip, full-model, and vLLM paths are intentionally out of scope for this goal.

## Stage Review And Checkpoint

- Fresh `$stage-review` subagent `01a01725-006f-7242-96df-7dd7c144b2f0` returned `clean-pass` with no required work.
- Review artifact: `doc/functional_decoder/stage_review.md`.
- Checkpoint scope: `models/autoports/qwen_qwen3_6_35b_a3b` only. The unrelated untracked `tt_metal/third_party/tt-cluster-descriptors/` workspace state is excluded.
- Stage commit SHA: pending until final local commit; a commit cannot contain its own final SHA, so the final SHA is reported after commit creation.
