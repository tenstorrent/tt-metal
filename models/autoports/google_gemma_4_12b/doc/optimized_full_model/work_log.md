# Optimized Full Model Work Log

## Scope

Goal: optimize the completed `google/gemma-4-12B` repo-local TTNN full model on the target T3K mesh, without starting vLLM integration.

Applied skills: `multichip`, `optimize`, and `autofix`. No new subtle regression required a fresh forked AutoFix pass in this stage; the earlier full-model special-token generation issue remains documented in `../full_model/work_log.md`.

Provenance:

- HF model: `google/gemma-4-12B`
- Local snapshot: `/home/moconnor/.cache/huggingface/hub/models--google--gemma-4-12B/snapshots/56820d7d8cbe8e47975a53325439ed272e91cff2`
- Hardware: Wormhole T3K, 8 devices
- Mesh/fabric: `ttnn.MeshShape(1, 8)`, `ttnn.FabricConfig.FABRIC_1D_RING`
- Baseline full-model report: `../full_model/README.md`
- Inherited optimized decoder report: `../optimized_multichip_decoder/README.md`

## Implementation Changes

Updated `tt/model.py`:

- Added full-model TP embedding and LM-head ownership around the optimized multichip decoder stack.
- Kept embedding hidden-sharded across TP8 and gathered to the full residual stream once at the model entry.
- Split LM-head weights into conservative BF16 prefill and optimized decode/last-token paths.
- Added decode LM-head DRAM-sharded matmul with BFP4 LoFi default, environment overrides, width-sharded input, and padded sharded output.
- Added preallocated decode input helpers and trace-safe `decode_forward_device_inputs`.
- Added on-device greedy sampling through `models.common.sampling.SamplingGenerator`.

Updated `tt/generator.py`:

- Added trace capture/replay for full-wrapper decode.
- Added trace teardown.
- Routed `decode_forward(..., enable_trace=True)` through trace replay.
- Kept high-level text generation host-masked by default for non-EOS special-token suppression.

Added `tests/test_optimized_full_model.py`:

- T3K-only profile harness.
- Warmed prefill.
- Signposted `PERF_PREFILL` and `PERF_DECODE` windows.
- Traced decode replay with `sample_on_device=True` and `return_ttnn=True`.

## Validation Commands

Prefill:

```bash
python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/google/gemma-4-12B \
  --reference models/autoports/google/gemma-4-12B/readiness_aime24_plain.refpt \
  --mesh-device T3K --fabric-config FABRIC_1D_RING \
  2>&1 | tee models/autoports/google/gemma-4-12B/doc/optimized_full_model/artifacts/run_prefill_check_aime24_plain_optimized.log
```

Result: `AGGREGATE top1=0.969 (31/32) top5=1.000 (32/32) top100=1.000 (32/32)`.

Teacher forcing:

```bash
python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/google/gemma-4-12B \
  --reference models/autoports/google/gemma-4-12B/readiness_aime24_plain.refpt \
  --mesh-device T3K --fabric-config FABRIC_1D_RING \
  2>&1 | tee models/autoports/google/gemma-4-12B/doc/optimized_full_model/artifacts/run_teacher_forcing_aime24_plain_optimized.log
```

Result: `AGGREGATE top1=0.938 (30/32) top5=1.000 (32/32) top100=1.000 (32/32)`.

Autoregressive:

```bash
python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/google/gemma-4-12B \
  --hf-model google/gemma-4-12B \
  --prompt-file models/autoports/google/gemma-4-12B/doc/full_model/artifacts/aime24_prompt_0_plain.txt \
  --mesh-device T3K --fabric-config FABRIC_1D_RING \
  --output-dir models/autoports/google/gemma-4-12B/doc/optimized_full_model/artifacts/autoregressive_aime24_plain_masked \
  --max-new-tokens 32 \
  2>&1 | tee models/autoports/google/gemma-4-12B/doc/optimized_full_model/artifacts/run_autoregressive_aime24_plain_masked_optimized.log
```

Result: HF and TT both produced 32 coherent AIME24 continuation tokens. TT is not token-identical after optimization, so the top-k gates above are the acceptance evidence.

Focused harness:

```bash
pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_optimized_full_model.py::test_optimized_full_model_prefill_and_traced_decode_profile \
  --tb=short --timeout=1200 \
  2>&1 | tee models/autoports/google/gemma-4-12B/doc/optimized_full_model/artifacts/test_optimized_full_model_profile_harness.log
```

Result: `1 passed, 3 warnings in 10.06s`.

Watcher:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DISABLE_DISPATCH=1 \
TT_METAL_LOGS_PATH=models/autoports/google/gemma-4-12B/doc/optimized_full_model/watcher_eth_no_dispatch \
GEMMA4_12B_FULL_MODEL_PROFILE_LAYERS=2 \
pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_optimized_full_model.py::test_optimized_full_model_prefill_and_traced_decode_profile \
  --tb=short --timeout=1200 \
  2>&1 | tee models/autoports/google/gemma-4-12B/doc/optimized_full_model/artifacts/watcher_optimized_full_model_profile_harness.log
```

Result: `1 passed, 3 warnings in 425.91s`. Watcher log detach reports zero Ethernet retraining events; no watcher assertion is present.

## Performance Commands

Host-masked compatibility path:

```bash
python - <<'PY' 2>&1 | tee models/autoports/google/gemma-4-12B/doc/optimized_full_model/artifacts/perf_batch1_aime24_plain_masked_host_argmax_optimized.log
# See artifact for exact inline script.
PY
```

Result:

```json
{
  "prompt_tokens": 149,
  "generated_tokens_total": 32,
  "decode_tokens_timed": 31,
  "ttft_ms": 121.92969070747495,
  "decode_s": 7.27522938977927,
  "decode_tokens_per_second_per_user": 4.261034029188259
}
```

Traced on-device sampling, final default:

```bash
python - <<'PY' 2>&1 | tee models/autoports/google/gemma-4-12B/doc/optimized_full_model/artifacts/perf_batch1_aime24_plain_traced_on_device_sampling_optimized.log
# See artifact for exact inline script.
PY
```

Result:

```json
{
  "lm_head_decode_dtype": "bfloat4_b",
  "lm_head_decode_fidelity": "lofi",
  "timed_replay_tokens": 16,
  "replay_ms_per_token": 43.31933718640357,
  "replay_tokens_per_second_per_user": 23.084379054485282
}
```

BF16 traced baseline for LM-head precision tuning:

```bash
GEMMA4_12B_FULL_MODEL_LM_HEAD_DECODE_DTYPE=bf16 \
GEMMA4_12B_FULL_MODEL_LM_HEAD_DECODE_FIDELITY=hifi2 \
python - <<'PY' 2>&1 | tee models/autoports/google/gemma-4-12B/doc/optimized_full_model/artifacts/perf_batch1_aime24_plain_traced_on_device_sampling_lm_head_bf16.log
# See artifact for exact inline script.
PY
```

Result: `22.664213958769157 tokens/s/user`, `44.122421444626525 ms/token`.

## Precision Trials

Teacher-forcing top-k stayed unchanged for all tested LM-head decode precisions:

| Trial | Teacher forcing result | Trace replay result |
| --- | --- | --- |
| BF16 HiFi2 | top5 32/32, top100 32/32 | 22.66 tokens/s/user |
| BFP8 HiFi2 | top5 32/32, top100 32/32 | 23.18 tokens/s/user |
| BFP4 HiFi2 | top5 32/32, top100 32/32 | 23.23 tokens/s/user |
| BFP4 LoFi | top5 32/32, top100 32/32 | 23.33 tokens/s/user in the precision artifact, 23.08 tokens/s/user in the refreshed canonical run |

Final default: BFP4 LoFi decode LM head. The refreshed canonical run is slightly slower than the first BFP4 LoFi trial but still faster than BF16 and remains within normal timing variance.

Decoder precision trials are inherited from `../optimized_multichip_decoder/precision_trials.jsonl`: BFP4 MLP decode weights and full-attention BFP8 QKV/O were rejected because decode PCC fell below acceptance bars.

## LM-Head Config Trial Notes

The decode LM-head profiler advice says to try `in0_block_w >= 2`. The following focused smoke failures document why `in0_block_w=1` remains:

- `artifacts/smoke_num_layers1_traced_decode.log`: DRAM-sharded matmul requires sharded output memory config.
- `artifacts/smoke_num_layers1_traced_decode_retry.log`: L1 circular buffers grew to 2,946,464 B, above 1,499,136 B.
- `artifacts/smoke_num_layers1_traced_decode_retry2.log`: L1 circular buffers grew to 2,413,984 B, above 1,499,136 B.
- `artifacts/smoke_num_layers1_traced_decode_retry3.log`: L1 circular buffers grew to 1,881,504 B, above 1,499,136 B.
- `artifacts/smoke_num_layers1_traced_decode_retry5.log`: accepted config, capture token 499, replay token 499, match true.

## Tracy and tt-perf-report

Capture:

```bash
rm -rf models/autoports/google/gemma-4-12B/doc/optimized_full_model/tracy/full_model_2_layers
GEMMA4_12B_FULL_MODEL_PROFILE_LAYERS=2 python -m tracy -r -p -v \
  --output-folder models/autoports/google/gemma-4-12B/doc/optimized_full_model/tracy/full_model_2_layers \
  -m pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_optimized_full_model.py::test_optimized_full_model_prefill_and_traced_decode_profile \
  --tb=short --timeout=1200 \
  2>&1 | tee models/autoports/google/gemma-4-12B/doc/optimized_full_model/artifacts/tracy_full_model_2_layers.log
```

Stable ops CSV:

```bash
cp models/autoports/google/gemma-4-12B/doc/optimized_full_model/tracy/full_model_2_layers/reports/2026_06_09_02_49_25/ops_perf_results_2026_06_09_02_49_25.csv \
  models/autoports/google/gemma-4-12B/doc/optimized_full_model/tracy/full_model_2_layers/ops.csv
```

Report commands:

```bash
tt-perf-report models/autoports/google/gemma-4-12B/doc/optimized_full_model/tracy/full_model_2_layers/ops.csv \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
  --csv models/autoports/google/gemma-4-12B/doc/optimized_full_model/tracy/full_model_2_layers/prefill_perf_report.csv \
  --summary-file models/autoports/google/gemma-4-12B/doc/optimized_full_model/tracy/full_model_2_layers/prefill_summary.csv

tt-perf-report models/autoports/google/gemma-4-12B/doc/optimized_full_model/tracy/full_model_2_layers/ops.csv \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
  --no-summary > models/autoports/google/gemma-4-12B/doc/optimized_full_model/tracy/full_model_2_layers/prefill_perf_report.txt

tt-perf-report models/autoports/google/gemma-4-12B/doc/optimized_full_model/tracy/full_model_2_layers/ops.csv \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --csv models/autoports/google/gemma-4-12B/doc/optimized_full_model/tracy/full_model_2_layers/decode_perf_report.csv \
  --summary-file models/autoports/google/gemma-4-12B/doc/optimized_full_model/tracy/full_model_2_layers/decode_summary.csv

tt-perf-report models/autoports/google/gemma-4-12B/doc/optimized_full_model/tracy/full_model_2_layers/ops.csv \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --no-summary > models/autoports/google/gemma-4-12B/doc/optimized_full_model/tracy/full_model_2_layers/decode_perf_report.txt
```

Findings:

- Prefill signpost: 72 device ops, 0 host ops, 5,498 us device time, 3,766 us op-to-op gap.
- Decode signpost: 111 device ops, 0 host ops, 7,997 us device time, 1,375 us op-to-op gap.
- Decode is dominated by on-device argmax, final logits all-gather, and the DRAM-sharded BFP4 LM-head matmul.
- The decode window is already trace replay. The report's trace-savings advice is treated as a merged-device/op-gap attribution limitation, not a hidden host fallback.

## Runtime Fallback Audit

Measured optimized decode path:

- `generator.decode_forward(..., enable_trace=True, sample_on_device=True, return_ttnn=True)`
- Trace captures full decode forward, final norm, decode LM head, softcap path, logits movement needed by sampling, and on-device greedy argmax.
- The replay result remains a TTNN token tensor; no full logits are read to host and no host argmax is used in the signposted decode path.

Named non-measured or accepted host boundaries:

- Model construction, weights, RoPE, page table, and cache tensors are setup-time host-to-device conversions.
- Per-token trace replay refreshes scalar token and position tensors into preallocated device input tensors with `ttnn.copy_host_to_device_tensor`.
- Readiness checks and qualitative generation convert tensors to torch for comparison.
- High-level `generate()` uses CPU logits plus non-EOS special-token suppression; this is an explicit production compromise for the current text-quality path because raw on-device argmax cannot apply that mask.

## Final Status

The optimized full-model state is complete for the repo-local TTNN autoport pipeline:

- Full model runs all major pieces across TP8 T3K, not just the decoder layer.
- Prefill and teacher-forcing decode pass AIME24 top-k readiness at the requested bar.
- Warmed before/after performance is recorded.
- Full-wrapper traced decode with on-device sampling is implemented and measured.
- Watcher and profiler artifacts exist.
- Remaining host boundaries and profiler limitations are named above.
