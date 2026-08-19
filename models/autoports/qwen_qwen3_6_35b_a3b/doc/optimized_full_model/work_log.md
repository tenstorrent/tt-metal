# Optimized Full Model Work Log

## 2026-08-19

Scope: optimize the completed Qwen full-model/generator path on the local
`2x2` Blackhole p300c mesh. No vLLM work was started.

Implementation:

- Added `QwenReadinessGenerator.measure_token_out_no_readback()` for
  serving-style traced token-out decode. The hot loop enqueues nonblocking
  trace replay without per-token sync/readback, token refresh, position refresh,
  RoPE refresh, or page-table rebuild.
- Kept the no-readback measurement cache at least as large as the model's
  prefill chunk so one-page synthetic cache sizing does not perturb traced
  sampler feedback state.
- Added `last_trace_counters` instrumentation to the traced token-out and
  traced teacher-forcing paths so the measured host boundaries are explicit.
- Kept the completed full-model split-sampling contract: sampler-ready sharded
  logits, `tt_out_tok` feedback, persistent token and position tensors,
  device-side RoPE advance, changed-only page-table handling, traced greedy
  sampling, and top-k/top-p-capable common sampler plumbing.
- Kept the optimized decoder stack policy: TP=2 over mesh columns, EP=2 over
  mesh rows, BF16 Ring collectives with `num_links=2`, BF16 residual layout,
  paged BF16 KV cache, BF16 linear state, and inherited dtype/fidelity
  decisions. No datatype frontier search was run.
- Added `tests/test_full_model.py::test_full_model_token_out_no_readback_measurement_smoke`.
- Added scripts for prompt-128/gen-128 no-readback measurement and terminal
  sampler-choice profiling.

Device setup:

```bash
timeout 60 tt-smi -ls --local \
  > models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/logs/tt_smi_initial.log
```

Result: four local Blackhole p300c devices visible.

```bash
timeout 300 env TT_METAL_WATCHER_DISABLE_ETH=1 ./python_env/bin/python - <<'PY'
import ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
PY
```

Result: mesh open/close smoke passed in `logs/mesh_open_smoke.log`.

```bash
tt-smi -s \
  > models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/logs/tt_smi_final.log
```

Result: final post-benchmark device snapshot passed; all four p300c devices
reported DRAM OK with no corrected or uncorrected GDDR errors.

Correctness and contract gates:

```bash
./python_env/bin/python -m py_compile \
  models/autoports/qwen_qwen3_6_35b_a3b/tt/generator.py \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/scripts/measure_token_out_no_readback.py \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/scripts/profile_terminal_sampler_choices.py \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_full_model.py
```

Result: passed.

```bash
./python_env/bin/python -m pytest -q \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_full_model.py
```

Result: `2 passed, 5 skipped`.

```bash
timeout 1800 env TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  RUN_QWEN36_FULL_MODEL_SMOKE=1 \
  ./python_env/bin/python -m pytest -q \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_full_model.py
```

Result: `7 passed, 2 warnings` in `logs/hardware_smokes_watcher_final.log`.

```bash
timeout 1800 env TT_METAL_WATCHER_DISABLE_ETH=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  RUN_QWEN36_FULL_MODEL_SMOKE=1 \
  ./python_env/bin/python -m pytest -q \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_full_model.py
```

Result: `7 passed, 2 warnings` in
`logs/synthetic_full_model_no_fallback_smoke_final.log`.

AIME24 checks:

```bash
timeout 3600 env TT_METAL_WATCHER_DISABLE_ETH=1 ./python_env/bin/python -m \
  models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/qwen_qwen3_6_35b_a3b \
  --reference models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/artifacts/aime24_chat_100.refpt \
  --mesh-device P300C --fabric-config FABRIC_1D_RING
```

Result: top-1 `96/100`, top-5 `100/100`, top-100 `100/100`.

```bash
timeout 3600 env TT_METAL_WATCHER_DISABLE_ETH=1 TT_READINESS_TRACE_REGION_SIZE=64000000 \
  ./python_env/bin/python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/qwen_qwen3_6_35b_a3b \
  --reference models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/artifacts/aime24_chat_100.refpt \
  --mesh-device P300C --fabric-config FABRIC_1D_RING
```

Result: top-1 `99/100`, top-5 `100/100`, top-100 `100/100`,
TTFT `8749.84 ms`, decode `16.38 t/s/u`, e2e `6.76 t/s/u`.

Autoregressive:

```bash
timeout 7200 env TT_METAL_WATCHER_DISABLE_ETH=1 TT_READINESS_TRACE_REGION_SIZE=64000000 \
  ./python_env/bin/python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/qwen_qwen3_6_35b_a3b \
  --hf-model Qwen/Qwen3.6-35B-A3B \
  --prompt-file models/common/readiness_check/autoregressive_prompt.txt \
  --mesh-device P300C --fabric-config FABRIC_1D_RING \
  --max-new-tokens 100 \
  --output-dir models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/artifacts/autoregressive_default_prompt_100
```

Result: HF and TT both produced `100` tokens.

```bash
timeout 300 ./python_env/bin/python models/common/readiness_check/check_degenerate_output.py \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/artifacts/autoregressive_default_prompt_100/autoregressive_meta.json \
  --scope autoregressive \
  --json models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/artifacts/autoregressive_default_prompt_100/degenerate_output_report.json
```

Result: no degenerate output detected; informational HF/TT token agreement
`14/100`.

Performance:

```bash
timeout 7200 env TT_METAL_WATCHER_DISABLE_ETH=1 TT_READINESS_TRACE_REGION_SIZE=64000000 \
  ./python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/scripts/measure_token_out_no_readback.py \
  --mesh-device P300C --fabric-config FABRIC_1D_RING \
  --prompt-len 128 --max-new-tokens 128 --include-readback-baseline \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/artifacts/token_out_no_readback_prompt128_gen128_warmed.json
```

Result: readback baseline decode `16.58 t/s/u`; optimized no-readback replay
decode `17.43 t/s/u`, e2e `9.52 t/s/u`, final token matched, no steady-state
host sync/readback.

```bash
timeout 3600 env TT_METAL_WATCHER_DISABLE_ETH=1 ./python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/scripts/profile_terminal_sampler_choices.py \
  --mesh-device P300C --fabric-config FABRIC_1D_RING --iterations 5
```

Result: common top-k1 composite gather selected, `10.901 ms`, matches host
argmax. Force-argmax async full-vocab all-gather was rejected because the local
2x2 p300c fabric could not route the requested full-mesh all-gather.

Lower-bound check:

- Optimized decoder screen latencies: linear decode `1.281 ms`, full decode
  `1.096 ms`.
- Decoder stack wall lower bound: `30 * 1.281 + 10 * 1.096 = 49.39 ms/token`.
- Inherited terminal path: `11.464 ms`.
- Stack plus terminal: `60.854 ms/token`.
- Optimized token-out replay: `57.359 ms/token`.

The measured optimized token-out replay is inside the stack-plus-terminal
envelope. Against the more aggressive tt-perf-report device-time stack plus
terminal (`55.721 ms/token`), the gap is `2.94%`, below the 10-15% threshold.

Profiler artifacts:

- copied optimized decoder final tt-perf-report tables to
  `tracy/inherited_optimized_multichip_decoder_final_reports/`;
- copied completed full-model terminal tt-perf-report tables to
  `tracy/inherited_full_model_terminal_path_reports/`;
- wrote summarized provenance to `artifacts/perf_summary.json`.

Stage review:

- Initial independent review: `more-work-needed`; finding was stale
  prompt-128/gen-128 no-readback artifact position metadata after the
  canonical position contract update.
- Rerun review: `clean-pass`, stage-review subagent
  `01a01aa9-8cf5-73f0-84f4-97a0df531484`.

Limitations:

- The prompt-128/gen-128 real-weight optimized measurement is batch 1.
- Batch-2, mixed prompt, fixed slot, inactive row, changed page-table, and
  non-aligned prompt support are covered by synthetic hardware/watcher tests.
- Long-context 262144-token behavior is preserved from the full-model context
  contract and inherited decoder evidence; no capability reduction was made.
- No vLLM integration was started.
