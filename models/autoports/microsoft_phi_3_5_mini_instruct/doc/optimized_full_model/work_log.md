# Optimized Full Model Work Log

Date: 2026-06-15

Goal: optimize the completed Phi-3.5-mini full model/generator path on the target 1x8 mesh without starting vLLM.

## Baseline Inventory

Current full model entrypoints:

- `tt/model.py`
- `tt/generator.py`

Prior completed full-model status:

- AIME24 teacher forcing: TTFT 221.54 ms, decode 36.88 t/s/u.
- Token-out greedy story path: TTFT 265.98 ms, decode 38.37 t/s/u.
- AIME24 prefill: top-1 96/100, top-5 100/100, top-100 100/100.
- AIME24 traced teacher forcing: top-1 91/100, top-5 100/100, top-100 100/100.

Optimized decoder lower bound from `doc/optimized_multichip_decoder`:

- Device layer latency: 543.090 us.
- Host traced layer latency: 559.258 us.
- 32-layer device stack lower bound: 17.37888 ms/token.
- 32-layer host traced stack lower bound: 17.896256 ms/token.

Hardware:

- `tt-smi -ls --local` showed 8 Wormhole devices before and after profiler/watcher runs.

## Code Changes

`tt/generator.py`:

- Added `benchmark_token_out_decode`.
- Added no-readback traced decode mode in `_decode_next_token_traced(readback=False)`.
- Added `TraceCounters.no_readback_decode_steps`.
- Preserved normal `generate()` behavior with sampled-token readback for readiness return contracts.
- Kept `tt_out_tok` feedback into the persistent trace token input.

`tt/model.py`:

- Changed LM-head padded vocab contract to `per_device_vocab_size=max(8192, next_power_of_two(tile_aligned_vocab_per_device))`.
- For Phi-3.5 mini this is per-device 8192, total padded vocab 65536.
- Left decoder stack policy unchanged.

`tests/test_optimized_full_model.py`:

- Added static strategy/fallback checks.
- Added opt-in one-layer smoke/profile tests for token-out no-readback and reduced full-path profiling.

## Commands And Results

Static checks:

```bash
python -m py_compile models/autoports/microsoft_phi_3_5_mini_instruct/tt/model.py \
  models/autoports/microsoft_phi_3_5_mini_instruct/tt/generator.py \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_full_model.py

pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_full_model.py::test_full_model_strategy_static \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_full_model.py::test_token_out_no_readback_runtime_fallback_audit_static -s
```

Result:

- `2 passed`
- Artifact: `logs/final_static_fallback_audit.log`

Final prefill:

```bash
python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/microsoft_phi_3_5_mini_instruct \
  --reference models/autoports/microsoft_phi_3_5_mini_instruct/readiness_aime24_chat_100.refpt \
  --mesh-device T3K --fabric-config FABRIC_1D_RING
```

Result:

- top-1 96/100
- top-5 100/100
- top-100 100/100
- Artifact: `logs/final_run_prefill_check_2026_06_15.log`

Final teacher forcing:

```bash
python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/microsoft_phi_3_5_mini_instruct \
  --reference models/autoports/microsoft_phi_3_5_mini_instruct/readiness_aime24_chat_100.refpt \
  --mesh-device T3K --fabric-config FABRIC_1D_RING
```

Result:

- top-1 91/100
- top-5 100/100
- top-100 100/100
- TTFT 226.93 ms
- decode 40.15 t/s/u
- e2e 37.13 t/s/u
- Artifact: `logs/final_run_teacher_forcing_2026_06_15.log`

Final autoregressive:

```bash
python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/microsoft_phi_3_5_mini_instruct \
  --hf-model microsoft/Phi-3.5-mini-instruct \
  --mesh-device T3K --fabric-config FABRIC_1D_RING \
  --max-new-tokens 128

python -m models.common.readiness_check.check_degenerate_output \
  --model-dir models/autoports/microsoft_phi_3_5_mini_instruct
```

Result:

- TT produced 128 tokens.
- Degeneracy scan: no degenerate output detected.
- Artifacts:
  - `logs/final_run_autoregressive_2026_06_15.log`
  - `logs/final_check_degenerate_output_2026_06_15.log`
  - `../../readiness_autoregressive/tt_completion.txt`

## Sampler Strategy Trials

Baseline no-readback token-out:

```bash
python - <<'PY'
# build_generator(...); benchmark_token_out_decode(prompt128, max_new_tokens=128, warmup_decode_steps=8)
PY
```

Result:

- Artifact: `perf/token_out_no_readback_prompt128_gen128.json`
- TTFT 227.151 ms
- decode 50.520 t/s/u
- 19.794 ms/token
- e2e 35.715 t/s/u
- counters: 0 sampled-token readbacks, 0 full-logit decode readbacks, 135 `tt_out_tok` feedback steps.

8192 selected no-readback token-out:

```bash
python - <<'PY'
# same prompt128/gen128 benchmark after LM-head per-device vocab 8192
PY
```

Result:

- Artifact: `perf/token_out_no_readback_prompt128_gen128_lmhead8192.json`
- TTFT 254.465 ms
- decode 56.431 t/s/u
- 17.721 ms/token
- e2e 38.821 t/s/u
- counters: 0 sampled-token readbacks, 0 full-logit decode readbacks, 135 `tt_out_tok` feedback steps.

Rejected sampler contracts:

- Explicit sampler pad 4032 to 4096:
  - `logs/token_out_no_readback_1layer_pad_on_warmed_64.log`
  - decode 293.05 t/s/u
- No explicit pad at 4032:
  - `logs/token_out_no_readback_1layer_pad_off_warmed_64.log`
  - decode 299.60 t/s/u
- LM-head padded to 4096:
  - `logs/token_out_no_readback_1layer_lmhead4096_warmed_64.log`
  - decode 296.71 t/s/u
  - reduced profile showed TopK still 1 core and 2290.80 us.

Accepted sampler contract:

- LM-head padded to 8192:
  - `logs/token_out_no_readback_1layer_lmhead8192_warmed_64.log`
  - one-layer decode 791.17 t/s/u
  - full 32-layer decode 56.43 t/s/u

## Profiling

Initial 4032 reduced profile:

```bash
PHI35_RUN_OPTIMIZED_FULL_MODEL_PERF=1 PHI35_PROFILE_MEASURED_DECODE_STEPS=1 \
PHI35_PROFILE_WARMUP_DECODE_STEPS=8 \
python -m tracy -r -p -v \
  -o models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_full_model/tracy/reduced_1layer \
  -m pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_full_model.py::test_reduced_full_model_token_out_profile_1x8_ring -s
```

Key result:

- Reduced token-out device time 3210.51 us.
- TopK 2255.32 us, 70.25%, 1 core.
- `perf/reduced_1layer_token_out_perf_summary.csv.csv`

Selected 8192 reduced profile:

```bash
PHI35_RUN_OPTIMIZED_FULL_MODEL_PERF=1 PHI35_PROFILE_MEASURED_DECODE_STEPS=1 \
PHI35_PROFILE_WARMUP_DECODE_STEPS=8 \
python -m tracy -r -p -v \
  -o models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_full_model/tracy/reduced_1layer_lmhead8192 \
  -m pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_full_model.py::test_reduced_full_model_token_out_profile_1x8_ring -s
```

Key result:

- Reduced token-out device time 1168.96 us.
- TopK 99.11 us, 8.48%, 17 cores.
- Sampling 63.61 us.
- LM-head matmul `32 x 3072 x 8192`: 207.993 us.
- `perf/reduced_1layer_lmhead8192_token_out_perf_report.csv`
- `perf/reduced_1layer_lmhead8192_token_out_perf_summary.csv.csv`
- `perf/reduced_1layer_lmhead8192_token_out_perf_table.txt`

`tt-perf-report` command:

```bash
tt-perf-report tracy/reduced_1layer_lmhead8192/reports/2026_06_15_18_18_41/ops_perf_results_2026_06_15_18_18_41.csv \
  --start-signpost PERF_FULL_TOKEN_OUT \
  --end-signpost PERF_FULL_TOKEN_OUT_END \
  --csv perf/reduced_1layer_lmhead8192_token_out_perf_report.csv \
  --summary-file perf/reduced_1layer_lmhead8192_token_out_perf_summary.csv \
  --no-color
```

## Lower Bound Accounting

Final token-out measured decode:

- 56.4306 t/s/u
- 17.7209 ms/token

Lower bound:

- 32 * 543.090 us = 17.37888 ms device stack.
- 32 * 559.258 us = 17.896256 ms host traced stack.
- Selected reduced one-layer full-path device time = 1168.96 us.
- Estimated terminal device work = 1168.96 - 543.09 = 625.87 us.
- Stack device lower bound + terminal estimate = 18.00475 ms/token.

Conclusion:

- The selected token-out path is within the lower-bound plus measured terminal-work envelope.
- The old generic TopK-dominated sampler gap was closed before completion.

## Watcher

Full 32-layer token-out watcher smoke:

```bash
RUN=2026_06_15_full_token_out_watcher10_lmhead8192
BASE=models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_full_model/watcher/$RUN
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_APPEND=1 \
TT_METAL_LOGS_PATH="$BASE" \
python - <<'PY'
# build full generator and run benchmark_token_out_decode(prompt128, max_new_tokens=16, warmup_decode_steps=2)
PY
```

Result:

- Full stack ran with no sampled-token readbacks and no full-logit decode readbacks.
- Watcher detached all eight devices.
- Watcher disabled features: None.
- Ethernet retraining events: 0.
- Minimum reported free stack: 416 bytes on TRISC0 in `sdpa.cpp`.
- Error scan had no matches:
  `watcher/2026_06_15_full_token_out_watcher10_lmhead8192/watcher_scan.log`

## Limitations

- TTFT for the prompt128 token-out benchmark regressed because the padded LM head doubles the vocab matmul width. The warmed decode win is larger and places token-out near the decoder-stack lower bound.
- Force-argmax was not enabled. The final greedy path is canonical split sampling and remains top-k/top-p-capable.
- The current CPU HF greedy autoregressive reference produced repetitive text under the local transformers path, so autoregressive quality is judged from the TT completion plus the TT degeneracy scan.
- Prefill is not trace-optimized in this stage.
- No broad full-model datatype frontier search was run.
- vLLM integration was not started.

