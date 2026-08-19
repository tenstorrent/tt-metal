# Full Model Work Log

## 2026-08-19 - watcher sampling gate

- Investigated watcher abort in traced token-out smoke.
- Root-caused failure to `ttnn.all_gather` native minimal-default writer in `models/common/sampling/tt_sampling.py::_perform_all_gather`.
- Added opt-in `use_composite_topk_all_gather` sampling path that converts top-k gather inputs to row-major, uses composite all-gather, then tilizes back before `ttnn.sampling`.
- Enabled the option only for the Qwen full-model sampling args.
- Verified the original watcher gate passes and fallback-guarded synthetic tests still pass.

Reports:

- `models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/AUTOTRIAGE.md`
- `models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/AUTOFIX.md`

## 2026-08-19 - full model implementation and signoff

- Added `tt/model.py::QwenFullModel` with replicated BF16 embedding/final
  RMSNorm, BF8 flat 4-way vocab-sharded LM head, full 40-layer
  `MultichipDecoder` stack, paged BF16 full-attention KV cache, BF16 linear
  state, and explicit cache/page-table/prompt-length/position APIs.
- Added `tt/generator.py::QwenReadinessGenerator` and `build_generator` for the
  standard Metal readiness generator contract.
- Kept the completed optimized multichip decoder policy: `2x2` p300c mesh,
  TP over columns, EP over rows, two-link BF16 Ring collectives, BF16 residual
  boundaries, paged BF16 cache, inherited weight dtypes/fidelity, and the
  rejected-candidate ledger from the optimized multichip decoder stage.
- Implemented public prefill chunking with 64-token windows, non-aligned prompt
  slicing, fixed-slot page-table row selection, mixed prompts, and inactive row
  handling.
- Implemented traced greedy token-out decode with on-device first-token sampling
  from prefill logits, separate persistent TT decode-input and sampler-output
  token buffers, persistent TT current-position tensor, and common on-device
  sampler inside the captured decode body.
- Preserved an explicit `host_sampling_compat` mode for readiness checks and
  tests requiring host sampling.
- Updated `models/common/readiness_check/mesh_device.py` so readiness tools can
  open the local `P300C` `2x2` mesh and pass a larger trace region through
  `TT_READINESS_TRACE_REGION_SIZE`.
- Updated `models/common/readiness_check/generate.py` to handle tokenizer
  mapping/tensor returns when generating the fresh chat-template reference.
- Added `tests/test_full_model.py` for exports, fallback-audit strings,
  non-aligned prompt, fixed-slot mixed prompt with inactive row, and traced
  token-out greedy comparison.

Fresh reference:

```bash
timeout 14400 ./python_env/bin/python -m models.common.readiness_check.generate \
  --hf-model Qwen/Qwen3.6-35B-A3B \
  --prompt-source aime24 --chat-template --gen-len 100 --top-k 100 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/artifacts/aime24_chat_100.refpt
```

Result: `161` prompt tokens and `100` continuation tokens.

Correctness:

```bash
timeout 3600 env TT_METAL_WATCHER_DISABLE_ETH=1 \
  ./python_env/bin/python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/qwen_qwen3_6_35b_a3b \
  --reference models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/artifacts/aime24_chat_100.refpt \
  --mesh-device P300C --fabric-config FABRIC_1D_RING
```

Result: top-1 `96/100`, top-5 `100/100`, top-100 `100/100`.

```bash
timeout 3600 env TT_METAL_WATCHER_DISABLE_ETH=1 \
  ./python_env/bin/python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/qwen_qwen3_6_35b_a3b \
  --reference models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/artifacts/aime24_chat_100.refpt \
  --mesh-device P300C --fabric-config FABRIC_1D_RING
```

Result after traced teacher-forcing remediation: top-1 `99/100`, top-5
`100/100`, top-100 `100/100`, TTFT `8747.52 ms`, decode `16.35 t/s/u`,
end-to-end `6.76 t/s/u`.

Autoregressive:

```bash
timeout 7200 env TT_METAL_WATCHER_DISABLE_ETH=1 TT_READINESS_TRACE_REGION_SIZE=64000000 \
  ./python_env/bin/python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/qwen_qwen3_6_35b_a3b \
  --hf-model Qwen/Qwen3.6-35B-A3B \
  --prompt-file models/common/readiness_check/autoregressive_prompt.txt \
  --mesh-device P300C --fabric-config FABRIC_1D_RING \
  --max-new-tokens 100 \
  --output-dir models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/artifacts/autoregressive_default_prompt_100
```

Result: HF and TT both produced `100` tokens. TT output is coherent English, no
repetition loop, no wrong-language drift, and no early duplicate-token failure.

Optimized token-out timing:

```bash
timeout 7200 env TT_METAL_WATCHER_DISABLE_ETH=1 TT_READINESS_TRACE_REGION_SIZE=64000000 \
  ./python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/scripts/measure_token_out_trace.py \
  --mesh-device P300C --fabric-config FABRIC_1D_RING --max-new-tokens 100
```

Result from `logs/token_out_trace_perf_default_prompt_100.log`: TTFT
`2930.03 ms`, `99` traced decode tokens, decode `16.42 t/s/u`, end-to-end
`11.16 t/s/u`, trace present, final exclusive position `159`.

Capacity:

- Recomputed per-device transformed TT weights: `15,985,073,536` bytes.
- Recomputed per-device runtime state at context 262144: `2,768,207,872` bytes.
- Recomputed per-device total: `18,753,281,408` bytes (`17.4654 GiB`).
- Real load plus full-context cache allocation passed in
  `logs/real_full_model_load_context_alloc.log`.
- `doc/context_contract.json` keeps advertised context `262144`; no hard
  physical limit was hit.

Final synthetic gates after the mixed-row fix:

```bash
./python_env/bin/python -m py_compile \
  models/common/sampling/tt_sampling.py \
  models/autoports/qwen_qwen3_6_35b_a3b/tt/model.py \
  models/autoports/qwen_qwen3_6_35b_a3b/tt/generator.py \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_full_model.py \
  models/common/readiness_check/generate.py \
  models/common/readiness_check/mesh_device.py
```

Result: passed.

```bash
./python_env/bin/python -m pytest -q \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_full_model.py
```

Result: `2 passed, 4 skipped`.

```bash
timeout 1800 env TT_METAL_WATCHER_DISABLE_ETH=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  RUN_QWEN36_FULL_MODEL_SMOKE=1 \
  ./python_env/bin/python -m pytest -q \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_full_model.py
```

Result: `6 passed, 2 warnings in 46.56s`.

```bash
timeout 1800 env TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  RUN_QWEN36_FULL_MODEL_SMOKE=1 \
  ./python_env/bin/python -m pytest -q \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_full_model.py
```

Result: `6 passed, 2 warnings in 50.05s`.

```bash
tt-smi -ls --local
```

Result: four local Blackhole p300c devices visible and resettable after the
watcher run.

Documentation added:

- `README.md`
- `runtime_fallback_audit.md`
- `sampling_trace_audit.md`
- `capacity.md`

## 2026-08-19 - stage-review remediation

Initial full-model stage review returned `more-work-needed` with four findings:

- teacher-forcing did not actually run traced decode when `enable_trace=True`;
- supplied decode page-table overrides were ignored when the caller also passed
  a cache object;
- the shared qualitative/degen suite was missing;
- terminal-path profiling was missing.

Fixes:

- Added traced teacher-forcing generation. The trace owns decode and sampling;
  the host callback only overwrites the next reference token between trace
  replays.
- Added `_cache_with_page_table()` so prefill/decode honor supplied page-table
  overrides without losing caller-owned cache state.
- Added changed-page-table synthetic coverage that verifies changed mappings
  alter decode logits and compile under trace capture.
- Fixed traced free-running feedback after the qualitative suite exposed token
  id 0 repetition on prompt 1. The trace now keeps separate buffers: 1-wide
  decode input and tile-width sampler output, with an in-trace device copy of
  sampled output slot 0 back to the decode input.
- Added `scripts/run_qualitative_chat_suite.py` and generated HF-tokenizer
  chat-template artifacts for the six shared prompts.
- Added `scripts/profile_terminal_path.py` and Blackhole-normalized
  `tt-perf-report` summaries for LM head, sampler, and terminal path.

Qualitative rerun:

```bash
timeout 7200 env TT_METAL_WATCHER_DISABLE_ETH=1 TT_READINESS_TRACE_REGION_SIZE=64000000 \
  ./python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/scripts/run_qualitative_chat_suite.py \
  --mesh-device P300C --fabric-config FABRIC_1D_RING --max-new-tokens 64
```

Result: prompts 0-4 matched HF for `64/64` generated tokens; prompt 5 first
diverged at generated token 44 but stayed coherent on the Python/Fibonacci
task. The degeneration checker reported no findings.

Autoregressive rerun after the split-buffer trace fix:

```bash
timeout 7200 env TT_METAL_WATCHER_DISABLE_ETH=1 TT_READINESS_TRACE_REGION_SIZE=64000000 \
  ./python_env/bin/python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/qwen_qwen3_6_35b_a3b \
  --hf-model Qwen/Qwen3.6-35B-A3B \
  --mesh-device P300C --fabric-config FABRIC_1D_RING \
  --max-new-tokens 100 \
  --output-dir models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/artifacts/autoregressive_default_prompt_100
```

Result: HF and TT both produced `100` tokens. TT output is coherent English and
the degeneration checker reported no findings; HF/TT token agreement is
informational at `14/100` because free-running greedy output naturally diverges.

Terminal-path profiling:

```bash
timeout 7200 env TT_METAL_WATCHER_DISABLE_ETH=1 TT_READINESS_TRACE_REGION_SIZE=64000000 \
  ./python_env/bin/python -m tracy -r -p -v --no-runtime-analysis \
  --op-support-count=5000 --check-exit-code \
  --output-folder models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/tracy/terminal_path_raw \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/scripts/profile_terminal_path.py \
  --mesh-device P300C --fabric-config FABRIC_1D_RING
```

Result: final norm plus LM head `0.510 ms`, sampler `10.938 ms`, terminal
subpath `11.464 ms`. Blackhole-normalized `tt-perf-report` shows
`TopKDeviceOperation` at `10,608 us`. The sampler is `18.0%` of the measured
full traced token-out decode step (`60.891 ms/token`), so it does not dominate
token-out decode.

Stage review:

- Final verdict: `clean-pass`.
- Review artifact: stage-review subagent `01a019fe-bb4f-77e2-9669-be816d42d0c0`.
