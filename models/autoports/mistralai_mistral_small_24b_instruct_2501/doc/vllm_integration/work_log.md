# vLLM integration work log

## Implementation

The adapter was added at `tt/generator_vllm.py` and the model was registered by `register_tt_models()` in the shared TT plugin. The adapter delegates construction, prefill, decode, sampling, trace management, cache use, and output conversion to `tt/generator.py`. It declares async decode and on-device sampling capabilities. The plugin also registers the exact Mistral tokenizer/renderer needed by the upstream model's Tekken chat template.

The adapter enforces the selected datatype-sweep configuration by constructing `build_generator()` without an adapter-local policy. It accepts only vLLM's cache object and forwards scheduler page tables and positions. Optional host sampling is guarded by `MISTRAL_SMALL_24B_VLLM_HOST_SAMPLING_COMPAT=1`; all production benchmarks used `sample_on_device_mode=all` without that variable.

## Debugging and regression evidence

Initial realistic chat prompts produced a correct prefill token followed by corrupt decode IDs. Raw-ID probes and model-versus-sampler argmax checks proved the sampler and tokenizer were not creating the corruption: traced model logits were already wrong while positions advanced correctly. The root cause was startup warmup executing a position-0 decode into vLLM's real BFP8 serving cache. The later real prefill rewrote already-quantized cache tiles, corrupting decode.

Warmup no longer mutates serving state. `allocate_kv_cache(..., shared_across_layers=False)` can construct one exact-geometry K/V pair shared across layers for disposable compile-only warmup. Trace preparation compiles against that scratch cache, deallocates it, records the trace against the vLLM cache without replay, recopies serving token/position/page state, and leaves the first actual decode as the first execution. `warmup_model_decode()` resets vLLM's cache after preparation. Final raw-ID generation was coherent and model argmax matched sampled output.

Row reuse also exposed stale physical block IDs in the unused tail of plugin page tables. `block_tables_for_rows()` now masks every column beyond the row's logical block count with `-1`, with a regression test. Padded inactive prefill rows no longer populate scheduler-owned cache state.

Focused final adapter/tokenizer/routing suite:

```text
46 passed
  model adapter tests
  corrected checkpoint tokenizer tests
  device-sampling limit/routing tests
```

The final full compatibility profile passed **72 tests with one expected skip in 504.80 seconds**. Its last issue was unseeded top-k rerun variety: the model-specific traced `Sampling1D` intentionally replays fixed slot seeds, so 7/8 identical request pairs repeated. Under the explicit compatibility environment only, stochastic batches now use vLLM's persistent host RNG; production remains on-device and traced. A deliberate production-server smoke rejected unsupported top-k 100, explicit seed/logprobs, min-p, and penalties before EngineCore admission. A supported request and `/health` passed after every rejection (`unsupported_sampling_survival.json`).

The review-requested full-logit discriminator ran the full 40-layer selected policy through `generator_vllm`, reading the production traced model-logit buffer only after replay for diagnostic comparison. Fresh-cache run-to-run and batch-position 0/1 tensors were bit-exact (max absolute difference 0, PCC 1.0, identical SHA256 and argmax 1278). The standalone eager full-model decode baseline retained argmax 1278 with PCC 0.9998519 and max BF16-path difference 0.375. From the repository root, the command was `TT_MISTRAL_TEXT_VER=mistral_small_24b_autoport HF_HUB_OFFLINE=1 python_env/bin/python models/autoports/mistralai_mistral_small_24b_instruct_2501/doc/vllm_integration/run_logit_determinism.py`; evidence is `readiness_vllm/logit_determinism.json`. This diagnostic readback is not present in production serving.

The shared readiness runner was narrowed after independent review: qualitative requests use chat completions only when the loaded checkpoint tokenizer declares a chat template, otherwise the prior text-completion behavior is retained. `fix_mistral_regex` is passed only for this exact Mistral checkpoint family. A focused helper regression covers both the Mistral and unrelated-model cases.

## Serving and evaluation commands

Production server:

```bash
TT_MISTRAL_TEXT_VER=mistral_small_24b_autoport \
PYTHONPATH="$PWD/vllm/plugins/vllm-tt-plugin/src:$PWD/vllm:$PWD" \
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages serve --model-dir models/autoports/mistralai_mistral_small_24b_instruct_2501 \
  --hf-model mistralai/Mistral-Small-24B-Instruct-2501 \
  --mesh-device P300x2 --max-num-seqs 32 --block-size 32 --max-model-len 32768 \
  --tt-config '{"sample_on_device_mode":"all","trace_region_size":200000000,"fabric_config":"FABRIC_1D"}'
```

Qualitative, primary benchmark, and CI burst were run through `models.common.readiness_check.run_vllm_server` against `http://127.0.0.1:8000`. The exact raw benchmark command strings are embedded in `vllm_benchmark.json` and `vllm_ci_serving_benchmark.json`. Sampling compatibility used the same server command with `MISTRAL_SMALL_24B_VLLM_HOST_SAMPLING_COMPAT=1`, followed by:

```bash
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages sampling --sampling-profile full \
  --server-url http://127.0.0.1:8000 \
  --model-dir models/autoports/mistralai_mistral_small_24b_instruct_2501 \
  --hf-model mistralai/Mistral-Small-24B-Instruct-2501
```

## Results

- Primary 128/128/1, concurrency 1, temperature 0: TTFT P50/P99 568.942/568.942 ms; TPOT mean/P50/P99 18.928/18.928/18.928 ms; ITL P50/P99 17.762/19.390 ms; 43.053 output tokens/s; TPOT-derived 52.833 tokens/s/user; 1/1 complete.
- CI burst 100/100/32, concurrency 32, temperature 0: TTFT P50/P99 1191.704/1192.844 ms; TPOT mean/P50/P99 19.618/19.419/23.825 ms; ITL P50/P99 17.935/67.958 ms; 1026.974 output tokens/s; 32/32 complete. Its TPOT-derived 50.973 tokens/s/user is secondary only.
- Non-aligned direct serving check: 37/37 prompt tokens observed, page size 32, completion returned, pass.
- Qualitative: 12/12 manually reviewed generations coherent and on-topic; no repetition loop, gibberish, wrong-language drift, request contamination, tokenizer markers, or mojibake. The sampled thermodynamics response has a minor count ambiguity documented in the verdict.

## Runtime and cleanup audit

Server ownership was checked with `pgrep -af 'run_vllm_server|vllm.entrypoints|EngineCore'`; hardware visibility was checked with `/home/mvasiljevic/.ttsmi-venv/bin/tt-smi -ls --local`. The final compatibility server was stopped through its owning readiness-runner process and exited cleanly. No serving process remained and all four Blackhole boards listed. The first bounded 1x4 mesh reopen then hit the recurring device-0 ERISC heartbeat timeout; with ownership clear, one bounded `tt-smi -r` restored the mesh, after which all four boards listed and a 1x4 open/close smoke passed. Earlier in bringup, one reset was mistakenly attempted before the prior runner had fully released ownership; the runner was then stopped explicitly and all subsequent recovery followed the required ownership check. No runtime fallback was used in performance measurements and no vLLM/API/EngineCore process remains.

## Commits

Stage-owned plugin and root commit SHAs are recorded here after final review and cleanup.
