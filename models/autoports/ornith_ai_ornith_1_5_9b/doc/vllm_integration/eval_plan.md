# Final serving evaluation plan

Source/artifact inspection, 2026-11-13. No server requests, generation, or device operations were run for this plan. Run the commands below serially against the final full-32-layer, B32 server, after its owner confirms readiness. Preserve the final launch command, revisions and server log. Reduced diagnostic results are not final serving acceptance.

## P2 routes and the actual sampler boundary

The intended final defaults are `enable_thinking=false`, effective `top_k=20`, `sample_on_device_mode=all`, and `seeded_sampling_policy=auto`. Record these as operator overrides; they differ from the checkpoint's model-card qualitative settings. An explicit request overrides generation defaults.

| Installed request profile | Relevant request fields | Expected final route |
| --- | --- | --- |
| `plain` | temperature 0, no seed | Device greedy |
| `lm_eval_non_reasoning` | temperature 0, seed 42 | Device greedy |
| `lm_eval_reasoning` | temperature 1, top-p .95, top-k 40, seed 42 | Host; explicit 40 exceeds the device draw boundary |
| `agentic` | temperature 1, top-p .95, no top-k/seed | Device if effective server default is top-k 20; host if unrestricted top-k survives default resolution |
| `model_card` | resolved from `generation_config.json` | Pinned checkpoint has no generation_config.json; installed harness reports unresolved/skipped |
| Shared qualitative | temperature 0 or .7, sampled top-p .9, no seed/top-k | Device with effective top-k 20 |

The installed profile JSON is authoritative. Do not replace reasoning top-k 40 with 32 or 20 to pass the exact-profile gate. A separately named custom profile can diagnose performance but cannot replace P2. Acceptance of top-k 40 through the host path is correct; its performance ratio still needs measurement.

Source evidence:

- `tt/generator_vllm.py` declares `supports_sample_on_device=True`, `max_device_top_k=32`, and device penalties. `tt/generator.py` constructs the canonical sampler with `max_top_k=32` and rejects stochastic top-k outside 1..32.
- Plugin `src/vllm_tt_plugin/model_runner.py::check_perform_device_sampling` sends a step to host when any stochastic row has top-k <1 or >32. Seeded policy `host`, TP4 logprobs, min-p, and other host processors can independently select host. Intermediate prefill chunks deliberately return logits without sampling. Do not interpret those chunks as unexpected final-sampling fallback.
- `models/common/sampling/tt_sampling.py` has configurable candidate collection, but its final draw calls `ttnn.sampling`. In `ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp`, `MAIN` hard-codes `nearest32_K=32`, `logk=5`, and one output candidate tile. The writer in `kernels/dataflow/writer_interleaved.cpp` traverses two 16-wide faces and uses `k-16` as its second-face range. For k>32 it reads outside that row's candidate faces. A wider gathered input, `max_top_k=64`, or a larger declared capability does not make this draw safe. The current canonical sampler cannot safely claim >32; a common-kernel change and discriminating candidate-membership/statistical/trace tests would be required.
- Installed `runtime/readiness_check/run_harness_requests.py::predict_route` merges checkpoint generation_config defaults with the raw request. It does **not** merge `--override-generation-config` from server settings. It can predict host for agentic while the effective top-k20 request actually uses device. Preserve the predictor result and explain it with launch settings and observed sampling markers; do not lie through `--max-device-top-k` or edit the installed harness/profile.

`doc/full_model/serving_contract.json` is a scoped generator contract, not a P2 pass. Its top-k/device rows cite `batch32_full.json`, `sampled_trace_probe.json`, `sampling_transition2.json`; host compatibility cites `serving_decode_probe_final2.json` and `runtime_fallback_probe.json`. These establish B32 generator operation, limited-k sampling and independent writable host logits. Newer `doc/vllm_integration/original_boundary_device.json` verifies reduced-model exact seeds and four penalty tables across original-prompt-boundary transitions. `diagnostic4/host_logprobs.json` exercises a reduced server's host path. None measures final full-model P2 top-k40 TPOT or establishes final B32 agentic routing.

Actionable model-side work is to retain the truthful 32 boundary, verify the final top-k20 defaults through observed routes, and expose matched qualitative controls (implemented below). The adapter already supplies the host-logit path needed by top-k40. No runtime capability widening is justified by this inspection. If the full-model ratio fails, measure where host-logit transfer and sampling time are spent before proposing an implementation change.

## Exact final-server probes

Run from the tt-metal root using the prepared environment. Set `SERVER_LOG` to the actual full-model launch log if it differs. These commands attach to the existing server; they do not launch a competing device process.

```bash
export TT_MODEL_BRINGUP_ROOT=/home/ttuser/ornith-9b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24
export PYTHONPATH="$TT_MODEL_BRINGUP_ROOT/runtime:$PWD${PYTHONPATH:+:$PYTHONPATH}"
MODEL_DIR=models/autoports/ornith_ai_ornith_1_5_9b
HF_MODEL=ornith-ai/Ornith-1.5-9B
HF_REVISION=489cb97981b8654bcfcf30ce1f94ed1b62e07b53
SERVER_URL=http://localhost:8000
SERVER_LOG="$MODEL_DIR/readiness_vllm/full_tracked/server.log"

python_env/bin/python -m readiness_check.run_harness_requests \
  --server-url "$SERVER_URL" --model-dir "$MODEL_DIR" \
  --hf-model "$HF_MODEL" --revision "$HF_REVISION" --server-log "$SERVER_LOG" \
  --users 1,32 --load-users 32 --output-tokens 128

python_env/bin/python -m readiness_check.run_chat_conformance \
  --server-url "$SERVER_URL" --model-dir "$MODEL_DIR" \
  --hf-model "$HF_MODEL" --revision "$HF_REVISION" --server-log "$SERVER_LOG" \
  --trust-remote-code

python_env/bin/python -m readiness_check.run_long_context --part accuracy \
  --server-url "$SERVER_URL" --model-dir "$MODEL_DIR" --hf-model "$HF_MODEL" \
  --revision "$HF_REVISION" --server-log "$SERVER_LOG" --tt-metal "$PWD"

python_env/bin/python -m readiness_check.run_vllm_server \
  --model-dir "$MODEL_DIR" --hf-model "$HF_MODEL" --stages qualitative \
  --server-url "$SERVER_URL" --server-log "$SERVER_LOG"

python_env/bin/python "$TT_MODEL_BRINGUP_ROOT/scripts/check_eval_readiness.py" \
  --model-dir "$MODEL_DIR" --stage vllm
```

P2 uses back-to-back plain/profile batches at users 1 and 32; default ratio limit is 1.15. The 32-user ratio is the performance gate; 1-user is advisory. `accepted_lm_eval_reasoning` is nonwaivable. The harness caps generated output to the chosen probe budget and separately checks whether the original profile's 32768-token budget fits. Do not label 128-token timing as a completed 32768-token generation.

P3 defaults to all cases, including thinking/tools where detected; final default thinking-off does not waive explicit thinking-on coverage. Use supported built-in reasoning/tool parsers and preserve parser settings. P5 accuracy defaults to 8192, 32768, 131072 context positions (clamped to served capacity), depths .1/.5/.9, greedy and thinking-off. Record any clamping. Final readiness requires actual probe outputs and the gate check, not these source predictions. A limitation entry requires the skill's documented investigation/evidence budget; do not pre-waive anticipated host-route slowdown.

## Comparable qualitative controls

Pinned model/revision: `ornith-ai/Ornith-1.5-9B` at `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, tokenizer `Qwen2Tokenizer`, chat template present. The shared prompt source is `$TT_MODEL_BRINGUP_ROOT/runtime/readiness_check/vllm_prompts.txt`; all six prompts use one user message and `add_generation_prompt=True`.

Existing exact artifact inventory:

| Artifact relative to model root | Completed HF samples | Why it cannot serve as the final default control |
| --- | --- | --- |
| `doc/full_model/qualitative_shared_seed0/hf/prompt01_seed0.json` through `prompt06_seed0.json` | Six | Default thinking on; temperature1, top-p.95, top-k20, presence1.5; 256-token budget |
| `doc/full_model/qualitative_entropy_control/hf/prompt04_seed0.json` through `prompt04_seed9.json` | Ten, thermodynamics prompt only | Same thinking-on/model-card settings; not all six prompts |
| `doc/full_model/qualitative_shared/hf/` | Two complete of three existing files | Ten-seed configuration is incomplete; same incompatible settings |
| `doc/datatype_sweep/qualitative/hf/`, `doc/optimized_full_model/qualitative/` | Earlier controls/reuse metadata | Their `qualitative_prompt_format.json` also records thinking-on and model-card settings |

The `qualitative_prompt_format.json` beside each run records the actual controls. `doc/datatype_sweep/qualitative_control_reuse.json` points back to the two full-model source runs. There is no existing matched thinking-off, temperature .7/top-p .9/presence0 HF control in these artifacts. Reuse the earlier samples only for their original model-card comparison.

`tests/full_model_qualitative_suite.py` now accepts `--thinking default|on|off`, `--temperature`, `--top-p`, `--top-k`, and `--presence-penalty`. Defaults, old `--greedy` behavior, and the blind rubric are preserved. Explicit thinking settings are included in shared metadata and prompt rendering; sampling overrides enter the shared signature. Temperature0 selects greedy on both backends. A changed runner hash intentionally prevents silently reusing backend artifacts from a different script; use new output directories.

After the serving job releases the devices, generate HF and same-code TT controls serially. These commands use all six prompts and 256 tokens, seed0; the TT backend uses the current full model on the same 1x4 mesh. No reduced layer selector is applied. The owner has already launched the HF greedy command into the directory below; inspect its completion rather than duplicate that job.

```bash
CHECKPOINT=/home/ttuser/ornith-9b/hf/hub/models--ornith-ai--Ornith-1.5-9B/snapshots/489cb97981b8654bcfcf30ce1f94ed1b62e07b53
CONTROL_SCRIPT="$MODEL_DIR/tests/full_model_qualitative_suite.py"
for backend in hf tt; do
  python_env/bin/python "$CONTROL_SCRIPT" --backend "$backend" \
    --checkpoint "$CHECKPOINT" --seeds 0 --max-new-tokens 256 \
    --thinking off --temperature 0 --top-p 1 --top-k 20 --presence-penalty 0 \
    --output-dir "$MODEL_DIR/readiness_vllm/qualitative_control_greedy"
  python_env/bin/python "$CONTROL_SCRIPT" --backend "$backend" \
    --checkpoint "$CHECKPOINT" --seeds 0 --max-new-tokens 256 \
    --thinking off --temperature .7 --top-p .9 --top-k 20 --presence-penalty 0 \
    --output-dir "$MODEL_DIR/readiness_vllm/qualitative_control_sampled"
done
```

The shared serving runner sends temperature0 for greedy and temperature.7/top-p.9 for sampled, max_tokens256. It omits seed, top-k and penalties; save the effective top-k20/presence0 server defaults alongside its output. Top-p is inactive for greedy. The sampled seed0 HF/TT controls are a distributional/qualitative comparison to the unseeded serving output, **not** an exact RNG replay. HF and TT RNG implementations also differ. Record this explicitly; repeated sampled-token equality is not a quality gate.

The current standalone control fixes TT context at 2048 while the final serving allocation is 262144. Canonical SDPA selects a different key chunk with context size; label this configuration difference. The control is useful for same-code/same-chip qualitative comparison but is not an exact-logit reference for the serving configuration. A later context option, defaulting to 2048, would permit an allocation-matched control. Do not change the running HF job's script hash or silently compare manifests generated before/after that extension; finish the current HF job first and use deliberate new provenance if adding it.

The standalone suite runs a fixed raw budget and scores through first EOS, while serving can stop at EOS. Compare the effective text and shared prefix, not the standalone post-EOS suffix. Both sides save rendered prompt IDs and current implementation hashes; retain them with the verdict. One seed is a smoke/control and does not satisfy the ten-seed failure investigation rule. If a quality failure needs investigation, use the original HF/model-card settings on both sides with at least ten matched seeds, preserve the existing blind rubric, and score before unblinding. Report concrete prompt IDs/excerpts and whether HF shares the failure; never gate on greedy reaching EOS.

## Host validation of the control extension

The CLI help, Python compilation and diff whitespace check succeed. Four cases executed the actual extracted policy block: unchanged defaults, legacy `--greedy`, sampled .7/.9/k20/presence0, and explicit temperature0 with those controls. The pinned local tokenizer was loaded without model weights and confirmed `enable_thinking=False` changes the rendered prompt. Pre-commit completed all applicable checks; Black reformatted the added rendering expression on the first pass. No HF model generation, TT execution, or server request was performed by the investigating agent for this change.

## Serving logit diagnostic: matched B32 cache geometry

`tests/vllm_logit_determinism_probe.py` has separate HTTP and TT modes. HTTP preserves raw greedy/logprobs20 responses for three explicit token prompts (129/131/130 tokens), two solo passes and two concurrent orderings. This is continuation/determinism coverage, not chat quality evidence.

TT mode now constructs the full32-layer `OrnithModel` and an external canonical cache with **32 slots, context262144, 8224 physical blocks**. It supplies this cache directly to `OrnithGenerator`; it does not use the earlier B3 standalone allocation. The32x8192 page table assigns each slot eight disjoint valid initial pages, including inactive slots needed during warmup/capture, and zero-fills unused tail entries to match the plugin. This also avoids introducing a different sentinel in any masked SDPA tail loads. Active requests occupy slots0/1/2. Prefill submits tokens[3,131] and tables[3,8192]; decode supplies tokens[32,1],32 positions (inactive positions-1) and the full table. Host-logit shapes are checked as[3,1,V] after prefill and[32,1,V] after decode, with masked decode required. The selected precision and200000000-byte trace region match the full serving setup. This B32/context geometry selects SDPA key chunk512; the replaced B3 geometry selected128.

TT teacher-forces the HTTP-selected tokens for the same three output positions, repeats the same row order and then reverses request order. Reports retain slot assignments, exact full-logit hashes/repeat errors, top20 candidates, HTTP-candidate ranks/logprob deltas, and the fixed1e-4 comparison. Physical page numbers and scheduler admission grouping can still differ from HTTP; matching dimensions alone does not establish bit-identical execution. `status: measured` is not a correctness pass. Full-vocabulary transfers are diagnostic only.

```bash
python_env/bin/python "$MODEL_DIR/tests/vllm_logit_determinism_probe.py" http \
  --output "$MODEL_DIR/readiness_vllm/full_tracked/logit_http.json"
# Only after the owner releases the server's devices:
python_env/bin/python "$MODEL_DIR/tests/vllm_logit_determinism_probe.py" tt \
  --http-result "$MODEL_DIR/readiness_vllm/full_tracked/logit_http.json" \
  --output "$MODEL_DIR/readiness_vllm/full_tracked/logit_tt_b32.json"
```

The B32 revision was checked with host mocks for explicit external-cache construction, disjoint warmup pages, three-row prefill,32-row masked teacher-forced decode, async read synchronization, repeat/reorder comparisons and cleanup. No device or server execution was performed by the author.


### P3 render_c07 advisory explanation

All12 behavioral cases and all10 critical checks pass. The retained `render_c07` advisory is a deprecated-field discrepancy in upstream `/tokenize`: legacy `reasoning_content` yields73 tokens, while current `reasoning` yields88. The original actual `/chat/completions` request already returned `Ada` with88 prompt tokens; its request validator normalizes the legacy alias, so the source chain and recorded usage support preserved chat history. No model/template change is indicated.

Authorized CPU-only tokenization checks reproduce73/88/88 server tokens for legacy/current/identical-dual fields with thinking off. The pinned HF template expects `reasoning_content`; copying canonical `reasoning` into that template alias gives exact token equality for the current API. Thinking on produces71/86/86, confirming the default thinking-off suffix is not the cause. Operators should use canonical `reasoning`; a direct HF reference should mirror the existing template alias, or a shared fixture may supply both fields identically. No new interface/default, plugin/upstream edit, installed-probe weakening or limitations waiver is needed. The original failed advisory remains unchanged.

Evidence: `doc/eval_readiness/chat_conformance/reasoning_field_analysis.md`, `reasoning_field_tokenization.json` (six exact CPU requests/responses and HF tokens), and `reasoning_field_host_proof.json` (actual installed schema-normalizer AST execution and source hashes). No new chat inference or device job was submitted during this investigation.


## P2 failure investigation: host sampler and CPU threads

CPU-only source audit against plugin `2a1504554aa7babb1d426dfbca8c7db8c0a795ef`, installed vLLM0.31.0 and the unchanged0.1.24 harness. No inference, device job or timing benchmark was submitted during P5/HF controls.

P2 `run_harness_requests.py:140-163,270-315` times plain then profile on the same endpoint for each concurrency. Its gate divides median per-request streamed TPOTs; `served_api.py:136-155` uses `(last nonempty content/reasoning chunk - first)/(usage completion_tokens - 1)`. Engine histogram deltas are also saved. Default output128 is a capped timing probe, not the profile's32768-token completion budget. Preserve per-request token counts, errors, route logs and metrics; do not substitute aggregate throughput or change the profile. Reasoning top-k40/top-p.95/seed42 must remain exact and host-routed at the truthful device bound32.

The existing plugin already optimizes seeded host sampling: `host_sampler.py:55-166` recognizes distinct per-request CPU generators, generates/prefetches noise in a bounded64MiB cache and groups sampling into at most4 row tasks. Pool capacity is at most32 or CPU affinity; grouping additionally depends on Torch intra-op threads. Temperature1 is already a no-op (`:171-184`). Do not benchmark only upstream sequential `random_sample` and assume that is this server's path. The remaining native top-p filter sorts the entire vocabulary, then applies top-k thresholds, softmax/cumsum/mask/scatter (`vllm/v1/sample/ops/topk_topp_sampler.py:449-505`), followed by another softmax and exponential-noise selection. Exact top-p.95 does not activate the top-k-only shortcut.

The model's host boundary also matters: `tt/generator.py:1141-1163,1176-1194` asynchronously reads all four vocabulary shards, converts each to FP32, concatenates, then clones the final view. At batch32/vocabulary248320, one full FP32 matrix is30.31MiB and BF16 transfer is15.16MiB before transport overhead. A model-owned follow-up could reduce redundant CPU allocations/copies while retaining private writable output, correct vocabulary/row order, inactive masks and pending-submission independence. Measure this separately before editing. Existing compact/selective host-logit capabilities are lane-DP row optimizations, not a way to send only40 vocabulary candidates or accelerate the full active front-packed B32 case. Claiming a larger device bound remains invalid.

**OMP is an existing supported launch control, but higher values are only a hypothesis.** Full-server log line12659 explicitly records Torch threads reduced8→1 for serving. A read-only process snapshot for EngineCore851944 showed `OMP_NUM_THREADS=8`, `VLLM_OMP_NUM_THREADS_SET_BY_VLLM=1`, MKL unset and affinity16 CPUs; that environment describes startup, not the post-warmup runtime count. Plugin `worker.py:534-536` invokes the installed `set_torch_threads_for_runtime`, whose documentation at `vllm/utils/torch_utils.py:259-291` preserves externally supplied OMP settings and warns that thread spin-wait can hurt serving. Plugin `tests/test_dp_modes.py:279-299` verifies automatic4→1 versus external4 retained. PyTorch documents OMP limits and MKL's precedence in [threading environment variables](https://docs.pytorch.org/docs/stable/threading_environment_variables.html).

If the exact32-user ratio exceeds1.15:

1. Retain the failing result. First rule out unrelated CPU competition, mismatched completed-token counts, errors, trace setup or unexpected routing; inspect both streamed and engine timing evidence. Do not run comparison benchmarks concurrently with HF controls or other serving load.
2. In an idle CPU window, benchmark the **actual** existing host sampler at B1/B32, full vocabulary, k40/p.95/temperature1 and distinct seed42 generators, retaining identical initial RNG states and measuring cold versus prefetched noise. Compare bounded intra-op counts1,2,4 (8 only if useful), plus conversion/concatenation costs. Record affinity/quota and actual `torch.get_num_threads`; no TT access is needed for that diagnostic.
3. If CPU evidence supports it, restart the unchanged server with explicit external `OMP_NUM_THREADS=2` or4, leaving MKL unset or deliberately matched, and repeat the unchanged P2. Do not set vLLM's internal auto-sizing marker. Capture actual runtime threads; compare both absolute plain/profile TPOT and ratio so slowing the baseline cannot masquerade as an improvement. Check other serving controls before retaining the launch setting. No new interface/default policy is required for this existing operator option, and no setting was changed by this audit.
4. If transfer/materialization dominates, investigate the model-owned copy path with ownership regressions and a fresh device correctness check before accepting an optimization. If full-vocabulary sorting still dominates, any shared host-sampler algorithm change needs its own eight-part proposal and seeded/tie/top-p compatibility evidence. No plugin/upstream change is proposed or authorized here merely because top-k40 is slower.

No predicted speedup or P2 pass is claimed. The1.15 threshold and installed profiles remain unchanged.
