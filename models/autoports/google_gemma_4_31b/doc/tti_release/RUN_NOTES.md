# TTI release handoff: google/gemma-4-31B

## Outcome

- Date: 2026-07-16 UTC.
- Result: **release-workflow-pass/readiness-fail**. The authoritative final6 workflow exited `0`; the regenerated handoff report correctly blocks release readiness on both mandatory Meta rows because no valid canonical-control waiver exists.
- Evaluated implementation: `models/autoports/google_gemma_4_31b` only.
- Server mode: existing external autoport vLLM server, no Docker and no TTI-managed local server.
- Host context: reservation container `spawner-exp-d-gemma31`; loopback endpoint `http://127.0.0.1:8000` on a P150x4 mesh.
- Docker fallback: not used. The copied spec records the matching release image `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.18.0-c49bb76-6b4a3a7`, but no container was launched from it.
- Context contract: `max_context=113280`, `max_model_len=113280`, and `max_num_batched_tokens=113280`, preserving `../context_contract.json`. The release sweep includes an actual 65,535-token input plus 128 output tokens.

## Provenance

- tt-metal release base: `2be0f245e2005c72b54ec9884ca10323dea30178`.
- vLLM checkout: `44b7853d448f3f8c5db7ed068a4f82ebfcd1065d`.
- TTI checkout/tag: `tt-inference-server` `v0.18.0`, base `d5913e816`.
- TTI local release-fix HEAD: `b803374e04c2460ea3bfabec4bfed832f2af532a`.
- TTI local commits, oldest first:
  - `f1a89cb4b` Fix v2 external autoport release specs.
  - `6ad299582` Preserve service port for external workflows.
  - `c5eb37b7a` Fix Gemma release eval and target wiring.
  - `e4d2307cd` Preserve Gemma release eval context and prompts.
  - `8a69f76d4` Fix Gemma IFEval scorer schema.
  - `569f62b01` Isolate vLLM conformance pytest discovery.
  - `507c74673` Allow long vLLM conformance requests.
  - `fdc353375` Bound vLLM determinism conformance samples.
  - `b803374e0` Fix Gemma Meta GPQA answer filtering.
- No commit was pushed.
- tt-metal Stage 11 model-fix commit: `97a16e1c982a27fbc2f4e27b65dbd6b077f9e34f` (`Fix Gemma 4 dynamic decode head grids`).
- tt-metal Stage 11 release-evidence commit: `56340a0e2a0` (`Record Gemma 4 TTI release handoff`).
- tt-metal Stage 11 handoff-metadata commit: `98de8fd3c6f98387da50726de4678e2bbe703aec` (`Complete Gemma 4 TTI handoff metadata`).
- tt-metal Stage 11 readiness-blocker commit: `6425539564d` (`Record Gemma 4 release readiness blockers`).
- tt-metal Stage 11 evidence-reconciliation commit: `f095b3c6498d9c18ff3d952eb5619088be2f2576` (`Reconcile Gemma 4 release evidence`).

## Commands and environment

The server was launched from the reservation container through the checkout readiness runner:

```text
python -u -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/google_gemma_4_31b \
  --hf-model <local google/gemma-4-31B snapshot> \
  --stages serve --mesh-device P150x4 --port 8000 \
  --max-num-seqs 32 --max-model-len 113280 --block-size 64 \
  --server-timeout 2400 \
  --tt-config '{"sample_on_device_mode":"all","trace_region_size":268435456,"fabric_config":"FABRIC_1D","trace_mode":"all","enable_model_warmup":true}' \
  --additional-server-args='--served-model-name google/gemma-4-31B ... --async-scheduling --chat-template models/autoports/google_gemma_4_31b/doc/vllm_integration/chat_template.jinja'
```

The authoritative client-side release command was:

```text
python3 run.py \
  --model gemma-4-31B-it \
  --runtime-model-spec-json ../autoport_release_spec.json \
  --tt-device p150x4 \
  --workflow release \
  --service-port 8000 \
  --server-url http://127.0.0.1 \
  --no-auth \
  --skip-system-sw-validation
```

Key non-secret server environment: `TT_GEMMA4_TEXT_VER=gemma4_31b_autoport`, `GEMMA4_31B_VLLM_HOST_SAMPLING_COMPAT=1`, offline Hugging Face/Transformers operation, checkout-local `PYTHONPATH`, writable `.exp_run/tt-metal-cache`, and a temporary full-model tensor cache. Client caches were isolated under `.exp_run/tti-release/gemma4-31b-20260716/client_cache`.

## Smoke gate

- Server health check: HTTP 200.
- One OpenAI-compatible request: HTTP 200; compact request/response copied under `smoke/`.
- Non-aligned prompt: PASS with 149 logical tokens; the request was not rounded by the client.
- Tiny TTI benchmark: PASS with `disable_trace_capture=true`, ISL 8, OSL 8, concurrency 1, one request.
- Smoke report: `smoke/SMOKE_REPORT.md`.

## Release results

- Copied report: `RELEASE_REPORT.md`.
- Copied report data: `release_report_data.json`.
- Successful authoritative run log: `tti_release_final6.log` (sanitized by the repository whitespace hook; SHA-256 `b479faf832817336a23a496d6044ac940256d19fa0753005a7407101c9982df3`).
- Original final6 harness acceptance: PASS, zero blockers, based on two missing-reference waivers that did not meet the release-waiver standard.
- Regenerated handoff acceptance: FAIL with two blockers (`meta_ifeval` and `meta_gpqa_cot`). The invalid waivers were removed from the copied model/runtime specs.
- Accuracy rows:
  - `meta_ifeval`: 25.181850822484343.
  - `meta_gpqa_cot`: 26.339285714285715 after correcting the GPQA answer filter; the original buggy score was 20.982142857142858.
- Both mandatory eval rows are retained and unwaived. No unrelated instruction-tuned threshold was borrowed. Without a complete canonical HF/GPU control for this base checkpoint, both rows remain readiness blockers.
- `eval_meta_gpqa_cot.json` is the corrected saved-output rescore. The original lm-eval result is retained as `eval_meta_gpqa_cot_original_buggy_parser.json` for auditability.
- Benchmarks: all 17 configured points completed with zero failed requests. The target point passed functional, complete, and sellable tiers; compact rows are in `benchmark_summary.csv`.
- Spec tests: PASS. Logger fork safety passed and vLLM chat/completions parameter conformance passed 21/21 in 5,802.38 seconds, including 9/9 penalty cases and all seed, determinism, logprobs, stop, `n`, and `max_tokens` cases.
- Dynamic batch repair received live release coverage at irregular active batches including 13, 17, 19, 23, 26, 29, and 31, plus the explicit concurrency-26 and concurrency-13 benchmark points.

## Implementation proof

- `runtime_model_spec.json` records `impl.code_path=models/autoports/google_gemma_4_31b`, `impl.impl_name=autoport-google-gemma-4-31b`, and `metadata.autoport_generator=models/autoports/google_gemma_4_31b/tt/generator_vllm.py`.
- The server command used `--model-dir models/autoports/google_gemma_4_31b` and imported the autoport vLLM generator.
- The copied report records `model_impl=autoport-google-gemma-4-31b`.
- The copied release artifacts do not identify `models/tt_transformers`, `models/demos`, or another packaged implementation.

## Recovery and fixes

- Optimized-vLLM predecessor recovery: two pre-model device-0 remote-Ethernet resume failures at core 31-25 were recovered by terminating only failed-run processes, bounded list/reset/list operations, and passing 1x4 mesh open/close smokes before resuming.
- Stage 11 initial server attempt: fabric ERISC compilation failed because the default `/home/odjuricic/.cache/tt-metal-cache` was not writable. Both TT cache variables were redirected to the checkout-local `.exp_run/tt-metal-cache`; the subsequent autoport server passed readiness and non-aligned checks.
- Post-release cleanup: the first mesh-open smoke reproduced the recoverable device-0 core 31-25 heartbeat failure after the server had shut down. With no device holders, a bounded reservation-container list/reset-all/list recovery succeeded; the repeated P150x4 smoke opened grid 11x10 with `FABRIC_1D` on all four devices and closed cleanly. The final list showed all four boards.
- Release harness/spec/API/eval/benchmark failures were repaired under `$autofix`; concise diagnosis and result reports are copied under `autofix/`.
- Meta-accuracy `$autofix` found that the GPQA task accepted `[A-Z]`, so the literal placeholder `X` embedded in every prompt was incorrectly treated as an answer. Restricting the filter to `[A-D]` changes the saved-output score from 94/448 (20.9821%) to 118/448 (26.3393%); the TTI fix and regression tests are committed in the local TTI checkout.
- An exact HF CPU control for GPQA row 0 completed in 223.36 seconds and matched the TT answer (`C`, with gold `B`). A batch-4/limit-4 control produced no completed rows before its clean 904.546-second timeout and reached 61.529 GiB sampled RSS. A subsequent exact batch-32/128-token probe also timed out at 900 seconds. The official Gemma 4 MTP drafter preserved the row-0 output exactly but was 1.7% slower; exact prompt lookup improved row 0 by 1.61x and a deliberately high-repetition TT sample by only 1.229x. For that second sample, the exact HF prefix remained coherent while TT had already entered a repetition loop, so TT output compressibility cannot justify a faster HF projection. The primary-source audit found no exact-base published, TTI, or internal CI reference, and no accessible pre-provisioned HF endpoint exists. The full evidence and exact external H100/H200 closure contract are recorded in `autofix/meta_accuracy_AUTOFIX_RESULT.md`; `$autofix` could not close the canonical-control gate locally.
- Model repair: dynamic decode shard grids now use exact row-wise multi-range core sets when an active batch cannot form one rectangle, with matching full-worker-grid concat subcore configuration. Exhaustive host tests passed and the full release exercised every irregular batch observed by IFEval.
- No context or request-length reduction, benchmark alignment, stock-model substitution, Docker fallback, or raw eval-sample reuse was used.

## Cleanup and device state

- Cleanup and final reservation-container device health evidence are recorded in `post_release_health.log`.
- No Stage 11 client, vLLM/API server, EngineCore, or Docker container was left running. The unrelated pre-existing `multigoal` tmux session was preserved.

## Independent stage review

- Final verdict: **MORE-WORK-NEEDED**.
- No fixable handoff inconsistency remains.
- The only material gap is the unwaived mandatory Meta reference/threshold requirement described above; `$autofix` exhausted the feasible local canonical-control path.
- Full verdict: `stage_review_final.md`.

## Artifact policy

Only small handoff artifacts and the 143 KiB successful workflow log were copied. Raw eval sample JSONL, detailed multi-megabyte benchmark request payloads, weights, Hugging Face caches, Docker layers, TT persistent caches, and the large server log remain outside this directory. `autofix/meta_accuracy_gpqa_corrected.json` records the offline rescore without copying raw samples.
