# Gemma 4 26B A4B IT — TTI release run notes

Status: **release workflow passed; Stage 11 readiness blocked**. Exact BF16 HF controls resolve IFEval as passing but expose an unwaived GPQA model-correctness failure. Required model-path Autofix was exhausted without reaching the mandatory GPQA gate.

## Evaluated implementation

- Hugging Face model: `google/gemma-4-26B-A4B-it`
- Generated implementation: `models/autoports/google_gemma_4_26b_a4b_it`
- This run does not use `models/tt_transformers`, `models/demos`, stock `tt-transformers`, or another packaged implementation.
- Autoport server entry point: `python -m models.common.readiness_check.run_vllm_server --model-dir models/autoports/google_gemma_4_26b_a4b_it ...`
- Context contract: `max_model_len=262144`; no context or request-length reduction was used.
- Tokenizer/prompt path: Hugging Face `GemmaTokenizer`, model chat template, OpenAI-compatible `/v1/chat/completions`.

## Host, devices, and server mode

- Reservation container hostname: `b30c965c728b`
- Device: four local P300C devices opened as a `1x4` mesh (`p300x2` TTI device spelling).
- Server mode: existing external autoport vLLM server, no Docker and no TTI-managed local server.
- Endpoint: `http://127.0.0.1:8000`; no authentication material was printed or copied.
- `tt-smi -ls --local` found all four devices and a `ttnn.MeshShape(1, 4)` open/close smoke passed before serving.
- The definitive release needed no recovery. During later Autofix A/B restarts,
  transient ARC/active-Ethernet heartbeat failures were recovered inside the
  reservation container with targeted `tt-smi -r`; no Docker or physical-host
  recovery was used.

## Revisions

- tt-metal server checkout: `4b17e185dea9`
- vLLM checkout recorded by the release spec: `938c45ed`
- TTI version/tag: `0.19.0`
- Initial TTI checkout: `e26e723b`
- TTI release wiring: `02e81d32`
- TTI parser/generation/conformance wiring: `e3ea566d`
- TTI GPQA timeout repair: `82e52455`
- TTI conformance repair: `61473555`
- Definitive TTI measured-timeout repair: `daa1fe6f`
- Exact CI-nightly subset references: `1fb8702b`
- tt-metal artifact checkpoint: `a08f7ac8f33`
- tt-metal notes/review checkpoint: `ee55f892bee`
- tt-metal exact-reference and Autofix checkpoint: `eb459b3bf9e`
- A final notes-only commit follows this checkpoint; use the repository log as the authoritative SHA to avoid a self-referential commit field.

## Commands and key environment

Autoport server:

```text
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir models/autoports/google_gemma_4_26b_a4b_it \
  --hf-model google/gemma-4-26B-A4B-it \
  --mesh-device P300x2 --max-num-seqs 32 --max-model-len 262144 \
  --sampling-profile full \
  --tt-config '{"trace_region_size":220000000,"fabric_config":"FABRIC_1D_RING"}'
```

Definitive no-Docker TTI release:

```text
ONLY_BENCHMARK_TARGETS=1 \
CACHE_ROOT=models/autoports/google_gemma_4_26b_a4b_it/doc/tti_release/release_gate_cache \
python3 run.py --workflow release \
  --runtime-model-spec-json models/autoports/google_gemma_4_26b_a4b_it/doc/tti_release/autoport_release_spec.json \
  --tt-device p300x2 --service-port 8000 \
  --server-url http://127.0.0.1 --no-auth \
  --skip-system-sw-validation --disable-trace-capture \
  --limit-samples-mode ci-nightly
```

The embedded runtime spec—not command-line overrides alone—sets `docker_server=false`, `local_server=false`, `service_port=8000`, workflow `release`, context `262144`, and the autoport code path.

## Smoke evidence

- `/health`: HTTP 200.
- Direct OpenAI-compatible chat request: passed; response saved as `openai_chat_smoke.json`.
- Tiny TTI benchmark: 8 input / 8 output tokens, one successful request, zero failures, trace capture disabled.
- The deliberately non-aligned smoke request passed without benchmark alignment or model-side padding restrictions.
- Optimized-vLLM evidence also covers logical prompt lengths 47, 2051, and 262143.

## Repairs and release interpretation

- `$autofix` repaired stable `meta_*` result alias parsing while preserving public lm-eval task identities.
- The lm-eval API backend's implicit 256-token default truncated GPQA reasoning. Gemma GPQA now uses the locally canonical reasoning allowance `max_gen_toks=32768`; context remains 262144.
- A fixed 1800-second lm-eval client timeout failed valid long requests. The final task-local timeout is 14400 seconds, derived from measured aggregate throughput and leaving the request unchanged.
- Generic penalty conformance heuristics produced false negatives despite materially changed fixed-seed outputs. The repaired contract fails identical outputs and accepts actual content changes without requiring unrelated length or whitespace statistics.
- `ci-nightly` evaluates a 5% accuracy subset. From definitive measured task times, a linear full-set projection is IFEval `318 s * 541/28 = 6,144 s` (1.71 h) plus GPQA `12,773 s * 198/10 = 252,905 s` (70.25 h), or about **71.96 hours** serially before server startup, benchmark, conformance, and report overhead. GPQA's concurrent long-tail requests make this only an estimate. No machine-readable reservation deadline was exposed in the container, so a continuous three-day device hold could not be justified or guaranteed; unrestricted readiness is not claimed.

## Final result

- Release readiness classification: **BLOCKED** (`release-workflow-pass/readiness-fail`).
- `meta_ifeval`: score **82.62** (strict instruction-level accuracy **0.8372**; strict prompt-level accuracy **0.7857**), 5% CI-nightly sample.
- `meta_gpqa_cot`: flexible-extract exact match **0.4000**, 5% CI-nightly sample.
- Benchmark: **PASS**, 8/8 requests, 0 failures, decode throughput **26.55 tok/s**, mean TTFT **267.45 ms**, target 26 tok/s / 300 ms.
- API/spec tests: **PASS**, 22/22 parameter cases plus logger-fork-safety; includes non-uniform seeding, penalties, stop, `n`, token limits, and logprobs.
- Exact BF16 HF CPU controls used snapshot `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, lm-eval commit `5416b8a97e8460cb74ae8cd96a457016fc0dc2e8`, seed 42, the unchanged chat template and 262144-token context. Aggregate-only artifacts are `hf_reference_ifeval.json`, `hf_reference_gpqa_cot.json`, and `HF_REFERENCE_NOTES.md`.
- `meta_ifeval` reference: **87.0432%** over the exact first 28 documents. TT 82.62% satisfies the sample-aware 95% gate (23 correct-equivalent versus threshold 23).
- `meta_gpqa_cot` reference: **100%** over the exact first 10 documents. TT 40% fails the sample-aware 95% gate (4 correct versus threshold 9). This failure is unwaived and blocking.
- Model-path Autofix localized the failure to iterative TT decode numerical drift: HF and TT match tokens 0-14 on a known failing document, then TT selects HF's second-ranked token at index 15. Prefill, tokenization, tracing, concurrency, and scheduler-row packing were isolated. The best feasible precision diagnostic scored only 4/10; see `AUTOFIX.md` and `AUTODEBUG_GPQA_DIVERGENCE.md`.
- Final report: `TTI_RELEASE_REPORT.md`; structured result: `tti_release_report_data.json`; captured spec: `tti_runtime_model_spec.json`.
- Stage review: **more-work-needed**; see `stage_review.md`. A clean-pass rereview is not possible while the unwaived mandatory GPQA gate fails.
- Autofix terminal state: **FAILED** after exhausting isolated decode precision/op-path candidates without reaching 9/10. Per the stage goal, work stops at this terminal condition rather than weakening the task or context contract.
- Cleanup/final hardware health: **PASS**. The TTI runner and autoport vLLM tmux sessions exited, no serving/engine process or container remains, `tt-smi -ls --local` reports all four P300C devices, and a post-run `ttnn.MeshShape(1, 4)` open/close passed. Transient workflow caches and raw sample dumps were removed after compact evidence was copied.
