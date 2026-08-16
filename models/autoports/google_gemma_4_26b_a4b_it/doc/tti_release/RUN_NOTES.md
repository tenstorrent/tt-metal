# Gemma 4 26B A4B IT — TTI release run notes

Status: **release workflow passed; Stage 11 readiness blocked**. The definitive no-Docker `ci-nightly` release returned `rc=0`, but independent review rejected the two mandatory `NA` accuracy comparisons.

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
- No ARC, ERISC, remote-Ethernet, reset, Docker, or host-level recovery was required.

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
- tt-metal artifact checkpoint: `a08f7ac8f33`
- tt-metal notes/review checkpoint: `ee55f892bee`
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
- Accuracy blocker: both mandatory rows are present and nonzero, but remain `NA` because no comparable full-set GPU reference or exact-subset `ModeReferenceScore` exists. The official GPQA 82.3% lacks recipe equivalence, no official IFEval value exists, and current TT scores were not self-baselined. A valid unblock requires paired HF/GPU controls on the exact CI documents and harness settings, then a TTI eval/report rerun.
- Final report: `TTI_RELEASE_REPORT.md`; structured result: `tti_release_report_data.json`; captured spec: `tti_runtime_model_spec.json`.
- Stage review: **more-work-needed**; see `stage_review.md`. `$autofix` exhausted local code/spec/waiver mechanisms and found that new external GPU control evidence is required.
- Cleanup/final hardware health: **PASS**. The TTI runner and autoport vLLM tmux sessions exited, no serving/engine process or container remains, `tt-smi -ls --local` reports all four P300C devices, and a post-run `ttnn.MeshShape(1, 4)` open/close passed. Transient workflow caches and raw sample dumps were removed after compact evidence was copied.
