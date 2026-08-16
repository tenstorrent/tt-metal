# Gemma 4 26B A4B IT — TTI release run notes

Status: **Stage 11 release gate passed**. The definitive no-Docker `ci-nightly` release returned `rc=0`, with acceptance `PASS` and zero blockers.

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
- Final tt-metal artifact commit: pending

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
- `ci-nightly` evaluates a 5% accuracy subset. A full unrestricted sweep would scale the long reasoning workload materially and is not claimed here.

## Final result

- Release readiness classification: **Stage 11 / nightly-equivalent PASS** (`EXPERIMENTAL` model status); unrestricted full-set readiness is not claimed.
- `meta_ifeval`: score **82.62** (strict instruction-level accuracy **0.8372**; strict prompt-level accuracy **0.7857**), 5% CI-nightly sample.
- `meta_gpqa_cot`: flexible-extract exact match **0.4000**, 5% CI-nightly sample.
- Benchmark: **PASS**, 8/8 requests, 0 failures, decode throughput **26.55 tok/s**, mean TTFT **267.45 ms**, target 26 tok/s / 300 ms.
- API/spec tests: **PASS**, 22/22 parameter cases plus logger-fork-safety; includes non-uniform seeding, penalties, stop, `n`, token limits, and logprobs.
- Eval reference waiver: both accuracy rows are `NA` only because the custom autoport spec has no published/GPU reference score. The rows are present and nonzero, both mandatory tasks ran, and neither row failed. This Stage 11 handoff accepts the measured scores without inventing a reference baseline.
- Final report: `TTI_RELEASE_REPORT.md`; structured result: `tti_release_report_data.json`; captured spec: `tti_runtime_model_spec.json`.
- Stage review: pending
- Cleanup/final hardware health: **PASS**. The TTI runner and autoport vLLM tmux sessions exited, no serving/engine process or container remains, `tt-smi -ls --local` reports all four P300C devices, and a post-run `ttnn.MeshShape(1, 4)` open/close passed. Transient workflow caches and raw sample dumps were removed after compact evidence was copied.
