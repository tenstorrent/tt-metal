---
name: vllm-integration
description: Integrate and validate the existing block-granular DiffusionGemma adapter with the tenstorrent/vllm TT plugin. Use per-block serving contracts, direct OpenAI-server tests, and traced block metrics; never apply per-token async/TPOT assumptions.
---

# DiffusionGemma vLLM integration

Load `diffusion-gemma` first. Serve through the tenstorrent/vllm TT plugin
fork, not upstream GPU vLLM.

## Existing implementation and status

- Adapter: `models/experimental/diffusion_gemma/tt/generator_vllm.py`.
- vLLM-free core: `tt/serving.py::BlockDiffusionServingSession`.
- Generator engine: `tt/generate.py`.
- Evidence: `models/experimental/diffusion_gemma/doc/vllm_integration/`.

The adapter is live on QB2 and has served real OpenAI completion/chat requests,
including two committed 256-token blocks. The scoped runner/scheduler changes
are recorded as plugin patches in the evidence directory.

Current advertised capabilities are:

```text
supports_prefix_caching = False
supports_async_decode = False
supports_sample_on_device = True
```

Do not advertise async decode or vLLM APC until their block-granular contracts
are implemented and tested. The local `DG_PREFIX_CACHE` prototype is not vLLM
APC.

## Current launch contract (2026-07-17)

Never benchmark or judge quality from an implicit launch:

- Set `DG_SPARSE_MOE=1`; it defaults off and otherwise denoise computes the dense 128-expert path.
- Set `DG_DEDUP_ARGMAX=1` for argmax controls; it defaults off.
- Set `DG_VLLM_MAX_DENOISE_STEPS` explicitly; unset means K=48.
- Set `DG_VLLM_GUMBEL_MODE` explicitly (`argmax` is a fast/RUN control; `chunked` is the
  production sampler).
- Set `DG_VLLM_TRACE` explicitly. Unset is eager. Trace block 0 is capture-inclusive and current
  growing contiguous-prefix shapes recapture across blocks.
- Pass `--generation-config vllm`; otherwise checkpoint config caps output at one 256-token block.
- Pass `--max-num-batched-tokens >= largest whole prompt`; TT chunked prefill is not scheduler
  admission and oversized prompts otherwise remain waiting.
- Model-side `DG_PREFILL_RAGGED_LONG` defaults on and slices prompts above 4096 through the ragged
  top-8 MoE path. For pure-prefill numbers, use
  `context_window_prefill_only_chunkedlong_20260713_msl65536.json`; the artifact without
  `chunkedlong` is the superseded dense-fallback control.
- Do not use `ignore_eos=true` for qualitative judgment; it exposes the post-EOS physical canvas.
- HTTP temperature/top-p/top-k/seed are not wired into the denoise loop.

## Block contract

- `prefill_forward` writes prompt KV, creates the stateful denoise adapter, and
  emits block 0.
- `decode_forward` emits one additional committed canvas block per active
  request.
- Output shape is `[num_requests, canvas_length]`, normally 256 tokens/request.
- Position and computed-token counts advance by `canvas_length`, not one.
- The model owns bidirectional denoise attention, frozen-prefix KV semantics,
  canvas sampling, self-conditioning, and commit append.

There is no per-token `tt_out_tok`, stale-token refresh, `+1` position update,
or token-feedback loop. Any runner/scheduler patch must preserve N-token output
blocks and bound-check `start + canvas_length <= max_model_len`.

## Cache and concurrency

The current denoise path reads a model-owned contiguous frozen-prefix cache and
runs one active sequence per cache. `allocate_kv_cache` exposes existing model
cache handles without double allocation.

Concurrent multi-sequence serving remains gated on:

- vLLM paged-cache ownership and per-request block tables;
- batched canvas decode (#47557);
- block-aware scheduler state.

Do not claim `max_num_seqs > 1` support from a loop that serializes independent
single-sequence sessions.

## Plugin registration

HF architecture:

```text
DiffusionGemmaForBlockDiffusion
```

Plugin architecture:

```text
TTDiffusionGemmaForBlockDiffusion
```

Register:

```text
models.experimental.diffusion_gemma.tt.generator_vllm:DiffusionGemmaForCausalLM
```

The plugin is in the external tenstorrent/vllm checkout. Keep reproducible
patches under `doc/vllm_integration/`; do not pretend the fork lives inside
tt-metal.

## Bring-up and tests

The old generic `models.common.readiness_check` package is not present in this
checkout. Do not invoke it.

Use the smallest representative path first:

```bash
python models/experimental/diffusion_gemma/demo/serving_smoke.py --help
```

Run the block-contract tests:

```bash
DG_RUN_DEVICE=1 python -m pytest \
  models/experimental/diffusion_gemma/tests/test_serving_block_contract.py -q
```

For a real server, use the project-matching tenstorrent/vllm environment and:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <checkpoint> \
  --served-model-name diffusiongemma-26B-A4B-it \
  --generation-config vllm \
  --max-model-len <validated-served-limit> \
  --max-num-batched-tokens <largest-whole-prompt> \
  --block-size 64 \
  --max-num-seqs 1
```

Then issue targeted `/v1/completions` and `/v1/chat/completions` requests,
including a non-256-aligned prompt and a 512-token/two-block request. Use the
exact launch and request details recorded in
`doc/vllm_integration/README.md`; do not copy machine-specific paths into the
skill.

## Correctness and qualitative checks

- Sampling stays on device: no host argmax or full-logits readback.
- Validate block 0 and a subsequent block, including position
  `prompt_len → +256 → +512`.
- Verify EOS/length trimming without corrupting physical whole-block commit.
- Compare served output with the same full-model RUN-path control.
- Use `qualitative-check`. The July-15 control demonstrates coherent TT output at the intrinsic
  bf16 diffusion floor, so persistent garbage is required work unless the same prompt/config
  control reproduces it. Check EOS-tail exposure, argmax-vs-chunked mode, K, and prompt formatting.
- Preserve non-aligned prompt lengths and the HF 262144 prompt+generated contract, but distinguish
  standalone allocation evidence from the exact live-vLLM `--max-model-len` actually validated.

## Trace and performance

Serving reuses the generator's traced denoise controller. Verify trace sets are keyed by the
visible frozen-prefix shape. The current contiguous-cache path must release/recapture after each
commit grows the prefix; same-ID cross-block replay is valid only after paged/fixed-shape prefix
inputs land. Verify no recapture within a fixed-shape denoise block and no stale prompt-only replay
after committed KV becomes visible.

Report:

- prefill+block-0 TTFT;
- mean/p99 block latency;
- blocks/second;
- tokens-per-block/second;
- step count and commit latency;
- request success and emitted block count.

Do not derive a headline from vLLM per-token TPOT/ITL or
`1000 / mean_tpot_ms`; the scheduler emits blocks. If generic vLLM JSON reports
those fields, retain them only as raw transport diagnostics and label them
non-semantic for DiffusionGemma.

Also do not use API-visible `completion_tokens / wall_time` as a device rate: EOS trimming changes
the numerator and `max_num_seqs=1` queueing changes the denominator. Use `DG_VLLM_METRIC` block
latency and `256 / block_latency`.

Do not run Tracy, `tt-perf-report`, or live-server device profiling in this
stage. Use same-harness before/after serving metrics and earlier non-serving
device profiles.

## Evidence

Maintain:

- `doc/vllm_integration/README.md` and `work_log.md`;
- `serving_test_suite.json` and `live_vllm_serving.json`;
- reduced and full-depth `serving_smoke_*.json`;
- traced-serving artifacts;
- plugin registration/model-runner/scheduler patches;
- exact fork revision and server command.

Done means the direct OpenAI server path uses this adapter, returns valid
responses for non-aligned and multi-block requests, preserves the context and
on-device sampling contracts, reports block metrics, and states unsupported
concurrency/APC/async capabilities honestly.
