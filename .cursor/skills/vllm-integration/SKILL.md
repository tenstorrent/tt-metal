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
  --max-model-len 262144 \
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
- Use `qualitative-check`; RUN-first coherent-or-degenerate behavior must be
  classified against #48291 rather than mistaken for an adapter regression.
- Preserve non-aligned prompt lengths and the 262144 context contract.

## Trace and performance

Serving reuses the generator's traced denoise controller. Verify one capture is
replayed across blocks and is not recaptured per request step.

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
