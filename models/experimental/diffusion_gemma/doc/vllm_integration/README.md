# DiffusionGemma — vLLM serving integration (#47466 / #47488)

## Status (lead)

- **vLLM sampling status:** on-device canvas sampling (`sample_on_device_mode=all` equivalent):
  Gumbel-max → entropy-budget accept → renoise, entirely on device. No host argmax, no
  full-logits readback. `model_capabilities = {prefix_caching: False, async_decode: False,
  sample_on_device: True}`.
- **Qualitative verdict: PASS (RUN-first).** The full-depth serving path reproduces the model's
  RUN-path output: with 16 denoise steps it emitted coherent text through the block-emission
  adapter — `你好！I'm doing well, thank you for asking. How can I help you today?` — matching the
  recorded `text_demo` visible-dialogue control almost exactly. So the adapter is **not a serving
  regression**. With the fast RUN config (4 steps / EOS-stop) it emits the expected EOS-heavy
  degenerate block (#48291 fidelity bar, not a serving bug). Control: the RUN visible-dialogue
  output + the R0.5 HF-vs-TT committed-argmax replay (`demo/replay_hf_tt.py`, `plan.md`).
- **Serving path: LIVE + MULTI-BLOCK (updated 2026-07-03).** A full live serving test suite ran on
  QB2: 7 real OpenAI requests (completions + chat, all non-256-aligned prompts) each returned HTTP
  200, and a **live 2-block serve** emitted two real committed 256-token denoise blocks
  (`32→288→544`). The dg-09 `#47488` scheduler-half blocker was reproduced live (on an ordinary
  completion) and **fixed** (`plugin_47488_scheduler.patch` + a `generator_vllm.py` session-stop
  deferral). See **§ Serving test suite** and `serving_test_suite.json`. Original block-0 bring-up:
- **Serving path: LIVE (block-0, dg-09).** A fresh, project-matching vLLM (built from the tenstorrent/vllm fork
  against the *current* tt-metal `ttnn`, NOT the stale ghcr image) now serves DiffusionGemma
  end-to-end on QB2. A real `POST /v1/completions` returns **HTTP 200** with a valid OpenAI
  response; the 256-token block flows through prefill → runner sampling → state/​output build →
  engine → response. The runner's one-committed-token assumption (the #47488 blocker) was
  **reproduced live** (`model_runner.py:2757` `reshape(sz)` → `RuntimeError: shape '[1]' is
  invalid for input of size 256`) and then **fixed live** with the scoped #47488 patch
  (generalize one-token → N-token blocks). See **§ Live serving verification (fresh vLLM)** below.
  The reduced-surface driver `demo/serving_smoke.py` remains the fast device-only contract proof.

### Primary single-user block metrics (reduced-surface serving driver)

> Metrics are reported **per-block**, never as `1000/mean_tpot_ms` (there is no per-token TPOT
> in block-diffusion). Numbers below are from the reduced (1-layer) target — an inner-loop
> contract proof, **not** the full-model accuracy/perf number. Full-depth numbers go under
> "Full-depth run" once captured.

All runs: 1 prompt, non-256-aligned prompt length, `--gumbel-mode argmax`, `--max-seq-len 1024`,
QB2 `(1,4)` mesh, block/canvas = 256. Reduced (1-layer) is the inner-loop contract proof; the
full-depth (30-layer) rows are the final evidence.

| Run | Blocks × tokens | Prefill TTFT (prefill+block0) | Mean block latency | Tokens/block/s | Position | Output |
|---|---|---|---|---|---|---|
| Reduced 1-layer, 2 steps | 2 × 256 = 512 | 8.24 s | 6.89 s | 37.1 | 32 → 544 | degenerate (reduced target) |
| Full-depth, 4 steps, EOS-stop | 1 × 256 (EOS-halt) | 65.98 s | 64.45 s | 3.97 | 32 → 288 | EOS-heavy (empty after skip-special; #48291) |
| Full-depth, 16 steps, no EOS-stop | 1 × 256 (halt @11) | 111.61 s | 110.20 s | 2.32 | 32 → 288 | `你好！I'm doing well, thank you for asking. How can I help you today?` |

Raw metrics: `serving_smoke_reduced.json`, `serving_smoke_fulldepth.json`,
`serving_smoke_fulldepth_visible.json`. Metrics are **per-block** — there is no per-token TPOT in
block-diffusion, so `1000/mean_tpot_ms` is intentionally not reported.

## How it works

DiffusionGemma emits a **256-token block per decode step**. The whole denoise loop lives inside
the model forward; the vLLM adapter (`tt/generator_vllm.py`,
`DiffusionGemmaForCausalLM(HybridAttentionForCausalLM)`) is a thin interface over the vLLM-free
block-emission core (`tt/serving.py`, `BlockDiffusionServingSession`), which delegates to the
existing `tt/generate.py` engine. See `work_log.md` for the full delegation map and the exact
low-level methods called.

- `prefill_forward` → session `prefill` (write prompt K/V + build stateful `DenoiseLogitsAdapter`)
  + `decode_block` for block 0 (mirrors the AR "prefill returns first token").
- `decode_forward` → one `decode_block` (block N) per active request.
- On-device sampling reused via `tt.sampling` / `tt.denoise_loop`; only per-step `[B,L]`
  decision tensors are read back for the data-dependent halt; `[B,L,vocab]` logits stay on device.

## Registration

HF architecture `DiffusionGemmaForBlockDiffusion` → plugin auto-prefixes `TT` →
`TTDiffusionGemmaForBlockDiffusion`. Registered via `_register_model_if_missing` in the
tenstorrent/vllm fork's `register_tt_models()` (`plugins/vllm-tt-plugin/.../platform.py`). The
plugin is an upstream fork (not vendored in tt-metal), so the exact edit is provided as
`doc/vllm_integration/plugin_registration.patch` (registers the bare HF arch + `TT` aliases →
`models.experimental.diffusion_gemma.tt.generator_vllm:DiffusionGemmaForCausalLM`).

## Served context & datatype (from `doc/context_contract.json`)

- **max_model_len = 262144** (HF `text_config.max_position_embeddings`; 256 × 1024). No reduction
  below the advertised context. Full-depth 256K weights+KV fit on QB2 (29.704 GiB/chip used,
  2.163 GiB free); full-vocab Gumbel materialization OOMs at 256K, so serving uses the chunked /
  argmax sampler (`DG_VLLM_GUMBEL_MODE=argmax|chunked`) which fits.
- **Non-aligned prompt lengths:** any valid prompt length up to max_model_len is served. The
  256-token *output* block granularity is not an input constraint; prefill pads to a 32-tile
  multiple internally. The reduced-surface run uses a deliberately non-256-aligned prompt.
- **Datatype policy:** bf16 weights + bf16 KV cache + bf16 CCL (self-conditioning softmax/
  soft-embedding accumulate fp32). Full-model bring-up policy (dg-07 sweep not run; fp32 experts
  don't fit QB2). Matches the gemma4 vLLM bridge.
- **Batch/concurrency:** batch-1 single-user latency is the headline. One contiguous model cache
  backs one active sequence today; concurrent batched serving (up to 32) needs the vLLM
  paged-cache ownership change (#47488) + batched canvas decode (#47557). Largest tested value
  recorded in `work_log.md`.

## The #47488 upstream runner/scheduler change (scoped)

The TT runner assumes one committed token per decode step. To emit a 256-token block it must:
1. accept `num_out_tokens == canvas_length` (drop `assert num_out_tokens == 1`,
   `model_runner.py:2471`; allow `[num_reqs, 256]` sampled ids at `:2378`/`:1878`);
2. advance `num_computed_tokens` / `num_tokens` by `canvas_length` per decode step
   (scheduler + `_apply_sampled_tokens_to_state` `:2479`/`:2508`);
3. bound-check `start_idx + canvas_length <= max_model_len`;
4. build the per-request output with 256 tokens/step (`:2437`);
5. route the frozen-prompt-prefix read through the paged cache + per-request block tables
   (cache-ownership half of #47488) for concurrent batched serving.

Contract already respected: phase-based batching, no chunked prefill, spec decode hard-blocked,
APC force-disabled for sliding-window, no prompt-length divisibility requirement. Full line
references in `work_log.md`.

## Live serving verification (fresh vLLM)

This is a real vLLM OpenAI server serving DiffusionGemma on QB2, built fresh against the current
tt-metal — **not** the stale `0.14.0-80180b9-7678b70` ghcr image (its baked tt-metal predates
`models/experimental/diffusion_gemma`).

### Fresh, project-matching vLLM environment (host install — no container needed)

Installed into the current DG venv `/home/zni/venvs/tt-diffusion-gemma` (Python 3.12.12; `ttnn`
editable from `/home/zni/tt-metal`; transformers 5.12.1; torch 2.11.0+cpu). The tenstorrent/vllm
fork is cloned at `/home/zni/tt-vllm` (branch `dev`, head `6b4a3a7`); its TT plugin is the pip
package `plugins/vllm-tt-plugin`.

```bash
source /home/zni/venvs/tt-diffusion-gemma/bin/activate
cd /home/zni/tt-vllm
# base vLLM: empty device target (no CUDA/kernels — ext_modules=[] for VLLM_TARGET_DEVICE=empty,
# just gRPC codegen). Runtime deps come from requirements/common.txt, which pins NO torch, so the
# venv's torch 2.11 + transformers 5.12.1 + ttnn are preserved. (Build isolation pulls torch 2.10
# into a throwaway env only; it never touches the venv.)
VLLM_TARGET_DEVICE=empty uv pip install -e . \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install -e plugins/vllm-tt-plugin
```

Result: `vllm 0.1.dev1+g6b4a3a7b4.empty` + `vllm-tt-plugin 0.0.0`; `platform_plugin() ->
vllm_tt_plugin.platform.TTPlatform` (TT platform auto-activates because `ttnn` imports);
`ttnn`/`transformers`/`torch` versions unchanged. No profiling forward runs at startup
(`determine_available_memory` returns a dummy and overrides `num_gpu_blocks`).

### Patches applied to the fork (`/home/zni/tt-vllm`, a separate repo)

Both saved verbatim under this dir and re-appliable with `git apply` from the vllm repo root:

- `plugin_47488_registration.patch` — `register_tt_models()` registers `DiffusionGemmaForBlockDiffusion`
  + TT aliases → `models.experimental.diffusion_gemma.tt.generator_vllm:DiffusionGemmaForCausalLM`.
- `plugin_47488_model_runner.patch` — the **#47488** block-granular runner change. Generalizes the
  runner's hard "one committed token per decode step" assumption to N-token blocks
  (`num_out_tokens = sampled_token_ids.shape[1]`; N=1 keeps the autoregressive path byte-identical,
  N=256 for DiffusionGemma). Four edit sites in `model_runner.py`:
  1. `_get_output_tokens` (~:2757) — keep the full `[sz, num_out]` device sample
     (`reshape(sz, -1)`) instead of `reshape(sz)`; the 1-token logprobs path stays guarded.
  2. `sample_tokens` DP-pack (~:2252) — accept `[num_reqs, N>=1]` instead of asserting `shape[1]==1`.
  3. `_apply_sampled_tokens_to_state` (~:2887) — drop the `num_out_tokens==1` assert; write the
     `[start:start+N]` block into `token_ids_cpu`, advance `num_tokens` by N, extend the output list
     by N (both the sync and captured/async paths).
  4. `_build_runner_output` (~:2839) — emit all N tokens per request in the `ModelRunnerOutput`
     token list vLLM's engine core appends + detokenizes.

### Live server launch

```bash
source /home/zni/venvs/tt-diffusion-gemma/bin/activate
export TT_METAL_HOME=/home/zni/tt-metal PYTHONPATH=/home/zni/tt-metal
export MESH_DEVICE=P150x4 DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it
export DG_VLLM_GUMBEL_MODE=argmax VLLM_RPC_TIMEOUT=1800000
export VLLM_ENABLE_V1_MULTIPROCESSING=0   # single-process V1 so tracebacks surface in the log
python -m vllm.entrypoints.openai.api_server \
  --model /home/zni/dg_models/diffusiongemma-26B-A4B-it \
  --served-model-name diffusiongemma-26B-A4B-it \
  --max-model-len 1024 --max-num-seqs 1 --block-size 64 \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "enable_model_warmup": false}}' \
  --host 127.0.0.1 --port 8000
```

Launch notes (all live, from the running server):
- `--block-size 64` is **required** — the TT `get_num_available_blocks_tt` multiplies by
  `cache_config.block_size`, which vLLM leaves `None` for this arch unless set (the fork's
  `server_example_tt.py` always passes `64`). Omitting it fails at KV-cache init with
  `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'`. This is a launch flag, not
  a model bug.
- `enable_model_warmup: false` skips the AR two-phase trace warmup (block-diffusion warms lazily on
  the first prefill/decode).
- Server comes up healthy: `GET /health` → 200, `/v1/models` lists `diffusiongemma-26B-A4B-it`,
  arch resolves to `DiffusionGemmaForBlockDiffusion` → TT class, 30-layer 26B builds on the `(1,4)`
  mesh (`GPU KV cache size: 21,824 tokens` at max-model-len 1024).
- `--max-model-len 1024` here is the **inner-loop bring-up value** (small KV → fast iterate), not
  the served ceiling. The context contract's `262144` is unchanged and is the value for final
  headline evidence; 1024 was used only to iterate the runner patch quickly.

### Live #47488 blocker → fix (both observed on the running server)

Before the runner patch, the first `POST /v1/completions` ran the full prefill + 256-token canvas
denoise loop and then died in the engine core:

```
File ".../vllm_tt_plugin/model_runner.py", line 2757, in _get_output_tokens
    next_token_ids = _take(tt_out).reshape(sz)
RuntimeError: shape '[1]' is invalid for input of size 256
```

i.e. the model committed a 256-token block but the runner tried to collapse it to one token —
exactly #47488. After applying `plugin_47488_model_runner.patch` and relaunching (a `tt-smi -r`
reset + mesh-smoke recovery was needed once because a hard-killed EngineCore left the ethernet
cores un-reset — `TT_THROW ... assert_active_ethernet_cores_to_reset`), the same request succeeds.

### Live request + response (patched runner)

```bash
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"diffusiongemma-26B-A4B-it","prompt":"Hello, how are you?","max_tokens":32,"temperature":0}'
```
```
HTTP 200  (wall 254s)
{"id":"cmpl-ab17c7b1523666b5","object":"text_completion", ...
 "choices":[{"index":0,"text":"","finish_reason":"stop", ...}],
 "usage":{"prompt_tokens":6,"total_tokens":7,"completion_tokens":1}}
```
Server-side per-block metric (LIVE, through the vLLM engine):
```
[DiffusionGemma vLLM] prefill row=0 prompt_len=6 cache_len=32 block0 next_pos=288 steps=35 latency=252.259s
POST /v1/completions HTTP/1.1 200 OK
```

- **256-token block emitted through vLLM**: absolute position advanced `32 → 288`
  (cache_len 32 + one 256-token canvas). Non-256-aligned prompt (6 tokens, padded to cache_len 32).
- **Per-block metrics** (block-diffusion cost profile — never `1000/mean_tpot_ms`): block-0
  latency **252.26 s** at **35 denoise steps** (early-halted before the 48-step cap), full 30-layer
  26B, `argmax` sampler, bf16, TP=4, `(1,4)` mesh.
- `text:""` / `completion_tokens:1`: the committed argmax canvas placed an EOS at position 0, so
  vLLM stopped at the first token and `skip_special_tokens` yields empty text. This is the known
  **RUN-first #48291** fidelity behavior (bf16/MoE/TP=4 argmax commits EOS-heavy blocks; no
  temperature cushion), **not a serving regression** — it matches the recorded full-depth 4-step
  `serving_smoke` result.

### Remaining live blocker: the #47488 *scheduler* half — RESOLVED 2026-07-03

> **Update (2026-07-03):** this blocker is now **fixed** — see **§ Serving test suite** above
> (`plugin_47488_scheduler.patch` + the `generator_vllm.py` session-stop deferral) and the live
> 2-block serve. The original capture below is retained for the record.

A second live request with `ignore_eos:true` (`max_tokens:48`, so >1 token must survive per step)
returns HTTP 500 and dies at:

```
vllm/v1/core/sched/async_scheduler.py:53  AsyncScheduler._update_request_with_output
    request.num_output_placeholders -= len(new_token_ids)
    assert request.num_output_placeholders >= 0        # AssertionError
```

Root cause: the async scheduler reserves **one** output placeholder per scheduled step
(`_update_after_schedule`: `num_output_placeholders += 1 + spec`), but a block-diffusion step emits
up to `canvas_length = 256` tokens. Whenever more than the reserved count survives the stop-trim,
the placeholder budget underflows. The first request passed only because it committed EOS at
position 0 → trimmed to 1 token → within the 1-placeholder budget.

This is the **scheduler half of #47488** (generalize placeholder reservation **and** the
`num_computed_tokens += canvas_length` advance, for the prefill-block and each decode-block). It is
beyond the scoped `model_runner.py` change (which is done and verified), and a correct fix is
coupled to vLLM's KV/num-computed accounting; doing it hastily risks silently corrupting generation
(per the vLLM-integration skill's overlap warning), so it is left as the precise, live-captured
remaining blocker rather than hacked. **Net:** batch-1 single-block serving where the request stops
within block 0 (the realistic DG default-stop path) works live end-to-end; multi-token-survival /
multi-block serving needs the #47488 scheduler half.

## Serving test suite (live QB2, 2026-07-03 — #47488 scheduler half RESOLVED)

A full live serving pass was run on QB2 against the recorded server (config in
`live_vllm_serving.json`), reusing the intact venv + patched plugin (both #47488 patches
reverse-apply clean — no rebuild). Full per-request evidence + per-block metrics are in
**`serving_test_suite.json`**. Headline results:

- **Device-gated test:** `pytest tests/test_serving_block_contract.py` → **7 passed** (incl. the
  device block-emission case: prefill + 2×256-token blocks on device, 8.38 s).
- **7 live OpenAI requests** (4 `/v1/completions` + 2 `/v1/chat/completions` + the 2-block serve),
  all with **non-256-aligned prompt lengths** (6, 5, 25, 31, 21 tokens) — every one returned
  **HTTP 200** with a valid OpenAI response and a 256-token block emission through the vLLM engine.
- **LIVE 2-BLOCK SERVE (the #47488 scheduler-half goal):** one request (`ignore_eos`,
  `max_tokens 512`) emitted **two REAL committed 256-token denoise blocks** —
  `prefill block0 32→288 (35 steps, 178.2 s)` then `decode block=1 288→544 (48 steps, 232.8 s,
  stop=False)`, position advanced `32→288→544 = cache_len + 2×256`, 512 committed tokens, HTTP 200.
- **Per-block metrics** (block-diffusion cost profile — never `1000/mean_tpot_ms`): block-0
  latency 178–276 s at 17–48 denoise steps; the real block-1 232.8 s at 48 steps; 256 committed
  tokens per block.
- **Qualitative (RUN-first, HF/RUN control):** one chat request produced coherent on-topic text
  through the serving path — `The vast blue expanse holds endless secrets beneath its rolling
  waves.` — matching the recorded visible-dialogue RUN control; other prompts committed
  EOS/degenerate canvases (the RUN-first #48291 fidelity limit). **PASS — not a serving regression.**

### The #47488 scheduler half — reproduced live, then FIXED live

The dg-09 record left the scheduler half as an open blocker. This pass reproduced it **live on an
ordinary completion** (not just `ignore_eos`): `POST /v1/completions {"prompt":"The capital of
France is","max_tokens":16}` ran the full block-0 denoise (48 steps, 233 s) and then died in
`vllm/v1/core/sched/async_scheduler.py:53  assert request.num_output_placeholders >= 0`
(`EngineDeadError`, HTTP 500). Root cause: `AsyncScheduler` reserves exactly **one** output
placeholder per scheduled step (`_update_after_schedule`: `num_output_placeholders += 1 + spec`),
but a block-diffusion step commits up to `canvas_length = 256` tokens — so **any** request whose
committed block-0 does not place a stop token within the first reserved position underflows. (The
dg-09 default-stop request passed only because it EOS-stopped at canvas position 0 → 1 token.)

**Fix (this session), two coupled pieces:**

1. **`plugin_47488_scheduler.patch`** (fork `scheduler.py`) — a `TTScheduler._update_request_with_output`
   override generalizing the 1-token async accounting to N-token block commits: clamp
   `num_output_placeholders` at 0 instead of asserting; advance `num_computed_tokens` by `n-1` to
   keep the autoregressive "num_computed lags committed output by exactly 1" invariant the
   running-loop scheduler math relies on; skip prefix-cache bookkeeping for block commits (prefix
   caching is disabled here). **`n == 1` is byte-identical to `AsyncScheduler`** — the autoregressive
   path is unchanged, and the override self-activates only when a step commits >1 token (only
   block-diffusion does). Safety: the DiffusionGemma model owns its own KV/position (`page_table=None`,
   contiguous cache), so vLLM's `num_computed_tokens` governs only stop/bound/bookkeeping — never
   what the model computes — so a mis-count cannot silently corrupt generation; correctness is
   verified by counting the emitted blocks.
2. **`tt/generator_vllm.py` adapter fix** — serving sessions are now built with `stop_token_ids=[]`
   so the **session** does not self-finish on an internal EOS; vLLM owns the stop decision. Without
   this, a committed block containing an EOS forced `session.finished=True` and the next decode step
   returned synthetic `256×stop_id` padding (defeating `ignore_eos`, so the "2-block" request
   returned 512 tokens with only ONE real block). With it, a genuine block-1 denoise runs.

The launch also needs **`--generation-config vllm`** for multi-block, to drop the model's
`generation_config.json` default `max_tokens=256` cap (else a request stops at exactly block-0).

**Net:** batch-1 single-block serving works for ordinary requests (no longer only the EOS-at-0 case),
and multi-block (≥2 committed denoise blocks per request) works live end-to-end. Remaining beyond
this: concurrent batched multi-sequence serving = #47488 paged-cache ownership + #47557 batched
canvas decode.

## How to reproduce the block-emission evidence

```bash
# live 2-block serve (fork scheduler + runner patches applied; adapter session-stop deferral):
#   1) apply the three fork patches under doc/vllm_integration/ to /home/zni/tt-vllm
#   2) launch with --generation-config vllm (removes the model max_tokens=256 cap):
#      bash launch_server_gencfg.sh    # = live_vllm_serving.json launch + `--generation-config vllm`
#   3) curl -s :8000/v1/completions -d '{"model":"diffusiongemma-26B-A4B-it",
#         "prompt":"Hello, how are you?","max_tokens":512,"temperature":0,"ignore_eos":true}'
#      -> HTTP 200, 512 tokens, server log shows block0 32->288 then decode block=1 288->544

# reduced-surface serving driver on QB2 (proves the block contract on the free device)
TT_LOGGER_LEVEL=ERROR DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it \
python -m models.experimental.diffusion_gemma.demo.serving_smoke \
  --mesh P150x4 --num-layers 1 --max-seq-len 1024 \
  --num-blocks 2 --canvas-length 256 --max-denoising-steps 2 \
  --gumbel-mode argmax --local-files-only \
  --metrics-json doc/vllm_integration/serving_smoke_reduced.json

# CPU + device block-contract tests
DG_RUN_DEVICE=1 python -m pytest models/experimental/diffusion_gemma/tests/test_serving_block_contract.py -q
```

For the **live vLLM path** (fresh host-installed vLLM + patched runner), use the launch + curl
commands in **§ Live serving verification (fresh vLLM)** above.

## Limitations

- Live full-vLLM-engine serving works end-to-end (see § Serving test suite): real
  `/v1/completions` and `/v1/chat/completions` requests return HTTP 200 and emit 256-token blocks,
  and **multi-block (≥2 committed denoise blocks per request) is demonstrated live**. It requires
  the three fork patches applied to `/home/zni/tt-vllm` (a separate repo, not vendored in tt-metal):
  `plugin_47488_registration.patch`, `plugin_47488_model_runner.patch` (runner N-token block), and
  `plugin_47488_scheduler.patch` (scheduler N-token placeholder/num_computed accounting) — plus the
  `generator_vllm.py` session-stop deferral and the `--generation-config vllm` launch flag for
  multi-block. Remaining: concurrent batched **multi-sequence** serving = #47488 paged-cache
  ownership + #47557 batched canvas decode (still `--max-num-seqs 1`).
- The shared `run_vllm_server` readiness harness (`models/common/readiness_check/`) does not exist
  in this tt-metal checkout, so the live serve is driven directly against
  `vllm.entrypoints.openai.api_server` (the fork's documented server path) rather than that runner.
- Single active sequence per contiguous model cache; concurrent batched serving = #47488
  (paged-cache ownership) + #47557 (batched canvas decode). Served with `--max-num-seqs 1`.
- Text quality is RUN-first / degenerate until #48291 (not a serving regression — the live serving
  path reproduces the RUN-path output; the EOS-first block gives empty text under the default stop
  policy and the `ignore_eos` control surfaces the same canvas content vLLM would; see the live
  request above and `plan.md` R0.5).
- The vLLM adapter *class* methods (`prefill_forward`, `decode_forward`, `get_kv_cache_spec`,
  `allocate_kv_cache*`, `initialize_vllm_model`) import `vllm.*` / need the runner, so they are
  proven by static inspection + `py_compile` + the device-tested block-emission core
  (`BlockDiffusionServingSession`, driven by `serving_smoke.py` / the block-contract test), not by
  running the adapter class itself. Adapter-class execution coverage lands when #47488 + the
  installed plugin make the engine path runnable.
- No Tracy / tt-perf-report / device-profiler collection in the vLLM stage (per skill).
