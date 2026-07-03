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
- **Serving path:** written to the block-granular contract. A live full-vLLM-engine run is
  **blocked upstream** — the tenstorrent/vllm TT runner asserts one committed token per decode
  step (`model_runner.py:2471`), and vLLM + the plugin exist only in a stale container image
  with no `run_vllm_server` harness present. The block-emission contract is proven on the free
  QB2 device by the reduced-surface driver `demo/serving_smoke.py`; the runner/scheduler change
  is **#47488** (scoped below).

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

## How to reproduce the block-emission evidence

```bash
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

## Limitations

- Live full-vLLM-engine serving is blocked on #47488 (runner one-token assert) + the
  container/image gating (stale image, no `run_vllm_server`, no host vLLM). The adapter is
  written to the block contract and proven via the reduced-surface driver.
- Single active sequence per contiguous model cache; concurrent batched serving = #47488 + #47557.
- Text quality is RUN-first / degenerate until #48291 (not a serving regression — the serving
  path reproduces the RUN-path visible-dialogue control; see `plan.md` R0.5 and the visible-text
  run above).
- The vLLM adapter *class* methods (`prefill_forward`, `decode_forward`, `get_kv_cache_spec`,
  `allocate_kv_cache*`, `initialize_vllm_model`) import `vllm.*` / need the runner, so they are
  proven by static inspection + `py_compile` + the device-tested block-emission core
  (`BlockDiffusionServingSession`, driven by `serving_smoke.py` / the block-contract test), not by
  running the adapter class itself. Adapter-class execution coverage lands when #47488 + the
  installed plugin make the engine path runnable.
- No Tracy / tt-perf-report / device-profiler collection in the vLLM stage (per skill).
