# DiffusionGemma vLLM integration — work log (#47466 / #47488)

Stage: SERVING (dg-09). Branch: `diffusion-gemma-function`. Device: QB2 / `bh-qbge-06`
(4× Blackhole, chips 0–3, TP=4, `(1,4)` mesh). RUN-first (degenerate output OK until #48291).

## Summary of what was built

| Artifact | Path | Purpose |
|---|---|---|
| vLLM adapter | `tt/generator_vllm.py` | `DiffusionGemmaForCausalLM(HybridAttentionForCausalLM)` — block-granular bridge |
| Block-emission core (vLLM-free) | `tt/serving.py` | `BlockDiffusionServingSession` — prefill + per-block decode state machine |
| Reduced-surface driver | `demo/serving_smoke.py` | drives prefill→N blocks on device without vLLM; per-block metrics |
| Block-contract tests | `tests/test_serving_block_contract.py` | CPU scaffolding + device block-emission gate |
| Context contract | `doc/context_contract.json` | served max_model_len, datatype policy, block granularity |
| Plugin registration | `doc/vllm_integration/plugin_registration.patch` | `register_tt_models()` diff for the tenstorrent/vllm fork |

## Architecture: how the adapter delegates

The whole denoise loop lives inside the tt-metal model's forward. The adapter is a thin
vLLM interface over the existing `tt/generate.py` engine:

```
prefill_forward(tokens, prompt_lens, ...)         # per request row
  → BlockDiffusionServingSession.prefill()
      → tt.generate.prefill_prompt_tokens()        # write prompt K/V (causal)
      → make_generation_logits_fn_builder_from_checkpoint_state()  # stateful DenoiseLogitsAdapter
  → BlockDiffusionServingSession.decode_block()     # emit block 0 (mirrors AR "first token")
      → tt.generate.denoise_and_commit_block()
          → tt.denoise_loop.denoise_block()          # ≤48 on-device denoise steps:
              → tt.sampling.gumbel_max / token_entropy
              → tt.denoise_loop.entropy_budget_accept (sort/cumsum/scatter) / renoise
          → tt.generate.commit_canvas_tokens()       # append committed clean argmax K/V

decode_forward(...)                                  # per request row, per step
  → BlockDiffusionServingSession.decode_block()      # emit block N (256 tokens)
```

No separate host sampling path, no full-logits readback: only per-step `[B,L]` decision
tensors (argmax/entropy/sampled/accept) are read back for the data-dependent halt; the
`[B,L,vocab]` logits stay on device. This is the reuse of the on-device canvas sampling path
required by the goal.

Low-level `tt/generate.py` methods delegated to: `prefill_prompt_tokens`,
`make_generation_logits_fn_builder_from_checkpoint_state`, `denoise_and_commit_block`
(→ `denoise_block` + `commit_canvas_tokens`), `decode_generation`, `tokenize_prompt`, and the
seeded canvas/noise/Gumbel hooks (`make_seeded_host_canvas_init_fn`,
`make_seeded_host_noise_tokens_fn`, `make_seeded_{host,chunked,}_gumbel_noise_fn`).

## Block-granular emission contract (explicit)

- The adapter emits a **256-token block per decode step** (`canvas_length`), not one token.
- The async-decode contract is redefined **per-BLOCK**: `supports_async_decode`, stale-input
  refresh, and page-table update are once per emitted 256-token block, not per token. We
  declare `supports_async_decode=False` (the per-block async path is unproven without the
  #47488 runner; never advertise async without proof).
- Position advances by `canvas_length` per decode step (`next_pos += 256`).

## The #47488 upstream runner/scheduler change (scoped precisely)

The current tenstorrent/vllm TT runner assumes **one committed token per decode step**.
Verified against the plugin source (`/tmp/ttvllm_check/`, image `0.14.0-80180b9-7678b70`):

| What | Where | Change for a 256-block |
|---|---|---|
| Hard assert one token | `model_runner.py:2471` `assert num_out_tokens == 1` | accept `num_out_tokens == canvas_length` |
| Sampled-id shape | `model_runner.py:2378` (`view(sz,1)`), `:1878-1880` (`shape[1]==1` assert) | allow `[num_reqs, 256]` |
| Runner output build | `model_runner.py:2437,2444` (one-token lists) | emit 256 tokens/req |
| Host position advance | `_apply_sampled_tokens_to_state` `:2479-2489`, `:2508-2518` (`+1`) | advance `num_tokens` by 256 |
| Context bound | same fns, `assert end_idx <= max_model_len` | check `start_idx + 256 <= max_model_len` |
| Decode input read | `model_runner.py:931-934` (`num_tokens-1`, single last token) | correct next-block position |
| Scheduler advance | `scheduler.py` / `_update_states` (`num_computed_tokens` from scheduler) | advance by `canvas_length`/step |

Contract we already respect (verified): phase-based batching (a step is all-prefill OR
all-decode — `scheduler.py:30-100`, `model_runner.py:911-914`); no chunked prefill
(`platform.py:339-341`, re-asserted `model_runner.py:251-252`); speculative decoding
hard-blocked (`platform.py:342-344`); APC force-disabled for sliding-window models
(`platform.py:512-521`); no prompt-length divisibility requirement (`model_runner.py:924-928`).

## Cache ownership (the second half of #47488)

The diffusion denoise-read path reads the frozen prompt prefix from the model-owned
**contiguous** `tt_model.tt_kv_cache` via `ttnn.slice` (`tt/denoise_forward.py`
`read_prompt_kv_cache_by_layer` / `read_prompt_kv_cache_slice`), NOT from a vLLM paged block
pool. Serving therefore runs in the generator/standalone cache-ownership mode: the model owns
its `max_model_len` cache (`create_kv_cache=True`) and is driven with `page_table=None`;
`allocate_kv_cache[_per_layer]` returns those existing handles (no double allocation). One
contiguous cache backs one active sequence. Routing the frozen-prefix read through a vLLM
paged cache + per-request block tables (concurrent batched serving) is #47488 + batched canvas
decode #47557.

## Datatype policy served

bf16 weights + bf16 KV cache + bf16 CCL (self-conditioning softmax/soft-embedding accumulate
fp32). This is the full-model bring-up policy (`create_tt_model` default); the dg-07
datatype-sweep stage has not run, and fp32 experts do not fit QB2 DRAM, so bf16/MoE is the
fastest policy that fits. Matches the gemma4 vLLM bridge. See `doc/context_contract.json`.

## Served context

max_model_len = 262144 (HF `text_config.max_position_embeddings`, = 256 × 1024). No reduction
below the advertised context. Physical-limit note: full-depth 256K weights+KV fit
(29.704 GiB/chip used, 2.163 GiB free); full-vocab Gumbel materialization OOMs at 256K, so
serving uses the chunked / argmax sampler (`DG_VLLM_GUMBEL_MODE`) which fits. See context
contract.

## Serving environment reality (why no live `run_vllm_server`)

- vLLM + the TT plugin exist **only inside a container image**
  (`ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.14.0-80180b9-7678b70`);
  no host vLLM, no running container.
- The container's baked tt-metal is stale (`80180b9`): no `models/experimental/diffusion_gemma`,
  no DiffusionGemma registration, and `models/common/readiness_check/` (the skill's
  `run_vllm_server`) **does not exist anywhere** on host or in the image.
- The upstream runner's `assert num_out_tokens == 1` hard-blocks block emission until #47488.

⇒ The skill's fallback applies verbatim: "If the runner+scheduler cannot accept block output,
the required upstream change is scoped (#47488) and the adapter is written to that contract."
Serving-path evidence is produced by the reduced-surface driver on the free device.

## Device runs (reduced-surface serving driver)

Command:
```
TT_LOGGER_LEVEL=ERROR DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it \
python -m models.experimental.diffusion_gemma.demo.serving_smoke \
  --mesh P150x4 --num-layers 1 --max-seq-len 1024 \
  --num-blocks 2 --canvas-length 256 --max-denoising-steps 2 \
  --gumbel-mode argmax --local-files-only \
  --metrics-json doc/vllm_integration/serving_smoke_reduced.json
```

### Reduced-surface run (1 layer) — inner-loop contract proof (2026-07-03, QB2)

`DG_VLLM_SERVING_SMOKE_SUCCESS prompt_len=24 prompt_aligned_256=False cache_len=32 blocks=2
tokens=512 canvas=256 ttft_s=8.238 mean_block_latency_s=6.894 tokens_per_block_per_s=37.14
final_next_pos=544 text_chars=4652`

- Non-256-aligned prompt (24) served; padded to cache_len=32; **2 × 256-token blocks emitted**;
  position advanced 32 → 544 (cache_len + 2×256). Per-block metrics captured (see
  `serving_smoke_reduced.json`). DRAM: baseline 0.0 → post-build 1.202 → post-prefill+block0
  2.616 GiB/chip.
- Output is degenerate `<unused*>` tokens — **expected for the 1-layer reduced target**; this is
  the block-emission *contract* proof (prefill → committed blocks → detok), not the model's
  output. Reduced target is an inner-loop tool only (per skill), not final accuracy/perf.

### Full-depth run (30 layers) — final block-emission + qualitative evidence (2026-07-03, QB2)

Command: same driver, no `--num-layers` (full 30), prompt `"Hello, how are you?"`,
`--max-seq-len 1024 --num-blocks 2 --max-denoising-steps 4 --gumbel-mode argmax`.
post-build DRAM 13.268 GiB/chip (matches plan's ~13.24 backbone footprint). prompt_len=19
(non-aligned).

`DG_VLLM_SERVING_SMOKE_SUCCESS prompt_len=19 prompt_aligned_256=False cache_len=32 blocks=1
tokens=256 canvas=256 ttft_s=65.980 mean_block_latency_s=64.450 tokens_per_block_per_s=3.97
final_next_pos=288 text_chars=0`

- Full 30-layer block emission works: **256-token block committed**, position advanced 32 → 288.
  Requested 2 blocks but block 0 committed an EOS token → the session's per-block stop check
  halted generation after 1 block (correct serving behavior). Per-block metrics: **TTFT 65.98 s,
  block latency 64.45 s, 3.97 tokens/block/s** at 4 denoise steps (argmax). (Slower than the
  1-layer proof because the per-step 262144-vocab logits + 30-layer forward dominate; this is the
  block-diffusion cost profile, reported per-block, never as `1000/mean_tpot_ms`.)
- `text_chars=0`: the committed block is EOS/special-heavy, so `skip_special_tokens` yields empty
  text — the **expected RUN-first #48291 degeneracy** (bf16/MoE/TP=4 argmax ≈50% vs HF; diffusion
  commits the clean argmax with no temperature cushion), not a serving regression. See the
  visible-text run below and the qualitative verdict.
- Artifacts: `serving_smoke_fulldepth.json`, `/tmp/dg_serving_smoke_fulldepth.log`.

### Full-depth visible-text run (qualitative control comparison)

Same driver, full 30 layers, `--num-blocks 1 --max-denoising-steps 16 --gumbel-mode argmax
--disable-eos-stop` (EOS-stop off so non-EOS positions surface, matching the recorded
`text_demo` visible-dialogue control that produced coherent text at more steps).

`DG_VLLM_SERVING_SMOKE_SUCCESS prompt_len=19 prompt_aligned_256=False cache_len=32 blocks=1
tokens=256 canvas=256 ttft_s=111.611 mean_block_latency_s=110.195 tokens_per_block_per_s=2.32
final_next_pos=288 text_chars=66`

- **Coherent visible output through the serving path:**
  `你好！I'm doing well, thank you for asking. How can I help you today?`
- This matches the recorded `text_demo` RUN control almost exactly (plan.md 2026-07-02:
  `你好！I'm doing well, thank you for asking! How can I help you today?`). ⇒ the serving
  block-emission path is **not a serving regression** — it faithfully reproduces the model's RUN
  output. The denoise loop early-halted at step 11/16 (`halted=True`) via the stable+confident
  stopping criterion, exercising the data-dependent halt through the serving session.
- Metrics: TTFT 111.6 s, block latency 110.2 s, 2.32 tokens/block/s (16-step cap; block-diffusion
  cost profile — reported per-block, never `1000/mean_tpot_ms`).
- Artifact: `serving_smoke_fulldepth_visible.json`, `/tmp/dg_serving_smoke_visible.log`.

### Qualitative verdict (HF-vs-TT control)

- **Verdict: PASS (RUN-first).** The serving path reproduces the model's RUN-path output. With
  sufficient denoise steps (16, halted 11) it emits coherent text identical to the recorded RUN
  control; with the fast RUN config (4 steps / EOS-stop) it emits the expected EOS-heavy
  degenerate block (#48291, not a serving bug).
- **Control:** the recorded `text_demo` visible-dialogue RUN output (same prompt, same on-device
  sampling + KV path) and the R0.5 HF-vs-TT committed-argmax replay (`demo/replay_hf_tt.py`,
  `plan.md` R0.5 — HF itself commits mostly EOS on this bf16/MoE/TP=4 path; the residual non-EOS
  miss is at the logits level, deferred with #48291). The serving output is **not materially
  worse** than the prompt-correct control → no serving regression.
- The shared `models/common/readiness_check/check_degenerate_output.py` degeneracy gate is
  unavailable in this checkout (the whole `readiness_check` harness is absent — see below); the
  visible-text vs RUN-control comparison serves the same purpose here.

### gemma4 isolation gate

`git status --porcelain -- models/demos/gemma4/` is empty — this stage edits **zero** gemma4
files (all changes under `models/experimental/diffusion_gemma/`). The `git diff main --
models/demos/gemma4/` delta is pre-existing branch state (main advanced: #47817/#47556/#47172;
plus the deferred F1/R-new decode-isolation item), not introduced by this stage.

## Process / device hygiene

- Device audited free before runs: 4× Blackhole (chips 0–3), no `vllm`/`EngineCore` processes.
- The driver opens/closes its own mesh (`_open_mesh_device`/`_close_mesh_device`, fabric
  disabled on close). No profiler/Tracy in the serving stage (per skill). No leftover
  processes after runs.

## SHAs

<!-- filled in at commit time -->
