# DiffusionGemma bring-up on tt-metal — agent guide

Working context for bringing up **Google DiffusionGemma 26B-A4B-it** on Tenstorrent hardware.
Tracking issue: **tenstorrent/tt-metal#47452** (label `DiffusionGemma`). Work branch: `zni/diffusion-gemma-bringup`.

## What it is
- HF [`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it) (released 2026-06-11, Apache-2.0). transformers class `DiffusionGemmaForBlockDiffusion`, `model_type=diffusion_gemma`.
- A discrete **text-diffusion** LLM. **Fine-tuned from the `google/gemma-4-26B-A4B` checkpoint** → the text backbone is identical to the Gemma-4 26B-A4B MoE; only the generation procedure + a few extra weights differ.
- Multimodal input (text / image / video → text). Bring up **text-first**.

### Text backbone config
30 layers · hidden 2816 · 16 heads / 8 KV · head_dim 256 · MoE 128 experts top-8 + 1 shared MLP · moe_intermediate 704 · sliding-window 1024 interleaved with full-attention · dual RoPE (θ=1e6 / 1e4) · final logit softcap 30 · vocab 262144 · canvas_length 256.
Vision tower: `gemma4_vision` (SigLIP-family: 27 layers, hidden 1152, patch 16, 280 tokens/img).

## How it generates (the key difference from autoregressive Gemma)
Block-autoregressive **multi-canvas diffusion**. Per 256-token block, the SAME backbone runs in three phases (shared weights, selected by attention mode):
1. **Prefill (encoder, causal)** — encode the prompt, write KV.
2. **Denoise (decoder, bidirectional)** — iteratively denoise a 256-token "canvas"; cross-attends to the prompt by concatenating encoder K/V in front of canvas K/V (prefix-style, not a separate cross-attn module). Read-only on the **prompt/committed** KV; the canvas's own K/V is **recomputed every step** (canvas tokens change each step → a 256-token mini-prefill against the frozen prefix) and is **never written into the frozen cache until commit**.
3. **Commit (encoder, causal)** — re-encode the finished canvas, append its KV, emit 256 tokens. Then the next block.

**Noise = RANDOM tokens, not a `[MASK]` token.** The canvas is initialized to random token ids and rejected positions are re-noised to random tokens (uniform discrete diffusion, not absorbing-mask).

**Per denoise step** (≤48 steps, often halts 12–16): temperature-scale (linear 0.8→0.4) → **Gumbel-max** `argmax(logits/T + gumbel)` → **entropy-budget acceptance** (accept most→least confident until accumulated entropy exceeds a budget) → re-noise the rest → stop when the argmax canvas is stable AND mean entropy < threshold. **Commit = clean argmax**, not the noisy sampled values.

**Self-conditioning** (extra weights beyond the backbone): previous-step softmax → probability-weighted average of token embeddings → through a small **gated MLP** → added to canvas embeddings. Active only in denoise; zeroed on encoder passes.

Algorithm reference: transformers `modeling_diffusion_gemma.py`; vLLM blog <https://vllm-project.github.io/2026/06/10/diffusion-gemma.html>.

## Reuse vs build (tt-metal)
**Reuse — the backbone is already in-repo.** `models/demos/gemma4/` is a near-complete, trace-compatible on-device Gemma-4 26B-A4B MoE: `tt/model.py`, `tt/moe.py`, `tt/router.py`, `tt/experts/`, `tt/shared_mlp.py`, `tt/attention/`, weight loading, CCL/TP. MoE / softcap / dual-RoPE / weight-loading match the target. On-device sampling: `models/common/sampling/generator.py`.
**Already present — do NOT rebuild:** K=V tying for full-attn layers (`attention_k_eq_v`, `tt/attention/weights.py:73`), scaleless V-norm (`tt/attention/prefill.py:61`, `decode.py:84`), the bounded-sliding hybrid KV cache, tokenizer/chat-template.

**Net-new (the real work):**
1. **Bidirectional canvas attention** — gemma4 prefill SDPA is hardcoded `is_causal=True` (`tt/attention/prefill.py:126,264`, `operations.py:333`); add a non-causal path (explicit `attn_mask`; ref `models/experimental/pi0/tt/ttnn_gemma.py:320` — `scaled_dot_product_attention(attn_mask=…, is_causal=False)` — and `models/tt_dit/encoders/gemma/model_gemma.py:253`). Denoise wants a **symmetric sliding window 2W+1** on local layers — ttnn SDPA has the symmetric-window geometry, but `sliding_window_size` and `attn_mask` are mutually exclusive (`sdpa_device_operation.cpp:67-68`), so bake the window into the mask. **Define the 2D mask geometry explicitly** (#47462): canvas↔canvas bidirectional (local layers symmetric 2W+1; does the window extend over the prompt prefix or is the prompt fully visible?), canvas→prompt visibility per layer type, canvas absolute/RoPE positions offset by `prompt_len`, and how the `[256, prompt_len+256]` mask is chunked for long prompts — don't ship a non-causal path that only works for an isolated 256 canvas. Long context (>32768) needs a non-causal path — the existing chunked-prefill long-context workaround is causal-only (`operations.py:25-29`, `prefill.py:106-130`).
2. **KV-cache phase state machine** (#47474, prereq) — gemma4 (re)writes KV every forward; no "read-frozen encoder KV" mode, and the bounded-sliding circular buffer would wrap on commit-append. Define **three KV storage classes**: (i) frozen prompt/committed KV (paged cache, read-only); (ii) **per-step canvas K/V** (recomputed every denoise step — ephemeral activations or a dedicated scratch zone — **must NOT be written into the cache during denoise**, or it corrupts the frozen prompt KV); (iii) commit-append (write the finished canvas's KV once). Specify the page / circular-buffer mapping for both local and full-attention layers.
3. **Discrete-diffusion decode loop** (#47463) — entropy = softmax+log+mul+sum; Gumbel-max; renoise = where+scatter. **Entropy-budget acceptance needs sort-by-confidence + cumulative-entropy threshold + scatter-back.** The primitives exist (`ttnn.sort` returns values **and** indices, `ttnn.cumsum`, `ttnn.topk`); the **unproven** parts are: (a) the **scatter-back / inverse-permutation** mapping per-rank accept decisions to the original canvas positions; (b) the **data-dependent cutoff** (accept until cumulative entropy exceeds budget) under **static Metal Trace**; (c) trace-ability + perf of the whole sort→cumsum→scatter chain over the 256-position axis. ⇒ **spike this before committing the loop** (#47463/#47472); the spike may conclude a new op/kernel or a small host fallback (256-element readback) is needed. Treat "no new kernels" as a hypothesis, not a fact. No `[MASK]` token — random-token init/renoise.
4. **Self-conditioning gated MLP** — extra module + weights.
5. **On-device canvas sampling** (#47472) — per-position over the 256 canvas; keep logits/probs on device.

**Do first (two stages, don't conflate them):** (1) bring up the existing **gemma4 implementation path** unchanged and PCC-validate vs HF on the **gemma4 checkpoint**; (2) then point it at the **DiffusionGemma checkpoint** and validate the **text weight mapping + causal-pass PCC** — catch missing/renamed weight keys, the extra self-conditioning weights, and config diffs (v-norm, K=V, canvas_length, …) — **before** adding any diffusion delta.

## Serving: tenstorrent/vllm TT plugin (NOT upstream vLLM)
We serve via the [tenstorrent/vllm](https://github.com/tenstorrent/vllm) TT plugin (`plugins/vllm-tt-plugin`, `dev` branch). The plugin has its own model_runner/worker/scheduler and **the tt-metal model owns its forward + attention + KV** — the runner passes only tokens/page_table/kv_cache/start_pos/prompt_lens/sampling. Consequences:
- vLLM's GPU attention backends / `model_states` / per-request causal tensor / `DiffusionSampler` do **not** run here. Bidirectional attention + the whole denoise loop live **inside the tt-metal model's `prefill_forward`/`decode_forward`** (loop internally, commit a 256-block, advance `start_pos`).
- **Speculative decoding is hard-blocked** (`platform.py:342`). (Upstream rides DiffusionGemma on engine spec-decode plumbing; we have no equivalent. tt-metal's only spec-decode is gemma4's draft-model EAGLE/MTP `tt/spec_decode.py`, not a reusable framework.)
- **Chunked prefill unsupported** (`platform.py:339-341`).
- **Continuous batching is phase-based** (a step is all-prefill OR all-decode; `docs/SCHEDULING.md`).
- **APC is model-gated and force-disabled for sliding-window models** (`platform.py:512-521`) → likely off for Gemma. **Not a Functional gate** — the parent #47452 lists APC under success criteria, but treat it as best-effort unless the plugin's sliding-window gating changes.
- Integration = implement a TT model class (`initialize_vllm_model` `loader.py:39`, `allocate_kv_cache_per_layer`, `prefill_forward` `model_runner.py:1999`, `decode_forward` `async_decode.py:473`, `model_capabilities`) + register in `register_tt_models()` (`platform.py`; HF arch auto-prefixed `TT`). Copy the existing gemma4 bridge `TTGemma4ForCausalLM`.
- Emitting a 256-token block per decode step likely needs an upstream tenstorrent/vllm runner+scheduler change (#47488).

## Correctness / determinism gotchas
- **Determinism:** token-for-token PCC vs torch requires **injecting the torch run's exact Gumbel noise + random-renoise token ids into the TT path** — on-device RNG won't match bit-exactly. Reserve regenerated noise for distributional checks.
- **top-k / top-p is NOT shipped** in the reference (transformers defers it; vLLM PR #45429 open/unmerged). Target shipped sampling first (temperature schedule + Gumbel-max + entropy-budget); treat top-k/p as forward-looking, not a gate.
- gemma4 has **no entropy computation** today — the entropy harness is net-new.
- The PCC harness must validate the diffusion *decisions* (entropy values, Gumbel-max argmax agreement, multi-step trajectory), not just logits — bfp8 small-probability drift can flip accept/renoise.

## Hardware staging
- 26B-A4B currently runs on **T3K (1×8) only** (`models/demos/gemma4/README.md`). **Validate the backbone on T3K first.**
- Near-term product target: **QB2 only**; **BHG (Galaxy) adapted later**, then broader HW. QB2 fit + Galaxy 4×8 TP enablement tracked in #47487.

## Milestones
- **Foundation (exit criteria)** — causal Gemma-4 26B-A4B backbone PCC-validated vs HF **on T3K** (#47461) + torch ref / PCC harness (#47468). This is a correctness gate, not a product target. QB2/Galaxy enablement (#47487) is a **hard prereq of the Functional milestone**, not of foundation.
- **Functional** — text-only · batch 1 · max ctx 256K · on-device sampling · vLLM · QB2 (gated on #47487). Perf TTFT ~50%, t/s/u ~100%.
- **Functional +** — + image/video inputs · all resolutions · + BHG/broader HW. Perf t/s/u ~200%.
- **Complete** — everything.
- Batching: batch=1 first, then batch=4 (#47557).

## Issue map (label `DiffusionGemma`, parent #47452)
- **Foundation:** #47468 torch ref + PCC harness · #47461 causal backbone (T3K) · #47487 HW enablement (QB2 + Galaxy)
- **Functional core:** #47474 KV phase state machine (prereq) · #47462 bidirectional forward · #47463 decode loop · #47472 on-device sampling · #47557 batched decode · #47464 functional e2e · #47465 perf · #47466 vLLM integration · #47488 vLLM block-granular runner/scheduler
- **Functional +:** #47467 multimodal
- **Infra / optional:** #47475 quant dequant · #47489 CI

## Methodology / references
`tech_reports/ttnn/TTNN-model-bringup.md` · `models/docs/model_bring_up.md` · `tech_reports/LLMs/llms.md` · `tech_reports/ttnn/comparison-mode.md`. PCC via `tests/ttnn/utils_for_testing.py`; profiling via `tools/tracy/profile_this.py`.

## Conventions
- **Commit messages must NOT include a `Co-Authored-By` trailer.**
