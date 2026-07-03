---
name: diffusion-gemma
description: Shared model context for the DiffusionGemma 26B-A4B-it bring-up on Tenstorrent hardware. Load this in EVERY DiffusionGemma stage before any stage-specific skill. It states what the model is, how it generates (block-autoregressive text diffusion, NOT autoregressive token-by-token), the reuse-vs-build boundary and the hard rule against editing the shared gemma4 backbone, how correctness is judged (diffusion decisions + injected noise, not teacher-forcing top-k), the QB2 memory budget, the module map, and the issue/stage ownership. Every other skill in this pipeline (functional-decoder, full-model, optimize, multichip, datatype-sweep, tt-enable-tracing, vllm-integration, tti-release, stage-review, ...) is a GENERIC autoregressive-LLM skill; this skill is the override that maps each of their autoregressive assumptions onto the diffusion path.
---

# DiffusionGemma — shared bring-up context

**Load this skill first in every DiffusionGemma stage.** The other skills in `models/experimental/diffusion_gemma/.agent/skills/`
were written for a generic *autoregressive* LLM autoport pipeline. This skill is the authority
that overrides their autoregressive assumptions for the text-diffusion path. When a stage skill and
this skill disagree, **this skill wins**, and the per-skill "DiffusionGemma adaptation" sections
inside those skills spell out the specific overrides.

Source of truth for the actual work lives in the module itself:
`models/experimental/diffusion_gemma/plan.md` (full plan + live status),
`models/experimental/diffusion_gemma/AGENTS.md` (terse working context),
`models/experimental/diffusion_gemma/QB2_MEMORY_BUDGET.md` (memory budget). Read `plan.md`
Part 0 first — it is the live "what to do next".

## This is not a greenfield bring-up

The DiffusionGemma module is **already substantially built and has RUN on QB2**. Do not
re-author the backbone or the diffusion delta from scratch. As of the current branch:
foundation (torch reference + PCC harness, causal backbone PCC on QB2, QB2 fit) is done; the
device pieces — KV-phase machine, bidirectional masked SDPA (both ≤32768 and the long-prompt
>32768 path), on-device canvas sampling, and the decode-loop control flow — are built and
validated; and the first real-26B multi-block prompt→text run passed on QB2. The remaining work
is **hardening, correctness (#48291), long-context/256K fit, precision, perf, and serving** — the
later stages of this pipeline, not the early "write a decoder" stages.

Consequences for how you use these skills:
- The generic stages 01–04 (author `functional_decoder.py`, then optimized/multichip/optimized-multichip
  decoders from scratch) **do not apply as fresh authoring**. The "decoder" is the gemma4 backbone
  (already tensor-parallel and optimized) plus the **diffusion delta** (bidirectional attention,
  three-phase KV, denoise loop, self-conditioning, canvas sampling), which already exists.
- Treat every generic stage as "validate / harden / optimize / integrate the EXISTING code",
  not "create the file the generic prompt names".

## What it is

Google **DiffusionGemma 26B-A4B-it** (`google/diffusiongemma-26B-A4B-it`, transformers class
`DiffusionGemmaForBlockDiffusion`, `model_type=diffusion_gemma`). A discrete **text-diffusion**
LLM **fine-tuned from `google/gemma-4-26B-A4B`** — so the text backbone is byte-for-byte the
in-repo Gemma-4 26B-A4B MoE; only the generation procedure and a few extra weights differ.
Bring up **text-first**; multimodal is later (Functional+).

Text backbone: 30 layers · hidden 2816 · 16 heads / 8 KV · head_dim 256 · MoE 128 experts top-8
+ 1 shared MLP · `moe_intermediate` 704 · sliding-window 1024 interleaved with full-attention ·
dual RoPE (θ=1e6 full / 1e4 sliding) · final logit softcap 30 · vocab 262144 · `canvas_length` 256 ·
`max_position_embeddings` 262144 (256K context).

## How it generates (the core difference from autoregressive Gemma)

Block-autoregressive **multi-canvas diffusion**. Per 256-token block, the **same backbone / shared
weights** runs in three phases, selected by attention mode:

1. **Prefill (encoder, causal)** — encode the prompt; write KV.
2. **Denoise (decoder, bidirectional)** — iteratively denoise a 256-token *canvas*. Cross-attends
   to the prompt by **concatenating encoder K/V in front of canvas K/V** (prefix-style, no separate
   cross-attention module). **Read-only** on the prompt/committed KV; the canvas's own K/V is
   **recomputed every step** (a 256-token mini-prefill against the frozen prefix) and is **never
   written into the frozen cache until commit**.
3. **Commit (encoder, causal)** — re-encode the finished canvas, append its KV, emit 256 tokens.
   Then the next block.

**Noise = RANDOM token ids, not a `[MASK]` token.** The canvas is initialized to random token ids;
rejected positions are re-noised to random tokens (uniform discrete diffusion, not absorbing-mask).

**Per denoise step** (≤48 steps; early-halt is data-dependent): temperature-scale (linear 0.8→0.4)
→ **Gumbel-max** `argmax(logits/T + gumbel)` → **entropy-budget acceptance** (accept most→least
confident until accumulated entropy exceeds a budget) → re-noise the rest → stop when the argmax
canvas is stable AND mean entropy < threshold, or the step cap. **Commit = the clean argmax**, not
the noisy sampled values.

**Self-conditioning** (extra weights beyond the backbone): previous-step softmax → probability-weighted
average of token embeddings → small **gated MLP** → added to canvas embeddings. Active only in
denoise; **zeroed on encoder (prefill/commit) passes**.

## The five structural differences every stage must respect

These are the reasons the generic autoregressive skills need overriding. When any skill says
"decode / token / greedy / teacher-forcing / autoregressive", map it through this table.

1. **Reuse, don't author. And never edit the shared backbone.** The backbone is
   `models/demos/gemma4/` — already tensor-parallel (TP=4), MoE-sharded, trace-compatible. The
   net-new work is the **diffusion delta** under `models/experimental/diffusion_gemma/`.
   **HARD RULE: do NOT modify `models/demos/gemma4/` or any other shared directory.** `git diff main
   -- models/demos/gemma4/` must stay empty. Any footprint change the backbone needs for the
   diffusion path (decode L1/DRAM reductions, commit-append) belongs in DiffusionGemma-local code
   (e.g. `tt/commit_decode.py`) that composes over the backbone — never in-place edits to gemma4.
   (This is the F1/F2 isolation requirement in `plan.md`; the R-new risk is exactly ungated
   shared-gemma4 edits.)
2. **"Decode" = the denoise loop, not token-by-token.** There is no per-token `tt_out_tok`
   feedback, no advancing current-position cursor within a block, no greedy/top-k next-token
   sampling. The unit of work is a **denoise step over a fixed 256-token canvas** (≤48 steps/block),
   then a **commit** that appends 256 tokens and advances position by 256. Per-token metrics
   (TTFT is still meaningful for prefill; "t/s/u" must be derived from per-step × steps + commit,
   or reported as tokens-per-block / blocks-per-second, never as `1000/mean_tpot_ms`).
3. **Correctness = diffusion decisions, not teacher-forcing top-k.** Do NOT gate on AIME24
   teacher-forcing top-1/top-5. Validate the diffusion **decisions**: per-step entropy values,
   Gumbel-max argmax agreement, entropy-budget accept/renoise agreement, and multi-step trajectory
   vs the torch reference. Token-for-token determinism requires **injecting the torch run's exact
   Gumbel noise + random-renoise token ids** into the TT path (on-device RNG will not match
   bit-exactly) — reference `reference/generate.py` replay hooks (`make_replay_canvas_init_fn`,
   `make_replay_noise_fn`) and `demo/replay_hf_tt.py`. `bfp8` small-probability drift can **flip**
   an accept/renoise decision even when the argmax is unchanged, so any precision metric must be
   sensitive to small-probability mass — top-1/top-5 explicitly is not.
4. **Data-dependent control flow vs static Metal Trace.** Entropy-budget acceptance is a
   data-dependent cutoff (sort → cumsum → scatter/inverse-permutation over the 256 canvas positions),
   and early-halt is data-dependent — both collide with static trace capture. The trace-safe shape
   is a **fixed step budget (always run the max, ≤48) with an on-device tensor mask**, never a host
   branch or a variable-length slice. Keep the cutoff decision as a device tensor; use tensor-valued
   index arguments for scatter/gather so indices stay device-resident. Warm the program cache for
   `sort`/`cumsum`/`scatter`/`gather`/entropy at the exact fixed canvas shape and argument values.
5. **Serving is block-granular through the tenstorrent/vllm TT plugin (a fork, not upstream).**
   The whole denoise loop lives **inside** the tt-metal model's `prefill_forward`/`decode_forward`,
   which emits a **256-token block per step**, not one token. Speculative decoding is **hard-blocked**
   in the plugin; **chunked prefill is unsupported**; continuous batching is **phase-based**; APC is
   force-disabled for sliding-window models (likely off for Gemma). Emitting a block per step likely
   needs an upstream tenstorrent/vllm runner+scheduler change (#47488). vLLM's GPU attention
   backends / DiffusionSampler / per-request causal masks do NOT run here — the tt-metal model owns
   its forward + attention + KV.

## QB2 hardware & memory budget

Near-term target is **QB2 only** (`P150x4` = 2 liquid-cooled Blackhole PCIe cards, 4 Tensix
processors, TP=4; 480 Tensix cores, 720 MB SRAM, 128 GB DDR6, ~1024 GB/s). Galaxy 4×8 is later.
The 26B-A4B backbone fits and runs on QB2 (experts TP-sharded; ~13.24 GiB/chip post-build at small
context). Sizing the batch/context ceiling must account for **weights + 256K KV + per-step canvas
K/V scratch (#47474 storage class ii) + non-causal long-context mask buffers (#47462)** — a clean
short-prompt causal run does NOT de-risk the 256K diffusion fit. See `QB2_MEMORY_BUDGET.md` and
#47487. "256K context" is the capacity limit for `prompt + generated`, not a requirement that
prompts be 256K tokens.

## Module map (what already exists — read before touching)

Under `models/experimental/diffusion_gemma/`:
- `tt/model.py` — `DiffusionGemma4Model`, wraps the gemma4 backbone; `tt/diffusion_attention.py`
  — bidirectional/masked SDPA + the staged-GQA maskless fallback (`_manual_gqa_attention`);
  `kv_phase.py` (module root) / three-phase KV; `tt/denoise_forward.py`, `tt/denoise_loop.py`
  — denoise step + loop; `tt/self_conditioning.py`; `tt/sampling.py`, `tt/sampling_params.py`
  — on-device canvas sampling; `tt/commit_decode.py` — DiffusionGemma-local commit-append (keeps
  the backbone untouched); `tt/generate.py` — prompt→text entry (`generate_text`,
  `generate_text_from_checkpoint_state`, multi-block commit loop).
- `reference/` — torch/HF reference: `hf_reference.py`, `denoise_loop.py`, `generate.py`,
  `sampling.py`, `self_conditioning.py`, `attention_mask.py`, `_upstream.py`. Use as the PCC and
  decision oracle; replay hooks live here.
- `weight_mapping.py` (`remap_state_dict` — backbone + self-cond split), `config.py`,
  `checkpoint.py`, `memory_budget.py`.
- `demo/text_demo.py` (emits `DG_TEXT_DEMO_SUCCESS`/`DG_TEXT_DEMO_FAILURE` markers — grep these,
  the RUN-first denoise path emits expected `TT_THROW` fallback noise even on success),
  `demo/replay_hf_tt.py` (HF-vs-TT committed replay harness).
- `tests/` — device-gated tests (`DG_RUN_DEVICE=1`, checkpoint via `DG_CKPT`). RUN regression:
  `tests/test_device_text_demo_run.py`. See `tests/` for KV-phase, bidirectional SDPA, canvas
  sampling, entropy, trajectory PCC, self-conditioning, memory-budget coverage.

## Issue map / stage ownership (parent #47452, label `DiffusionGemma`)

- Foundation: #47468 torch ref + PCC harness ✅ · #47461 causal backbone (QB2) + self-cond loader ✅
- HW: #47487 QB2 256K budget + Galaxy 4×8 TP (budget must include canvas scratch + non-causal mask)
- Core: #47474 KV phase machine ✅ · #47462 bidirectional forward ✅ · #47463 decode loop (built;
  fidelity gated by #48291) · #47472 on-device sampling ✅ · #47557 batched decode · #47464
  functional e2e (**RUN done**) · #47465 perf · #47466 vLLM · #47488 vLLM block-granular
  runner/scheduler (upstream tenstorrent/vllm `dev`)
- Correctness: **#48291** decision-fidelity bar (bf16/MoE/TP=4 argmax ≈50% vs HF; diffusion commits
  the clean argmax so there is no temperature cushion). RUN-first: defer #48291, but decide it
  **before shipping** usable output.
- Later: #47467 multimodal · #47475 quant · #47489 CI

## Conventions (hard requirements)

- **Commit messages must NOT include a `Co-Authored-By` trailer.**
- **Commit AND push after each meaningful, verified batch of changes** — don't accumulate a large
  uncommitted pile. Land increments on the working branch (`diffusion-gemma-function`); log SHAs.
- **Do NOT edit `models/demos/gemma4/` or any shared directory.** Keep every fix inside
  `models/experimental/diffusion_gemma/`. `git diff main -- models/demos/gemma4/` empty is a gate.
- **Do not skip device tests by default.** Run the relevant QB2 device test whenever hardware/env
  is available; only skip when genuinely inapplicable or blocked, and record the reason.
- **RUN-first vs correctness:** the current priority is a reproducible prompt→text device run at
  small scale; degenerate/EOS-heavy output is acceptable for the RUN milestone. Fidelity (#48291,
  R0.5/R0.6 replay) is a separate, deferred track — do not let it block a run.
- Stage evidence should go under `models/experimental/diffusion_gemma/doc/<stage>/` (README.md +
  work_log.md + artifacts); the context contract, once created, is
  `models/experimental/diffusion_gemma/doc/context_contract.json`. This `doc/` tree does not exist
  yet — create it on first stage work (the layout mirrors `models/autoports/*/doc/`).
- Grep `DG_TEXT_DEMO_SUCCESS` / `DG_TEXT_DEMO_FAILURE` for run outcome — not the fallback log noise.
