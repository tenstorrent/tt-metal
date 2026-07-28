# Official DiffusionGemma sampler with traced early halt (2026-07-22)

> **2026-07-28 — the `host` Gumbel mode used/recommended below was DELETED.** It was measured NOT
> to be the TT language-drift cause: it drifts on exactly the same prompts as `device`, repairs 0,
> and costs 1.40x per request. The real cause was the canvas attending prefill pad keys, fixed in
> `d0936d4da4f`. **Every measurement below stands exactly as recorded**; only the recommendation to
> use, keep, or fall back to `host` is void. `device` is the only materialized Gumbel source and
> therefore the only mode valid under up-front capture (`argmax`/`chunked` are not materialized).

## Configuration

The official-semantics serving configuration now combines:

- IID full-vocabulary Gumbel-max through `DG_VLLM_GUMBEL_MODE=host`;
- the released `t_max=0.8` to `t_min=0.4` denoise temperature schedule;
- released entropy budget `0.1`;
- released stable-and-confident halt thresholds (`stability_threshold=1`,
  `confidence_threshold=0.005`);
- one-step traced early-halt windows through
  `DG_DENOISE_EARLY_HALT=1 DG_DENOISE_EARLY_HALT_WINDOW=1`;
- reusable up-front capture.

HTTP `temperature`, `top_k`, `top_p`, and `seed` are not checkpoint sampler
parameters and are not wired into the model-owned denoise loop. The GPQA launcher
therefore no longer presents those arguments as effective settings.

Dynamic Gumbel previously selected the fixed-budget single-step trace in the
vLLM wrapper even when early halt was enabled. It now selects
`traced_early_halt_block` when the halt window is one. A larger window fails
loudly because the Gumbel seed/noise must be refreshed between denoise steps.

`chunked` is not used for the official-quality launcher: its current
1024-wide TT RNG has a known innermost-axis distribution bias. `device` uses
the distribution-tested permuted-vocabulary workaround, but the full-vocabulary
temporary pads to an 8 GiB allocation and OOMs on this full-depth configuration.
`host` generates the same IID Gumbel distribution with torch and copies one
bounded full-vocabulary input into the persistent traced buffer. It is slower
than chunked sampling but preserves the released sampling semantics.

## Device validation

Full 30-layer DiffusionGemma on 4× Blackhole p300c, TP=4:

- `DG_UPFRONT_CAPTURE=1`
- reveal-mask `p_max=4096` (the full GPQA launcher size)
- 48-step cap
- host-generated IID Gumbel
- early-halt window 1
- two sequential prompts through one persistent capture

Results:

- request 0 halted at K=17;
- request 1 halted at K=19;
- both committed blocks, halt decisions, and realized K values exactly matched
  eager early halt under the same canvas/Gumbel/renoise seeds;
- `capture_events` stayed 48 across both requests;
- request 1 reported `gumbel_mode=materialized` and `reveal_mask_reuse`;
- both requests released normally;
- pytest result: `1 passed` in 151.25 s.

CPU coverage:

- 12 focused vLLM sampler/trace-selection tests passed;
- 36 sampling/up-front/real-Transformers parity tests passed;
- final combined up-front/vLLM regression: 39 passed, 5 device-gated skips,
  with the unrelated stale launch fixture deselected;
- formatting, shell syntax, lints, and diff whitespace checks passed.

The unrelated pre-existing
`test_server_launch_forces_optimized_trace_stack_and_full_prefill_budget`
fixture still lacks its `fixed_budget` field; it fails before reaching this
sampler selection code.

A fresh independent stage review returned `clean-pass` after the launcher moved
off biased chunked RNG and the device gate added exact eager-vs-traced
token/K/halt comparisons.

## Thinking-template contract — #48291 doc-0 garbage root cause + fix (2026-07-24)

The #48291 comment-5062743522 doc-0 GPQA garbage (`níní…1111… the the…`) was traced to a
**malformed thinking-template contract in the eval invocation**, NOT the host-Gumbel sampler,
bf16 precision, the reveal mask, or the trace.

Mechanism: an eval invocation injected a literal `<|think|>` *system* message while the server's
`enable_thinking=true` was not applied. The checkpoint `chat_template.jinja` then (under
`add_generation_prompt` + `not enable_thinking`) appends an empty, already-closed thought channel
`<|channel>thought\n<channel|>`. Asked to answer after a finished-empty thought, block 0 never
converges (48 steps, `halted=false`), commits the clean argmax of a near-flat/noise canvas into
the frozen KV, and later blocks condition on that noise and collapse to the trivial `1`/`\`
repetition attractor (which halts early because a constant canvas is stable + near-zero entropy).
Verify device-free with the checkpoint tokenizer `apply_chat_template`: `enable_thinking != true`
⇒ the empty-closed-thought suffix; `enable_thinking=true` ⇒ a clean `<|turn>model\n` prompt.

Fix — an eval-invocation contract, NOT an inference-server code change: rely on server-side
`--default-chat-template-kwargs '{"enable_thinking":true}'` and do NOT inject a manual `<|think|>`
token. The committed `run_upfront_gpqa.sh THINKING_MODE=1`, the tt-inference-server DG model spec,
and its GPQA eval config (`use_chat_api=True`) already use this correct contract; the garbage
artifacts came from an older/ad-hoc invocation that bypassed it.

Device confirmation (P150x4, 2026-07-24, msl=4096 gen=3072, host-Gumbel seed 0, thinking): doc-0
`exact_match=1`, ends `\boxed{C}`, coherent Heisenberg-uncertainty reasoning, and EVERY block
halts early (block 0 `halted=true` + K=16/9/8/5) vs the failing run's blocks 0–8 all 48-step
`halted=false`. `prompt_len` 157 (correct) vs 161 (malformed).

Secondary robustness gap (amplifier, not the origin): the K-step cap commits a non-halted block's
argmax anyway (matches upstream Transformers `generation_diffusion_gemma.py`), so one bad block
poisons every later block via KV. A flag/abort-on-non-halt guard is a serving-robustness follow-up
in `tt/generate.py` / `tt/traced_denoise.py`.

## TTFT / block throughput — dense vs tuned sparse MoE (2026-07-24, msl=4096, gen=3072, thinking)

DiffusionGemma emits a whole 256-token block per step-loop, so "TTFT" here is the time to the
first 256-token block (prefill + block-0 denoise) — structurally unlike autoregressive TTFT.

| metric | dense-128 (now fail-loud) | tuned sparse MoE (default) |
| --- | --- | --- |
| per denoise step | ~4.95 s | ~0.9 s |
| doc-0 TTFT (first block) | 79.5 s | 14.5 s |
| doc-0 steady t/s (decode-only) | 5.2 | 33.1 |
| doc-1 TTFT | 145.4 s | 18.7 s |
| doc-1 steady t/s | 2.6 | 20.4 |

The ~5× factor is the true-sparse token-gather MoE (MoE dominates ~89% of the denoise step). As of
2026-07-24 the sparse path is the DEFAULT (`DG_SPARSE_MOE` default on) and the dense-128 path is
fail-loud unless `DG_ALLOW_DENSE_MOE=1` — see the README guardrails. Cross-run block counts differ
under `DG_SPARSE_MOE_TUNED` (bf16 chaos): doc-0 ran 5 vs 7 blocks but stays correct (`\boxed{C}`);
doc-0 remained coherent, while doc-1 flipped answer across bf16 configs (a #48291
decision-fidelity sensitivity, not this bug).
