# Plan: recover traced-denoise replay via a paged prefix (high-level)

> **SUPERSEDED 2026-07-19 — see `paged_prefix_denoise_design.md`.** The §7 "fixed-max prefix +
> reveal-mask" recovery is now IMPLEMENTED and **bit-exact on the full 30-layer 26B model** on QB2
> (`DG_DENOISE_REVEAL_MASK`, capture-once, 0 recapture, 1.68× faster; reveal==recapture==eager
> committed_sha256). The §1–6 true-paged read (this doc) is now **Phase 2** (T8, 128K/256K endgame)
> whose primitives (`return_lse` SDPA kernel + `merge_attention_partials`) are also device-verified.
> This doc is retained as the recon record.

Status: PLAN ONLY (no implementation). Approach chosen: **paged prefix read**.

## 0. Goal
Eliminate the per-block Metal-trace **recapture** that regressed steady serving from ~18 to ~3.6
tok/s, i.e. restore **capture-once / replay-many**, while keeping the **growing-prefix correctness**
committed by `ec5b64b4891` (tokens committed in earlier blocks must be visible to later blocks'
cross-attention). Use a paged prefix so the prefix becomes a **fixed-shape, replayable input**, and
so the denoise path aligns with the future 256K / vLLM-paged direction.

## 1. Core idea
- Route the denoise cross-attention over the frozen prefix through a **paged read**: the SDPA reads
  prefix KV via a **fixed-shape `page_table`**.
- The `page_table` (plus any "revealed length / last-page" bookkeeping) is a **persistent written
  input**, refreshed **outside** the trace each block to expose newly committed pages — without
  changing any tensor shape inside the trace.
- Trace shape is therefore invariant → **one capture replays every block**. Reuse the proven
  "canvas-RoPE written input" mechanism (buffers allocated before capture, refreshed per block
  outside capture); the `page_table` refresh lives at the same point.
- Prefix growth changes from "widen the read span" to "update the `page_table`"; the
  `invalidate_prefix_growth` recapture guard is then removed / demoted to a fallback.

## 2. Phases
- **A — Recon & feasibility.** Confirm the available paged prefix-read primitive (does the
  denoise / backbone attention already have a paged SDPA; `page_table` semantics; partial-last-page
  handling); the full-attention (K=V) layers vs the 25 sliding layers; how commit writes a new
  block's KV into the page pool; and the relationship to the current contiguous model-owned cache and
  the just-landed 32K KV-spec fix.
- **B — Design freeze.** Fixed `page_table` shape + reveal semantics; boundary handling (256-token
  commits aligned to page size is the simplest case); paged read for both attention-layer kinds; the
  per-block refresh point; coexistence with the self-cond / per-step-noise persistent buffers.
- **C — Implementation (gated, DG-local).** Add the paged prefix-read path + fixed-shape `page_table`
  written input; change post-commit prefix growth to a `page_table` update; controller prepares /
  refreshes the `page_table` and drops the recapture guard. Behind a default-OFF flag until device-
  verified.
- **D — Correctness verification.** Golden = the current recapture path (already growing-prefix
  correct). Require multi-block **bit-exactness** (committed argmax / KV / logits) + replay
  determinism + an "updated-input → output changes" replay test + boundary cases.
- **E — Performance verification (QB2).** Steady block ~71–84 s → ~14 s (3.6 → ~18 tok/s); confirm
  `recapture_after_block0 = false` and a single capture; page-pool + trace-region memory; multiple
  block counts and context lengths.
- **F — Wrap-up.** Evidence under `doc/vllm_integration/`; keep default-OFF until verified, then
  decide default-on; commit + push; update the issue comment if warranted.

## 3. Correctness strategy
- Golden is the current recapture path; the new paged replay must be multi-block bit-exact.
- Watch: paged-read vs the original contiguous maskless-read numerics; the reveal/last-page boundary
  must **never expose uncommitted tokens** to the cross-attention; must not disturb the #48291
  decision-fidelity premise (small-probability drift must not flip accept/renoise).

## 4. Risks / unknowns
- Whether a paged SDPA usable by the denoise path already exists — if not, effort rises materially.
- Full-attention (K=V, head_dim 512) layers vs the 25 sliding layers reading differently.
- Reveal-length / partial-last-page boundary.
- Page-pool + trace-region memory; the denoise may need to read the page pool rather than a
  contiguous slice.
- Independent of, but must stay self-consistent with, the 32K admission fix (a vLLM-accounting-layer
  change; this is a device-side denoise read-path change).

## 5. Evidence to leave
Multi-block bit-exact + replay determinism + updated-input replay test; before/after steady tok/s
(3.6 → ~18) with recapture count = 0; page-pool + trace-region memory; no host fallback inside
capture; gated default-OFF.

## 6. Effort
~2–4 days including device verification; toward the upper end (or more) if the denoise has no
ready-made paged SDPA.

## 7. Feasibility recon (2026-07-15) — upper-end/kernel-level, and early-halt is the cheaper recovery

Source recon (read-only) verdict: **there is no ready-made paged SDPA that satisfies the denoise
cross-attention's semantics, so the paged-prefix path is upper-end-of-2–4-days-or-beyond
(kernel-level), not a drop-in.**

- **What's reusable (favors the estimate):** `ttnn.transformer.chunked_scaled_dot_product_attention`
  is a real paged prefill SDPA with a fixed-shape `page_table_tensor` and a trace-safe device-tensor
  offset (`chunk_start_idx_tensor`, "update on device, no recompile") — exactly the persistent
  written-input the plan wants. gemma4 already runs paged decode/prefill + hybrid page tables
  (`decode.py`, `prefill.py`, `kv_cache_hybrid.py`); DiffusionGemma's own commit path already
  writes/reads paged (`commit_decode.py:216-289`); the committed prefix is already in the cache.
- **Why it is NOT a drop-in (the crux):**
  1. The paged prefill op is **causal-only, no `attn_mask`/`is_causal`** (`sdpa_nanobind.cpp:460-471`),
     but the denoise op runs `is_causal=False` (`diffusion_attention.py:200`). canvas→prefix is
     causal-equivalent (`chunk_start_idx=prompt_len`), but **canvas→canvas is bidirectional** and
     cannot be expressed causally in one paged SDPA.
  2. Merging a paged-causal prefix read with a local non-causal canvas SDPA needs flash-style
     online-softmax (LSE) accumulation; the base SDPA returns only the output (no LSE), so there is
     **no ready-made merge op** — requires a two-pass custom merge OR a C++/LLK extension of the
     paged/chunked SDPA to accept a mask / non-causal mode.
  3. **Sliding layers (25 of 30) have no paged prefill primitive** — gemma4 slides with a *non-paged*
     windowed slicer (`operations.py:320-376`) over bounded, modulo-mapped caches; they need a
     different paged-read design than the 5 full-attention layers.
- **The sole remaining recapture cause** is the growing concatenated prefix K/V (dim-2 grows by
  `canvas_len`/block via `read_prompt_kv_cache_slice` + the `invalidate_prefix_growth` guard at
  `traced_denoise.py:434-442`); RoPE is already trace-fixed via the constant-shape
  `canvas_rope_provider`.
- **Cheaper alternative to evaluate first — fixed-max prefix + reveal-mask:** slice a *fixed-max*
  prefix span and reveal committed positions with a persistent `attn_mask` written-input (the masked
  denoise SDPA path already exists — `diffusion_attention.py:199-208`, mask builder
  `denoise_forward.py:118-146`), so trace shape stays fixed → capture-once/replay-many with NO new
  kernel. Costs extra masked compute over the padded span (fine for the current bounded
  `max_num_seqs=1` contexts; wasteful toward 256K) and must be bit-exactness-checked vs the
  contiguous-concat golden and must never expose uncommitted tokens.
- **Highest-leverage, already-tractable recovery: early-halt.** Post the tanh-GELU fix the trajectory
  converges (halts ~7–15 of 48 steps — device-observed on the canonical prompt). Recapture cost is
  per-trace-per-block, so halting at ~K steps cuts BOTH replays and captured traces by ~48/K,
  recovering most of the 3.6-tok/s regression with zero attention-path change. Land/enable this
  first; scope the paged-prefix (or reveal-mask) work against what remains.

## 8. Implemented: `DG_DENOISE_FROZEN_PREFIX` capture-once (device-verified 2026-07-15)

Restored the pre-`ec5b64b4891` capture-once/replay-many mechanism as an **opt-in flag** (default
OFF): the recapture-on-growth guard in `traced_denoise.py` is gated so the block-0 trace is reused
across blocks (`frozen_prefix_reuse` metric instead of `invalidate_prefix_growth`). Unit-tested
(`test_frozen_prefix_reuses_block0_trace_without_recapture`) + full-30L device measurement
(`probe_early_halt --mode perf`, canonical prompt, 3 blocks):

| config (30L, frozen prefix) | steady t/s | block | steps/blk | vs regressed 3.03 |
| --- | --- | --- | --- | --- |
| fixed-48 | **11.92** | 21.5 s | [48,48,48] | ~3.9× |
| **frozen + early-halt** | **47.84** | 5.35 s | [9,17,2] halted | ~16× |

- Capture-once confirmed: 2 capture events, 8 `frozen_prefix_reuse`, **0 recapture**; per-step cost
  1.746 s (capture+replay) → **0.42 s** (replay-only).
- **Early-halt now FIRES** under the real 0.005 threshold on the coherent model (halts [9,17,2] of 48,
  bit-exact vs the full path) — the tanh-GELU fix unlocked it; the old "never clears 0.005" note is
  withdrawn. Frozen **+** early-halt is the fastest config (47.84 t/s), far exceeding the historical
  ~17.9 t/s fixed-48 baseline.
- fixed-48 frozen (11.92) is below the historical 17.9 because the model now runs the *correct*
  tanh-GELU (heavier than the old fast-approx GELU the baseline was measured under) and this config
  omits the fused MoE-dispatch kernel (`DG_MOE_DISPATCH_FUSED2`, default off) — headroom, not a
  regression.
- **TRADE-OFF (why default-OFF):** frozen prefix means later blocks read the block-0 prefix slice —
  they do NOT attend to earlier blocks' committed KV. Bit-correct for single-block generation; a
  speed-over-multiblock-fidelity trade for >1 block. The correct-AND-fast path still needs the
  fixed-shape reveal-mask (§7) or paged read (§1–6).
