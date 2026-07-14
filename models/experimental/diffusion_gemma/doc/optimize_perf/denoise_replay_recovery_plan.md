# Plan: recover traced-denoise replay via a paged prefix (high-level)

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
