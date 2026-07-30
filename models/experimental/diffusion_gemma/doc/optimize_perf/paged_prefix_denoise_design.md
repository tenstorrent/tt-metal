# Capture-once denoise attention — reveal-mask (shipped) and paged prefix (open)

Status: open — Phase 1 (reveal mask) shipped and is intrinsic to the default up-front path; Phase 2
(T8 paged full-attention read + LSE merge) and T11 (128K/256K perf+memory) remain. Every flag this
design named is deleted or default-ON ([flag triage](flag_triage_20260728.md)), and the "golden =
per-block-recapture path" it compares against is no longer executable.
Owns: the Phase-2 design and its rejected alternatives, the kernel/merge API contract, the
denoise-attention gotchas, and the adapter-lifetime mechanism absorbed from the deleted
`upfront_warmup_plan.md`. Over cap: two open work items plus a lifetime trap.
See also: [refuted list](../REFUTED.md), [optimize-perf hub](README.md), [plan](../../plan.md).

## Problem and contract

The traced denoise loop recaptured every block because the denoise attention's prefix K/V and mask
grow by `canvas_len` (256) per block while a Metal trace bakes shapes at capture — the growing
`read_prompt_kv_cache_slice` plus the `prompt_len`-keyed `invalidate_prefix_growth` guard were the
root cause. RoPE was already trace-fixed via the constant-shape `canvas_rope_provider` written input.

- **C1 — no recapture.** Every per-block-varying shape becomes a constant-shape **written input**
  refreshed outside capture.
- **C2 — early halt untouched.** All read spans, page tables, tails and masks are constant *within*
  a block and change only between blocks.
- **C3 — no leak.** Later blocks attend earlier blocks' committed KV; uncommitted or future
  positions are never read. `frozen_prefix` violates this; this design does not.

Topology: **30 layers = 5 full-attention** (K=V-tied, head_dim 512, every 6th) **+ 25
sliding-window** (head_dim 256, W=1024). Mask geometry is owned by [plan](../../plan.md).

## Full-attention 5 layers — paged chunked SDPA + LSE merge (Phase 2, OPEN)

Canvas Q (C=256 rows at abs pos `prompt_len … prompt_len+C-1`) attends `[prefix(P committed) ++
canvas(C)]`, bidirectional on canvas. Decompose into two partials plus a merge:

- **Prefix partial** is causal-equivalent (the canvas is entirely after the prefix): with
  `chunk_start_idx = prompt_len`, paged `ttnn.transformer.chunked_scaled_dot_product_attention` over
  the prefix pages gives each query all P committed keys. No leak — the page table exposes only
  committed pages *and* `chunk_start_idx` causal-bounds the read. `chunk_start_idx_tensor` as a `[1]`
  int32 **device** tensor is trace-safe (runtime offset, no recompile), proven in `llama3_70b_galaxy`.
- **Canvas partial** is the existing non-causal C×C local SDPA (`_sdpa_q_chunked`,
  `is_causal=False`). Fixed shape, no kernel.
- **Merge** = flash online-softmax combine, needing each partial's LSE.

**REJECTED merge alternatives:** (a) two-pass recompute re-materializes the C×P prefix scores,
defeating the long-context reason paged exists; (b) mask-into-the-chunked-op adds a general additive
mask plus a non-causal mode to a causal-only paged kernel — a strictly larger LLK surface than
emitting a statistic the kernel already computes. Chosen instead: `return_lse`, exposing
`lse = m + log(l)` from the existing `compute_streaming.hpp` reduction (already emitted by the
ring-joint path's `cb_lse_out`), combined by a pure-ttnn `merge_attention_partials`.

```
scaled_dot_product_attention(..., return_lse: bool = False)          # default → identical to today
chunked_scaled_dot_product_attention(..., return_lse: bool = False)
#   return_lse=True → (output[1,H,C,vhd], lse[1,H,C,1] fp32 = m + log(l))
merge_attention_partials(out_a, lse_a, out_b, lse_b) -> out
#   m = max(lse_a, lse_b); wa = exp(lse_a - m); wb = exp(lse_b - m)
#   out = (out_a*wa + out_b*wb) / (wa + wb)      # algebraically exact
```

Because the merge is algebraically exact, bf16 rescale drift gates on **decision agreement, not
bitwise** equality. The 0.95 argmax-agreement gate is mis-specified and unreachable by any bf16
implementation; the usable floor is **≥0.992 at production-48 steps** — do not chase precision
([refuted list](../REFUTED.md)).

**Primitive status — one half survives, the other was reverted.** `tt/attention_merge.py ::
merge_attention_partials` is DONE and still in the tree (device 3/3, `tests/test_attention_merge.py`),
but never wired into `tt/diffusion_attention.py`. Its producer, the `return_lse` SDPA kernel extension,
**was reverted on 2026-07-30**: it lived in `ttnn/cpp/` with no live consumer, so the no-shared-edits
rule took it out. It did pass 6/6 on QB2 first (`return_lse=False` byte-identical, LSE ≈
`torch.logsumexp`) and is recoverable from `2e18c599bd3` — see
[the T6 plan](return_lse_kernel_plan.md). **Wiring this section therefore now needs the ttnn extension
re-landed as its own upstream PR first**, not just a call-site change.

## Sliding 25 layers — no paging needed

Since C=256 < W=1024, the union over all canvas queries is exactly the **last W committed prefix
rows**, so a persistent constant-shape `[1, kv, W, hd]` tail buffer refreshed by `ttnn.copy` outside
capture suffices: O(W), context-independent, no leak (W=1024 and commit=256 are both 32-aligned, so
the copy is exact).

This design called the traced path "maskless ALL-ATTEND" and filed true sliding as a Phase-2
decision change needing its own #48291 re-validation (T10). **No longer true:** a bounded
sliding-window prefix read shipped 2026-07-29 (`a25adba5260`), is unconditional on the traced path
whenever retention is enforced, and is bit-identical (sha256 equal 10/10) — so T10's re-gate is moot
as stated.

## Device results (Phase 1)

- Full 30-layer 26B, 3 blocks: `committed_sha256` **identical** across eager / recapture-golden /
  reveal-mask (`31a59b3e…`); capture-once with `reveal_mask_reuse` every block, 0 recapture;
  **1.68× faster** (13.0 vs 21.9 s per block; 19.6 vs 11.7 tok/blk/s).
- Lazy capture: coherent prompt halting [8,2], `committed_sha256 == golden` (`19ada7e9…`) bit-exact,
  `capture_events` 48 → 8, TTFT 108.8 → 54.0 s (~2×), steady state unchanged.

**MEASUREMENT TRAP:** early halt does **not** fire on the `live_context_sweep` repetitive FILLER
prompt (it degenerates and runs the full 48 steps); it fires on coherent prompts, so a filler-prompt
benchmark structurally cannot measure early-halt speed.

The design's motivating arithmetic — "71–84 s/block ≈ 3.6 tok/s vs eager ~7" and "3.6 → ~18 tok/s
(+early-halt ~47)" — is pre-MoE-work and is not a throughput claim; current throughput is owned by
the [optimize-perf hub](README.md).

## Adapter lifetime — why recapture happened at all (from the deleted warmup plan)

The trace was **already shape-invariant and reused across blocks**. The only thing forcing
per-request recapture was **lifetime**: `DenoiseLogitsAdapter` owns the trace-baked persistent
buffers (`_canvas_rope_bufs`, `_reveal_mask_buf`, `signal_buf`, `_vocab_offsets` /
`_embedding_weight_sharded`) and was rebuilt on every prefill, reallocating them at new addresses
and invalidating the block-0 trace.

**Correctness argument for reusing one capture:** controller buffers, adapter buffers and the
model-owned `tt_kv_cache` read at a fixed `p_max` keep stable addresses, while per-request content
flows in *outside* the trace — prefill overwrites cache `[0:cache_len]`, `rebind_prompt` resets
`prompt_len` / `q_rope_offset` / reader-floor and re-hides the stale `[cache_len:p_max]` tail, and
the replay re-refreshes canvas RoPE. `reset_prompt_len` must exist separately from the grow-only
`set_prompt_len` because a request boundary has to allow ANY tile-aligned `prompt_len <= read_span`,
including a shrink. Mock-capture cache pollution is benign: the mock's committed KV lands in the
`[mock_len:p_max]` tail, which every real request masks out or overwrites.

**THE ONE BEHAVIOR ACCEPTED:** up-front capture fixes the prefix read span `p_max` at startup and
the denoise reads that FULL span every step, O(`p_max`) independent of the true prompt length, so
servable prompt + generated is capped at `p_max`. The full contract is owned by the
[optimize-perf hub](README.md).

**SHARPEST EDGE, named in advance:** the session reset guard is the single point keeping the
persistent adapter alive — miss it and released traces are used after free (garbage or crash);
over-retain on the non-up-front path and traces leak. It must be strictly flag-scoped. DG serving is
single-sequence (`prefill_forward` rejects `num_reqs > 1`), so one persistent adapter is compatible;
concurrent batched serving is #47488 / #47557, out of scope.

**VERIFICATION DESIGN that shipped as `tests/test_upfront_capture.py`:** (1) reuse across prompts,
asserting `capture_events == 1` while A vs B committed outputs DIFFER — proving the new prompt's
KV/mask flow in rather than a stale replay; (2) bit-exact committed sha256 across up-front,
per-request and eager; (3) a multi-request A→B→A coherence smoke guarding the "garbled while serving
but clean standalone" stale-cross-request-state failure mode. Do **not** collect Tracy or profiler
data from a live server (tt-enable-tracing rule) — use direct-session or block-harness evidence.

## Gotchas

- Do NOT reuse `build_canvas_denoise_mask(P_max, C)` for reveal, and do NOT let the sliding mask be
  sometimes-`None` — both in [refuted list](../REFUTED.md).
- The masked SDPA has **no** `_manual_gqa_attention` L1-CB fallback (that fires only when
  `attn_mask is None`), which caps reveal-mask `P_max` on the head_dim-512 layers — a reason paged
  is the real long-context answer.
- Refresh via persistent buffer + `ttnn.copy`, **never** `ttnn.from_torch` in-trace. Chunked-SDPA
  Q-chunk length must be a multiple of 128. Uncommitted rows must be **zero-initialized**:
  `NaN + -inf = NaN` poisons softmax.
- The full-span read `[0:cache_len]` **clones** the cache via `read_prompt_kv_cache_slice` to avoid
  aliasing or freeing it; the buffer-aliasing trap behind that is owned by
  [per-layer prefix spans](per_layer_prefix_spans.md), the tile-aligned `ttnn.fill_cache` write by
  [commit batching](commit_batching.md).
- Trace-lifetime rule — allocate EVERY persistent cross-replay buffer in `prepare_*` BEFORE
  `begin_trace_capture`, or trace scratch clobbers it on every replay; a capture overflow poisons the
  device (`tt-smi -r`), so guards fail loud at startup, never mid-serving. Owned by the
  [optimize-perf hub](README.md).
- `prompt_len` must be **decoupled** — it drives read-span, `q_rope_offset` and mask anchor at once.
  Read-span moves to the page table or a fixed `P_max`, reveal to `chunk_start_idx`; RoPE offset and
  mask anchor stay on `prompt_len`, asserting `revealed_len == chunk_start_idx == prompt_len`.

## Still open

- **T8** — swap the 5 full-attention layers to paged read + merge. Needs paged-cache infrastructure
  over DG's contiguous KV cache: the hard part.
- **T11** — Phase-2 128K/256K perf and memory.

Verification environment: 4× Blackhole `p300c`, venv `/home/zni/venvs/tt-diffusion-gemma`,
checkpoint `~/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it`.
