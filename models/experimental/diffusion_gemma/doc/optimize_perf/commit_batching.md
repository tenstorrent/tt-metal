# Commit batching — one causal prefill-append instead of 256 decode-appends (#47557)

## What / why

After each denoise block the generation loop **commits** the block's 256 clean-argmax canvas
tokens into the frozen Gemma4 KV cache (three-phase KV: prefill → denoise → **commit**). The
baseline commit (`tt/generate.py::commit_canvas_tokens`) does this with **256 sequential
single-token decode-appends** — one full 30-layer `commit_decode_forward` per committed token
(**~31.5 s / 256-token block** on QB2; see `README.md` headline table). The 256 forwards, not the
KV writes, are the cost.

The batched commit (`tt/commit_batched.py::commit_canvas_tokens_batched`) collapses those 256
forwards into **one causal masked prefill** over the whole 256-token canvas. Expected ~7× on the
commit step and ~1.25× on end-to-end block throughput (the ~48-step denoise loop is the other
large per-block term: `README.md` shows commit = 31.5 s of a ~232 s block).

It is **opt-in and guarded** (`DG_COMMIT_BATCHED=1`, or pass `commit_fn` to
`denoise_and_commit_block`). The sequential path stays the default until this is validated on
device (`verify_commit_batching.py`). **No `models/demos/gemma4/` edits** — the batched commit
composes over the importable Gemma4 ops exactly like `tt/commit_decode.py` and
`tt/diffusion_attention.py`.

## The equivalence argument (code inspection)

**Claim.** For a fixed committed token block, the batched commit writes, per layer and per
committed position, K/V that are the *same function* of the same inputs as the 256 sequential
decode-appends — same absolute positions, same per-head norm, same RoPE, same K/V projections,
same causal visibility, same cache layout. The two paths write **algebraically identical** K/V;
they differ only in **op implementation** (prefill vs decode kernels for the same math), a small
numerical drift (bfp8 / tile-reduction order), not an algebraic difference. So the correct
assertion is **high PCC / small max-abs-diff**, not literal bit-identity — the same relationship
prefill and decode already have in the backbone. Each component is pinned to code below.

### 1. The sequential commit *is* a causal prefill

In `commit_decode_forward` → `_commit_attention_decode_forward`, for committed token `i` at
absolute position `start_pos + i`:

- its K/V are computed from **that token's** per-layer input hidden state and written at
  `start_pos + i` (`paged_update_cache(update_idxs_tensor=cache_pos)`, `commit_decode.py:236-257`);
- its decode SDPA (`scaled_dot_product_attention_decode`, `cur_pos = start_pos+i`) attends to
  **all cache positions `≤ start_pos + i`** — the frozen prefix (prompt + prior blocks,
  `0..start_pos-1`) plus already-committed canvas tokens `0..i-1` plus token `i` itself. Token
  `j < i` was written when *it* was the current decode token, so by the time token `i` runs the
  cache holds `prefix ++ canvas[0..i]` at every layer.

That is exactly the causal-prefill visibility of a chunk appended at the end of the cache. In a
decoder transformer each token's per-layer representation depends only on itself and earlier
tokens, computed identically token-by-token or all-at-once — the standard
autoregressive-equals-prefill identity, of which the commit is a concrete instance.

### 2. The batched commit reproduces that visibility with an explicit causal mask

`commit_batched.py::_sdpa_causal_masked` runs `scaled_dot_product_attention(is_causal=False,
attn_mask=<additive [1,1,C,start_pos+C] causal mask>)`. The mask
(`build_canvas_denoise_mask(prefix_len=start_pos, canvas_len=C, causal=True, ...)`) sets, for
canvas query `i` (absolute `start_pos+i`) and key `p`:

- **full-attention layer:** attend iff `p ≤ start_pos + i` → whole prefix visible, canvas causal
  (`p = start_pos+j` visible iff `j ≤ i`);
- **sliding layer:** additionally `start_pos + i - p < sliding_window` (last `sliding_window`
  positions), matching the decode SDPA's `sliding_window_size`.

Verified on host against a brute-force enumeration of the sequential decode-append visibility
(`reference/attention_mask.py::build_canvas_denoise_mask(causal=True)`; the landing commit runs
that check across `P ∈ {32,64,1024,1280,0,96}`). The batched SDPA's `attn_mask` handling and the
L1-clash fallback (`_manual_gqa_attention_masked`, which *keeps* the mask) reuse the validated
denoise SDPA path — only the mask content changes.

> **Sliding-window edge caveat.** The mask uses the HF causal-sliding convention
> (`0 ≤ q−k < window`); the device reference is the decode SDPA's `sliding_window_size`. They
> agree where the window does **not** bite (`start_pos + C ≤ sliding_window`, i.e. committed
> context ≤ 768 with `window = 1024`), covering RUN-first. Where it bites, a one-position edge
> mismatch is possible and must be confirmed on device — the verify harness compares sliding
> layers' K/V directly at a long-enough context.

### 3. K/V values: same projection + per-head norm + RoPE, same positions

| step | sequential (`commit_decode.py`) | batched (`commit_batched.py`) |
|------|----------------------------------|-------------------------------|
| QKV projection | `apply_qkv_projection` | `apply_qkv_projection` (identical weights) |
| head split | `split_qkv_heads_decode` | `split_qkv_heads_prefill` (decode vs prefill layout of the same split) |
| Q norm | `_apply_per_head_norm(q_norm_weight, with_scale=True)` | `apply_per_head_norm(q_norm_weight, with_scale=True)` |
| K norm | `_apply_per_head_norm(k_norm_weight, with_scale=True)` | `apply_per_head_norm(k_norm_weight, with_scale=True)` |
| V norm | `_apply_per_head_norm(None, with_scale=False)` | `apply_per_head_norm(None, with_scale=False)` |
| RoPE | `_apply_rope_decode_peruser` / `apply_rope` at `start_pos+i` | `_apply_rope_chunked(start_offset=start_pos)` → `start_pos+i` |

Both use the **same RMS eps** (`config.rms_norm_eps`) and the **same absolute RoPE position**
`start_pos + i` for canvas token `i`. The frozen prefix K/V are read back already-RoPE'd/normed
(written by prompt prefill / prior commits) and are **not** re-RoPE'd in either path — matching
the denoise prefix contract (`diffusion_attention.denoise_attention` `k_rope_offset` logic). The
decode vs prefill head-split/RoPE ops compute the same values in different tile layouts — the
numerical-drift term, not an algebraic one.

### 4. Same cache positions and layout

- **Positions.** Batched writes canvas token `i` at absolute `start_pos + i`
  (`_write_canvas_kv_contiguous`, `update_idxs=[start_pos+t]`) — the *same* indices the sequential
  path uses. `start_pos = cache_len + N·256` is a multiple of 32 (prompt padded to 32, canvas
  256), so all seq bounds are tile-aligned.
- **Layout.** Both write the same contiguous per-layer cache tensor
  `[1, num_local_kv_heads, max_seq, head_dim]` (`tt_kv_cache[i]`) via the same non-paged
  `paged_update_cache`. A single-sequence (batch-1) contiguous cache addresses one seq position
  per non-paged update, so the default `write_batch=1` = one op per position = provably the same
  write as sequential. `write_batch>1` is an opt-in fast write (1-block-paged trick) to be
  device-validated before default.
- **KV-sharing.** `write-then-read-from-cache`: a non-shared layer writes its canvas K/V, then the
  SDPA reads `cache[0 : start_pos+C]`. A KV-shared layer (`kv_shared_layer_map[i]`, E2B/E4B) skips
  its write; its earlier **source** layer already wrote the shared cache tensor, so the shared
  layer's read sees `prefix ++ canvas`. Mirrors the sequential `is_kv_shared` handling
  (`commit_decode.py:154,568`).

### 5. Where the two paths legitimately differ (numerical, not algebraic)

- **SDPA kernel:** decode flash-decode (per token) vs prefill flash-attention (masked, 256
  queries). Same `softmax(QKᵀ·scale)·V`; different tiling/accumulation.
- **Head split / RoPE / per-head norm / MoE:** decode (1-token, sparse-matmul MoE via
  `commit_decode._commit_experts_decode_forward`) vs prefill (256-token, gathered-expert MoE via
  `denoise_forward._denoise_moe_forward`). Same routing (top-8, softmax, geglu, weighted sum),
  different kernels. MoE numerics feed the next layer's K/V, so drift compounds mildly down the
  stack — bounded and measured per-layer by the verify harness.

**Net:** the batched commit is algebraically the 256 sequential appends. Expected KV-cache
agreement is high PCC (≥ ~0.999 early layers, decreasing slightly with depth), with the
sliding-edge caveat. `verify_commit_batching.py` asserts this per layer and reports commit_ms
before/after.

## Honesty flags (do not force these)

- **Not bit-identical.** Prefill and decode kernels differ numerically; assert PCC, not equality.
  Low PCC on a specific layer means that layer's decode↔prefill op mapping (head split / RoPE /
  MoE) needs closer reconciliation before this ships as default.
- **Contiguous cache + `page_table=None` only** (the standalone / serving RUN path). Paged / vLLM
  hybrid-cache commit still uses the sequential path; the batched SDPA-read for paged caches is
  intentionally `NotImplemented` (batched paged decode is #47557 / #47488).
- **Sliding-window edge** unproven where the window bites (> ~768 committed tokens).
- **`write_batch>1`** (fast contiguous write) unproven on device; default is the per-position
  write that matches the sequential op exactly.

## How to enable / verify

```bash
# Enable the batched commit for a run (opt-in):
DG_COMMIT_BATCHED=1 python -m models.experimental.diffusion_gemma.demo.text_demo ...

# Device verify (KV bit-equivalence + commit_ms before/after) — run when QB2 is free:
DG_CKPT=/path/to/diffusiongemma-26B-A4B-it \
  python models/experimental/diffusion_gemma/doc/optimize_perf/verify_commit_batching.py \
  --mesh 1x4 --num-layers 30 --max-seq-len 1024 --prompt "The capital of France is"
```
