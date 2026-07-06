# DiffusionGemma — chunked (bounded-memory) long-context prefill (#47466 / Agent D)

## Goal

The shared gemma4 backbone's multi-chunk prefill is broken: `chunk_start_idx` and
`chunk_page_table` are accepted for signature-compat and then discarded
(`models/demos/gemma4/tt/model.py`: `prepare_inputs_prefill` ~L1298 `del chunk_start_idx`;
`ttnn_prefill_forward` ~L1436 `del ... chunk_page_table, chunk_start_idx`). This gives a wrong
per-chunk RoPE offset, no cross-chunk attention past a single chunk, and prefill memory
proportional to the whole prompt (OOM past ~64k).

Deliver a DG-local, bounded-memory multi-chunk prefill that WORKS, without editing
`models/demos/gemma4/`, behind a flag `DG_CHUNKED_PREFILL` (default OFF). Prove it with a device
correctness check: 512-token prompt as 2×256 chunks == a single 1×512 prefill, PCC ≥ 0.999.

## What I changed (all DG-local — gemma4 untouched)

- **`tt/chunked_prefill.py`** (new) — the fix, composed over the unmodified backbone:
  - `chunked_prefill_attention_forward(...)` — a COPY of gemma4
    `attention/prefill.py::_prefill_forward_single`, signature-compatible with gemma4
    `attention.prefill_forward`, with the three fixes:
    1. RoPE: cos/sin arrive **pre-sliced** to the chunk's absolute positions
       (`chunk_start_idx : chunk_start_idx+L`) via a `rope_mats` dict the driver passes to the
       backbone's `__call__` (which bypasses `_get_rope_mats`, the routine that always slices from
       position 0).
    2. KV fill: `paged_fill_cache` uses `_CHUNK_CTX.chunk_page_table` (this chunk's blocks), not
       the full page table.
    3. SDPA: `chunked_scaled_dot_product_attention` over the **full** `page_table` with
       `chunk_start_idx` → the chunk's queries attend the entire KV prefix (all prior chunks),
       reading prior chunks straight from the paged cache (never materialized → bounded memory).
  - `_swap_prefill_attention()` — a scoped context manager that rebinds
    `models.demos.gemma4.tt.attention.prefill_forward` to the fixed routine for the duration of a
    chunked call, then restores it. This is **runtime composition, not a source edit** — the whole
    backbone graph (layers, MoE, KV-sharing, norms, lm_head) stays the real unmodified backbone, so
    chunked-vs-single is apples-to-apples, and `git diff main -- models/demos/gemma4/` stays empty.
    When no chunk context is active the swapped routine defers to the saved gemma4 `prefill_forward`
    (so a stray monkeypatch can't change stock behavior).
  - `chunked_prefill(model, prompt_embeds, ...)` — the driver: loops the prompt in `chunk_size`
    tokens, builds the per-chunk `chunk_page_table = page_table[:, start_block:end_block]` and the
    offset RoPE dict, runs the backbone `__call__` once per chunk with only that chunk's
    activations resident, and returns the final chunk's logits. Mirrors the
    `models/tt_transformers/tt/generator.py` chunk math (`chunk_start_idx = num_cached_tokens`,
    `chunk_page_table = source_page_table[:, chunk_start_block:chunk_end_block]`).
  - `chunked_prefill_enabled()` — reads `DG_CHUNKED_PREFILL` (default OFF).
- **`tests/test_device_chunked_prefill.py`** (new) — the device acceptance gate (below).
- **`tests/test_chunked_prefill_math.py`** (new) — CPU-only structural tests (flag default, chunk
  block math, page-table slicing vs the tt_transformers contract). **5 passed** (device-free).

## Reference contract used (from `models/tt_transformers/tt/attention.py` + `generator.py`)

- `page_table` = full per-user page table (blocks `0..chunk_end`) → SDPA `page_table_tensor`
  (attend the whole prefix).
- `chunk_page_table` = `page_table[:, chunk_start_block:chunk_end_block]` → `paged_fill_cache`
  (write only this chunk).
- `chunk_start_idx` = absolute chunk start → RoPE slice offset + SDPA causal-mask offset.
This is exactly what gemma4 discards; the fix re-honors it in DG-local code.

## Device evidence

- **Device correctness (2×256 vs 1×512): PCC = 1.0 — PASS** (2026-07-06, QB2 `P150x4`, `(1,4)` mesh,
  TP=4). `DG_RUN_DEVICE=1 MESH_DEVICE=P150x4 pytest tests/test_device_chunked_prefill.py` →
  **1 passed** in 15.48s. Log line: `[chunked-prefill] last-token logits PCC (2x256 vs 1x512): 1.0`.
  A 512-token prompt prefilled as two 256-token chunks over a paged KV cache reproduces the stock
  single 1×512 prefill's last-token logits **exactly** (PCC 1.0 ≥ 0.999 bar). This is only possible
  when BOTH the per-chunk RoPE offset (`chunk_start_idx`) and cross-chunk attention
  (`chunk_page_table` fill + full-`page_table` `chunked_scaled_dot_product_attention`) are correct:
  chunk-1's queries (positions 256–511) attend chunk-0's KV read straight from the paged cache, and
  are RoPE-rotated to their true absolute positions.
- CPU structural: `pytest tests/test_chunked_prefill_math.py` → **5 passed** (device-free).
- gemma4 isolation gate: `git diff main -- models/demos/gemma4/` still shows only the pre-existing
  branch commits (24 files, spec-decode/LM-head/RoPE), **none from this feature** — my increment
  adds zero gemma4 edits (all new files under `models/experimental/diffusion_gemma/`).

### Exact device invocation

```bash
source /home/zni/venvs/tt-diffusion-gemma/bin/activate
export TT_METAL_HOME=/home/zni/tt-metal TT_METAL_RUNTIME_ROOT=/home/zni/tt-metal \
       ARCH_NAME=blackhole PYTHONPATH=/home/zni/tt-metal-chunk \
       DG_RUN_DEVICE=1 MESH_DEVICE=P150x4 TT_LOGGER_LEVEL=ERROR
flock /tmp/dg-mesh.lock timeout 900 \
  python -m pytest models/experimental/diffusion_gemma/tests/test_device_chunked_prefill.py -v -s
```

> Env note: this is a `git worktree` at `/home/zni/tt-metal-chunk`, but the compiled runtime
> firmware + editable `ttnn` live in the built tree `/home/zni/tt-metal`. So `TT_METAL_HOME`
> **must** point at `/home/zni/tt-metal` (else the brisc linker script `firmware_brisc.ld` is
> missing and every kernel build fails), while `PYTHONPATH=/home/zni/tt-metal-chunk` keeps `models`
> resolving to this worktree's code. `ttnn` resolves from the editable install regardless.

## Memory bound achieved

Prefill working-set is `O(chunk_size)`, not `O(prompt_len)`: each chunk projects/RoPEs/fills only
its own `chunk_size` tokens; prior chunks live in the paged KV cache and are read directly by
`chunked_scaled_dot_product_attention` (never materialized as an activation). So a prompt of any
length up to the served context prefills in a fixed per-step footprint set by `chunk_size` (default
256) + the paged KV cache (which is required capacity, not prefill scratch). This removes the
single-chunk `O(prompt_len)` activation blowup that OOMs the stock path past ~64k. (Quantified
byte figures to be added from the device run / `memory_budget.py`.)

## Agent D2 increment — sliding-window chunked prefill PAST the window (2026-07-06)

Removed Agent D's `NotImplementedError` for sliding-window layers once total context exceeds the
sliding window (1024). The paged `chunked_scaled_dot_product_attention` op is **causal-only** (no
window mask), so it over-attends for sliding layers past the window. Fix (all DG-local, gemma4
still untouched), in `tt/chunked_prefill.py`:

- **Per-sliding-layer bounded rolling K/V window buffer**, threaded across chunks. New
  `_SlidingWindowState` (dict keyed by `id(weights)` — a stable per-layer key, since each layer's
  `AttentionWeights` object is unique and alive for the whole prefill) carried on `_ChunkContext`;
  built/released by the `chunked_prefill` driver so it lives for the whole prefill and is freed at
  the end. Full-attention layers are **unchanged** (paged `chunked_scaled_dot_product_attention`).
- **`_bounded_sliding_sdpa`** — per chunk, for each sliding layer: (1) append this chunk's RoPE'd
  K/V to the layer's buffer (`clone`/`concat` → independent of the caller's `tt_k`/`tt_v`, which
  are deallocated after); (2) run the causal+sliding SDPA over the buffer for this chunk's queries
  (front-align the chunk's Q to the buffer tail with `hist_len` zero rows so `is_causal` requires
  `Q.s == K.s`; keep only the tail `chunk_len` output rows); (3) **trim** the buffer back to the
  last `sliding_window` positions. A sliding query at absolute pos `p` attends only `(p-window, p]`,
  all inside the trimmed buffer, so the bounded result is identical to a single full-length
  sliding-window prefill.
- **`_sliding_window_square_sdpa`** — the single square causal+sliding SDPA, the DG-local core of
  gemma4 `operations.chunked_prefill_sdpa_sliding`. NOTE it does **not** slice-and-deallocate its
  inputs (unlike the gemma4 strided routine): the buffer is *persistent* and a full-range
  `ttnn.slice` **aliases** its input, so deallocating that slice would free the live buffer — the
  exact `input_tensor.is_allocated()` FATAL hit on the first attempt when the gemma4 routine was
  reused verbatim. gemma4 never hits this (it only runs `chunked_prefill_sdpa_sliding` for
  `seq_len > 32768` → always multi-stride → never a full-range slice). Since the bounded window
  buffer is a single stride (`<= window + chunk`), one direct SDPA call replaces the strided loop.
- **RoPE**: sliding layers get the sliding rope cache (θ=1e4) sliced to the chunk's absolute
  positions via `_chunk_rope_mats` → `model.__call__(rope_mats=dict)` (per-layer-type dict lookup
  `rope_mats[layer_types[i]]`, `model.py:731–737`); the K/V appended to the buffer is therefore
  RoPE'd at the correct absolute offset. Verified by the device gate below (a wrong sliding offset
  would collapse PCC).

### Device evidence — sliding past the window: PCC = 0.99997 — PASS

QB2 `P150x4`, `(1,4)` mesh, TP=4, 2026-07-06. `DG_RUN_DEVICE=1 MESH_DEVICE=P150x4 pytest
tests/test_device_chunked_prefill.py -v -s` → **2 passed** in 10.94s:

- `test_chunked_prefill_sliding_past_window` (NEW): **2048-token prompt as 8×256 chunks vs single
  1×2048**, `sliding_window=1024` (so sliding layers EXCEED the window at chunk 4, start=1024).
  Last-token (pos 2047) logits **PCC = 0.9999728 ≥ 0.999**. Log:
  `[chunked-prefill] SLIDING last-token logits PCC (8x256 vs 1x2048, window 1024): 0.9999728`.
  This FAILS with the old code by construction: the removed sliding branch raised
  `NotImplementedError` once `chunk_start_idx + chunk_len > window` (chunk 4). Even without that
  guard, the causal-only paged op would over-attend (query 2047 would see positions `[0,1023]` the
  window excludes), collapsing PCC. Matching the single 1×2048 prefill (which applies
  `sliding_window_size=1024`) at 0.99997 proves the rolling buffer reproduces the exact window mask
  across chunk boundaries.
- `test_chunked_prefill_matches_single` (Agent D's gate, still green under the new sliding path):
  512 as 2×256, window 1024 (> prompt). **PCC = 0.9999766** (was 1.0 on the old paged-causal
  sliding path; now routed through the rolling buffer — still ≥ 0.999). Log:
  `[chunked-prefill] last-token logits PCC (2x256 vs 1x512, window 1024): 0.9999766`.
- CPU structural: `pytest tests/test_chunked_prefill_math.py` → **5 passed** (device-free), unchanged.
- gemma4 isolation gate: `git diff main -- models/demos/gemma4/` unchanged by this increment (only
  files under `models/experimental/diffusion_gemma/` touched).

> Device note: the mesh hit the recurring **eth core 29-25** reset-timeout on the first device open
> (a known QB2 flake, not caused by this code — it failed at `device.py` setup before any test
> ran); a fresh device open on retry recovered it (no `tt-smi -r` by this agent). See
> `../work_log.md` and the DG serving-env memory for the recurring pattern.

### Memory bound achieved (updated)

- **Full-attention layers:** `O(chunk_size)` prefill scratch — prior chunks live in the paged KV
  cache and are read directly by `chunked_scaled_dot_product_attention` (never materialized).
- **Sliding-window layers:** `O(chunk_size + sliding_window)` — the rolling buffer holds at most
  `sliding_window + chunk_size` positions at peak (trimmed to `sliding_window` after each chunk).
- Both are bounded (independent of prompt length), so prompts far past the single-chunk
  `O(prompt_len)` OOM cliff (>64k) prefill in a fixed footprint set by `chunk_size` + `window` +
  the paged KV cache (required capacity, not prefill scratch).

## Scope / OPEN QUESTIONS

- **Single-user (`batch_size == 1`)** prototype. Batched chunked prefill = #47557 (batched canvas)
  + #47488 (paged-cache ownership). The swapped attention raises `NotImplementedError` for
  `batch_size > 1`.
- **Sliding-window layers now work for prompts of any length** (this increment) — resolved.
- **Perf caveat (sliding layers):** the bounded buffer SDPA computes `hist_len` (≤ `sliding_window`)
  discarded history-query rows per chunk (front-padded zero Q) because `is_causal` needs a square
  `Q.s == K.s`. With `chunk_size=256`, `window=1024` that is ~5× the useful query rows on sliding
  layers. It is correctness-exact and memory-bounded (the task's goal); a larger `chunk_size`
  amortizes it (e.g. `chunk_size=1024` → ~2×). Not on the correctness-gate path.
- **Sliding paged-cache decode-after-chunked-prefill:** the sliding layers still `paged_fill_cache`
  their K/V (unchanged), but a bounded circular sliding cache filled per-chunk across chunk
  boundaries (`cache_position_modulo`) is not exercised by the prefill-logits gate; decode-readable
  sliding KV across chunks is a separate follow-up (the buffer serves only the prefill SDPA).
- **Integration wiring**: the flag + `chunked_prefill()` entry are delivered; hooking the DG
  serving/generate prefill to dispatch on `DG_CHUNKED_PREFILL` (vs the stock single-chunk path) is
  the next wiring step, gated behind the flag so default behavior is unchanged.

## SHAs

- Agent D (2×256==1×512 PCC=1.0): commits `6181cf1f62c` + `8f840f32ce3` (see branch log).
- Agent D2 (sliding past window, PCC=0.99997): logged on commit below.
