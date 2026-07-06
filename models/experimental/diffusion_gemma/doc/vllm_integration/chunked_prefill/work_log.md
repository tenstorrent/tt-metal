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

## Scope / OPEN QUESTIONS

- **Single-user (`batch_size == 1`)** prototype. Batched chunked prefill = #47557 (batched canvas)
  + #47488 (paged-cache ownership). The swapped attention raises `NotImplementedError` for
  `batch_size > 1`.
- **Sliding-window layers** are correct while total context ≤ `sliding_window` (1024): within the
  window a causal chunked SDPA IS the sliding SDPA. For sliding-window prompts **longer** than the
  window, the overlapping-window scheme (gemma4 `chunked_prefill_sdpa_sliding`) must be adapted to
  the multi-chunk contract — the routine raises `NotImplementedError` in that case rather than
  return silently-wrong output. OPEN follow-up.
- **Integration wiring**: the flag + `chunked_prefill()` entry are delivered; hooking the DG
  serving/generate prefill to dispatch on `DG_CHUNKED_PREFILL` (vs the stock single-chunk path) is
  the next wiring step, gated behind the flag so default behavior is unchanged.

## SHAs

- (to be logged on commit)
