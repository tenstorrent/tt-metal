# Chunked (bounded-memory) long-context prefill

Status: current but **UNWIRED** — nothing in `tt/` or `demo/` imports `tt/chunked_prefill.py`, so
the module is reachable only from its two test files. There is no flag to turn it on:
`DG_CHUNKED_PREFILL` had zero dispatch sites and was deleted 2026-07-28
([flag triage](../../optimize_perf/flag_triage_20260728.md)). Serving's actual long-prompt path
today is `DG_PREFILL_RAGGED_LONG` (default ON, 4096-token ragged slices), **not** this one. Slightly
over the 100-line cap: the device crash trap, the two PCC gates and the repro command are never cut
for length.
Owns: the gemma4 chunked-prefill bug, the composition-not-edit workaround, its device PCC gates, and
the `O(chunk_size [+ sliding_window])` memory bound.
See also: [refuted list](../../REFUTED.md) · [vLLM-native plan](../vllm_native_plan.md) ·
[plan](../../../plan.md)

## The bug (#47466)

gemma4 multi-chunk prefill is broken: `chunk_start_idx` and `chunk_page_table` are accepted for
signature-compat and then **discarded** — `models/demos/gemma4/tt/model.py` `del chunk_start_idx`
~L1298 and `del ... chunk_page_table, chunk_start_idx` ~L1436, with `del start_pos` at :1291 and the
"Gemma4 doesn't chunk-prefill" comment (see also :1385). Three consequences: the per-chunk RoPE
offset is wrong (prefill RoPE always slices from position 0), there is no cross-chunk attention past
a single chunk, and a single chunk uses prefill memory proportional to the whole prompt (OOM past
~64k). DG may not fix it by editing the shared backbone: [no shared edits](../../../AGENTS.md).

## The fix is composition, not a backbone edit

`_swap_prefill_attention()` scopes a rebind of
`models.demos.gemma4.tt.attention.prefill_forward` to the DG-local fixed routine for the duration of
a chunked call and restores it on exit, so `git diff main -- models/demos/gemma4/` stays empty and
the whole backbone graph (layers, MoE, KV-sharing, norms, lm_head) is the real unmodified backbone.
When no chunk context is active the swapped routine defers to the saved gemma4 `prefill_forward`, so
a stray monkeypatch cannot change stock behavior. It raises `NotImplementedError` for
`batch_size > 1`.

| concern | stock gemma4 | DG chunked fix |
|---|---|---|
| RoPE | `cos[:, :, :seq_len]` (always from 0) | offset `rope_mats` dict → `cos[:, :, start:start+L]` |
| KV write | `paged_fill_cache(page_table)` | `paged_fill_cache(chunk_page_table)` — this chunk's blocks only |
| attention (full) | SDPA over this chunk only | `chunked_scaled_dot_product_attention` over the FULL `page_table` with `chunk_start_idx`, so the chunk's queries attend the whole prefix |
| attention (sliding, past window) | causal-only paged op over-attends → `NotImplementedError` | bounded rolling in-memory K/V window buffer + square causal+sliding SDPA over the buffer |

**Reference contract** copied from `models/tt_transformers/tt/{attention,generator}.py`:
`page_table` is the full per-user table and goes to the SDPA `page_table_tensor`;
`chunk_page_table = page_table[:, chunk_start_block:chunk_end_block]` goes to `paged_fill_cache`;
`chunk_start_idx = num_cached_tokens` is both the RoPE slice offset and the SDPA causal-mask offset.
RoPE reaches the chunk by **bypassing** the backbone's `_get_rope_mats` (the routine that always
slices from position 0) — cos/sin arrive pre-sliced to `chunk_start_idx : chunk_start_idx + L` via
`model.__call__(rope_mats=dict)`, resolved per layer type by `rope_mats[layer_types[i]]`
(`model.py:731-737`). Sliding layers get the sliding rope cache (theta=1e4) sliced to the chunk's
absolute positions via `_chunk_rope_mats`; a wrong sliding offset would collapse the device PCC.

**Rolling-buffer algorithm**, per chunk per sliding layer: append this chunk's RoPE'd K/V via
clone/concat (independent of the caller's `tt_k`/`tt_v`, which are deallocated after); front-align
the chunk's Q to the buffer tail with `hist_len` zero rows because `is_causal` requires `Q.s == K.s`;
keep only the tail `chunk_len` output rows; trim the buffer to the last `sliding_window` positions.
`_SlidingWindowState` is keyed by `id(weights)` — a stable per-layer key, since each layer's
`AttentionWeights` object is unique and alive for the whole prefill — and is built and released by
the `chunked_prefill` driver, so it lives exactly one prefill.

> **DEVICE CRASH TRAP.** `_sliding_window_square_sdpa` must **not** slice-and-deallocate its inputs
> the way the gemma4 strided routine does: the window buffer is **persistent** and a full-range
> `ttnn.slice` **aliases** its input, so deallocating that slice frees the live buffer. Reusing
> gemma4's routine verbatim hit the `input_tensor.is_allocated()` FATAL on the first attempt. gemma4
> never hits it because it only runs `chunked_prefill_sdpa_sliding` for `seq_len > 32768`, which is
> always multi-stride and therefore never a full-range slice. Because the bounded buffer is a single
> stride (`<= window + chunk`), ONE direct SDPA call replaces gemma4's strided loop.

## Device gates (QB2 P150x4, `(1,4)` mesh, TP=4, 2026-07-06; bar >= 0.999)

1. **Full-attention / within-window:** a 512-token prompt as **2x256 chunks** reproduces a single
   **1x512** prefill's last-token logits at **PCC 0.99998**. This gate measured **1.0** on the
   original paged-causal sliding path and **0.9999766** after being rerouted through the rolling
   buffer — both readings recorded; the small drop is the buffer path, not a regression below the bar.
2. **Sliding-window PAST the window:** a 2048-token prompt as **8x256 chunks** reproduces a single
   **1x2048** prefill at **PCC 0.99997** with `sliding_window=1024`, so the sliding layers exceed the
   window at chunk 4 (start=1024). This is the case the paged causal-only op cannot handle: it
   **fails with the old code**, which raised `NotImplementedError` once total context exceeded the
   window, and passes only with the bounded rolling K/V window buffer.

```bash
# env: see plan.md; worktree trap (TT_METAL_HOME vs PYTHONPATH): see ../README.md#device-hygiene
DG_RUN_DEVICE=1 MESH_DEVICE=P150x4 TT_LOGGER_LEVEL=ERROR ARCH_NAME=blackhole \
flock /tmp/dg-mesh.lock timeout 900 \
  python -m pytest models/experimental/diffusion_gemma/tests/test_device_chunked_prefill.py -v -s
# -> 2 passed in 10.94 s
```

Exact log lines to look for:
`[chunked-prefill] SLIDING last-token logits PCC (8x256 vs 1x2048, window 1024): 0.9999728` and
`[chunked-prefill] last-token logits PCC (2x256 vs 1x512, window 1024): 0.9999766`.
CPU structural gate: `pytest tests/test_chunked_prefill_math.py` → **5 passed**, device-free,
covering block math and page-table slicing against the tt_transformers contract. SHAs for the
2x256 == 1x512 work: `6181cf1f62c` + `8f840f32ce3`. Device note: the mesh hit the recurring eth core
29-25 reset timeout on the first device open, at `device.py` setup before any test ran; a fresh
device open on retry recovered it with no `tt-smi -r`.

## Memory bound

- **Full-attention layers: `O(chunk_size)`** (default 256) — each chunk projects/RoPEs/fills only its
  own tokens and all prior chunks are read straight from the paged KV cache, never materialized as
  an activation.
- **Sliding-window layers: `O(chunk_size + sliding_window)`** — peak is `sliding_window + chunk_size`
  before the post-chunk trim.
- **Why the bound is exact:** a sliding query at absolute position `p` attends only `(p-window, p]`,
  all of which is inside the trimmed buffer, so the bounded result is identical to a single
  full-length sliding prefill — which is what the 0.99997 PCC confirms.

Both bounds are independent of `prompt_len`, so prompts far past the stock single-chunk
`O(prompt_len)` OOM cliff (>64k) prefill in a fixed footprint set by
`chunk_size` + `sliding_window` + the paged KV cache (required capacity, not prefill scratch).

## Files, scope, open items

- `tt/chunked_prefill.py` (attention routine + driver + sliding rolling-window buffer),
  `tests/test_device_chunked_prefill.py` (the two device PCC gates),
  `tests/test_chunked_prefill_math.py` (CPU structural tests).
- **Scope:** single-user `batch_size == 1` prototype; batched chunked prefill needs #47557 + #47488.
- **PERF CAVEAT (open, quantified):** the square causal SDPA computes `hist_len`
  (`<= sliding_window`) **discarded** history-query rows per chunk — at `chunk_size=256`,
  `window=1024` that is ~5x the useful query rows on sliding layers; `chunk_size=1024` amortizes it
  to ~2x. Correctness-exact, memory-bounded, and not on the correctness-gate path.
- **OPEN FOLLOW-UP:** sliding paged-cache **decode after** chunked prefill is not exercised. The
  sliding layers still `paged_fill_cache` their K/V, but a bounded circular sliding cache filled
  per-chunk across chunk boundaries (`cache_position_modulo`) is separate work; the rolling buffer
  serves only the prefill SDPA.
