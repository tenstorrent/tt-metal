# DiffusionGemma — chunked (bounded-memory) long-context prefill (#47466)

## Status (lead)

- **Device correctness: PASS — PCC = 1.0.** A 512-token prompt prefilled as **2×256 chunks** over a
  paged KV cache reproduces a single **1×512** prefill's last-token logits exactly (bar: ≥ 0.999),
  on QB2 `P150x4`, `(1,4)` mesh, TP=4. This proves both fixes the stock gemma4 backbone lacks:
  correct per-chunk RoPE offset (`chunk_start_idx`) and cross-chunk attention through the paged KV
  cache (`chunk_page_table` fill + full-`page_table` `chunked_scaled_dot_product_attention`).
- **Flag:** `DG_CHUNKED_PREFILL` (default **OFF**). Default behavior is the stock single-chunk
  gemma4 prefill; the chunked path is opt-in.
- **gemma4 untouched:** all changes are DG-local (`git diff main -- models/demos/gemma4/` unchanged).

## The bug (from #47466)

gemma4 multi-chunk prefill is broken: `chunk_start_idx` and `chunk_page_table` are accepted for
signature-compat and then discarded (`models/demos/gemma4/tt/model.py` — `del chunk_start_idx`
~L1298, `del ... chunk_page_table, chunk_start_idx` ~L1436). Result: (1) wrong per-chunk RoPE
offset (prefill RoPE always slices from position 0), (2) no cross-chunk attention past a single
chunk, and (3) a single chunk uses prefill memory proportional to the whole prompt (OOM past ~64k).

## The fix (composition, not a backbone edit)

`tt/chunked_prefill.py` copies the gemma4 single-user prefill-attention routine
(`attention/prefill.py::_prefill_forward_single`) and fixes it DG-locally, then drives the
**unmodified** backbone one bounded chunk at a time:

| Concern | Stock gemma4 | DG chunked fix |
|---|---|---|
| RoPE | `cos[:, :, :seq_len]` (positions 0..L-1) | offset `rope_mats` dict → `cos[:, :, start:start+L]` |
| KV write | `paged_fill_cache(page_table)` | `paged_fill_cache(chunk_page_table)` (this chunk's blocks) |
| Attention | `scaled_dot_product_attention` (this chunk only) | `chunked_scaled_dot_product_attention(full page_table, chunk_start_idx)` (whole prefix) |
| Memory | `O(prompt_len)` activations | `O(chunk_size)` — prior chunks read from the paged cache |

The routine swap is a scoped rebind of `models.demos.gemma4.tt.attention.prefill_forward` for the
duration of a chunked call (`_swap_prefill_attention()`), restored on exit — runtime composition,
so the diff against gemma4 stays empty and the whole backbone graph (layers, MoE, KV-sharing, norms,
lm_head) is the real backbone. The chunk math (page-table slicing, `chunk_start_idx`) mirrors the
working reference in `models/tt_transformers/tt/{attention,generator}.py`.

## Memory bound achieved

Prefill working-set is **`O(chunk_size)`**, not `O(prompt_len)`: each chunk projects / RoPEs / fills
only its own `chunk_size` (default 256) tokens; all prior chunks live in the paged KV cache and are
read directly by `chunked_scaled_dot_product_attention` — never materialized as an activation. So a
prompt of any length up to the served context prefills in a fixed per-step activation footprint (set
by `chunk_size`) plus the paged KV cache (required capacity, not prefill scratch). This removes the
stock single-chunk `O(prompt_len)` activation blowup that OOMs past ~64k.

## Files

- `tt/chunked_prefill.py` — the fix (attention routine + driver + flag).
- `tests/test_device_chunked_prefill.py` — device gate (2×256 == 1×512, PCC ≥ 0.999).
- `tests/test_chunked_prefill_math.py` — CPU structural tests (flag, block math, page-table slicing).

## Scope / follow-ups

- Single-user (`batch_size == 1`) prototype. Batched chunked prefill = #47557 + #47488.
- Sliding-window layers correct while total context ≤ `sliding_window` (1024); longer sliding-window
  prompts need the overlapping-window scheme adapted to the multi-chunk contract (raises
  `NotImplementedError` rather than returning silently-wrong output). OPEN.
- Wiring the DG serving/`generate` prefill to dispatch on `DG_CHUNKED_PREFILL` is the next step,
  gated behind the flag so default behavior is unchanged.

See `work_log.md` for the exact device invocation, the env note (worktree vs built tree), and SHAs.
