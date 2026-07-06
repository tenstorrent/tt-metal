# DiffusionGemma — chunked (bounded-memory) long-context prefill (#47466)

## Status (lead)

- **Device correctness: PASS.** Two device gates on QB2 `P150x4`, `(1,4)` mesh, TP=4 (bar: ≥ 0.999):
  - **Full-attention + within-window (Agent D):** a 512-token prompt as **2×256 chunks** reproduces a
    single **1×512** prefill's last-token logits — **PCC = 0.99998** (was 1.0 on the old
    paged-causal sliding path; now routed through the rolling window buffer, still ≥ 0.999).
  - **Sliding-window PAST the window (Agent D2):** a **2048-token** prompt as **8×256 chunks**
    reproduces a single **1×2048** prefill — **PCC = 0.99997**. With `sliding_window=1024` the
    sliding layers EXCEED the window (chunk 4, start=1024), so this is the case the paged
    causal-only op cannot handle. It **fails with the old code** (which raised `NotImplementedError`
    once total context > window) and **passes** with the bounded rolling K/V window buffer.
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
| Attention (full) | `scaled_dot_product_attention` (this chunk only) | `chunked_scaled_dot_product_attention(full page_table, chunk_start_idx)` (whole prefix) |
| Attention (sliding, past window) | causal-only paged op over-attends → `NotImplementedError` | bounded rolling in-memory K/V window buffer + square causal+sliding SDPA over the buffer |
| Memory (full) | `O(prompt_len)` activations | `O(chunk_size)` — prior chunks read from the paged cache |
| Memory (sliding) | `O(prompt_len)` activations | `O(chunk_size + sliding_window)` — trimmed rolling buffer |

The routine swap is a scoped rebind of `models.demos.gemma4.tt.attention.prefill_forward` for the
duration of a chunked call (`_swap_prefill_attention()`), restored on exit — runtime composition,
so the diff against gemma4 stays empty and the whole backbone graph (layers, MoE, KV-sharing, norms,
lm_head) is the real backbone. The chunk math (page-table slicing, `chunk_start_idx`) mirrors the
working reference in `models/tt_transformers/tt/{attention,generator}.py`.

## Memory bound achieved

Prefill working-set is bounded (independent of prompt length), so a prompt of any length up to the
served context prefills in a fixed per-step footprint. This removes the stock single-chunk
`O(prompt_len)` activation blowup that OOMs past ~64k, for **both** attention kinds:

- **Full-attention layers: `O(chunk_size)`.** Each chunk projects / RoPEs / fills only its own
  `chunk_size` (default 256) tokens; all prior chunks live in the paged KV cache and are read
  directly by `chunked_scaled_dot_product_attention` — never materialized as an activation.
- **Sliding-window layers: `O(chunk_size + sliding_window)`.** The paged chunked SDPA op is
  causal-only (no window mask), so sliding layers instead thread a bounded rolling in-memory K/V
  window buffer: each chunk appends its RoPE'd K/V, runs the square causal+sliding SDPA over the
  buffer for this chunk's queries, then trims the buffer to the last `sliding_window` positions
  (peak `sliding_window + chunk_size`). A sliding query at pos `p` only attends `(p-window, p]`, so
  the trimmed buffer always contains every key it needs — making the bounded result identical to a
  single full-length sliding prefill (device PCC 0.99997 above).

So >64k prompts are bounded: the footprint is set by `chunk_size` + `sliding_window` + the paged KV
cache (required capacity, not prefill scratch), not by `prompt_len`.

## Files

- `tt/chunked_prefill.py` — the fix (attention routine + driver + flag + sliding rolling-window buffer).
- `tests/test_device_chunked_prefill.py` — device gates: 2×256 == 1×512 (full/within-window) and
  8×256 == 1×2048 (sliding PAST the window), both PCC ≥ 0.999.
- `tests/test_chunked_prefill_math.py` — CPU structural tests (flag, block math, page-table slicing).

## Scope / follow-ups

- Single-user (`batch_size == 1`) prototype. Batched chunked prefill = #47557 + #47488.
- Sliding-window layers now work for prompts of **any** length, including longer than the window
  (Agent D2, device PCC 0.99997). Perf caveat: the square causal SDPA computes ≤ `sliding_window`
  discarded history-query rows per chunk; larger `chunk_size` amortizes it. See `work_log.md`.
- Wiring the DG serving/`generate` prefill to dispatch on `DG_CHUNKED_PREFILL` is the next step,
  gated behind the flag so default behavior is unchanged.

See `work_log.md` for the exact device invocation, the env note (worktree vs built tree), and SHAs.
