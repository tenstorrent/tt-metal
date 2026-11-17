# WS8 · Prefix-cache continuation prefill (A, 17 Nov 2026)

Branch `ws8-prefix` (4 commits), landed on `integration` together with B's `ws5-l1-fix` (415ff12e8a). Gate: B on the
merged commit (WS5 on), A on f8cc89d + WS8 (WS5 off). Status: **landed**.

## Why

The agentic evals with prefix caching (`GEMMA4_PREFIX_CACHING=1`, `--enable-prefix-caching`) stalled. A cached-prefix
hit runs `prefill_forward(start_pos > 0)`, which went through `MultichipDecoder._prefill_continuation`: one eager
batch-1 decode per suffix token, ~5-14 tok/s. Every agent turn appends a few hundred to a few thousand tokens, so
one turn blocked the server for minutes. `GEMMA4_PREFIX_TRACE_SUFFIX=1` covered slot 0 only. Without prefix caching,
re-prefilling the whole context every turn starved decode (9 agents: ~6 tok/s per user, many 10 s windows with no
generated token).

## What changed

`Gemma4Generator.prefill_forward`: every cached-prefix continuation runs as same-user decode rows, up to 32 positions
per step, through the batched decode layer (the WS6 verify contract: per-row RoPE and SDPA positions, K/V written in
row order by one `PagedUpdateRows` per tensor and layer, one write table per layer group). One 32-row step is traced
per (cache, page-table shape) and replayed per chunk; a short chunk pads with position -1 rows (no cache write, no
attention). `GEMMA4_26B_WS8_ROWS=0` restores the token-at-a-time path, `GEMMA4_26B_WS8_TRACE=0` runs the rows
eagerly. B's `ws5-l1-fix` makes the token-at-a-time path free the L1 it creates under WS5 (it clashed with prefill).

## Result

| Continuation after an 8K cached prefix (`ws8_bench.py`, full model) | Time | tok/s |
|---|---|---|
| Token-at-a-time (before) | 32 tokens: 2.31 s | 14 |
| WS8 rows, eager | 2048: 5.49 s | 373 |
| **WS8 rows, traced** (A, no WS5) | 512: 0.46 s / 2048: 1.84 s | **1118 / 1112** |
| **WS8 rows, traced** (B, merged with WS5) | 512 / 2048 | **1146 / 1131** |
| Cold prefill of the whole prompt, for comparison | 10240: 5.41 s | 1894 |

Proof ≥ 1000 tok/s ✓. A 2048-token turn after 8K now costs 1.8 s instead of 5.4 s re-prefill (or ~150 s before).

## Accuracy

- `tests/test_ws8_rows_prefill.py` (2 layers; page-aligned, partial page + two chunks, two full chunks, past the
  sliding window): rows vs token-at-a-time last-logit PCC 0.9937-0.9999, same top-1, within 0.01 of the serial path vs
  a cold prefill; next-step PCC 0.99989-0.99997 with `GEMMA4_26B_WS5_FUSED=0` (≥ 0.999: the K/V writes are right) and
  0.99741-0.99997 with WS5 on (≥ 0.995: the serial reference writes K/V through WS5's merged-MoE weights). B: 4/4 in
  both modes; cached prefix and neighbour pages bit-identical.
- Full model, real text (last position of a 32-token suffix after 8K, `last_logits` in the JSONs). Full-vocab logit
  PCC is tail-dominated (the existing serial path is 0.89-0.95 against a cold prefill), so the bar is relative:

| Text, build | rows/cold | serial/cold | rows/serial | top-1 equal | top-5 overlap rows/cold, serial/cold |
|---|---|---|---|---|---|
| A's text, 2K prefix, no WS5 | 0.889 | 0.887 | 0.817 | ✓ | 3, 3 |
| A's text, 8K, no WS5 | 0.945 | 0.926 | 0.967 | ✓ | 3, 4 |
| A's text, 8K, WS5 (B) | 0.945 | 0.946 | 0.982 | ✓ | 3, 4 |
| B's text, 8K, WS5 off / on (B) | 0.920 | 0.945 / 0.947 | 0.959 / 0.960 | ✓ | **2**, 3 |

On B's text the rows path is 0.025 further from the cold prefill than the serial path and shares 2 of its top 5
(serial: 3). It is the same with WS5 on and off, so it is a property of the batched (expert-parallel) layer at that
position, not of WS5. Landed on the end-to-end evidence below; Terminal89 / SWE100 on the final commit are the check.
- Common gate on the merged commit (B): `test_device_numerics` + `test_prefix_caching` 16/16, quick_check PASS,
  B1 step 4.995 ms (4K 5.43), B8/16/32 17.60/20.06/22.38 ms, serving 1 user 198.4 tok/s, 16 users 63.1 tok/s/user,
  GPQA-10 9/10 (serve_bench runs without prefix caching).

## Served agentic runs with prefix caching (A, f8cc89d + WS8, before WS5)

`~/gemma26-eval/agentic` (31B QB2 CI recipe and task lists; `results/WS8/agentic_smoke_scores.txt`):

| Cohort | Resolved | Exceptions | Wall time |
|---|---|---|---|
| SWE-bench Verified, first 5 (5 concurrent) | 4/5 | 0 | 35 min |
| Terminal-Bench 2, first 5 (5 concurrent) | 1/5 | 0 | 3 h (one task to its budget) |
| SWE-bench Verified, first 32 of SWE100 (28 concurrent) | **17/32 = 53.1 %** | 0 | 2 h 28 min |

Prefix-cache hit rate 94-96 % of prompt tokens. Generation ~190 tok/s total at ~28 agents (was ~50 with stalls
without prefix caching). Reference: Gemma 4 31B on SWE100, QB2 CI 51-64 %, GPU 64.8 %.

## Known limits

- Rows run through the batched decode layer (expert-parallel MoE), not WS5's fused batch-1 layer: same numerics class
  as WS1 batched decode.
- A continuation longer than ~16 % of the context is still faster as a cold prefill (1100 vs ~1900 tok/s). The rows
  path is used for every continuation; vLLM's scheduler, not the model, picks prefix reuse.
- The traced step is released with the generator's other traces and recaptured on the next continuation.
