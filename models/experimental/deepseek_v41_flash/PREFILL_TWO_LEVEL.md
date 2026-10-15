# Two-level candidate select in V4.1 prefill

Status: phases 1-2 implemented (eager and traced prefill, server cap removed), not yet validated on device.
Phase 0's query-sliced reference indexer is not done, so `full_model_reference.py` cannot yet generate references
much past 16384 tokens. Phase 3 (sparse prefill attention) is a separate decision.

## Why

Prefill refuses any layer holding more than `candidate_topk_blocks * candidate_block_size` = 2048 * 8 = 16384
compressed entries (`DeepSeekV41PrefillAttention._check_chunk`, `new_static_buffers`). Layer 20 is a ratio-1 kv
source, so a prompt is capped at 16384 tokens. The server (`demo/server.py`, `_V41Prefiller`) caps
`--prefill-max-len` there and replays the rest of a longer prompt through decode, one token per step.

Decode already implements the two-level select (`fused_lightning_select_kv` with `candidate_blocks_output`, then
`build_candidate_pool` for the consumers).

## What the reference computes

From the checkpoint's `inference/model.py` (`Indexer.forward`, `select_candidate_blocks`), with V4.1's config:

| Layers | Ratio | Role |
| --- | --- | --- |
| 2-19 (groups 2, 8, 14) | 2 | kv / index sources; never use candidates |
| 20 | 1 | kv source, index source, `candidate_source_layer_id` |
| 21-23 | 1 | reuse layer 20's selection |
| 24, 28, 32, 36 | 1 | index sources restricted to layer 20's candidates |
| other 25-39 | 1 | reuse their group's latest selection |

Per query token:

* Level one (layer 20): score each block of 8 entries by its best index score (unreachable entries are `-inf`),
  pin the block holding the query's newest entry (`(compress_len - 1) // 8`) to `+inf`, keep the top 2048 blocks,
  drop picks that came back `-inf`. The result is a per-entry boolean mask. Layer 20's own top-512 is unrestricted.
* Level two (layers 24, 28, 32, 36): mask the layer's own scores to the candidates (`-inf` outside), then top-512.

Below 16384 entries every block is kept, so the result equals today's plain select.

Facts that shape the design:

* Layers 2-19 only hit the 16384 check by accident: nothing algorithmic is missing for them.
* Decode recomputes `cand_blocks` and its pool from the full index-key cache every step, so prefill hands decode
  no candidate state; committing entries and keys as today is enough.
* The layers are split evenly, five per stage, so layer 20 shares stage 4 with layer 24 and layers 28-36 are on
  later stages. Candidates cross stage edges like the plain selection does (which every edge but layer 20's
  carries).

## Phase 0: validation baseline (host only)

1. `tests/prefill/full_model_reference.py` runs the checkpoint's model, which already includes the pre-filter, so
   it is exact at any length. Its indexer einsum materializes `[T, heads, E]` (~275 GB at 32K tokens): run the
   reference indexer in query slices (exact, since every query is independent).
2. Generate references at 20480 tokens (just past the threshold) and 32768.

## Phase 1: eager prefill (`forward`)

In `tt/prefill/attention.py`, `DeepSeekV41PrefillAttention`:

1. `PrefillShared.candidates`: additive TILE mask `[1, 1, T, E]` (0 kept, `-inf` dropped), or `None` while
   `E <= 16384` (keep everything).
2. Layer 20 builds it from the scores `_index_scores` already returns (causal cut included):
   * per-block max: eight stride-8 ROW_MAJOR slices combined with `ttnn.maximum` -> `[T, E / 8]`;
   * add a host-built pin table, `+inf` at block `((start + i + 1) // rate - 1) // 8` of query `i`;
   * threshold at the 2048th block score per query (`ttnn.topk` then `min`, as `_select_entries` does), and drop
     blocks scoring `-inf`;
   * `repeat_interleave` by 8 back to entries; `log` of the 0/1 mask gives the additive form.
3. Index sources with `uses_candidates` add `shared.candidates` to their scores before the top-512 threshold.
4. Drop the `max_entries` check in `_check_chunk`. In `tt/prefill/model.py`, carry `candidates` across stage
   edges in `_move_shared`, as `entry_sel` is, and free it with the other shared state.
5. Validate: `test_full_model_prefill_verify.py` at `DEEPSEEK_V4_VERIFY_LEN=20480`, then 32768 (remove its
   `total <= max_entries` assert).

## Phase 2: traced prefill (`forward_static`, `DeepSeekV41TracedPrefill`)

1. Drop the refusal in `new_static_buffers`. Entry buffers then follow the real capacity (~512 MB of entries and
   ~128 MB of keys per ratio-1 copy at 512K tokens).
2. `PrefillStaticStep` gains a persistent per-tier pin table next to `index_cuts`. `topk`'s
   `k = min(2048, cap / 8)` is fixed per tier, so the trace stays static.
3. Cross-stage hand-off: send the compact block mask `[T, cap / 8]` over the existing D2D sockets and expand it
   on the receiving stage (8 MB per 1024-token chunk at 32K, 128 MB at 512K). Past ~128K, send the per-query
   block indices instead (`[T, 2048]` int32, a fixed 8 MB) and scatter them on arrival.
4. Server: remove the cap in `_V41Prefiller` (`demo/server.py`); update the 16384 notes in `tt/prefill/model.py`
   and `tt/prefill/attention.py`.
5. Validate: `test_prefill_decode_demo.py` on a prompt longer than 16384 tokens. It already compares the logits
   after prefill with replaying the prompt through decode.

## Phase 3: scaling (separate decision)

Phases 1-2 keep dense masked SDPA: each chunk's cost grows with the entries so far, so a prompt costs
O(length^2). At 512K tokens each `[1024, E]` mask or score tensor is ~1 GB, and SDPA reads every entry for each
of the 20 ratio-1 layers per chunk. Estimated practical limit: ~64K-128K tokens. Beyond that, prefill needs a
sparse attention op that gathers each query's 512 selected entries plus its window (a multi-query
`fused_lightning_select_kv`): new kernel work.

## Risks to check first

* `ttnn.topk` with k = 2048 over up to 65536 columns. Test it first. Fallback: a fixed-iteration binary search
  on the threshold (count the scores at or above it), which is also trace-safe.
* Ties: thresholding keeps them, so slightly more than 2048 blocks or 512 entries can survive. Today's top-k
  has the same behaviour.
* bf16 block maxima: near-equal block scores can flip which blocks are kept. Measure with the per-layer PCC of
  the Phase 1 test.

## Device runs

Phases 1 and 2 are validated on the Galaxy; the commands go to whoever holds the card (see
`.cursor/rules/device-access.mdc`).
