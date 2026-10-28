# DSpark parallel verify for V4.1-Flash: plan

Status: plan. Follows `DSPARK.md`, whose verify (section 3.1) runs the block `[t, d1..dk]` as k+1
pipelined single-token steps. This plan replaces that with **one decode step that carries all
T = k+1 rows through every layer at once**, so the drafter's 5 proposals (k = 5, T = 6) are checked
by a single pass of the 7-stage pipeline.

## 1. Why

Measured with the pipelined verify (`DSPARK.md` 3.4): 1 token 30.7 ms, 5 tokens 55.2 ms, i.e. each
extra row costs one slowest-stage time `s` (~6 ms). A verify of T rows costs `(S + T - 1) * s` for
S stages.

A wide step costs `S * s_T`, where `s_T` is a stage's time with T rows. Decode on these stages is
bound by weight reads and op latency, not by FLOPs, so a batched op over 6 rows costs about the same as
over 1. The break-even point with S = 7 and T = 6 is `s_6 < 12/7 * s_1 = 1.7 s_1`.

| s_6 / s_1 | verify ms (T = 6) | vs pipelined (~61 ms) |
|---|---|---|
| 1.0 | ~31 | 2.0x faster |
| 1.3 (target) | ~40 | 1.5x |
| 1.7 | ~52 | break-even |
| 6 (every op looped per row) | ~185 | 3x slower |

**The whole plan depends on every per-layer op being truly batched, not looped per row.** Looping is
fine as a correctness scaffold (Phase 1-2), and it is never the shipping path.

## 2. What "a wide step" has to compute

Rows j = 0..T-1 sit at positions `p + j` of **one session**. That is not the same as V4's batched
users, which each have their own caches:

* Row j must see the **causal** prefix: KV of positions `<= p + j`, which includes rows `0..j` of this
  step and nothing after them.
* Every compressed layer's indexer selects **per row**: row j scores the compressed keys that exist at
  `p + j` and gets its own top-k set.
* Ratio-1 and ratio-2 compressors close groups **inside the block**. Row j's group is visible to rows
  after j.
* The result must equal the T single-token steps exactly (up to bf16), so the spec stream stays
  equal to plain greedy.

## 3. Op-by-op inventory at T rows

"Batched today" means the op already takes T ROW_MAJOR replica rows (the V4 multi-user decode path,
`matmul_decode` up to 8 rows).

| Piece | Today | Needed | Kind |
|---|---|---|---|
| Packet, step context (`_build_packet`, `_step_context`) | `_decode_batch` tokens, one position | T tokens plus base `p`; `pos_f = p + arange(T)` on device, so every RoPE row / slot / entry / mask is per row | host + trace |
| Embedding, RoPE rows, norms | batched | none | none |
| q_a, q_b, kv, o_b, compressor wkv / wgate, indexer wq_b / weights_proj, router gate (`LinearDecode`) | batched (<= 8 rows) | check fused-norm epilogues and output core grids at T rows | test |
| o_a (`BatchedLinearDecode`) | `assert m == 1` | M = T: per-row group-major layout, one weight read | **kernel** |
| mHC (`DeepSeekV4HyperConnection(row_major_streams=True)`) | `assert t == 1` (8-core single-user program) | multi-row RM `fused_hyperconnection` and `fn` (`LinearDecodeLargeK` with T rows): `[T*hc, D/8]` per collapse core | **kernel** |
| Routed + shared experts (`fused_experts`) | ROW_MAJOR path is one token row; the TILE path pads to 32 rows and its CBs do not fit on the output stage | multi-row RM replica: take the union of the rows' top-6, read each expert's weights once, and apply it to the rows routed to it | **kernel (largest item)** |
| Sliding attention (ratio 0) | one ring, `cur_pos` | per-row ring view `ring_j = ring * keep_j + scatter_j @ kv_new` (rows `0..j` in their slots), then `flash_multi_latent_attention_decode` with B = T users, K `[T, 1, W, Dh]` | trace (existing ops) |
| Ratio-2 compressor (`_latent`) | one row; pools from `win_kv` on odd positions | pair rows inside the block (`p` even: pairs (0,1), (2,3), ...; `p` odd: row 0 pairs with `win_kv`'s slot). Pool every closed pair, then leave the trailing open row in `win_kv`. The pairing is a variant (`p % 2`) | trace |
| Ratio-1 compressor | one entry | T entries | trace |
| Index-key / `comp_kv` writes | one row | T rows at their entries (`paged_update_cache` with T update indices, or T calls) | trace |
| Select (`fused_lightning_select_kv`, `merge_select_kv`) | `B == 1`, `Sq == 1` | row j: its own `cur_pos` / key count, its own output rows. Scaffold: T calls. Target: one multi-query select that reads the keys once and writes T top-k sets | **kernel** |
| `sel_tile` / `sel_kv` | one `[1, 1, W + topk, Dh]` per group | `[T, 1, W + topk, Dh]`; each row's ring part built as for sliding layers | alloc |
| Two-level candidates (`build_candidate_pool`, `cand_blocks`, `pool_*`) | per group | per row (T pools), or fall back (section 4.4) | trace / fallback |
| Compressed SDPA | B = 1, mask or `cur_pos` | B = T, per-row `cur_pos` (or a per-row mask before the window fills) | trace |
| Engram (`write_rows`, `engram_gate`) | `engram.batch` rows, op takes `[B, 1, hc, D]` | `hash_session(slot, ids[1, T], p)` already hashes a run of one session; build the Engram layer with `batch = T` | host |
| Taps (stash, stage 7) | one row at `pos % stash_rows` | T rows | trace |
| Final collapse, norm, lm_head | batched | plus in-trace argmax, so the output socket carries T ids + top-2 gaps, not T x 258 KB of logits | trace |
| D2D socket buffers (`streams_in`, `pre_mix_in`, cache hand-off) | sized by `_decode_batch` | T rows; cache hand-off rows (`new_kv_row` / `new_key_row`) become T rows | alloc |

## 4. Design

### 4.1 One verify trace per variant, at fixed T

* `prepare_spec_decode(..., verify_rows=T)` allocates the T-row buffers next to the single-row
  ones; plain decode and prefill keep using the existing traces.
* Each submesh captures a verify trace per reachable variant. The variant is
  `(p % 2, select mode, causal)`: the r=2 pairing depends on the parity of the block's first row.
* T is fixed per capture (6). A round that verifies fewer drafts fills the extra rows with any token,
  and the accept rule treats them as rejected. They write positions `p + k + 1 ..`, which the fixup
  restores exactly as it does for rejected drafts. Capacity checks (`ensure_session_capacity`) must
  leave T - 1 positions of slack.

### 4.2 Causality inside the step

* **Sliding rings.** The step computes the T new kv rows first. For row j, the attention reads a
  per-row copy of the ring with the slots of `p .. p + j` replaced by rows `0 .. j`; slots of later rows
  keep their old values (positions `p + j - 127 ..`, which row j does see). The one-hot `scatter_j` /
  `keep_j` come from the packet (`[T, W, T]`, built on device from `pos_f`). After attention, all T rows
  go into the persistent ring at `(p + j) % W`.
* **Compressed groups.** Write the block's T index keys / `comp_kv` rows first, then select per row
  with that row's key count (`_index_position(pos_f_j)`). Entries past row j's position are not counted,
  so they are invisible to it. This matches the single-step path, because entries are keyed by
  position.
* **Masks / cur_pos.** `_compressed_mask` and `_compressed_curpos` take `pos_f` `[1, 1, 1, T]` and give
  per-row outputs.

### 4.3 Rollback

Same state as `DSPARK.md` 3.2, written T rows at a time:

* Ring shadows and the fixup trace are unchanged. The fixup restores the slots of the previous
  round's rejected positions.
* `win_hist_kv` / `win_hist_gate` take T rows per step at `(p + j) % hist`. The fixup still restores
  the window slot of the last verified position.
* Position-keyed state (`comp_kv`, index keys, stash, Engram hash) needs no rollback, as before.

Later (Phase 5): defer the ring commit instead. The step writes its rows into a staging buffer; the
next step, once it knows `m`, commits rows `0..m` into the ring before its own attention. That removes
the shadow copy and the fixup trace.

### 4.4 Fallback

A block that straddles a boundary goes through the existing pipelined verify:

* the first row below `sliding_window - 1` and the last at or above it (the causal switch);
* the two-level threshold (`_two_level`), if Phase 2 does not build per-row candidate pools;
* the end of `max_seq`.

Prefill (`decode_prompt_traced`) also stays single-row until Phase 5, where it can feed the prompt
T rows at a time on the same traces.

### 4.5 Host loop (`tt/spec.py`)

```
1. spec_fixup(rejected, frontier=p-1)
2. ids, gaps = decode_verify_traced([t, d1..dk, pad..], p)   # one step, T rows, device argmax
3. m = leading j with X[j+1] == ids[j];  emit ids[0..m]
4. spec_write_main(p..p+k, written=[j <= m])
5. spec_draft(ids[m], frontier=p+m)
6. rejected <- p+m+1 .. p+T-1;  p <- p+m+1
```

`DSparkSpeculator` gets a `parallel_verify` switch. The pipelined path stays, both as the fallback
and as the reference in tests.

## 5. Files

Per `PORTING.md`: V4 hooks default to V4 behavior, and the V4.1 files hold only what changed.

| File | Change |
|---|---|
| `tt/model.py` | `verify_rows`, T-row buffers, `_step_context` over `pos_f [1, 1, 1, T]`, verify traces and variant key, `decode_verify_traced`, fallback test |
| `tt/decode/attention.py` | `decode` with `S = T` rows (`decode_inputs(..., rows=T)`): T-row compressor, per-row ring views, per-row select, B = T SDPA; T-row `sel_tile` in `build_decode_caches` |
| `tt/decode/decoder_layer.py` | T-row mHC; MoE through the multi-row experts |
| `tt/decode/moe.py`, V4 `tt/decode/moe.py` | T-row replica path (lift `assert t == 1` behind the op's new support) |
| V4 `tt/decode/hyperconnection.py` | `row_major_streams` with T rows |
| V4 `tt/layers.py` (`BatchedLinearDecode`) | M = T |
| `tt/decode/engram.py` | `batch = T` for a run of one session |
| `tt/spec.py` | `parallel_verify` round |
| `ttnn/.../deepseek/moe/fused_experts` | multi-row ROW_MAJOR replica input, expert union |
| `ttnn/.../deepseek/hyperconnection` | multi-row single-user program |
| `ttnn/.../deepseek/fused_lightning_select_kv`, `merge_select_kv` | T queries over one key set, per-row key count |
| `tests/decode/test_verify_rows.py` (new) | per layer kind: one T-row step vs T single-row steps |
| `tests/decode/test_dspark_spec.py` | parallel-verify mode in the exactness and acceptance runs |

## 6. Phases and exit criteria

Every device run needs approval first.

**Phase 0: go / no-go measurement.** Device micro-benchmarks on one stage:
* `LinearDecode` (q_a, q_b, o_b, shared expert) at 1 vs 6 rows;
* `fused_experts` once vs 6 calls;
* `fused_hyperconnection` once vs 6 calls;
* `fused_lightning_select_kv` at 8k and 64k keys, once vs 6 calls;
* flash-MLA decode at B = 1 vs B = 6.

Exit: a per-op `s_6 / s_1` table, and a projected stage ratio below 1.5 once the kernel items land.
If the select or the experts scale close to linearly and cannot be fixed, stop here.

**Phase 1: sliding-only correctness, looped scaffold.** T-row packet and context, the sliding
attention with per-row ring views, MoE / mHC / o_a looped per row, Engram at `batch = T`, taps,
device argmax. Exit: on the sliding layers, one T-row step equals T single steps (logits PCC >=
0.999, identical ring contents), at positions below and above the window and across the wrap.

**Phase 2: compressed layers.** Ratio-1 / ratio-2 compressors in the block (both parities), T entry
writes, per-row select (looped), per-row `sel_tile`, B = T SDPA. Two-level falls back at first. Exit:
every layer kind (r=2 source, r=1 source, index consumer, candidate source and consumer) matches T
single steps; selections match row for row.

**Phase 3: kernels.** Multi-row `fused_experts` (expert union), multi-row `fused_hyperconnection`, o_a
at M = T, multi-query select. Each one has its own op test against the looped scaffold. Exit:
`s_6 / s_1 <= 1.3` on every stage, measured with the profiler, and nothing over the L1 budget of the
output stage.

**Phase 4: spec integration.** `parallel_verify` in `DSparkSpeculator`, the fixed-T trace, the
fallback. Exit:
* the corrupted-oracle stream equals plain greedy over >= 256 tokens for k = 1..5 (crossing the ring
  wrap, both r=2 phases, and the fallback boundaries);
* the DSpark stream equals plain greedy;
* tok/s/user and the round split reported against the pipelined verify.

**Phase 5: follow-ups.**
* Deferred ring commit, which removes the fixup.
* Per-row candidate pools, which removes the two-level fallback.
* Prompt replay T rows per step.
* The drafter's per-row `fused_experts` / `fused_hyperconnection` loops (`tt/decode/dspark.py`, ~13 ms
  draft trace) moved onto the same multi-row kernels.

## 7. Risks

* **Output-stage L1.** T-row replicas, per-row `sel_tile`s and the multi-row expert CBs sit beside the
  prefetch rings, the same constraint that forced the drafter onto single-row ops. Budget each kernel
  for stage 7 first.
* **Expert union.** 6 rows x top-6 can reach 36 distinct experts. In the worst case the weight reads
  are the same as 6 calls, and the win comes only from overlap and from removing per-op latency.
  Phase 0 has to measure how much overlap real text gives.
* **Select at long context.** It scales with keys, not weights. Looping it 6 times at 64k+ keys can
  dominate the stage. The multi-query kernel is required, not optional.
* **Trace count.** The verify variants double the traces per stage. Check `TRACE_SIZE_SCALING.md`.
* **Numerics.** A T-row matmul or SDPA can reduce in a different order than the single-row one. Accept
  bf16-level logit differences, and report the top1-top2 gap at any divergence from greedy (as in
  `DSPARK.md` Phase 4).
