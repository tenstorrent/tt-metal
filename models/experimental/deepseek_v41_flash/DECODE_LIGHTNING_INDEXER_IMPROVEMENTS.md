# Decode lightning indexer: improvements

Status: plan. Measured on `dsv41_decode_32k_32L` (32 layers, decode at position ~32k, 8 stages x TP4
on Blackhole, batch 1): `generated/profiler/reports/dsv41_decode_32k_32L/2026_10_30_08_39_55/`.
Op times are device kernel durations (`ops_perf_results_*.csv.gz`). Phase times inside
`fused_lightning_select_kv` come from its `SEL-*` device zones in `profile_log_device.csv`, at
1350 MHz.

Companion: `DECODE_SDPA_IMPROVEMENTS.md` (the attention that reads the selection).

## 1. What runs

Of the first 32 layers, 2, 8, 14, 20, 24 and 28 are `index_source_layer_ids`. Each scores the
compressed index keys and keeps the top `index_topk = 512`. The other compressed layers reuse the
selection. Layers 2, 8, 14 and 20 are also `kv_source_layer_ids` (they run the compressor). Layer 20
is the `candidate_source_layer_id`: it also keeps the best `candidate_topk_blocks = 2048` blocks of
`candidate_block_size = 8` keys, and layers 24 and 28 select only among those.

At position 32k the select scores `(cur_pos + 1) // 4` = 8192 keys of `index_head_dim` 128 against
`index_n_heads` = 32 query heads. That is a 2 MB key read and ~67 MFLOP, a few microseconds of
roofline. It is unsplit: `num_parts = 1`, since `KEY_SPLIT_MIN_KEYS` is 80k keys.

| piece | per call | calls / token | per token |
|---|---|---|---|
| `FusedLightningSelectKv`, plain (layers 2, 8, 14, 24, 28) | 125-131 us | 5 | 0.64 ms |
| `FusedLightningSelectKv`, candidate source (layer 20) | **250 us** | 1 | 0.25 ms |
| query / head-weight replication before the select (6 ops, section 3.3) | 30.8 us | 6 | 0.18 ms |
| `BuildCandidatePool` (layers 24, 28) | 37.2 us | 2 | 0.07 ms |
| indexer `wq_b` (10.0) + its RoPE (6.5) + `weights_proj` on **1 core** (7.7) | 24.2 us | 6 | 0.15 ms |
| compressor `wkv` / `wgate`, 16 cores, bf16 | 33.4 us | 7 | 0.23 ms |
| `sel_kv` -> `sel_tile` `Tilize` | 7.6 us | 6 | 0.05 ms |
| **total** | | | **~1.6 ms** (6% of a 27.8 ms token) |

The select's cost grows with context. Its per-core scans are linear in keys, and its pass count and
syncs are fixed. So this share grows past 32k, while SDPA's does not.

## 2. Inside the select: 2/3 of it is radix passes

Per-RISC durations: BRISC 130 us, NCRISC 128 us, the three TRISCs ~16 us. Scoring (compute) is
about an eighth of the op. Everything else is the cross-core top-k and the gather, run by the
dataflow kernels (`writer_fused_lightning_select_kv.cpp`). The coordinator core's timeline for one
plain select (device 0):

| us from start | phase (`SEL-*` zone) | duration |
|---|---|---|
| 0 - 2.6 | write this step's new row (`NEW-ROW`) | 1.5 |
| 2.6 - 16.6 | score + pass-0 histogram (`HIST`) | 14 |
| 16.6 - 101 | **4 radix passes**, each: signal / release (1.3), `OWNER-SUM` (6.2), `COORD-WAIT` + `COORD-WALK` (2.9), `REFINE-HIST` (6-12) | **~21.5 each, 86 total** |
| 101 - 108 | list the selected rows (`LIST`) | 6.2 |
| 108 - 126 | gather 512 rows of `comp_kv` into `sel_kv` (`GATHER-RD` / `-WR`) | 18 |

86 cores hold keys, and 64 are histogram owners. Three findings drive the plan:

1. **The passes do not narrow.** The key is the fp32 score mapped to an order-preserving uint32,
   radixed 8 bits at a time from the top (`kHistBits = 8`, `num_hist_bins = 256`). The top 8 bits
   are the sign and 7 exponent bits, nearly the same for every score. So pass 0 keeps nearly every
   key in the threshold bin, and so does pass 1. `REFINE-HIST` (the rescan of the kept keys)
   takes 11.1, 12.4, 12.4 and 6.2 us in passes 0-3. Only the last one shrinks. Two of the four passes
   are spent on bits that carry no information.
2. **The gather runs on 3 of 120 cores.** Each core gathers its own selected keys, and the top 512
   cluster in a few cores' contiguous key blocks. Three cores move all 512 KB (15-18 us). The other
   117 finish in 0.07 us.
3. **The candidate source repeats it all for blocks.** Layer 20's `SEL-CAND` (64 us on the
   coordinator) builds each block's best key and runs a second 4-pass `radix_select` over blocks.
   At 32k there are 8192 / 8 = 1024 blocks, fewer than the 2048 kept, so **every block is
   selected**. That select is pure overhead below position 4 x 2048 x 8 = 64k. The main key select
   of layer 20 is also slower (`HIST` 24 us, `REFINE-HIST` 16-19 us per pass).

## 3. Changes

### 3.1 Radix from the first informative bit (2 passes instead of 4)

During `HIST`, each core already visits every local key. Track each row's min and max order key
there. Then, before the pass-0 histogram, run one small exchange: every active core writes its two
words per row into a coordinator slot, as the pass counts already travel, and the coordinator
multicasts the start bit `first = 31 - clz(min ^ max)` with the release it already sends. Passes then
take 8 bits from `first` down instead of from bit 31. All keys share the bits above `first`, so this
is exact: the same selection, ties still broken by index order.

The pass-0 histogram has to wait for `first`, so `HIST` splits into score-and-min/max, then
histogram. The exchange is one signal / release round (~1.3 us today) and the histogram is one more
scan of the local keys.

With 8192 keys spread over a 16-bit informative window, a threshold bin after 2 passes holds about
one key, and the existing stopping rule ("a row stops once its threshold bin can be taken whole")
ends there.

* Wider first pass as an option: 11 bits (2048 bins) fits a 32k-key row in one pass plus a short
  refinement, at 8 KB of histogram per core. Measure it against 2 x 8 bits.
* The block select of 3.2 gets the same treatment.

**Expected: ~-38 us per select** (two passes of ~21.5 us, less the exchange and the extra scan),
**~-0.25 ms/token** with the candidate source's block passes.

### 3.2 Skip the block select when every block is kept

In the `CANDIDATE_SOURCE` section, `num_blocks = div_up(num_keys, candidate_block_size)` is known on
device. When `num_blocks <= candidate_topk_blocks`, the answer is all blocks in order. Each core
lists its `num_local_blocks` at offset `first_local_block` with no histogram or `radix_select`, and
`pool_count = num_keys`. This is a runtime branch, so one trace still serves every position.

The consumers (layers 24 and 28) then run `build_candidate_pool` (37 us each) to copy every key into
the pool. With the same flag (`pool_count == num_keys`) they can score `idx_key_cache` directly
through its page table, skipping the copy. That is a runtime choice between two key tensors the op
already knows.

**Expected below 64k positions: -60 us (layer 20) and -74 us (pool builds), -0.13 ms/token.**

### 3.3 Replicate the query inside the op

`DeepSeekV41Indexer.select_kv` gives the op a query and head-weight replica on every core
(`_replicate_heads`). It costs 6 ops and 30.8 us per select:

* query: `ReshapeView` 9.4 us, `RepeatCodegen` 9.5 us, `InterleavedToSharded` 8.0 us;
* weights: `ShardedToInterleaved` 0.4 us, `RepeatCodegen` 2.4 us, `InterleavedToSharded` 1.1 us.

The replicas are 8 KB (32 heads x 128, bf16) and 64 B. The reader can fetch them itself from the one
L1 copy `wq_b` / `weights_proj` leave, or core 0 can multicast them once (the op already
multicasts headers over its `mcast_bbox`). The `query` / `head_weights` "borrowed from this core's
replicated shards" in `reader_fused_lightning_select_kv.cpp` become a read into the same buffers.
**Expected: -28 us per select, -0.17 ms/token.**

### 3.4 Gather on every core

After the final allocation, core 0 knows each core's `(output_offset, num_contribute)`. The
row-id lists already travel to core 0's page in `PARTIAL_OUTPUT` / candidate mode. Do that in all
modes, multicast the 512-entry list (2 KB), and let core `c` gather rows
`[c * k / N, (c + 1) * k / N)`, ~4-5 rows instead of up to ~170. That is one more multicast
(~1-2 us) for an 18 us gather that becomes ~2 us. While gathering, group rows by 32 and tilize them
in compute, so the op writes `sel_tile` directly and the `Tilize` op goes away (see
`DECODE_SDPA_IMPROVEMENTS.md` 3.5). **Expected: -15 us per select plus the 7.6 us `Tilize`, about
-0.14 ms/token.**

### 3.5 The per-pass sync (second order)

Each pass's `OWNER-SUM` (6.2 us) has 64 owners each read their histogram slice from all 86 active
cores. After 3.1 there are 2 passes, so this matters half as much. Still worth trying:

* fewer, fuller scoring cores: 8192 keys on 32 cores is 256 keys each. That doubles the per-core
  scans, which 3.1 shortens, and cuts the owners' reads by 2.7x;
* a two-level histogram reduction (row-wise, then column-wise) instead of every owner reading every
  core.

`REFINE-HIST` also runs ~95 cycles per key, a scalar loop over the fp32 score tiles in L1.
Unrolling it and keeping the order key of the previous pass (instead of recomputing it from the
score) is worth a look once the pass count is down.

### 3.6 Upstream of the select

* `weights_proj` (`K` 5120, `N` 32, `n_blocks: 1` in `DECODE_LAYOUTS`) reads 320 KB on one core:
  7.7 us. Split `K` over 8-16 cores (a K-split `matmul_decode` with a reduction) for ~2 us:
  -0.03 ms/token.
* The indexer q RoPE (6.5 us, 64 cores) is the same shape as the attention q RoPE. Fold it into
  the `wq_b` epilogue with the plan in `docs/fuse_partial_rope_into_matmul_decode.md`.
* Compressor `wkv` / `wgate`: 33.4 us each on 16 cores with bf16 weights (5.2 MB each, ~157 GB/s),
  7 calls per token. They stay bf16 because the checkpoint keeps them in bf16 (the
  `DeepSeekV41Attention` docstring), so the lever is cores, not dtype. Concatenate them into one `[5120, 1024]` matmul on 32-64 cores reading the
  same activation replica. The current `n_blocks: 16` is chosen so they share kv's all-gather
  replica, and a wider grid needs its own. Expected ~10-12 us per pair instead of 67:
  -0.2 ms/token.

## 4. Order of work and expected result

| step | change | where | per token |
|---|---|---|---|
| 1 | radix from the first informative bit (3.1) | writer kernel | -0.25 ms |
| 2 | skip the all-kept block select; consumers skip the pool copy (3.2) | writer kernel, `build_candidate_pool` / select reader | -0.13 ms (below 64k) |
| 3 | in-op query replication (3.3) | reader kernel, `select_kv` | -0.17 ms |
| 4 | gather on every core, tiled output (3.4) | reader / writer / compute | -0.14 ms |
| 5 | compressor pair as one wider matmul, `weights_proj` K-split (3.6) | `DECODE_LAYOUTS`, `tt/decode/attention.py` | -0.23 ms |
| 6 | sync and scan tuning (3.5) | writer kernel | TBD |

Steps 1-5 come to ~0.9 ms of the token at 32k, more at longer context. The select op alone would fall
from ~130 to ~75-80 us before 3.5's sync and scan tuning.

## 5. Validation

* Exactness: the selection must stay bit-identical. 3.1 and 3.2 are exact by construction, and
  3.4 changes only who copies the rows.
  `tests/ttnn/unit_tests/operations/experimental/deepseek/test_fused_lightning_select_kv.py` compares
  the selected indices and gathered rows against torch `topk`. Add cases for score ranges that span
  few and many exponents, all-equal scores (pure tie-break), and `num_blocks` just below, at and
  above `candidate_topk_blocks`.
* Perf: `test_fused_lightning_select_kv_multi_row_perf.py` and `test_candidate_pool_perf.py`, plus
  the `SEL-*` zones. The pass count shows directly as the number of `SEL-COORD-WALK` zones per call
  (4 today, 8 at the candidate source).
* End to end: `test_full_model_decode_demo.py` at `DEEPSEEK_V4_START_POS=32768` (and 64k+, past the
  candidate threshold), text unchanged and tok/s/u up. Then re-profile 32L.

## 6. Open questions

1. Is the 16-bit informative window typical, or do some layers' scores span more exponents? The
   min/max start bit adapts either way, but the pass count does not. Log `first` per layer in a
   debug build.
2. Below 64k, can the candidate source skip its block work entirely, including the block-key
   pass that overwrites the scores (`scores_l1[b] = best`)?
3. Past 80k keys the key split (`num_parts`, `merge_select_kv`) takes over. Re-measure there after
   steps 1-4, since a cheaper unsplit select moves `KEY_SPLIT_MIN_KEYS` up.
