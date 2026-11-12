## P.38 perf (8x4, 2026-11-06): both KV caches bf16 for now (CKV_FORMAT=bf16, WINDOW_KV_FORMAT=bf16_tiles defaults)
- Owner 2026-11-06 (PD migration first cut). Defaults changed in tt/settings.py; old path selectable:
  `DS41_CKV_FORMAT=bfp4_rows DS41_WINDOW_KV_FORMAT=bfp8_tiles`. INDEX_K_FORMAT stays bfp4. Contract unchanged (served
  caches were already bf16, DS41-143); `state.formats()` unchanged (all bf16).
- Chosen: bf16_tiles kept (per-chip region + neighbour send, P.18 / P.26), compressed layers read the bf16 history table
  plus the bf16 region through a new sdpa fork option, `ttnn.bringup.sparse_sdpa(kv_format=BF16, kv_window=region)`
  (KV_WINDOW: chunk 0 from kv_window, top-k chunks from kv, [window | top-k] lists with two sentinel tails, as ROW_BFP4;
  DS41-212). The table is history only (tt/compressor.history_only: base 0, [1, 1, hist, 512] bf16; no window region),
  so no region -> table copy per layer and no list compaction. Stock sparse_sdpa cannot take two tensors.
- Decoder tail (replay128, 32 rows a chip < 128): bf16_tiles now falls back to `bf16` (WindowTables, P.31 style):
  window-only layers the replicated bf16 row table through the fork (BF16, max_grid; the stock op takes no core cap),
  compressed layers `WindowTables.window_rows` (load-time [128 + S, 512] bf16) as kv_window (with a bf16 or a bfp4_rows
  history). Before, bf16_tiles asserted there. model.build_tail asserts history_only() instead of CKV_FORMAT=bfp4_rows.
- Rule 18: the fork call takes max_grid (11 x 10 = 110 cores on the Galaxy); slice_write of window_rows max_grid.
  OV-23: no Python-built op added (the sdpa fork is a C++ device op; WindowSend stays prepared). OV-3: nothing padded.
  OV-25: only the new fork tests run (test_sparse_sdpa_kv_window_bf16.py 10 passed; row_bfp4 test_refusals 1 passed).
- Memory: the bf16 table row is 1024 B vs 320 B (bfp4_rows), 3.2x; the plan budgets the table in bf16 (plan.md
  "sparse-attention table" 1.00 GiB a chip at 1M on 4x4), so it fits as planned. Region per layer 294912 B a chip
  (P.26) vs 166464 B (bfp8_tiles).
- Measured (`orchestrator measure --task P.38`, change on, ab/P.38/new; vs P.37's run of the old path, ab/P.37/new):
  quick: every layer / state within its limit, host_transfers_per_layer 0. compress_kv err L02 0.027 (old 0.115),
  L08 0.056 (0.136), L14 0.127 (0.176), L20 0.098 (0.156); layers L00-L38 5-15 % lower (L20 0.124 vs 0.135, L38 0.117
  vs 0.124); L39 0.825 vs 0.741 (limit 1.533; the end-of-model outlier, chaotic); window_kv L01-L19 lower, L21-L29
  slightly higher (0.100 vs 0.094 at L21, downstream of the changed history; limits 0.155+). p51200: chunk wall
  203.7 vs 194.0 ms, device 183.57 vs 177.27 ms, attention section 50.01 vs 44.12 ms (bf16 rows: 3.2x the history
  bytes, 2x the window bytes), compressor 1.90 vs 1.77; profiled host dispatch 186 vs 156 ms (1300 vs 1254 calls:
  bf16_tiles' to_rm + slice_write a layer instead of one ckv_bfp4). Gate units passed here: test_short_turn (wall
  158.2 ms, device 146.16), test_contract, test_kv_layout, test_adapter_acks; test_ladder / test_profile left to the
  orchestrator.
- Not changed: tests/bringup/test_logit_kl.py asserts the shipped WINDOW_KV_FORMAT is bfp8_tiles (owner 2026-11-01);
  it now fails that assert (diagnostic test, not in this gate; the owner's 2026-11-06 default supersedes it).
- Re-run: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/sdpa/tests/unit/test_sparse_sdpa_kv_window_bf16.py`;
  `python -m models.demos.common.bringup.orchestrator measure --task P.38 --spec <spec>`.

## P.39 perf (8x4, 2026-11-06): served index keys as Blaze decode's 192-byte fp8 rows (INDEX_K_SERVED=fp8_row192)
- Owner-authorized contract change (PD migration first cut). Setting `INDEX_K_SERVED` (tt/settings.py): `fp8_row192`
  (default) / `bf16` (the old contract). A/B: on `DS41_INDEX_K_SERVED=fp8_row192`, off `DS41_INDEX_K_SERVED=bf16`
  (both with `DS41_ENGRAM_TABLE=resident`).
- Row: uint8 [.., 192] a compressed position: e4m3 of key / scale (128 B) | scale fp32 LE (4 B) | 60 zero B; pad rows
  (groups past actual_end) all zero. Records 16 x 192 = 3072 B (2_index_k_r2), 32 x 192 = 6144 B (4_index_k_r1); same
  ND shard / bank / SP placement / TP replication as before.
- Write path (tt/compressor.py `fp8_rows`, `TtServedRows._put`): the chunk's server-order bf16 key slab (as before) ->
  to_layout TILE (capped) -> stock `ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(round_scale_to_power_of_two
  =False)` (TILE path: one core per 32 x 128 block, no grid option: 10 / 20 cores on 8x4, asserted <= MAX_WORKER_CORES)
  -> new Python op `ttnn.bringup.pack_fp8_rows` (fp8_rows_ttnn: data movement only, prepared, max_grid; 20 / 40 cores)
  -> `ttnn.bringup.update_padded_kv_cache` into the served uint8 slab (the copy path takes uint8 as is) and into each
  re-index mirror. Same slab feeds the bf16 compute copy as before.
- Re-index layers: tt-blaze's REINDEX stage (csa_attention/op.py:1126-1161) quantizes the stage payload's
  `index_cache_row` = the producer's post-norm post-RoPE bf16 key (stage_payload.py:29-32); the only producer before
  24-36 is the candidate-source stage, layer 20 (csa_candidate_pool_stage/op.py:271-281). So 4_index_k_r1 covers
  [20, 24, 28, 32, 36] under fp8_row192 (`compressor.reindex_layers`, kv_contract / caches `of` map); layer 20's
  TtServedRows writes the same rows into the mirrors (`mirrors`: the engine slabs, or own tensors in the ladder) in
  the same chunk, before layer 20's ack.
- Indexer choice: a persistent bf16 compute copy (the old served slab, now private, `TtServedRows.slab`), so the
  indexer's numerics are those of the old path (INDEX_K_FORMAT=bfp4 rounds it as before). Cost: memory (8x4, 1M, a
  slot, a chip: compute copy 80 MiB; served 60 MiB vs 80 MiB bf16; mirrors 96 MiB; total 236 vs 80 MiB) and per chunk
  a source: tilize + cast + pack + 1 (r2) / 5 (r1) extra update_padded_kv_cache on W / r rows. Reading the fp8 rows
  instead would dequantize the whole capacity each chunk (the stock per_token_cast_back takes a core a row: > 110).
- Byte checks (OV: the brief's "exact torch recipe" does not hold on this device, DS41-213): served bytes == the stock
  cast packed (test_kv_layout: every index layer incl. mirrors, 0 rows differ; RM-input cast = TILE-input cast); tt-
  blaze's `_quantize_scaled_keys`: scales byte-equal, ~1.1 % of values one e4m3 step away (the cast lands one step
  higher at truncation edges and keeps subnormals), never more. tt-blaze's own test compares its writer with the cast
  (`_glm_quantized_key`), not the recipe. Raw fp8 bytes must be read from the host buffer (DS41-214).
- Test readers: caches.decode / fp8_decode (independent of tt/), fp8_row_failures (zero padding, positive scale, the
  row's largest |e4m3| 416 or 448, all-zero pad rows), bytes_equal on raw rows; acks: a re-index layer's records
  byte-equal to layer 20's; golden comparison of decoded rows at the unchanged limits (re-index layers against their
  source's golden and limit, server_rules.state_threshold). kv_contract.read_rows decodes for the producer read-back.
  `state.served` / `to_torch` decode the served rows (load_served of decoded rows re-encodes: d2p is off).
- Rules: 18 (cast asserted <= cap, pack / tilize / update capped); OV-23 pack_fp8_rows prepared (patch_count 3);
  OV-3 nothing padded to fit an op (the 60 zero bytes are the contract's row); OV-25 only the new op's unit test run.
- Re-run: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/fp8_rows_ttnn/tests/unit/test_pack_fp8_rows.py`;
  `python -m models.demos.common.bringup gate P.39 --spec <spec> --check --unit P.39#test_kv_layout`;
  `python -m models.demos.common.bringup.orchestrator measure --task P.39 --spec <spec>`.
- Measured (`orchestrator measure --task P.39`, change on, ab/P.39/new; vs P.38's change-on run = this task's off path
  before the code changed, ab/P.38/new; the orchestrator measures the real off side): quick: no metric over its limit,
  host_transfers_per_layer 0; every err_layer_L* identical to P.38 to the last digit (the indexer scores the same bf16
  compute copy); err_state_index_k (the served rows decoded) L02 0.0525 (bf16 0.0156, limit 0.236), L08 0.0612 (0.0340,
  0.260), L14 0.0997 (0.0876, 0.292), L20 0.0856 (0.0707, 0.276): the fp8 rounding's cost, far inside the limits
  (calib x the reference's own fp4 drift). p51200: chunk wall 200.8 vs 203.7 ms (noise), device 182.97 vs 183.57 ms,
  compressor section 1.98 vs 1.90 ms (+0.08: tilize + cast + pack + the extra writes), indexer 6.98 vs 6.99,
  profiled host dispatch 175.1 vs 186.4 ms (run noise). pack_fp8_rows host 74 us a call (32 chips, prepared).
- Gate units passed here on the final code: test_kv_layout (byte-exact), test_adapter_acks (0 failures; 4 chunks x 40
  acks), test_contract (producer index_k PCC 0.9927, limit 0.9465; contract_checks_failed 0, acks_early 0). Left to
  the orchestrator: test_ladder, test_profile, test_short_turn.

## P.40 perf (8x4, 2026-11-07): index keys bf16 vs bfp8 vs bfp4 (INDEX_K_FORMAT gains bfp8)
- Change: `INDEX_K_FORMAT` (tt/settings.py) choices bf16 / bfp4 / bfp8; tt/indexer.py `INDEX_K_DTYPES` maps them to
  the score's key dtype: bfp8 = the tiled key slab rounded to bfloat8_b by ttnn.bringup.bfp_round (prepared, compact,
  into the layer's persistent buffer, as bfp4) and the ring's gathered-key buffer bfloat8_b. bf16 / bfp4 unchanged.
  Default stays bfp4 (owner decides). Query keeps INDEX_Q_FORMAT=bfp4; served keys stay INDEX_K_SERVED=fp8_row192.
  No op change: ring_indexer_score_dsa already takes q and k each in its own format; bfp_round already does bfloat8_b.
  Prepared ops touched (OV-23): bfp_round (prepared + compact). Cores: bfp_round on max_grid() (110), unchanged.
- Fork test: indexer_score tests gain a q bfp4 / k bfp8 ratio-2 case (q_dtype / k_dtype). Not run on the 8x4: a
  2x2 FABRIC_2D sub-mesh fails the router handshake (K-68; I ran it once, 24 s, no hang, no reset). For F.1 / a 2x2 box.
- Measured, same code (f5072f7f1d2aa663), env DS41_ENGRAM_TABLE=resident + the arm: bfp8 = `orchestrator measure
  --task P.40` (ab/P.40/new: quick + p51200); bf16 / bfp4 = the same batch plans and env (core/batch, BRINGUP_AB=1,
  BRINGUP_BATCH_MAX_SEQ=56320) by /scratch/dnijemcevic/p40ab/drive.py -> /scratch/dnijemcevic/p40ab/<arm>/; ladder
  rung last per arm (err_final_hidden, top5, every state metric) in p40ab/<arm>/ladder. All runs rc 0, nothing over a limit.

  | metric (limit) | bf16 | bfp4 (default) | bfp8 (new) |
  |---|---|---|---|
  | err_final_hidden (<= 0.242) | 0.1319 | 0.1334 | 0.1313 |
  | top5_overlap (>= 0.994) | 1.000 | 1.000 | 1.000 |
  | err_state_topk_idx L02 / L08 / L14 / L20 | .079 / .069 / .068 / .087 | .164 / .151 / .119 / .151 | .080 / .070 / .067 / .087 |
  | err_state_topk_idx L24 / L28 / L32 / L36 | .134 / .083 / .094 / .086 | .253 / .156 / .154 / .146 | .134 / .083 / .093 / .086 |
  | (limits L02..L36: .217 .172 .166 .222 .318 .216 .222 .212) | | | |
  | overlap_state_topk_idx L02 / L08 / L14 / L20 | .921 / .931 / .933 / .913 | .836 / .849 / .881 / .849 | .920 / .930 / .933 / .913 |
  | overlap_state_topk_idx L24 / L28 / L32 / L36 | .867 / .917 / .907 / .914 | .747 / .844 / .846 / .854 | .867 / .918 / .907 / .914 |
  | err_state_index_k L02 / L08 / L14 / L20 | .0525 / .0607 / .0985 / .0870 | .0525 / .0612 / .0997 / .0856 | .0525 / .0607 / .0981 / .0853 |
  | device_ms_indexer (p51200) | 7.18 | 6.97 | 7.04 |
  | device_ms_total (p51200) | 183.27 | 185.11 | 183.56 |
  | chunk_wall_ms (p51200; spread) | 204.3 (6.8) | 202.9 (12.0) | 206.5 (10.2) |
  | chunk_wall_ms (quick, last chunk) | 179.8 | 180.8 | 181.4 |
  | ring key buffers a chip, 56320 / 1,044,480 tokens | 93.7 / 1738 MB | 26.4 / 489 MB | 49.8 / 923 MB |

  Reading: bfp8 scores like bf16 (top-k overlap within 0.001 of bf16 at every layer); bfp4 loses ~6-12 points of
  overlap (L24: 0.747) but stays inside every limit. Time: indexer 7.0-7.2 ms a chunk in all three (diffs ~0.2 ms,
  run noise level); wall within spread. err_state_index_k is the served fp8 rows (INDEX_K_SERVED), so the same in every
  arm by design. Key-buffer bytes: 8 indexer layers each keep a replicated [T, 128] buffer (2 / 8 / 14 at T = seq / 2,
  20 / 24 / 28 / 32 / 36 at T = seq); bytes a 128-row value: bf16 256, bfp8 136, bfp4 72.
- Caveat (not in my paths): the F137 CPU diagnosis (reference/ds41_ref.py "index_k" site, bringup/hooks.py) models
  only the bfp4 grid for index keys; under bfp8 it would model bfp4. Diagnosis only, never gated.
- Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.40 --spec <spec>`; other arms:
  `python /scratch/dnijemcevic/p40ab/drive.py bf16:ql bfp4:ql bfp8:l` then devjob `bash /scratch/dnijemcevic/p40ab/run.sh`.

## P.41 perf (8x4, 2026-11-07): one bfp8 index-key copy, served 8-bit rows built from it (INDEX_K_COPY=bfp8)
- Setting `INDEX_K_COPY` (tt/settings.py): `bfp8` (default) / `bf16` (the path before, unchanged). A/B: on
  `DS41_INDEX_K_COPY=bfp8`, off `DS41_INDEX_K_COPY=bf16` (both `DS41_ENGRAM_TABLE=resident`).
- bfp8 needs INDEX_K_FORMAT=bfp8 and INDEX_K_SERVED=fp8_row192. Any other value of either: FALLS BACK to the bf16
  copy with one log line (`compressor.index_k_copy`), so INDEX_K_FORMAT bf16 / bfp4 and INDEX_K_SERVED bf16 keep working.
- Change (tt/compressor.py `TtServedRows`): no bf16 slab; `k8` = [1, 1, rows / r, 128] bfloat8_b TILE, DRAM
  interleaved, zeroed at load (served.zero_fill, capped), same local rows (block-cyclic order) as the slab was.
  Per chunk (`_put8`): the server-order rows tiled -> start % S == 0: ttnn.bringup.bfp_round in place at tile row
  (start / S) W / r (same on every chip), then that slice of k8 -> typecast bf16 (exact) -> stock per_token_cast_to_fp8
  -> pack_fp8_rows -> served + mirrors (`fp8_rows_tile`). Other starts (follow-ups, pulled-back last chunk; the c_start
  row's rows sit elsewhere, 16 rows off a tile for ratio 2 at 32 mod 64): `_region` (DS41-215): W blocks c, c + 1 of
  k8 read back (exact), update_padded_kv_cache into that ROW_MAJOR region with start / end shifted by c S, bfp_round
  back in place (old rows re-round to themselves); served rows from bfp_round of the chunk's rows.
  tt/indexer.py `_keys`: returns cache.k8 as it is (kept, never freed); no per-chunk slice / tilize / round of the
  history. Rounding is the same rule on the same bf16 values as P.40's bfp8 arm, so the scored keys are the same.
- Difference to the bf16 path: an aligned write with a partial end zeroes the copy's rows past ceil32(end) of that
  chunk (bfp_round writes the whole W / r rows; the bf16 slab's update left them). Never read: the causal mask hides
  them and the next chunk rewrites from its start.
- Test change (owner-authorized): test_kv_layout's recipe and stock-cast reference input = the golden keys rounded to
  bfloat8_b (ckv_bfp4_ttnn/tests/reference.round_values, mbits 7) when k8 is in use. Nothing else changed.
- New test `tests/test_index_k_copy.py` (8x4, 22 s): ratio 2 and 1, writes at 0 (partial end), 0, S, S + 1312 (32 mod
  64, partial end), 2S + 2944, 3S + 1312 (last block): k8 == the bf16 path's copy of host-rounded keys and served /
  mirror rows byte-equal, every chip, every write. Plus the fallback rule.
- Cores (rule 18): bfp_round max_grid (110); slice / typecast / tilize / untilize (cores.to_rm) on the capped
  sub_core_grids; update_padded_kv_cache max_grid; cast 10 / 20 cores (asserted); pack unchanged.
- OV-23: bfp_round prepared (non-compact: row0 changes per chunk); pack_fp8_rows prepared. OV-25: no fork suite run.
- Memory, index-key copies a chip a slot (3 ratio-2 sources + 1 ratio-1; bf16 256 B, bfp8 136 B a 128-value row;
  before = the bf16 copy + the indexers' persistent bfp8 rounded buffers of layers 2 / 8 / 14 / 20):
  56,320 tokens 6.90 -> 2.39 MB; 1,044,480 tokens 127.95 -> 44.39 MB. Also gone: the per-chunk transient tilized copy
  of the whole history (1M, layer 20: 33 MB RM + 33 MB TILE). Unchanged: served fp8 rows + mirrors, ring key buffers.
- Re-run: `scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/test_index_k_copy.py`;
  `python -m models.demos.common.bringup gate P.41 --spec <spec> --check --unit P.41#test_kv_layout`;
  `python -m models.demos.common.bringup.orchestrator measure --task P.41 --spec <spec>`.
- Gotcha (fixed in this step): test_adapter_acks first failed "Input Tensor is not allocated" at a follow-up (chunk
  [2496, 6072), max_seq two chunks): the region read was a full-range ttnn.slice of k8, a wrapper over k8's buffer, and
  was freed (DS41-52). `_k8_rows` / `_k8_free` compare buffer addresses; the new test covers capacities of 1 and 2 chunks.
- Measured, change on (`orchestrator measure --task P.41`, ab/P.41/new) vs the off path's numbers = P.40's change-on
  run (ab/P.40/new: INDEX_K_FORMAT=bfp8 with the bf16 copy, the same code the off side runs; the orchestrator measures
  the real off side). Ladder rung last: gate unit P.41#test_ladder (P.40 bfp8 arm: p40ab/bfp8/ladder).

  | metric (limit) | off: bf16 copy | on: bfp8 copy |
  |---|---|---|
  | index-key copies a chip, 56,320 / 1,044,480 tokens | 6.90 / 127.95 MB | 2.39 / 44.39 MB |
  | err_state_index_k L02 / L08 / L14 / L20 (.236 .260 .292 .276) | .0525 / .0607 / .0981 / .0853 | .0525 / .0609 / .0981 / .0854 |
  | err_state_topk_idx L02 / L08 / L14 / L20 | .0798 / .0698 / .0674 / .0871 | identical |
  | err_state_topk_idx L24 / L28 / L32 / L36 | .1335 / .0825 / .0929 / .0859 | identical |
  | overlap_state_topk_idx L02..L36 | .920 .930 .933 .913 .867 .917 .907 .914 | identical |
  | err_final_hidden (<= 0.242) | 0.1313 | 0.1313 |
  | top5_overlap (>= 0.994) | 1.000 | 1.000 |
  | device_ms_indexer (p51200) | 7.04 | 6.83 |
  | device_ms_compressor (p51200) | 1.95 | 1.98 |
  | device_ms_total (p51200) | 183.56 | 183.80 |
  | chunk_wall_ms (p51200; spread) | 206.5 (10.2) | 204.7 (30.3) |
  | chunk_wall_ms (quick, last chunk) | 181.4 | 180.7 |

  Reading: the scored keys are the same values (top-k identical to the last digit); the served rows now come from
  the bfp8 keys, a rounding before the e4m3 one (index_k error +0.0002 at most, limits ~0.25). Time at 51,200 (cache
  capacity 56,320): indexer -0.21 ms (no whole-capacity tilize + round per chunk), compressor +0.03 (the slice and
  typecast before the cast); totals and wall within run noise. The removed per-chunk work scales with the capacity,
  so the saving should grow toward 1M tokens (not measured: the orchestrator profiles at 51,200).
- Gate units passed here on the final code: test_kv_layout (0 failures; encoding byte-exact against the stock cast of
  the bfp8 keys), test_adapter_acks (0 failures), test_contract, test_ladder. Left to the orchestrator: test_profile,
  test_short_turn.

## P.42 perf (8x4, 2026-11-07): re-index aliases + 128-row window snapshot at cap (owner-authorized contract change)
- Settings (tt/settings.py): `REINDEX_SERVE` alias (default) / copy (P.39); `WINDOW_KV_KEEP` gains snapshot128 (default;
  full / ring / none kept); `KV_BLOCK_SIZE` 64 (the server's kv_block_size; cap = (end - 1) // it x it). A/B: on
  `REINDEX_SERVE=alias WINDOW_KV_KEEP=snapshot128`, off `copy full` (both ENGRAM_TABLE=resident).
- Aliases: tt-d-gen 48b79127fb1 has no address-uniqueness check (make_kv_chunk_table.cpp:1060 refuses duplicate
  coordinates only; kv_chunk_table.cpp validConfigs, migration_strategy_builder.cpp, kv_chunk_index_builder.cpp:191,
  tableWithinDramGeometry: none), nor has tt-metal's KvChunkAddressTable. kv_contract: config `alloc` = layer 20 only,
  alias entries = layer 20's (noc_addr, size, device group) at the same (slot, position); model.mirrors() -> [].
- Snapshot: TtWindowCache(keep snapshot128) = P.32's 192-row ring for attention + `snap` (CacheSlab [.., 1, 128, 512],
  ND-sharded 32-row records, replicated) + `pin` (the 128 rows before max_seq - chunk: the server's pulled-back last
  chunk needs that carry, outside the ring; found by test_runner_contract's engine case). TtDS41Model.layer ->
  `_snapshot` -> `write_snapshot(end)` after the layer's window write, before its ack, on every chunk not marked
  last=False (tt-d-gen's chunk header has no last flag). Rows: `_snapshot_rows` (ring slices; owned blocks: stock
  all_gather TP then SP of the ring itself into load-time buffers, WindowTables.snapshot_gather_buf; pieces sliced
  straight into p % 128 order), written by update_padded_kv_cache(kv_actual_global=0) (DS41-217). Not written for
  DECODER_SKIP=replay128's tail layers.
- Table: 0_window_kv tensor_type Snapshot, records at 0/32/64/96 only, device group = whole mesh. Declared
  max_sequence_length = max_seq, NOT 128: tt-metal's table TT_FATALs a lookup at >= it (kv_chunk_address_table.cpp:138)
  and the framework's test_contract reads every config at every chunk's first / last block; tt-d-gen's conformSnapshot
  reads it as a 128 extent (SnapshotOfExtent), what decode declares. One-line flip: kv_contract address_table
  `_config(L, seq_len, ..)` -> `c["seq"] or seq_len` once testing/contract.py skips Snapshot configs.
- FRAMEWORK CONFLICT (not fixable in my paths): test_contract FAIL `acks_early` 40 (of 800 blocks; limit 0): it
  re-reads every config's first block after the request; the snapshot's record 0 is rewritten by the next chunk by
  design (one per layer). Its other checks pass (producer snapshot PCC 0.987-0.999 over 96 golden rows a layer).
- DS41-216: snapshot replicas differed on whole TP columns / mesh rows at the gather ring's antipode with the model's
  high_bw gathers (2 of 2 acks runs), a stock whole-mesh gather (3 of 3, also with topology Linear); stock TP-then-SP
  gathers 3 of 3 clean (the shipped path). Golden checks (chip (0, 0)'s copy) passed in every run.
- Measured, change on (`orchestrator measure --task P.42`, ab/P.42/new) vs P.41's change-on run (ab/P.41/new; the
  orchestrator measures the real off side, full + copy): every err_layer_L00..L39 identical to the last digit (L39
  0.7067, limit 1.533), host_transfers_per_layer 0. p51200: chunk_wall_ms 210.0 (spread 27.1) vs 204.7 (30.3);
  device_ms_total 192.17 vs 183.80 (+8.4: the snapshot writes, in the ffn_residual section 13.60 vs 5.07, ~0.21 ms a
  layer, mostly the two gathers); write_snapshot host 22.7 ms a chunk (40 layers). Quick last-chunk wall 186.8 vs 180.7.
- Memory a chip a slot (8x4): window KV full 68.75 MiB at 56,320 / 1281.25 MiB at 1,048,576 -> 20 MiB at any length
  (40 x (192 ring + 192 pin + 128 snapshot) KiB), + one 6.75 MiB gather buffer pair a model; re-index copies 5.16 /
  96.09 MiB -> 0. Total before 73.91 / 1377.34 MiB, after 20 MiB.
- Gate units on the final code: test_kv_layout PASS, test_adapter_acks PASS, test_runner_contract PASS (tt-d-gen
  engine 48b79127fb1, pulled-back chunk), test_contract FAIL acks_early 40 (the framework conflict above). Left to the
  orchestrator: test_ladder, test_profile, test_short_turn.
- Tests: new tests/test_window_snapshot.py (8x4, exact on 32 chips: ends 100, 5120, 6120, 10310 (start 32 mod 64),
  15337, the pulled-back chunk via the pin, 40); caches.snapshot_failures / table_rules (Snapshot type, no record
  past 128, whole-mesh group, alias == source entry, alias records out of the overlap check); acks: snapshot vs golden
  [cap - 128, cap) at every ack and at the end + a 100-token request; runner: final and turn-1 snapshots.
  Per-position-only rules (reused-prefix bytes, last block, bytecmp, pulled-back rewrite of layer 0) skip the snapshot.
- Re-run: `scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_window_snapshot.py`;
  `python -m models.demos.common.bringup gate P.42 --spec <spec> --check --unit P.42#test_adapter_acks`;
  `python -m models.demos.common.bringup.orchestrator measure --task P.42 --spec <spec>`.

## P.42 perf attempt 2 (8x4, 2026-11-07)
- Attempt 1 was stopped because approvals.yaml / tasks.yaml changed. Those edits are the owner's P.43 queue entry
  (10:30, during attempt 1), not the agent's. Left as they are, not reverted.
- No code change in attempt 2. The gate status on the same code: test_kv_layout, test_adapter_acks and
  test_runner_contract kept PASS; test_contract FAIL. `orchestrator measure --task P.42` reused quick + p51200
  (rc 0, every err_layer within its limit, L39 0.7067 vs 1.533; p51200 chunk_wall_ms 210.0, device_ms_total 192.17).
- FAIL contract (framework, DS41-218): `acks_early` 40 (limit 0) of 800 blocks; contract_checks_failed 1. tt-d-gen
  ships Snapshot configs only with the last chunk (prefill_reader.cpp:235, ALL tensors only when is_last_chunk), so
  the bytes it reads are final. The fix belongs in testing/contract.py, which is outside this step's paths.
- Open serving question (DS41-219): at 48b79127fb1 a PD prefill's last actual_end is prompt_len - 1. Our cap and
  decode's differ when prompt_len % 64 == 1. Decode has to confirm which length its cap uses.

## P.43 perf (8x4, 2026-11-07): SKIP_TAIL_START=cap (last chunk / replay128 tail from decode's cap) + Snapshot configs in the framework ack check
- Setting `SKIP_TAIL_START` cap (default) | end (P.42). A/B: on `DECODER_SKIP=replay128 WINDOW_KV_KEEP=snapshot128
  SKIP_TAIL_START=cap ENGRAM_TABLE=resident`, off the same with `end`. DECODER_SKIP=off and chunks not marked last: no change.
- Model (tt/model.py): `last_start(end, request_start)` = the latest 32-aligned start of a request's last chunk
  (cap: exact min(end - 2688, cap - 2668), replay128 min(end - 128, cap - 128); end: exact end - 2688, replay128 no
  limit; clamped at the request start / 0). `tail_start` (replay128 tail: cap: min(ceil32(end - 128), cap - 128)).
  `_check_last_start` raises ValueError at the first decoder layer of a last chunk past last_start (exact only when a
  decoder carry is masked; replay128 unless the chunk is the request's own start). Under cap every layer's snapshot is
  written on the last chunk; replay128's decoder layers from the tail state's window ring
  (`TtWindowCache.write_snapshot(end, src=)`, the main cache's gather buffers and slab; the high_bw gathers kept).
- Caller rule (tests): the chunk before the last ends at last_start(end) (no overlap, so the encoder's carry is in
  the 192-row ring; a pulled-back overlapping start would truncate every layer's window under snapshot128).
- Framework (testing/contract.py): `AckReads`: history configs unchanged (each chunk's first / last block); Snapshot
  configs (`is_snapshot_config`) only at the last chunk's acks, every record of [0, max_sequence_length). Unit test
  tests/test_contract_acks.py (fake table, host only). kv_contract declares 0_window_kv max_sequence_length 128
  (`c["seq"] or seq_len`); caches.table_rules expects 128 for the snapshot.
- Test (tests/test_window_snapshot.py::test_skip_snapshot): every layer, s10240 tokens, ends 5184 / 5150 / 100. The
  limit for the step-dump layers comes from the official per-position window KV on the same rows (official_kv,
  DS41-193): with the ring drift, L20 rows [0, 64) (a plain no-skip chunk) failed 0.182 vs 0.144, the official itself
  is 0.180 there. Results: exact every layer <= 0.76 of limit; replay128 L00-L20 <= 0.78, L21-L39 diag 0.97-2.2x
  (DS41-221); refusal of last_start + 32 raises in both modes.
- Last-chunk start (8x4, chunk 5120), before (end) -> after (cap): replay128 3000: 0 -> 0 (tail 2880 -> 2816); 5150:
  5120 -> 4992 (tail 5024 -> 4992); 5184: 5120 -> 4992 (tail 5056 -> 4992). exact 3000: 288 -> 256; 5150: 2432 ->
  2432; 5184: 2496 -> 2432.
- Gate units passed on this code: test_kv_layout, test_window_snapshot (replay128, exact), test_contract (acks_early
  0, contract_checks_failed 0, 800 blocks), test_adapter_acks, test_runner_contract.
- Re-run: `DS41_DECODER_SKIP=exact DS41_WINDOW_KV_KEEP=snapshot128 DS41_ENGRAM_TABLE=resident scripts/run_safe_pytest.sh
  --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_window_snapshot.py` (~5 min incl. load);
  `scripts/run_safe_pytest.sh models/demos/common/bringup/tests/test_contract_acks.py` (host only).
- test_short_turn (replay128, cap): tail [51520, 51712) (192 rows; was [51584, 51712)), final hidden rel at the last
  token 0.108 (limit 0.242; DS41-205 had 0.199 with the 128-row tail), top-5 / top-1 hit; short_turn_ms 170.9,
  short_turn_device_ms 147.29.
- `orchestrator measure --task P.43` (change on; the harness never marks a chunk last, so the skip does not act):
  quick L39 0.7067 (limit 1.533, identical to P.42), host_transfers_per_layer 0, wall 186.9; p51200 chunk_wall_ms
  204.4 (spread 39.2), device_ms_total 189.02 (P.42's on run: 206.5 / 188.74).

## P.35 perf (8x4, 2026-11-07): NORM_IN_MHC, attn_norm / ffn_norm inside the collapse pass
- What: setting NORM_IN_MHC (on / off; on by default). Fork option `ttnn.bringup.mhc_post(..., collapse_pre=,
  norm_weight=w, norm_eps=)`: the y that POST_COLLAPSE writes for the next hc step comes out normalized,
  bf16((y * rsqrt(sum y^2 / C + eps)) * w), y = the bf16 collapse.
  - tt/mhc.py: `TtHcMixes.norm` (set by the block) and `norm_kwargs()` add norm_weight / norm_eps in collapse_kwargs.
    The weight is the norm's own replicated [1, 1, 1, H] fp32 tile.
  - tt/model.py: the block sets `attn_hc.norm = attn_norm` and `ffn_hc.norm = ffn_norm` when both of these hold:
    `model.norm_in_mhc` (POST_COLLAPSE's conditions + the setting), and a whole-row norm (not split) with a bf16
    output.
  - attn_hc / ffn_hc mark `ctx.extra["attn_normed"/"ffn_normed"]` when the y came from the residual's pass. The norm
    step then hands its input on. Layer 0's attn_hc (mhc_pre_xing) keeps its rms_norm: 1 program a chunk.
- Kernel design and the measured cost are in the fork CHANGELOG ("norm_weight"). In short:
  - y^2 is summed exactly into a spare column (25) of the partial rows.
  - The rstd is computed in the root's sum window.
  - The root's reader sends the 32 values per row straight into the owners' L1. With a spare core per row, the roots
    sit on cores with no units.
  - The norm runs on the SFPU after the hc.
- Traced, 8x4, [160, 5120] a chip: mhc_post 134.4 -> 139.4 us a call. The rms_norm it replaces is 18.7 us, so about
  -13.7 us device a call: -1.1 ms device a chunk (80 calls) and 80 programs fewer. Each removed rms_norm also had
  ~45-60 us of host a call (P.43 profile: attn_norm 2.4 + ffn_norm 4.7 ms host a chunk).
- First design: +27 us a call. A device-zone timeline (zones added only while probing, then removed) traced the
  cost to the BRISC word-copy transposes (DS41-222) and to the rstd waiting behind the root's Sinkhorn and
  transposes.
- Precision: not bit-identical to off. The rms_norm fork's statistic is FPU-grade (README Limits); this one is exact
  fp32. Fork test: y_n is within one bf16 step of float64 everywhere, and 1.0-1.3e-5 of the elements are one step off.
  X' and hc are bit-identical to POST_COLLAPSE without the norm.
- Gotcha: a spare-core root (no units) kept hw_startup's bf16 formats; it gave NaN hc until the reconfig (DS41-223).
  The hc rows after the norm phase need srcA / the packer back on fp32.
- Not done: the bfp8 twin for q_a (+0.55 ms). The rms_norm fork's twin is a separate pack path; the y_n tile is in
  DEST, so a precise bfp8 pack of the same DEST value could follow in a later pick. The collapse program's own
  transposes (3 x 5.6-8.2 us on its critical path) are another candidate pick: about -1 ms device a chunk, and they
  are left unchanged so off stays today's program.
- Boxes: [1280, 5120] (2x2 own rows, 8 units a core, no spare cores: roots stay on unit cores) and [640, 1280] pass
  in the fork test. The scratch shape is unchanged.
- `orchestrator measure --task P.35` (switch on; final code): rc 0.
  - quick: every one of the 205 err limits held; worst ratio 0.746 (err_state_window_kv_L25 0.147 / 0.197).
    L00-L38 within ~1 % of P.43's, L39 0.911 (limit 1.533; P.43 0.707, the chaotic last layer).
    host_transfers_per_layer 0. chunk_wall 186.7 ms.
  - p51200: wall 203.2 ms. Sections: attn_norm 0.70 -> 0.017 ms (layer 0 only), ffn_norm 0.70 -> 0, attn_residual
    5.31 -> 5.52 (P.43 profile at rung last; mhc_post is position-independent). 53280 programs (-2528: 79 launches a
    chip).
  - p512000: wall 247.0 ms (spread 52.6). The off side and the gate are the orchestrator's.
- Re-run: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/mhc_post_ttnn/tests/unit/test_mhc_post_norm.py`;
  `python -m models.demos.common.bringup.orchestrator measure --spec <spec> --task P.35`.
- Attempt 2 (2026-11-07): no code change. Attempt 1 failed only on the command-log check: a python3 heredoc doing
  host arithmetic counts as device code. The code is attempt 1's. pre-commit passed on every changed file and
  rewrote nothing. `orchestrator measure --task P.35` reused quick / p51200 / p512000 (code 99663929581d1fa6,
  measured 22:55-23:02 after the last kernel edit at 22:54): rc 0, with the numbers above.

## P.36 perf (8x4, 2026-11-08): FFN_FEED, the MoE head's x written by attn_residual's mhc_post over the fabric
- What: setting FFN_FEED (on / off, default on) + FFN_FEED_LINKS (2). Fork option `ttnn.bringup.mhc_post(...,
  norm_weight=, feed_output=X, feed_semaphores=[ready, arrival], feed_num_links=, feed_cluster_axis=1)`: the pass that
  writes ffn_norm's output (POST_COLLAPSE + NORM_IN_MHC) also writes the MoE head's column-split input X
  [S/SP, H/TP] on every TP chip, as all_to_all_async_generic(in_dim 2, out_dim 3) did (DS41-138). Fork CHANGELOG
  "feed_output" has the kernel design.
  - tt/ffn_feed.py: the two global semaphores, made once a mesh at model load (after ccl.setup, before the MoE blocks
    place their persistent L1), over the capped worker grid.
  - tt/moe_block_main_adapter.py: `can_feed()` (OWN_ROWS path, no token loops / splits / padding; reset in at_chunk),
    `feed_buffer()` (persistent bf16 [S/SP, H/TP] per (mesh, chunk), pinned, allocated at load: DS41-146),
    `__call__(fed=True)` -> `_own_rows` skips the all-to-all and reads the buffer (not freed).
  - tt/mhc.py: `TtHcMixes.feed` (callable) / `feed_kwargs()` / `fed`; tt/model.py: the block sets `ffn_hc.feed` when
    model.ffn_feed (norm_in_mhc, MOE_IMPL=block, TP >= 2), ffn_norm's hc has the norm and the MoE layer can_feed;
    ffn_hc marks ctx.extra["ffn_fed"], experts passes fed=True.
  - Fallback (no refusal): the carried block (MOE_BLOCK_SRC=carried, no can_feed) and the main block's own forward
    run the TP all-to-all as before.
- Bit-identical: quick (rung last, every layer vs golden): all 205 err metrics equal to P.35's to the last digit,
  pcc_chunk_out 0.6054857891895873 both, host_transfers_per_layer 0. Fork test: X bit-identical to the stock
  all-to-all of y.
- Measured (`orchestrator measure --task P.36`, change on, final code; vs P.35's change-on run = this task's off path,
  unpaired): p51200 wall 203.2 -> 203.1 ms, device 187.16 -> 186.92 (attn_residual 5.52 -> 7.06, experts 107.13 ->
  105.28); p512000 wall 247.0 -> 246.1, device 225.30 -> 223.95 (attn_residual 5.51 -> 7.64, experts 106.67 ->
  103.36). Launches 1744 -> 1704 a chunk (-40 all_to_all_async_generic), programs 53280 -> 52000 (-40 a chip).
  The skew wait moved into attn_residual (the brief's risk): the op ends when every TP peer's block has landed.
  The paired off / on table is the orchestrator's.
- Re-measured on main (op alone, 8x4 torus, [160, 5120], 2 links, 50-100 calls; probe `/tmp/p36_timing_probe.py`,
  not kept): host enqueue mhc_post 101.8 us, + feed 109.6 us, the all-to-all it replaces 98.4 us; synced loop 146.7 /
  192.1 / 98.0 us. So about -90 us host and -50 us device a call when the chips are in step.
- Gotchas:
  - The model's mesh is FABRIC_2D_TORUS_XY (TP axis a 4-ring): ports grouped by position hung at the ready fence;
    group targets by get_eth_forwarding_direction (DS41-224). The fork test runs both fabrics.
  - One program per chip (create_descriptor with the coordinate) cost 336 us host a call; the uniform program reads
    its chip from the routing table and the rest from a common-arg table: 110 us (DS41-225).
  - tt-probe saves probes under tests/ttnn/unit_tests/operations/<name>/ (outside this task's paths): deleted.
- Rule 18: the op's cores = units + spare-core roots + 2 L ports inside the 11 x 10 cap (8x4 [160, 5120]: 100 + 5 + 4
  = 109). Rule 4: FABRIC_2D / FABRIC_2D_TORUS_XY only. OV-23: a C++ descriptor op (addresses patched on cache hits),
  110 us host a call.
- Re-run: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/mhc_post_ttnn/tests/unit/test_mhc_post_feed.py`
  (24 passed); `python -m models.demos.common.bringup.orchestrator measure --spec <spec> --task P.36`.
- Attempt 2 (2026-11-08): no code change. Attempt 1 failed on two checks only: a python3 heredoc that edited
  settings.py was read as device code, and the DS41-225 bullet was 602 chars (now cut to fit by rewording one
  phrase). pre-commit on every changed file: passes (the only failure is check-large-files on the orchestrator's
  dashboard/*.html, which are not this step's files). `orchestrator measure --task P.36` reused quick / p51200 /
  p512000 (same code and switch): rc 0, with attempt 1's numbers above (quick chunk_wall 184.7 ms, p51200 203.1,
  p512000 246.1).

## P.44 (perf, attempt 1, 2026-11-10): served index rows encoded from the chunk's bf16 keys (INDEX_K_SERVED_FROM)

- Expected saving (written before measuring): per index-key write with start % chunk == 0, the k8 slice and the
  bfp8 -> bf16 typecast go (2 launches); at other starts the separate bfp_round to a bfp8 temp and the typecast go
  (2 launches). 4 index-key sources a chunk (layers 2, 8, 14, 20) -> 8 launches fewer: device ~-0.05 to -0.2 ms
  (640 / 320 rows x 128 a chip each, tiny ops), host ~-0.5 to -1 ms a chunk (~60-100 us a launch on 32 chips). The
  main point is correctness of the served bytes (fp8(key) as Blaze's writer), not speed. No persistent memory
  change; the per-chunk transient peak is lower (the tiled bf16 rows live through the encode instead of a k8 slice +
  a bf16 typecast copy).
- Change: new Setting INDEX_K_SERVED_FROM (bf16 default | bfp8 = P.41). `_put8` with bf16: start % chunk == 0 rounds
  the tiled rows into k8 in place as before and encodes the served / mirror rows from the same tiled bf16 rows
  (fp8_rows_tile), then frees the tile; any other start runs `_region` (unchanged) and encodes from the tile; the
  separate bfp_round to a bfp8 temp, the k8 slice and the bfp8 -> bf16 typecast are skipped. bfp8 keeps P.41's
  path byte for byte. INDEX_K_COPY=bf16 is untouched (its served rows were already from bf16). No new op, no new
  core grid (fp8_rows_tile / bfp_round as before: rule 18 unchanged; bfp_round prepared=True as before, OV-23).
- Tests (owner-authorized): test_kv_layout's served-row reference input is the unrounded golden keys under bf16
  (rounded under bfp8); test_index_k_copy adds a third cache (INDEX_K_COPY=bf16, unrounded keys) whose served /
  mirror bytes B's must equal under bf16; the k8 == rounded-copy assertions unchanged.
- Results (change on): gate units test_index_k_copy PASS (served + mirror byte-exact with the bf16-key encode on 32
  chips, all 6 write patterns incl. 1312 / 2944 follow-ups), test_kv_layout PASS (served bytes == stock
  per_token_cast_to_fp8 of the golden keys, every byte, all 4_index_k_r1 / 2_index_k_r2 layers).
  `orchestrator measure --task P.44` rc 0: quick chunk_wall 181.1 ms (spread 4.9); p51200 wall 205.9 (spread 22.1),
  device_ms_total 185.77, device_ms_indexer 6.88. err_state_index_k L02/08/14/20 0.0526 / 0.0607 / 0.1015 / 0.0868
  (limits 0.236 / 0.260 / 0.292 / 0.276); topk overlap 0.866-0.933. The paired off/on table is the orchestrator's.
- Host-only: golden index keys of chunk 2 (s10240 rung) of layers 2/8/14/20, CPU writer recipe (caches.blaze_rows):
  fp8(key) vs fp8(bfp8(key)) differ in 18.9 % of e4m3 values, 49.6 % of scales, 100 % of rows (DS41-226).
- Memory (per chip, k8 = the only index-key copy, 136 B a row): 56,320 tok 2.28 MiB, 1,044,480 tok 42.33 MiB, before
  = after (no persistent buffer added or removed; the per-chunk transient is smaller).
- Gotcha: I edited settings.py / compressor.py / tests with `python - <<EOF` text-replace heredocs and ran a host-only
  CPU script (/tmp/p44/share.py: golden keys + blaze_rows, imports ttnn, opens no device). Neither touches a device;
  noted because P.36 attempt 1 was flagged for a python heredoc editing settings.py.
- Re-run: `gate P.44 --spec <spec> --check --unit 'P.44#test_index_k_copy'` / `'P.44#test_kv_layout'`;
  `python -m models.demos.common.bringup.orchestrator measure --spec <spec> --task P.44`.

## P.50 (perf, attempt 1): COMBINE_SPLIT ported from FIX.23 onto the pre-replication base

- Expected per chunk (written before the change; perf_guide: this is a correctness change that adds work, no saving):
  device +~1 ms (routed_experts_combine +0.95 ms a chunk on the replica branch: one extra SFPU pass and a second fp32
  sum a token, ~24 us a layer x 40), host 0 (same launches, same program count; the switch is a compile arg). Wall:
  within noise (FIX.23: 183.2 -> 183.5 ms clean wall, spreads 15.6 / 6.5).
- Ported (DS41-235, owner: the fix only, no redesign): `git diff 4c79ce65ece 4e90f611e86` hunks for the combine fork
  (common/moe_block_config.{hpp,cpp}, common_nanobind.cpp, routed_experts_combine/device/{combine_host,combine_program}
  .{hpp,cpp}, kernels/compute/accumulator_compute.cpp, tests/routed_experts_combine/test_experts_combine_checks.py:
  applied cleanly, offsets only), README / CHANGELOG / INDEX lines, setting COMBINE_SPLIT (default off, owner) and
  its plumbing in tt/moe_block_main_adapter.py, tests/test_determinism.py, testing/b2b.py (ids() only),
  DS41-233 / 234 / 235 (+ detail). Not ported: replicas, slot maps, expert_slot_copy, tile_gather, probe_determinism.
- One addition of mine: the adapter's block cache key (`_BLOCKS`) now includes the split flag (DS41-185: settings are
  frozen at load; a refresh to the other value in one process would have reused the other side's program).
- The test also records `determinism_max_abs_diff_<COMBINE_SPLIT>` (off / on). The gate's command pins
  DS41_COMBINE_SPLIT=on, so the orchestrator's A/B cannot run the off side of this test; I ran it by hand.
- Determinism (40 layers, 4 chunks of 5120 from 0, 3 sequences a prompt, novel / code interleaved, run 0 = cold):
  off (DS41_COMBINE_SPLIT=off, results /scratch/dnijemcevic/bringup/p50_det_off/P.50.json): FAILS,
  determinism_max_abs_diff_off = 0.2542; novel cold vs warm chunk 0: 0.2417 (255 rows), chunks 1-3 0;
  code run 1 [0.0078, 0.154, 0.254, 0.219] (31 / 390 / 1362 / 271 rows), run 2 [0.225, 0.0195, 0.227, 0] rows
  [392, 132, 1050, 0]. on (gate unit P.50#test_determinism): PASS, determinism_max_abs_diff_on = 0, every chunk.
- Re-run: `PYTHONPATH=$PWD DS41_COMBINE_SPLIT=off|on scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_determinism.py` (~5 min); A/B: `python -m
  models.demos.common.bringup.orchestrator measure --task P.50`.
- `orchestrator measure --task P.50` (quick A/B, on side only so far; ab/P.50/new/side.json): quick rung rc 0, every
  one of 102 limits held (largest share err_state_window_kv_L25 0.746), host_transfers_per_layer 0, clean chunk wall
  184.3 ms (spread 23.0); at 51200: chunk wall 206.2 ms (spread 46.8), device total 185.85 ms, experts 104.35 ms.
  The off side and the combine op's own time: the orchestrator's A/B (I did not profile, per the brief).

## P.49 (perf, attempt 1): PREFILL_END=cap, prefill stops at decode's cap; the cap rule equals decode's (DS41-219)

- Expected per chunk (written before measuring; perf_guide section 1: a contract change, not a speed pick): a
  non-last chunk ends on a 64 block (cold starts, chunk 5120), so its cap is its end: device 0, host 0 (no op added,
  removed or resized; the same programs). The A/B (quick rung, 51,200) times full non-last chunks: expected flat, within
  the noise. The request's last chunk: the model still runs the chunk at its fixed padded size (rows past the end are
  pad, masked), so device about 0 there too; only a last chunk wholly past cap (prompt ends < 64 tokens into a chunk)
  skips every layer's compute (saves one whole chunk, ~185 ms device) and only rewrites the 40 snapshots and acks.
- Cap rule (DS41-219, tt-d-gen 48b79127fb1): PD prefill gets actual_end = prompt_compute_end = max(prompt_len - 1, 1)
  (engine/include/engine/runtime/types.hpp:101-110; the last chunk ends there, engine/src/runtime/prefill_writer.cpp:
  125-131); decode asks for [0, prompt_len - 1) (backend_runtime.cpp:817) and its cap is reusable_prefix_cap(64,
  prompt_len) = (prompt_len - 1) // 64 x 64 (engine/include/engine/control/prefix_indexer.hpp:78-81). So cap =
  actual_end // 64 x 64 (tt/attention.py snapshot_cap; was (actual_end - 1) // 64 x 64). Same for every length:
  server_rules.self_check checks lengths 1..258 at blocks 32 and 64. Non-last chunks end on a block because a
  recurrent slot always starts cold (backend_runtime.cpp:1063-1064), so their cap is their end.
- Change: setting PREFILL_END (cap default | end). tt/runners/adapter.py _run runs the chunk with the model's end =
  attention.prefill_end(actual_end); positions >= cap become PAD in the host ids (engram hash, id history) and pad rows
  in the model (no window ring, snapshot, compressed KV, index key writes). Acks still name the engine's range. A chunk
  with cap <= start computes nothing: _acks_only rewrites each layer's snapshot at cap, syncs, acks (K-71). Only the
  serving runtime reads the switch; the ladder / profile call model.layer directly, so the A/B rungs are unchanged.
- Tests updated to the new rule (caches.served_end): test_adapter_acks, test_runner_contract (ranges [lo, cap),
  [0, cap)), test_window_snapshot (test_snapshot_cap values), server_rules (prompt_compute_end, decode_cap). New
  tests/test_prefill_end.py (served_bytes_differ, past_cap_written, end_past_written).
- Bit-identity (test_prefill_end, 40 layers, end -> slot 0, cap -> slot 1, read through the table over UMD):
  prompt_len (mod 64, actual_end, cap, chunks): 191 (63, 190, 128, 1), 4097 (1, 4096, 4096, 1), 4098 (2, 4097, 4096,
  1), 5122 (2, 5121, 5120, 2; last chunk computes nothing), 7231 (63, 7230, 7168, 2), 10176 (0, 10175, 10112, 2);
  both prompts: 0 of 24,960 served records differ; the cap run wrote 0 records in [cap, actual_end); the end run wrote
  some there in every case that has such records (proves the two sides differ).
- Gate units: test_prefill_end PASS, test_adapter_acks PASS (the slot-0 last-block check [2496, 2536) is empty
  under cap: printed, not run), test_determinism PASS (max |diff| 0, both prompts, cold and warm).
- test_contract FAILS under PREFILL_END=cap (DS41-236), checks that measure positions the cap path no longer computes
  (request actual_end 10208 = seq 10240 - contract_tail_pad 32; cap 10176):
    FAIL pcc_contract_state_window_kv: 0.8525 (layer 38; every layer 0.853-0.868) < limit 0.9434 (P.43's limit; this
      run's hook raised before recording it): the window ring state at 10208 vs the device ring, which stops at 10176
    FAIL contract_state_exact_engram_tail: layers 1, 14 [5306, 14, 237] != golden [16, 43, 783] (ids before 10208 vs
      before 10176)
  Not failing: producer KV over [0, 10208) compress_kv 0.98722 (limit 0.9370; P.43 end 0.98714), index_k 0.99175
  (0.9465; P.43 0.99230), window_kv snapshot at cap 10176 min 0.98705 (P.43 0.98686). Per the brief I did not change
  the check, did not run measure / the profile, and stopped. The owner decides; proposals:
    (a) run test_contract with DS41_PREFILL_END=end in the gate (the full-prompt path is kept), and let
        test_prefill_end carry its verdict to cap: every served byte below cap is identical between the two;
    (b) or judge the cap path at the handoff point: KV over [0, cap) and the fixed state at cap, which needs a golden
        state snapshot at 10176 (or contract_tail_pad 64 with that snapshot): a golden / spec change.
- Perf: not measured (stopped before measure, as the brief says). Expected: 0 on the A/B chunks (full, non-last).
- Re-run: `gate P.49 --spec <spec> --check --unit 'P.49#test_prefill_end'` (~8 min);
  `DS41_PREFILL_END=end|cap` selects the side.

## FIX.25 (overseer fix, 2026-11-12): determinism in every gate

What changed (framework; the model's code untouched):
- `testing/model_determinism.py` (new, the one mechanism): the loaded device model runs the novel prompt (the spec's
  prompt file) and the code prompt (DS41-234's, `testing/b2b.py`) interleaved, 4 chunks of 5120 each from position 0,
  3 runs each, every sequence on a fresh state. Run 0 of novel is the cold one (the first sequence after the load,
  DS41-233); every later run follows the other prompt (determinism.py's A, B, A). Each later run must equal its
  prompt's run 0 bit for bit: every chunk's final hidden and, after the last chunk, every layer's served state
  (`state.served`: window KV, compressed KV, index keys; a model without it: `state.to_torch`). Compared with
  testing/determinism.py's snapshot / diff (bits: -0 vs +0 and NaN payloads count). On failure the message names the
  prompt, the run, the first differing chunk and the first differing layer state. Metrics: determinism_max_abs_diff
  (gated == 0; a bits-only difference records 2^-149 so it still fails), _novel / _code, _state_, elements_differ,
  per-chunk lists, determinism_s (cost). Coverage in defaults.yaml `tests.model_determinism` (chunks 4, runs 3).
- `tests/test_determinism.py` (framework, new): that check on the spec's model. The model's own
  `tests/test_determinism.py` now calls the same function (keeps P.49 / FIX.25 / FIX.26 gates that name it and P.50's
  `determinism_max_abs_diff_<COMBINE_SPLIT>` metric); no second mechanism.
- `core/gate.py` `with_determinism` / `needs_determinism` / `gate_limits`: every device gate (`device: true`) that runs
  a framework model-level test (ladder, profile, positions, contract, smoke) or is a full-model or fix step, and names
  no determinism test itself, runs the check without its tasks.yaml entry naming it. It is put FIRST into the gate's
  first unit whose runs share one process at the shipped configuration (no switch variable), so with
  tests.one_process it costs no model load and its first sequence is the cold one; without such a unit it is the
  gate's own first unit `<task>#test_determinism`. `gate_limits` adds `determinism_max_abs_diff: == 0` to the gate's
  limits (run_units, run_gate, delta_status, the brief's thresholds, the A/B gate_on); a tasks.yaml limit is never
  replaced. No switch turns it off (owner: no exemption for the model; tests.nondeterministic stays component-only).
- `orchestrator.py` A/B: a `determinism` item on both sides (quick: added when the pick's gate runs the check; full:
  the gate's injected segment), first in the side's process (`core/batch.phase` -1). The accept (gate_on, the reuse of
  the 'on' side) fails without it passing on the 'on' side; the table shows both sides. A 'before' side (ab.before)
  measured earlier stays valid without it.
- `core/batch.py`: test_determinism.py is a shared test; `hints` reads target.chunk only when a run needs a position
  (it raised on a spec without target: 10 test_perf_ab selftests failed at HEAD for that, now pass).
- `testing/b2b.py` `code_text`: the code prompt falls back to the same kind of text built from the repo's TTNN op
  sources when /scratch/.../code.txt is missing (another box).
- Selftest `selftest/test_model_determinism.py` (10 tests): a deterministic fake passes with 0; a deliberately
  non-deterministic fake (`fixture_model.NondeterministicFakeModel`, a term that depends on the call count, like an
  arrival-order sum) fails naming chunk 0 and layer state L02; a -0 vs +0 difference fails; injection rules; and end
  to end, a ladder gate (tasks.yaml names no determinism test) through run_gate passes with the deterministic fake and
  FAILS with the non-deterministic one (FIXTURE_NONDETERMINISTIC=1), the ladder never running after it.
  test_overseer's fix-job test now expects the check once, first. Full selftest suite vs HEAD: no new failure.

Numbers (Galaxy 8x4, 40 layers, COMBINE_SPLIT=on, the shipped defaults):
- Batched probe, exactly as an injected gate unit runs (determinism, then the `last` ladder rung, one process):
  bit-identical (hidden every chunk, every layer's served state, both prompts, cold and warm); process 347 s = load
  191 s + determinism 93.5 s + ladder 52 s (warm). A ladder unit alone took 266-285 s (FIX.23 / P.48), so the check
  adds about 65-75 s per gate; its cold sequence pays the program compiles the ladder then skips. The warm sequences
  are ~1.5 s each: the 4 x 3 x 2 coverage costs ~10 s over the one cold sequence it cannot avoid.
- Coverage choice: 4 chunks reach every chunk DS41-233 / 234 moved with COMBINE_SPLIT=off (P.50: 0.2542, rows
  [392, 132, 1050, 0]); 3 runs give cold vs warm and warm vs warm after the other prompt. Fewer would save seconds
  only; more chunks need a longer state (max chunks = target.seq / chunk).
- A gate without a shared unit (e.g. only contract tests) pays its own load: ~260-300 s (the model test's 260 s).

Gotcha: I ran `python -m pytest` once for the CPU selftests (an argument error, nothing ran); the selftests then ran
through scripts/run_safe_pytest.sh (CPU only, the runner takes the box lock). And one devjob was started with the bare
safe runner (usage error, rc 3, no device access).

Re-run: `gate FIX.25 --spec <spec> --check`; the batched probe: `BRINGUP_BATCH_LM_HEAD=1 BRINGUP_BATCH_CMD='<determinism
cmd> && BRINGUP_RUNG=last <ladder cmd>' scripts/run_safe_pytest.sh --no-precompile --run-all
models/demos/common/bringup/tests/test_batch.py`; selftest: `scripts/run_safe_pytest.sh --run-all --no-precompile
models/demos/common/bringup/selftest/test_model_determinism.py`.

## FIX.26 (overseer, attempt 1): test_contract judges PREFILL_END=cap at cap (2026-11-12)

What changed (DS41-236 -> DS41-238; framework finding F264):
- How the golden snapshots are made: `reference/generate_golden.py` runs the native reference (fp32, keep_fp32) chunk by
  chunk from 0 and saves the fixed state (window ring, compress_partial, engram tail) before every dumped chunk and,
  on the contract rung, after `seq - contract_tail_pad` = 10208 tokens (a deep copy of the state before the last
  chunk, then [5120, 10208)). Nothing at 10176.
- New `models/demos/common/bringup/reference/handoff.py`. `handoff golden`: same reference, same calls: chunk 0 from
  position 0, then a check that the state at 5120 equals the golden's stored state byte for byte (fixed and growing,
  all 40 layers; it did: same reference, same code), then [5120, 10176) and the fixed state saved as
  `kv_cache/handoff_layer_<i>_at_10176.safetensors` (40 files), sha256 in `handoff.json`, tied to the golden's
  content_hash. The handoff files are "derived" (golden.derived: name starts with `handoff`), so the pinned golden
  (content_hash, files.json, frozen manifest) is unchanged. Golden.state(i, at=p) falls back to them (sha-checked).
  The point comes from the model: hooks.contract_handoffs(spec, 10208) = [snapshot_cap(10208)] = [10176].
- `handoff drift`: the limits follow the framework's existing rule (F128 / F134: calib x the official bf16 reference's
  drift on the same comparison): the official reference from 0, last chunk ending at 10176, vs the native golden;
  same 51 contract_state_* keys as drift-s10240.json, written to `handoff-drift-s10240.json` {"10176": ...}. No
  existing limit or drift file was edited. drift.replay(full_last=False), contract_entries(growing="tail") are new
  options (defaults unchanged).
- testing/contract.py: at = hooks.contract_end(spec, actual_end) (DS41: tt/attention.prefill_end, so 10176 under
  PREFILL_END=cap, 10208 under end); producer / independent read-back over [0, at rounded to 32), fixed state at at,
  limits from drift-<rung>.json when at == actual_end, else the handoff drift. New metric contract_state_at.
- tt/runners/kv_contract.golden_window: also takes the handoff snapshots, so a non-step layer's 128-row window
  snapshot at cap is compared on all 128 rows (before: 96; [10048, 10080) was in no golden ring).
- selftest/test_handoff.py (3 CPU tests on the fixture).

Numbers:
- CPU: handoff golden 1727 s (chunk 0 858 s, [5120, 10176) 852 s, 16 threads); handoff drift 2171 s (bf16, 16
  threads, run side by side). Golden cap vs 10208 snapshot: window ring PCC 0.72-0.78 (L1 / L20 / L39), engram tail
  [5306, 14, 237] vs [16, 43, 783] (the device's "wrong" tail in DS41-236 was the right one for cap).
- New limits at cap (end limits in brackets): window_kv 0.94787 (0.94341), compress_kv 0.93695 (0.93703), index_k
  0.94648 (0.94653), compress_partial 1.0.
- test_contract, PREFILL_END=cap (default), before the golden_window change: contract_state_at 10176, ring 0.98793,
  producer window_kv 0.98705 / compress_kv 0.98877 / index_k 0.99334, engram tail exact, checks failed 0, acks_early 0.

Gotchas: Golden.verify() is False on this golden already (F164 trimmed 32 files); the index route (files.json) is the
check, and the handoff files add nothing to it. The framework selftests that build the fixture prompt fail at HEAD
(text source models/tt_transformers/... moved to models/ttt_compat, #62523); test_handoff.py sets text.source.

For the owner: P.49's gate unit can go back to plain test_contract (no DS41_PREFILL_END=end):
`scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_contract.py`.

Re-run: `python -m models.demos.common.bringup.reference.handoff golden --spec <spec>` then `... handoff drift`
(golden skips points it already has), `python -m models.demos.common.bringup.reference.golden sync --spec <spec>`, `gate FIX.26 --check`.
- Full gate `gate FIX.26 --check` (after the golden_window change): PASS, 4 units, 1301 s. test_contract (cap):
  judged at 10176, KV over [0, 10176) window 0.98793 / compress 0.98877 / index 0.99334, every layer's window
  snapshot on 128 rows (min 0.98793), ring min 0.98793, engram tail equal on both engram layers, checks failed 0;
  test_contract.3.0 (end): judged at 10208, ring 0.98725 vs 0.94341, KV 0.98793 / 0.98879 / 0.99336, checks failed 0;
  test_ladder pass; test_determinism max abs diff 0.

## FIX.27 (overseer fix, attempt 1): ledger files split into chunks below the 500 KB repo limit (framework F265)
What: FIX.26's commit was refused because state_metrics.json was 515,706 bytes (over the repo's 500 KB hook).
The three ledger records that grow with every task are now chunk files (core/chunked.py, Ledger stores):
`state.chunks/`, `state_metrics.chunks/`, `history_state.chunks/` in ledgers/galaxy_8x4-full, `0001.json`, `0002.json`,
... Rule: a task's record stays in its chunk; a new task goes to the newest chunk, or opens the next one when that
chunk would pass 256 KB. Nothing is compacted, rounded or dropped (the old per-save re-trim of every task's metrics
now applies only to the entry being written). BREADCRUMBS.md and supervision.md roll at a gate commit once past
256 KB: their older sections move unchanged to `BREADCRUMBS.0001.md`, ...; the live file keeps the newest <= 64 KB.
Migration (once, `Ledger.migrate()`), each chunk-merge equal to the old file as parsed JSON:
- state_metrics.json 515,706 B -> 0001.json 256,402 + 0002.json 259,307
- history_state.json 290,493 B -> 0001.json 255,251 + 0002.json 35,243
- state.json 146,416 B -> 0001.json 146,416
Each chunk dir has MIGRATED.json (old size, sha256, git blob id, a hash per task). Pre-migration copies:
/scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1/FIX.27_premigration/.
Gotcha: the orchestrator now running FIX.27 was started with the old ledger code. state.json is left in place
(byte-identical to the chunk) so that code still sees every status. Its writes after the migration (small
state_metrics.json / history_state.json, a state.json) are merged over the chunks by the new reader, and the first
new-code write folds them in and deletes them. Simulated on a copy: no other task's record changed.
Restart the orchestrator before FIX.26's commit-only so the new code does it.
Files checked (bytes today; growth per task over the last 14 tasks):
state_metrics.json 515,706 (+8.0 KB) chunked; history_state.json 290,493 (+3.0 KB) chunked; state.json 146,416
(+2.4 KB) chunked; BREADCRUMBS.md 384,453 (+4.4 KB) rolls; supervision.md 201,778 (+0.7 KB) rolls; tasks.yaml
245,491 (+4.2 KB) NOT chunked (hand-edited; ~60 tasks of headroom); known_issues.md 84,786 (+0.8 KB) and
known_issues_detail.md 101,558 not chunked (read whole by agents); results/* one file per task (largest 291,685,
X/FIX ops.json.gz); results/knowledge_use one file per attempt (~5 KB); dashboard/*.html 1.5 MB, never committed (F210).
Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.27 --spec <spec> --check`.
Gate: `gate FIX.27 --check` refuses before running anything: "gate calls pytest directly" (core/gate.py
device_policy_errors: no gate command may call pytest directly, CPU gates included). The command is tasks.yaml's,
which an agent may not edit. The framework's own selftest gates use the safe runner (dev/tasks.yaml), and the same
units pass that way: `scripts/run_safe_pytest.sh --no-precompile --run-all models/demos/common/bringup/selftest/test_core.py
models/demos/common/bringup/selftest/test_ledger_chunks.py` -> 65 passed, ledger_files_over_limit = 0 (19 s, no device
opened). Overseer: change FIX.27's gate cmd to that line. Full selftest dir: 182 failed / 576 passed. 164 are the missing
fixture text (tale-of-two-cities, known), and 17 A/B / runner-contract tests fail identically on unchanged HEAD.
The one this change broke (test_seeded: `git add state.json`) is fixed.

## FIX.27 (overseer fix, attempt 2): no code change; the gate command is still the blocker
- `gate FIX.27 --check` again refused before running: "gate calls pytest directly" (core/gate.py:149
  device_policy_errors; tasks.yaml FIX.27 cmd is `python -m pytest ...`, which an agent may not edit). Not loosened:
  the policy is the owner's (selftest test_core.py::test_device_policy pins `python -m pytest` as refused, device or not).
- The ledger after attempt 1: the orchestrator (old code) has since rewritten state.json (87 tasks, FIX.27's record
  differs: its attempt / history, merged field by field over the chunks), state_metrics.json `{}` and
  history_state.json 925 B (FIX.27's agent run). The chunks read 87 state / 79 metrics records.
- Same units via the safe runner: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --no-precompile --run-all -q
  models/demos/common/bringup/selftest/test_core.py models/demos/common/bringup/selftest/test_ledger_chunks.py`
  -> 65 passed (19 s). pre-commit on every changed file passes, except check-large-files on
  dashboard/index.html 1.59 MB and teletext.html 1.58 MB: generated pages, kept out of every gate commit
  (gate.GENERATED, F210), rewritten whole after each gate, not appended to; not chunked.
- Overseer: replace FIX.27's gate cmd with the safe-runner line above, then it passes as is.

## FIX.27 (overseer fix, attempt 3): gate passes with the corrected command
- The overseer changed FIX.27's gate cmd to `scripts/run_safe_pytest.sh --run-all --no-precompile <test_core.py>
  <test_ledger_chunks.py>`. No code change in this attempt.
- `gate FIX.27 --check` -> PASS, ledger_files_over_limit = 0 (65 passed). Chunks: state_metrics 256,402 + 259,307 B,
  history_state 255,251 + 35,243 B, state 146,416 B (1 chunk); live state.json 147,268 B, state_metrics.json 3 B.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.27 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`

## FIX.27 (overseer fix, attempt 4): the gate's commit no longer names a file git has already deleted (OV-38)
- Cause: the WIP commit left the deletions of state.json, state_metrics.json and history_state.json staged (gone from
  disk and index, still in HEAD). gate._commit_files lists staged paths (`git diff --cached`), and git_commit ran
  `git add -A -- <them>` -> "pathspec did not match any files". `git commit -- <them>` itself commits the deletion fine.
- Fix: core/gate.py `_stage` (the one git add path of git_commit): adds only files that are on disk or in the index;
  staged deletions stay staged and go in through the commit's pathspec. core/chunked.py `ChunkedJson.paths()` (the one
  list of ledger files, used by Ledger.record_paths): the old single file only while it is on disk or git still has it
  (index or HEAD); once its deletion is committed the old name is never listed again.
- Selftest test_ledger_chunks.py::test_a_gate_commit_after_the_migration_was_staged_names_no_vanished_file: old files
  committed, migrated, deletions staged, gate commit -> PASS and commits; a whole-ledger commit; then the layout names
  only *.chunks and a second gate commits. test_the_ledger_reads_and_writes_every_chunk updated to the new rule.
- Dry check on the live ledger: commit list stages with `git add --dry-run` rc 0; the three staged deletions are
  committed by pathspec.
- Sizes now: state_metrics.chunks 256,402 + 259,355 B; history_state.chunks 255,251 + 39,767 B; state.chunks 147,652 B;
  BREADCRUMBS.md 64 KB live + chunks 261 KB / 65 KB; supervision.md 202 KB; tasks.yaml 246 KB; overseer.yaml 64 KB.
  results/FIX.5_profile.json is 627 KB but tracked since Oct 27 and written once (not appended): left as is.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.27 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`
