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

## P.51 (perf, attempt 1): DECODER_SKIP=decode (Kartik's cut), per-position encoder replay below global_write_start
- Expected per chunk (written before any code change; perf_guide sections 1-2). A serving chunk under decode runs
  layers 0-19 in full, layer 20's KV projection only (attn_hc .. compressor, P.30's kv_only) and skips layers 21-39.
  P.30 measured that skipped chunk at 83.6 vs 167.9 ms device at 51200; with today's ~189 ms device (P.42 off) I
  expect ~95-100 ms device a non-replay chunk (-~90 ms, about half), host about -45 % (19.x of 40 layers' launches
  gone; ~-90 ms of host dispatch if the chunk stays host-bound). A replay chunk (first chunk of a prefix hit) costs
  the same as a normal one (the replay rows are computed; only their history writes are skipped) plus one row shift
  a KV-source cache write (~2 x 6 small ops, < 1 ms). The framework's A/B (quick rung, p51200) drives the model with
  last unset, as the harness always has: there decode runs every layer, the same path as off, so expected 0 there;
  the skip shows only on chunks the serving runtime marks (and with BRINGUP_CHUNK_LAST=0, which I time separately).
- What changed. Setting `DECODER_SKIP` gains `decode` (default stays `off`; keep_default). Model (tt/model.py):
  decode builds the decoder list as exact does (layer 20 = kv_only, 21-39 skipped on a chunk marked last=False);
  `begin_chunk(.., write_start=g)`: g > start -> replay (every window truncated at start, the `_window_policy` replay
  branch now also for decode), `write_from(start)` = max(start, g); the compressor step skips a chunk wholly below g
  and passes `write_from` to `TtCompressor.write`; `kv_only` returns at once when g >= end; decode writes layers
  0-19's snapshot on every chunk (`_snap`); `last_start` / `_check_last_start` act for exact / replay128 only;
  `use_decoder_skip` takes decode / off on any model loaded with a skip mode. g > start under another mode, or with
  WINDOW_KV_KEEP=full, raises. tt/compressor.py `_write_from`: the chunk's gathered natural-order rows rotated up by
  (g - start) / r (slice + concat in ROW_MAJOR), the moved rows masked to zero (they lie at positions >= start + S >=
  ceil32(end), which update_padded_kv_cache's valid_global never writes anyway), written as a chunk at g through the
  usual any-start path (server order, bf16 / fp8 / bfp8 copies, `_region`); the private attention table is written as
  today and its rows of groups [start / r, g / r) put back (`AttnTable.rows_copy` / `restore`). Adapter
  (tt/runners/adapter.py): `prefill_chunk(.., global_write_start=None)` (None -> actual_start), under decode every
  chunk last=False; acks unchanged (every layer, after a sync).
- Decisions. (a) The skip acts where the caller marks the chunk (the serving runtime always does under decode); the
  framework harness (ladder, determinism, profile) passes last unset and so runs every layer under decode too: those
  gates measure the off path (bit-identical code path), and BRINGUP_CHUNK_LAST=0 times the skipped chunk. (b) The
  window ring (snapshot128's working rows) is written for replayed positions: it is not served, and the next chunk's
  carry / the snapshot need it; the brief's attention.py cache.write lines are ring writes under snapshot128 and the
  served full slab only under WINDOW_KV_KEEP=full, where a replay is refused. (c) Layers 20-39's snapshot records are
  not written under decode (stated in serving_contract.md / migration_configs.md). (d) test_contract and
  test_adapter_acks check layers 20-39's windows / ring state, which decode no longer computes: they run on off
  (DS41-240), as the brief's item (5) says.
- Gotcha: `devjob wait` blocks past a 2-minute tool call; run it as a background task.
- Results (8x4, 40 layers). tests/test_decoder_skip_decode.py (one model load, decode vs off in-process, 3 slots, 6.8
  min): (1) served_bytes_differ 0 of 20,912 records (history [0, P) + layers 0-19's snapshot), N 300 / 4320 / 5248 /
  7320 / 10368 (P 128 / 4160 / 5120 / 7168 / 10240) x novel / code; decoder_snapshot_written 0; acks once per layer
  per chunk, both modes. (2) Prefix hit (Xp 4096, replay chunk [3968, 7168), g 4096): replay_rows_written 0 (both
  prompts); rows [4096, 7168) vs a fresh decode prefill, novel: history min PCC 0.9952 (1_compress_kv_r2 L14), max rel
  0.098; layers 0-19's snapshot min PCC 0.9920 (L18), max rel 0.126; code: worst PCC 0.9733 (snapshot L18), max rel
  0.231. Against the golden (novel; server_rules.state_threshold, official drift for dumped layers' windows): replay
  0 checks over the limits, fresh 0; e.g. 3_compress_kv_r1 L20 PCC replay 0.99295 / fresh 0.99326, 1_compress_kv_r2
  L14 0.98947 / 0.98975, snapshot L14 0.99008 / 0.99148.
- Gate check (default env = off; 1428 s): test_ladder (determinism 0, err_final_hidden 0.1321, top5 1.0, every
  err_layer / err_state within its limit), test_decoder_skip_decode, test_contract (contract_checks_failed 0,
  acks_early 0), test_adapter_acks: all pass; chunk_wall_ms / device_ms_total MISSING (the orchestrator's A/B
  profile supplies them).
- `orchestrator measure --task P.51` (change on = decode; the harness's chunks with last unset, so the off path):
  determinism 0; quick L00 0.0195 ... L38 0.1167, L39 0.9115 (limit 1.533, DS41-188), host_transfers_per_layer 0, wall
  180.9; p51200 chunk_wall_ms 204.9 (spread 4.7), device_ms_total 185.11.
- The skipped chunk (decode, BRINGUP_CHUNK_LAST=0, p51200, one profiled process): device_ms_total 93.38 vs 185.11
  (-91.7 ms, -49.5 %; expected ~95-100), profiled wall 110.8 vs 196.4 (-85.6 ms); experts 53.11 vs 103.06, attention
  23.60 vs 48.09, indexer 1.38 vs 6.80, compressor 2.01 vs 1.96 (layer 20's KV head stays). In the adapter test (wall
  with a sync before each of the 40 acks) 5120-row chunks at positions 0 / 5120: decode ~176 ms vs off ~282-300 ms.
- Re-run: `scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_decoder_skip_decode.py`;
  the skipped chunk's time: `DS41_DECODER_SKIP=decode DS41_ENGRAM_TABLE=resident BRINGUP_CHUNK_LAST=0 BRINGUP_POS=51200
  BRINGUP_PROFILE_HOST=1 scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_profile.py`.
- For the owner: tasks.yaml P.51 has no `ab_nonlast: true`, so the orchestrator's A/B times only chunks with last
  unset (identical on both sides); adding it would put the skipped chunk (p51200_nonlast) in the table. Open:
  tt-d-gen's prefix-hit request must carry the slot's id history below actual_start for the engram tail (the same
  slot has it; a migrated prefix needs load_tokens), as for every follow-up today.

## FIX.28 (overseer, OWNER 2026-11-13): the gate tests run DECODER_SKIP=decode too

- What. The gate tests drive the skipped path themselves, on the same model load (loaded with the shipped
  DECODER_SKIP=off; `TtDS41Model.use_decoder_skip("decode")` now works on an off-loaded model: the decoder lists are
  host-only, built by `_build_decoder`, dropped again by `use_decoder_skip("off")`). The full-path checks are unchanged
  and run first.
  - test_determinism: after the full-path check, `check(..., run=decode_sequence, suffix="_decoder_skip")`
    (testing/model_determinism.py takes `run` / `suffix`). Per prompt (novel, code), 3 runs (cold, warm): a 4-chunk
    prompt (N 20480) stops at P = (N-128)//64*64 = 20352 (tt/attention.decoder_skip_stop); request A [0, 7680), then
    request B from 7552 with global_write_start 7680 (replay of [7552, 7680), no history write) up to P; every chunk
    last=False. Compared bit for bit: layers 0-19's snapshots after each chunk, every history cache decode reads over
    [0, P) (layer 20's KV head included) and layers 0-19's rings.
  - test_contract: hooks.contract_variants -> testing/contract.py `variants` (F266): a request [0, P) on the next slot,
    P = 10048 for the contract's 10208 tokens, each chunk global_write_start = actual_start, last=False; the same ack /
    ack-timing / producer / independent / fixed-state checks, judged at P against FIX.26's handoff golden at P
    (hooks.contract_handoffs adds decoder_skip_stop; golden made with BRINGUP_CPU_THREADS=16, DS41-241), plus
    nothing written in [P, N). Metrics suffixed `_decoder_skip`; `contract_checks_failed` counts both requests.
  - test_adapter_acks: `decoder_skip_requests`: slot 0 end P (10048) and slot 1 end P+40 (10088: not a multiple of
    64; serving_contract.md P.49 rule: served end = end//64*64 = P, checked); acks per layer per chunk, history at
    every ack, layers 0-19's snapshots at the cap, final history [0, P) and snapshots at P vs the golden
    (caches.GoldenKV(handoffs=True)), [P, N) unwritten.
  - Under decode only layers 20-39's window snapshot / ring are not judged (never written; decode does not read
    them): kv_contract.WINDOW_UNWRITTEN, adapter.contract_state_pcc, the cap-0 path in adapter.py (DS41-240,
    rewritten).
- Results (8x4, 40 layers, gate --check, all PASS): ladder err_final_hidden 0.1321 (limit 0.2422), top5 1.0;
  determinism_max_abs_diff 0 (full) and determinism_max_abs_diff_decoder_skip 0 (decode: 5 chunks per sequence, 3
  runs x 2 prompts, states 0, 15.9 s); contract (full / decode): contract_checks_failed 0 / 0, acks_early 0 / 0, 800 /
  800 ack blocks, state judged at 10176 / 10048, window ring PCC 0.98793 (limit 0.94787) / 0.98938 (limit 0.96071),
  compress_kv 0.98877 / 0.98868 (limits 0.9370 / 0.9366), index_k 0.99334 / 0.99329 (0.9465 / 0.9463), engram tail
  exact; unwritten_records_changed_decoder_skip 0 of 60; adapter acks: decode_acks_failures 0, served end 10048 for
  both requests, 0 of 120 records past P changed; test_decoder_skip_decode served_bytes_differ 0 of 20,912.
- New metrics and limits: `determinism_max_abs_diff_decoder_skip` == 0 (in this gate); the `*_decoder_skip` PCC
  limits come from the same rule as the full path (calib x official drift at the judged point); no existing limit
  changed. Suggested lines for P.51's gate: `determinism_max_abs_diff_decoder_skip == 0`,
  `contract_checks_failed_decoder_skip == 0`, `acks_early_decoder_skip == 0`,
  `unwritten_records_changed_decoder_skip == 0`, `decode_acks_failures == 0` (tasks.yaml not edited).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.28 --spec models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.

## FIX.29 (overseer fix, OWNER 2026-11-13: "LOAD THE MODEL ONLY ONCE") - one model load per gate

What changed
- One load path for the model: `tt/runners/adapter.load_model` (TtDS41Model). Both the bring-up hook
  (`bringup/hooks.py device_model`) and the serving runtime (`DS41PrefillRuntime.__init__`) call it, so the
  contract tests still build the model through the engine's adapter, and that is the same load as everyone else's.
  load_model goes through `testing/batch.shared_model`: outside a batch process it just loads (and counts); inside one
  it returns the batch's model (loaded once, RoPE for the longest sequence any test needs, the LM head if one needs it).
- `testing/batch.shared_model` (framework): one cache per process; reuse only when key / mesh / layers / config
  (chunk, expert cache, checkpoint) / load-time switch values (hooks.batch_settings, read from the environment) match
  and the seq / LM head suffice; else a warned, counted reload. hooks.batch_call_settings = (PREFILL_END,): read per
  chunk, so a test may change it on the shared model. Every real load records `model_loads` (value: this process;
  `total`: running total across processes, DS41-242). After each run: Settings snapshots re-read (a run's monkeypatch
  is undone), gc, the run's changed metrics and result files written to the plan results.
- `core/batch.py`: `shareable()` = SHARED_TESTS (now also test_contract.py, test_contract_acks.py) or a test file
  declaring `BATCH_SHARED = True` (tests/test_determinism.py, test_decoder_skip_decode.py,
  bringup/contract/test_adapter_acks.py, test_prefill_end.py, test_window_snapshot.py). `gate_groups` /
  `gate_entries`: consecutive units of one model configuration form one process (determinism runs first: FIX.25's
  cold sequence); hints now always cover target.seq and the ladders' seqs.
- `core/gate.py run_units` + `_run_group`: the units that will run (not kept / reused) are grouped; a group of >= 2
  runs as ONE batch process (logs/<task>.check.shared/: plan, results, per-unit logs, process.log); each unit then gets
  its own rc, metrics, result files and log section, so verdicts, stamps and the delta work as before; a unit the
  process never reached gets rc 1 with the reason. `gate_model_loads` = model loads of this gate's processes.
- test_decoder_skip_decode: no longer loads with DECODER_SKIP=decode (monkeypatch); it asserts the model is as loaded
  (shipped off), runs `use_decoder_skip("decode")` / off per request, and restores `loaded_skip` at the end. Truly
  load-time switches (a different process / reload): DECODER_SKIP=exact / replay128 (replay128 builds the tail
  model), DEEP_LAYERS, WINDOW_KV_KEEP, every weight / format switch; decode is host lists only (FIX.28).
- (2) Agents' command: `scripts/run_safe_pytest.sh --one-process --run-all <test> <test> ...` (one process, the
  model loaded once; BRINGUP_BATCH_KEEP_GOING=1 runs past a failure). bringup-engineer.md rule 3 tells agents.
- Diagnostic only: `BRINGUP_INJECT=rowshift2` (testing/batch.inject): the run's 2nd embed call gets its ids rolled
  by one (an off-by-one, once: not repeatable), undone after the run. Never in a gate.

Measured (8x4, 40 layers, FIX.28's gate command as is)
- Loads: 5 (FIX.28, one per unit) -> 1 (gate_model_loads 1; load 214 s).
- Wall: FIX.28 units 352 + 342 + 340 + 485 + 468 = 1987 s (33 min) -> FIX.29 gate 728 s (12 min; the shared process
  717 s: determinism 331 incl. the load, ladder 68, contract 23, acks 133, decode 137).
- Metrics vs results/FIX.28.json: 502 identical (every err_*, top5, determinism 0 / 0, contract, acks,
  served_bytes_differ 0 of 20,912, ...); 10 differ, all timing or new: prefill_seconds / chunk_seconds_c10 68.1 ->
  10.3 s (the ladder's first chunk is no longer the process's cold one), model_load_s 206.8 -> 213.6,
  determinism_s 96.2 -> 95.7, determinism_s_decoder_skip 15.9 -> 16.8, decode_test_chunk_wall_ms off / decode
  287.0 / 191.1 -> 284.5 / 187.3; new: gate_model_loads 1, model_loads 1, model_shared 1.
- Injected failure still caught in the shared process (BRINGUP_INJECT=rowshift2, `--one-process`):
  model test_determinism FAIL ("identical inputs gave different outputs ... first differing chunk 1", max |diff|
  4.5); test_contract FAIL (producer read-back PCC compress_kv 0.736 < 0.937, index_k 0.850 < 0.946);
  test_adapter_acks FAIL (layer 0 snapshot pcc 0.170 < 0.998); test_decoder_skip_decode FAIL (novel N 300: 128 of
  128 records differ off vs decode). Contract, acks and decode ran in one process past each other's failure
  (BRINGUP_BATCH_KEEP_GOING=1), one load.
- CPU selftests: selftest/test_gate_one_process.py (one process, gate_model_loads 1; separate processes count 2 and
  fail; a failing shared run fails its unit and stops), test_one_process.py / test_model_determinism.py updated
  (the contract test may share now). The full selftest suite's other failures are the same at HEAD (checked on a
  clean worktree of HEAD for the gate-related files).

Shared / not shared, and why
- Shared in one process: the mesh, the model weights, the program cache, the engram table.
- Not shared: each test's KV caches and states (freed after each run; the adapter's slabs are never freed by the
  model, only its own buffers), its environment, results, log and pytest session (test modules re-imported).
- Still another process / load: a unit with another switch value in its command (model_key), a test that sets a
  load-time switch itself (reload, warned and counted in gate_model_loads), a test file not declared shareable.

Re-run
- `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.29 --spec models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`
- Several tests by hand: `scripts/run_safe_pytest.sh --one-process --run-all <test files>`

## FIX.31 (overseer fix, OWNER 2026-11-13: "bookkeeping eats the Galaxy") - count every device run once

What changed (framework only; no model, test or fork code)
- (1) Every `scripts/run_safe_pytest.sh` run of a bring-up spec files its result (core/runrec.py, called by
  tt_safe_run.sh `ttrun_runrec_begin` / its EXIT hook `ttrun_runrec_end`): key = tests + measuring pytest args (no
  precompile flags, no --run-all, no --one-process) + `stamps.measuring_env` + code key + goldens key + waivers;
  records in runs/<run>/stamps/_runs/<key>.json (newest first, result files kept). A `BRINGUP_BATCH_CMD` batch
  (--one-process, F259 unit_command) becomes the same plan as a file first, so each of its runs is filed. The gate
  (`core/gate._lookup`) reuses a unit from: its stamp; its stamp made before FIX.31 (legacy code / env key aliased,
  `stamps.ensure_aliases`, runs/<run>/stamps/code_aliases.json); every run's newest runner record, all PASS; another
  task's gate record of the same runs (old records: command recovered by its hash from that task's gate). Never a
  FAIL / HANG / NOT RUN, never an older PASS behind a newer failure.
- `stamps.env_key` = `measuring_env`: without TT_METAL_CACHE, TT_METAL_HOME when it is this checkout, the launch
  variables (BRINGUP_BATCH*, BRINGUP_HOST_MEM_*, BRINGUP_DEVJOB_DIR, BRINGUP_RUNREC). DS41-244: a shell's gate
  (e8e6...) and the orchestrator's (4413...) were different keys, so `gate --status` called passes stale.
- (2) `overseer fix ... --unit TASK#label --selftest <file>`: the fix's gate is its selftests + the named units only
  (raw segments of the task, BRINGUP_TASK=<task>), with the limits those units recorded (newest PASS in the store).
  Without them the old chaining stays. A model-level unit still brings FIX.25's determinism unit (owner: no exemption).
- (3) `stamps.BOOKKEEPING`: gate.py, batch.py, run_safe_pytest.sh, tt_safe_run.sh count in the code key by their
  measuring part only (python: top-level defs by name; shell: functions / `# --- <title> ---` sections). Listed =
  bookkeeping; anything unlisted (a new function too) counts. Measuring parts kept: gate.py `gate_env`, `_run_group`
  and imports; batch.py `parse`, `hints`, `gate_entries`, `plan_from_cmd`, ITEM_VARS, ...; the safe runner's card,
  env, pytest command, hang / reset. core/hostmem.py, runrec.py, multigate.py: wholly bookkeeping.
- (4) An agent ended by an API / auth / overload error (orchestrator.API_ERROR: 403 / 529 / overloaded / Cloudflare)
  is INFRA: no attempt charged, no gate (changed tree or not), back-off 60 s doubling to 30 min, 8 in a row stops the
  task for a person; a cut-off attempt's changes are named in the next brief.
- (5) core/multigate.prerun: `pending_full_gates(include=[X.3])` collects every pending unit of the picks and the
  milestone, drops what each gate would keep / reuse, runs each run once (dedup by run keys) in one batch process per
  model configuration (grouped by key, not adjacency; past failures), then each task's gate takes its results
  (`run_gate(pre=...)`): own verdict, metrics, commit. `with_determinism` makes the check its own unit (never inside
  another unit's command: P.51's ladder re-ran). `batch.launch_form`: a unit without --no-precompile joins a shared
  process with kernels compiled on demand, the log says so; `batch.why_not` logs why a unit runs alone.
  `batch.model_key` folds a switch at its test-process value into unset (DS41-245: ENGRAM_TABLE=resident is NOT
  folded; its test value is mmap).
- (6) watch: a live gate (runs/<run>/live/gate-<pid>-<task>.json, written by run_gate), a safe-runner run
  (live/<pid>.json) or a chain (`python -m ...watch live --pid $$`) keeps the watch waiting; results land in
  runs/<run>/gates.log (`pid=` lines; the orchestrator's own skipped, `gate check` lines never wake it); `--grace`
  (60 s) bridges a chain's gap.

Decisions
- Precompile inside a batch: not done (disagree with "precompile once, then batch"): the up_front_collect pass would
  run test_batch.py's whole plan under NO_DISPATCH, and testing/batch.shared_model would keep the model that pass
  built from shape-only tensors for the real pass. F151 already drops the pass for model-level steps (kernels cached).
- model_key and gate_groups are launch: the device side reloads whenever a load-time switch differs
  (testing/batch.shared_model compares hooks.batch_settings), so a key fold cannot share a wrong model.
- Code key: legacy (1117c19bb2f25445) of HEAD 82a197f0edf == FIX.30's records; new key of HEAD == today's tree
  (b4e915a478bc066f) after every FIX.31 edit: the edits are all bookkeeping.

Selftests (CPU, through the safe runner): selftest/test_count_runs_once.py (20), test_back_to_back_gates.py (6),
test_fix_claim_and_infra.py (7), test_watch.py (+4); updated test_model_determinism.py / test_one_process.py (the
determinism unit is its own). Full selftest suite: 183 failures, 180 of them identical at HEAD (clean worktree;
missing tale-of-two-cities.txt.bz2 and others), the other 3 were these updated tests.

Re-run
- `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.31 --spec models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`
- selftests: `scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/selftest/test_count_runs_once.py` (etc.)

Measured (8x4, `gate FIX.31 --check`, after the last edit)
- Units: 5; reused 3, run 2. FIX.31#test_determinism: X.3's pass on the same tree (19:35, a shell's env key aliased);
  FIX.30#test_ladder: X.3's pass of the same runs (X.3's command has --no-precompile, FIX.30's not); the CPU selftest
  FIX.30#test_one_process: a hand run filed by the safe runner. Run: test_profile (2 runs) + test_positions in ONE
  process, one model load (131 s): 316 s device, gate 322 s. FIX.30 ran the same two units in 2613 + 819 = 3432 s
  (gate_model_loads 4 -> 1).
- Not 0 device minutes: FIX.30's own records carry env key 6d56329acc1a26b2, which no current launcher reproduces
  and whose contents were never stored (DS41-244); profile and positions had no other pass on this tree.
- Metrics vs results/FIX.30.json: accuracy identical (err_final_hidden 0.13964838470398958 both, top5 1.0, op_rows 1538,
  pos_chunk 5120); timing within noise (prefill_ms_full 1916.5 -> 1918.2, chunk_wall_ms 182.6 -> 183.5,
  device_ms_total 172.52 -> 172.12).
- First check attempt failed in FIX.30's selftest test_a_runs_record_gives_its_configuration_a_need...: not hermetic
  (asserted BRINGUP_HOST_MEM_NEED_GB unset; the gate now has a measured need for that unit). Fixed in the test.
- Note: selftest/* is NO_MEASURE (F120), so a selftest edit does not change the code key; a selftest gate unit is
  only re-run when its newest run record failed.

## P.52 (perf, attempt 2): the MoE expert-replica work (P.45 / P.46 / P.47) ported onto model_bringup
- Expected per chunk (written before any device run; perf_guide sections 1-2): MOE_REPLICAS=runtime with no copies
  vs off = 0 ms device (one more, empty expert slot a chip: the flat experts skip a slot with no tokens; P.47 measured
  185.77 vs 185.64 ms, +0.13 = noise), 0 ms host (no extra launch: the slot map rides on the router bias tile;
  set_replicas is never called by the harness). Accuracy and every output byte identical (no copy = the same routing).
- Ported (code only; no ledger, result or R.* / P.48 / FIX.23-probe file): `git diff 12ac1ca9d6c 1e4b42b2d98` (merge
  base .. P.47) on tt/, tests/ and ttnn/ttnn/bringup, applied with `git apply -3`. In that range only four commits touch
  those paths: ffe570b0b4f (P.46), 1884f5ec92d (P.45), 18daae54d6e (removes P.46's tile_gather), 1e4b42b2d98 (P.47).
  New: tt/expert_slots.py, tests/test_{moe_counts,moe_replicas,expert_slot_copy}.py,
  moe_block_main_ttnn/expert_slot_copy/ (op). Changed: tt/{model,moe_block_main_adapter,settings}.py, the fused head
  (router slot map; RouterSpec.candidates), common/moe_block_config.*, CMakeLists, nanobind, INDEX / README / CHANGELOG.
  Known issues DS41-227..230 copied under their old ids (still free here); the repo_map row for expert_slots.py.
- Conflicts (5; this branch's behaviour kept): adapter block() cache key -> (mesh, chunk, COMBINE_SPLIT, spc), and
  MoEBlockConfig keeps combine_snap_split; model.begin_chunk: P.51's write_start / replay code kept, P.45's
  `_counts_start` line after it; INDEX / README / CHANGELOG: both entries kept (COMBINE_SPLIT first). The fused head,
  the combine fork and settings.py merged without conflict. Switches MOE_REPLICAS / MOE_COUNTS stay default off.
- test_moe_counts asserted DS41_MOE_COUNTS=on (P.45's gate set it); P.52's unit runs it bare, so the test now switches
  MOE_COUNTS on itself (monkeypatch) and writes the files to pytest's tmp_path when the env does not say otherwise.
- Results (8x4, COMBINE_SPLIT=on): test_expert_slot_copy PASS (32 copies 2.79 ms, 228 GB/s); test_moe_replicas PASS
  (replicas_checks_failed 0; 32 copies on layer 20 vs off max |diff| 0.0, every (a)-(e) bit-identity check ok);
  test_moe_counts PASS (11 files, 40 layers each). `measure --task P.52` (MOE_REPLICAS=runtime, nothing filled):
  determinism 0 (novel, code, state); quick L00 0.0195 .. L38 0.1167, L39 0.9115 (= P.51's), host transfers 0, wall
  184.7; p51200 device_ms_total 185.7 (P.51 off 185.11; P.47 185.64 / 185.77), experts 103.9, wall 203.0 (spread 6.5).
  As expected: 0 within noise. The off side and the ladder / profile units: the orchestrator's A/B.
- Gotcha: `measure --part` names are the A/B runs (determinism, quick, p51200), not the test file stems.
- Re-run: `gate P.52 --spec <spec> --check --unit 'P.52#test_moe_replicas'` (likewise test_expert_slot_copy,
  test_moe_counts); `python -m models.demos.common.bringup.orchestrator measure --spec <spec> --task P.52`.

## P.53 (perf, attempt 1): MOE_FILL=piggyback, the replica fill carried by the combine's port senders
- Expected per chunk (written before any device run; perf_guide sections 1-2, 4 "moved time"): the change removes
  P.47's serial expert_slot_copy programs from the chunk. Per copy round on the 8x4 (32 copies, one a chip) that is
  2.79 ms device a round (0.71-0.78 ms for one copy alone); with the copies hidden in the combine's idle link time the
  device saving is up to that, minus what the carrying combines grow (the owner's mark: < 5 % of a ~1.5 ms combine,
  i.e. < ~0.08 ms a carrying call). The links carry ~10 % during the combine, but the copy also adds DRAM reads on the
  source and writes on the destination while the experts stream their weights (weight-bound): that, not the link,
  is the likely cost. Host: ~0 (runtime args only, no launch; set_replicas enqueues no program). With
  MOE_REPLICAS=runtime and nothing filled (the gate's ladder / profile: no set_replicas in the harness) the A/B
  measures ~0 by construction: the fill path with no job only adds one empty check a port round.
- Design as built (MOE_FILL=piggyback; MOE_FILL default serial = P.47, the owner's text and tasks.yaml keep_default):
  - Kernel (routed_experts_combine port writer, `REC_FILL`): the port's own RISC (RISCV_0, NoC1) reads the pieces from
    its chip's DRAM into 4 staging slots at the end of its arena (16 KB run table + 4 x 10880 B) and sends them on the
    connection it already holds, to the destination's DRAM. A piece goes only in a main-loop round that sent no row and
    while the channel holds < MOE_FILL_CAP packets; whatever of the call's budget is left goes out after the main loop
    while the credits drain (bounded by the host's budget; no wait on another core: no new hang path in the kernel).
    Not "an idle core hands pieces to the port": the port's own sender reads (no hand-off protocol); measured enough.
  - Host schedules the pieces (tt/expert_fill.py FillQueue): progress lives on the host, the jobs are runtime args each
    call (no read-back, no host transfer in the layer loop). One layer's copies a call, never on that layer's own call,
    MOE_FILL_PIECES (125 = 1/16 slot) a source port a call, split over its jobs (<= 4). Copies ride link 0 of the
    first hop only: the next call's start-barrier increments to the destination travel the same connection and route,
    so after one more MoE call the pieces (and the flushed done increment) have landed; the layer's slot map then
    flips by a device copy of a map staged at set_replicas. set_replicas unmaps changed chips at once (host write, as
    P.47) and the entry waits one call (superseded pieces fenced).
  - MOE_REPLICA_DEMO=rotate (diagnostic, default off): a new 32-copy neighbour table on one layer before every chunk run
    (tt/model.py begin_chunk -> adapter.replica_demo), for step 3 only.
- Step 1 microbenchmark (tests/test_moe_fill_piggyback.py; carrier layer 20, copies into layer 19; combine device time,
  slowest chip, median; fill on / off alternating in one window behind 100 matmuls, DS41-248): no job -0.5 %. Added %
  (hidden % of pieces), cap 4: 1/4 slot a call pair -1.3 (92.5), neighbour 32 +0.2 (95.1), random 32 +88 (55.5);
  1/8 -0.2 / -0.5 / +2.6 (86.6); 1/16 -0.7 / 0.0 / -0.6 (96.6); 1/32 +0.8 / -0.1 / -0.2 (100). cap 1: random 1/16 +23.
  Whole slot a call (earlier run, no lead): +90 % pair .. +760 % random. Every slot bit-exact, done words exact, 100
  calls back to back no hang, carrier outputs bit-identical. PASS (< 5 %) at <= 1/8 slot a call; DS41-246.
- Step 2 (same test, section 3): set_replicas(neighbour table) queues 32 copies, none mapped; the filled layer's own call
  while pending carries none and its output is unchanged; mapped after 18 carrier calls (1 wait + 16 + 1 fence);
  slots bit-exact; output with the table bit-identical to none mapped, rows reach the spare slots; half the chips
  changed: unmapped at once, mapped after 18 calls; emptied: unmapped. All checks ok.
- Step 3: NOT measured. The first profile (MOE_FILL=serial + MOE_REPLICA_DEMO=rotate, rung last) hung the board in the
  timed runs (DS41-247). Per the owner's safety rule: stopped, no reset, no further device run (so no final gate
  --check after the clang-format pass; the format changed whitespace only, the unit test_moe_fill_piggyback passed on
  the code before it).
- What limits it: a call can hide ~1/8-1/16 of a slot a source port; one layer's copies a call; at most 4 copies a
  source port a call; the fill rides link 0 only. A 32-copy table needs ~18 MoE calls (< one chunk's 21 prefill MoE
  layers); tables for many layers at once queue (one layer a call).
- Re-run: `gate P.53 --spec <spec> --check --unit 'P.53#test_moe_fill_piggyback'` (~9 min: devjob); step 3 (after the
  owner's reset): `DS41_MOE_REPLICAS=runtime DS41_MOE_FILL={serial,piggyback} DS41_ENGRAM_TABLE=resident
  DS41_MOE_REPLICA_DEMO=rotate BRINGUP_PROFILE_WALL=1 scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/common/bringup/tests/test_profile.py` (and BRINGUP_POS=51200).
- stock ttnn.matmul instead of ttnn.bringup.mhc_post: tests/test_moe_fill_piggyback.py only, a timing lead (100
  matmuls 4096^3 on 11 x 10 = 110 cores ahead of each profiled window), not a model path. Cores of the change: none
  added (the combine's 8 port cores a chip; the 110-core cap holds).

## P.55a (perf, attempt 1): MoE exact replicas, step 1 (go/no-go): the bound and the copy deadline
- Expected per chunk (written before the device run; perf_guide 1-2): this step changes no model code, so 0 ms device
  and 0 ms host for the model. The bound it measures: the brief's replica table saved ~24.5 ms device a chunk with
  32 learned copies (novel), so the exact split should reach ~25-35 ms at most (the busiest chip down to ~1.1-1.2x
  the mean; the ceiling is busiest - mean, ~70 ms with the CPU model on P.45's chunk-10 counts), and the copy deadline
  should cut that by roughly the receivers' wait for a 0.71+ ms copy (light chips reach their replica at ~0.5-0.7 ms).
- Design as built (no model code changed; MOE_REPLICAS stays off, nothing to A/B): tests/test_moe_exact_bound.py
  loads the model with MOE_COUNTS=on (read back, no files) and wraps the MoE block's three ops (fused_head,
  sparse_dispatch_2d, routed_experts_combine) to know each call's device-operation ids. Per workload it runs chunk 10
  (tokens [51200, 56320)) three times under the real-time profiler (no device profiler, no op mode): novel and code
  after a device prefill of chunks 0-9 (the real chunk 10, then two re-runs on the profile's steady path), random ids
  (the p51200 position profile's tokens, fresh state: one compile run, then three). Per run and layer: every slot's
  tokens and the program's per-chip device time. tests/moe_exact_model.py (CPU, numpy) fits the per-chip expert time
  and plans the exact split per layer, max mode and deadline mode (definitions in its docstring; DS41-249 detail).
  720 layer plans run in a spawn pool (~7 s; spawn, not fork, so no child inherits the open mesh).
- Prefix: the device's own chunks 0-9, not the golden prefix, for both prompts (the code prompt has no golden; P.45's
  count files were made the same way). The quick rung's chunk is chunk 10 at 51,200, so "quick" and "51,200" are the
  same tokens; the third workload is the position profile's random ids at 51,200 (DS41-228: not a prompt's routing).
- Results (gate unit P.55a, PASS; medians of 3 runs, ms a chunk; save = busiest chip's experts time before - after):

  | workload (p51200) | measured rec (slowest / model) | max save | ceiling | deadline save | copies on time | 2 paths |
  |---|---|---|---|---|---|---|
  | novel | 56.3 / 49.9 | 21.9 | 23.0 | 10.1 | 0 of 335 | 20.9 |
  | code | 47.0 / 38.7 | 10.6 | 11.9 | 2.9 | 0 of 87 | 10.1 |
  | randids | 63.2 / 58.1 | 30.9 | 32.0 | 16.5 | 5 of 524 | 27.7 |

  Fit: A 38.9 us (DRAM rate, fixed), B 7.55 us a row tile, C0 0.235 ms, R^2 0.966, rmse 0.068 ms (360 layer calls).
  Three gate runs of the unit (same code but the output format) agreed within ~0.3 ms on every number.
  Per-layer tables (measured, model, busiest, mean, max, deadline, every chip's own finish, every expert's tokens):
  results/P.55a_exact_bound.json and the unit log.
- Verdict (owner's line, < ~5 ms a chunk surviving = NO-GO): novel GO (10.1), code NO-GO (2.9), random ids GO (16.5).
  The test does not decide for the owner; step 2 was not started.
- What limits it (DS41-249): the light chips sit on the DRAM floor (12 experts x 38.9 us = 0.467 ms) and reach their
  replica before any whole-expert copy can land (>= 0.75 ms on one link pair), so the receivers idle ~0.3 ms; most
  of the loss comes back if a copy is split over two paths. Also: ~94 % of the copies outlast the home's own compute
  of the expert (the zero-extra-read source window); the dispatch's imbalance and the combine's link traffic are not
  modelled. The max-mode bound sits at the floor too: a perfect split leaves ~0.47 ms a layer, the ceiling 23.0 ms
  (novel) is about what the brief's learned 32-copy table measured (~24.5 ms device a chunk).
- Gotcha: record_json writes indent=1 (a number a line); per-chip lists went in as strings to stay under 500 KB.
- Re-run: `gate P.55a --spec <spec> --check --unit P.55a` (~4 min on the 8x4, devjob).

## P.55b (perf, attempt 1): MoE exact replicas, step 2: the streamed copy inside the layer's own combine
- Expected per chunk (written before the first device run; perf_guide 1-2, 4 "moved time"): the model's chunk path
  copies nothing in step 2 (MOE_REPLICAS=exact = the runtime layout, 13 slots, the spare slot empty; the device planner
  and dispatch split are P.55c), so 0 ms device and 0 ms host vs runtime (P.52: runtime vs off +0.1 ms = noise).
  Microbenchmark guesses: one 19.9 MB copy over two link paths ~0.36-0.45 ms (P.47: 0.71 ms on one); a real split
  adds what the receivers wait past their own work (~0.2-0.5 ms a layer in a balanced random-input call).
- Read first: P.55a's verdict (OV-39: novel 10.1 GO, code 2.9 NO-GO, random 16.5): the stop rule (both under ~5 ms)
  does not trigger, so step 2 ran.
- Design as built (MOE_REPLICAS=exact; MOE_EXACT_BLOCK 32, MOE_EXACT_FANOUT 2, MOE_EXACT_CAP 8, MOE_EXACT_LIMIT_MS 20):
  - Kernel: routed_experts_combine's port writer, REC_FILL + REC_FX (`set_combine_fill_exact`; P.53's REC_FILL path
    untouched). Each port core reads its jobs from a DRAM page (jobs tensor, page dir x links + link; chip page 8: the
    waits), so P.55c's device planner can write them; a job sends pieces [p0, p1) of a run table to one chip on that
    port's link, rows first; a flushed atomic increment of a flag word on the receiver every 32 pieces (the receiving
    router flushes the block's writes first: a flag means the block is in DRAM); a relay job reads a piece only once
    its own chip's flag word counts its block. Flag words by call parity (zeroed by the chip's first port for the next
    call); the chip's first port is the waiter: stamps block arrivals, waits for every target after its own work
    (bounded: never a hang), every port writes 16 stamp words. 448 persistent words on the collector; +3328 B of each
    port's arena; cores: none added (the 8 port cores; rule 18 holds).
  - Source (OV-39 (3)): the home RE-READS the slot from DRAM (the port's own reads, 4 staging slots), no L1 hand-off
    from the gate/up cores: in ~94 % of P.55a's copies the copy outlasts the home's compute of that expert, so its L1
    blocks are gone; cost: one extra 19.9 MB read a sending edge, inside the measured added time.
  - Paths (OV-39 (2)): every edge sends half its pieces on each link of the first hop (2 links); `(home, slot, [r],
    via)`: a quarter on each of 4 links, half relayed through a free chip's spare slot (route chosen so it avoids the
    straight link). Several receivers of one expert: a relay tree (tt/expert_exact.plan_tree: nearest first, a parent
    whose port toward the child is still free, at most 2 children) - two copies on one port serialize (code split
    1.80 ms before that rule, 1.24 after). Not a fabric multicast: a 2D multicast writes the same address on every
    chip of a rectangle, so non-receivers in it would lose their spare slot.
  - Streaming (OV-39 (1)): the per-block flags and their arrival stamps are there (first block 0.04-0.07 ms); the
    receiver computing on landed blocks is P.55c (the flat expert has to poll the flag before a block).
- Results (gate unit P.55b PASS; layer 20, random chunk; combine alone ~0.86 ms, chips' own work 0.62-0.65 ms):

  | case | copies | latency ms | added ms |
  |---|---|---|---|
  | no job | 0 | - | +0.009 |
  | pair, 1 link | 1 | 1.397 | +0.630 |
  | pair, both links | 1 | 0.890 | +0.106 |
  | pair, 4 links (detour) | 1 | 0.761 | +0.008 |
  | neighbour 32 | 32 | 0.954 | +0.254 |
  | real split novel (4 hot experts, depth 5) | 23 | 1.254 | +0.372 |
  | real split code (4 hot experts, depth 3) | 10 | 1.235 | +0.371 |

  exact_copy_bitexact 1, exact_copy_runs_ok 100 (0.20 s, every gate / wait clean), exact_combine_identical 1,
  exact_copy_added_ms 0.372, exact_copy_latency_ms 1.254 (novel split). Deadline (P.55a's model, chunk 10 at 51,200,
  a copy edge = the measured pair): both links 0.89 ms -> novel 5.87 ms a chunk, 0 of 174 on time; code 1.25, 0 of
  41; 4 links 0.76 ms -> novel 8.74, 0 of 269; code 2.44, 0 of 74 (P.55a assumed 0.36 ms: 20.9 / 10.1).
- The final `gate P.55b --check` (PASS, determinism 0; the recorded metrics): exact_copy_added_ms 0.4185,
  exact_copy_latency_ms 1.229 (novel), code 1.229 / +0.297, pair both links 0.876 / +0.137, 4 links 0.772 / -0.004,
  neighbour 0.961 / +0.195; the model: both links novel 6.16 ms (0 of 183) code 1.36 (0 of 46), 4 links novel 8.55
  (0 of 259) code 2.30 (0 of 63). Run-to-run spread ~0.02-0.05 ms on latency, ~0.05-0.1 ms on added.
- `measure --task P.55b` (after the last edit): determinism 0 (novel, code, state), quick min layer PCC 0.60548578 at
  L39 (bit-identical to P.52 / P.53's), host transfers 0, wall 231.9 ms (spread 25.1). Its stamp's env is EMPTY: P.55b's
  task has no A/B switch entry, so it ran the defaults (MOE_REPLICAS=off, and not DS41_ENGRAM_TABLE=resident, which
  P.52 / P.53's measures set: 182-185 ms). The wall is not comparable with theirs; no code on the off path changed
  (the stock combine's kernel compiles without REC_FILL / REC_FX). The orchestrator's A/B sets the switches.
- What limits it (DS41-250): a link carries ~7 GB/s of copy while the combine's main loop runs (rows first, the experts
  streaming weights from the same DRAM; 47 us a 340 KiB block), ~20 GB/s after (17 us); one link averages 15.5 GB/s,
  the same as expert_slot_copy. So a whole expert lands at 0.76-0.89 ms on a pair, ~1.25 ms through trees, after a
  light receiver's own work (0.47 ms in P.55a's real chunk, 0.63 here): no copy is on time; the receiver waits.
  Levers left: more links (detours through free chips, 3-4 directions), copy pieces not waiting for idle rounds,
  partial replicas (only the blocks a receiver needs before its deadline).
- Gotchas: metrics.results_dir() outside the gate is generated/bringup_adhoc (the test falls back to the ledger for
  P.55a's record); a port's waiter stamps live in the persistent exact words, so they describe the last call only.
- Re-run: `gate P.55b --spec <spec> --check --unit P.55b` (~6.5 min, devjob); `measure --task P.55b`.

## P.55c (perf, attempt 1): MoE exact replicas, step 3: the exact split in the whole model
- Expected per chunk (written before the first device result; perf_guide 1-2, 4 "moved time"; chunk 10 at 51,200):
  - fill (fallback c': spare slots filled before the timed chunk, copies free): device -10 to -20 ms novel, -5 to -10
    code (P.55a's max-mode bound 21.9 / 10.6 ms, less what dispatch imbalance and the plan's hash split leave), host 0.
  - stream (the design: the copy in the layer's own combine, receivers wait for it): device -3 to -6 ms novel, -1 to
    -1.5 code (P.55b's deadline model at 0.89-1.25 ms a copy: 6.2 / 1.4 ms) minus the copies' added program time
    (+0.3-0.4 ms a layer with copies, partly hidden behind the busiest chip): likely ~0 to -4 novel, ~0 or worse code.
    Host: +~1 ms (a runtime-arg update of the combine a layer).
  - not in either: a device planner. The split is decided in the fused head's router rows (router_rows.hpp
    slot_of) BEFORE any chip knows the other chips' counts (the counts exist only after the routing exchange), so an
    on-device exact plan needs a second head pass a layer (or a cross-chip count exchange inside the head): ~0.2 ms
    a layer, ~4 ms a chunk, i.e. about the whole stream gain.
- Design as built (MOE_REPLICAS=exact; default off, keep_default):
  - Split (planner -> dispatch): a layer's plan [(home chip, home slot, kept tokens, [(receiver, tokens)])] becomes the
    slot map with each holder's share as a range of the token's 15-bit hash (router_rows.hpp slot_of, the uneven
    split: copy word slot | lb << 9; old maps keep the even split bit for bit; tt/moe_block_main_adapter.
    exact_map_words). The fused head then routes every (token, expert) to the home or a replica; dispatch, counts,
    combine follow the slots unchanged.
  - Order: the flat expert already runs its experts biggest first (expert_schedule.hpp se_dyn_build), so the home
    computes its hot expert first. Receivers: with the exact copy (REC_FX) the combine builds the flat expert with
    fx_spare: every role puts the spare slot (the replica) last and never pins it.
  - Copy: P.55b's REC_FX as built (home re-reads the slot, both links of the first hop, relay trees, a ready flag a
    32-piece block). New: the chip's waiter port writes the call's token into the landed word (exact word 448) once
    every waited flag word hit its target; the receivers' gate/up and down weight readers read the replica's weights
    only after it (fx_wait.hpp, a 4 B poll every ~1 us, bounded). Whole copy, not block by block (OV-39 (1): at most
    the replica's own compute, ~0.05-0.1 ms of a ~0.9 ms gap). While any layer has a stream plan, every MoE call of
    the block runs the exact program (empty jobs where none): the flag words alternate by call parity (a stock call in
    between would leave a parity unzeroed).
  - Planner: NOT on device (DS41-253). The plans are computed on the host (tests/moe_exact_model.Plan, P.55a's A / B)
    from a counting run of the same chunk; a device planner fed the same counts plans the same, so the numbers below
    leave out only its own cost and a count exchange. Fallback c' (fill): set_exact(fill=True), spare slots filled by
    expert_slot_copy before the chunk (outside the profiled window).
- Results (the final `gate P.55c --check`, unit test_moe_exact_replicas; chunk 10 at 51,200 = the quick rung's chunk,
  after the device's own prefill of chunks 0-9; re-runs on the steady path; device ms a chunk = median of 3 profiled
  runs; wall = median of 5 unprofiled runs, first host call to the sync, MOE_COUNTS on - not the framework's clean wall):

  | prompt | side | device ms | wall ms | experts+combine | dispatch | replicas | max diff vs off |
  |---|---|---|---|---|---|---|---|
  | novel | off | 187.6 | 272.6 | 57.1 | 20.4 | 0 | (off repeat 0) |
  | novel | fill (copies free) | 156.6 (-31.0) | 249.8 (-22.8) | 38.1 | 11.2 | 512 | 0 |
  | novel | exact (streamed) | 194.5 (+6.9) | 280.3 (+7.7) | 68.9 | 18.0 | 290 | 0 |
  | code | off | 170.7 | 259.7 | 46.6 | 14.4 | 0 | (off repeat 0) |
  | code | fill (copies free) | 157.9 (-12.8) | 249.1 (-10.6) | 39.1 | 11.1 | 295 | 0 |
  | code | exact (streamed) | 180.2 (+9.5) | 274.1 (+14.4) | 53.4 | 13.9 | 67 | 0 |

  The run before (same code but the wall runs): novel 185.9 / 161.5 / 195.0, code 170.5 / 158.3 / 175.6 device ms.
  Every side bit-identical to off on every MoE layer's output, repeats bit-identical (the exact side with every
  receiver's spare slot spoiled first: the wait works). Copy cost in these real chunks (OV-40 (3)): a layer with copies
  2.10 ms (novel, 23 layers) / 1.98 (code, 8) vs 0.93 / 0.96 with the same split's copies in place and 1.62 / 1.51 off;
  layers without copies +0.07 / +0.02 ms (the exact program with empty jobs, after a copy layer's skew).
- What limits it (DS41-251): a copy tree lands 1.25 ms (one tree, P.55b) to ~1.9 ms (19 receivers of one expert) into
  the layer, while the off layer's busiest chip is done at 1.0-1.9 ms; the plan assumed 0.89 ms and its model
  predicted 1.21 ms for the copy layers (measured 2.10). Recalibrated (P.55a's model, whole-copy wait): 1.25 ms a copy
  -> novel 0.77 ms a chunk (0 of 44 copies on time), code 0.06; 1.6 ms -> 0. The split itself is worth 12-24 ms
  (fill); the copy has to be in place before the layer (another layer's call, P.53's piggyback, or a learned table,
  P.54) to keep it.
- The model's chunk path (the orchestrator's A/B, ladder, profile): exact without a plan == runtime, the stock combine:
  ~0 by construction (P.52: runtime vs off within noise); the A/B's random tokens could not show the split anyway
  (DS41-228).
- Cores: none added (rule 18 holds: the port cores and the flat expert's cores, inside the 11 x 10 cap). Host: 6
  runtime args a flat weight-reader core and the port's token, written every exact call.
- Gotchas: DS41-252 (the steady path's first re-run is no baseline: attempt 1 read 4256 max diff, all of it the
  baseline); the flat expert's biggest-first order (a receiver's replica would go first); flag parity (above).
- Re-run: `gate P.55c --spec <spec> --check --unit 'P.55c#test_moe_exact_replicas'` (~5 min on the 8x4, devjob);
  `python -m models.demos.common.bringup.orchestrator measure --spec <spec> --task P.55c`.
- `measure --task P.55c` (change on: MOE_REPLICAS=exact ENGRAM_TABLE=resident, no plan in the chunk path, after the
  last code edit): determinism 0 (novel, code, state); quick min layer PCC 0.60548579 at L39 (= P.52 / P.53 / P.55b:
  the off path's bytes), host transfers per layer 0, wall 182.6 ms (spread 3.9); p51200 wall 206.9 (spread 11.4).
  Final gate --check: PASS (ladder rung last within every limit, determinism 0; profile chunk_wall 182.7, device 171.4).

## P.56 (perf, attempt 1): MoE replicas learned from the previous chunk (MOE_REPLICAS=prev)
- Expected per chunk (written before any code change or device run; perf_guide 1-2, 4 "moved time"; chunks 1..10 of
  a real prompt, chunk 0 runs off by design):
  - device: P.55c's split with the copies in place saved 31.0 ms (novel) / 12.8 ms (code) at chunk 10; a plan from
    chunk K-1's counts keeps ~80-89 % of the in-sample gain (owner's CPU model) -> ~25 / ~10 ms if every copy were in
    place. The fill is the limit: P.53's piggyback hides ~1/16 slot a source port a call (125 pieces of ~1830), so a
    port moves ~2.7 slots over the 38 other MoE calls of a chunk, ~8-10 new copies a layer a chunk over the mesh;
    copies that stay hot are kept (no re-copy), so the tables fill over the first 2-4 chunks. Costs: one planner
    program a layer (counts exchange over the fabric + the plan, 1 core + up to 4 sender cores) ~0.03-0.05 ms device
    -> +1.2-2 ms; the exact combine program (REC_FX) on every MoE call +0.02-0.07 ms a layer (P.55c) -> +1-2 ms; the
    carried pieces ~0-1 % of the combine. Net guess: novel -8 to -15 ms device from chunk ~3 on (less at chunks 1-2),
    code -2 to -6 ms. The orchestrator's random-token profile (DS41-228) shows ~ +2 to +4 ms (costs only).
  - host: +1 prepared launch a layer (the planner, 32 chip programs, addresses patched: ~60-100 us) and the combine's
    fill args every call (~20 us) -> +3 to +5 ms a chunk; the off-map guard adds a launch only when a layer's plan is
    not ready (chunk 0 of a request: 40 copies).
- Design as built (MOE_REPLICAS=prev, needs MOE_FILL=piggyback; tt/expert_prev.py, tt/kernels/moe_prev_planner.cpp):
  - Learn (on device, no host round trip, no sync): after EVERY MoE call a prepared planner program (ROLE 1: up to 4
    sender cores a chip send the chip's own head.counts slots to every other chip over the 2D fabric, a fused write +
    atomic increment, tail_send's pattern; ROLE 0: one planner core a chip waits for the 31 rows, adds each spare
    slot's tokens to the expert it held, plans). The plan is integer arithmetic, identical on every chip: the busiest
    chip's biggest remaining expert gets the least loaded free receiver and x = (busiest - receiver - 165) / 2 tokens
    (MOE_PREV_FIXED 165 = one replica's weight read in tokens, P.55a's fit), stop below 64 tokens (MOE_PREV_MIN); a
    receiver already holding the expert is kept within 165 tokens (MOE_PREV_SLACK). Split = the uneven hash split
    (P.55c's slot_of), the map written straight into the layer's router bias (fp32 tile) for its NEXT call.
  - Fill: the plan's NEW copies (spare slot not already holding the expert) are cut into 2000 pieces and given to the
    calls o+1 .. o+38 in order (o = the layer's call ordinal; 40 MoE calls a chunk), at most MOE_FILL_PIECES (125) a
    source port a call and 30 jobs a page, budgets for all chips simulated identically on every chip (global state);
    the planner writes its own chip's jobs into a ring of 44 REC_FX job tensors (one a call ordinal), each job naming
    the copied layer's weight buffers (fork change: words 6 / 7). Every MoE call runs the exact combine with ring[o %
    44]; call o+39 is the fence (its start barrier follows the pieces on link 0 of the first hop). A copy that does not
    fit its window is not made (receiver dropped, counted).
  - Guard (host, no transfer): a layer's plan is used only if its next call comes >= 40 ordinals after the plan; else
    the load-time empty map is copied back (device copy) and the call runs off. Request start (begin_chunk at 0 / a
    replay): one reset program (all job page headers zeroed, a new epoch: every layer state reads empty); chunk 0 runs
    off; nothing crosses requests (spare bytes from the old request are re-copied, not reused).
  - Not P.53's REC_FILL path: its jobs are host runtime args, and the plan exists only on the device; the carry is
    P.53's rules (first hop link 0, 125 pieces a port a call, cap 4, fence) on P.55's device-readable REC_FX pages.
    P.55c's split (uneven map) and receiver order (replica last, fx_spare) are reused; P.54's exp/p54_wip not used.
- Checks (tests/test_moe_prev_replicas.py, gate unit): every chunk's last-layer output prev vs off max |diff| 0, prev
  repeat 0, off repeat 0; a host mirror (expert_prev.plan_host) replays all 440 planner runs a pass from the read-back
  counts: every layer state, slot map, the global state and the pending job pages equal the device's, every chip the
  same (attempt-1 run: 0 mismatches of 477 checks a prompt).
- Results, attempt-1 full run (8x4, fresh requests, chunks 1..10 back to back after chunk 0; device ms = program real-
  time profiler, slowest chip a program, summed; wall = unprofiled, sync before / after each chunk):

  | prompt | device off -> prev (ms) | experts phase off -> prev | wall off -> prev | planner (device) | ready layers |
  |---|---|---|---|---|---|
  | novel | 174.9 -> 157.3 (-17.6) | 54.6 -> 44.1 | 195.3 -> 193.5 | 5.0 | 40/40 every chunk 1..10 |
  | code | 163.1 -> 155.3 (-7.8) | 48.1 -> 42.1 | 191.1 -> 194.3 | 6.3 | 40/40 every chunk 1..10 |

  Per chunk (novel, device off -> prev): c1 173.6 -> 155.7, c2 171.8 -> 155.7, c3 176.5 -> 156.2, c4 175.0 -> 159.7,
  c5 173.4 -> 156.2, c6 177.4 -> 159.0, c7 175.2 -> 157.7, c8 173.7 -> 155.4, c9 174.7 -> 156.5, c10 177.7 -> 160.8;
  code c1 170.7 -> 153.9, c2 167.2 -> 155.0, c3 160.9 -> 154.2, c4 159.2 -> 157.4, c5 160.5 -> 151.2, c6 166.6 -> 158.0,
  c7 159.2 -> 153.2, c8 159.7 -> 157.7, c9 164.3 -> 157.2, c10 163.1 -> 155.4. Chunk 0 (off by design, planner only):
  +13 / +9 ms. Receivers mapped a chunk: novel 204-233 (kept 90-144, new 77-222), code 232-338; slots complete in time
  = every mapped slot (by construction); copies dropped for budget: novel 252 at c1 -> 6-31 from c4, code 130-330.
  Full table per chunk and layer: results/P.56_prev_replicas.json.
- MOE_FILL_PIECES=250 (1/8 slot a port a call, same code + the planner trims below): drops ~0 but the experts phase is
  the same (novel 44.06, code 42.04 ms): the extra copies are marginal; device novel 174.9 -> 156.0, code 162.3 ->
  152.5 (planner 4.0 / 5.2 ms). Default stays 125 (P.53's validated hiding).
- Planner cost: core phases (cycles, chip 0, full model): reads 1.2k, counts wait 20k (15 us; 284k in the host-bound
  4-layer quick model = launch skew, DS41-248), plan 36-50k, map + writes 7k after trims (36k before: base map tile
  precomputed, five-step msb, batched job page read-modify-write). Program ~0.1-0.13 ms a layer = 4-5 ms a chunk.
- What limits it: (1) the wall: the chunk is host-bound (~190 ms wall vs ~157 device), so the device gain does not
  show in the clean wall per chunk; (2) the planner's 4-5 ms a chunk (a 32-chip program a layer, the exchange's skew);
  (3) one spare slot a chip a layer; (4) the plan is chunk K-1's routing (P.55c's in-sample fill reached 156.6 novel at
  chunk 10 vs 160.8 here); (5) DECODER_SKIP modes that skip layers on some chunks break the 40-call rhythm: the guard
  then runs those layers off.
- Gotchas: this Galaxy is Blackhole (-mcpu=tt-bh): job pages are written whole (1 KB read-modify-write) to keep DRAM
  accesses 64 B aligned; DS41-255 (NONE); G's epoch starts at 1 so zeroed layer states read empty.
- stock ttnn.copy (router bias <- load-time bias) is the guard's device copy, not a compute op; no INDEX op applies.
  Cores: planner 1 + up to 4 sender cores a chip (rule 18 holds); the combine adds none.
- Re-run: `gate P.56 --spec <spec> --check --unit 'P.56#test_moe_prev_replicas'` (~8 min, devjob);
  `P56_QUICK=1 scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/
  test_moe_prev_replicas.py` (4 layers, chunks 0-3, a whole copy a call: iterating only).
- `measure --task P.56` (MOE_REPLICAS=prev MOE_FILL=piggyback ENGRAM_TABLE=resident): determinism 0 (novel, code,
  state); quick min layer PCC 0.60548579 at L39 (= P.52-P.55c), host transfers per layer 0, wall 182.3 (spread 7.1);
  p51200 device_ms_total 160.22, device_ms_experts 80.02, wall 188.4 (spread 11.6), 1656 launches (the planner's
  run_prepared is not counted), 0 new programs. Comparable off sides (P.52-P.55c old): device 186.0-187.5, experts
  104.8-106.0, wall 201.6-203.8. The p51200 profile re-runs one chunk, so its plan is in-sample (random ids): it
  overstates the real prompts' gain (DS41-228).
- FINAL (the gate unit on the final code, `gate P.56 --check --unit 'P.56#test_moe_prev_replicas'`: PASS, 365 s;
  results/P.56.json, results/P.56_prev_replicas.json), chunks 1..10 mean:

  | prompt | device off -> prev (ms) | experts phase off -> prev | wall off -> prev | planner (device) | max diff / repeat / mirror |
  |---|---|---|---|---|---|
  | novel | 175.15 -> 156.10 (-19.1, -10.9 %) | 54.74 -> 44.15 | 190.6 -> 191.6 | 4.08 | 0 / 0 / 0 of 477 |
  | code | 164.00 -> 153.08 (-10.9, -6.7 %) | 48.15 -> 41.98 | 192.6 -> 192.1 | 5.31 | 0 / 0 / 0 of 477 |

  Per chunk device off -> prev: novel c1 173.9 -> 155.2, c2 173.4 -> 152.9, c3 176.5 -> 155.3, c4 175.0 -> 154.5, c5
  173.0 -> 153.8, c6 177.8 -> 158.5, c7 175.2 -> 156.8, c8 173.9 -> 154.0, c9 174.8 -> 161.2, c10 178.2 -> 159.0; code
  c1 170.4 -> 152.2, c2 167.7 -> 152.3, c3 165.0 -> 150.4, c4 160.7 -> 157.0, c5 161.3 -> 150.4, c6 166.0 -> 156.2, c7
  160.6 -> 151.0, c8 159.3 -> 151.9, c9 164.7 -> 156.1, c10 164.4 -> 153.4. The plans (receivers kept / new / dropped
  per chunk) equal attempt 1's exactly: the same chunk sequence plans the same. Expected (above): novel -8 to -15,
  code -2 to -6 device: beaten (the kept copies fill the tables from chunk 1, P.53's budget was enough).
- Host (OV-23): the planner's dispatch (run_prepared, 32 chip programs, 8 + 3 x senders addresses patched) 92.5 us a
  call (median), the pre-call hook (guard + the combine's fill args) 4.9 us: ~3.9 ms a chunk of host; the chunk's wall
  did not move (host-bound at ~190 ms on both sides).

## P.57 (perf, attempt 1): host-bound wall with P.56 on: where the host time goes, fixability, one-layer trace prototype
- Expected (written before any device run; perf_guide 1, 3): analysis only, no shipped change, so device 0 / host 0
  saved by this step. Guess for the breakdown of the ~190 ms host per chunk (novel chunk 10, prev): ttnn op dispatch
  (b) ~110-130 ms (~1650 outermost calls x ~70 us on 32 chips), Python in tt/ (a) ~30-45 ms, prepared launches (c)
  ~8-12 ms (~100 calls x ~90 us), transfers / waits (d) < 3 ms (host_transfers_per_layer = 0; the engram worker join),
  per-chunk host work (e) ~5-10 ms (begin_chunk with ENGRAM_ASYNC; P.56's guard ~0.2 ms). Trace (whole chunk) would
  take (a)+(b)+(c) off the critical path: wall -> ~device + ~10 ms, i.e. ~190 -> ~165 ms prev (-25 to -30 ms a chunk);
  a one-layer replay should cost < 0.1 ms host vs ~4-5 ms untraced.
- What I did: tests/test_host_bound.py (no tt/ change, nothing shipped). One model load; side off / prev switched in the
  loaded model (P.56's `_side`). Per side: the novel prompt chunks 0..10 back to back (warm, profiled = real-time
  profiler on chunk 10, clean, instrumented passes) and one warm chunk at 51,200 (random ids, fresh state,
  steady_state; clean x5, profiled, instrumented, then testing/profile.py host_profile = BRINGUP_PROFILE_HOST's
  cProfile). Instrumentation (HostClock): FastOperation.__call__, ttnn.bringup.run_prepared, transfer functions,
  PrevReplicas.before/after, the engram worker join, each charged its exclusive time; run_block's signposts give the
  sections, Python = a section's time minus its wrapped calls. Trace prototype: side off, layer 21 at novel chunk 10.
- Terms: host = first host call to the last op enqueued; wall = to the end of the closing sync; drain = wall - host;
  device = each program's slowest chip, summed (rt profiler, never per-op mode).
- Results (final run; c1-10 = mean of the clean pass, chunks 1..10):

  | chunk | host | wall | device | wall explained |
  |---|---|---|---|---|
  | novel c1-10 off | 160.5 | 180.9 | 177.2 (c10) | device-bound: wall = device + ~4 |
  | novel c1-10 prev | 185.0 | 186.6 | 160.0 (c10) | host-bound: wall = host + 1.6 drain |
  | p51200 off | 163.5 | 202.5 | 194.6 | device-bound: wall = device + 7.9 (start-of-chunk idle) |
  | p51200 prev | 181.7 | 184.1 | 162.8 | host-bound: wall = host + 2.4 |

  So wall = max(host, device) + 2-8 ms holds; with prev OFF the chunk is not host-bound; P.56 adds ~18-24 ms of host
  (combine 402 -> 731 us a call x 40 = +13.5, planner run_prepared 99 us x 40 = +3.9, guard +0.6), which moves the
  regime and eats its own 17 ms device gain.
- Host parts, novel chunk 10, prev (instrumented 195.7 vs clean 190.3 ms; op calls 1732 + 40 run_prepared):
  (a) Python in tt/ 34.4 (attention 11.0, ffn_residual 7.0, experts 4.1, ~6 of it ttnn.deallocate); (b) op launches
  139.9 (C++ 135.3, Python forks 4.6; ~80 us a call); (c) run_prepared 8.9; (d) transfers / waits 10.0 (layer 1 waits
  7.6 ms for the async engram job, the rows' copy 2.4); (e) per chunk 2.5 (begin_chunk 0.5, ids 1.4, guard 0.6);
  (f) other 0. Same shape off and at p51200 (op launches 122-140, Python 32-35). Full lists: results/
  P.57_host_bound.json (per op: calls, us a call; per section).
- Fix list (ms a chunk of host; the wall gains only down to the device, ~160 ms with prev):
  1. Segment trace: the MoE / FFN half of each layer (experts + ffn_residual + router / shared expert / moe_add, ~93 ms
     host, no per-start argument) as 40 traces replayed between eager attention: host ~185 -> ~95, wall -> device,
     ~ -25 ms wall (prev). Needs 4 below, fixed input / output buffers. ~8-16 agent-hours.
  2. Whole-chunk trace: host -> ~10 ms (inputs only); traced device ~10 % lower (prototype) -> wall ~145-150, ~ -35
     to -40 ms. Needs every blocker below lifted: ~25-40 agent-hours.
  3. P.56 ring slot on the device (the combine reads o % W from G[0]; one fixed job-ring address): -12 to -13 ms host
     (combine back to ~400 us). ~3-5 h (fork option).
  4. Engram job one chunk ahead (begin_chunk of chunk k+1 issued during chunk k; prefill knows the ids): -8 to -10 ms
     host (the layer-1 wait). ~2-4 h (+ a serving-contract hook).
  5. moe_block_main fused_head (301 us) / combine (402 us off) host: one mesh program with common args (DS41-145/147
     pattern): ~ -10 to -14 ms. ~6-10 h.
  6. Attention-side launch cuts (slice 211, rope 136, slice_write 164, to_layout / untilize / typecast / concat ~230
     calls): -10 to -20 ms. ~6-12 h over a few picks.
  7. all_gather (106 us x 92) / mhc_post (100 us x 82) cheaper launches: -5 to -8 ms. ~4-6 h.
  8. Python glue trims (dealloc batching, section glue): -5 to -10 ms. ~4-8 h. (A trace removes 5-8 outright.)
- Trace verdict: FEASIBLE at a fixed position (prototype), NOT yet chunk to chunk. Prototype (layer 21, side off, 47
  programs, input cloned each run): capture after 3 warm runs, no capture error, 20 replays bit-identical, no hang;
  host a layer 6.45 -> 0.020 ms, device a layer 3.93 -> 3.54 ms (trace_bit_identical 1). Blockers (detail in
  known_issues_detail.md, DS41-256): per-start host args (rope slices, kv-write positions, window rows, scalars,
  slice_write offsets, ckv row0, indexer start / kv_len) - per-start traces or start as a device tensor; per-chunk
  stores allocated in the chunk (top-k / candidates) - preallocate; token ids input buffer; engram wait inside layer 1;
  P.56's host ring slot and guard. Not blockers: host reads (0), CCL semaphores / device call counters (replayed fine),
  allocations inside the capture. DS41-100's address-keyed blocker is gone.
- Gotchas: a single-chunk clean wall varies +-10 ms run to run (off chunk 10: 169.6 vs 182.2 host in two runs); judge on
  the c1-10 means. The rt-profiled pass's span / idle are inflated by the profiler's host overhead (span 220-250 ms).
  The gate run writes results/P.57_host_*.pstats (host_profile's raw stats) beside the json.
- Re-run: `gate P.57 --spec <spec> --check --unit P.57` (~7 min with the model load; devjob).

## P.58 (perf, attempt 1): TP=4 / TP=8 experts go/no-go (CPU model + dispatch/combine microbenchmark, no model change)
- Expected per chunk (written before any code or device run; perf_guide 1-2, 4 "moved time"; chunk 10 at 51,200):
  no model code changes, so 0 ms device and 0 ms host for the shipped model. The check itself: compute side as the
  overseer's estimate (experts phase novel EP 49.9 -> TP4 ~29 / TP8 ~28 ms, code 38.7 -> ~28), maybe a few ms worse
  with 4x / 8x more, smaller weight reads and 32-row padding per quarter expert. Communication: a token's rows go to
  ~4.4 groups x 4 chips = ~17.6 chips (TP4) or ~3.3 x 8 = ~26 (TP8) instead of ~5.5 (EP), and the combine carries 4x
  / 8x the rows (24 / 48 partial rows a token instead of 6); P.55a measured dispatch ~20 ms a chunk (novel), so my
  guess is dispatch +20 to +40 ms and combine +10 to +30 ms a chunk for TP4 (more for TP8): TP4 chunk ~ 175 - 21 +
  30..70 = ~185-225 ms vs EP + P.56 156 ms (novel) -> likely NO-GO unless the fabric hides most of the extra rows.
  Host: TP4 as a real design adds no launch (same ops), so ~0.
- No setting / switch added: the task is a check with no model change (the brief's generic "switch" text has nothing
  to apply to); nothing ships, MOE_* settings untouched.
- What I built: tests/moe_tp_model.py (CPU model: groups = mesh rows (TP4) / columns (TP8), chip k = r C + c as
  flush_counts writes it; scaled and fitted variants; bias steering) and tests/test_moe_tp4_check.py (one mesh open,
  ~3 min of device work). TP is emulated with the fork's ops unchanged: each 1/g slice of an expert is an expert of
  its own on the g chips of its group, with the same router row and bias, so the fused head builds the TP route masks
  itself and the combine sums the g partial rows in fp32 (reduce = 0, inside the combine). Three op limits forced
  approximations (each also a cost of a real build): flat op cannot split 576 / 288 (stand-ins 512 / 256, compute x
  1.125); fused head top-K <= 16 (TP4 run at 2 / 3 / 4 experts a token, TP8 at 1 / 2, cost fitted vs K / vs chips a
  row goes to and taken at 6 experts); routing planner N x slots <= 1024 (head-driven runs at 32 slots a chip, the
  compute runs at the true 48 / 2 x 48).
- Results (gate unit P.58; DS41-257, detail in known_issues_detail.md):

  | chunk 10 at 51,200, device ms | novel | code |
  |---|---|---|
  | EP off (P.56 measured, c1-10) | 175.2 | 164.0 |
  | EP + P.56 prev (measured) | 156.1 | 153.1 |
  | TP4 (model: EP off + experts diff + dispatch added) | 215.3 | 209.1 |
  | TP8 (same; compute upper bound, comms extrapolated) | 334.8 | 329.8 |
  | experts phase EP / TP4 / TP8 (flat measured at real counts + combine cost) | 57.4 / 94.8 / 197.9 | 45.3 / 86.3 / 190.0 |
  | flat op alone a chunk EP / TP4 / TP8 | 52.5 / 86.0 / 125.9 | 42.5 / 77.5 / 119.0 |
  | dispatch added / combine added a layer, TP4 | +0.068 / +0.099 ms | +0.103 / +0.148 ms |
  | dispatch added / combine added a layer, TP8 | +0.48 / +1.68 ms | +0.53 / +1.70 ms |

  (The iteration run before the gate, same code but the check: TP4 215.0 / 207.7, TP8 311.7 / 313.6; the TP8 combine
  is a linear extrapolation from K 8 / 16 to 48 and moves the most.)
  CPU model with P.55a's fit scaled (A / g, B / g): EP 49.9 / TP4 28.3 / TP8 26.0 (novel), the overseer's estimate;
  with the device's own fits at the slice widths: EP ~37-42, TP4 ~64-69, TP8 ~78-81 ms a chunk over chunks 0-10 of
  P.45's counts (compute only, both prompts). Busiest / mean (compute) novel chunk 10: EP 2.32, TP4 1.16, TP8 1.04 -
  the balance is real, but each TP chip's row work does not shrink with the width (B 7.66 us a 32-row tile at 2304,
  6.25 at 512, 4.85 at 256).
- VERDICT: NO-GO for TP4 and TP8. Both are slower than EP off, let alone EP + P.56, on both prompts; the loss is the
  compute (the flat op's per-row cost), not the fabric (TP4's comms add only ~7-10 ms a chunk). Precision is not the
  issue (rel err vs fp32 EP 0.0560, TP4 0.0561, TP8 0.0564). Effort for a real TP4: a narrow-expert flat op (x read
  once per row for all of a chip's slices, y partials summed on the chip before the combine) + head / planner limits
  (top-K 24, 1536 slots) + 576-wide split: ~40-60 agent-hours, and it would still need the per-row cost to fall ~4x
  to break even with P.56.
- Gotchas: the realtime profiler window must hold only the timed calls (a warm call first: the first dispatch of a
  routing is ~1 ms, program build); a handful of tokens (<= 6 of 5120) split an expert's slices when two experts tie
  within the router's rounding (check >= 99.5 %).
- Re-run: `gate P.58 --spec <spec> --check --unit P.58` (~4 min with the mesh open; devjob).
- Final `gate P.58 --check`: PASS (determinism 0; P.58 unit 152 s, tp_checks_failed 0). `measure --task P.58` (no
  switch, defaults; no model code changed): determinism 0 (novel, code, state), quick min layer PCC 0.60548579 at L39
  (= P.52-P.56: the off path's bytes), host transfers per layer 0, wall 234.3 ms (spread 19.0; defaults without
  ENGRAM_TABLE=resident, comparable with P.55b's 231.9, not with the 182 ms measures that set it).

## P.59 (perf, attempt 1): flat expert x pieces of at least 8 KB (EXPERT_X_PIECES), then the replica / TP baseline re-measured
- Expected per chunk (written before any device run; perf_guide 1-2, 5): device -4 to -8 ms (chunk 10, MOE_REPLICAS off),
  all in the experts phase (routed_experts_combine; P.56 off: 54.7 ms novel / 48.2 code a chunk). Reason: Marko's
  one-chip benchmark (12 experts x M rows) gives -21 % at M 256 and -17 % at M 2048, ~0 at M 64; our busiest chip has
  ~4,000 token-slots a layer (~M 330 uniform) but the light chips sit at the DRAM floor (DS41-249), so only the busiest
  chip's row term (B, per 32-row tile) shrinks: ~-0.1 to -0.2 ms a layer. With P.56 prev the gain is smaller (the
  busiest chip already offloaded). Host: 0 (same program count; a different compile define only). Caveat: the change
  does nothing when the plan's sub-block is 128 rows (x block 34,816 B: 4 pieces of 8,704 B already); it acts only at
  64-row sub-blocks (17,408 B: legacy 4 x 4,352 B, min8k 2 x 8,704 B). The program factory's log line now prints the
  piece bytes: check it.
- What changed: fork option `FlatLayoutOptions.x_min_piece_bytes` (moe_block_main_ttnn flat_routed_expert; 0 = the
  source's 4 pieces when aligned; > 0 = the most of 4 / 2 / 1 pieces of >= that many bytes: Marko Staletovic's
  1e2ab798da1 x-pieces hunk, his :127-131, taken by hand behind the option because a plain 3-way merge would replace
  the source's line; his DW split, gate/up hot/cold loop and TEMP env hooks not taken). In the plan key and program
  hash (appended only when set). Model setting EXPERT_X_PIECES (legacy | min8k, default min8k) ->
  tt/moe_block_main_adapter.py x_min_piece_bytes() -> the block's FlatLayoutOptions (and the _BLOCKS key). The
  factory's log line prints the piece bytes. No new op, no host work, no cores (rule 18 unchanged).
- Piece size the plan really gets (rule: check it): the model's combine-embedded flat plan and the standalone plan
  both pick mt 2 (64-row sub-blocks) at H 5120, I 2304, 12 experts, 110-core cap: x block 17,408 B -> legacy 4 x
  4,352 B, min8k 2 x 8,704 B (the model's log: "pieces 2 of 8704 B"). P.58's TP slice widths (512 / 256) plan mt 4:
  34,816 B -> 4 x 8,704 B on both sides (no change).
- Flat op alone (tests/test_expert_x_pieces.py::test_expert_x_pieces, 8x4, every chip 12 experts x M tokens,
  real-time profiler, slowest chip, median of 10 behind 40 matmuls; outputs compared bit for bit on the written rows):

  | M a expert | legacy ms | min8k ms | change | bit-identical |
  |---|---|---|---|---|
  | 64 | 0.657 | 0.661 | +0.5 % | yes |
  | 256 | 1.012 | 0.822 | -18.8 % | yes |
  | 512 | 1.760 | 1.447 | -17.8 % | yes |
  | 1024 | 3.243 | 2.683 | -17.3 % | yes |

- Model level, P.56's real-prompt test (tests/test_moe_prev_replicas.py, DS41_EXPERT_X_PIECES=legacy / min8k, two
  processes, same day; chunks 1..10 mean; device = slowest chip a program, summed; experts phase =
  routed_experts_combine; every maxdiff / repeat / mirror check 0 on both sides; routing counts identical):

  | prompt | side | device off | device prev | experts off | experts prev | prev gain (device) |
  |---|---|---|---|---|---|---|
  | novel | legacy | 174.89 | 156.43 | 54.63 | 44.26 | -18.46 |
  | novel | min8k | 166.73 (-8.16) | 154.86 (-1.57) | 46.14 (-8.49) | 42.85 (-1.41) | -11.87 |
  | code | legacy | 163.33 | 153.59 | 48.14 | 42.03 | -9.74 |
  | code | min8k | 156.56 (-6.77) | 151.82 (-1.77) | 41.87 (-6.27) | 40.83 (-1.20) | -4.74 |

  Wall (unprofiled, host-bound): novel off 193.4 -> 187.7, prev 189.4 -> 190.9; code off 183.3 -> 178.5, prev 189.2
  -> 186.8 (noise-level except off). Per chunk: generated/bringup_adhoc/P.59_{legacy,min8k}_prev_replicas.json.
- P.55a's timing model refit (tests/test_moe_exact_bound.py, chunk 10, 3 workloads x 3 runs, 360 layer calls; A is
  fixed at the DRAM rate by the model's design, 12 x 19.9 MB at 512 GB/s; B and C0 least squares):

  | fit | A us / active expert | B us / 32-row tile | C0 ms | R^2 | rmse ms |
  |---|---|---|---|---|---|
  | P.55a (2026-11-13) | 38.9 | 7.55 | 0.235 | 0.966 | 0.068 |
  | legacy (same day) | 38.9 | 7.66 | 0.220 | 0.963 | 0.072 |
  | min8k | 38.9 | 5.96 | 0.249 | 0.970 | 0.050 |

  With it, P.55a's exact-split bound shrinks (legacy -> min8k): novel max save 22.4 -> 13.4 ms, deadline 10.4 -> 4.2,
  2-path 21.4 -> 13.1, ceiling 23.5 -> 15.2; code max 11.0 -> 5.1, deadline 3.1 -> 0.9; busiest / mean novel 2.33 ->
  1.90. Measured experts at chunk 10 (shortest chip) novel 57.1 -> 49.5, code 47.3 -> 41.7 ms.
- P.58's compute-side estimate redone (test_expert_x_pieces_layouts: the flat op at P.55a's real chunk-10 counts per
  layout, both sides, one plan and one weight set; + P.58's measured combine / dispatch costs a layer, unchanged:
  x pieces do not touch them; EP off / prev = this step's P.56 run). Device ms a chunk at 51,200:

  | | novel legacy | novel min8k | code legacy | code min8k |
  |---|---|---|---|---|
  | EP off (measured) | 174.9 | 166.7 | 163.3 | 156.6 |
  | EP + P.56 prev (measured) | 156.4 | 154.9 | 153.6 | 151.8 |
  | TP4 (EP off + experts diff + dispatch added) | 215.1 | 215.7 | 208.5 | 207.4 |
  | TP8 | 334.3 | 335.0 | 328.8 | 327.8 |
  | flat a chunk EP / TP4 / TP8 | 52.5 / 86.0 / 125.6 | 43.8 / 86.0 / 125.8 | 42.4 / 77.6 / 118.7 | 36.8 / 77.6 / 118.8 |

  Calibration slopes (A us a slot / B us a row tile): EP 51.3 / 7.75 -> 51.2 / 6.46; TP4 14.5 / 6.27 -> 14.6 /
  6.26; TP8 13.6 / 4.84 -> 13.7 / 4.84. CPU model with the P.55a fit scaled (A / g, B / g, the overseer's estimate):
  novel EP 49.9 -> 42.0, TP4 27.8 -> 27.4, TP8 25.5 -> 26.4; code EP 38.6 -> 33.8, TP4 27.0 -> 27.7, TP8 26.0 -> 27.1.
  Verdict unchanged (TP4 / TP8 NO-GO); EP's lead over TP widens by ~8 ms (novel).
- Reading: min8k is a real device win on the off path (-8 novel / -7 code ms a chunk, all in the experts phase, on
  target with the expected -4 to -8); with P.56 prev on it adds only ~-1.6 ms because the replicas already cut the
  busiest chip's row work, so P.56's own gain over the new baseline is -11.9 (novel) / -4.7 (code), down from -18.5 /
  -9.7. The chunk stays host-bound (wall ~187-191 ms).
- measure --task P.59 (EXPERT_X_PIECES=min8k ENGRAM_TABLE=resident): determinism 0 (novel, code, state); quick min
  layer PCC 0.60548579 at L39 (= P.52-P.58: the same output bytes), host transfers per layer 0, wall 175.9 (spread
  6.3); p51200 (random ids, MOE_REPLICAS off) device_ms_total 176.62, device_ms_experts 94.67, wall 192.4 (spread
  10.3), 1656 launches, 0 new programs (earlier off sides: device 186.0-187.5, experts 104.8-106.0).
- Re-run: `scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/
  test_expert_x_pieces.py` (~1 + ~2.5 min); the baseline: P.56 / P.55a tests with DS41_EXPERT_X_PIECES=legacy|min8k
  and BRINGUP_RECORD_TASK=P.59_<side> (~6 + ~4 min each, devjob).
- Final `gate P.59 --check`: PASS (determinism 0; rung last err_final_hidden 0.132 vs 0.242, top5 1.0; profile
  device_ms_total 164.02, chunk_wall_ms 176.7). x_pieces_* and baseline_* metrics recorded in results/P.59.json.

## P.60 (perf, attempt 1): P.56's host overhead off the critical path (MOE_PREV_HOST legacy | device)
- Expected per chunk (written before any code change or device run; perf_guide 1, 3; real novel / code chunks 1..10,
  MOE_REPLICAS=prev on both sides): device ~0 (+-0.3 ms: the combine reads one more 64 B word a port, the planner
  advances the same counter it already keeps); host -12 to -16 ms (P.57: combine 731 -> ~400-450 us a call x 40 =
  -11 to -13; planner launch 92.5-99 -> ~60-70 us x 40 = -1 to -1.5; guard -0.3). The wall moves by about the host
  cut only while the chunk stays host-bound: novel prev host ~185 vs device ~155 (min8k) -> wall ~186 -> ~172-175;
  code (device ~152) similar. Caveat: P.57 attributes the combine's +330 us to the ring slot changing every call; that
  is a hypothesis I check first with a host probe (the cost could be the exact path's per-call args or program size,
  which a device-side slot would not remove).
- Where the host time really went (tests/test_moe_prev_host_probe.py, a diagnostic: 4-layer model, novel chunks 0..3,
  plus temporary C++ timers in the combine's launch, removed): handing every call ring[0] (a fixed jobs address) did
  NOT help (978 vs 856 us a combine call), so P.57's "ring slot chosen on the host" was not the cost (DS41-256's
  attribution: contradicted, DS41-259). The cost was routed_experts_combine_program_factory.cpp set_fill: with the
  exact fill every cache hit rewrote every fill port's args and every flat-expert weight reader's wait args on 32
  chips (a new token and jobs address each call): ~350-470 us of a ~810-930 us launch (off: ~440-520 us, set_fill 1).
- What changed (MOE_PREV_HOST=device, the new default; legacy = P.56 as built, selectable):
  1. Fork (moe_block_main routed_experts_combine; option, defaults keep the old behaviour): set_combine_fill_exact(...,
     ring=W, ordinal=G, wait=False). The jobs tensor holds W page sets; each REC_FX port reads word 0 of G (the
     planner's call ordinal) with one 64 B DRAM read and takes set G[0] mod W; wait False: the flat expert's readers
     do not wait for a landed word and there is no token (prev's copies land in earlier calls, fenced; its page of
     waits is always empty, so the readers' wait never had anything to wait for). With a ring the factory keeps a
     signature of the fill args per program: a cache hit with the same args writes nothing.
  2. Model: tt/expert_prev.py one job tensor [chips, 1, W x 9, 256] (built per host mode, PrevReplicas.mode), the same
     fill args every call (gate_up / down fixed: every prev job names its own layer's buffers in words 6 / 7);
     kernels/moe_prev_planner.cpp compile arg 20 RINGPG (pages of a set; 0 = legacy): one ring address instead of 44.
  3. Planner launch: not made cheaper beyond the 43 ring args. Measured (probe, PROBE_PLANNER=1): run_prepared of the
     planner 55.7 us a call with the same io 40x, 58.1 us alternating 4 layers' io (the patches cost ~2 us), but ~95-104
     us inside the chunk (host caches cold between calls, cf. DS41-131). A cheaper launch needs one mesh program for
     the 32 chips (DS41-225's table pattern: per-chip fabric channels and sender groups from a common-arg table) or
     folding the planner into the combine (races on G[0] and slot g's header zeroing, shared fabric links); both are
     fabric rewrites with hang risk, not done with the owner away. Expected ~-50 us a call, ~-2 ms a chunk.
- Bit-identity: same device programs (the combine's hash has no fill-call field; the planner's differs only in
  RINGPG), same plans: prev_plans_identical 1 (every layer's receivers / kept / new / dropped / loads, chunks 0..10,
  both prompts), every chunk's last-layer output = off's (maxdiff 0), repeat 0, host mirror 0 mismatches of 477 checks
  in both modes.
- Results (gate unit P.60#test_moe_prev_replicas, PASS, 422 s; one process, the model loaded once, sides switched in
  it: off, device (the setting's), legacy; chunks 1..10 mean of fresh back-to-back requests; host = first host call
  to the last layer's enqueue; clean wall = to the sync after the last layer (diff and final norm untimed); device =
  real-time profiler, each program's slowest chip, summed, a separate pass; EXPERT_X_PIECES=min8k default):

  | prompt | side | host ms | clean wall ms | device ms | experts ms | combine us a call | planner us a call |
  |---|---|---|---|---|---|---|---|
  | novel | off | 164.7 | 181.0 | 166.97 | 46.43 | 339.6 | - |
  | novel | prev legacy | 188.1 | 189.1 | 155.08 | 43.14 | 689.9 | 92.1 |
  | novel | prev device | 172.4 (-15.7) | 179.3 (-9.8) | 152.88 (-2.2) | 40.91 | 360.3 | 95.0 |
  | code | off | 162.3 | 172.2 | 155.57 | 41.50 | 342.2 | - |
  | code | prev legacy | 182.7 | 183.5 | 151.88 | 40.96 | 684.7 | 91.1 |
  | code | prev device | 170.1 (-12.6) | 175.0 (-8.5) | 149.24 (-2.6) | 38.86 | 364.4 | 95.7 |

  The guard + fill args (`before`) 5.9-6.0 us a call on both. Per-chunk host / wall in results/P.60_prev_replicas.json
  (single chunks vary +-10 ms; novel device c10 212 ms host is one outlier). The device -2.2 / -2.6 ms (experts phase)
  is one profiled pass a side: plausibly the readers' removed landed-word polls, not proven.
- Reading: expected host -12 to -16 ms: met (-15.7 / -12.6). The chunk is still host-bound with prev on (host 172 vs
  device 153 novel), so the wall gains less than the host (-9.8 / -8.5); prev now beats off on the clean wall for novel
  (179.3 vs 181.0) and is ~3 ms behind on code (175.0 vs 172.2). What is left of P.56's host: planner ~3.8 ms + the
  combine's ~0.8 ms over off. Trace (P.61): the combine's fill args are now the same every call (no host value), one
  of DS41-256's blockers gone; the guard stays on the host (it decides per layer, ready on steady chunks).
- Cores: none added (rule 18 unchanged). Prepared: the planner and reset stay prepared (OV-23); the combine is a C++ op.
- Re-run: `gate P.60 --spec <spec> --check --unit 'P.60#test_moe_prev_replicas'` (~7 min, devjob); host probe
  `PROBE_VARIANTS=off,legacy,device [PROBE_PLANNER=1] scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_moe_prev_host_probe.py` (~1 min, 4 layers).
- `measure --task P.60` (change on: MOE_REPLICAS=prev MOE_FILL=piggyback MOE_PREV_HOST=device ENGRAM_TABLE=resident):
  determinism 0 (novel, code, state); quick min layer PCC 0.60548579 at L39 (= P.52-P.59: the same output bytes), host
  transfers per layer 0, wall 168.2 ms (spread 6.2; P.56's prev measure 182.3); p51200 (random ids, in-sample plan,
  DS41-228) device_ms_total 157.56, device_ms_experts 76.62, wall 177.7 (spread 6.5; P.56: 188.4), host overhead 20.1,
  1656 launches (the planner's run_prepared not counted).

## P.61 (perf, attempt 1): MOE_TRACE=moe, the MoE half of every layer replayed from a Metal trace
- Expected per chunk (written before any device run; perf_guide 1, 3; P.57 / P.60 numbers, MOE_REPLICAS=prev,
  MOE_PREV_HOST=device): host -80 to -90 ms (P.57: the MoE / FFN half ~90 ms of host a chunk - experts 64.7,
  ffn_residual 27.3 incl. Python; a replay ~0.02-0.05 ms a layer + the guard / planner bookkeeping ~0.01 ms), so host
  ~172 -> ~85-95 ms (novel). Device -2 to -8 ms (P.57's one-layer prototype: 3.93 -> 3.54 ms a layer with the whole
  layer traced; the MoE half alone gets a part of that: fewer dispatch gaps between its ~15 programs). Wall: the chunk
  turns device-bound: novel prev 179.3 -> ~155-160 (-20 to -25 ms), code 175.0 -> ~150-155; prev off 181 -> ~165-172
  (it was device-bound already: only the device cut and the start-of-chunk idle). At 51,200 (P.60 measure: wall 177.7,
  device 157.6) -> ~158-162.
- What I built (MOE_TRACE = off | moe, tt/settings.py; default off on this branch until accepted, per the task):
  tt/moe_trace.py `SegmentTrace` on every main-model block (tt/model.py `TtDS41Block._segment`): ffn_norm only notes
  its input, router .. moe_add do nothing, ffn_residual runs the whole MoE half (ffn_norm, the MoE block call: head,
  dispatch, experts + combine, tilize + add, P.56's planner; ffn_residual's mhc_post with the next layer's attn_hc
  row) - eager for the first 2 calls of a key, then captured once (~25 ms a layer; layer 0's first 186 ms), then
  replayed with execute_trace every call. Inputs: attn_residual's mhc_post writes the new streams into one load-time
  buffer (all layers) and ffn_hc's hc into one a layer (TtHcResidual.fused out= / hc_out=, mhc_post output_tensor /
  hc_output); the head's x is already in the FFN_FEED buffer: no copy before a replay. Kept outside the trace each
  call: P.56's guard (`TtMoEBlockMainLayer.pre_call`: a host decision on host ordinals; its reset copy of the router
  bias is an eager op, issued only on a request's first chunk) and the planner's ordinals / MOE_COUNTS registry
  (`post_call_host`, `PrevReplicas.after_host`). Only the steady path traces (OWN_ROWS, FFN_FEED, POST_COLLAPSE,
  MOE_BLOCK_SRC=main with its own head; MOE_REPLICAS off or prev with MOE_PREV_HOST=device); anything else, the
  decoder tail and DEEP_LAYERS=tail's source layer run eager. No device_params change: with trace_region_size 0 Metal
  puts the trace's command buffer top-down in DRAM (tt_metal/distributed/mesh_trace.cpp:65).
- Decisions: (1) DS41-260: every DRAM buffer the capture allocates stays allocated while the trace lives (frees held,
  op outputs kept, a graph capture raises on a DRAM buffer allocated and freed inside the capture): 9.66 MiB a chip a
  layer, ~386 MiB for 40 layers a key (the gate's prev test has 2 keys: off and prev). L1 (the block's arena) is freed
  as in the eager call. (2) DS41-261: a capture first waits for the engram worker's uploads. (3) DS41-262: the
  rt profiler's accounting of replays (testing/profiler.py trace_replay / remap_replays), so device_ms_total and the
  sections stay right with the change on. (4) P.56's guard stays a host decision outside the trace (the brief said
  "device-side"): it issues nothing on a steady chunk, so it is no blocker and needs no new device program.
- Results (tests/test_moe_trace.py, one process, 40 layers, the model loaded once with MOE_TRACE=moe; untraced = the
  same model with every SegmentTrace.enabled False; chunks 1..10 mean of fresh requests of 11 chunks; host = first host
  call to the last layer's enqueue; clean wall = to the closing sync, nothing profiled; device = rt profiler, a separate
  pass, each program's slowest chip summed; moe host = host time inside the MoE halves):

  | prompt / side | host ms | clean wall ms | device ms | moe host ms |
  |---|---|---|---|---|
  | novel prev, untraced -> traced | 164.9 -> 114.3 | 173.0 -> 166.9 (-6.1) | 155.4 -> 150.0 | 56.0 -> 2.4 |
  | novel off | 159.8 -> 117.0 | 176.4 -> 174.6 (-1.9) | 168.2 -> 164.8 | 51.4 -> 2.0 |
  | code prev | 167.8 -> 117.7 | 173.5 -> 161.8 (-11.7) | 150.6 -> 146.2 | 56.5 -> 2.7 |
  | code off | 163.4 -> 114.2 | 173.1 -> 166.7 (-6.4) | 157.0 -> 154.6 | 52.0 -> 2.3 |
  | p51200 prev (random ids, median of 5) | 168.4 -> 121.5 | 179.1 (spread 54.9) -> 173.1 (spread 9.5) | 165.5 -> 158.9 | |

  Every traced chunk's output equal to untraced bit for bit (all 4 passes, 2 traced clean passes each, and p51200).
  Captures 80 (40 layers x prev / off), median 24.6 ms, max 186 ms (the first); 7240 replays in the run, no error.
- Reading: host -47 to -51 ms a chunk (expected -80 to -90: P.57's "~90 ms MoE half" counted more than the half; the
  half's own host was 51-56 ms, now 2-3). The chunk is no longer host-bound (host ~115 vs device ~150), so the wall
  gains only the device cut and the host-bound spans: -2 to -12 ms (expected -20 to -25 prev). The device itself got
  2-5 ms faster (fewer late-dispatch waits in the CCL programs). Wall - device is still ~15-17 ms: a diagnostic pass
  (P61_ONLY=novel:prev, the rt timeline per chip) puts only ~2.7 ms a chunk at the traces (1.8 ms of idle before the 40
  replays - Metal drains the device before each one, tt_metal/impl/trace/dispatch.cpp - 0.26 after, 0.59 inside); the
  rest is eager idle (the chunk's start before the device has work, layer 1's wait for the engram job), which the
  profiled pass inflates and cannot size.
- Re-run: `scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_moe_trace.py`
  (~12 min with the load, devjob); P61_QUICK=1 (4 layers, ~1.5 min); P61_ONLY=novel:prev (one pass set + the gaps).
- `measure --task P.61` (change on: MOE_TRACE=moe ENGRAM_TABLE=resident): determinism 0 (novel, code, state); quick
  min layer PCC 0.60548579 at L39 (= P.52-P.60: the same output bytes), host transfers per layer 0, clean wall 163.7 ms
  (spread 5.3; P.60's measure 168.2); p51200 (random ids) device_ms_total 149.24 (P.60 157.56), device_ms_experts 69.13
  (76.62), ffn_residual 9.89, wall 178.6 with a 26.2 ms spread (P.60 177.7, spread 6.5: this single run is noisy; the
  test's 5-run medians above give 179.1 -> 173.1), host launches 1296 (1656), host overhead 29.4. The replays' device
  time lands in its sections (profiler.trace_replay): experts and ffn_residual are filled, not 0.
- Gate check: run with DS41_MOE_TRACE=moe exported (the setting's default is off on this branch; the orchestrator's
  A/B sets both sides).
- `gate P.61 --check` (DS41_MOE_TRACE=moe): every unit PASS - determinism 0, the ladder's last rung (every err_layer /
  err_state within its noise floor, err_final_hidden 0.132 <= 0.242, top5 1.0), test_moe_prev_replicas (device mode:
  prev = off bit for bit, repeat 0, mirror 0 mismatches of 477, novel device 164.8 -> 150.3, wall 178.4 -> 166.5; its
  per-op experts / planner columns read 0 under the trace: its spy times the launches, which a replay does not make),
  the profile (chunk_wall_ms 164.2, device_ms_total 141.52).
- Next (not done): the remaining wall - device (~15 ms a chunk) is eager idle at the chunk's start and layer 1's
  engram wait (DS41-256 fix 4: the engram job one chunk ahead), and ~2.7 ms at the 40 replay boundaries (Metal's drain
  before each replay: fewer, longer traces need the attention's start-dependent arguments on the device).

## P.62 (perf, attempt 1): why prev costs host over off, accounted call by call; MOE_PREV_HOST2 legacy | lean
- Expected per chunk (written before the first measurement of the change; perf_guide 3; real novel / code chunks 1..10,
  MOE_REPLICAS=prev MOE_PREV_HOST=device on both A/B sides): device ~0 (+-0.3 ms: the planner's program is the same
  work, its senders read a table instead of their args); host -2 to -3 ms of the ~7.7 ms prev adds over off: the
  planner launch ~95 -> ~45-55 us a call as ONE program for the 32 chips (DS41-225: 336 -> 110 us for a feed op) =
  -1.6 to -2.0 ms; the combine's fill args set once instead of a C++ call a layer (~5 us) = -0.2 ms. The rest of the
  7.7 ms is not attributed yet (STEP 1 measures it); the target (prev <= off + 1 ms) needs that part too.
- Terms: host = first host call of a chunk to the last layer's enqueue (includes the engram worker waits at layers 1
  and 14); busy = host - those waits (the main thread's own work); wall = to the sync after the last layer; device =
  rt profiler, each program's slowest chip summed. prev = MOE_REPLICAS=prev; off = prev switched off in the same model.
  legacy / device / lean = the prev host paths P.56 / P.60 / P.62 (MOE_PREV_HOST, MOE_PREV_HOST2). All means: chunks
  1..10 of fresh requests, novel and code prompts, 8x4.
- STEP 1, accounting (tests/test_prev_host_account.py, new, diagnostic: one load, sides alternating, 10 clean rounds
  a side for host / busy, 2 instrumented rounds for per-op exclusive host (P.57's HostClock), 1 cProfile round).
  prev - off, us a chunk:

  | term | novel legacy | novel lean | code legacy | code lean |
  |---|---|---|---|---|
  | planner launch (run_prepared, 40 a chunk) | +4083 | +3772 | +4080 | +3735 |
  | routed_experts_combine launch (fill path) | +1521 | +1078 | +1333 | +1504 |
  | fused_head launch | +354 | +156 | +499 | +535 |
  | PrevReplicas.before Python (guard + fill args) | +266 | +76 | +272 | +83 |
  | PrevReplicas.after Python (instrumented, wrapper included) | +443 | +435 | +446 | +439 |
  | sum | +6667 | +5517 | +6630 | +6296 |
  | clean busy difference (10 rounds, +- standard error) | +6701 +-1484 | +5289 +-1159 | +6231 +-873 | +6555 +-1023 |
  | residual (busy - sum) | +34 | -228 | -399 | +259 |

  Every other op's difference changes sign between runs (+-0.1..0.8 ms each, both directions: noise). The engram
  wait in host differs by -1.9..+0.5 ms between sides (clean) and -2..-3.4 ms (instrumented: a slower main thread
  waits less at layer 14). P.60's "~2.5 ms not attributed" was this noise (one pass a side; a pass's mean swings +-4
  ms, DS41-264). So prev's host over off = one extra 32-chip launch a MoE layer (~93-102 us) + the combine's fill path
  (~15-38 us a call) + ~5 us Python a call; nothing else.
- STEP 2, what lean (MOE_PREV_HOST2=lean, default) changes, bit-identical (same programs' work, same plans):
  (1) planner and reset as ONE mesh program (kernel compile arg 21 MESH1: the chip's position from the routing
  table's my_device_id and a byte table; the senders' targets from a 2-bit first-hop table and their eth channel
  overwritten into the connection's first runtime arg, DS41-225's pattern; tt/expert_prev.py _mesh_tables /
  _planner_desc_mesh); (2) the combine's fill args set once (they are the same every call under MOE_PREV_HOST=device;
  fill_off() / set_mode() re-arm); (3) after() without per-call settings reads. Gate unit (one pass a side) host us a
  call lean vs device: planner 85.3-85.5 vs 91.5-92.2, before 2.0-2.1 vs 6.7; combine 352-354 vs 352-357 (off
  336-341): lean saves ~11 us a call = ~0.45 ms a chunk. Expected -2 to -3 ms: missed, because one mesh program costs
  the same to launch as 32 (DS41-263: on a prepared hit only per-device dispatch is left).
- Not done, with the reason:
  - Fold the planner into the fused head: the head drives the fabric on every link (x all-gather, reduce-scatter,
    routing exchange); the planner's senders need their own connection per direction on the same EDM channels.
  - Fold it into the combine: the combine holds 110 cores (rule 18 cap) and its port cores are the fabric workers; it
    needs the counts packets inside combine_port_writer.cpp (31 x 64 B a chip, after the main loop, link-0 ports) and
    the planner core in a program that already runs (the next layer's head; it needs no fabric), with the G[0] /
    ring-slot order kept. A fork rewrite in two C++ ops with hang risk: the next step if the owner wants prev's host
    <= off + 1 ms (saves ~3.7 ms a chunk; the combine fill path's ~0.6-1.5 ms is the other part).
  - Guard device-side: on steady chunks it issues no device work; its Python is 2.0 us a call under lean (0.08 ms a
    chunk). A device guard would cost a read in the head; nothing to gain.
  - Combine fill-path host trims (set_fill builds a 10-word signature per program per call on 32 programs; the fill
    words' TensorSpec per call): estimated <= 0.3 ms a chunk, below the noise; not worth a fork rebuild here.
- Gate unit P.62#test_moe_prev_replicas (PASS; one pass a side, so host/wall carry +-4 ms noise):

  | prompt | side | host ms | wall ms | device ms | experts ms |
  |---|---|---|---|---|---|
  | novel | off | 170.7 | 186.2 | 167.43 | 46.42 |
  | novel | prev lean (P.62) | 168.5 | 176.0 | 153.87 | 41.18 |
  | novel | prev device (P.60) | 172.2 | 179.4 | 152.18 | 40.70 |
  | novel | prev legacy (P.56) | 180.5 | 181.7 | 155.02 | 42.99 |
  | code | off | 164.5 | 174.7 | 157.46 | 42.26 |
  | code | prev lean | 167.0 | 172.9 | 148.43 | 38.36 |
  | code | prev device | 167.4 | 172.8 | 148.66 | 38.44 |
  | code | prev legacy | 181.9 | 182.8 | 151.39 | 40.67 |

  maxdiff 0 every mode, repeat 0, mirror 0 mismatches, plans identical across the three host paths (both prompts),
  ready 1.0. Planner device ms a chunk 3.76 / 4.96 (novel / code), unchanged by MESH1.
- Cores (rule 18): planner 1 + up to 4 sender cores a chip (cores (0,0), (1..4,0)), as before. Prepared (OV-23): the
  planner and the reset stay prepared programs; no other Python-built op touched.
- Gotcha: never edit a kernel the running job has not compiled yet: the first accounting run read the kernel at its
  first prev call after my MESH1 edit (22 compile args expected, 21 given) and failed to compile (no device harm).
- Re-run: accounting `P62_SIDES=off,prev:legacy,prev:lean scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_prev_host_account.py` (~11 min with the load, devjob); gate unit
  `gate P.62 --spec <spec> --check --unit 'P.62#test_moe_prev_replicas'`; quick `P56_QUICK=1` on that test (~3 min).
- `gate P.62 --check`: PASS, every unit (determinism 0; ladder rung last: every err_layer / err_state within its noise
  floor, err_final_hidden 0.132 <= 0.242, top5 1.0; test_moe_prev_replicas as above; profile chunk_wall_ms 170.1,
  device_ms_total 151.4).
- `measure --task P.62` (change on: MOE_PREV_HOST2=lean ENGRAM_TABLE=resident): determinism 0 (novel, code, state);
  quick min layer PCC 0.60548579 at L39 (= P.52-P.61: the same output bytes), host transfers per layer 0, clean wall
  171.3 ms (spread 8.2); p51200 (random ids, DS41-228) device_ms_total 156.49, device_ms_experts 75.99, wall 177.6
  (spread 42.6: two runs of 10 at 207.7 / 215.9 ms, loadavg 11.8; the other eight 173.3-182.2), host overhead 21.1,
  1656 launches (run_prepared not counted), 0 new programs.

## P.62b (perf, attempt 1): P.56's planner folded into programs that already run (MOE_PLANNER launch | folded)
- Expected per chunk (written before any code change; perf_guide 3 and 4 "moved time"; real novel / code chunks 1..10,
  MOE_REPLICAS=prev on both sides): host -3.5 to -3.9 ms (P.62's accounting: the planner launch +3.7-3.8 ms a chunk,
  40 x ~85-95 us, removed; plus ~+0.1-0.3 ms back: one more patched head arg a call, the folded tail's desc). Device
  -2 to -4 ms (the planner program's ~0.1 ms a layer, 3.8-5.0 ms a chunk, is mostly its launch / exchange skew; the
  plan's own ~30-40 us runs on the head's watcher RISC after its work, hidden while the head's other cores run; the
  count packets ride the combine's ports after their rows: +~1-3 us a call). Target: prev host <= off host + 1.5 ms.
- Design (before code): the head has no free core on the 8x4 (11 x 10 grid at the 110 cap: 10 x 10 matmul cores + the
  mover column, both data-movement RISCs taken), so the planner is a tail of the head's watcher kernel (NCRISC, after
  its watch and Y send), its L1 the arena past the head's own bytes (head 948 KB of a 1386 KB arena). The count packets
  go out from the combine's link-0 port senders after their main loop (one chip port also writes the chip's own row
  locally, tagged with the call ordinal + 1 so the next head knows a plan is pending).
- Built (MOE_PLANNER = launch | folded, tt/settings.py, default folded; folded needs MOE_PREV_HOST=device +
  MOE_PREV_HOST2=lean, the OWN_ROWS head path and >= 2 MoE layers a chunk, else that call runs launch):
  1. Fork routed_experts_combine (`set_combine_fill_exact(..., prev_counts, prev_landing, prev_sem, prev_spc,
     prev_core)`, fill args 12..17, in the fill signature): after the main loop and the fill drain each link-0 port
     reads the chip's counts row (64 B-aligned window into a free staging slot) and sends it to the chips its first hop
     leads to (fused write into landing row (G[0] & 1) x chips + me, atomic increment of the arrival word on the
     watcher core); the chip's first port also writes its own landing row, last word = G[0] + 1 (the tag).
  2. Fork fused_head (`tail=HeadTail(ct, l1)`, `tail_desc=`; `fused_head_tail_core`): the watcher kernel
     (xg_watcher.cpp, NCRISC) runs `kernels/prev_planner.hpp` after its work; L1 = arena + al64(head arena bytes);
     the pending layer's addresses from a 64 B descriptor a layer (SRC_TAIL, patched per call). The planner's body
     moved into that header (tt/kernels/moe_prev_planner.cpp ROLE 0 includes it: one copy of the arithmetic). Folded
     input (counts arg 0): own row from the landing rows, plan only if its tag == G[0] + 1, bounded count wait.
  3. tt/expert_prev.py: `head_tail()` (kwargs for the head: the pending call's descriptor), `on_head()` (fill args
     with the prev counts, once), after() books the ordinal and leaves the call pending; the block's last MoE layer
     plans at once with the standalone folded planner (`_plan_now`: one prepared mesh program on the watcher cell, no
     senders) so nothing is pending at a chunk's end; `flush()` at reset / fill_off / set_mode / a non-folded call.
     Arrival word: a global semaphore on the watcher cell (L1_SMALL). Adapter: `_fold_ok`, the head kwargs; MOE_TRACE
     does not trace folded calls (the last layer's plan is an eager launch after() makes).
- Cores (rule 18): none added. Count packets on the combine's existing port cores; the tail on the head's existing
  watcher core; the standalone folded planner 1 core a chip (the watcher cell), once a chunk.
- Results (tests/test_prev_host_account.py, P62_SIDES=off,prev:lean:launch,prev:lean:folded, 10 alternating clean
  rounds a side, chunks 1..10 of fresh requests; host = to the last enqueue, busy = host - engram wait, wall = to the
  sync; the test's prev:lean side fixed first, DS41-267):

  | prompt | side | host ms | busy ms | wall ms | prev - off host (busy +- se) |
  |---|---|---|---|---|---|
  | novel | off | 163.43 | 151.60 | 177.29 | |
  | novel | prev lean launch (P.62) | 169.95 | 158.70 | 175.84 | +6.53 (+7.11 +- 1.20) |
  | novel | prev lean folded (P.62b) | 164.42 | 152.89 | 170.92 | +0.99 (+1.30 +- 1.00) |
  | code | off | 165.43 | 154.37 | 174.94 | |
  | code | prev lean launch | 168.65 | 158.21 | 173.68 | +3.22 (+3.84 +- 1.02) |
  | code | prev lean folded | 165.40 | 154.13 | 170.67 | -0.03 (-0.24 +- 1.23) |

  Target prev host <= off host + 1.5 ms: met (novel +1.0, code 0.0). Folded's identified host over off (instrumented,
  novel, us a chunk): combine launch +563, PrevReplicas.after +180 (wrapper incl.), fused_head +167 (the tail's
  patched arg), the one standalone plan +151, before +101; launch's planner run_prepared +4057 is gone.
- Device (gate unit P.62b#test_moe_prev_replicas, one profiled pass a side, the setting's mode first = lean folded):
  novel off 166.82 -> folded 150.11 ms (P.60 device path 152.45, legacy 155.07); code 156.95 -> 145.75 (149.38,
  152.23); planner program 0.04 / 0.08 ms a chunk (was 3.7 / 4.9). The tail's plan ~31k cycles + map ~7k, its count
  wait ~1.8k (quick model; the launch path's wait 622k = launch skew). Every check: maxdiff 0, repeat 0, mirror 0
  mismatches of 477 (both prompts, three paths), plans identical across legacy / device / lean-folded.
- `measure --task P.62b` (change on: MOE_PLANNER=folded ENGRAM_TABLE=resident): determinism 0 (novel, code, state);
  quick min layer PCC 0.60548579 at L39 (= P.52-P.62), host transfers per layer 0, clean wall 167.1 ms (spread 5.5;
  P.62's measure 171.3); p51200 (random ids) device_ms_total 155.65 (P.62 156.49), device_ms_experts 74.59 (75.99),
  wall 176.0 (spread 50.7, loadavg 7-9), host overhead 20.4, 1656 launches (run_prepared not counted).
- Gotchas: DS41-265 (the head's patch store is per variant: the first tail run planned layer 0 at the wrong ordinals
  because the tail's descriptor arg was never patched); DS41-266 (no free head core); DS41-267 (P.62's accounting
  prev:lean side was P.60's device path). Fork suites not run (OV-25).
- Re-run: accounting `P62_SIDES=off,prev:lean:launch,prev:lean:folded scripts/run_safe_pytest.sh --run-all
  --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_prev_host_account.py` (~8 min, devjob); gate unit
  `gate P.62b --spec <spec> --check --unit 'P.62b#test_moe_prev_replicas'` (~8 min); quick `P56_QUICK=1` on that test
  (~1.5 min).

## P.63 (perf, attempt 1): the fused head's and the combine's cache-hit patching (MESH_PROGRAMS legacy | on)
- Expected per chunk (written after the 1-layer probe OV-43 asked for, before any A/B or gate run; real novel / code
  chunks 1..10, 40 MoE layers): host -5 to -7 ms (combine ~-115 us and head ~-40 us a call, x 40); device 0 (the same
  programs and the same runtime-arg values: host only); wall: as much of the host cut as the chunk is host-bound
  (P.62b: host ~164-170 ms vs device ~146-150 ms, so most of it). Launches a chunk unchanged (0).
- Terms: host = first host call of a chunk to the last layer's enqueue; wall = to the sync after the last layer;
  device = rt profiler, each program's slowest chip summed; us a call = host time of one op launch (Python wrapper
  inclusive); prev / off = MOE_REPLICAS=prev on / switched off in the same model; legacy / on = MESH_PROGRAMS.
- STEP 1, the probe OV-43 asked for (tests/test_mesh_program_probe.py, new, diagnostic; 4 layers, novel chunks 1..3;
  temporary C++ timers in the fork, removed): combine ~470 us a call = hit patching ~250 (the flat expert's 766 address
  args a chip, 6.4 us a chip; the combine's own 132 args 0.3 us) + hash 4 + Metal's enqueue of 32 programs ~215; head
  ~340 us = patching 3.6 us x 32 chips (~2,640 args a chip, the moved groups written) + ~220 enqueue. One patch
  ~8 ns = GetRuntimeArgs (get_kernel's two hash lookups + a shared_ptr copy). Metal's enqueue already packs a
  multi-device program once and copies it per device (fd_mesh_command_queue.cpp write_program_commands_to_devices),
  so a mesh program saves the patch loop's x32, not the per-device copies (DS41-263).
- Decision: not ONE mesh program (the task's literal ask). The probe says the cost a mesh program removes is the patch
  loop; caching each patched arg's RuntimeArgsData* per program removes most of it with no kernel change, no fabric
  rewrite of two 110-core ops and no hang risk. What a mesh program would still add on top: the remaining ~3 us
  (combine) / ~2 us (head) a chip of patching = ~150 us a MoE layer, ~6 ms a chunk, needing DS41-225 tables in the
  head's and the combine's fabric kernels (head_program.cpp 3,100 lines, combine_program.cpp 1,000): a separate pick.
- Built (MESH_PROGRAMS = legacy | on, tt/settings.py, default on; fork option cached_slots on fused_head and
  routed_experts_combine, default False): HeadLastAddresses::slots, FlatProgramHandles::slots, CombineHandles::slots
  (one RuntimeArgsData* a patch, filled on the program's first hit) and the combine's Program* a chip
  (shared_variables_t::program). Same values written: programs, runtime args and outputs bit-identical.
- Probe, no timers (3 alternating rounds x 4 passes a side, median us a call): combine 468.2 -> 351.9, head 342.0 ->
  300.6 (sparse_dispatch_2d 102 / 104, unchanged): -157 us a MoE layer, ~-6.3 ms host a chunk at 40 layers.
- Gate unit P.63#test_moe_prev_replicas (PASS, change on, one pass a side; prev = lean folded): combine us a call
  245.7 / 249.4 (novel / code; P.62 lean 352-354, P.60 device in this run 250.6 / 251.6 - its own hit also on). prev
  chunks 1..10: novel host 155.6 / wall 166.5 / device 149.85 ms; code 156.3 / 165.2 / 145.16 (P.62b's gate unit:
  host ~164-170). off: code host 151.0 / wall 166.8 / device 155.9; novel off host 242.6 / wall 260.2 is a slow side
  (chunk 1 already 163 ms, most chunks ~250: host load, loadavg ~12; one pass a side, DS41-264). maxdiff 0, mirror 0
  mismatches of 477, plans as P.62b.
- `measure --task P.63` parts (change on: MESH_PROGRAMS=on ENGRAM_TABLE=resident): determinism 0 (novel, code,
  state); quick min layer PCC 0.60548579 at L39 (= P.52-P.62b: the same output bytes), host transfers per layer 0,
  clean wall 166.6 ms (spread 22.5; P.62b 167.1); p51200 (random ids) wall 175.2 (spread 7.7, loadavg 11.8; P.62b
  176.0), device_ms_total 155.07 (155.65), device_ms_experts 74.07, host overhead 20.1 (20.4), 1656 launches (same),
  0 new programs. At p51200 the chunk is device-bound (wall = device + ~20), so the host cut shows little there.
- Expected vs got: host -5 to -7 ms a chunk expected; the per-call numbers give ~-6.3 ms; the A/B (orchestrator) gives
  the chunk numbers. Launches a chunk: unchanged (host-only change). Cores (rule 18): none changed. Prepared (OV-23):
  no Python-built op touched (both ops are C++ device ops). Fork suites not run (OV-25).
- Re-run: probe `PROBE_SIDES=legacy,on,legacy,on scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_mesh_program_probe.py -s` (~2 min); gate unit `gate P.63 --spec
  <spec> --check --unit 'P.63#test_moe_prev_replicas'` (~11 min, devjob).

## P.63b (perf, attempt 1): one program for the whole mesh, fused head and combine (MESH_PROGRAMS2 legacy | on) - HUNG, stopped
- Expected per chunk (the estimate I worked from before writing code; this section itself was written at the end,
  after the hang): host -5 to -7 ms (40 MoE layers x ~150 us: the head's ~2 us and the combine's ~3 us a chip of
  cache-hit patching after P.63, x31 chips saved, plus the framework's per-program hit work); device +0.1 to +0.2 ms
  (every head / combine kernel reads its table row from DRAM before it starts, ~1-2 us an op); wall: most of the host
  cut while the chunk is host-bound (P.63: host ~156 ms vs device ~145-150 ms).
- Terms: per-chip program = today's 32 programs a launch, one per chip; mesh program = ONE program every chip runs
  with identical runtime args; table = a DRAM buffer (the same on every chip) holding, per chip, what differs; a chip
  finds its row by its fabric chip id (my_device_id in the routing table in its L1, DS41-225's pattern). us a call =
  host time of one op launch.
- STEP 1, what differs between the 32 programs (DS41_MESH_DIFF=<prefix>, diagnostic, 2-layer probe; DS41-269):
  head - kernels, cores, CBs, semaphores, compile args the same; 4 defines carry the chip index; 4,280 per-core args a
  chip differ (compute kernels included, lists of chip-dependent length); patched addresses at chip-dependent indices.
  combine - port and accumulator cores differ by chip (placed next to each chip's Ethernet cores), so CBs and
  semaphores differ; 3,047 args a chip differ. And each launch copies its args to every chip: ~83 KB (head) / ~79 KB
  (combine) a chip (DS41-270) - the ~220 us "enqueue" a mesh program keeps (DS41-263), and why per-chip data must not
  go into the args.
- Built (head only): fork option `fused_head(mesh_program=)`, hashed; model setting MESH_PROGRAMS2 (tt/settings.py).
  common/mesh_uniform.cpp assembles the mesh program from the per-chip descriptors (args padded to each kernel's
  longest - free, the dispatcher sends max length; differing defines -> args; per chip and slot a delta, absolute, or
  anchor + offset entry; anchors = extra args the head patches per call) and checks on the host that every chip's args
  are rebuilt exactly from the mesh args and its rows. common/device/kernels/mesh_overlay.hpp: data-movement kernels
  apply their row first thing; a compute kernel's entries are applied by a data-movement kernel on its core into the
  TRISC args, released by a flag. Head: 220 units, 4,599 entries (chip 1), 138 anchors, table 6.76 MB, 26,113 args a
  launch (legacy ~20.7K: +6 % bytes copied). ra_plan.hpp / route_words.hpp read the chip index at run time.
- Runs (8x4, tests/test_mesh_program_probe.py, 1 layer, novel chunks 0..3): (1) before the anchors / compute
  entries the assembly refused and fell back to per-chip programs: outputs bit-identical to P.63 (on vs mesh side);
  (2) with them the assembly passed its host check, and the FIRST mesh head launch HUNG: dispatch timeout
  (metal_context.cpp:806), PCIe MMIO timeouts, triage could not attach; the safe runner reset and recovered the box
  ("All Detected Links are healthy"). I did not trigger the reset. Owner safety rule: no device run after that.
- Not measured: host us a call, host / device / wall per chunk on novel and code chunks 1..10, the 51,200 position -
  nothing ran in mesh mode. The P.63 numbers stand.
- Not done: the combine's mesh program (needs a uniform port / accumulator placement option in port_placement.cpp /
  combine_program.cpp first; the overlay machinery is generic and would carry its args, but its flat expert and combine
  builders create Programs, not descriptors, so the assembly needs a Program-path variant and a define plumbed into
  their kernel creation).
- State left for the orchestrator: MESH_PROGRAMS2 default legacy (the gate's frozen tests then run P.63's path,
  unchanged); MESH_PROGRAMS2=on raises at model load (a clean error, not another hang and reset). The A/B "change on"
  side (MESH_PROGRAMS2=on) will therefore fail at load: do not expect a measurement from this attempt.
- Next (for whoever picks it up): find the hang off the 8x4 first (2x2 or loudbox with TT_METAL_WATCHER, or slow
  dispatch); suspects in DS41-271's detail (compute flag release, the row's NOC read, desc decoding). Before more
  work, weigh it: DS41-270 says the args' bytes copied 32x (~2.6 MB a head call) are the bigger host cost, and a mesh
  program keeps them; fewer arg bytes (common args, a DRAM arg table per layer) would cut host time for both paths.
- Cores (rule 18): none changed (the same kernels on the same cores). Prepared (OV-23): no Python-built op touched.
  Fork suites not run (OV-25). Gate not run (stopped after the hang).
- Re-run (do NOT run the mesh side on the 8x4 until the cause is fixed): diff `DS41_MESH_DIFF=/tmp/p63b/v
  PROBE_LAYERS=2 PROBE_PASSES=1 PROBE_SIDES=on PROBE_COMPARE=0 scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_mesh_program_probe.py -s` (~1 min); mesh side needs
  PROBE_ALLOW_MESH=1 and PROBE_SIDES=on,mesh.

> Overseer, 2026-11-14 15:45: P.63b was STOPPED and its code REVERTED (mesh head hung the 8x4 on first launch; DS41-269/270: small payoff for a large rewrite). Its diff is kept in /scratch/dnijemcevic/replication/p63b/. The entries DS41-269..271 stay as findings.

## P.64 (perf, attempt 1): fewer attention-side launches - the window write as one prepared program (ATTN_FUSE legacy | on)
- Census first (tests/test_attn_launch_census.py, new, diagnostic; 8 layers 0,1,2,3,14,20,21,24, novel chunk 3, legacy):
  every layer's attention issues the same window-write group (WINDOW_KV_FORMAT=bf16_tiles, WINDOW_KV_KEEP=snapshot128,
  tt/attention._window_region): untilize of the chip's own rows (cores.to_rm, ~58 us), the ring write
  (WindowRing.put: 3 x ttnn.slice ~57 us + 3 x ttnn.bringup.slice_write ~33 us), the region write (_put_rows:
  slice_write ~75 us) and the neighbour send (WindowSend, run_prepared ~100 us): 9 launches, ~520 us host a layer.
  Reuse layers (30 of 40): 16 attention launches a layer, 7 of them this group's slices / slice_writes.
- Expected per chunk (written before any code change; perf_guide 3 and 4: launches on a host-bound chunk): the
  region write, the 3 ring slices and the 3 ring slice_writes go into the send's prepared program (a copy kernel on
  the second data-movement RISC of the send's core row; the send reads the carry from the untilized rows instead of
  the region): 9 -> 2 launches a layer, -280 launches a chunk (40 layers); host -10 to -13 ms a chunk (~-320 us a
  layer: -171 slices, -99 slice_writes, -75 region write, +~20-30 us for the bigger prepared call with its value
  slots); device ~0 to -1 ms (7 small programs and their dispatch gaps gone, a ~10-20 us copy on 4 cores beside the
  send); wall: as much of the host cut as the chunk is host-bound (P.63: host ~156 ms vs device ~145-150 ms), so
  ~-5 to -10 ms on real chunks 1..10, ~0 at p51200 (device-bound there). Bit-identical outputs (data movement only).
- Not in this pick (next candidates, from the census): the window snapshot (write_snapshot: 2 high_bw_all_gathers of
  every chip's whole 192-row ring + slice + update_padded_kv_cache, 4 launches ~470 us host a layer, plus the
  gathers' device time) - a fabric broadcast of the owner's 128 rows would replace it; q_a's input typecast; the
  compressor layers' slice / to_layout / mesh_partition (4 layers only).
- Terms: launch = one outermost ttnn op call or one run_prepared; host = first host call of a chunk to the last
  layer's enqueue; wall = to the sync after the last layer (clean: profiler off); device = rt profiler, each program's
  slowest chip summed; us a call = host time of one launch (Python wrapper inclusive); prev / off = MOE_REPLICAS=prev
  on / switched off in the same model; legacy / on = ATTN_FUSE.
- Built (ATTN_FUSE = legacy | on, tt/settings.py, default on):
  - tt/window_send.py `WindowSend.fused(region, tail, rows, ring, pieces)`: one prepared program a call (io [region,
    tail, rows, ring]; the ring pieces 12 value slots, unused ones n = 0). Send kernel = the legacy kernel reading the
    sent pages and the last chip's kept pages from `rows`' last 128 rows (_KERNEL_FUSED, made by text edits of
    _KERNEL, asserted); copy kernel (_COPY_KERNEL) on the send core's NCRISC (COPY_CORES = 1): rows -> region rows
    [128, 128 + q_rows) and -> the ring pieces. Its own fabric connection (setup_fabric_connection on the new program).
  - tt/attention.py: `WindowRing.plan` (put's host bookkeeping + pieces; put = plan + the old ops),
    `TtWindowCache.write_own(defer=True)` returns the pieces (the carry still taken first, the pin's pieces still
    copied by the old ops), `_window_region` with ATTN_FUSE=on, a bf16 region, a ring and > 1 chip: no region
    slice_write, the fused call at step 4 (after the carry's own writes to the kept tail, as the send was).
  - Applies to every box with > 1 chip and q_rows >= 128 (the region path's own condition); 1 chip / bfp8 regions /
    WINDOW_KV_KEEP=full: legacy path unchanged.
- Cores (rule 18): the fused program uses core (0, 0) a chip, both data-movement RISCs (was: the send on (0, 0) +
  slice / slice_write on an 8-core row and the region slice_write on max_grid). Prepared (OV-23): yes, the only
  Python-built op touched. Fork suites not run (OV-25); no fork changed.
- Bits: data movement only. tests/test_window_write_fused.py (new): fused vs legacy ops on random rows, 5 piece
  patterns x 3 layers x copy cores 1/2/4/8, every chip's region, kept tail and ring byte-equal, and equal to the send's
  definition. tests/test_attn_launch_census.py: 8-layer model, novel chunks 0..3, legacy vs on: 0 of 88 differ (every
  chip: layer streams, served snapshots, kept tails).
- Per call (probe, 8x4, "two pieces" case, median): legacy group (region slice_write + 2 x slice + 2 x slice_write +
  send) host 306 us a layer, device 64.5 us (6 programs); fused host 74 us (1 copy core; 81 / 96 / 123 at 2 / 4 / 8),
  device 54 us (1 program). In the model (census): fused run_prepared 86-150 us a call vs the legacy send 65-100 us.
- Launches: census 8 layers 462 -> 406 (-7 a layer at chunk 3: 3 pieces; -5 at chunk 5: 2 pieces, one whole); p51200
  1656 (P.63) -> 1376 a chunk.
- Gate unit P.64#test_moe_prev_replicas (PASS, change on, one pass a side; prev = lean folded), chunks 1..10 mean:

  | prompt | side | host ms | wall ms | device ms | (P.63 gate unit, same side: host / wall / device) |
  |---|---|---|---|---|---|
  | novel | prev | 144.4 | 162.9 | 148.33 | 155.6 / 166.5 / 149.85 |
  | novel | off | 207.4 | 235.2 | 167.71 | slow side again (P.63 242.6 / 260.2; DS41-264) |
  | code | prev | 150.0 | 164.5 | 144.89 | 156.3 / 165.2 / 145.16 |
  | code | off | 146.9 | 169.7 | 156.07 | 151.0 / 166.8 / 155.9 |

  maxdiff 0, repeat 0, mirror 0 mismatches of 477 (both prompts, three host paths). The host is now at or below the
  device time on prev (host 144-150 vs device 145-148 ms), so further host cuts show in the wall only as the device
  shrinks too.
- `measure --task P.64` parts (change on: ATTN_FUSE=on ENGRAM_TABLE=resident): determinism 0 (novel, code, state);
  quick min layer PCC 0.60548579 at L39 (= P.52-P.63: the same output bytes), host transfers per layer 0, clean wall
  161.8 ms (spread 9.7; P.63 166.6); p51200 (random ids) wall 173.7 (spread 22.5; P.63 175.2), device_ms_total 154.93
  (155.07), device_ms_experts 74.17, device_ms_attention 47.69, host overhead 18.8 (20.1), 1376 launches (1656), host
  launch ms 110.6.
- Expected vs got: launches -280 a chunk (exact); host -10 to -13 ms expected; per-call numbers give ~-8 to -9 ms
  (40 layers x ~200-230 us; fewer pieces at some starts); the gate unit's prev host -6 to -11 ms vs P.63's (one pass a
  side, +-4 ms noise); device ~0 (-0.1 to -1.5); wall -1 to -5 ms on real chunks (the chunk is now device-bound on prev).
  The orchestrator's A/B gives the on / off table.
- Next host pick from the census: write_snapshot (2 high_bw_all_gathers of every chip's whole ring + slice +
  update_padded_kv_cache a layer, ~4 launches ~470 us host and the gathers' device time) -> a fabric broadcast of the
  owner chips' 128 rows (e.g. a third kernel in this same fused program); then q_a's input typecast, the compressor
  layers' slice / to_layout / mesh_partition (4 layers).
- Re-run: probe `WW_CORES=1,2,4,8 scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_window_write_fused.py -s` (~30 s); census `CENSUS_SIDES=legacy,on
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_attn_launch_census.py
  -s` (~2.5 min; CENSUS_COMPARE=0 CENSUS_PASSES=3 for host ms); gate unit `gate P.64 --spec <spec> --check --unit
  'P.64#test_moe_prev_replicas'` (~9 min, devjob).

## P.65 (perf, attempt 1): the window snapshot as one prepared fabric broadcast (CHEAP_CCL legacy | on)
- Expected per chunk (written before any code change; perf_guide 3 / 8; OV-45): P.57's census charges
  high_bw_all_gather 92 calls x ~106 us and mhc_post 82 x ~100 us. 80 of the 92 gathers are the window snapshot
  (tt/attention.write_snapshot: 2 gathers of every chip's whole 192-row ring a layer), followed by 1-2 slices, a concat
  and update_padded_kv_cache: 4-6 launches, ~470 us host a layer. mhc_post is a C++ fork op at ~100 us, already at the
  floor of one launch on 32 chips (DS41-263), so only folding it into another program would remove it: not this pick.
  The change: one prepared program a layer (tt/snapshot_send.py) that copies the owner chip's 128 snapshot rows
  straight into every chip's slab over a fabric broadcast tree (1-hop unicasts: the owner's column first, then each
  row), block by block. Host: 4-6 launches -> 1 (~100-130 us prepared), about -350 us a layer, so -12 to -15 ms a chunk
  (40 layers) and -120 to -200 launches a chunk. Device: the two gathers move ~6 MB into each chip a layer; the tree
  moves 128 KB into each chip. Expected 0 to -3 ms a chunk (the slices / concat / update programs gone, the gathers'
  link time replaced by a ~50-80 us tree). Wall: as much of the host cut as the chunk is host-bound (P.64: prev host
  144-150 vs device 145-148 ms, so small on real chunks; more on the off side). Outputs are bit-identical (data movement
  only). Memory: the 6 MiB gather buffer is no longer allocated under on.
- Terms: launch = one outermost ttnn op call or one run_prepared; host = first host call of a chunk to the last
  layer's enqueue; wall = to the sync after the last layer (clean: profiler off); device = rt profiler, each program's
  slowest chip summed; us a call = host time of one launch (Python wrapper inclusive); prev / off = MOE_REPLICAS=prev
  (lean, the shipped path) / switched off in the same model; legacy / on = CHEAP_CCL.
- Built (CHEAP_CCL = legacy | on, tt/settings.py, default on): tt/snapshot_send.py `SnapshotSend` (module docstring:
  tree, home core, port cores); tt/attention.py `WindowTables.snapshot_send()` (None on 1 chip or legacy),
  `TtWindowCache.write_snapshot` builds the 4 blocks (zero / local / owner chip, ring row) exactly as `_snapshot_rows`
  picks them and makes one call; under on the 6 MiB gather buffer is not allocated at load. Programs: one per block
  pattern (steady state: one, the last chip owns all 4), prepared (OV-23), io [ring, slab], value slots page0 (the
  slab's (slot, layer) batch x 128) and the 4 ring rows. Cores (rule 18): 1 home + up to 4 port cores a chip (logical
  (0..4, 0)). Applies to every box with > 1 chip (tree from mesh_plan's ports and torus wraps; 2x2 / 2x4 lines too).
- Bits: tests/test_snapshot_send.py (new): two caches on one slab tensor (batches 1 and 2, batch 0 a guard), the same
  rows, legacy vs on (on's batch poisoned first) vs the definition, every chip, 6 ends (cap 64, whole chunk owned by
  chip 31, cap 6144 owned by chips 5 + 6, a replicated follow-up, owners 30 + 31 at an end mid-record, cap 0): all
  equal, guard untouched. First version's bug (fixed before any number here): snapshot_send() returned the object
  under legacy too, so the "legacy" side ran the new path.
- Per call (probe, 8x4, steady pattern, median): host 352.3 -> 108.2 us; back-to-back wall 317.4 -> 72.5 us; device
  163.2 -> 66.8 us, programs 4 -> 1.
- `measure --task P.65` (change on: CHEAP_CCL=on ENGRAM_TABLE=resident): determinism 0 (novel, code, state); quick
  min layer PCC 0.60548579 at L39 (= P.52-P.64: the same output bytes), host transfers per layer 0, clean wall 159.4
  ms (spread 61.5; P.64 161.8); p51200 (random ids) wall 168.0 (spread 50.3: one 214.8 run, the rest 164.5-177.2,
  loadavg 13.6; P.64 173.7), device_ms_total 149.69 (P.64 154.93), host_launch_calls 1136 (P.64 1376), host_launch_ms
  90.3 (110.6), host overhead 18.3 (18.8), 0 new programs.
- Gate unit P.65#test_moe_prev_replicas (PASS, change on, one pass a side), chunks 1..10 mean, host / wall / device ms
  (P.64's gate unit in brackets): novel off 138.0 / 173.2 / 163.03 (207.4 / 235.2 / 167.71, a slow side); novel prev
  138.1 / 164.2 / 145.24 (144.4 / 162.9 / 148.33); code off 143.7 / 169.0 / 152.57 (146.9 / 169.7 / 156.07); code prev
  138.6 / 158.7 / 141.33 (150.0 / 164.5 / 144.89). maxdiff 0, repeat 0, mirror 0 mismatches (both prompts, three host
  paths).
- Expected vs got: launches -240 a chunk (exact; 6 a layer at p51200 -> 1); host -12 to -15 ms expected, per-call
  numbers give -9.8 ms (40 x 244 us), p51200 host_launch_ms -20.3; device 0 to -3 expected, got -3.3 to -5.2 ms (the
  gathers' link time was bigger than I guessed: 163 us a layer); wall -1 to -6 ms on real chunks (one pass a side,
  +-4 ms noise, DS41-264), -5.7 at p51200. The orchestrator's A/B gives the on / off table.
- Not done: mhc_post (82 calls ~100 us, a C++ fork op already at one launch's floor on 32 chips, DS41-263: only
  folding it into a neighbouring program removes it) and the other 12 all_gathers (compressor layers): next picks.
  The tree's device time could drop further with sub-block pipelining (depth 6 x 32 KB blocks); not needed for the
  host goal.
- Re-run: probe `scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_snapshot_send.py -s` (~1 min); gate unit `gate P.65 --spec <spec>
  --check --unit 'P.65#test_moe_prev_replicas'` (~9 min, devjob); `orchestrator measure --spec <spec> --task P.65`.

## P.66 (perf, attempt 1): per-layer Python glue trimmed (LEAN_PY legacy | on)
- Expected per chunk (written before the change was measured; perf_guide 2 / 3): census of the current code (novel,
  prev, chunks 0..10, tests/test_lean_py.py P66_CENSUS): 630 ccl.dealloc frees a chunk at ~7.8 us (4.9 ms), 386 more
  dealloc calls on pinned tensors (Python only), 63 Python allocations (40 of them the MoE arena, 88 us each, L1:
  not removable without the arena time-sharing L1). Removable without changing any op: the block's per-call step
  wrappers (a dict of 15 closures, a set and a dict per step: ~0.7 ms), the pybind is_allocated before the pinned
  check (~0.3 ms), and three fixed-shape per-layer outputs read only inside their layer written into load-time
  buffers (alloc in the op + free ~35 us each, DS41-172): the ffn residual's dead stats row (37 a chunk), wo_b's
  output (40), the MoE output's add (40): ~3.9 ms. Host -4 to -6 ms a chunk, device 0, launches 0 (the same ops).
  The wall moves only where the chunk is host-bound (prev: host ~138-144 vs device ~141-148 ms, P.65).
- Terms: host = first host call of a chunk to the last layer's enqueue; busy = host minus the engram worker wait; wall =
  to the sync after the last layer (clean: no profiler); prev / off = MOE_REPLICAS=prev (lean host path, shipped) /
  switched off in the loaded model; legacy / on = LEAN_PY, switched in the loaded model (tt/lean.set_mode). Means of
  chunks 1..10 of fresh requests, 4 rounds with the sides alternating, +- = standard error of the paired difference.
- Built (LEAN_PY = legacy | on, tt/settings.py, default on; tt/lean.py holds the switch as `ON[0]`):
  1. tt/model.py: the block's step wrappers made once per layer (`_lean_overrides`; the caller's ids in
     `self._protect` during the call), a step with nothing to free runs unwrapped; `_free` checks ccl._KEEP (the
     load-time tensors) before the pybind `is_allocated`. MOE_TRACE=moe keeps the per-call path.
  2. Three per-layer outputs read only inside their layer written into load-time buffers (tt/host_alloc.py, reserved at
     load only under on; pinned, so the block's free skips them; every next writer is queued after the readers on the
     same chip): ffn_residual's mhc_post stats row on the POST_COLLAPSE path, which nothing reads ("fork.post.dead_row",
     stats_output=); wo_b's output = the attention output under OWN_ROWS ("attn.wo_b_out", optional_output_tensor);
     the MoE output's ttnn.add on the own rows ("moe.out", output_tensor=; not under MOE_TRACE).
  3. tt/attention.write_own: no per-chip `_pin_put` call when the chunk misses the pin's span (32 -> 0 a layer on
     most chunks); host_alloc.fork_out: a tuple shape as its key (no generator).
  Same ops, same order, same programs' arguments except the given outputs: no launch added or removed.
- Probe (tests/test_lean_py.py, new; one model load, the 4 sides alternating, outputs of chunks 1 and 10 vs the first
  side): max |diff| 0 on every side, both prompts.

  | wl, moe | host legacy -> on | busy diff | wall legacy -> on |
  |---|---|---|---|
  | novel, prev | 138.1 -> 133.0 | -4.76 +- 0.98 | 161.5 -> 160.2 (-1.3 +- 0.9) |
  | novel, off | 136.9 -> 135.6 | -2.44 +- 1.35 | 171.8 -> 173.7 (+2.0 +- 1.6) |
  | code, prev | 140.0 -> 135.7 | -4.27 +- 1.95 | 160.0 -> 159.0 (-1.1 +- 2.9) |
  | code, off | 137.8 -> 134.4 | -3.08 +- 1.77 | 164.7 -> 164.1 (-0.6 +- 2.4) |

  Census a chunk (novel, prev): frees 630 -> 513 (4.93 -> 4.30 ms); cProfile: ttnn op calls -5.4 ms (the outputs no
  longer allocated), the wrappers' dict / set building gone, _pin_put 1280 -> 128 calls. Launches unchanged.
- `measure --task P.66` (change on: LEAN_PY=on ENGRAM_TABLE=resident): determinism 0 (novel, code, state); quick
  pcc_chunk_out 0.60548579 (= P.65: the same output bytes), host transfers per layer 0, clean wall 159.4 ms (spread
  5.5); p51200 (random ids) wall 167.7 (spread 5.7; P.65 168.0), device_ms_total 150.27 (P.65 149.69),
  host_launch_calls 1136 (= P.65), host overhead 17.4 (18.3). The p51200 chunk is device-bound (wall ~ device + 17):
  the host cut cannot show there.
- Gate `P.66 --check`: PASS (determinism + ladder last, test_moe_prev_replicas, test_profile x2; 1200 s). Its
  prev-replicas unit (change on only, one pass a side, +-4 ms noise, DS41-264), chunks 1..10, host / wall / device
  ms: novel off 131.3 / 172.2 / 162.75, prev lean 141.0 / 169.9 / 146.05; code off 134.0 / 166.3 / 152.50, prev lean
  134.9 / 158.8 / 141.80; maxdiff 0, repeat 0, mirror 0 mismatches. The on / off difference is the alternated probe
  above, not these single passes.
- Expected vs got: host -4 to -6 expected, -4.3 to -5.1 ms got with prev (-1.2 to -3.4 with off, noisier); device 0.
  The wall does not follow: in this probe wall ~160 ms vs host ~133-138 ms, i.e. the chunk is device-bound here
  (perf_guide 1), so the host cut is headroom until the device time drops.
- Left (not in this pick): the MoE arena's L1 allocation (40 x 88 us + its free; L1 is time-shared, a load-time arena
  would clash with other ops' CBs), the boundary streams (h, pre: 82 frees, the caller owns them), sparse_sdpa's q /
  o rows (2 x 38 frees; rope_chain's output_tensor is scalar-form only), nlp_concat_heads' output (no output option in
  the fork), run_block (framework, ~1.6 ms under cProfile). Prepared (OV-23): no Python-built op touched. OV-3:
  nothing padded. Cores (rule 18): unchanged (the same ops on the same grids). Fork suites not run (OV-25).
- Re-run: `P66_CLEAN=4 PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_lean_py.py -s` (~9 min with the load; devjob).

## P.67 (perf, attempt 1): the replica branch's wins merged into model_bringup (owner 2026-11-14)
- How merged: NOT git merge / cherry-pick commits; one 3-way apply (`git apply -3`) of
  `git diff 6bdf324e716 b9a45fb0b80 -- . ':(exclude)models/demos/deepseek_v41_flash_d_p/bringup/ledgers'` (the
  branch's whole code / test / fork / knowledge delta since the fork point, without its ledger), so the orchestrator's
  commit is an ordinary one (no MERGE_HEAD). Every file only the branch changed is byte-equal to b9a45fb0b80 (after
  the id renumbering below); tt/attention.py and tt/model.py merged without conflict (model_bringup's P.51
  decoder_skip_stop / FIX.28 use_decoder_skip + _build_decoder kept beside the branch's lean / moe_trace / replica
  hooks); orchestrator.py: both sides had the same rel() fix (a05a31f806b == 894ffd47779), nothing to merge.
- known_issues id clash: model_bringup added DS41-241..245 (FIX.28-FIX.31) and rewrote DS41-240; the branch added
  DS41-241..269. Kept model_bringup's 240..245 as they are (its ledger refers to them); the branch's 241..269 are
  renumbered +5 -> DS41-246..274 in every file the branch brought (known_issues, known_issues_detail, BREADCRUMBS,
  code comments, tests, fork CHANGELOG / README; also the forms DS41-a..b and DS41-a/b). The branch's ledger, which
  still says the old numbers, stays on the branch. BREADCRUMBS / repo_map: both sides' sections kept.
- Defaults as on the branch: EXPERT_X_PIECES=min8k, MOE_REPLICAS=prev, MOE_FILL=piggyback, MOE_PREV_HOST=device,
  MOE_PREV_HOST2=lean, MOE_PLANNER=folded, MESH_PROGRAMS=on, ATTN_FUSE=on, CHEAP_CCL=on, LEAN_PY=on, MOE_TRACE=off;
  model_bringup's COMBINE_SPLIT=on, DECODER_SKIP (off default; decode via use_decoder_skip), PREFILL_END=cap kept.
- Decoder skip vs replicas: expert_prev's guard uses a layer's plan only when its next call is >= NCALL ordinals
  later; a chunk that skips layers 21-39 makes fewer MoE calls, so the next chunk's layers run replicas off (host
  only, no transfer) - correct, a missed gain on skipped chunks only. Checked on the gate's test_decoder_skip_decode.
- Expected per chunk (before measuring; perf_guide 1-2), the A/B's last golden chunk vs today's model_bringup
  (X.3: wall 182.2 ms, device 171.9, host overhead 10.4): device -10 to -15 ms (P.56 replicas + P.59 x pieces;
  workload-dependent, the branch measured on novel chunks), host -20 to -30 ms of launch work (P.60-P.66) of which
  only the part on the critical path shows: wall ~182 -> ~160 ms (the branch's P.66: 159.4 ms, spread 5.5).
  Rebuild: `cmake --build build_Release --target install -j 48` (40 s).
- Gotcha (DS41-275): the first `orchestrator measure --task P.67` ran the 'new' side without ab.env
  (DS41_ENGRAM_TABLE=resident; unset = mmap under pytest, DS41-245) and showed the merge 29 ms SLOWER (quick 211.9,
  p51200 226.7 ms, not settled) although device fell 35 ms. Fixed measure() in orchestrator.py to run ab.before picks
  with ab.env on both sides, as ab_report() already does (F230); re-measured.
- Measured (measure --task P.67, one process, both sides DS41_ENGRAM_TABLE=resident), before -> after:
  quick (last golden chunk) wall 182.8 -> 157.9 ms (spread 8.4 / 8.1), errors identical (worst err_state_window_kv_L25
  0.147 / limit 0.197, min layer PCC 0.6055 both); p51200 wall 202.9 -> 166.9 ms (spread 11.7 / 32.5), device
  187.3 -> 152.0 ms (experts 105.8 -> 74.9, ffn_residual 11.3 -> 7.4, attention 48.1 -> 47.8), host launches
  1656 -> 1136 (122.4 -> 89.1 ms), host_overhead 15.6 -> 14.9, cProfile 204.7 -> 170.8; determinism 0 both sides.
  Expected -20 ms wall: got -25 (quick) / -36 (p51200); device -35 ms, more than the -10..-15 expected.
- Prepared (OV-23): no new Python-built op beyond the branch's (the branch's own P.56-P.66 notes cover them). OV-3:
  nothing padded. Cores (rule 18): unchanged from the branch. Fork suites not run (OV-25).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.orchestrator measure --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --task P.67` (devjob, ~8 min).
- Gate (`gate P.67 --check`): first full run FAILED one unit: test_decoder_skip_decode, served_bytes_differ = 16
  (novel / code N 5248 P 5120 and N 10368 P 10240: 4 records each, ('0_window_kv', layer 0, rows 0/32/64/96): layer
  0's served window snapshot of the DECODER_SKIP=off run != the decode run; every other case 0); the one-process
  batch stopped there, so contract / acks / determinism / replica units did not run. Then the same test passed with
  DS41_CHEAP_CCL=legacy (0 differ), with DS41_ATTN_FUSE=legacy (0 differ), and AGAIN WITH BOTH ON
  (`gate --check --unit P.67#test_decoder_skip_decode`, 0 differ): an intermittent race, 1 of 4 runs; the legacy
  passes do not localize it. DS41-276. Second full `gate P.67 --check`: every unit PASS (ladder, profile wall 159.1 ms
  / device 143.9 ms, decoder skip, determinism x2, contract, acks, replica test). Not fixed: the owner should know
  the decode-skip snapshot path has a flake after this merge (suspects: P.64's fused window write and P.65's
  SnapshotSend, both new fabric programs on core row 0 with no cross-chip barrier between them).

## FIX.32 (overseer fix, attempt 1): the P.67 "served-snapshot race" was a prepared-program cache key (DS41-277)
- Step 1 (first): ATTN_FUSE and CHEAP_CCL defaults set to legacy (code and 'on' kept). Proof: test_decoder_skip_decode
  20 times in one process after test_determinism on the same model (the order of the failing P.67 batch), both
  legacy: 20 of 20 PASS, served_bytes_differ 0 every run (tests/test_snapshot_race_loop.py, new).
- Cause (not a race): `SnapshotSend._pp` (P.65, tt/snapshot_send.py) cached one prepared program per block pattern.
  Its TensorAccessor compile-time args hold the slab's shape, and an ND-sharded accessor takes the outermost page
  coordinate as page_id % tensor_shape[0] (tt_metal/hw/inc/api/tensor/tensor_accessor.h:445). test_determinism runs
  whole chunks on ladder caches with their own 1-batch slab ([1, 1, 128, 512]) and so builds the steady pattern's
  program (31, 31, 31, 31: the last chip owns the 4 blocks). Reused for the engine's [slots x layers, 1, 128, 512]
  slab, every (slot, layer) write landed in batch 0 = slot 0, layer 0. In the test only chunk-aligned stops use that
  pattern; slot 0 layer 0 then holds the last write, the others stale bytes that are equal in both slots, hence exactly
  4 records (layer 0) a case. The failing P.67 batch had 4 of 4 such cases differ (deterministic), the 3 passing runs
  were processes without test_determinism first.
- Evidence: tests/test_snapshot_send_geometry.py (new, ~10 s): own 1-batch slab first, then an engine-like 3-batch
  slab at batch 2. Before the fix: 64 failures (all 32 chips: batch 2 not written, batch 0 overwritten). After: 0.
  Model level before the fix (test_determinism, then CHEAP_CCL=on alone, 2 runs): 2 of 2 FAIL, served_bytes_differ 16,
  the gate's exact records; DIAG: layers 0 and 1 wrong on every chip against the owner's ring.
- Fix: the program cache key is (pattern, ring geometry, slab geometry), geometry = shape, dtype, layout, memory
  config (`snapshot_send._geom`). Not a barrier or semaphore change: the fabric ordering was sound (the router flushes
  the write, acked, before a fused increment: fabric_edm_packet_transmission.hpp flush_write_to_noc_pipeline).
  WindowSend (P.64) builds its programs once too, but its io tensors (region, kept tail, rows, ring) have one interleaved
  geometry in every path (asserted in pages and bytes), so it is left as is.
- Step 2 proof (one process: test_determinism with both on, then 20 runs a side): both on 20 / 20 PASS, ATTN_FUSE alone
  20 / 20, CHEAP_CCL alone 20 / 20, served_bytes_differ 0 in all 60 runs. Bit-identical to legacy: tests/
  test_snapshot_send.py (legacy == on == definition, every chip, 6 cases) and tests/test_window_write_fused.py (5 piece
  patterns, every chip) pass. Host us of the snapshot call with the key: 112.3 (P.65: 108.2; geometry probe 94.3).
- Step 3: both defaults back to on.
- Cost of legacy vs on (test_moe_prev_replicas, chunks 1..10 mean, shipped prev lean path, host / wall / device ms):
  see the table below (legacy: env DS41_ATTN_FUSE=legacy DS41_CHEAP_CCL=legacy; on: the gate's unit, same code).

## FIX.33 (overseer fix, attempt 3): model_bringup without expert replication (owner 2026-11-15)
- Attempt 2 left no change in the tree (checked `git diff`); this attempt starts from HEAD `6777771a502`.
- Method: replica code taken out by rebuilding, not by hand-deleting. Fork `moe_block_main_ttnn` and INDEX.md = their
  pre-replica state (`6bdf324e716`, the replica branch's base; model_bringup never changed the fork after it) +
  `git apply --3way` of P.59 (`fd96a8d0456`) and P.63 (`a39123e5c25`) only. tt/moe_block_main_adapter.py, model.py,
  residual.py = pre-P.67 model_bringup (`ce1b9047afd`: has FIX.28/29) + P.59, P.63, P.66 (`b9a45fb0b80`). Every
  conflict was a replica field beside a kept one (HeadTail / tail next to cached_slots; fill / fill_exact next to
  cached_slots; the block key's spc; P.61's hc_out / output_tensor in residual._collapse); kept side only.
- KEEP (P.67 file / hunk): attention.py, window_send.py (P.64 ATTN_FUSE), snapshot_send.py (P.65 CHEAP_CCL + FIX.32's
  geometry key), lean.py, host_alloc.py (P.66), settings ATTN_FUSE / CHEAP_CCL / LEAN_PY / MESH_PROGRAMS (P.63) /
  EXPERT_X_PIECES (P.59), the adapter's x_min_piece_bytes / cached_slots / _lean_out, model.py's _lean_overrides,
  residual.py's dead_row_site; fork x_min_piece_bytes and cached_slots; tests test_attn_launch_census,
  test_window_write_fused, test_snapshot_send, test_mesh_program_probe, test_expert_x_pieces (its P.59 test),
  test_lean_py, test_snapshot_race_loop, test_snapshot_send_geometry; orchestrator.py (894ffd47779, framework).
- REMOVE: settings MOE_REPLICAS, MOE_PREV_HOST, MOE_PREV_HOST2, MOE_PLANNER, MOE_TRACE, MOE_PREV_FIXED / MIN / SLACK /
  NEW, MOE_EXACT_BLOCK / FANOUT / CAP / LIMIT_MS, MOE_FILL / _PIECES / _CAP, MOE_REPLICA_DEMO, MOE_COUNTS / _DIR / _TAG
  (P.45 counts: the replica study's); tt/expert_prev.py, expert_exact.py, expert_fill.py, expert_slots.py, moe_trace.py,
  kernels/moe_prev_planner.cpp; the adapter's replica parts (~770 lines: slots, set_replicas, exact / prev / fill,
  counts, _MAIN_LAYERS); model.py's seg / _segment / counts / replica_demo / prev_reset; residual.py's trace outputs;
  host_alloc's MOE_TRACE condition; testing/profiler.py's trace_replay / remap_replays (P.61's only caller);
  tests moe_exact_model, moe_tp_model, test_expert_slot_copy, test_host_bound (P.57, prev on), test_moe_counts,
  test_moe_exact_bound / _copy / _replicas, test_moe_fill_piggyback, test_moe_prev_host_probe, test_moe_prev_replicas,
  test_moe_replicas, test_moe_tp4_check, test_moe_trace, test_prev_host_account, and test_expert_x_pieces_layouts
  (P.58's TP study on P.55a counts); fork expert_slot_copy/, prev_planner.hpp, fx_wait.hpp, REC_FILL / REC_FX /
  prev counts in combine_port_writer.cpp, the head's HeadTail, the router's slot map, SE_FX_SPARE, RouterSpec slot
  fields, the resources' fill words, their CHANGELOG / README / INDEX text (one FIX.33 CHANGELOG entry records it).
- Orthogonal wins that leaned on replica code: (1) test_lean_py / test_mesh_program_probe / test_attn_launch_census
  took prompt ids / maxdiff from the replica tests -> tests/workloads.py (`prompt_ids`, `maxdiff`); test_lean_py's
  sides are LEAN_PY legacy / on only (was glue x prev/off). (2) test_mesh_program_probe toggled cached_slots through
  the replica list `_MAIN_LAYERS` -> the model's own blocks (`b.moe_block`). (3) LEAN_PY's moe.out buffer was gated on
  MOE_TRACE=off -> unconditional (OWN_ROWS). (4) P.63 / P.59 fork hunks sat next to replica fields; nothing of theirs
  used them. No orthogonal behaviour changed.
- Rebuilt (`./build_metal.sh`, rc 0; `ttnn.bringup.moe_block_main` has no expert_slot_copy / fill / tail entry).
- Gate (FIX.33 --check, all 7 units PASS): ladder rung last: all 303 err_* metrics equal FIX.32's (replicas prev on)
  to the last digit, err_final_hidden 0.13211 (limit 0.24219), top5 1.0; determinism 0.0 (full and DECODER_SKIP=decode);
  decoder_skip_decode, contract, adapter acks pass (served_bytes_differ 0, acks_early 0).
- Numbers (gate profile, 8x4, chunk 5120): chunk_wall_ms 170.4 (spread 4.6; profiled 171.4), device_ms_total 159.37
  (experts 84.87, attention 43.47), host overhead 11.0 ms, profiled timeline host dispatch 148.6 ms. vs P.67 A/B: off
  pre-P.67 182.8, replicas on 157.9 (FIX.32's gate: 159.8 wall, 143.29 device, experts 70.46): replicas-off is +10.6 ms
  wall, +16.1 device (all in experts) vs FIX.32, as expected (~10 ms); -12.4 ms vs pre-P.67 from P.59 / P.63-P.66.
- Grep (tt/, tests/, hooks.py, the fork, testing/): no MOE_REPLICAS / expert_prev; `prev_` only in pre-existing
  unrelated names (prev_end, prev_hi) and one docstring of tests/workloads.py naming the deleted file it came from.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.33 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`

## FIX.34 (overseer, OWNER 2026-11-15): realistic prefill perf at 51k..1M under DECODER_SKIP=decode (no model change)

- New test `tests/test_skip_positions.py` (BATCH_SHARED; one model load for all points; no model code changed). Points
  51200, 128000, 256000, 512000, 768000, 1024000 (SKIP_POS overrides). Per point two chunks, each timed alone:
  steady = `begin_chunk(.., p, last=False)`; resumed = `begin_chunk(.., p, last=False, write_start=p + 128)` (128 rows
  encoder replay, windows truncated at p, 4992 new rows).
- Preceding chunk: every iteration (warm-up, timed, profiled) first runs the REAL previous chunk tokens[p-5120:p] at
  p-5120 (last=False, untimed, synced), then the timed chunk; never the same chunk twice in a row; no steady_state()
  (the window carry / engram tail / any MoE state are the previous chunk's). Long-range caches: a fresh state sized
  p+5120 and random ids (seed 0), test_profile's BRINGUP_POS mechanism (F230, testing/profile.PositionGolden).
- Wall: testing/profile.warm_up (F196/F200: until no new program and settled), then 10 timed runs with cache misses
  refused; median and spread; begin_chunk to the sync. Device: one rt-profiler run (F155); `<kind>_device_ms_p<p>` =
  the slowest chip's total; `_secmax` = test_profile's device_ms_total (slowest chip per section, summed); phases as
  test_profile, plus section chunk_input (begin_chunk + embedding, ~0.2 ms, uncounted in test_profile).
- Settings: DS41_DECODER_SKIP=decode (gate env); DS41_ENGRAM_TABLE set to its shipped default (resident) when unset,
  since pytest otherwise reads F222's test value mmap (DS41-245). Everything else shipped defaults.
- Gotcha (DS41-279): the 8x4 spec's target.seq is 56320, so the loaded model's position tables stop there; the first
  gate run reported p128000..p1024000 "not run: p + chunk > max_seq 56320". The test now raises target.seq in memory
  to the longest point (as profile.run_profile does for BRINGUP_POS), capped at deploy.seq (1,044,480).
- Memory (checked first): every point fits. Allocator DRAM a chip (lockstep mesh): load 15.90 GiB (max_seq 1,029,120),
  with the point's state 16.13 (p51200) .. 21.71 GiB (p1024000) of 31.83; prior evidence P.32 test_dram at 1M
  (1,049,600, served slabs): peak 18.39 GiB. Values are after allocation, not an in-op peak (test_dram's peak is
  ~+1.2 GiB over load).
- Results (8x4, chunk 5120, gate check log FIX.34.check.log):

| point | chunk | tokens | wall ms (spread) | device ms | host ms | tok/s | experts | attention | indexer | attn_res | ffn_res | compressor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 51200 | steady | 5120 | 106.7 (7.8) | 81.6 | 25.1 | 47985 | 46.9 | 22.9 | 1.3 | 4.3 | 4.1 | 1.9 |
| 51200 | resumed | 4992 | 107.2 (5.8) | 81.4 | 25.8 | 46567 | 45.9 | 22.9 | 1.3 | 4.2 | 4.2 | 2.7 |
| 128000 | steady | 5120 | 107.6 (46.2) | 82.4 | 25.2 | 47584 | 46.2 | 22.7 | 2.5 | 4.2 | 3.9 | 1.7 |
| 128000 | resumed | 4992 | 109.4 (10.2) | 82.8 | 26.6 | 45631 | 46.0 | 22.8 | 2.5 | 4.2 | 3.9 | 2.7 |
| 256000 | steady | 5120 | 106.4 (25.2) | 85.7 | 20.7 | 48120 | 47.4 | 23.0 | 4.7 | 4.3 | 4.0 | 1.9 |
| 256000 | resumed | 4992 | 112.9 (37.4) | 85.1 | 27.8 | 44216 | 46.1 | 23.1 | 4.7 | 4.2 | 3.9 | 2.8 |
| 512000 | steady | 5120 | 112.5 (24.6) | 88.5 | 24.0 | 45511 | 46.2 | 22.8 | 8.9 | 4.3 | 3.9 | 1.7 |
| 512000 | resumed | 4992 | 114.3 (35.3) | 89.5 | 24.8 | 43675 | 47.1 | 22.2 | 8.9 | 3.6 | 3.6 | 2.7 |
| 768000 | steady | 5120 | 115.7 (5.3) | 93.0 | 22.7 | 44252 | 46.5 | 23.0 | 13.1 | 4.3 | 4.2 | 1.7 |
| 768000 | resumed | 4992 | 120.3 (6.2) | 95.3 | 25.0 | 41496 | 47.3 | 22.9 | 13.1 | 4.3 | 4.1 | 2.9 |
| 1024000 | steady | 5120 | 120.4 (6.3) | 99.1 | 21.3 | 42525 | 47.5 | 23.1 | 17.3 | 4.1 | 4.1 | 2.1 |
| 1024000 | resumed | 4992 | 125.5 (6.8) | 97.9 | 27.6 | 39777 | 46.2 | 23.0 | 17.3 | 4.3 | 4.1 | 2.7 |

  (device = slowest chip; section columns in ms, slowest chip per section; the full table with every section is the
  metric skip_positions_table in results/FIX.34.json.) What grows with position: the indexer (1.3 -> 17.3 ms, linear in
  the keys scored); experts, attention and the residuals are flat. Resumed costs ~+1 ms compressor (the replay) and up
  to ~5 ms wall; its tok/s counts only the 4992 new rows. Spread outliers (one run of 10 at +20..40 ms at 128k-512k)
  are host-side; the medians are stable.
- Re-run: `PYTHONPATH=$PWD DS41_DECODER_SKIP=decode scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_skip_positions.py` (via devjob: ~11 min with the load), or the gate:
  `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.34 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`

## FIX.35 (overseer, OWNER 2026-11-15): measured end-to-end prefill 0 -> 1,024,000 under DECODER_SKIP=decode (no model change)

- New test `tests/test_full_prefill.py` (BATCH_SHARED, one load). Drives the serving runtime itself
  (adapter.DS41PrefillRuntime: build_runtime, allocate_kv_cache 1 slot at deploy.seq 1,044,480, runtime.compile), one
  request in slot 0 of 200 x 5120 chunks via `runtime.prefill_chunk` back to back; the runtime marks every chunk
  last=False under decode. Engine inputs (server_rules.server_payload) put on the device before the clock (the H2D
  socket has them there). No sink / D2H acks (sink would sync per layer; D2H is a device op per layer, not run).
  The only per-chunk sync is the runtime's own: `_host_ids` reads the engine input back (a blocking read behind the
  previous chunk's work). The test times it by wrapping `runtime._host_ids` (timing only). 6 boundary syncs.
- Tokens: the canonical prompt whole (chat-wrapped Tale of Two Cities, 189,939 ids; its first 56,320 = prompt.json,
  asserted) + 256 WikiText-2 raw train articles (/mnt/models/agentic-prefill-goldens/wikitext-2-raw-v1-train.txt),
  1,098,205 ids; 200 distinct chunks (asserted); the warm-up uses ids after the first 1,024,000.
- Gotcha DS41-280: spec text.source (models/tt_transformers/...) no longer exists; read models/ttt_compat/tests copy
  (same sha). Gotcha DS41-281: the 3-chunk warm-up is not enough: the first full request built 4 programs at every
  chunk 3..199 (788, +11 s: 33.18 s); the test repeats the request until one builds 0 (here the 2nd) and measures it.
- Device time: a separate pass under the rt profiler with its own array-backed callback (3.96 M program records, 0
  dropped; 16-bit ids unwrapped against the host id at arrival). That pass's wall is 25.17 s vs 21.87 unprofiled, so
  its span / gap are upper bounds; program busy time is not affected.
- Results (8x4, warm request, gate log FIX.35.units.log):

| boundary | cumulative wall s | tok/s | sync ms |
|---|---|---|---|
| 51200 | 1.01 | 50460 | 15.1 |
| 128000 | 2.59 | 49331 | 16.3 |
| 256000 | 5.17 | 49520 | 18.3 |
| 512000 | 10.51 | 48713 | 24.6 |
| 768000 | 16.02 | 47938 | 26.9 |
| 1024000 | 21.87 | 46823 | 27.3 |

  Per chunk: host enqueue (after the read-back) 86.1 ms mean (p10 80.3, p90 91.3, max 133.4); read-back 22.6 ms mean
  (idle read-back 2.4 ms, so the host waited ~20 ms for the device every chunk); device (slowest chip's program
  time) 80.7 ms mean, 70.6 (chunk 0) -> 92.6 (indexer grows); gap between chunks 6.3 ms mean (max 21.2).
  Sync at the boundaries 15-27 ms = the device's lag behind the host. The next chunk's read-back would wait the same.
- Verdict: device-bound. The host is ahead of the device at the end of every chunk (host_bound_chunks 0 of 200: the
  next read-back waited >= 7.8 ms beyond idle on every chunk). Host time shows in the wall only at chunk boundaries:
  the runtime's per-chunk read-back + begin_chunk host work before the first program, 1.26 s of 21.87 s (5.8%,
  profiled pass). Wall 21.87 s vs sum of program time 16.14 s: the other ~5.7 s is the device between programs
  inside a chunk (dispatch / CCL waits), not host. Host enqueue (86 ms) exceeds program time (81 ms) on 159 chunks,
  but not the device's timeline per chunk (~109 ms).
- vs FIX.34 (one chunk alone, random ids, zero history, wall incl. its sync): 106.7-120.4 ms wall a chunk there;
  here 21.87 s / 200 = 109 ms a chunk on average. Device ms here 75.3 (p51200 chunk) .. 86.2 (last) vs 81.6 .. 99.1
  there. Real text routes differently from random ids. Its "host 25 ms" is wall - device (unhidden), not enqueue time.
- Re-run: `PYTHONPATH=$PWD DS41_DECODER_SKIP=decode scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_full_prefill.py` (via devjob, ~7 min with the load), or
  `gate FIX.35 --spec models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.

## P.72 (perf, owner A/B): STATIC_TABLES=off still runs? (8x4)

- Estimate (written before the change): the change touches only the off side, so on (the default) saves 0 device
  and 0 host per chunk. Expected A/B (on vs off) on the 8x4 at realistic positions: off adds, per compressed KV source
  a chunk, the history slice + tilize + mesh all-gather (TP then SP, 32 chips) + untilize, and the per-chunk fp32
  top-k remap; device +5..+20 ms a chunk growing with position (the gathered history is max_seq / r rows), host
  +3..+6 ms (~15 extra launches a source). 2x2 8-layer reference: 103.5 (on) vs 118.8 ms (off).
- Broken (DS41-282): with STATIC_TABLES=off the first compressed layer asserted "WINDOW_KV_FORMAT=bfp8_tiles needs
  STATIC_TABLES=on" (devjob p72_off_skip). Since P.38 the default WINDOW_KV_FORMAT=bf16_tiles gives every chip owning
  >= 128 rows a per-chip bf16 window region; the P.19 off path only handled the replicated window table.
- Fix (off path only, tt/sparse_attention.py): with a bf16 region, off writes the window into the region
  (_window_region), uses the gathered compressed history alone as the table, and builds the index as
  [region window ids | top-k rows at offset 0] (new TtSparseAttention._index_region); sparse_sdpa gets the region as
  its BF16 kv_window, as _static does. The memo key gains the region flag. The on path (_static) is not touched.
- Checks, STATIC_TABLES=off, both PASS on the fixed tree: devjob p72_off_skip2 (DS41_DECODER_SKIP=decode
  test_decoder_skip_decode.py, rc 0, 343 s) and p72_off_det (common/bringup/tests/test_determinism.py, rc 0, 201 s).
- Attempt 1 was cut off by a node loss; its on-side measure hung at p256000 in test_skip_positions (dispatch timeout,
  cores 15-2/15-3), on the untouched on path: left to the orchestrator (OV-42). Attempt 2 ran nothing on the device.
- Re-run: `DS41_STATIC_TABLES=off DS41_DECODER_SKIP=decode python -m models.demos.common.bringup.devjob start x --
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_decoder_skip_decode.py`

## FIX.36 (overseer fix, attempt 1): the intermittent DECODER_SKIP=decode hang, and test_skip_positions misreporting it

- (3) Test fixed (tests/test_skip_positions.py `_alloc_failure`, DS41-284): a point is 'does not fit' only when a line
  of the error says "Out of Memory" / "Not enough space to allocate" / "allocation failed" / "cannot allocate"
  (bank_manager.cpp's wording); a message with "timeout" / "potential hang" / "timed out", or any other error,
  re-raises and fails the test. Cause of the misreport: FIX.34 checked the first line only for "memory" / "alloc",
  and a hang's first line is "TT_THROW @ .../system_memory_manager.cpp:806: tt::exception".
- (1) Reproduced on the device (devjob fix36_loop3, new tests/test_skip_hang_loop.py, one load): the test's per-point
  work looped over p256000 / p1024000, 23 runs a kind, the last under the rt profiler. Rounds 0-12 clean: 2,392
  chunks in ~7.5 min, layer 19's output of the p256000 steady chunk bit-identical every round (max_abs_diff 0.0).
  Hung in round 13 at `p256000 resumed run0 prev`: the first chunk after the steady kind's profiled run (finish_rt:
  sync, 1 s idle, unregister). The 23:18 P.72 hang was also ~3 s after a profiled run (p768000's, then p1024000's
  new state and first warm-up runs).
- What hangs: unknown. tt-triage could not attach (UMD discovery, eth-core read: "MMIO per-op timeout: 4B load took
  55527 us"). Cores 15-2 / 15-3, waited on at teardown on every chip, are the prefetch / dispatch cores
  (wait_for_dispatch_cores), not a model program. Chip 15 (phys 15 / M0D14, one of the 4 chips with "4 eth channels
  but only 2 routing planes" at open) had a core whose MMIO died (RISC reset and L1 ECC skipped); tensix reads still
  answered on every chip. A spinning kernel leaves MMIO readable; this did not.
- Excluded by reading: equality semaphore waits in the model's fabric sends (all wait_min + atomic take-back);
  ENGRAM_ASYNC's worker on CQ 0 (FDMeshCommandQueue locks every enqueue / write; pinned writes block); the rt
  profiler as a new factor (its push kernel and receiver run from mesh open in every run).
- (2) Not done: no fix, so no 20-run proof. Brief: stop device work at the first hang. The safe runner started
  recover-hosts (sys-triage, glx reset) by itself at 04:34; nothing else ran after it. Leads in order (each may cost a
  box reset): same chip again? (re-run the loop); idle-then-burst / rule 18 core audit of the skip-positions flow;
  HANGLOOP_PROFILE=0; watcher. Detail: known_issues_detail.md "The intermittent DECODER_SKIP=decode hang ...".
- Gate not run (device hung). Logs: /scratch/dnijemcevic/fix36/ (loop3.log, triage log, inspector_loop3/).
- Re-run the reproduction: `DS41_DECODER_SKIP=decode HANGLOOP_MINUTES=20 python -m models.demos.common.bringup.devjob
  start loop -- scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_skip_hang_loop.py -s`

## FIX.36 (overseer fix, attempt 2): the hang's cause, rule 18 (120-core programs), capped

- OV-43 lead 1, no profiler: `HANGLOOP_PROFILE=0` hung in round 7 (`p256000 resumed run5 cur`, mid-stream), chip 3's
  MMIO / PCIe dead (PcieHangError). Not the profiler, not idle, and not always chip 15 (15, 3, then 7).
- New `HANGLOOP_WATCH=1` (default) in tests/test_skip_hang_loop.py: one rt-profiler callback for the whole loop; per
  kernel set the most cores (printed each round, OVER above tt/cores.max_worker_cores; the test asserts none), per chip
  the last 6 programs done, printed on an error. Overhead: ~43 s a round vs ~32 s, 0 records dropped.
- It found 2 kernel sets on 120 cores every chunk: eltwise_binary_no_bcast (20 a chunk a chip, the MoE output add,
  moe_block_main_adapter.py, not capped since MOE_BLOCK_SRC=main) and eltwise_binary_sfpu_scalar (5 a chunk).
- Unfixed watch run hung in round 16, chip 7 dead: chips 4-7 / 28-31 never finished rid 37292 (ring_attention
  all_gather), the others ran on to 37294 / 37304: one chip lost, its CCL partners waiting (DS41-283, detail file).
- Fix (DS41-285, rule 18): sub_core_grids=sub_cores(..) on every stock binary op left uncapped in tt/: moe add (x2),
  attention mask lt (x4), compressor lt (x2), indexer eq + floor, engram maximum -> cores.maximum (where(ge) on the
  capped grid, bit-identical), sparse_attention off-path remap (op helper, floor, ge). Elementwise: same values.
- Proof: devjob fix36_fix1 (`HANGLOOP_PROFILE=0`, 30 min): 41 rounds = 7,544 chunks clean, 0 programs over 110,
  determinism max_abs_diff 0.0 in all 40 round comparisons. Unfixed: hangs at rounds 13 / 7 / 16. I count "20 clean
  runs" as 20 rounds of the loop (each = test_skip_positions' work for 2 points); 41 > 20.
- (3) the test fix from attempt 1 (DS41-284) stands.
- Re-run: `DS41_DECODER_SKIP=decode HANGLOOP_PROFILE=0 HANGLOOP_MINUTES=30 python -m models.demos.common.bringup.devjob
  start loop -- scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_skip_hang_loop.py -s`
- Gate `FIX.36 --check`: PASS (525 s): test_skip_positions all 6 points fit and ran (no hang), determinism 0.0.
  Cost of the caps, device ms (slowest chip) vs FIX.34: p51200 steady 81.0 (81.6), p256000 84.5 (85.7), p1024000 97.3
  (99.1); wall 107.2 (106.7) .. 123.2 (120.4): within run-to-run noise.
- Box report for the owner: the four chips with "only 2 routing planes" (phys 7 / 15 / 31 / 23) are not singled out:
  the dead chip was 15, 3, 7 in three hangs (two of them in that set, one not).
