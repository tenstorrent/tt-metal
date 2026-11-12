## P.20 perf (galaxy 8x4, attempt 1): SP8_RESWEEP, the SP=8 x TP=4 attention's matmuls swept on the Galaxy

- Change-on side: DS41_ATTN_SP32=off, ATTN_SP32_PROJ=off, INDEXER_TOPK_LOCAL=off, MATMUL_CONFIG=galaxy,
  ENGRAM_TABLE=resident. Off side: the three on (P.19's SP=32). No switch default changed (F104).
- (1) Off paths: the model ran unchanged with the three off, so nothing was fixed. That covers three warm runs in
  each capture process, at 51200 and 512000, and the quick measure. The off paths are P.15's load-time buffers, P.16's
  fork outputs and P.11's packing. No code change in tt/ besides the table.
- (2) Capture (test_capture, new P23_TAG=sp8 so P.19's calls_tuned*.json stay as they are): calls_sp8.json (51200)
  and calls_sp8_p512000.json are identical: 187 calls, 12 distinct, same configs, 1357 KiB free for CBs at every
  call. No dense matmul shape depends on the context. Sweep: test_sweep_card on 12 single cards in parallel, one
  distinct call each (P23_SHARD=i/12, P23_CALLS=calls_sp8.json, P23_NAME=_sp8, P23_MIN_MS=0). Merged with
  `P23_NAME=_sp8 merge_shards(12)`, giving sweep_sp8.json and galaxy_table_sp8.py. All shards rc 0 in about 3 min.
- tt/mm_configs.GALAXY: 7 SP=8 keys added (12 -> 19 entries). 5 SP=8 calls share a key with P.19's entries
  (engram_kv, hc.fn, indexer.wq_b, compressor 320 / 640 x 512 x 128). Their sweep winners equal P.19's entries, and
  those entries are unchanged. Acceptance is the same as P.19. All 7 are bit-identical to auto. One new winner:
  attn.wo_b 640x2048x5120, minimal M4 K8 N16 sb2x2, 0.0775 -> 0.0739 ms (-4.6 %). The rule is kept for the other 6.
  Cores (OV-31, no 110 cap in this pick): minimal 12x10 = 120 (wq_b, wo_b, q_a); 2D 11x10 = 110 (wo_a, compressor
  1024); 8x10 = 80 (compressor 512); 1x10 (weights_proj).
- (3) Per site, galaxy-tuned device kernel ms on one card. These are calls a chunk and ms a call / a chunk, at the
  5120-token chunk:
  | site | SP=8 shape (per chip) | calls | ms a call | ms a chunk | SP=32 shape (P.19) | calls | ms a call | ms a chunk |
  |---|---|---|---|---|---|---|---|---|
  | q_a | 640x1280 x 1280x1792 bfp8 -> bf16 (K split, + TP sum) | 40 | 0.0251 | 1.005 | 160x5120 x 5120x1792 | 40 | 0.0450 | 1.801 |
  | attn.wq_b | 640x1280 x 1280x8192 bfp8 -> bf16 | 40 | 0.0751 | 3.006 | 160x1280 x 1280x32768 | 40 | 0.1897 | 7.587 |
  | attn.wo_a | 2x640x4096 x 2x4096x1024 bfp8 -> bfp8 | 40 | 0.0593 | 2.371 | 8x160x4096 x 8x4096x1024 | 40 | 0.2060 | 8.240 |
  | attn.wo_b | 640x2048 x 2048x5120 bfp8 -> bf16 (+ RS) | 40 | 0.0739 | 2.957 | 160x8192 x 8192x5120 | 40 | 0.1506 | 6.024 |
  | indexer.wq_b | 160x1280 x 1280x4096 bfp8 -> bf16 | 8 | 0.0269 | 0.215 | same | 8 | 0.0269 | 0.215 |
  | indexer.weights_proj | 640x1280 x 1280x32 bf16 -> bf16 | 8 | 0.0093 | 0.075 | 160x5120 x 5120x32 | 8 | 0.0168 | 0.135 |
  | compressor | 640x1280 x 1280x1024 bf16 -> bf16 | 3 | 0.0199 | 0.060 | 160x5120 x 5120x1024 -> fp32 | 3 | 0.0431 | 0.129 |
  | compressor | 640x1280 x 1280x512 bf16 -> bf16 | 1 | 0.0151 | 0.015 | 160x5120 x 5120x512 -> fp32 | 1 | 0.0293 | 0.029 |
  | compressor | 320x512 x 512x128 bf16 | 3 | 0.0042 | 0.013 | same | 3 | 0.0042 | 0.013 |
  | compressor | 640x512 x 512x128 bf16 | 1 | 0.0067 | 0.007 | same | 1 | 0.0067 | 0.007 |
  | engram_kv | 640x6144 x 6144x6400 bfp8 -> bf16 | 2 | 0.2241 | 0.448 | same | 2 | 0.2237 | 0.447 |
  | hc.fn | 640x5120 bf16 x 5120x32 fp32 -> fp32 | 1 | 0.0298 | 0.030 | same | 1 | 0.0298 | 0.030 |
  Matmul total a chunk: SP=8 10.20 ms (tuned rule 10.44) vs SP=32 24.66 ms. SP=8's matmuls are TP partials (q_a,
  compressor, weights_proj: K split; wo_b) or column splits, so the TP reduce-scatters / gathers and the head <-> seq
  all_to_alls that SP=32 removed are not in these numbers. The orchestrator's profiles at 51200 / 512000 compare the
  whole chunk.
- Quick A/B (change on, `measure --task P.20 --part quick`): rc 0, all 102 limits pass. L39 err 0.8918 / 1.533
  (P.19 SP=32: 0.9144), min layer PCC 0.532 (L39). The tightest are err_state_topk_idx_L08 at 0.90 of its limit and
  window_kv L28 at 0.88. host_transfers_per_layer 0, chunk_wall_ms 374.2 (spread 70.3; P.19's quick on SP=32:
  301.9). No profile was run (brief).
- OV-3: nothing padded. OV-23: no Python-built op touched. OV-25: no fork suites run.
- Re-run: captures `DS41_ATTN_SP32=off DS41_ATTN_SP32_PROJ=off DS41_INDEXER_TOPK_LOCAL=off DS41_MATMUL_CONFIG=galaxy
  DS41_ENGRAM_TABLE=resident P23_TAG=sp8 [BRINGUP_POS=512000] scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/perf/test_matmul_sweep.py -k capture`. Sweep:
  `/scratch/dnijemcevic/p20/sweep.sh` (12 shards; logs in /scratch/dnijemcevic/p20/sweep). A/B: `python -m
  models.demos.common.bringup.orchestrator measure --task P.20`.

## FIX.15 overseer (galaxy 8x4, attempt 3): GALAXY_CORE_CAP, no op on more than 110 Tensix cores of a chip

Owner rule 18 (2026-10-30): a Galaxy's PSU trips on a chip's di/dt; at most 110 of the 120 cores an op. Attempt 2's
edits were in the tree (OV-34): kept, rebuilt (`./build_metal.sh` rc 0), finished. Attempt 1 / 2 stopped by the
overseer (P.20 revert, then the card-0 reset). The P.20 section above is the rejected pick's record, left as found.
- Setting `MAX_WORKER_CORES` (tt/settings.py, default 110): the cap on a Galaxy (cluster type GALAXY / TG /
  BLACKHOLE_GALAXY), the whole grid elsewhere; 0 = no cap. One module reads it: tt/cores.py (`worker_grid` 11 x 10,
  `max_grid()` for fork options, `sub_cores()` for stock `sub_core_grids`; None off a Galaxy, so other boxes run the
  calls as before).
- Forks, option `max_grid` (default None = today's grid; attempt 2): sparse_sdpa, rms_norm (both programs),
  mhc_pre_xing / mhc_pre_xing_pack, mhc_post (+ stats), rotary_embedding_indexed, update_padded_kv_cache (both
  forms), ckv_bfp4 / bfp_round (prepared path kept, OV-23), IndexerScoreProgramConfig `max_grid_x / max_grid_y`.
  New forks (stock op without a grid option): embedding, nlp_concat_heads, nlp_create_qkv_heads
  (nlp_create_q_heads_split), slice_write; README / CHANGELOG / INDEX written, tests/source.yaml skeletons (selection
  and baseline: F.1; no fork suite run here, OV-25).
- This attempt: (1) MoE block: `MoEOptions.max_grid` (moe_block.py; DS41 passes cores.max_grid): the fused head's
  engine + mover column inside 11 columns (HeadSE3 `max_grid`; planner: engine 10 x 10, MT 2, KB 10, predicted 231
  us), its x copy as a capped same-dtype typecast (ttnn.copy takes no grid), the bfp8 cast capped, the warm-up / router
  uploads host-built. (2) cores.maximum = where(ge(a, b), a, b) on the capped grid (bit-exact select; ttnn.maximum's
  binding takes no sub_core_grids; no fork of a binary op for 4 calls a chunk). (3) cores.typecast for ROW_MAJOR
  (tilize -> typecast -> untilize, capped; stock refuses a sub grid on RM): the indexer's block maxes.
  (4) served.slice_rows: no output_tensor under the cap (slice lands it with an uncapped ttnn.copy) and a whole-slab
  read as 2 slices + concat (a no-op slice of the ND-sharded slab is a whole-grid to_memory_config) (DS41-176).
  (5) cores.from_torch: host build + to_device under the cap (from_torch(device=) tilizes / typecasts on the whole
  grid), every tt/ upload; served._zero_fill_capped: DRAMZeroFill's kernel on 11 x 10, prepared (DS41-177).
  (6) compressor divide got its sub_core_grids.
- Audit (tests/perf/test_core_audit.py, real-time profiler core_count per program; DS41-179): before (attempt 2,
  01:09, today's code) 77 (section, op, shape) rows over 110 (120 cores; ring_indexer_score_dsa 114); after:
  0 over in the warm chunk AND in the load + compile chunk (`CORES_LOAD=1 CORES_AUDIT_STRICT=1`: PASS, max 110).
- Matmuls: capped program_config keys on grid 11 x 10, so the P.19 12 x 10 table entries miss on the Galaxy (kept for
  an uncapped 12 x 10 grid). Capture on the capped code (calls_cap110.json: 187 calls, 12 distinct, 1358 KiB free for
  CBs), swept on the 8x4 MESH (OV-34; `test_sweep` now holds candidates to cores.worker_grid): 12 calls in 5:44, all
  bit-identical to auto. ms a call, P.19 12 x 10 table -> 11 x 10 rule -> 11 x 10 best (now in GALAXY):
  q_a x40 0.0450 -> 0.0551 -> 0.0474 (1d bw16); attn.wq_b x40 0.1897 -> 0.1972 (rule); attn.wo_a x40 0.2060 ->
  0.2062; attn.wo_b x40 0.1506 -> 0.1364; engram_kv x2 0.2237 -> 0.2806 -> 0.2284 (minimal M4 K16 N8); compressor /
  indexer / hc.fn within 0.001. Swept total 24.66 (P.19) -> 24.52 ms a chunk.
- Per op (audit, one warm 5120-token chunk, kernel ms a chunk, all chips max; cores before -> after): attention
  linear 120 -> 110, 13.87 -> 13.52; sparse_sdpa 120 -> 110, 11.90 -> 11.61; rotary (attention) 120 -> 110, 4.90 ->
  4.96; mhc_pre_xing x2 120 -> 110, 8.06 -> 8.00; mhc_post x2 120 -> 100 / 110, 7.65 -> 7.73; nlp_concat_heads 120 ->
  110, 2.47 -> 2.58; nlp_create_q_heads_split 120 -> 110, 1.98 -> 2.04; slice_write 120 -> 110, 0.85 -> 0.90;
  ring_indexer_score_dsa 114 -> 104, 3.02 -> 3.31; topk_large_indices 120 -> 110, 1.48 -> 1.49; rms_norm (q_a, ffn,
  kv) 120 -> 110, unchanged; engram_kv minimal_matmul 120 -> 110, 0.455 -> 0.462; attention RM untilize
  (to_layout -> untilize_with_unpadding) 120 -> 107, 4.03 -> ~5.1; attention typecast 120 -> 80, 1.52 -> 1.81; q_a
  typecast 120 -> 100, 0.29 -> 0.55; experts x copy + bfp8 cast 120 -> 100, 0.67 -> 1.21 (DS41-180); MoE fused head
  120 -> 110 (experts section unchanged); flat_routed_expert 109 (unchanged). Full table:
  /scratch/dnijemcevic/fix15/per_op.md; JSON generated/ds41_cores/cores_before.json, cores_after_ms.json.
- Chunk (DS41_ENGRAM_TABLE=resident; before = P.19's shipped side): rung (gate profile) device 191.63 -> 195.03 ms,
  warm wall 302.5 -> 303.7; p51200 device 201.47 -> 205.54, wall 306.4 -> 302.6; p512000 device 239.13 -> 241.32,
  wall 308.6 -> 306.7 (indexer 43.74 -> 46.89 there: the ring indexer score on 104 cores). Gate FIX.15 --check PASS
  (112 checks; err_final_hidden 0.18973 (P.19 0.18954), top5 1.0, contract 0 failed, acks_early 0, smoke 1).
- Not done / open: a re-tune of the capped stand-ins (sub-grid typecast, untilize_with_unpadding, ~2 ms a chunk) and
  of the ring indexer score's work split at 110 (it lands on 104); galaxy_4x4 not run (the fused head at 1280 rows a
  row on 10 engine columns is unverified there; the planner raises if no layout fits). tt-probe.sh mesh mode reset
  the whole box after its (passing) probe (DS41-178): one probe this attempt, then only run_safe_pytest.
- OV-3: nothing padded. OV-23: the one Python-built program added (capped zero fill) runs prepared; ckv_bfp4 stays
  prepared. OV-25: no fork suite run. OV-34: no single-card job; the audit, the probe and the sweep on the 8x4 mesh.
- Re-run: audit `CORES_LOAD=1 PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/perf/test_core_audit.py`; sweep: `-k capture` on test_matmul_sweep.py,
  `cp generated/ds41_p23/calls_galaxy.json generated/ds41_p23/calls_cap110.json`, then `P23_CALLS=calls_cap110.json
  P23_MIN_MS=0 ... test_matmul_sweep.py -k "test_sweep and not card"`; positions: `DS41_ENGRAM_TABLE=resident
  BRINGUP_PROFILE_WALL=1 BRINGUP_POS=<p> BRINGUP_PROFILE_HOST=1 ... test_profile.py`, then `BRINGUP_WALL_CLEAN=1` the
  same; gate: `python -m models.demos.common.bringup gate FIX.15 --spec <spec> --check`.

## P.18 WINDOW_TILES (perf, galaxy 8x4, 2026-10-31): a per-chip window region, one neighbour send

- Setting WINDOW_KV_FORMAT gains `bfp8_tiles` (the default; `bfp8_blocks` = the path before, unchanged). Per layer, every
  chip keeps its OWN window region `TtWindowAttention.region` [1, 1, (128 + q_rows) / 32, 18496] uint8 (8x4: 288 rows,
  9 pages): [carry 128 | its own 160 query rows] as bfloat8_b pages of one tile row (32 rows), the same 16-value
  exponent groups and nearest-even rounding as bfp8_blocks (ckv_bfp4 blocks=True), so the same values.
  Per chunk (`tt/attention._window_region`): (1) the served 0_window_kv write unchanged (TP gather + local write at
  chunk-aligned starts; the full mesh gather + server-order write at other starts); (2) the chip's own rows (pad rows
  zeroed by a per-chip iota) rounded into region rows 128.. (one ckv_bfp4, 5 cores); (3) chip 0's carry: the slot's
  kept tail `TtWindowCache.rtail` when this layer's previous chunk ended at `start`, else the served rows packed into it
  - on the last chip alone (its local slice, no gather) whenever the carry's owner is the last chip (every
  chunk-aligned start), else `cache.carry` (gathers; non-aligned prefix hits); start 0 reads none; (4) ONE neighbour
  send (`tt/window_send.py:WindowSend`, a Python fabric program, one core a chip, prepared): chip q's last 4 pages ->
  chip q + 1's first 4, the last chip's kept tail -> chip 0, then the last chip keeps its own last 4 pages.
  sparse_sdpa (unchanged fork, ROW_BFP8 / ROW_BFP4 kv_window) reads the region with load-time per-chip index lists
  (`WindowTables.region_lists`: query d reads region rows d - d0 + 1 ..; positions < 0 dropped for chip 0 at starts
  < 128). Removed a layer: the SP gather of the blocks, its slice_write, the carry gathers (TP + SP), their slices.
- Deviation from the brief (owner please read): the region is stored as 32-row block pages, not TILE-layout tiles,
  and the sparse_sdpa fork is unchanged (no band-read option): a query's 128 keys start mid tile, so a band read from
  tiles needs a 5-tile window chunk (the kernel's Skt = 4 is compile-time across its softmax code) or a tile -> row
  re-tile in compute; the block page is a tile row's bytes row-major, read with 2 NoC reads a block (~70 KB a query
  vs 87 KB of tiles + a convert) and the existing reader already reads the window as a consecutive band from its first
  id (DS41-182). Same values either way.
- No ready fence in the send: the region is per layer and every layer's MoE exchange orders a chip's next write to a
  receiver after the receiver's read (tt/window_send.py docstring). Routes from tt-metal's dimension-order tables with
  the open fabric's torus wraps (combine_rf route / mesh_plan); 8x4 torus: (r,3) -> (r+1,0) is Y+ then X+ (wrap).
- Byte test `tt/perf_window_tiles.py` (8x4 mesh, regions of 9 and 24 pages, 3 prepared calls, io patched): PASS;
  host enqueue a warm call 220-300 us (32 per-chip programs). Rule 18: send 1 core, own-row ckv_bfp4 5 cores; core
  audit (`CORES_TAG=_p18`, generated/ds41_cores/cores_p18.json) PASS, max 110.
- Accuracy: measure quick (last chunk after the golden prefix, every layer vs golden): every layer and state within
  its limit; L00 0.025307 (= FIX.15), L03 0.07673, L20 0.16442, L39 0.85131 (P.19 0.91435); test_runner_contract
  (prefix hit at 2496, non-aligned start) PASS, 0 failures (gate unit P.18#test_runner_contract).
- Speed (this run, change on; before = FIX.15's shipped side, its BREADCRUMBS numbers): quick wall 280.5 ms (rung,
  FIX.15 303.7); p51200 device 205.54 -> 193.32 ms, wall 302.6 -> 285.2; p512000 device 241.32 -> 234.07, wall
  306.7 -> 289.1. Section attention 66.6 -> 59.24 ms (p51200; P.19-side profile, the same window path as FIX.15).
  Per op (core audit, calls a chunk / cores; the audit recorded no device ms for fork ops this run): attention
  fabric_all_gather 160 -> 40 (the served write's TP gather stays), slice_write 80 -> 0, slice 120 -> 40,
  ckv_bfp4 80 -> 80 on 20 -> 5 cores (own 160 rows + the carry instead of the SP row's 640 + the carry), window send
  +40 (1 core). Host (cProfile, p51200, a chunk): window path 60.6 -> 36.4 ms, all_gather 294 -> 174 calls (39.3 ->
  24.7 ms), the send 3.8 ms / 40 calls; ttnn op launches 2543 -> 2282.
- Window memory a chip (8x4, chunk 5120): before (bfp8_blocks + P.11) window blocks 164 pages 3.03 MB + packed
  blocks 20 + 160 pages 3.33 MB + carry buffers 148 pages 2.74 MB = 9.10 MB shared, + 4 pages (74 KB) a layer a slot;
  after: regions 9 pages a layer = 6.66 MB (40 layers) + index lists 0.41 MB, + the kept tail 74 KB a layer a slot.
- OV-3: nothing padded. OV-23: the one Python-built program added (the send) runs prepared (patch count checked,
  2 slots a chip). OV-25: no fork suite run (no fork changed). OV-35: DS41-175 updated.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_window_tiles.py -s`; `python -m models.demos.common.bringup.orchestrator
  measure --task P.18 --spec <spec>`; old path: DS41_WINDOW_KV_FORMAT=bfp8_blocks.
- Final check (after the hooks reformatted window_send.py): measure quick on the final code: wall 278.2 ms (spread 7.5),
  every layer and state within its limit, layer errors identical to the first run. Its p51200 / p512000 position runs
  were cut: my shell moved the job to the background and killed it at its 10 min limit; SAFE_PYTEST caught the SIGTERM
  and marked cards 0-31 dirty (09:05). No device job run after it (agent rule 19): the next job resets the box
  (DS41-183). The position numbers above are from the first measure (same code before black's reformat).

## FIX.16 (overseer, galaxy_8x4): steady_state(start) on the device model (framework F236)

- What: `TtDS41Model.steady_state(start)` (tt/model.py) -> `TtWindowCache.steady(start)` (tt/attention.py) on every
  layer of the active state. With a start (before each TIMED profile / wall / cProfile run) it marks the kept window
  carry as the tail of a whole previous chunk ending at `start`, in the load-time buffers that chunk would leave it in:
  bfp8_tiles (default) `rtail_end = start` (the last chip's neighbour send rewrites `rtail` every chunk); bfp8_blocks +
  P.11 `_tail_blocks = (start, _tail_buf)`; GLUE_FOLD bf16 rows re-keyed only when kept (nothing allocated). With None
  (before an accuracy run) every kept tail is dropped: the carry is rebuilt from the served cache, as before. Host
  only, no device op, nothing allocated. The carry's values are the last run's tail, not [start - 128, start): timing
  only. The ladder and the contract never call it.
- Why: every timed run repeats the same chunk, so step 3 of `_window_region` (slice 1760x512 + to_layout 128x512 +
  ckv_bfp4 into rtail) ran in every layer although consecutive serving chunks skip it (owner 2026-10-31).
- Switch: `STEADY_STATE` on (default) / off (no-op; the old timed path, comparison only), tt/settings.py.
- Measured (p51200 profile rung, chunk [51200, 56320), 40 layers; off = `DS41_STEADY_STATE=off`, same code, ad-hoc
  results /tmp/fix16_off; on = this gate): clean wall median 279.1 -> 262.5 ms (-16.6; owner estimate 16.3 host);
  device sections attention 59.10 -> 57.49 ms (-1.62), others within +-0.2 (experts 87.04 / 87.13, indexer 7.23 / 7.24);
  gate device_ms_total 184.64.
  Op counts (per layer section, ops file): attention 22 -> 19 rows a layer: exactly `bringup.ckv_bfp4` (the carry's),
  `slice`, `to_layout` gone in every layer; 407 of 416 layer sections op-for-op identical, the other 9 are layer 1's
  first sections that the off run's real-time profiler attributed to one "(other)" row (attribution artifact of that
  run). ckv_bfp4 a chunk 80 -> 40 (on 5 cores each, unchanged; no new op, rule 18 n/a).
- Gate: PASS FIX.16 (P.17#test_ladder, P.17#test_profile; ladder errors unchanged, e.g. err_final_hidden 0.18973,
  top5 1.0; chunk_wall_ms 262.5 < 434, device_ms_total 184.64 <= 230), 1333 s.
- OV-23: no Python-built op added or touched. OV-25: no fork suite run. OV-3: nothing padded.
- Note: profile units take ~12 min, over the 10-min foreground tool limit: device jobs ran as background tasks with a
  2 h limit, waited on (never two at once).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.16 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`; baseline:
  `DS41_STEADY_STATE=off BRINGUP_RESULTS_DIR=/tmp/x BRINGUP_RECORD_TASK=off BRINGUP_WALL_CLEAN=1 DS41_ENGRAM_TABLE=resident
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_profile.py` (after the
  BRINGUP_PROFILE_WALL=1 run).

## P.21 perf (galaxy 8x4, attempt 2): CHEAP_BUNDLE (#9 HOST_GLUE, #7 NORM_INTO_MOE, #8 NORM_GEMM_DTYPE; #10 deferred)
Attempt 1 left no changes in the tree. Owner pick: opportunities #9, #7, #8, #10 (report /scratch/dnijemcevic/opportunities/report.md).
Each item has its own setting (tt/settings.py, default on, off = the path before P.21).
- **HOST_GLUE (#9)**: `core/model_settings.Settings` gained `snapshot()` / `thaw()` / `refresh()`; with no snapshot, get() reads
  the environment as before, so every other model is unchanged. TtDS41Model.__init__ thaws, then snapshots when HOST_GLUE=on, so
  settings.get serves the load-time values with no os.environ read (~1500 calls a chunk). tt/mm_configs.py `_site_config`
  memoizes each call site's program config while the settings are frozen. The key is (id(w), id(ckc), dtypes, batched, rule),
  checked against the identity of w and ckc and against x.shape (DS41-88: ids alone can be reused). It is cleared when the
  snapshot changes. tt/perf_flat_program_key.py calls settings.refresh() after its monkeypatch.setenv (DS41-185). 0 ops removed.
  Host estimate from the report: ~4.7 ms (get) + ~3.9 ms (program_config) cum.
- **NORM_INTO_MOE (#7)**: ffn_norm (column split, bf16, the fork) writes `MoEBlock.input_buffers()`: the fused head's
  persistent x, plus its bfloat8_b xq through the new rms_norm option `bfp8_twin`. The block reads them in place:
  `forward(x, xq_ready=True)` skips the head's bf16->bf16 copy and its bfp8 typecast, and asserts that x is its own buffer
  when xq_ready is set. Both buffers are pinned (ccl.pin), so the block's `_free` leaves them alone. The bytes are the same
  (bfp8_twin = typecast(bf16 out, bfp8) bit for bit, DS41-184). 2 ops a layer, 80 a chunk; DS41-180 measured these two
  capped ops at 1.21 ms of device a chunk. moe_block.py changes: `input_buffers()`, `forward(xq_ready=)`,
  `FusedHead.__call__(xq_ready=)`; the defaults are the old behaviour.
- **NORM_GEMM_DTYPE (#8)**, ATTN_SP32 path only. A step's device readers come from block_steps (kv reads q_a's projection).
  - attn_norm: tried bfloat8_b where q_a alone reads it (32 layers), then dropped. With dtype=bfp8 the norm leaves the fork's
    default program for the split program's LOCAL mode, one core a row tile (5 cores for 160 rows): device_ms_attn_norm
    went 2.6-3.9 -> 5.58 ms a chunk (p51200), against ~0.45 ms for the q_a typecasts it removed (DS41-186). attn_norm stays
    bf16 + typecast.
  - q_norm, in those 32 layers: gamma x 512^-0.5 folded in fp32 at load, writes bfloat8_b. attention._q_stock takes it as
    wq_b's input (`q_prescaled`), so the q scale multiply goes.
  - q_norm, in the 8 indexer layers: writes bf16 (the attention scales it as before) plus a bfp8_twin into one pinned
    buffer (q_a._twin_buffer). The indexer reads that as wq_b's input (`qr_twin`), so its typecast goes.
  - Both bfp8 norm outputs use the new fork option `pack_precise`: one rounding fp32 -> bfp8 instead of fp32 -> bf16 -> bfp8
    (DS41-137). The twin is bit-identical to the typecast it replaces. 40 ops a chunk (32 q multiplies + 8 indexer typecasts).
- **rms_norm fork** (split program; CHANGELOG "pack_precise and bfp8_twin"):
  - pack_precise sets the compute config's bfp8_pack_precise, plus the define RMS_SPLIT_PACK_PRECISE so the binary gets its
    own cache key.
  - bfp8_twin: in pass B, an SFPU rounds y to bf16 ties-away into the free DEST half (gamma's slots; DEST block halved
    without gamma). It is packed to CB c_8 (bfp8) and written at the same pages.
  - tests/unit/test_rms_norm_ttnn_twin.py: 7 passed on the 8x4 (FABRIC_2D, 11 x 10 grid), bit-exact against the typecasts,
    with negative controls and refusals. No fork regression suites were run (OV-25). Cores: max_grid, 110.
- **TINY_KERNELS (#10) not done**: the compressor _pool and the engram _gate_math each need a new Python-built op on the
  prepared path (OV-23).
  - _pool: a reader that re-lays row pairs (2r, 2r+1) of the TILE input into even / odd tiles (64-byte face-row segments
    via L1), then the SFPU max / exp / div / mul chain.
  - _gate_math: a reader that moves columns 4-7 / 8-11 of the [R, 32] partials to columns 0-3 (word copies in L1), then
    rsqrt / sqrt / sigmoid on the SFPU.
  - It is not a fit for mhc_pre_xing_pack's engram mode: the gates need the TP-summed partials.
  - Deferred: a kernel hang on the Galaxy costs a whole-box reset, and it is the smallest item in the bundle (~0.4 ms
    device / ~8.5 ms host, report). Its setting is not added (no no-op switch).
- measure --task P.21 with attn_norm bfp8 still in (code c5e0d552): every part rc 0. quick: all limits pass, wall 258.7 ms.
  p51200: wall 253.7 ms, device 194.48 ms, host cProfile 270.3 ms, 2010 launches; P.18 new was 2282, though the runs
  between differ. p512000: wall 286.5 ms, device 231.71 ms. Then attn_norm went back to bf16 (above); the final measure
  is the next bullet.
- **Final measure** (`measure --task P.21`, change on, code 8ff681ee, attn_norm bf16): rc 0 on every part.
  - quick: every err_* within its limit (worst err_state_topk_idx_L08 at 0.903), min layer PCC 0.725 (L39),
    host_transfers_per_layer 0; wall 321.9 ms with spread 487 (one outlier, the quick run's own).
  - p51200: wall 254.1 ms, device total 190.79 ms (attn_norm 3.20, q_a 3.44, ffn_norm 3.43, experts 93.3), host cProfile
    269.8 ms, 2042 launches.
  - p512000: wall 270.7 ms, device 231.35 ms.
  - Gate limits: wall < 434, device <= 260. The off side (all three switches off) is the orchestrator's A/B.
- Earlier accuracy run, quick (change on, measure --part quick, code d7193162, before the clang-format / TINY_KERNELS-removal edits):
  rc 0, every err_* within its limit (worst err_state_topk_idx_L08 at 0.902 of its limit; P.20's was 0.901). min layer
  PCC 0.599 (L39). chunk_wall_ms 259.5 (spread 22.1). host_transfers_per_layer 0. Not profiled (brief: the orchestrator
  measures off vs on at 51200 / 512000).
- Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.21` (quick);
  `scripts/run_safe_pytest.sh --mesh --run-all ttnn/ttnn/bringup/rms_norm_ttnn/tests/unit/test_rms_norm_ttnn_twin.py`.

## P.22 perf: ROPE_CHAINS (opportunities #2 and #4 as one pick), 2026-10-31
- **What**: the three RoPE chains as one call each of `ttnn.bringup.rotary_embedding_indexed` in a new chain mode
  (fork options `input_heads`, ROW_MAJOR input, `output_layout`, `output_dtype`, `group_heads`; a second device
  operation `device/rope_chain_device_operation.cpp` with its own kernels `device/kernels/chain/`). Setting ROPE_CHAINS
  (default on; off = the path before P.22). Model: `tt/rope.py:rope_chain` / `rope_chains`, `tt/attention.py`
  (`_q` sp32 branch, `_out` sp32 branch; `_q_stock` split into `_q_proj` + split/rope, `_out_proj` into concat/cast +
  `_out_wo`), `tt/indexer.py:_query_chain` (scores skips `_round` when the query is already bfloat4_b),
  `tt/host_alloc.py` (no q / o rope output buffers when on).
  - #4 attention q: wq_b [1, 1, 160, 64 x 512] bf16 TILE -> (heads read in place, tail roped, untilized) -> sparse_sdpa's
    [1, 64, 160, 512] ROW_MAJOR q. Removes nlp_create_q_heads_split + to_rm (untilize): 2 ops a layer, 80 a chunk.
  - #2 attention output: sparse_sdpa's ROW_MAJOR output -> (tilized, inverse rope, packed bfloat8_b, 8-head groups) ->
    wo_a's [1, 8, 160, 4096] input in the HOST_ALLOC "attn.oa" buffer. Removes to_layout + nlp_concat_heads + typecast
    (+ 2 reshape views): 3 ops a layer, 120 a chunk.
  - #4 indexer q: wq_b [1, 1, 160, 32 x 128] -> bfloat4_b [1, 32, 160, 128] (RNE, bfp_round's SFPU rule, into the
    LOWBIT_HOST query buffer). Removes split + bfp_round: 2 ops a layer, 16 a chunk (8 indexer layers).
  - Total: 216 launches a chunk fewer. No sparse_sdpa option was needed: the heads <-> tokens transposition between
    sparse_sdpa (32 heads of one token a tile) and RoPE / wo_a (32 tokens of one head a tile) stays in the ROW_MAJOR
    tensor; the rope op tilizes / untilizes it in its compute kernel.
- **Precision**: unchanged (bit-identical). Fork tests `tests/unit/test_chain.py`: 8 passed on the 8x4, each fused call
  equal to the unfused chain on every chip, miss and hit. Model: `tt/perf_rope_chains.py` (on / off alternating, one
  process, profile rung's last chunk): every run's final hidden state bit-identical, pcc vs golden 0.725092 both sides.
- **Gotcha (DS41-137 used)**: the first bfp8 output differed from ttnn.typecast by one bfp8 step (max |d| 0.0625): the
  packer without bfp8_pack_precise rounds twice; typecast packs precise. The chain compute sets bfp8_pack_precise for
  a bfloat8_b output plus a define (CHAIN_BFP8_PACK_PRECISE: the kernel cache key leaves the flag out).
- **Host design**: a ProgramDescriptor factory with plain uint32 addresses in common args and a factory-level
  override_runtime_arguments (DS41-187: an output_tensor in tensor_args would push the binding fast path to a full
  rebuild per hit); block split from [per, extra] common args and the core's logical (x, y) (as ckv_share, DS41-145:
  no per-core runtime args). One program per mesh range (DS41-147).
- **Cores**: every chain call on max_grid 11 x 10 = 110 cores (attention: 320 blocks a chip; indexer: 160).
- **Measured** (not the gate's profile): q chain host a call 259.9 us (3 ops) -> 61.4 us (fused). Model warm wall,
  same process, median of 3: off 273.8 ms, on 250.1 ms (-23.7 ms a chunk). Device ms a chunk: the orchestrator's A/B.
- stock `ttnn.to_layout` (to_rm) instead of the chain is not used on the default path any more; the off path keeps it.
- Re-run: `scripts/run_safe_pytest.sh --run-all --no-precompile ttnn/ttnn/bringup/rotary_embedding_indexed/tests/unit/test_chain.py -k 8x4`;
  `DS41_ENGRAM_TABLE=resident DS41_ROPE_CHAINS=on scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tt/perf_rope_chains.py -s`;
  `python -m models.demos.common.bringup.orchestrator measure --task P.22`.
- **Final measure** (`measure --task P.22`, change on: ROPE_CHAINS=on ENGRAM_TABLE=resident): rc 0 on every part.
  (A first attempt was killed by the agent tool's 10-minute background limit after its quick part; rerun in full.)
  - quick: every err_* within its limit (99 checked; worst err_state_topk_idx_L08 at 0.903, as P.21), min layer PCC
    0.725 (L39, as P.21), host_transfers_per_layer 0, chunk_wall_ms 232.3.
  - p51200: wall 235.1 ms, device total 179.55 ms (attention 47.16, indexer 7.06), host cProfile 246.2 ms, 1746
    launches. P.21's final measure (before this change): wall 254.1, device 190.79, cProfile 269.8, 2042 launches.
  - p512000: wall 253.3 ms, device 219.0 ms (P.21: 270.7 / 231.35).
  - The off side of the same run is the orchestrator's A/B; the P.21 numbers are another run.

## P.23 perf (galaxy 8x4, attempt 1, 2026-10-31): OWN_ROWS, the residual / mHC streams on each chip's own 160 rows (opportunity #1)
- **What**: setting OWN_ROWS (tt/settings.py, default on; off = the column split, the path before P.23;
  tt/common.own_rows: needs ATTN_SP32, MHC_FUSED, GLUE_FOLD, MOE_IMPL=block, MOE_HEAD=fused). The block boundary,
  h_mid and the pending pre are [1, 1, S/(SP TP), 4 x H] / [.., 24] on chip (r, c) = row block r TP + c (the rows
  sparse_sdpa reads and the MoE block's owner rows; tt/layout.own_rows_to_device / _to_host for the harness).
  - mHC (tt/mhc.py `rows=`): hc_fn^T whole on every chip (stream-major), the partial row is the whole row: no
    fabric_all_gather of the [S/SP, 32] rows, mhc_pre_xing without `partials`; mhc_post(stats_weight=) emits the whole
    row. Removes 2 gathers a layer.
  - attn_norm: no cols_to_rows all_to_all (the collapse already writes the chip's rows); the attention keeps its rows
    (TtWindowAttention.keep_rows: no exit all_to_all). Removes 2 a2a a layer.
  - ffn_norm (tt/norm.py build_norm rows=True for ffn_norm): one local whole-row rms_norm in FFN_NORM_DTYPE; removes the
    stats-only rms_norm and the stats all_gather (2 a layer). NORM_INTO_MOE does not apply (the norm writes its
    FORK_OUT_BUFFERS site; the head's a2a writes the head's x).
  - MoE (models/demos/moe_block, behind options, defaults unchanged): the block's existing `MoEOptions(residual="owners")`
    (Marko's: the fused head fed through one TP all_to_all [S/N, H] -> [S/SP, H/TP]) plus two new options:
    `owners_a2a_into_head` (the a2a writes the head's persistent x buffer, persistent_output_buffer, BEFORE the arena
    is taken: no copy) and tail "fused" with owners = `TailA2A(local=True)` (tt/tail_a2a.py: the workers' tilize + the
    shared expert's add straight into [S/N, H], no ports / all-to-all, one program for the mesh, workers inside
    max_grid: 20 cores). The ops tail's add got sub_core_grids from max_grid (rule 18). tt/prepared_tail.py follows
    TailA2A's `io` / `descriptor` / `out_shape`; tt/host_alloc.reserve_tail_out reserves [S/N, H].
    Added a layer: the head's a2a + the head's xq typecast (P.21's bfp8 twin had folded it into ffn_norm).
  - Engram (layers with engram): tt/engram_kv.py own_rows = chip column c holds the BLOCK-major columns
    [c 5H/TP, (c+1) 5H/TP) of [key 0..3 | value], so ONE a2a (concat columns, split rows) of the unchanged [640, 6400]
    product lands the chip's own [160, 5 x 5120] in block order (same weight bytes, no replicated wkv); tt/engram_add.py
    own_rows: whole weight, gates from the whole rows (no exact tp_sum).
  - Embedding: one cols_to_rows a2a a chunk; final norm local (split=False), to_host concatenates the chips' rows.
- **Not done (owner's sketch)**: the x_gather3 row-shard variant in the fused head (the head gathers [160, 5120] row
  shards instead of an a2a to the width split). x_gather3 / se_engine3 / router_stream consume the gather by K shard
  (XREADY per K step, the engine loads its own K columns from DRAM, the router computes partial logits from the chip's
  H/TP columns): a row-shard gather changes the engine's pipeline, the watcher protocol and the router's partial-logit
  exchange together (~5000 lines of fused kernels). Used instead: the owners path's one TP a2a into the head's x (the
  same bytes as the attention's old exit a2a) + the xq typecast: 2 launches a layer the variant would save.
- **Precision**: no lowering. Only summation orders change: the mHC mixes / sums of squares and the ffn / attn norm
  statistics are full-K fp32 sums on one chip instead of TP partials added in mesh order; the engram gate sums likewise.
  The MoE (head, router, experts, combine) sees the same x bytes as before.
- **Measured** (`measure --task P.23 --part quick`, rc 0, change on): every err_* within its limit; L00-L32 equal to
  P.22's on-side to 3 digits, L33-L38 +0.8..1.2 %, L39 0.700 -> 0.758 (limit 1.533; pcc 0.725 -> 0.684): rounding,
  DS41-188 (L39 moved 0.83 -> 0.72 at P.10, 0.87 -> 0.91 at P.17, 0.85 -> 0.70 at P.21 on rounding-only changes);
  host_transfers_per_layer 0; chunk_wall_ms 212.9 (P.22 on-side quick: 232.3; spread 94.9).
  Launches a chunk (count, not profiled): -240 (80 mHC row gathers, 80 attention a2a, 40 stats rms_norm, 40 stats
  gathers) +80 (40 head a2a, 40 xq typecasts) -2 engram exact TP sums +2 engram a2a +1 embedding a2a: ~-160; the fused
  tail's a2a work is gone from its program. Device / host ms a chunk at 51200 / 512000: the orchestrator's A/B (the
  brief: no profiling by the agent).
- **Tests**: models/demos/moe_block/tests/test_tail_a2a_local.py (local mode vs torch to a bf16 ulp, eager and
  prepared, with / without se, H 5120 / 4096): 8 passed on the 8x4 (MOE_MESH=8x4 MOE_FABRIC=torus_xy).
- **Cores**: TailA2A local 20 workers (11 x 10 cap); every other op unchanged or already capped (mhc / norm forks
  max_grid; a2a CCL workers as before).
- Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.23`;
  `MOE_MESH=8x4 MOE_FABRIC=torus_xy scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/moe_block/tests/test_tail_a2a_local.py`.
- Stock calls touched (F113): stock ttnn.add instead of ttnn.bringup.combine_rf: the MoE block's owners ops tail add
  (an existing call, now given max_grid's sub_core_grids; DS41 runs the fused local tail instead). The new a2a calls
  (engram_kv, embedding) are stock ttnn.experimental.all_to_all_async_generic, as tt/common.cols_to_rows (no fork lists it).
- Gate unit `P.23#test_ladder` (rung last, `gate P.23 --check --unit P.23#test_ladder`): PASS, worst layer pcc 0.6836
  (L39), final hidden pcc 0.98197 rel 0.1902, top1 0.9971, top5 1.0000.

## P.24 perf (galaxy 8x4, attempt 1, 2026-10-31): TINY_KERNELS, the compressor pooling and the engram gate math as one fork program each (opportunity #10)
- **What**: setting TINY_KERNELS (tt/settings.py, default on; off = the op chains, the path before P.24).
  - Compressor (tt/compressor.py `_pool`): the ratio-2 pair pooling (reshape, 4 slices, maximum, 2 subtract, 2 exp,
    add, 2 divide, 2 multiply, add: 16 launches a layer on [S/SP, 1024] fp32) is ONE call of the new fork
    `ttnn.bringup.pair_pool` (ttnn/ttnn/bringup/pair_pool_ttnn, Python ProgramDescriptor op, registered in PYTHON_OPS).
    Prepared always (prepare_program / run_prepared, the 2 address slots patched; OV-23), common runtime args only (no
    per-core args, DS41-145), one program for the mesh range (DS41-147); output = a load-time buffer
    (host_alloc.reserve_forks, site `fork.compressor.pool`, DS41-146) when FORK_OUT_BUFFERS=on. Taken when p is fp32 and
    rows % 64 == 0 (every box: S/SP = 640 / 1280 / 2560), else the chain.
  - Engram (tt/engram_add.py): with MHC_FUSED=on and OWN_ROWS=on (the pass's sums are whole rows) the engram pass
    `mhc_pre_xing_pack` takes the new option `engram_gate_eps` and writes the gates into columns 12..15 of its row; the
    add's `mhc_post` reads them with `post_col=12` (comb_col 0 on the identity [R, 16]). Removes _gate_math (3 slices,
    2 x (multiply, add, rsqrt), 3 multiply, abs, maximum, sqrt, neg, ltz, where, sigmoid: ~19 launches a layer). With
    OWN_ROWS=off (TP partial sums) the gates stay the chain (the epilogue would see partial sums).
- **Fork changes**: mhc_pre_ttnn (C++, rebuilt): option `engram_gate_eps` (default None: the same program, args, CBs,
  hash). On: the writer lays the partial rows coefficient-major (column k = SFPU vector k, lane = token row), the root
  sums them as before (lane-wise, so columns 0..11 are the option-off bytes), the compute runs the gate steps on the
  summed tile, the writer re-lays the row row-major in its free stage page. New fork pair_pool_ttnn.
- **Precision**: none changed. Both kernels use the stock ops' per-vector SFPU bodies (`_calculate_sqrt_body_` rsqrt /
  sqrt, `_sfpu_sigmoid_<true>`, `_sfpu_exp_accurate_<true>`, the fp32 SFPU divide body) in the chains' order, each
  step one fp32 rounding (results stored to DEST between steps: no fused multiply-add). Result: bit-identical to the
  device op chains (unit tests: 0 ulp on every value) and the quick A/B's 103 err metrics equal P.23's change-on side
  to the last digit (L39 0.7581, L38 0.1650; worst err_state_window_kv_L28 at 0.911 of its limit; 102 limits pass;
  host_transfers_per_layer 0).
- **Gotchas**: DS41-189 (SFPI internal compiler error on a runtime-indexed DEST loop around the sqrt body: one template
  instance per stream); DS41-190 (sqrt / rsqrt and the reciprocal share a programmable constant register: two phases,
  re-init between); the gate scratch vectors (columns 16..18) must be zeroed before the pack; ttnn.Shape has no slice
  `__getitem__` (index it).
- **Cores**: pair_pool min(units, 110) = 110 a chip on the 8x4 (160 units: 10 tile rows x 16 column tiles, max_grid
  11 x 10); the engram pass unchanged (its max_grid).
- **Host**: pair_pool 75.8 us a warm call on 32 chips (prepared, output buffer; test_pair_pool_prepared_and_output).
- **Measured** (`measure --task P.24`, change on: TINY_KERNELS=on ENGRAM_TABLE=resident; rc 0 on every part):
  quick wall 194.0 ms (spread 111.6). p51200: wall 203.3 ms, device total 169.67 ms (compressor 1.78, engram_add 0.56),
  1482 launches (host_launch_ms 123.8, cProfile 208.5). p512000: wall 226.4, device 208.34 (compressor 1.78,
  engram_add 0.56), 1482 launches. P.23's change-on side (another run): device 172.01 / 207.73, compressor 2.05 / 1.95,
  engram_add 0.65 / 0.65, 1571 launches, wall 217.7 / 237.1. The off side of this change is the orchestrator's A/B.
- **Tests** (8x4, whole-box mesh, FABRIC_2D; no fork regression suites, OV-25):
  `scripts/run_safe_pytest.sh --run-all --no-precompile ttnn/ttnn/bringup/pair_pool_ttnn/tests/unit/test_pair_pool.py`
  (6 passed) and `.../mhc_pre_ttnn/tests/unit/test_engram_pack.py` (11 passed: the new test_engram_gate x3 +
  refusals, and the file's option-off cases).
- No stock op added on the model path (F113): the model calls only the two forks.
- Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.24`.

## P.25 perf (galaxy 8x4, attempt 1, 2026-10-31): MM_RETUNE_CAP, every dense matmul re-swept within the 110-core cap (MATMUL_CONFIG=galaxy110)
- **Switch**: MATMUL_CONFIG gains `galaxy110` (new default; tt/settings.py). galaxy110 = tt/mm_configs.GALAXY110 (12
  entries, keyed by table_key at grid 11 x 10), the rule for any call not in it. `galaxy` (GALAXY, P.19 + FIX.15) is
  unchanged. mm_configs: `table()` gives the mode's table; the program_config cache key carries the mode name.
- **Capture** (8x4 mesh, `DS41_MATMUL_CONFIG=galaxy DS41_ENGRAM_TABLE=resident P23_TAG=p25 ... -k capture`,
  7m38s): generated/ds41_p23/calls_p25.json, 187 calls, 12 distinct, 1357 KiB free L1 for CBs at every call (budget
  1342 KiB), chunk out vs golden rel 0.75809. Against FIX.15's capture: 11 calls the same. One changed: hc.fn is now
  [160, 20480] x [20480, 32] fp32 (P.23 OWN_ROWS: whole rows, stream-major; was [640, 5120]). The changed dtypes and
  layouts of P.21-P.24 sit in the norms and RoPE chains feeding these matmuls. The matmul inputs' shapes and dtypes are
  unchanged.
- **Sweep** (8x4 MESH, agent rule 19; one process; `P23_CALLS=calls_p25.json P23_CURRENT=galaxy P23_MIN_MS=0
  P23_FULL=1 ... -k "test_sweep and not card"`). Each candidate is held to tt/cores.worker_grid (11 x 10). New in
  test_matmul_sweep.py:
  - P23_CURRENT=galaxy: "current" (the before) = galaxy's config.
  - A pick must name its grid with at most 110 cores, so auto and minimal's default grid are never picked.
  - Per-config `cores` reported.
  - P23_TAG / P23_NAME name the files.
  - Wider candidates: minimal M / N blocks that divide the call's tiles (M5 for 160 rows, N10), and 1D in0 at
    per_core_N up to twice the narrowest (other core counts).
  - Run 1 (P.19 candidate set, _p25): 2m45s, no winner. Run 2 (wider set, _p25w): 12m56s, one winner.
  - Acceptance as P.19: deterministic, within one bf16 rounding of auto, error vs float64 <= galaxy's x 1.01, > 1 %
    faster. Every entry is bit-identical to auto (max |d| 0).
- **Per call** (ms a call, device kernel time on the mesh; cores a chip; calls a chunk):
  | site | shape (per chip) | calls | before (galaxy) | after (galaxy110) |
  |---|---|---|---|---|
  | attn.wo_a | 8x160x4096 x 8x4096x1024 bfp8 -> bfp8 | 40 | 0.2062, 55 (2D nofuse 11x5 bw32) | same |
  | attn.wq_b | 160x1280 x 1280x32768 bfp8 -> bf16 | 40 | 0.1978, 103 (1D in0 pn10) | same |
  | attn.wo_b | 160x8192 x 8192x5120 bfp8 -> bf16 | 40 | 0.1361, 80 (1D in0 pn2) | same |
  | q_a | 160x5120 x 5120x1792 bfp8 -> bf16 | 40 | 0.0473, 56 (1D in0 pn1 bw16) | same |
  | indexer.wq_b | 160x1280 x 1280x4096 bfp8 -> bf16 | 8 | 0.0285, 64 | same |
  | indexer.weights_proj | 160x5120 x 5120x32 bf16 | 8 | 0.0169, 5 | same |
  | engram_kv | 640x6144 x 6144x6400 bfp8 -> bf16 | 2 | 0.2295, 110 (minimal M4 K16 N8 sb1x4) | **0.2246, 110 (minimal M4 K8 N10 sb2x1)** |
  | compressor | 160x5120 x 5120x1024 bf16 -> fp32 | 3 | 0.0430, 55 | same |
  | compressor | 160x5120 x 5120x512 bf16 -> fp32 | 1 | 0.0297, 40 | same |
  | compressor | 320x512 x 512x128 bf16 | 3 | 0.0042, 40 | same |
  | compressor | 640x512 x 512x128 bf16 | 1 | 0.0067, 40 | same |
  | hc.fn | 160x20480 bf16 x 20480x32 fp32 -> fp32 | 1 | 0.0888, 5 (rule: 2D 1x5) | same (now a table entry) |
  Swept total 24.584 -> 24.574 ms a chunk (-0.010). No call runs over 110 cores. The 160-row projections are
  weight-DRAM bound. minimal with M_block 5 is slower than 1D in0 (wq_b 0.215 vs 0.198, q_a 0.067 vs 0.047). More
  cores at other per_core_N do not help (wq_b pn16 on 64 cores 0.216). DS41-191.
- **Measured** (`measure --task P.25`, change on: MATMUL_CONFIG=galaxy110 ENGRAM_TABLE=resident, code 96d85e228ed6efec;
  rc 0 on every part). quick: 102 limits pass (worst err_state_window_kv_L28 0.911, err_state_topk_idx_L08 0.901 of
  the limit), L39 0.7581 (= P.24, bit-identical configs), host_transfers_per_layer 0, wall 193.5 ms (spread 11.7).
  p51200: wall 202.5, device 169.55 ms (engram_kv 0.96). p512000: wall 227.5, device 208.14 (engram_kv 0.93).
  P.24's change-on side (another run): device 169.67 / 208.34. The galaxy-vs-galaxy110 A/B is the orchestrator's;
  expected difference ~0.01 ms a chunk, below run-to-run noise.
- **Not done**: no memory-config change (DRAM-sharded weights would be the next matmul lever; out of a program-config
  pick). No model profile (brief). No fork suites (OV-25).
- OV-3: nothing padded. OV-23: no Python-built op added or touched. OV-25: no fork suite run.
- Re-run: see the test_matmul_sweep.py docstring (P.25 lines). Logs: /scratch/dnijemcevic/p25/{capture,sweep,sweep_w}.log;
  generated/ds41_p23/sweep_p25w.json, galaxy_table_p25w.py. A/B: `python -m models.demos.common.bringup.orchestrator
  measure --task P.25`.

## FIX.17 overseer fix (galaxy 8x4, attempt 1, 2026-10-31): DEEP_LAYERS=all|tail (default all), layers 21-39 only on the prompt's tail
- **Switch**: DEEP_LAYERS (tt/settings.py), default `all` = today's model, unchanged (the only code on the all path is
  two set lookups in `TtDS41Model.layer`). `tail`:
  - `TtDS41Model.__init__`: deep layers = the loaded layers above the last KV / candidate source (max(kv_sources,
    cand_source) + 1 = 21, so 21-39); they must be consecutive with layer 20 loaded; tail_rows = (n_deep + 1) x
    (window - 1) + 1 = 2541 (the owner's 20 x 127 + 1).
  - Normal chunk: `layer(20)` runs as before, its output DevH (streams, pre and the mHC stats row for layer 21) goes to
    `DS41DeviceState.keep_deep_input` (`deep_in` {start: (DevH, end)}, this chunk's and the previous one's; no copy) and
    returns the `Skipped` marker; `layer(21..39)` return it unchanged (free of it is a no-op). Nothing of 21-39 runs.
  - `tail_pass(state, end=None, on_layer=None)` (the caller signals the prompt's end; no length needed before): the
    deep layers over the kept chunks holding [end - 2541, end): the last chunk, plus the previous one when the last holds
    fewer than 2541 rows; each at its own chunk start through the normal block (every layer's window carry as its cache
    holds it: zeros, or the previous tail chunk's kept tail; the top-k / candidate pools of layer 20 at that start;
    ChunkMemo cleared before each). Returns layer 39's output of the last chunk; then `final_norm` / logits as usual.
  - Serving keep_history=False: `put` keeps the previous chunk's top-k / pool too (the two-chunk tail reads them).
  - tt/runners/adapter.py refuses DEEP_LAYERS=tail (the engine reads every layer's window KV).
- **Not the owner's 3072-row chunk**: the tail runs as whole 5120-row chunks at their own starts. The model is built
  for one chunk size (WindowTables, the load-time host_alloc / fork-output buffers, program configs, the top-k and
  candidate stores keyed by chunk start, the window carry keyed by chunk end); a 3072-row pass would need a second
  geometry of all of them and layer 20's pools re-split over 96 rows a chip. Cost: 1 chunk (75.7 ms at 51200) or 2 when
  the last chunk holds < 2541 rows, about 0.1 s a prompt.
- **Accuracy** (tests/perf/test_deep_tail.py, rung `last`, golden prefix of layers 0-20 only, layers 21-39 caches zero,
  the ladder's drift limits): pass. Layers 0-20 over the chunk: rel 0.025-0.188, each within its limit (e.g. L20 0.165 /
  0.252). Layers 21-39 over the tail rows [end - 2541, end): 0.138-0.165, L39 0.613 / 1.533 (ladder L39 0.758: DS41-188
  rounding swing). Final hidden (tail rows) 0.191 / 0.242; top-1 1.000, top-5 1.000 (limit 0.994) over 188 sampled tail
  rows, last token top-1 match; layers 0-20 states within limits (worst rel 0.202, worst index overlap 0.836).
  Exactness: with a zero carry at chunk start s, layer 20 + k is exact for rows >= s + 127 k; the tail starts 2579 rows
  into the chunk here, past 19 x 127 = 2413.
- **Measured** (8x4, one job at a time, `DS41_ENGRAM_TABLE=resident`, test_profile.py with BRINGUP_POS, the gate's two
  commands: clean wall median, then device from the rt profiler; results in /scratch/dnijemcevic/fix17):
  | chunk at | all: wall / device ms | tail: wall / device ms |
  |---|---|---|
  | 51200 | 200.7 / 171.07 | 122.6 / 92.77 |
  | 512000 | 228.3 / 207.56 | 128.0 / 105.02 |
  Tail pass (layers 21-39, one 5120-row chunk, synced wall / rt device): 75.7 / 77.18 ms at 51200, 100.3 / 103.90 ms at
  512000 (test_deep_tail.py, DEEP_TAIL_POS=512000 for the second). Programs warm in every run (cache 140 entries tail,
  162 all; no misses in the timed runs).
- **Prompt tok/s** (chunk walls linear in position between the two measured starts, extrapolated past 512000, plus the
  tail pass at the end; as /scratch/dnijemcevic/opportunities/prefill_tps.py's arithmetic):
  | prompt | all: wall s, tok/s | tail: wall s (tail pass), tok/s | x | device-only tok/s all -> tail |
  |---|---|---|---|---|
  | 128k (131072, 26 chunks, 1 tail chunk) | 5.24, 25023 | 3.27 (80 ms), 40067 | 1.60 | 29295 -> 52376 |
  | 512k (524288, 103 chunks, 2 tail chunks) | 21.97, 23867 | 13.08 (201 ms), 40075 | 1.68 | 27120 -> 50711 |
  | 1M (1044480, 204 chunks, 1 tail chunk) | 46.67, 22382 | 26.26 (128 ms), 39776 | 1.78 | 24595 -> 48355 |
- Gotchas: keeping layer 20's DevH across chunks (no copy) did not add programs (F196 warm in 10 runs; DS41-146
  addresses: no miss). to_host of the Skipped marker fails: a harness that reads layer 20's output in tail mode reads
  `state.deep_in[start][0]`.
- OV-3: nothing padded (whole chunks as the model runs them). OV-23: no Python-built op added or touched. OV-25: no fork
  suite run.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/perf/test_deep_tail.py` (DEEP_TAIL_POS=512000: timing only);
  `DS41_DEEP_LAYERS=tail BRINGUP_POS=51200 BRINGUP_PROFILE_WALL=1 DS41_ENGRAM_TABLE=resident scripts/run_safe_pytest.sh
  --run-all --no-precompile models/demos/common/bringup/tests/test_profile.py`.

## FIX.18 overseer fix (galaxy 8x4, attempt 1, 2026-11-01): OWN_ROWS_KV_MARGIN, answered `disagree`, no code change
- **Request**: find what in P.23 (OWN_ROWS) moves the window KV of layers 20 / 35 / 38 (acks and runner-contract checks),
  and fix it in the OWN_ROWS path.
- **Found (device probes, on vs off, DS41_OWN_ROWS env; /scratch/dnijemcevic/fix18/)**: no defect. The values
  reproduce the gate exactly (L20 0.98947, L35 0.98910, L38 0.98994; off 0.99195 / 0.99857 / 0.99651).
  - Golden input per step (L0-3, 14, 20): every step equally far from the golden; attention / attn_norm / window KV
    bit-identical on vs off.
  - The first difference is attn_hc's fp32 mixes (summation order, <= 2.3e-4 rel), which flips 0.13 % of the bf16 stream elements.
  - MoE block on the same x (owners a2a into the head, the local tail vs the column split): bit-identical.
  - Over all positions L20 wkv on 0.1836 / off 0.1842, the same per TP column. Over 286 sliding windows, on - off has mean
    +0.0005 and std 0.004; the failing window is the 16th percentile.
  - Follow-up rows [2496, 2536) as accurate vs the golden as the first ones in both modes; the "last block" check
    compares two device runs.
  - DS41-192 (detail in known_issues_detail.md), the same mechanism as DS41-141 / DS41-125.
- **Decision**: per the brief ("a genuine one-rounding difference with no lower-precision choice behind it: stop and
  report"), nothing changed; the evidence goes to the owner for a waiver (or a per-position drift for the contract limit).
  Speed unchanged (no code change, no device / wall numbers taken).
- Gate not re-run: the code is K.1R's (3a1a390acd8 = FIX.17 + ledger commits); K.1R's failing values are reproduced to the digit
  by the probes. adapter_acks / runner_contract will fail as in K.1R; contract / kv_layout / smoke passed at K.1R.
- Re-run: `DS41_OWN_ROWS=on|off FIX18_OUT=<dir> scripts/tt-probe.sh --mesh fix18_dump < /scratch/dnijemcevic/fix18/probe_fu.py`
  (and probe.py with FIX18_LAST=20, probe_moe.py); analysis `python3 /scratch/dnijemcevic/fix18/ana.py gin|chain 0,1,2,3,14,20`,
  `ana_fu.py`.

## FIX.19 overseer fix (galaxy 8x4, attempt 1, 2026-11-01): CONTRACT_WINDOW_DRIFT, window-KV check like with like
- **Request (owner, 2026-11-01)**: judge the dumped layers' window KV in test_adapter_acks / test_runner_contract over
  many samples, against calib x the reference's drift over the SAME positions. Until now it was one 128-row window
  against the final ring's drift (DS41-125 decision B).
- **Changed** (tests/bringup/contract only, no model code):
  - `official_kv.py` (new) runs the official reference (numerics official, bf16 = reference.shipped_dtype, layers 0-24
    resident) over both chunks of the s10240 golden's tokens. It keeps "L<i>.kv" for the dumped layers
    0, 1, 2, 3, 14, 20, 21, 24 in `<golden dir>-official/window_kv.safetensors` (+ .json carrying the golden's
    content_hash; a file from another golden is refused). It took 530.5 + 557.8 s = **1096 s** (CPU, no device).
    The golden dumps had no per-position official KV: drift-s10240.json keeps only the ring.
    The new file's last 128 rows reproduce that ring entry exactly (L20 pcc 0.99739, rel 0.0722).
  - `caches.GoldenKV(window="drift")`, mode "drift": a dumped layer's window KV over EVERY position of the checked range.
    Each kv_dump_compare channel (nope / pe) is checked against `GoldenKV.limits`: drift._float_entry(official,
    golden) on the same rows and channel -> e2e_limits.pcc_limit (1 - calib^2 x max(1 - pcc, 1 - one bf16
    rounding)), floor R.PREFILL_GOLDEN_PCC. No fixed number (F138). `drift_summary()` prints each layer's worst range.
  - test_adapter_acks / test_runner_contract use window="drift" (docstrings updated). Unchanged: compressed caches,
    non-dumped layers (ring snapshots vs R.state_threshold), the 0.99 last-block rule (tt-d-gen's own; L35 / L38 stay
    waived), the byte checks.
- **Numbers** (DS41-193, detail in known_issues_detail.md). Official vs golden L20 nope: [0, 2536) 0.98311, last 128 of
  it 0.99191, ring 0.99660. Old limit 0.98957 = the ring's margin. FIX.18's device dumps (OWN_ROWS on) against the new limits:
  - [0, 2536): nope 0.97809 vs 0.93244, pe 0.99310 vs 0.97721
  - last 128 rows: nope 0.98947 vs 0.96765
  - 286 windows: device median 0.98761, official median 0.98926, correlation 0.95
  The device is as far from the golden as the reference at its own precision.
- **Waiver**: the layer-20 lines of tests.unit_waivers (both tests) can go; the L35 / L38 last-block lines stay. For the
  overseer (spec is not mine to edit).
- **Gate: not completed.** I started the gate in the background with a 600 s job limit. Before the limit could cut it
  mid-device-run, I stopped it (SIGTERM) 10 s into test_contract's UMD device open. run_safe_pytest's pessimistic
  marker left /tmp/tt-device-{0..31}.dirty, so the next mesh run will `tt-smi -r` the whole box. Rule 19 forbids
  triggering a reset, so I ran nothing more on the device and did not touch the markers. The CPU check of the new path
  on the FIX.18 device rows passes, and a noise input fails it. All 5 units are left to the orchestrator's gate.
- OV-3: nothing padded. OV-23: no Python-built op added or touched. OV-25: no fork suite run.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.deepseek_v41_flash_d_p.tests.bringup.contract.official_kv` (only for
  a new golden); `PYTHONPATH=$PWD python /scratch/dnijemcevic/fix19/predict.py`; gate `gate FIX.19 --check`.

## P.26 perf (8x4, 2026-11-01): WINDOW_KV_FORMAT=bf16_tiles; contract err_*_wkv_* diagnostics
- Detail: known_issues_detail.md "P.26 WINDOW_KV_BF16" (file at the 500 KB limit, DS41-194).
## FIX.20-22/P.27-33: known_issues_detail.md
## P.34 perf (8x4, 2026-11-06): POST_COLLAPSE (mhc_post collapse_pre) - known_issues_detail.md "P.34 POST_COLLAPSE"
## Weight cache: one layout, shared master (owner 2026-11-06; fork CHANGELOG "One weight cache layout", framework F261)
- Routed-expert cache = the fork's bank-major `.bm.bin` only (hit = the layer's .bm.bin; a miss writes only those);
  `.tensorbin` is never read (old caches: tools/convert_expert_cache.py). FLAT_CACHE_BANK_MAJOR is gone.
- TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES: tt/settings.py export_pinned_limit sets 16 GiB (32 with bfp8 weights)
  when unset, from bringup/hooks.py device_params (tests, before the mesh opens) and tt/runners/adapter.py (runner, at
  import). An explicit value is kept (test_runner_contract.py passes 0 by default: the KMD 2.9 workaround): the
  stager then logs one FLAT_CACHE line and takes the slower copy.
- galaxy spec paths.tt_cache_shared = /data/dnijemcevic/bringup/deepseek_v41_flash_d_p/tt_cache: the master; the
  orchestrator pulls it into art/tt_cache at run start and pushes new layers in the background
  (`python -m models.demos.common.bringup.core.weight_cache status|pull|push --spec <galaxy spec>`).

## P.37 perf (8x4, 2026-11-06): the carried block's features on MOE_BLOCK_SRC=main
- **What each feature needed in the main block** (C++ fork ttnn/ttnn/bringup/moe_block_main_ttnn, Python
  models/demos/moe_block_main, adapter tt/moe_block_main_adapter.py; fork CHANGELOG "MoEDtypes.shared_activations bf8
  and shared_pack_precise"):
  - GEMM_INPUTS=bfp8 in the head (FIX.9, DS41-132; prerequisite of SE_PACK_PRECISE): main's C++ head never read
    MoEDtypes.shared_activations, so main ran the shared expert's x / h in bf16 (DS41-211). Ported se_engine3's
    x_dtype / h_dtype: HeadShape / SeEngineArgs x_dtype, 1088 B tiles in the engine's regions and CBs, SE_XTB /
    SE_HTB / SE_PREP_TB / XTB defines in se_brisc, se_compute, the x gather; a new head input xq (bf8 x; the engine's
    own-shard load and the gather read it, the router keeps bf16 x from DRAM: SE_RTRX_L1 off with bf8 x own).
  - SE_PACK_PRECISE (DS41-137): MoEDtypes.shared_pack_precise -> HeadProgramConfig.pack_precise -> the SE compute's
    bfp8_pack_precise + define SE_PACK_PRECISE; in the program hash and the plan / setup keys.
  - NORM_INTO_MOE (P.21): input_buffers() returns a persistent bf16 x and the block's persistent bf8 xq (pinned);
    ffn_norm writes both (rms_norm bfp8_twin), __call__(xq_ready=True) hands xq to MoEBlock(xq=). Off: MoEBlock.to_xq
    (one typecast on the capped grid). As on carried, model.py engages it only with OWN_ROWS=off (the default OWN_ROWS=on
    feeds the head from a TP all-to-all: one typecast a layer, both blocks).
  - DECODER_SKIP=replay128 (P.31): at_chunk builds the (mesh, 1024)'s own MoEBlock; the routed weights are shared when
    the tail block's flat plan has the same weight-layout facts (_WEIGHT_KEYS; 8x4: shared for all 20 tail layers),
    else laid out for it; the head weights per block at first call. model.build_tail reserves the 48 KB L1 only for
    carried (main keeps its words in L1_SMALL and allocates its arena a call; the reservation was never given back).
  - EXPERTS_PRECISION: main now asserts glm like carried.
- **Audit** (every setting the carried block / adapter / its _RoutedWeights reads; value: works / ported / out of scope):
  | setting | values | main |
  |---|---|---|
  | MOE_IMPL | block / flat | works (block selects the adapter) |
  | MOE_BLOCK_SRC | carried / main | works |
  | MOE_HEAD | fused | works; ops: NOT ported (main has no ops head: the fused head alone writes the dispatch's route / plan tensors; asserts) |
  | MOE_TAIL | ops / fused | works: main has one tail (C++ fused_tail; OWN_ROWS: the local tail), both values run it |
  | GEMM_INPUTS | bfp8 / bf16 | ported (bf8 x / h in the head; bf16 = the source's program) |
  | SE_PACK_PRECISE | on / off | ported |
  | NORM_INTO_MOE | on / off | ported (bit-identical, test_moe_block_main_parity.py) |
  | DECODER_SKIP | off / exact / replay128 | off, exact work; replay128 ported (at_chunk) |
  | OWN_ROWS | on / off | works (pre-existing own-rows path; off: MoEBlock's forward) |
  | GLUE_FOLD | on / off | works |
  | MAX_WORKER_CORES | Galaxy 110 | works (set_max_grid; the new typecast takes the same cap) |
  | EXPERTS_IMPL | bfp4 / bfp4_mse / bfp8 | works; bfp8_down4: out of scope (owner) |
  | EXPERTS_PRECISION | glm | ported (assert, as carried); full: refused on both |
  | EXPERTS_WEIGHT_CACHE | on / off | works (expert_cache_prefix) |
  | EXPERTS_LAYOUT | marko / legacy | works: placement of carried's flat op only; main's op plans its own layout (same math) |
  | EXPERTS_PREFETCH | on / off | works: carried's bank-major read-ahead; main reads its own cache files (no result change) |
  | EXPERTS_SUM, COMBINE_ACC, COMBINE_PLANNER, MATMUL_FIDELITY | all | n/a: read by TtExperts for its weighted combine, which the carried block never calls |
  | TAIL_A2A_PREPARED, PREPARED_ALL, HOST_ALLOC (+ FORK_OUT_BUFFERS / ROPE_CHAINS / TINY_KERNELS via host_alloc) | all | out of scope (owner): the C++ ops patch addresses on cache hits |
- **Tests added**: fork `ttnn/ttnn/bringup/moe_block_main_ttnn/tests/test_fused_head_bf8.py` (bf8 accuracy +
  determinism; the packed h bit for bit, both options: crafted x / selection weights so h = silu(10) u exactly and the
  output = h; 1,777,311 decided values, 232,605 telling one rounding from two: 0 mismatches each way). DS41
  `tests/test_moe_block_main_parity.py`: SE_PACK_PRECISE on / off through the adapter's block_dtypes / block_dims (same
  bit check), NORM_INTO_MOE on vs off on layer 0 of s10240 chunk 0 (OWN_ROWS=off): max |diff| 0.0, xq_ready taken.
- Re-run: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/moe_block_main_ttnn/tests/test_fused_head_bf8.py`;
  `scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_moe_block_main_parity.py`.
- Gotchas: the fork builds into build_Release; python loads the installed copies (cmake --build build --target
  install). make_head_model builds its own SeEngineArgs beside head_plan's: a new engine arg goes in both (DS41-211).
  The fork test needs mesh_device parametrized (8, 4): the default fixture opens 1x32 and the torus tie check refuses.
- Rule 18: no new op on more than the cap (the head's programs are the fork's, inside set_max_grid; the xq typecast on
  set_max_grid's 11 x 10). Pre-existing on main's OWN_ROWS path: ttnn.to_layout / ttnn.add without sub_core_grids
  (not changed here).
- Note for the A/B: GEMM_INPUTS=bfp8 now reaches main's head on both sides (on and off differ in SE_PACK_PRECISE /
  NORM_INTO_MOE only); main's "before" numbers (2.91 / 1.24 ms) were with bf16 x / h.
- **Measured** (`orchestrator measure --task P.37`, change on: MOE_BLOCK_SRC=main SE_PACK_PRECISE=on NORM_INTO_MOE=on,
  OWN_ROWS=on default; ab/P.37/new): quick: every layer and state within its limit, host_transfers_per_layer 0,
  chunk wall 185.0 ms. p51200: chunk wall 194.0 ms, device 177.27 ms, experts section 106.8 ms = 2.67 ms a layer
  (main before 2.91, carried 2.39); host: the MoE layer's _own_rows 49.19 ms cum / 40 = 1.23 ms a layer (main before
  1.24, carried 1.70). The device gain is the head's bf8 x / h (GEMM_INPUTS reaching it), not the switch: the off side
  (SE_PACK_PRECISE=off NORM_INTO_MOE=off) has bf8 x / h too; under OWN_ROWS=on NORM_INTO_MOE does not engage (as on
  carried). Short turn (gate unit, replay128 on main): final hidden rel 0.14151 (limit 0.24219), top-1 / top-5 hit.
- P.37 attempt 2 (F113 stock-call check):
- stock ttnn.all_gather instead of ttnn.bringup.combine_rf.<module>: main's fused head contract takes x_full (the mesh row's x, bf8 xq when GEMM_INPUTS=bfp8) gathered over the TP axis in DRAM only when its planner picks the x-in-DRAM fallback (blk.x_dram); this is the source block's own call (models/demos/moe_block_main/tt/moe_block.py MoEBlock forward, unchanged since 494262dfccb apart from gathering xq), and combine_rf's fabric all-gather is a Python ProgramDescriptor builder sized for the carried block's token-split path, not a drop-in for a [1, 1, rows, H] DRAM gather. No code change in attempt 2; the measure results of attempt 1 stand.
