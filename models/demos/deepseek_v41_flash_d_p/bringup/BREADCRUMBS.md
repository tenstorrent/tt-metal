## PPB.7 (PB.7 perf, attempt 2): DS41-339 fixed at the cause (moe_head shared expert reduced by columns)

- Plan (OV-65, before the first device run): the moe_head fork already has a columns scatter (scatter -1, Kimi-K3's
  latent path): every output column's adds run in its column block's ring order, never in the order of the TP chip
  that owns the row (DS41-128's mechanism, the tp_sum fix by columns). New fork option
  `MoEBlockConfig.shared_scatter_columns` (default off = the source's rows scatter): HeadShape.scatter_columns ->
  reduce_scatter_dim -1 without a latent, and the ring's halves split by matmul core COLUMN (the -1 path split them by
  core index = by core row, i.e. by row again). The model (Setting SE_SCATTER=cols) then turns y [S/SP, H/TP] into
  owner rows with one TP all-to-all (all_to_all_async_generic in_dim 3 -> out_dim 2, the OWN_ROWS path's own
  converse) before the y_own + se add. Butterfly / int32-snap zones rejected: a new transport schedule or 2x the RS
  bytes inside a fused kernel; the columns path is built and tested upstream.
- Expected cost per 8192 chunk: + one TP all-to-all of [1024, 1280] bf16 a chip a layer (~0.06-0.1 ms device: the
  engram_kv a2a of [640, 6400] is 0.10-0.16 ms) -> +1.3-2 ms device over 21 layers (walls), +2.5-4 ms over 40; host
  +~0.03 ms a layer (one op launch). The RS bytes are unchanged (3 of 4 blocks either way).
- Op test first (no model load: mesh-only): tests/test_moe_head_scatter_columns.py (fork check
  ttnn/ttnn/bringup/moe_block_ttnn/tests/test_shared_scatter_columns.py): rows rolled 32 within every SP row.
- Op test (resident, mesh only, 0 model loads): PASS. rows scatter (off) chunk 2048, rows rolled 32: 1024 of 2048
  rows differ, max 0.015625 (1 bf16 ulp) = the reproduction; columns (on) chunk 2048 and 8192: 0 rows differ,
  determinism 0, pcc vs torch 0.99956 (off 0.99957), one row block (8192: GY 8 MT 2 NCH 2 GX 11).
- Load (one line, PACKED_RULES): the device oracle needs the gate's 40-layer accuracy load (OV-61: one load per gate);
  iteration on it with DS41_PACKED_LAYERS=0,1,2,3 first (call-time), the full oracle after on the same held model.
- Gotcha 1 (DS41-340): the op test planned without the model's core cap in the runner's process, which froze the
  process's moe_block cap at 0x0; the next model test died (TT_FATAL core cap). Fixed in the test (set_max_grid with
  tt/cores.max_grid first); cost one --fresh process.
- Gotcha 2 (a real bug, found by the 4-layer oracle: alone vs today 7.5, every segment): the port kernels skip a
  sub-block whose rows are all padding (x_gather_block_real) reading `block` as a ROW block; by columns a block is a
  column block, so a packed chunk's (actual_end) real rows lost partials. Fix: named arg reduce_scatter_columns (=
  the option): a core's sub-block of any block is live when any of its row tiles is real. The op test now also runs a
  padded chunk (real rows vs the unpadded call: 0).
- Oracle rung (layers 0-3, the gate's 40-layer load, DS41_PACKED_LAYERS=0,1,2,3): alone vs today 0, segment
  equivalence 0, split vs alone 0, slot digests 0, determinism 0 (attempt 1: 0.25 / 0.094 / 60 from layer 0).
- Fork regression with the option off: not run (PACKED_RULES: no fork suites on the 8x4); off = the source's programs
  (reduce_scatter_dim -2, half() and the new named arg 0 unchanged); the op test's off case runs it.
- Ops / cores (rule 18): the head's matmul cores unchanged (11 x 8 = 88 at 8192); + one stock
  all_to_all_async_generic a layer (CCL, not a matmul).
- Gate check #1 (after the columns fix; 1193 s): glued oracle 40 layers FAIL segment equiv 7.0, split 704, digests
  102 (alone vs today 0, determinism 0); only glued1 H (from layer 32) and glued2 P (from layer 20) left. Walls ok:
  4x2k_p139264 73,140 new tok/s (>= 67k), 5x1600 16.1k, agentx glued 20.6k / 25.4k / 16.1k vs lanes 32.6k / 29.9k /
  36.9k, DRAM headroom 6x_p1m 6.886 GB; determinism 0.
- Probe tests/packed/test_glued_probe_layer.py (diagnostic, not gated; DS41_PACKED_LAYERS=0..20, GLUED_PROBE_LAYER=20,
  glued2 vs P alone): layers 0-19 0; layer 20: attention input / q_a / kv / attention / output bit-identical; the layer
  output differs in ONE row (1358 of 1600, 0.031, 1071 columns) -> the MoE. Cause DS41-341: the fused head's router
  adds the owner chip's fp32 logits partial first, then the arrivals in the owner's order (router_compute.cpp,
  head_model.cpp arrival_order): last-bit logits differences by owner chip, a rare gate-weight flip. Fix: fork option
  MoEBlockConfig.router_canonical_sum (chip order once all are in). Probe again: layer 20 0 rows.
- DECIDED (OV-58): one model Setting for both fork options, renamed MOE_HEAD_ORDER=free|source (default free;
  load-time). The op test now also checks the routing rows (ids + gate-weight bits) of moved rows (on: 0).
- Gate check #2 (`gate PB.7 --check`, after the last code edit, hooks clean except the orchestrator's dashboard HTML
  size check): PASS PB.7. Glued oracle 40 layers: segment equiv 0, alone vs today 0, split vs alone 0, slot digests
  0, determinism 0; packed oracle lane equiv 0, alone vs today 0; glued reference every gseg within its limits
  (top5 1.0); walls 4x2k_p139264 71,830 new tok/s (gate #1, columns only: 73,140; attempt 1, source order: 72,975),
  5x1600 16.1k, agentx glued 20.6k / 25.4k / 16.1k vs lanes 32.4k / 29.8k / 36.7k, DRAM headroom 6x_p1m 6.886 GB;
  determinism 0. `orchestrator measure --task PB.7` (quick, switch on): rc 0, no limit exceeded.
- Cost of MOE_HEAD_ORDER=free on the 4 x 2k lane chunk: ~-1.2 ms a chunk vs attempt 1 (112.4 -> ~114 ms; the
  se all_to_all a layer + the router's adds after the last arrival), within the run-to-run spread seen before.
- Re-run: op test `resident submit --cmd "scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_moe_head_scatter_columns.py"` (mesh only); probe: the gate's glued env
  + DS41_PACKED_LAYERS=0,..,20 GLUED_PROBE_LAYER=20 tests/packed/test_glued_probe_layer.py; gate `gate PB.7 --check`.

## PPB.7B (PB.7B perf, attempt 1): glued packing v2, per-row segment arguments (PACKED_GLUED_ONECALL off | on)

- Expected (written before any device run, perf_guide s2): path (b) costs ~77-95 ms device a moved segment a chunk
  (PB.7: 5 x 1600 at 139k 496 ms vs the 4 x 2k lane chunk 112 ms). v2 removes every move: the attention runs once a
  layer with per-row arguments; what is left over the lane chunk is the passes where a chip holds two segments (the
  indexer ring + top-k once more per extra pass, ~2-2.5 ms an index layer at 139k, 3 index layers in 0-20; the
  sparse_sdpa extra pass on the window keys of the other rows, ~0.1-0.2 ms a sparse layer; a few ChipRows copies) and
  the per-descriptor table uploads (~5 a chunk). Device 5 x 1600 @ 139k: ~496 -> ~110-125 ms; host: ~-70 ms of path
  (b)'s per-segment launches, + ~5-10 ms of pass launches. So glued 5 x 1600 ~120-130 ms (8000 tokens, ~62-66k new
  tok/s) vs lanes 2 chunks 200 ms (40k); AgentX mixes glued ~85-96 % real rows at ~120-135 ms a chunk vs lanes 41-51
  %: glued should win by ~1.5-2x.
- DECIDED (OV-58): v2 without a C++ fork change first: per-chip "passes" (pass p = each chip's p-th segment) over the
  existing per-chip lane options (sdpa kv_lanes, the indexer's lanes ring with lane_rows = R: each chip its own lane),
  a chip whose pass segment starts inside it scores its query / gate / bias rows rolled (its lane start is then that
  segment row's position, never negative: a fresh segment can start mid-chip). A per-token kv selection in the sdpa
  fork and more than 4 key buffers a ring call are the follow-ups if the passes measure costly.
- Load reasons: (1) tests/packed/test_glued_ops.py: "needs only the mesh: the two new prepared programs (ChipRows,
  GluedSend) bit-exact vs the host, smallest shapes"; (2) oracle rung DS41_PACKED_LAYERS=0,1,2,3 then the
  characteristic layers: "needs the gate's 40-layer accuracy load (OV-61) but computes only the listed layers".
- Load reason (3): the walls (tests/packed/test_glued_walls.py, GLUED_WALL_CASES=5x1600_p139264,agentx_m1 first): "needs
  the performance-class load (layers 0-20, DECODER_SKIP=decode, resident engram table) because the walls time the
  serving-equivalent chunk".
- Built: `tt/packed_glued.py` (GluedGeo: chip pieces / passes / glued rope pieces / per-row window index lists;
  ChipRows: one prepared per-chip row-piece copy (a per-chip table page uploaded once a chunk, page prefixes for the
  logits); GluedSend: the glued window region a layer in one prepared program; GluedCall(OneCall): compressor,
  indexer calls, window / sparse attention), `tt/rope.GluedStarts` + `glued_slices` (RopeTable / CompressRope /
  TtCompressor take them), Setting PACKED_GLUED_ONECALL (call-time, hooks batch_call_settings), PackedChunk wires
  GluedCall for a glued chunk that `glued_call_ok` takes (else path (b)). Tests: tests/packed/test_glued_ops.py
  (ChipRows / GluedSend bit-exact vs the host on the 8x4; region lists on all four boxes, host), diagnostics
  test_glued_v2_probe.py (one segment's top-k / sdpa / logits at a layer, glued vs alone), test_glued_v2_phases.py.
- Op test (mesh only): ChipRows rows / tiles (bf16, bfp4) / GluedSend (2 runs, a shared tail tensor): 0 chips differ.
  Gotcha: the mesh-only op test and the 40-layer oracle in ONE runner process: the oracle's first single-request
  chunk failed (mhc_post: "Input Tensor is not allocated", layer 1); the same oracle in a fresh process ran (DS41-316's
  family: run mesh-only tests in their own process, or after the model's tests).
- Oracle rung layers 0-3, first run: layers 0-1 (window) 0 everywhere; from layer 2 the segments starting inside a chip
  differed (0.057). Probe (test_glued_v2_probe.py): logits bit-identical, same top-k SETS, 180 of 768 rows in another
  ORDER = DS41-342: topk_large_indices splits a chip's last 256 mod cores rows over core pairs and ties follow the
  split, so a row's top-k depends on its chip row. Fix: each call's scores rolled per chip to the segment's alone
  chip row ((g - off) mod R) before the top-k (first max kv_len columns only), the top-k rows mapped back.
- Oracle after the fix: layers 0-3 and the characteristic layers [0, 1, 2, 3, 14, 20, 21, 24] (the candidate source
  20 and re-index 24 included): alone vs today 0, segment equivalence 0, split 0, slot digests 0, determinism 0.
- Walls, first v2 (GLUED_WALL_RUNS=3, gate walls env, one 21-layer load):
  | case | glued v2 | lanes | path (b) (PB.7) |
  | 5x1600_p139264 | 146.1 ms, 54.7k new tok/s (device 131.0) | 2 chunks 203.2 ms, 39.4k | 496 ms, 16.1k |
  | agentx_m1 | 3 chunks 432.8 ms, 54.4k (96 % real rows) | 7 chunks 731.2 ms, 32.2k (41 %) | 20.7k |
- Fork: ring_indexer_score_dsa's lanes ring takes up to 8 key buffers (was 4; only the validation limit; CHANGELOG
  PB.7B): a pass of 5-6 segments is one ring call. Oracle layers 0-3 after the rebuild: all 0.
- Phase split (tests/packed/test_glued_v2_phases.py: the walls with each profiled chunk's device phases), 5 x 1600 @
  139k, glued 125.8 ms device vs the 4-lane chunk 98.1: the extra sparse_sdpa pass + its row copy (attn_pass) 12.6 ms,
  the second indexer call (indexer_pass) 5.6 ms, the top-k phase copies (indexer_phase) 0.9 ms, attn_residual +5.9 ms
  (the collective after the attention waiting for the chips holding two segments); AgentX m1 chunk 0 132.1 vs 102.7.
- DECIDED (OV-58): the sdpa pass goes into the fork: sparse_sdpa kv_lane_rows (each query row its own lane's kv, the
  kv address per token from per-core runtime args set every call: no program per map). The indexer's second pass
  stays (a per-row-lane ring needs the score kernel to take two lanes a device: the next step if it measures worth).
- Fork: sparse_sdpa `kv_lane_rows` (CHANGELOG PB.7B): oracle layers 0-3 and the characteristic layers: all 0.
- Walls with kv_lane_rows (test_glued_v2_phases.py = the walls + phases, GLUED_WALL_RUNS=3, one 21-layer load):
  | case | glued v2 | lanes | glued / lanes |
  | 5x1600_p139264 | 132.2 ms, 60.5k new tok/s (device 115.3) | 2 chunks 206.6 ms, 38.7k | +56 % |
  | agentx_m1 | 3 chunks 419.2 ms, 56.2k | 7 chunks 730.0 ms, 32.2k | +74 % |
  | agentx_m2 | 4 chunks 540.8 ms, 52.2k | 8 chunks 941.9 ms, 30.0k | +74 % |
  5 x 1600 device phases: attention 37.9 (lane chunk 30.6 with 6400 real rows), indexer 8.3 + indexer_pass 5.6 +
  indexer_phase 0.9, attn_residual 6.9 (was 11.4 with the sdpa passes). AgentX m1 chunk 0 (histories 35k-444k):
  indexer_pass 15.8 ms (a second ring over long histories for the chips holding two segments), attn_region 8.3.
- What is left (measured, not built): the indexer's second pass (a ring over every chip for the few chips holding a
  second segment: 5.6 ms at 139k, 15.8 ms on a 35k-444k mix) needs the score kernel to take two lanes a device; the
  top-k phase copies (0.9-2.1 ms); host: new descriptors build new programs for the lane maps (sdpa kv_lane_of_chip,
  the ring's lane_buffer are hashed) and upload ~5 per-chunk tables (the walls warm them up; serving would pay them on
  each new descriptor).
- Op test after the last C++ change (mesh only, its own process): ChipRows rows / row prefixes / tiles (bf16, bfp4),
  GluedSend (2 runs): 0 chips differ; region lists on 8x4 / 4x4 / 2x4 / 2x2: 0 rows wrong.
- Re-run: op test `resident submit --cmd "BRINGUP_CHUNK=8192 scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/packed/test_glued_ops.py"` (its own process: DS41-316); oracle rungs: the
  gate's glued env + DS41_PACKED_LAYERS=0,1,2,3 (then 0,1,2,3,14,20,21,24) test_glued_oracle.py; probe: +
  GLUED_PROBE_LAYER / _COMP / _SEG test_glued_v2_probe.py; phases: the walls env + GLUED_WALL_CASES / GLUED_WALL_RUNS
  test_glued_v2_phases.py; gate `PYTHONPATH=$PWD python -m models.demos.common.bringup gate PB.7B --spec $BRINGUP_SPEC
  --check` (2 h tool timeout, background).
- Gate (`gate PB.7B --check`, after the last code edit, hooks clean): PASS PB.7B (1569 s). Glued oracle, 40 layers:
  segment equivalence 0, alone vs today 0, split vs alone 0, slot digests 0, determinism 0; packed (lane) oracle 40
  layers: lane equivalence 0, alone vs today 0; glued reference: every gseg within its limits (final rel 0.13, top-5
  1.0); determinism 0.0. Walls (table results/PB.7B_glued_walls.md, 5 runs a chunk):
  | case | glued v2 | lanes / path (b) (PB.7) |
  | 4x2k_p139264 | 113.4 ms, 72.2k (the lane chunk; bar 67k) | PB.7 71.8k |
  | 5x1600_p139264 | 134.4 ms, 59.5k | lanes 202.3 ms, 39.5k; path (b) 497 ms |
  | n2 / n3 / n5 / n6 | 105.8 / 122.4 / 138.2 / 151.5 ms | path (b) 241 / 326 / 497 / 583 ms |
  | 6x_p1m | 228.1 ms, DRAM headroom 6.69 GB | path (b) 774 ms |
  | agentx_m1 / m2 / m3 | 57.4k / 52.7k / 47.0k new tok/s | lanes 32.0k / 29.9k / 36.6k (+80 / +77 / +28 %) |
- Final check `orchestrator measure --task PB.7B` (quick, switch on, the single-request path): rc 0, chunk_wall_ms
  216.8, min layer PCC 0.732 (L39, as PB.7), host_transfers_per_layer 0.

## PPB.7C (PB.7C perf, attempt 2): dynamic glued layouts, one indexer ring call (PACKED_GLUED_DYNAMIC off | on)

- Attempt 1 was stopped by the overseer before any change (OV-66: PB.8's tree); the tree was clean of model code at
  the start of attempt 2 (only ledger files modified).
- Expected saving per chunk (written before any code change, perf_guide s2):
  - device: the second indexer ring pass goes (PB.7B measured 5.6 ms at 139k on 5 x 1600, 15.8 ms on the AgentX m1
    chunk with 35k-444k histories) plus the rolled q / w / bias copies of a rolled call (~0.2-0.5 ms); the ring call
    itself gets slower by the per-core key reads (k multicast off when groups of one column read different lanes:
    the K bytes x 8 rows, ~0.1 ms at 139k, ~0.5 ms at 547k) and by q_chunk 32 (same products, smaller matmul
    blocks; estimate +0-10 % of the first call's ~8 ms). Net: -4 to -5 ms at 139k, -12 to -15 ms on long mixes.
    The top-k phase copies stay where a pass's rows sit at another chip row than alone (DS41-342), ~0.9-2 ms.
  - host: with FRESH layouts every chunk (serving), PB.7B pays a program build per new lane map (sdpa
    kv_lane_of_chip / ring lane_buffer hashed; glued rope slices and concats keyed on segment lengths; per-descriptor
    uploads ~5-10 a chunk): estimated tens to hundreds of ms per new layout (one ring program ~50-100 ms to build).
    Target 0 builds after warm-up and one upload a chunk; the per-layout host table work stays (~1-3 ms).
  - So fresh-layout AgentX walls: PB.7B (repeated layouts) 57.4k / 52.7k / 47.0k new tok/s are the ceiling the
    fresh layouts should reach, and the one-call indexer adds ~+5-10 % on top.
- Load reason (1): tests/packed/test_glued_dynamic_probe.py (diagnostic, not gated): "needs the walls' performance
  load (layers 0-20, decode) because the program-cache misses of a new layout come from every layer's steps; it
  names the ops (wraps ttnn callables, counts program-cache entries)". On the code before any PB.7C change.
- Probe result (PB.7B code, 3 fresh mixes = 9 chunks after 2 warm-up mixes): 322 program-cache entries added (~36 a
  chunk). Sites: the glued rope slices + concats (rope.py / compressor.py glued_slices, 152), DeviceScalars.at slices
  of the served writes (served.py:215, 50), the multi-segment served writes' part slices / embedding / reshape / fp8
  (compressor.write_lanes, ~35), snapshot_send patterns (20), ring_indexer_score_dsa lane maps (12), history-gather
  tail slices / concat / gather / pack (position-dependent, ~30), sparse_sdpa (7), the window ring carry (3).
- Load reason (2): tests/packed/test_ring_ranges_ops.py: "needs only the held model's indexer objects (dtypes, ring
  semaphores, program settings) and the mesh: the fork's lane_ranges vs PB.7B's lanes calls on random data".
- Fork: ring_indexer_score_dsa `lane_ranges` (C++ / kernels; CHANGELOG PB.7C): per device up to 8 (row range, key
  buffer, start, kv_len) entries as runtime args (reader / compute / writer common args, re-patched every call; the
  buffer map too: every buffer's address bound in the reader's common args), each group of q rows looks up its range
  (kernels/indexer_score_ranges.hpp: its key accessors, kv_len, causal geometry; the epilogue's chunk start), key
  multicast off in this mode (a column's groups may read different buffers), each buffer gathered to the largest
  kv_len of the ranges reading it. Only the presence is hashed. q_chunk 32 (a range is whole q chunks).
- Op test tests/packed/test_ring_ranges_ops.py (held 21-layer model's indexer objects, random data, T 65536):
  PASS, 1128 (chip, range) pairs bit-identical to PB.7B's per-chip lanes calls on host-rolled rows (q_chunk 64),
  scores and the candidate layer's block maxes: two segments meeting inside a chip at every tile-row offset 32..224
  (the second fresh at 0 and continuing at 4096), short (0 / 32 / 64 / 2048) and long histories (to the capacity),
  six segments in one ring.
- Load reason (3): the glued oracle rung DS41_PACKED_LAYERS=0,1,2,3,14,20,21,24 with PACKED_GLUED_DYNAMIC=on: "needs
  the gate's 40-layer accuracy load (OV-61) but computes only the characteristic layers (the index layers, the
  candidate source 20, the re-index 24)".
- Model path (PACKED_GLUED_DYNAMIC=on, tt/packed_glued.GluedCall._index_dynamic): ONE ring call a layer (lane_ranges,
  q_chunk 32, GLUED_SEGS key buffers: the unused ones distinct scratch shards, DS41-343), then per pass its top-k at
  the alone chip rows (DS41-342's phase copies stay) with each chip's own valid length, keep copies to the segment
  rows; the candidate layer's short pools (nbv < ctopk) as the -inf padded pool at a fixed width (min() with a per-chip
  cap row; the block-max buffer zero-initialised so a never-written column is finite).
- Other layout-dependent programs made runtime (each behind the same switch; tt/page_copy.DYN_PARTS names them for a
  diagnostic bisect, tests/packed/test_glued_dyn_bisect.py): glued rope rows (tt/lane_rows.GluedRows: the pieces as
  prepared-program values), served-write scalars and the server-order row (tt/page_copy.pages_of instead of slices
  keyed on the position), the window ring carry (page copies), each segment's served writes on its own (the one-gather
  write's shapes follow the segment count; padding it would gather rows nobody needs, OV-3), sparse_sdpa kv_lane_rows
  without the lane map in its hash (sdpa fork, GLUED_SEGS lane tensors), the window snapshot broadcast with the block
  owners as values (tt/snapshot_send: every chip derives its tree role; host check of the role rule vs plan() on 8x4 /
  4x4 / 2x4 / 2x2 with and without wraps: 0 mismatches), the history pack / tail slices and concat as page copies
  (whole histories in PACK_BUCKET pieces).
- Debug log: oracle rung on the characteristic layers failed (inf) with every part on; bisect (drop one part at a
  time): only 'ranges'; the v2 probe showed segment P's logits from another key buffer: DS41-343 (repeats of segment
  0's k_local as padding). With distinct scratch shards: oracle rung (layers 0,1,2,3,14,20,21,24, every part on, hide
  on): alone vs today 0, segment equivalence 0, split 0, digests 0, determinism 0.
- Load reason (4): the probe again on the walls load: "the performance load; counts the programs a new layout still
  builds with every part on".
- Probe on the walls load, every part on (fresh process: the ENGRAM_TABLE switch): 3 fresh mixes = 9 chunks after 3
  warm-up mixes: 1 program-cache entry (was 322 on PB.7B's code): the combined history-tail gather keyed on the
  segments' tail rows. Fixed: one input / output size for every layout (history_gather.tails_max_rows: two periods a
  segment x GLUED_SEGS) and high_bw_all_gather's gathered_dim_size (runtime) moves only the plans' rows (no extra bytes).
- Load reason (5): the walls (GLUED_WALL_CASES=5x1600_p139264,agentx_m1 first): "the performance load: fresh-layout
  walls, builds after warm-up, ring calls a layer".
- Known issues used: DS41-342 (the top-k at the alone chip rows: the phase copies stay after the one ring call),
  DS41-328 (one key capacity a ring: asserted), DS41-316 (the op test shares the held model, no mesh-only request),
  DS41-286 (the top-k's tie order follows its core split: rows moved, not the core count), DS41-277 (PageCopy keyed on
  the accessor compile-time args, an ND-sharded slab's geometry included). New: DS41-343.
- Walls, first fresh-layout run (GLUED_WALL_CASES=5x1600_p139264,agentx_m1, GLUED_WALL_RUNS=3; served writes then
  per segment): 5x1600 glued 142.4 ms = 56.2k new tok/s (device 116.9; PB.7B 134.4 ms / 59.5k), lanes 213.6 ms;
  agentx_m1 glued, 3 fresh mixes (seeds 1000-1002, 10 chunks): 43.4k new tok/s, program builds after warm-up 0, ring
  calls a layer 1, the same last mix's layouts repeated 46.4k (host 8.0 ms a new layout a chunk); lanes fresh 28.5k
  (21 builds: the lane path is not made dynamic). The per-segment writes cost host (~+8 ms on 5 x 1600):
  DECIDED (OV-58) back to PB.7B's one-gather write (its shapes follow the segment count only: one program a count,
  built by the n2..n6 / first chunks; padding it would move rows no segment needs, OV-3).
- Phases (tests/packed/test_glued_v2_phases.py, 5 x 1600 @ 139k, one-gather writes back): glued 140.8 ms wall, device
  115.7 (PB.7B 115.3): indexer 10.60 (PB.7B's first call 8.3), indexer_pass 1.44 (now only the passes' top-k; PB.7B
  5.6 with its second ring), indexer_phase 1.79 (0.9). The one ring call is ~2.3 ms slower than PB.7B's first call over
  the 3 index layers (q_chunk 32: each k tile serves one q tile row; key multicast off: every core reads its keys), so
  the indexer saves ~1 ms net at 139k, not the 5.6 expected. Not built (next step): runtime key multicast per device
  (on where all of a chip's ranges read one buffer: most chips), q_chunk 64 where every range is 64-aligned.
- 4 x 2k (the 67k bar's lane chunk) with every part on: 120.2 ms = 68.2k (PB.7B 113.4 / 72.2k; device 103.5 vs 100.0):
  the dynamic parts cost a fixed-lane chunk ~3 ms device + ~3 ms host and buy it nothing (its layouts repeat by
  construction). DECIDED (OV-58): a fixed-lane chunk keeps PB.7B's programs (tt/page_copy.KIND, set by PackedChunk);
  glued, single-segment and PACKED=off chunks take the dynamic parts. Host trims: the snapshot's values packed (page0,
  4 ring rows in one word, 4 owner codes in one word: 3 values a core instead of 9), GluedRows' pieces 2 words each.
- Final oracle rung (characteristic layers, every part on, the lane-chunk gate and the packed values): alone vs today
  0, segment equivalence 0, split 0, digests 0, determinism 0.
- Not done (owner item 2, one upload a chunk): the per-chunk tables (keep rows, compressed keep, window index lists per
  class tuple, server-order index rows per segment, top-k valid lengths, the short-pool caps, ChipRows piece tables,
  the MoE pad words) stay small per-descriptor uploads (cached by value: ~10-15 a new layout). They are the bulk of the
  ~8 ms a new layout costs (the window index lists are a [32, 256, 128] host build). Next: one int32 table a chunk
  appended to the engine input (make_input writes it with the ids, the tt-d-gen side) and one prepared unpack program
  into the persistent per-kind tensors, the window index lists generated on the device from a few words a segment.
- Re-run: op test `resident submit --cmd "<the walls env> scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/packed/test_ring_ranges_ops.py"` (shares the held model); host check
  `PYTHONPATH=$PWD python -m pytest -q models/demos/deepseek_v41_flash_d_p/tests/packed/test_snapshot_dyn_host.py`;
  oracle rung: the gate's glued env + DS41_PACKED_LAYERS=0,1,2,3,14,20,21,24 test_glued_oracle.py (or
  test_glued_dyn_bisect.py with GLUED_DYN_DROPS); program-cache misses by op: the walls env +
  test_glued_dynamic_probe.py; walls: the walls env + GLUED_WALL_CASES / GLUED_WALL_RUNS / GLUED_WARM_MIXES
  test_glued_walls.py; gate `gate PB.7C --spec $BRINGUP_SPEC --check` (devjob).
- Gate check #1 (`gate PB.7C --check`, 1456 s): FAIL. Passed: determinism 0; glued oracle 40 layers (segment equiv 0,
  alone vs today 0, split 0, digests 0, determinism 0); glued reference (every gseg within its floor, top5 1.0); packed
  oracle (lane equiv 0, alone vs today 0). Walls: 4x2k_p139264 71,452 new tok/s (>= 67k; the lane chunk keeps PB.7B's
  programs), 5x1600 57,472 (139.2 ms), n2 / n3 / n5 / n6 112.3 / 124.8 / 142.0 / 150.0 ms, 6x_p1m 239.0 ms (DRAM
  headroom 6.658 GB), agentx_m1 glued fresh: 5 new mixes (17 chunks) 42.6k new tok/s, builds after warm-up 0, ring
  calls a layer 1, the last mix repeated 56.3k (12.2 ms of host a new layout a chunk). Failed: (a) the lanes side of
  agentx_m1 crashed ("Tensor is not allocated", DS41-344: packed_onecall's constant cache evicted by insertion order;
  fixed: LRU), so lanes_serving_tok_s_agentx_m* and agentx_m2 / m3 missing; (b) glued_indexer_ring_calls_per_layer_n1 =
  0: n1 is a single-segment chunk (today's path, one ring call a layer) and the metric counted GluedCall's calls only;
  now counted at the op (every ring_indexer_score_dsa call between two layer ends, the most any layer made).
- Gate check #2's walls (fresh process): every glued chunk took path (b) ("the indexer fork's lanes ring is not built")
  and 4x2k fell to 63.8k: the new op counter wrapped ttnn.bringup.ring_indexer_score_dsa before packed_onecall /
  packed_glued were first imported, and those modules read the op's binding docstring at import to see which fork
  options are built (the wrapper had none). Fixed in the test: import them first, the wrapper carries the docstring.
- Gate check #3 (only the walls unit re-ran; the 40-layer units passed in #2 on the same code): PASS PB.7C.
  Accuracy (#2): glued oracle 40 layers segment equiv 0, alone vs today 0, split 0, digests 0, determinism 0; packed
  oracle lane equiv 0, alone vs today 0; glued reference every gseg within its floor (top5 1.0); determinism 0.
  Walls (table results/PB.7C_glued_walls.md):
  | case | PB.7C | PB.7B (gate, repeated layouts) |
  | 4x2k_p139264 (lane chunk) | 115.3 ms, 71.1k (bar 67k) | 113.4 ms, 72.2k |
  | 5x1600_p139264 | 140.4 ms, 57.0k (device 115.1) | 134.4 ms, 59.5k |
  | n2 / n3 / n5 / n6 | 111.1 / 124.0 / 143.4 / 150.7 ms | 105.8 / 122.4 / 138.2 / 151.5 ms |
  | 6x_p1m | 238.6 ms, DRAM headroom 6.658 GB | 228.1 ms, 6.69 GB |
  | agentx m1 / m2 / m3, FRESH layouts (5 new mixes each) | 42.5k / 46.8k / 46.2k new tok/s, builds 0 / 0 / 0, ring calls 1 | (PB.7B mixes repeated: 57.4k / 52.7k / 47.0k; other mixes) |
  | the last fresh mix's layouts repeated | 53.3k / 55.2k / 51.9k | |
  | lanes fresh | 25.1k / 29.2k / 30.2k (builds 211 / 122 / 128: the lane path keeps PB.7B's programs) | 32.0k / 29.9k / 36.6k |
  Every glued case 1 ring call a layer (n1: today's path, 1). Host a new layout costs on the glued side: 3.4 / 11.8 /
  1.0 ms a chunk (the last mix fresh vs repeated); the fresh-vs-repeat gap over all mixes is larger (~15-20 %): the
  per-descriptor uploads and host tables (not merged into one upload, see above).
- Final check `orchestrator measure --task PB.7C` (quick, switch on, the single-request path): rc 0, chunk_wall_ms 222.0
  (spread 31.6; PB.7B 216.8), min layer PCC 0.732 (L39, as PB.7B), host_transfers_per_layer 0.
- For the owner: the gated items hold (one ring call a layer, 0 program builds after warm-up on fresh AgentX layouts,
  bit-identical oracles, 4 x 2k >= 67k). Below the ask: the one ring call saves ~1 ms device at 139k, not 5.6 (q_chunk
  32 and no key multicast make the call ~2.3 ms slower; next: runtime key multicast per chip, q_chunk 64 where
  aligned); the per-chunk tables are not yet one upload with the ids (next: an int32 descriptor table appended to the
  engine input and one prepared unpack; the window index lists made on the device).

## PPB.7C (PB.7C perf, attempt 3): per-chunk tables derived on the device from the chunk header (PACKED_GLUED_DYNAMIC)

- Start: attempt 2's tree stands (one ring call a layer, 0 builds after warm-up); OV-67: the per-descriptor host tables
  and uploads go. Overseer notes OV-3 / OV-58 / OV-61 / OV-66 / OV-67 read first.
- Expected saving per chunk (written before any code change, perf_guide): device ~0 (+ a few small derive programs,
  ~0.05-0.2 ms a chunk); host on a NEW layout: attempt 2 measured 3.4 / 11.8 / 1.0 ms a chunk (AgentX m1 / m2 / m3,
  fresh minus repeated): the ~10-15 per-descriptor uploads (keep rows, compressed keep, window index lists
  [32, 256, 128] built on the host, server-order rows a segment, top-k / candidate valid lengths a pass, candidate
  caps, ChipRows piece tables, MoE pad words) and their host builds -> ~0 (target <= 1 ms); a repeated layout pays
  the same as a fresh one (no per-descriptor caches): ~+0-0.5 ms host a chunk for the derive launches.
- Load reason (1): tests/packed/test_glued_host_probe.py (diagnostic, not gated) on attempt 2's code: "needs the walls'
  performance load (layers 0-20, decode, resident engram table) because the host time a new layout costs comes from
  every layer's steps; it attributes it (enqueue ms, uploads, cProfile fresh - repeat)".
- Load reason (2): tests/packed/test_glued_derive_ops.py: "shares the held walls model (mesh + its indexers' parameters only): the derived tables vs the host functions, random layouts".
- Load reason (3): test_glued_host_probe.py again on the attempt-3 code (the held walls model): "fresh - repeat host ms and uploads a chunk with the derived tables".
- Load reason (4): the glued oracle rung DS41_PACKED_LAYERS=0,1,2,3,14,20,21,24 (the gate's 40-layer accuracy load, OV-61; computes only the characteristic layers): "bit-identity of the derived tables path before the gate".
- Probe (1) on attempt 2's code (3 warm mixes, 3 fresh mixes each run fresh then repeated, 10 chunks): enqueue 218.8
  vs 150.4 ms a chunk fresh vs repeat (that probe's warm-up was short: the MoE's share of the gap was first-seen
  programs; the walls warm 4 mixes). The per-layout host tables by cProfile, a chunk: _gconst 15.8 ms in all, of it the
  window index lists 10.5 (GluedGeo.region_lists 8.5 on the host + the upload), the compressed keep 2.4, the
  server-order rows 1.3, the keep rows 1.0; 8 more from_torch a chunk.
- Built (PACKED_GLUED_DYNAMIC=on; off keeps attempt 2 / PB.7B): `tt/glued_derive.py`. The engine input of a glued
  chunk carries the chunk header (tt-d-gen's G1 fields, 32 uint32 words: n, (slot, actual_start, actual_end,
  global_write_start) a segment) after every SP row's ids: `[SP, 1, W + 32]`, ONE upload (packed_device.make_input;
  ids_from_input cuts it). `Derive`: one kernel source, five sections (keep + MoE end words, region index lists,
  compressed keep, idx = per pass valid lengths / candidate valid lengths / candidate caps + the ChipRows piece tables at
  fixed slots, srv = the served writes' server-order rows), each a prepared program (chip index baked in at build; the
  header's byte offset, the masked-carry bits, the writing segments as per-run values) into persistent pinned outputs;
  a section runs once a chunk on first use (again only when its values change). The region section on 16 cores
  (8 x 2: DS41-345), the rest on core (0, 0); data movement only. GluedCall (dynamic): _keep_rows / _ckeep /
  _region_index / _srv / _glued_write / _index_dynamic read the derived outputs (ChipRows via `chip_rows_at` with the
  derived table page and a fixed slot); PackedChunk's MoE span from the derived end words; ChipTables only on the
  non-dynamic path (lazy). The header reaches the model through run_chunk(inp=) / begin_packed(inp=) / the adapter's
  prefill_packed (the engine input); without it (tests driving host ids) Derive uploads the 32 words alone.
- DECIDED (OV-58): the kernels' per-call runtime args that follow the layout (ring lane_ranges, sparse_sdpa
  kv_lane_rows, GluedSend's carries / ring pieces, the snapshot owners, the glued rope pieces, page-copy offsets) stay
  runtime args computed from the descriptor every chunk (no cache: a fresh layout costs what a repeated one does; they
  ride with the launch, no upload), as the owner's 16:00 rule (1) names them (the sparse_sdpa kv_lane_rows pattern).
  Moving them into device tables too would need every fork kernel to read its args from DRAM; the probe shows no host
  ms left for a new layout, so that is not done here.
- Op test (2) tests/packed/test_glued_derive_ops.py (shares the held model; random AgentX fills + two segments meeting
  inside a chip at every tile-row offset, 1-token / idle segments, six segments, a full chunk; header from the engine
  input and from the fallback upload): 30 cases x 2 sources, 0 mismatches vs GluedGeo.keep_rows / region_lists (mask 0
  and random), real_prefix end words, the compressed keep, Call's phase / keep / candidate pieces, the valid lengths and
  caps, server_rows + off. First run: DS41-345 (a 16 x 1 core row on the dispatch columns).
- Probe (3) on the attempt-3 code (5 warm mixes, 4 fresh mixes run fresh then repeated, 14 chunks): enqueue 142.5 vs
  145.8 ms, wall 151.1 vs 154.4 ms a chunk (fresh - repeat -3.3 ms: noise), 0 programs built, no table uploads (0.29
  from_torch a fresh chunk: the slot id-history page of note_tokens, a host input).
- Oracle rung (4), characteristic layers 0,1,2,3,14,20,21,24, every dynamic part on (header by the fallback upload):
  alone vs today 0, segment equivalence 0, split vs alone 0, slot digests 0, determinism 0.
- Walls test change: glued_new_layout_ms_<case> (gated <= 1.5) is now every timed mix's chunk walls fresh minus the
  same mix run again right after (all ~15-17 timed chunks, was the last mix only: 3-4 chunks, noisy); the median is
  recorded too (glued_new_layout_ms_median_<case>); the repeated-layout tok/s from all the repeats.
- Known issues used: DS41-342 (the phase rolls to the alone chip rows stay: their piece tables are derived, slot
  'phase'), DS41-316 (the op test shares the held model's request; a failed request still cost a fresh process: the
  released model's memory did not come back), DS41-343 (the ring's padding buffers stay distinct scratch tensors;
  untouched). New: DS41-345.
- Re-run: op test `resident submit --cmd "<the walls env> scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/packed/test_glued_derive_ops.py"`; host probe: the walls env +
  GLUED_PROBE_MIXES=4 test_glued_host_probe.py; oracle rung: the gate's glued env + DS41_PACKED_LAYERS=0,1,2,3,14,20,21,24
  test_glued_oracle.py; gate `gate PB.7C --spec $BRINGUP_SPEC --check` (devjob).
- Gate check (`gate PB.7C --check`, after the last code edit, hooks clean): PASS PB.7C. Glued oracle 40 layers:
  segment equivalence 0, alone vs today 0, split vs alone 0, slot digests 0, determinism 0; packed oracle lane equiv 0,
  alone vs today 0; glued reference every gseg within its floor (final 0.132 vs 0.242); determinism 0.0. Walls
  (results/PB.7C_glued_walls.md):
  | case | attempt 3 (derived tables) | attempt 2 (host tables) |
  | 4x2k_p139264 (lane chunk) | 112.9 ms, 72.6k (bar 67k) | 115.3 ms, 71.1k |
  | 5x1600_p139264 | 138.7 ms, 57.7k | 140.4 ms, 57.0k |
  | n1 / n2 / n3 / n5 / n6 | 99.2 / 112.6 / 122.2 / 139.3 / 152.1 ms | 98.9 / 111.1 / 124.0 / 143.4 / 150.7 ms |
  | 6x_p1m | 237.6 ms, DRAM headroom 6.66 GB | 238.6 ms, 6.66 GB |
  | agentx m1 / m2 / m3 FRESH (5 new mixes each) | 47.2k / 51.0k / 51.7k new tok/s | 42.5k / 46.8k / 46.2k |
  | the same mixes repeated | 46.8k / 51.5k / 51.7k | 53.3k / 55.2k / 51.9k (last mix only) |
  | glued_new_layout_ms (fresh - repeat, every timed chunk) | -1.40 / 1.38 / 0.17 (medians -1.13 / 0.88 / -0.66) | 3.36 / 11.79 / 1.01 (last mix) |
  | program builds after warm-up, ring calls a layer | 0 / 0 / 0, 1 | 0 / 0 / 0, 1 |
  | lanes fresh | 25.1k / 28.6k / 30.1k (builds 212 / 122 / 128: the lane path keeps PB.7B's programs) | 25.1k / 29.2k / 30.2k |
  Fresh AgentX glued walls +11 / +9 / +12 % over attempt 2; fresh now equals repeated (the noise is ~+-1.5 ms a chunk).
- Final check `orchestrator measure --task PB.7C` (quick, switch on, the single-request path): rc 0, chunk_wall_ms 240.0
  (spread 27.7; attempt 2 222.0 spread 31.6, PB.7B 216.8), min layer PCC 0.732 (L39, as before),
  host_transfers_per_layer 0. That path runs none of attempt 3's code (Derive runs only in a glued chunk's GluedCall;
  the header only in a glued chunk's input): the gap is inside the run-to-run spread.
- For the owner: the per-chunk host tables are gone from the glued path (one upload a chunk: ids + header; tables
  derived on the device; no per-descriptor cache); fresh AgentX layouts now cost what repeated ones do, and the fresh
  walls rose ~10 %. Not moved: the kernels' per-call runtime args (computed from the descriptor each chunk, no upload,
  DECIDED above). The packed_glued module docstring still says "uploaded once a descriptor" for the non-dynamic path
  (PACKED_GLUED_DYNAMIC=off); left untouched after the gate so the recorded runs stay valid.

## PPB.9F (PB.9F perf, attempt 1): HONEST fresh walls (WALLS_FRESH=1), a measurement: tests only

- Overseer notes OV-3 / OV-58 / OV-61 / OV-67 / OV-68 read first; PACKED_RULES.md read. A MEASUREMENT task (owner 19:00):
  no tt/ or fork change; a path that fails on fresh positions is recorded, not fixed.
- Expected saving per chunk (perf_guide, written before any code change): device 0 ms, host 0 ms (nothing in the model
  changes). What the fresh mode should SHOW: on the single-request and fixed-lane paths, program builds after warm-up > 0
  (OV-68's slices: RoPE begin=start, compressed-row RoPE start//r, DeviceScalars.at, slab.slice_rows slot x layers +
  row, replay offsets) and a fresh-vs-repeated wall gap of the builds' host cost (each build ~ms); glued ~0 (PB.7C).
- Built: tests/packed/fresh_common.py (AgentX draws; `Slots`: serving's slots = a DS41ContractKV of WALLS_FRESH_SLOTS (32)
  slots over the loaded layers, states bound to their slabs as the adapter's state(), warm-up pool 0-15 / timed 16-31;
  `BuildProbe`: the PB.7C dynamic probe's wrap, a build charged to the innermost wrapped call, its arguments diffed
  against the previous call at the same site = the attribute that changed; the table writer). WALLS_FRESH=1 in
  test_skip_positions.py (single<chunk>_steady / _resumed / _partial, PACKED=off through run_chunk: the adapter's path
  from the engine input), test_packed_walls.py (lanes4x2k), test_glued_walls.py (glued_/lanes_agentx_m1..3,
  glued_5x1600; PB.7C's fresh flow + slots, profiled repeats, probe). Gated names: <case>_fresh_tok_s,
  program_builds_after_warmup_<case>; glued's old list metric `<side>_fresh_mixes_*` renamed `<side>_mixes_timed_*`
  (a list under the gate's `*_fresh*` > 0 fails float()).
- DECIDED (OV-58): fresh slots = the contract's caches (own new_state() slabs are slot 0 / one batch: a slot-keyed program
  never shows there); the slot index is the slab's batch row (slot x layers + row) as in serving. 32 slots at the 1M
  capacity ~ +4.7 GiB a chip (PB.5C: ~0.148 GiB a slot over layers 0-20).
- DECIDED: builds after warm-up = the cache entries of every fresh timed iteration (the untimed real previous chunk
  included: in serving it is a fresh chunk too); the repeats and profiled runs are not counted.
- Load reason: "needs the gate's perf load (layers 0-20, decode, 8192, resident engram table) because the walls are
  whole-chunk walls through every executed layer; one smoke (1 warm / 1 timed / 1 probe draw a case, all three tests in
  one process) then the gate units on the same held model".
- Smoke (resident, 1 draw a case): every path ran on contract slots; the lane path built ~50 programs a fresh chunk.
- Gate check #1 (`gate PB.9F --check`, 497 s): PASS. Glued fresh walls showed 8 of ~50 chunks at 0.4-1.4 s with 0
  program-cache builds (glued -28..-58 % vs repeated). Two changes to the measurement (tests only): (a) `PrepCount`
  counts ttnn.bringup.prepare_program calls in the timed runs (a prepared program whose per-run values are value
  slots keeps one cache hash, so a new prepare adds no cache entry: it was invisible), and the probe attributes host
  gaps before a call and slow calls by site; (b) every slot's state is built before timing (the adapter builds a
  slot's state lazily on its first request: once a process, not a chunk; it was paid inside the first fresh chunks of
  each slot). Cause of the spikes not proven: with (b) none came back (max 275 ms), prepared builds on the glued side 0.
- Gate check #2 (after the last code edit, hooks clean): PASS PB.9F (637 s; determinism 0.0). Results
  (results/PB.9F_fresh_walls.md; 8192, every accepted packing switch on, layers 0-20, decode):
  | case | warm-repeated tok/s | fresh tok/s | builds after warm-up (cache + prepared) | host ms/chunk fresh (rep) | device ms |
  | single8192_steady | 72.2k | 72.6k | 1 + 1 / 8 iterations | 13.2 (11.3) | 102.9 |
  | single8192_resumed | 72.3k | 72.5k | 0 + 0 | 17.1 (15.2) | 98.0 |
  | single8192_partial (last chunks, 372-4583 rows) | 14.1k | 14.1k | 0 + 0 | 11.5 (12.8) | 122.0 |
  | lanes4x2k | 67.8k | 59.0k (-13 %) | 346 + 3 / 8 iterations (16 chunks) | 38.3 (14.8) | 109.2 |
  | lanes_agentx_m1 / m2 / m3 | 33.6k / 37.1k / 36.0k | 23.3k / 26.5k / 28.9k (-31 / -29 / -20 %) | 403+22 / 251+11 / 264+5 | 90 / 53 / 45 (16 / 13 / 16) | 99 / 93 / 96 |
  | glued_agentx_m1 / m2 / m3 | 47.9k / 52.4k / 52.5k | 45.1k / 51.5k / 52.0k (-6 / -2 / -1 %) | 1 / 0 / 0, prepared 0 | 41 / 21 / 24 (22 / 19 / 23) | 130 / 113 / 117 |
  | glued_5x1600 | 52.5k | 48.1k (-8 %) | 0 + 0 | 36.7 (17.2) | 131.3 |
  The single-request path builds nothing at 8192 here: PACKED_GLUED_DYNAMIC=on gives it PB.7C's dynamic parts (RoPE
  rows, DeviceScalars picks as page copies); OV-68's slices are on the default path, PB.9F5 (5120, switches off)
  measures that. The lane path's builds (probe, by op and changed argument): ttnn.slice at served.py:155 slice_rows
  (slice_start / slice_end: the slab batch row slot x layers + row, the rows a history needs) from
  history_gather._pack / _tail_part / prefetch_prepare; served.py:211 DeviceScalars._pick (k = start // 32) from
  compressor._put_into 730-732; history_gather.py:376 _tail_part; snapshot_send run_prepared (new owner patterns,
  prepared); packed_onecall ring_indexer_score_dsa (chunk_start_idx) on AgentX lanes. Its fresh host cost is
  +23..+74 ms a chunk (DS41-346).
- Known issues: DS41-185 / DS41-314 used (the switches inside the tests through packed_common.switch, never
  settings.refresh); DS41-344 (the lanes side's LRU constants: no crash on fresh AgentX lanes); new DS41-346.
- Re-run: gate `gate PB.9F --spec $BRINGUP_SPEC --check` (devjob); one test: the gate's env +
  WALLS_FRESH_WARM / _RUNS / _PROBE / _PROF / _SLOTS / _CASES (e.g. WALLS_FRESH_CASES=steady,lanes4x2k,agentx_m1)
  through `resident submit`. PB.9F5: the same test_skip_positions.py at BRINGUP_CHUNK=5120.
- OV-67 numbers (fresh minus the same chunks repeated, every timed chunk; gate check #2): glued_new_layout_ms
  agentx m1 / m2 / m3 8.5 / 2.5 / 1.5 ms (medians 0.2 / 0.4 / -0.3), 5x1600 14.0 (median 17.0) with 0 builds and 0
  prepares: on serving slots at fresh positions the glued 5 x 1.6k chunk still pays ~15 ms host a new layout that no
  probe gap > 5 ms names (many small costs; not attributed here). Lanes 49.7 / 44.1 / 27.5 ms (the builds, DS41-346).
  One ring call a layer on every glued case.
- Final check `orchestrator measure --task PB.9F`: not run. The orchestrator run is paused (its PAUSE file exists:
  "paused before A/B on: one process for quick; `resume` continues the task", rc 4); the pause is the overseer's, left
  as it is. Nothing in tt/ or a fork changed (tests only), so the quick A/B has no switch to compare.
## MB.1 perf (attempt 1): honest 5120 fresh walls on the shipped branch (WALLS_FRESH=1): a measurement, tests only

- Read first: OV-3, OV-57, rule 20, the skill's standing decisions of 2026-11-20, perf_guide. A MEASUREMENT task (owner):
  no tt/ or fork change; a path that builds on fresh positions is recorded, not fixed (MB.2 fixes it).
- Expected saving per chunk (perf_guide, written before any code change): device 0 ms, host 0 ms (nothing in the model
  changes). What it should SHOW (OV-57, PB.9F5 on the packed branch with its switches off: ~22 builds a fresh 5120
  chunk, fresh 37-51k vs repeated 61-63k tok/s): builds after warm-up > 0 on every kind, from ttnn.slice keys
  (served.py slice_rows / DeviceScalars._pick, rope.py RoPE rows, compressor.py compressed-row RoPE incl. the table
  INPUT shape, history_gather tail, attention.py srv_row), and a fresh-vs-repeated host gap of the builds' cost.
- Port: tests/fresh_common.py (PB.9F a72c8c6918d tests/packed/fresh_common.py without the lane / glued parts:
  agentx_mix dropped; `Slots` now = the adapter's engine-owned KV cache + runtime.state(kv, slot), every slot's state
  built untimed). tests/test_skip_positions.py WALLS_FRESH=1: the timed call is `runtime.prefill_chunk(inp, kv,
  slot_id, actual_start, actual_end, global_write_start)` of the adapter's runtime (built as prefill_runner builds it,
  adapter.build_runtime -> allocate_kv_cache(WALLS_FRESH_SLOTS) -> compile), not PB.9F's packed run_chunk (this
  branch has no tt/packed*). The runtime loads through adapter.load_model (FIX.29), so the batch's held model, never
  a second copy; the engine's device input (server_rules.server_payload -> caches.payload_tensor) is made before the
  clock.
- Load reason: "needs the gate's perf load (layers 0-20, decode, 5120, resident engram table, position tables to
  1,044,480) because the walls are whole-chunk walls through every executed layer at AgentX positions up to ~1M; one
  smoke (1 warm / 1 timed / 1 probe draw a kind) then the gate units on the same held model; 8192 once (report only)".
- Smoke (resident, 1 draw a kind, task MB.1.smoke, results /tmp): every kind ran on the adapter runtime and the held
  model (no second load: the runtime's adapter.load_model got the batch's model, FIX.29); probe names the sites.
- Gate check (`gate MB.1 --check`, after the last test edit, hooks clean): PASS (470 s; determinism 0.0). Results
  (results/MB.1_fresh_walls.md), 5120, mean of 8 timed chunks, fresh vs repeated:
  | case | fresh ms | repeated ms | builds (8 iterations, prev + timed chunk) + prepared | host ms fresh (rep) | device ms |
  | single5120_steady | 95.5 | 80.8 | 177 + 2 | 36.7 (13.6) | 66.9 |
  | single5120_resumed | 91.4 | 81.1 | 179 + 2 | 26.3 (17.2) | 63.4 |
  | single5120_partial | 123.5 | 102.6 | 156 + 8 | 37.6 (17.7) | 85.5 |
  ~11 builds a fresh chunk, all ttnn.slice (slice_start / slice_end in the program hash) + snapshot_send prepares on
  last chunks; sites in DS41-347 and the results file. Fresh tok/s 53.6k / 54.6k / 14.4k (repeated 63.4k / 61.6k /
  17.3k). The numbers match PB.9F5 (packed branch, switches off, same seeds) within ~2 ms.
- 8192 (report only, task MB.1.c8192, results in /tmp, copied into MB.1_fresh_walls.md): builds 161 / 175 / 154, same
  sites. PB.9F's 0 at 8192 came from its command's DS41_PACKED_GLUED_DYNAMIC=on (a72c8c6918d tasks.yaml): on that branch
  page_copy.dyn() covers every non-lane chunk, single-request included, at exactly these sites (rope / compressor
  "rows", served "pick", history_gather "hist", attention "pages"). This branch has none of it.
- Rule 20 per-chunk values: no model change. The test passes per chunk only what the engine sends (the [sp, 1, chunk /
  sp] uint32 payload made before the clock, and prefill_chunk's slot_id / actual_start / actual_end /
  global_write_start). The values that reach a program key today (found, not fixed): start (RoPE rows, compressed-row
  RoPE start // r, DeviceScalars k = start // 32, ceil(end / 32)), slot (slab batch row slot x layers + row), the
  history length (slice_rows / _tail rows), end of a last chunk (snapshot owner patterns).
- Known issues: DS41-206 / DS41-208 used (decode truncates a reused slot's window instead of refusing, so slots can be
  drawn at any position); DS41-281 out of date (the op is now named: knowledge_use report); new DS41-347.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate MB.1 --spec $BRINGUP_SPEC --check` (devjob); one
  case: the gate's env + WALLS_FRESH_WARM / _RUNS / _PROBE / _PROF / _SLOTS / _CASES through `resident submit`;
  8192: the same command with BRINGUP_CHUNK=8192 and `--task MB.1.c8192 --results <dir>`.
- Final `orchestrator measure --task MB.1`: not run (no switch: tests only; nothing in tt/ or a fork changed).

## MB.2 perf (attempt 1): no program build in serving on the single-request path (POS_DYNAMIC off | on)

- Read first: OV-3, OV-57, rule 20, perf_guide, MB.1's results (DS41-347). Expected saving per chunk, written before
  any code change (perf_guide section 2): FRESH chunks (new starts / slots / ends, serving's real traffic): host -10 to
  -21 ms a chunk (MB.1 fresh vs repeated: steady 95.5 / 80.8, resumed 91.4 / 81.1, partial 123.5 / 102.6 ms; the gap is
  the ~11 program builds a fresh chunk), so fresh ~= warm; WARM-REPEATED chunks: host ~0 (a prepared page copy costs
  about what a cached slice costs; the ids re-upload removed: -0.3 to -1 ms host), device +0.05 to +0.2 ms (the ids
  reorder: one SP all_gather of 20 KB + a 4-core copy kernel; page copies instead of slices: same bytes, fewer cores).
- Load reason: "test_no_rebuild needs the gate's perf load (layers 0-20, decode, position tables to 1,044,480) because
  the fresh chunks are serving's at AgentX positions up to ~1M; the iteration ran it alone on the held model (143 s)".
- What (POS_DYNAMIC=on, call-time switch: hooks.batch_call_settings + batch_call_apply refreshes only this switch in a
  held model's snapshot, DS41-314's rule): tt/page_copy.py (ported from the batching branch's PB.7C page_copy:
  prepared, value slots s0 / d0 / n / nb / ss / ds, 4 data-movement cores a chip; new: a frozen per-mesh-row source
  offset ``sp_stride`` and ``IdsReorder``). Sites (MB.1 probe, OV-57): rope.RopeTable._rows (each mesh row its natural
  block's tile pages, no mesh_partition), compressor.CompressRope._rows (rows by mesh row, tiled there),
  served.DeviceScalars._pick (k = start // 32, ceil(end / 32), slot), served.CacheSlab.slice_rows (page = b R + row:
  every caller, history_gather _pack / _tail included), attention.WindowTables.srv_row, compressor._write_from (replay
  rotation = 2 page copies), history_gather._pack_fixed (whole history in PACK_BUCKET pieces, the last moved down to
  end at R; a tail in one piece of ceil32(2 hs) + 32 rows), _tail (always 2 hs rows: h moved down, rows the prefetch
  already gathered, same bytes), snapshot_send._call_dyn (the batching branch's pattern-as-values program, gated by
  POS_DYNAMIC), index_keys._decode (fork option unpack_fp8_rows(whole_grid=True)). adapter._run: the embedding ids from
  model.reorder_ids(inp, start, run_end - start) (SP all_gather of the engine input into a load-time buffer + one
  core a chip: server order -> natural, pad / vocab clamp to vocab - 1), embed(dev_ids=): no _host_emb_ids upload; the
  readback stays for note_tokens / the engram hash (engrams on the host, owner). adapter.compile under POS_DYNAMIC=on
  warms the finite set: [0, c), [c, 2c), the pulled-back partial last chunk [c + 1632, 2c + 1620) with ws 2c (decode).
- Cores (rule 18): page copies 4 cores a chip, IdsReorder 1 core a chip, snapshot send home + port cores (as before),
  unpack_fp8_rows whole_grid = max_grid's 110 cores (as its large calls before); no matmul changed.
- Rule 20, every per-chunk value touched and how it reaches the device: start (RoPE / compressed-RoPE first page,
  DeviceScalars index, srv_row index, IdsReorder (start mod W) / 32 and (start / W) mod SP) = prepared value slots;
  end (ceil(end / 32) index, IdsReorder valid = run_end - start, the snapshot's cap -> block codes / ring rows packed in
  2 value slots) = value slots; slot (DeviceScalars index, slab batch page b R) = value slots; history length (pack
  pieces' first rows, tail h, unpack rows / bands) = value slots on fixed shapes; replay offset (lo - start) / r = page
  copy counts. Shapes depend only on the chunk, the mesh and the slab geometry. Host per chunk: the engine's ids
  readback (engram hash) only; nothing uploaded but the engram rows.
- Gotcha: test_no_rebuild's first on / off check reused a draw's slot: 13.2 apart from stale history rows (DS41-348);
  the scratch diagnostic then showed every component (rope, crope, srv_row, scalars, ids) and every layer of both
  chunks 0.0 on vs off.
- test_no_rebuild (gate unit, 143 s): 8 fresh kinds (aligned, steady, resumed, partial, pulled-back with replay,
  acks-only, steady, partial) after runtime.compile only: builds 0, prepares 0, ids uploads 0, probe 0 entries; on vs
  off on the pulled-back chunk 0.0.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate MB.2 --spec $BRINGUP_SPEC --check --unit
  MB.2#test_no_rebuild` (devjob); fork test: ttnn/ttnn/bringup/fp8_rows_ttnn/tests/unit/test_unpack_fp8_rows.py.
- Fresh walls, POS_DYNAMIC=on (measure --part test, ab/MB.2/new/test_skip_positions; copied to
  results/MB.2_fresh_walls.md), 5120, 8 timed chunks a case, fresh vs warm-repeated (MB.1 fresh / repeated before):
  | case | fresh tok/s | warm tok/s | builds + prepared | host ms fresh (rep) | device ms |
  | single5120_steady | 61391 (53.6k) | 61240 (63.4k) | 0 + 0 | 14.6 (14.4) | 67.8 |
  | single5120_resumed | 59873 (54.6k) | 59865 (61.6k) | 0 + 0 | 16.9 (17.1) | 64.2 |
  | single5120_partial | 18634 (14.4k) | 18614 (17.3k) | 0 + 0 | 15.4 (13.4) | 86.3 |
  Fresh == warm (the target). Warm-repeated steady / resumed ~2-3 ms a chunk slower than MB.1's repeated (the history
  pack in PACK_BUCKET pieces, the 2 hs tail; same seeds): the owner's A/B table decides.
- Gate grouping (F283): the gate refused 2 executed-layer loads, test_skip_positions' command sets BRINGUP_CHUNK=5120
  (the spec's own chunk) and test_no_rebuild's does not; tasks.yaml and core/batch.py are not this step's: the test
  declares BATCH_ITEM_VARS = ("BRINGUP_CHUNK",) (its chunk is the spec's, F271's override when set; a run at another
  chunk still gets its own model: shared_model compares the load's chunk). Framework fix to consider: normalize
  BRINGUP_CHUNK equal to target.chunk as unset (core/batch.normalized).
- Gate check (`gate MB.2 --check`, after the last code edit, hooks clean; 604 s, 2 loads: executed + full): every unit
  rc 0 and every metric ok (ladder err_layer L00-L39 / err_state within their noise floors, determinism 0.0,
  contract_checks_failed 0, outside_history 0, program_builds_after_warmup_* 0, ids_uploads_per_chunk 0.0, every
  <case>_fresh_tok_s > 0) BUT one: `prefix_hit_limit_exceed_fresh = 0 (need > 0)`: the threshold glob `*_fresh*`
  catches test_decoder_skip_decode's limit-exceedance count (0 = good; 0 in FIX.39 / X.5): DS41-349, a tasks.yaml
  threshold to narrow (`*_fresh_tok_s*`); thresholds and frozen tests are not this step's to change.
- Final `orchestrator measure --task MB.2` (switch on): test_skip_positions rc 0, test_no_rebuild rc 0. Fork
  fp8_rows_ttnn: full suite (safe runner directly: its test module opens its own mesh, which the resident runner's
  held mesh cannot serve) 10 passed with the option off + whole_grid passed after its program count was keyed to the
  current mesh (the precompile pass's mesh held one too).
- Not done (follow-ups): 8192 fresh walls not re-measured (the same code paths; MB.1's 8192 sites are the 5120 ones);
  the tail's 2 hs / overlapping pack pieces could become runtime row counts (slice_write / ckv_bfp4) to drop the few
  re-moved rows (OV-3 answer); the A/B profile was not run (the orchestrator's table decides).

## PB.MB perf (attempt 1): model_bringup's no-rebuild fix (MB.1 + MB.2) into the batching branch (POS_DYNAMIC)

- Expected saving (written before any device run): 8192 single request ~0 device ms, ~0.3-1 ms host a chunk (the ids
  reordered on the device: no second ids upload; the 8192 path already built 0 programs fresh, PB.9F). 5120 single
  request (PB.MB5): ~22 builds a fresh chunk -> 0; fresh 37.2k / 50.6k tok/s (PB.9F5 steady / resumed) -> ~60k (MB.2's
  61.4k / 59.9k). Packed / glued chunks: none intended (the same runtime-value parts PB.7C already had; glued history
  packs now MB.2's fixed row counts: a few re-moved rows, << 1 ms).
- Applied: `git diff bcab6af973b dnijemcevic/model_bringup -- <model paths> ttnn/ttnn/bringup | git apply --3way`
  (path-limited; no ledger, no framework). Clean: model.py (embed(dev_ids=), reorder_ids), adapter.py (ids reorder,
  compile's chunk kinds), settings.py (POS_DYNAMIC, default on), the fp8_rows fork's whole_grid. Conflicts resolved:
  - ONE predicate a site, `tt/page_copy.dyn(part)`: PACKED_GLUED_DYNAMIC=on turns on GLUED_PARTS (PB.7C's 8),
    POS_DYNAMIC=on turns on POS_PARTS (rows, pick, pages, snap, hist + MB.2's slab, replay, decode); never inside a
    fixed-lane chunk (KIND "lanes": out of scope, still PB.7B's programs). MB.2's `on()` sites (served.slice_rows,
    compressor replay rotation, index_keys decode, snapshot send, srv_row, DeviceScalars._pick, history pack / tail)
    all call dyn(<part>) now; adapter-level ones (IdsReorder, compile kinds) call on().
  - RoPE / compressed-RoPE rows: ONE implementation, PB.7C's (a plain start = one glued piece -> lane_rows.GluedRows,
    pieces as runtime values); MB.2's per-mesh-row `sp_stride` page copies dropped (their only users), page_copy's
    kernel back to PB.7C's arg list (+ MB.2's sharded-geometry key and the n == 0 skip).
  - History pack: MB.2's `_pack_fixed` (fixed row counts: PACK_BUCKET pieces, a tail of ceil32(2 hs) + 32 rows moved
    down) for both paths (PB.7C's tail piece was b - a rows: one program a tail length); `_tail` takes the largest
    tail under dyn("hist"); `_tail_part` keeps this branch's (loc, pages_of) form.
  - index_keys._decode: this branch's grid choice (PB.6L small rectangle / wide grid) kept, `whole_grid=dyn("decode")`
    of that grid: at most two programs, none a band count.
  - snapshot_send: dyn("snap") (MB.2's cap and PB.7C's glued patterns: the same _call_dyn).
  - hooks.batch_call_settings: this branch's list + POS_DYNAMIC; batch_call_apply refreshes POS_DYNAMIC in the load's
    snapshot (MB.2, DS41-314).
  - tests: ONE fresh-walls module, tests/fresh_common.py (MB.1's + PB.9F's agentx_mix and both `Slots` modes:
    `Slots(runtime, kv)` adapter slots, `Slots(model, mesh, cap)` contract caches); tests/packed/fresh_common.py is
    that module object (sys.modules alias), so the packed walls' imports stay. test_skip_positions: MB.1's fresh walls
    through adapter.prefill_chunk replace PB.9F's run_chunk(PACKED=off) walls.
  - known_issues: model_bringup's DS41-321 / 322 / 323 collide with this branch's: renumbered DS41-347 / 348 / 349
    (BREADCRUMBS' MB sections and page_copy.py too).
- Load note (PB.MB): first device run = the test_no_rebuild unit at 8192 (layers 0-20, the gate unit itself: needs the served runtime over the executed layers because the check is the serving path end to end; no smaller piece covers prefill_chunk + compile kinds).
- DECIDED (OV-58): index_keys._decode under POS_DYNAMIC uses MB.2's one whole-grid program for every range, not PB.6L's small rectangle below 16k rows (that grid choice is a history-length-keyed program: test_no_rebuild 8192 built 1 at a 525k resume, DS41-350); the rectangle's host saving at short histories is given up for 0 builds (rule 20 binds).
- Per-chunk values this change touches and how they reach the device (rule 20): chunk start -> RoPE / compressed-RoPE
  rows (GluedRows piece values), DeviceScalars index (_pick page offset), server-order row index (srv_row page
  offset), IdsReorder ((start mod W) / 32, (start / W) mod SP, valid = run_end - start: prepared value slots); slot x
  layer -> served slab batch row (slice_rows page offset); history length -> pack pieces (fixed row counts, ckv_bfp4
  row0 a value slot), tail start (moved down to the largest tail; page offsets); replay offset -> two page copies'
  offsets; last-chunk cap -> snapshot block codes (values); decode range -> unpack_fp8_rows row_lo / band count
  (values; one whole-grid program). No host table, no program key on any of them.
- Results (8192, gate `gate PB.MB --check`): PASS, every unit: packed oracle and glued oracle (every *_max_abs_diff /
  digest mismatch == 0), test_chunk8192_reference (err_final_hidden / top5 within limits), test_skip_positions
  (WALLS_FRESH through adapter.prefill_chunk: program_builds_after_warmup_single8192_* == 0; fresh 69.1k steady /
  70.5k resumed / 13.7k partial), test_no_rebuild (0 builds, 0 prepares, 0 ids uploads, on vs off 0.0),
  test_determinism 0.0. First no-rebuild run FAILED on 1 build at index_keys._decode (DS41-350; fixed, see DECIDED).
- F283 gate note: test_determinism's command has no BRINGUP_CHUNK=8192, so it loads the 5120 40-layer model and the
  8192 oracles are refused in the same gate process ("needs another model ... refused, spec tests.one_load"); the
  oracles ran on the next `--check` (determinism kept). A tasks.yaml matter (BRINGUP_CHUNK=8192 on that unit), not the
  model's.
- PB.MB5's units at 5120 run by the agent (resident submit --task PB.MB, request 20261121-003957; the gate itself is
  BLOCKED until PB.MB passes): ladder last (err_final_hidden 0.135 <= 0.242, top5 1.0 >= 0.994), determinism 0.0,
  decoder skip (contract_checks_failed_decoder_skip 0), contract 0, outside history 0.0, fresh walls 5120 (0 builds
  every case; 56.3k steady with one 132.9 ms outlier chunk, 57.8k resumed, 18.2k partial), test_no_rebuild 0 / 0 / 0.0.
- Fresh vs warm table with PB.9F / PB.9F5 / MB.2: results/PB.MB_fresh_walls.md.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate PB.MB --spec $BRINGUP_SPEC --check` (devjob);
  5120: the PB.MB5 gate command through `resident submit --spec $BRINGUP_SPEC --task PB.MB --cmd "..."`.
- Follow-ups: the 8192 adapter walls' host +5-7 ms vs PB.9F's run_chunk walls (same device ms) is the adapter's
  per-chunk host path (MB.2 saw the same at 5120): a host-path pick; fixed lanes still rebuild (PB.9X).
- Final check `orchestrator measure --task PB.MB` (quick, POS_DYNAMIC on): rc 0, chunk_wall_ms 247.7 (spread 25.4; PB.7C 240.0 spread 27.7, PB.7B 216.8 on the same path), min layer PCC 0.732 (L39, as before), host_transfers_per_layer 0.

## PB.9X perf (attempt 1): every packed chunk through the glued path (PACKED_ROUTE glued | lanes)

- Read first: OV-3 / OV-58 / OV-61 / OV-67 / OV-68 / OV-69, PACKED_RULES.md, rule 20, perf_guide.
- Expected saving per chunk (written before any code change, perf_guide s2): FRESH lane-shaped chunks (4 x 2k at new
  slots / positions): host -20 to -25 ms a chunk (PB.9F lanes4x2k fresh host 38.3 ms vs repeated 14.8: the ~45 slice
  builds a fresh chunk -> 0 through glued's runtime-value parts); device +0 to +10 ms a chunk (the glued one-call path
  on a lane-shaped layout vs the lane path's 109 ms: PB.7C n2 / n3 112.6 / 122.2 ms walls suggest ~+5 ms); net fresh
  4x2k ~59.0k -> ~63-68k tok/s; warm-repeated 4x2k ~0 to -5 % (67.8k lanes -> maybe 63-67k glued). Padded lanes /
  AgentX lane mixes: fresh 23-29k -> glued's 45-52k (they become glued back-to-back chunks: fewer chunks, no pad rows).
  Glued's remaining fresh builds (agentx_m1 1, 5x1600 0-2): host ~-1 to -3 ms on the chunks that hit them.
- Premise check (DISAGREE noted): PB.7C's "glued 4x2k 72.6k" was the LANE path (PB.7C.json glued_path_4x2k_p139264 =
  "lanes": GluedChunk.as_lanes sent it there) at a repeated position; no 4 x 2k chunk has ever run the glued path.
- Load reason (1): test_packed_oracle at DS41_PACKED_LAYERS=0,1,2,3,14,20,21,24 on the gate's 40-layer 8192 accuracy
  load (OV-61; computes only the characteristic layers): "the lane headers now run as glued chunks: bit-identity of
  alone / lane equivalence / slot digests needs the device path; the characteristic layers cover every layer role".
- Oracle rung (characteristic layers, resident request 20261121-012516): alone vs today 0.0, lane equivalence 0.0,
  slot digests 0, determinism 0.0 with every lane header run as its glued chunk (packed1 2 passes a chip).
- Load reason (2): test_packed_walls + test_glued_walls fresh (smoke: 3 timed draws a case, 6 probe draws) on the gate's
  perf load (layers 0-20, decode, resident engram table): "the builds after warm-up and the fresh walls are whole-chunk
  numbers through every executed layer at AgentX positions to ~1M; no smaller piece shows a program key of a layer".
- Load reason (3): a fresh resident process (the C++ rebuild restarts it: an empty program cache, so builds after the
  warm-up are honest) on the perf load: walls smoke + build probe + host probe: "the builds after serving's warm-up
  and the host enqueue of a glued chunk need every executed layer; nothing smaller shows a layer's program key".
- What (PACKED_ROUTE=glued, default; lanes = PB.7's routing for the A/B row only; call-time, hooks.batch_call_settings,
  refreshed in batch_call_apply like POS_DYNAMIC, DS41-314):
  - tt/packed_device.as_run_desc: every packed chunk runs glued (a lane-shaped GluedChunk as itself; a ChunkDescriptor
    with > 1 lane as ChunkDescriptor.as_glued: its live segments in lane order, back to back, no pad rows). run_chunk /
    make_input / ids_from_input re-lay a routed lane header (tokens, states in segment order, the engine input = the
    glued one with the header); tests' packed_common.boundary_rows reads the routed segment's glued rows.
  - adapter.prefill_packed: the routed descriptor end to end; embedding ids made ON THE DEVICE from the engine input's
    header (OV-69: page_copy.GluedIdsReorder via packed_device.glued_dev_ids: one SP all_gather + a one-core kernel
    deriving every segment's engine / model offsets from the header; no host value), the readback stays for the engram
    hash. Checked: tests/packed/test_glued_ids_ops.py, 25 layouts (lane headers, padded, empty lanes, 1-token / idle
    segments, AgentX fills) 0 mismatches vs the host ids, 0 builds after the first.
  - Builds after warm-up (probe tests/packed/test_glued_build_probe.py, DS41-351): history_gather.tail_prepare takes the
    largest tail (as MB.2's _tail); adapter.compile (and the glued walls' warm-up) runs packed_device.warm_glued_chunks
    (22 chunks, ~4 s: every segment count fresh / continued, lane-shaped exact, near the slot's end, 1-token + idle).
    Fresh process after warm-up: every probe layout (histories 0 .. 1,032,192, 1-6 segments, padded / empty lanes) 0.
  - Host (glued 4 x 2k was host-bound: enqueue 126-136 ms vs device ~104; lane path 101-118): prepare_program COMMON
    value slots (C++ fork change, arg index | 1 << 31) used by unpack_fp8_rows(common_values) 0.344 -> 0.055 ms a call
    (24 a chunk), GluedRows pieces 0.404 -> 0.201 ms (8 a chunk), ckv_bfp4(dynamic_row0) in the history pack 0.068 ->
    0.049 ms (36 a chunk); setting COMMON_VALUE_SLOTS (default on), bytes equal (tests/packed/test_common_values_ops.py,
    0 mismatches). Fork tests: fp8_rows test_unpack_fp8_rows_common_values + _whole_grid passed (safe runner directly: the
    module opens its own mesh); ckv_bfp4 test_ckv_bfp4_dynamic_row0 written, not run (single card: rule 19).
- Cores (rule 18): GluedIdsReorder 1 core a chip (data movement); no matmul changed; unpack / ckv_bfp4 grids unchanged.
- DECIDED (OV-58): the fixed-lane path stays behind PACKED_ROUTE=lanes for the A/B row (not deleted); test_packed_walls'
  fresh lane cases go through the adapter runtime (build_runtime -> allocate_kv_cache -> compile -> prefill_packed).
- DECIDED (OV-58): warm-up = a finite enumerated set (rule 20 / OV-68 allow it); write_lanes keeps one program a
  writing-segment count (padding its gather to 6 would move rows no segment needs, OV-3).
- Rule 20, per-chunk values this change touches: segment starts / ends / slots of a lane header -> the glued header
  words (make_input, one upload with the ids) -> on the device: Derive tables (PB.7C) and GluedIdsReorder offsets; tail
  start h (history) -> page offsets / ckv_bfp4 row0 common value slot; unpack range -> common args; GluedRows pieces ->
  common args. No program key, no host table; ids uploaded 0 times (device reorder).
- Gate checks (after the last code edit; hooks clean): accuracy all ok: packed oracle (40 layers, every lane header
  through glued) alone vs today 0.0, lane equivalence 0.0, slot digests 0; glued oracle every metric 0; glued reference
  err_gseg* / top5_overlap_gseg* within their noise floors; determinism 0.0. Builds: every program_builds_after_warmup_*,
  glued_program_builds_after_warmup*, lanes_program_builds_after_warmup* == 0; ids_uploads_per_chunk 0. FAIL (left for
  the owner, never reverted): glued_serving_tok_s_4x2k_p139264 = 61.9k / 60.7k fresh (gate checks 1 / 3; need >= 67000;
  repeated 68.9k / 67.3k). Fresh table and per-segment overhead: results/PB.9X_fresh_walls.md (summary section).
  Lane header 4 x 2k through prefill_packed fresh 63.0k (PB.9F fixed lanes 59.0k fresh, 346 builds; 67.8k repeated);
  AgentX lane packing 30.1 / 34.0 / 32.6k (PB.9F 23.3 / 26.5 / 28.9k); glued AgentX 49.1 / 53.8 / 53.5k (45.1 / 51.5
  / 52.0k); 5 x 1.6k 51.9k (48.1k); 6 x 1344 47.9k; padded 4 x ~1.6k 43.2k.
- Gotcha (test fix): test_glued_walls' fresh builds metric counted the cache entries of the whole timed section, the
  profiled passes included; in a fresh process the profiler's own programs built there (2 entries, every draw's own
  count 0). It now takes PB.9F's definition (the fresh timed chunks' entries only).
- Gotcha: a gate check in a process started minutes before showed 200-280 ms host outliers on glued chunks with 0
  builds (gate check 2); the same code in a held process: none.
- Gotcha (F283, as PB.MB): test_determinism's 5120 load and the 8192 oracles cannot share a gate process; the second
  --check ran the oracles.
- Load reasons (4)-(6): the gate checks themselves (the units' loads), and test_common_values_ops / the host probe on
  the perf load (the byte checks of the common value slots need the mesh only; the host probe needs every layer).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate PB.9X --spec $BRINGUP_SPEC --check` (devjob);
  probes: tests/packed/test_glued_build_probe.py, test_onecall_host.py (PB3A_PROBE_OPNAMES=1, DS41_PACKED_ROUTE=glued |
  lanes), test_common_values_ops.py, test_glued_ids_ops.py with the walls env through `resident submit`.
- Follow-ups for 67k on 4 x 2k: the glued sparse_sdpa call's host cost (kv_lane_rows, 6 padded lane tensors: 0.45 ms a
  call vs 0.18 on the lane path), the 80 snapshot sends a chunk (one program a segment a layer), the ~13 ms fresh-vs-
  repeat gap of the glued walls (0 builds; not attributed).
## PB.8 perf (attempt 1): mhc_post v2 - feed lands in the ports' L1, W-share placement, hc transposes (MHC_POST_V2 False | True)

- Expected saving (written before code; perf_guide sections 1-4): device a chunk at 5120 (80 mhc_post calls: 40 ffn
  residual no feed, 40 attn residual with the feed): idea 1 (feed lands in the port L1: no y_output DRAM write + no
  port DRAM re-read, the tail after the norm) ~-5 to -8 us an attn call = -0.2 to -0.3 ms; idea 2 (the W block's
  sharing cores placed to cut the forward's NoC congestion) + the root's hc transposes off the BRISC (~-6 to -10 us)
  ~-8 to -15 us a call x 80 = -0.6 to -1.2 ms. Total -0.8 to -1.5 ms device a chunk; host ~0 (same launches, one
  program, C++ descriptor op, OV-23). Idea 2's "fused SFPU mix" is already ours (mix_vector: each operand loaded once a
  vector, 4 accumulators in registers since P.20); what is left there is the per-column coefficient re-unpack.
- Needs a fresh device open (scripts/run_safe_pytest.sh, not the resident runner) for the zone probe only: the device
  profiler must be on at the open and its CSV completes at the close (P.83). Op-level, no model load; correctness
  runs go through the resident runner (tests/test_mhc_post_fast.py).
- Baseline zones (op probe, fast, [160, 5120], slowest chip): ffn 110.5 us, its end = the root's hc path (wr_fin
  cols_to_soa 6.4, Sinkhorn ~5.5, wr_hc soa_to_cols 5.4: ~8 us past the units' last y write at ~102); attn 142.3 us,
  the feed tail 42 us past the units (fd_read: port DRAM re-reads of the staged y 24-34 us, then the sends); the units'
  columns land up to 61 us, the slowest partial ~85 us. [256, 5120] (packed 8192): 163 / 249 us, two units a core (no
  W share there).
- Change (fork option `v2`, CHANGELOG "v2"; model: tt/settings.py MHC_POST_V2, tt/mhc.py collapse_kwargs passes
  v2 = MHC_POST_FAST and MHC_POST_V2): (1) feed landing in the ports' L1 (idea 1): cb_feed = one slot a packet of the
  busiest port + a flag page; workers write each other-chip y tile at its stream position, then a flag word a unit;
  the port pushes packets as their units' flags set (streams as published); (2) idea 2 adapted to OUR op: a W block's
  5 cores on one grid column segment, each core's W share by ONE NoC multicast (his interleaved strips are for his
  row-group combine tree; ours has no tree, so the segment + multicast is the same idea: short, shared W traffic);
  his "fused SFPU mix" is already ours (P.20 mix_vector: each operand loaded once, 4 accumulators in registers), so
  nothing to take; (3) the pending hc transposes: on the root's SFPU (exact lane moves), not the BRISC.
- DECIDED (OV-58): the brief's switch must be listed in bringup/hooks.py batch_call_settings, which this step may not
  edit: MHC_POST_V2 is read per call in tt/mhc.py (one load could serve both), but until hooks.py lists it the batch
  runner treats it as load-time (a run with another value reloads the model). One-line follow-up for the overseer:
  add "MHC_POST_V2" to batch_call_settings' tuple.
- Measured (op probe, 8x4, device profiler, slowest chip, median of 4-6 calls; fast -> v2, us):
  [160, 5120] ffn 111.1 -> 102.6 (-8.5: the hc path; the units' phase unchanged), attn 143-150 -> 138.4 (fast varies
  between probe opens; -5 to -12); [256, 5120] ffn 161.8 -> 158.2, attn 249.6 -> 240.6. Expected a 5120 chunk: 40 x
  -8.5 + 40 x ~-6 = ~-0.6 ms device (below the brief's ~1 ms keep line; the owner decides from the A/B). Not moved:
  the multicast W share (units' column reads + mix bound the units: rd_in end max ~60, c_mix end max ~78), 8 sender
  headers (link / packet bound, DS41-355).
- Bit-identical to fast on every output and deterministic over 3 repeats: fork test 8 / 8 on the resident mesh
  (tests/test_mhc_post_v2.py -> the fork's check_v2: [64, 640] ragged +-feed, [160, 5120] +-feed, [256, 5120] +-feed,
  [640, 1280] feed, [1280, 5120] (landing too big: staging)); the probe's v2-vs-fast diffs 0 on X', stats, hc, feed.
- Gotchas: DS41-354 (kernel-config buffer under the device profiler: v2 first did not fit, fixed by sharing the mix
  code); the rotate direction of subvec_shflror1 (lane i -> i + 1) was taken from the bit-identical hc check (HCT_DIR);
  a semaphore word is 0 at every launch, so it carries the landing address safely; the flags are zeroed before the
  address is published (workers write nothing before they have it).
- Cores: unchanged (109 of 110 with the feed; the ports moved to the last column). Host: no new launches, one program.
- Re-run: correctness `python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd
  "scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_mhc_post_v2.py"`
  (MHCV2_CASES=<ids> picks cases); zones (fresh open): `MHCP_FAST=v2both MHCP_T=160,256
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/perf/test_mhc_post_probe.py -s`.
- Gate (`gate PB.8 --check`, switch on, after the last code edit): PASS: determinism 0, contract 0 / 0, outside_history
  0, every err_layer / err_state within its noise floor, err_final_hidden 0.135 (<= 0.242), top5 1.0; walls (switch on)
  steady_wall_ms p145408 84.3, p1024000 97.4, agentic 78.8, multilingual 82.8. The on / off device A/B is the
  orchestrator's (measure ran the change-on walls; no profile run by this step).
## PB.8R perf (attempt 1): re-apply PB.8's mhc_post v2 unchanged (OWNER 2026-11-20; MHC_POST_V2 False | True)

- Expected saving (written before measuring; PB.8's own numbers, perf_guide sec. 2): device ~-0.6 ms a 5120 chunk
  (40 ffn calls x ~-8.5 us + 40 attn calls x ~-6 us), ~-0.5 ms a packed 8192 chunk ([256, 5120]: -3.6 / -9 us a
  call); host ~0 (same launches, one C++ program a call; no new Python-built op).
- Applied: `git cherry-pick -n 0ef6b7107f5` onto 9b01cce1d8f; the code merged clean (mhc_post_ttnn and tt/mhc.py
  untouched since 561fc994f0f); only BREADCRUMBS.md / known_issues.md conflicted (both appended): kept both, PB.8's
  entries renumbered DS41-343/344 -> DS41-354/355 (343/344 were taken by PB.7C since), in its BREADCRUMBS section too.
  untracked.tgz's two tests are byte-identical to the commit's (already in the tree).
- PB.8's follow-up done: bringup/hooks.py batch_call_settings lists MHC_POST_V2 (call-time), and batch_call_apply
  refreshes it in the load-time snapshot like POS_DYNAMIC / PACKED_ROUTE (DS41-314: only that key, never a full
  refresh). tt/mhc.py reads it per mhc_post call; the fork keeps both programs (v2 is in the program key; a serving
  process uses one value, so no build after warm-up).
- Per-chunk values touched: none (no position, slot or layout enters the change; same op arguments per chunk).
- Rebuilt: `cmake --build build_Release --target install -j 48`.
- Fork test (bit-identical to fast, 3 repeats each, through the resident runner; mesh only, no model load): 8 / 8 ok
  (small_ragged +-feed, ds41_8x4 +-feed, packed8k +-feed, 640x1280_feed, 2x2). No redesign, no re-tune.
- `orchestrator measure --task PB.8R` (switch on): the 4 perf units rc 0 in one process.
- Gate `--check` (switch on): every accuracy unit PASS (err_final_hidden 0.135 <= 0.242, top5 1.0, determinism 0,
  contract 0 / 0, outside_history 0); walls steady p145408 84.4, p1024000 99.4, agentic 77.8, multilingual 84.5 ms.
  test_no_rebuild was refused twice by the one-load rule (DS41-356, OV-61): ENGRAM_TABLE (it read F222's mmap, the
  other perf units pin the shipped resident) -> tests/test_no_rebuild.py now pins it the same way (config only, no
  threshold); then seq (a skip_positions run moved to a fresh process held 1029120, no_rebuild needs 1044480; the
  run order came from the resident runner still holding my first attempt's 40-layer model) -> PASS alone with
  `gate PB.8R --check --unit PB.8R#test_no_rebuild`. All units passed on the final code.
- Re-run: fork test `python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd
  "scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_mhc_post_v2.py"`;
  gate `PYTHONPATH=$PWD python -m models.demos.common.bringup gate PB.8R --spec $BRINGUP_SPEC --check` (devjob).
## PB.9C perf (attempt 1): the performance LADDER per path at real fixed histories + no-rebuild PASS/FAIL per path (tests only)

- Expected saving (written before any run; perf_guide sec. 1-4): 0 ms device, 0 ms host a chunk: a measurement step,
  no model change (tests/, the report). What it adds is attribution: the serving entry's +5-7 ms a chunk over the
  harness wall (PB.MB follow-up) split item by item, and the D2H ack's cost the walls never paid.
- Load reasons (one-line checks, owner's iteration rule): (1) tests/perf/test_d2h_ack_probe.py, mesh only on the
  resident runner's held perf load (0 opens, 0 loads): "the D2H ack service + LayerAckService on this mesh, with the
  model's sub-device manager loaded / cleared between acks: the smallest piece that shows a hang or a pin failure";
  (2) the gate's units on its one executed-layers load (layers 0-20, BRINGUP_CHUNK=8192): "every ladder number is a
  whole-chunk number through every executed layer at histories to 1M; no smaller piece is the ladder".
- What (tests only, no model change):
  - tests/packed/ladder_common.py (new): positions / cases (LADDER_POS, LADDER_CASES; LADDER_SERVING=0 drops the
    serving column), time_kind (testing/profile.warm_up + WALL_RUNS median, misses refused), packed_ladder (steady /
    resumed at h, every segment its own serving slot of the adapter's KV cache, the real previous chunk first, segment
    order rotated every run, one profiled run: device ms, phases, traced gathers), D2HAcks (tt-d-gen's D2H ack:
    metadata-only D2HStreamService + LayerAckService + a layer-completion ring popped every chunk; record_count 1 so
    any ack count decodes), HostSplit (per-call host ms of wrapped callables, nested calls taken out, REST per chunk).
  - test_packed_walls / test_glued_walls: ladder mode (LADDER_POS in BATCH_POS_VARS, LADDER_CASES / LADDER_SERVING in
    BATCH_ITEM_VARS); paths lanes4x2k (PACKED_ROUTE=lanes, PACKED_GLUED=off), glued4x2k, glued5x1600; host split of
    glued 4 x 2k at 139264 (harness / serving / serving_noack / *_freshpos).
  - test_skip_positions: the serving column per point (runtime.prefill_chunk + D2H ack, slot 0 of a 1 + 7 slot KV
    cache over the same load) and the single host split at 139264 (+ fresh slot / position sides); the load's max_seq
    is the walls' capacity when the column is on (F283 / DS41-356: the units share one load in any order).
  - test_no_rebuild: at 8192 + PACKED=on the packed paths through prefill_packed at fresh slots (pool 11..19) and
    histories: no_rebuild_pass_<path>, program_builds_after_warmup_<path> (single8192, glued4x2k, glued5x1600,
    lanes4x2k; lanes not asserted, its error recorded).
  - tests/perf/test_d2h_ack_probe.py: mesh-only ack probe (315 acks, sub-device manager loaded / cleared between bursts).
- DECIDED (OV-58): `orchestrator measure --task PB.9C` not run: it is the quick A/B profile of a model switch; this step
  has no switch and no model change, and the gate's units already measured everything on the final test code.
- DECIDED (OV-58): the ladder tables are metrics (ladder_<path>_table), not results/<task>_ladder_*.md files (outside
  this step's paths); the owner's report is results/packed_ladder_20261121.md, composed from results/PB.9C.json.
- Gate `gate PB.9C --check` (devjob pb9c_gate4, after the last test edit; hooks clean): every ladder metric > 0,
  no_rebuild_pass_single8192 1, glued5x1600 1, program_builds_after_warmup_serving 0, ids_uploads_per_chunk 0,
  determinism 0.0; FAIL no_rebuild_pass_glued4x2k = 0 (need == 1): 1 build + 1 prepare_program @
  tt/packed_glued.py:805 GluedSend.__call__ on a fresh carry-source mix (DS41-360; a model gap, not fixable in a
  tests-only step). lanes4x2k 153 builds (recorded, not gated).
- Findings (results/packed_ladder_20261121.md): DS41-357 resumed glued chunks 3.1-3.4x steady (replay segment ->
  path (b)); DS41-358 the D2H ack ~0.2 ms host each (glued 4 x 2k: 84 acks 16.7 ms, host-bound so exposed; single
  8192: 21 acks 4.3 ms, half hidden); serving entry's extra at fixed histories single +2.6..+7.5 ms (ids readback 1.8,
  un-shuffle 0.5, reorder 0.3, note_tokens 0.07, minus the harness upload 0.4, plus the exposed acks); fresh slots /
  positions add nothing measurable at 0 builds.
- Gotchas: DS41-359 (adapter._run leaves model.state on its slot; the skip_positions harness then read that ring);
  DS41-356 again (skip_positions first in the perf process loaded 1032192, the walls need 1048576: refused); a test
  file edit makes the resident runner reload the model, and three of the reloads did not get the device memory back
  (restart = device open: 7 opens this task); writing results/<task>_*.md from a test lands outside a step's paths.
- Rule 20, per-chunk values: none touched in the model. The tests feed only what tt-d-gen sends (engine ids + the
  descriptor / chunk header, made before the clock); no host table.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate PB.9C --spec $BRINGUP_SPEC --check` (devjob);
  ack probe: `resident submit --cmd "<the gate's env> scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/perf/test_d2h_ack_probe.py"`; report: compose from results/PB.9C.json.

## PB.NR (2026-11-21, perf role, TESTS ONLY): the no-rebuild check runs cold, packed paths recorded per path
- Expected saving per chunk: 0 ms device, 0 ms host (no model code changes; this step makes the build count honest).
- Load needed: the gate's own loads only (test_no_rebuild must run in a fresh process: a held process's program cache
  is exactly what made PB.9C5's 5120 PASS suspect, F297); no extra load of mine.
- Changed (tests/test_no_rebuild.py only): `BATCH_COLD = True` (F297: core/batch puts it in a fresh process, first
  after the determinism check); `NO_REBUILD_PACKED_STRICT` in BATCH_ITEM_VARS (=1 asserts the glued paths and lets a
  glued error raise, the PB.9C behaviour; default records only); the single-request check always asserts 0 builds;
  `NO_REBUILD VERDICT <path>: PASS|FAIL (n builds)` printed per path before the asserts; docstring says why cold.
- Cold results (same model code 286f4654ec5): single8192 PASS (0), glued4x2k FAIL 16 (steady 1+1 prepare DS41-360,
  resumed 14), glued5x1600 FAIL 13 (all on the resumed draw; PB.9C's PASS was a warm cache), lanes4x2k FAIL 224,
  single5120 PASS (0). New: DS41-361 (path (b) of a replay chunk is never warmed by runtime.compile and slices at
  the segment offset).
- Gotcha: the gate batches both test_no_rebuild units (8192 and 5120) into one cold process; the 5120 one needs another
  model (chunk / seq) and tests.one_load refuses it ("needs another model ... switches []"). It passed when run alone:
  `gate PB.NR --check --unit PB.NR#test_no_rebuild.2.0` (its own fresh process). The orchestrator's gate must run it
  apart too (one geometry a process, F283).
- Rule 20: no per-chunk value touched (tests only; no model code changed).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate PB.NR --spec $BRINGUP_SPEC --check` (devjob);
  the 5120 unit alone with `--unit PB.NR#test_no_rebuild.2.0`.
- `orchestrator measure --task PB.NR` (devjob pbnr_measure) returned rc 4: "paused before A/B on: one process for
  quick; `resume` continues the task". Left for the orchestrator (no model code changed, so the A/B has nothing to compare).

## PB.9G perf (attempt 1): chip-aligned lanes are the one packed path (PACKED_CHIPS on | off)

- Read first: OV-3 / OV-58 / OV-61 / OV-67, PACKED_RULES.md (CHIP-ALIGNED LANES, NO SLOW FALLBACK), rule 20, perf_guide.
- Expected saving per chunk (written before any device run, perf_guide s1-4): 4 x 2k steady at 139264 (PB.9C: glued
  129.4 ms, fixed lanes 114.8): device ~0 to -3 ms (one pass a chip, no ChipRows phase / keep copies, the ids reorder
  without its SP all_gather), host -2 to -8 ms (identity piece tables skipped, no per-row pieces; the sdpa call's host
  cost is the big lever: PB.9X measured 0.45 vs 0.18 ms a call x ~18). Resumed 4 x 2k (PB.9C glued 419 ms through path
  (b)): -270 ms (replay segments in the one call: ~steady + the replay's writes, ~130-145 ms). 4 x 1.6k + split: one
  chunk of 7424 new tokens on 32 chips (glued 5 x 1600 was 137.5 ms for 8000 tokens): ~125-135 ms. Builds: steady 0
  (DS41-360 fixed: one GluedSend program for every carry mix), resumed 0 (DS41-361: no path (b) left).
- DECIDED (OV-58): the chip-aligned path is the one-call path of tt/packed_glued.py (GluedCall, PB.7C's device-derived
  tables) on a chip-aligned GluedChunk (align = rows a chip): on whole-chip segments every per-row quantity of that
  path IS a per-chip one (one segment a chip from its row 0: one pass, identity rolls, the ring's ranges one a chip),
  i.e. packed_onecall's per-chip design with glued_derive's device tables; generalising packed_onecall's fixed-lane
  class instead would re-do PB.7C's derivation for it (its per-tuple host constants and slices are the 224 builds).
- Load reason (1): tests/packed/test_glued_derive_ops.py + test_glued_ids_ops.py on the resident runner's held perf
  load (mesh only, 0 loads): "the derive and ids kernels' chip-aligned offsets / membership / replay shift vs the host
  functions: op level, no layer runs".
- Load reason (2): test_packed_oracle at DS41_PACKED_LAYERS=0,1,2,3,14,20,21,24 on the 40-layer 8192 accuracy load:
  "bit-identity of chip-aligned chunks (replay, 1-chip, 1-token, split, empty chips) needs the device path of every
  layer role; the characteristic layers cover them".
- What (PACKED_CHIPS=on default; off = PB.9X's back-to-back glued layout through the same one call; call-time,
  hooks.batch_call_settings, refreshed in batch_call_apply like POS_DYNAMIC; PACKED_ROUTE is gone with the lanes):
  - tt/packed.GluedChunk(align): a segment's rows (engine = model) [off_k, off_k + ext_k), ext_k = its length rounded
    up to whole chips (>= 1 chip; a 1-token / idle segment owns one chip), off_k = the extents before it; fill(align)
    splits a request on a kv block inside the chips left; aligned() re-lays a chunk.
  - tt/packed_device: as_run_desc(desc, mesh) makes every packed header a chip-aligned GluedChunk (CHIP_ROWS set by
    the model); relay_ids / seg_index re-lay ids and lanes; PackedChunk raises on a chunk the one call cannot run (no
    path (b): GluedMover, _glued_attention gone; no fixed lanes: LaneMover gone); STATS counts onecall / trivial /
    fallback (0) chunks; warm_glued_chunks enumerates chip-aligned kinds (1..6 segments, fresh / continued / replay /
    far, 1-token + idle).
  - tt/glued_derive (ALIGN_ROWS) and page_copy.GluedIdsReorder (ALIGN): offsets from the header's lengths; keep, the
    compressed keep and the MoE end words by segment membership (a segment's last chip past cap masked as today's pad
    rows); srv rows of a replay segment: write_start's server order, its rows from write_start - start on. The ids
    reorder reads its own mesh row only (no SP all_gather) when aligned.
  - tt/packed_glued: replay segments in the one call (_compress writes [gws, cap) through the shifted index rows;
    glued_call_ok refuses a replay only without DECODER_SKIP=decode); GluedGeo rope pieces cover pad rows; sparse_sdpa
    on chip-aligned chunks by kv_lanes with the map a runtime value (sdpa fork kv_lane_map_runtime, below);
    GluedSend orders its carry tensors by accessor kind with an anchor snapshot (DS41-360: one program for every mix).
  - tt/model._begin_engram_glued: zero jobs for a segment's last-chip pad rows and idle chips.
  - tt/packed_onecall: the fixed-lane code removed (Lanes, LaneCopy, LaneSend, the lane kernels, OneCall's lane
    methods); OneCall is GluedCall's base (attention, compress_only, _query, _keep, _phase) and raises if built.
  - Fork sdpa: kv_lane_map_runtime (default False; CHANGELOG / README / INDEX). Cores: no new grid (data movement and
    the derive programs keep their rectangles; no matmul changed). Build: ./build_metal.sh 45 s, clean.
  - Tests: test_packed_oracle phase B (DECODER_SKIP=decode: replay R / R1 (1 chip), S1 (1 chip), T (1 token), X1 / X2
    (a split request), 5-segment chunks, empty chips; alone-vs-today digest keys printed); test_packed_walls ladder
    cases 4x2k / 4x1600s (paths lanes4x2k, lanes4x1600s); ladder_common CASES["4x1600s"]; test_no_rebuild paths
    lanes4x2k / lanes4x1600s / lanes6mix (strict: all asserted; packed_fallback_chunks_<path>); test_glued_derive_ops
    and test_glued_ids_ops on chip-aligned layouts (replay, 1-chip, idle, split); test_onecall_host on the walls'
    load; removed test_onecall_ops / test_onecall_sizes / test_glued_probe (the removed code's tests).
- Op checks (resident, mesh only on the perf load): test_glued_derive_ops 47 layouts x 2 header sources, 0
  mismatches (18 chip-aligned incl. replay srv rows). Oracle rung (characteristic layers, accuracy load): alone vs
  today 0.0, lane equivalence 0.0, slot digests 0, determinism 0.0 (phase B included).
- Ladder rung (139264 only, harness wall, LADDER_SERVING=0): lanes4x2k steady 123.3 ms (device 104.1), resumed
  122.0 (PB.9C glued 129.4 / 419.0, fixed lanes 114.8 / 143.2); lanes4x1600s steady 132.4 (device 108.0), resumed
  131.0. Host probe (test_onecall_host, 4 x 2k @ 139264): walls 119 ms, run_chunk enqueue 109.5 ms; cProfile: 1021
  ttnn op calls 54 ms own time, 80 snapshot sends 11 ms cum (one mesh program a segment a layer), 20 GluedSend 4 ms,
  72 page copies 5 ms (DS41-364).
- Snapshot sends (PB.9G, part of PACKED_CHIPS=on): every segment's served snapshot of a layer (and every idle
  segment's) in ONE prepared program (tt/snapshot_send.SnapshotSend.multi: the dynamic kernels looped over up to 6
  segments, one arrival counter a (segment, block) (24 global semaphores), one relay counter a port core (program
  semaphore 0, incremented in block order), bounded waits; per segment its ring / slab addresses, page0, ring rows and
  block codes as values; one program a segment count, built by the warm-up's counts): 20 launches a 4-segment chunk
  instead of 80 (the cProfile's 11 ms). write_snapshot(batch=) collects, TtDS41Model.layer_packed sends.
- Gotcha: a `str.replace(old, new)` in my edit script hit a second, identical line (page_copy.IdsReorder got
  GluedIdsReorder's `inp if self.align` line): the single-request no-rebuild crashed in the first gate check
  ('IdsReorder' has no 'align'). Fixed; edit scripts now count matches.
- Rule 20, per-chunk values this change touches and how they reach the device: segment starts / ends / write_starts /
  slots -> the chunk header (tt-d-gen's fields, one upload with the ids) -> derived on the device: chip-aligned
  offsets (glued_derive ALIGN_ROWS, GluedIdsReorder ALIGN), keep / compressed keep / MoE end words (membership), the
  window index lists (+ the replay class-0 mask: a per-run value), the replay write rows (srv: write_start's server
  order + write_start - start). Per-call runtime values (no program key, no upload): the ring's lane ranges, the sdpa
  lane map (kv_lane_map_runtime), GluedSend's carries / pieces (its program key no longer follows the carry mix), the
  snapshot programs' per-segment addresses / rows / codes, GluedRows' rope pieces. The host keeps only what decides the
  op sequence (segment count, which segments write). No host table.
- After SnapshotSend.multi (rung, 139264, harness wall): lanes4x2k steady 121.4 ms (was 123.3; device 104.4),
  resumed 121.3; lanes4x1600s steady 127.4 (was 132.4; device 107.2), resumed 127.8. Oracle rung (characteristic
  layers) with multi: alone vs today 0.0, lane equivalence 0.0, slot digests 0, determinism 0.0.
- First gate check (pb9g_gate1, before multi): determinism 0.0; oracle on all 40 layers: alone vs today 0.0, lane
  equivalence 0.0, slot digests 0, determinism 0.0, packed_fallback_chunks 0; test_no_rebuild crashed on the
  IdsReorder slip above (fixed), so its batch's walls did not run. alone-vs-today served digests: 40 a phase-B
  segment, all decoder layers' (20..39) window / snapshot records (DS41-366; not served under decode).
- Gate check 2 (pb9g_gate2, code with SnapshotSend.multi): every unit ok but ladder_lanes4x2k_steady_wall_ms_p139264
  = 123.4 (need <= 117: FAIL); resumed 121.2 (<= 150 ok); no-rebuild cold strict PASS on every path (0 builds);
  ids uploads 0; packed_fallback_chunks 0; tables in results/PB.9G_ladder.md.
- Indexer at chip-aligned lanes (after gate check 2): the one ring call's lane_ranges take one whole-chip range a chip
  (pad rows past a segment's rows computed at its positions, never kept), so the ring runs the model's q chunk (64)
  instead of the glued path's 32 (_ranges / program_config in GluedCall._index_dynamic): at 1M the 4 x 2k chunk is
  device-bound with the indexer at 51 ms (fixed lanes ~42). Oracle rung (characteristic layers): bit-identical.
- Switch for the A/B: PACKED_CHIPS (DS41_PACKED_CHIPS) on = the change (chip-aligned lanes + per-layer snapshot
  program + whole-chip indexer ranges + kv_lane_map_runtime), off = PB.9X's back-to-back glued layout through the
  same one call (replay segments included: path (b) no longer exists, so off cannot be PB.9X's resumed path).
- Device opens this task: ~10 (resident restarts: C++ build, model configuration changes between the accuracy load
  and the executed-layers load, cold no-rebuild processes).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate PB.9G --spec $BRINGUP_SPEC --check` (devjob);
  rungs: `resident submit --cmd "DS41_PACKED_LAYERS=0,1,2,3,14,20,21,24 <oracle env> scripts/run_safe_pytest.sh
  --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/packed/test_packed_oracle.py"`, the walls with
  `LADDER_POS=139264 LADDER_CASES=4x2k,4x1600s LADDER_SERVING=0`; op checks test_glued_derive_ops / test_glued_ids_ops;
  host probe test_onecall_host (walls env).
- REVERTED (measured worse): the whole-chip indexer ranges at the model's q chunk 64. Gate check 3 (pb9g_gate3, that
  code): indexer at 1M 58.8 ms (4 x 2k; q 32: 51.4), 69.7 (4 x 1.6k; 60.0); 4 x 2k @ 1M steady 184.6 ms (180.3),
  4 x 1.6k 199.3 (193.8); 139264 unchanged within noise (123.1 / 132.4). Back to the glued ring's q chunk 32 (the
  shipped code is gate check 2's: no-rebuild cold PASS everywhere, oracle / determinism 0, steady 4 x 2k @ 139264
  123.4 ms FAIL vs 117).
- `orchestrator measure --task PB.9G` (devjob pb9g_measure, final code): rc 0, quick side on: min layer PCC 0.732
  (L39, as earlier tasks' 0.7318), host_transfers_per_layer 0, chunk wall 221.9 ms (the single-request quick chunk;
  PACKED_CHIPS does not touch it).
- Report: results/PB.9G_ladder.md (gate check 2's ladder of the shipped code next to PB.9C; the no-rebuild and
  accuracy verdicts; for the owner). A gate check deletes results/<task>_*.md written before it: write it after.

## PB.MB4 perf (attempt 1): model_bringup's MB.4 (one fused history pack, ttnn.bringup.ckv_bfp4_rows) on the batching branch
- Read first: OV-3, OV-58, OV-61, PACKED_RULES, rule 20, perf_guide; MB.4 = 32d12f46bda (model_bringup).
- Applied: `git show 32d12f46bda -- <tt, tests, ttnn/ttnn/bringup, BREADCRUMBS, known_issues> | git apply --3way`.
  Conflicts: prepared_program (C++ + README + CHANGELOG): kept THIS branch's (PB.9X already has common value slots,
  `(k, 1 << 31 | i)`, the same encoding MB.4's ckv_bfp4_rows passes; patch_count counts them): no C++ change, no
  rebuild. history_gather._pack: `dyn("hist")` (POS_DYNAMIC or PACKED_GLUED_DYNAMIC) -> `_pack_rows` (ONE
  ckv_bfp4_rows a pack); MB.2 / PB.9X's `_pack_fixed` (2048-row pieces: alloc + 4-core page copy + tilize + ckv_bfp4
  compact dynamic_row0 each) and `_dyn_row0` removed (one implementation, no dead path). Every pack goes through
  `_pack`: single-request prefetch / tail, packed prefetch_prepare (gather_hide items) and tail_prepare -> _tail_part.
  ckv_bfp4's dynamic_row0 option stays in the fork (its own test, test_common_values_ops); COMMON_VALUE_SLOTS still
  drives index_keys / lane_rows. INDEX / ckv_bfp4 CHANGELOG: both entries kept. MB.4's known issues 324-327 are
  this file's DS41-367..370 (model_bringup numbered from 321 after the branches split).
- Switch: DECIDED (OV-58): no new A/B setting. The task says one implementation, no dead path; the old pieces are
  the previous commit (fd76e0b40cd) and the off side of POS_DYNAMIC / PACKED_GLUED_DYNAMIC (slice + tilize +
  ckv_bfp4) is unchanged.
- Expected saving per chunk (before any device run; MB.4 measured on model_bringup -2.5 ms device / -6.3 ms host at
  1M single 5120 with per-core-arg pieces; here PB.9X's compact dynamic_row0 pieces are cheaper on the host):
  single 8192: 139264 device -0.3, host -0.5 ms; 524288 device -1.0, host -1.5; 1024000 device -2.0, host -2.5 ms.
  4 x 2k (4 segments' histories: ~4x the pieces): 139264 device -0.5..-1, host -1..-2; 1024000 device -3..-6 (part
  of the packs sit beside the gathers, PACKS_A_REGION), host -5..-8 ms. Bit-identical (records are per row).
- Load reason: "needs the gate's executed-layers load (layers 0-20, decode, 8192, resident engram) because the
  pack's cost exists only at 139k-1M histories through every source layer, single and packed; the gate's 5 units in
  ONE load answer accuracy, no-rebuild and the ladder together". Op check first on the held mesh (test_ckv_bfp4_rows).
- Gotcha (DS41-371): the first gate stopped at the load ("MoEBlockConfig has no attribute shared_scatter_columns"):
  the C++ build was model_bringup's (MB.4 built it at 11:23). ./build_metal.sh (rc 0) and the gate ran.
- Op test (held mesh, before the load; DS41-316): tests/test_ckv_bfp4_rows.py 3 passed on this branch's build.
- Gate `gate PB.MB4 --check` (devjob pbmb4gate2, 977 s): PASS, every unit. Oracles 0.0 (alone vs today, lane equiv,
  slot digests, determinism; 40-layer determinism 0.0); cold strict no-rebuild PASS (test_no_rebuild first after the
  load: 0 builds / prepares / ids uploads; single8192, lanes4x2k, lanes4x1600s, lanes6mix).
- Ladder (results/PB.MB4_ladder.md). Single 8192 vs PB.9C, steady wall ms: 139264 100.6 -> 100.0, 524288 111.0 ->
  107.5, 1024000 131.7 -> 127.4 (device -0.8 / -1.2 / -2.8); resumed 104.8 / 116.2 / 135.2 -> 102.8 / 111.0 / 132.1.
  Chip-aligned 4 x 2k vs PB.9G: 139264 120.8 -> 116.8 (device 104.8 -> 101.6), 1024000 178.0 -> 161.2 (device
  157.1 -> 146.7, exposed host 20.9 -> 14.5; attention section -8.7 ms: the packs run as compute steps of the
  gather-hide regions); resumed 120.8 / 178.5 -> 115.7 / 160.2; 524288 (no PB.9G point) 134.9 / 134.7.
  vs the estimate: single at 1M device -2.8 (est. -2.0), 4 x 2k at 1M device -10.4 (est. -3..-6), host -6.4 (est.
  -5..-8).
- Cores (rule 18): ckv_bfp4_rows on max_grid (110 a Galaxy chip; no matmul). Removed: the 4-core page copies.
- Rule 20, per-chunk values this change touches: the pack's source row (slab.batch R + a: slot, layer, start), its
  row count (ceil32(history) - a) and the record row a, for every segment's prefetch and tail: common-arg value slots
  of ONE prepared program per (slab geometry, record-buffer geometry, max_grid), written by run_prepared each call;
  nothing from the host but those three words a call; no program keyed on a position, slot or length (cold
  no-rebuild PASS).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate PB.MB4 --spec $BRINGUP_SPEC --check` (devjob);
  the op test: `resident submit --cmd "scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_ckv_bfp4_rows.py"`; after a branch switch: ./build_metal.sh first.

### Imported with MB.4 (model_bringup's BREADCRUMBS section, ids renumbered: its DS41-324..327 = DS41-367..370 here)
#### (model_bringup) MB.4 perf (attempt 1): the long-history slowdown (POS_DYNAMIC off | on), explained and fixed at the cause
- Read first: OV-57, rule 20, perf_guide, MB.3's ladder (DS41-367). Load reason: "needs the gate's ladder load (layers
  0-20, decode, 5120, resident engram table, position tables to 1,044,480) because the gap only exists at 524k-1M
  histories through every executed layer; one probe (tests/perf/test_mb4_history_probe.py) measures on and off in that
  one load at 4 points".
- Measurement (probe, before any code change; harness steady chunk, its real previous chunk first; on / off ms):
  51200 74.15 / 74.52, 524288 83.25 / 81.66, 786432 89.40 / 84.79, 1024000 92.59 / 86.81 (11-19: 74.1 / 79.5 / 84.9
  / 87.4): POS_DYNAMIC=off IS 11-19 (the harness adds nothing); on vs off output 0.0 at every point. The whole gap is
  history_gather.prefetch (the history pack at the chunk's first engram wait, 3 sources): at 1024000 device 6.08 vs
  4.13 ms (page_copy 4 cores a chip 1.99 ms vs slice + concat 0.51; ckv_bfp4 in 2048-row pieces 0.67 vs 0.28; tilize
  0.33 vs 0.23) and HOST 6.85 vs 1.40 ms (_pack_fixed: per 2048-row piece a tensor alloc + page copy + tilize +
  ckv_bfp4, ~4 pieces a source at 1M). Both grow with the history (pieces = history / 2048).
- Expected saving per chunk (written before the code change): one fork call per pack (ttnn.bringup.ckv_bfp4_rows:
  the slab's ROW_MAJOR rows read, tilized and packed in one program; source row, destination row and band count are
  value slots): at 1024000 device -2.0 to -2.4 ms vs on (prefetch ~3.7-4.0 ms: the gather stays), host -5 to -5.5 ms
  vs on (one dispatch a source instead of ~12); wall ~86-87 ms (<= off). At 51200 ~0.
- What (POS_DYNAMIC=on; off unchanged = the slices): tt/history_gather._pack_rows replaces MB.2's _pack_fixed: ONE
  ttnn.bringup.ckv_bfp4_rows(slab, loc, slab.batch R + a, ceil32(hi) - a, a) per pack (the prefetch's whole history
  and the tail's chunk periods), straight from the served slab: no page copy, no tilize op, no PACK_BUCKET pieces.
  Fork ckv_bfp4_ttnn: new entry ckv_bfp4_rows (reader kernels/ckv_bfp4_rows_reader.cpp; compute / writer behind
  CKV_ROWS_IN: tilize_block into cb 18 on the core, then the existing round + record pack). Fork prepared_program:
  value slots on COMMON runtime args ((k, 0x80000000 | i)); ckv_bfp4_rows passes its 3 values (src row, band count,
  record row) as common args: per-core values cost 0.31 ms host a call on 110 cores x 32 chips (DS41-369), common
  0.06 ms (tests/perf/test_ckv_rows_host.py). Built with ./build_metal.sh (rc 0).
- Gotcha (DS41-370): hw_startup(cb_in, cb_rows) leaves the packer at bfloat4_b; tilize_init does not reconfigure it
  (records garbage), and pack_reconfig_data_format AFTER tilize_init broke rows 16-31: reconfigure BEFORE the init.
  BH's DEST remap is switched off around the tilize (the row pack needs it on; untested whether the tilize does).
- Op test (tests/test_ckv_bfp4_rows.py -> the fork's tests/unit/test_ckv_bfp4_rows.py, resident runner, held mesh):
  4 passed: byte-exact vs the host rule and vs ckv_bfp4(to_layout(slice)) on interleaved and ND-sharded sources,
  several bands a core, one prepared program for 4 value sets, deterministic, refusals. The fork's single-device
  unit suites (test_ckv_bfp4.py, test_bfp_round.py, prepared_program's 2x2 cases) were not run: rule 19 (no
  single-card job on the Galaxy); the default paths' code is unchanged (new define / new slot kind only).
- Cores (rule 18): ckv_bfp4_rows on max_grid (110 a Galaxy chip; no matmul). Removed: MB.2's 4-core page copies of
  the history pack.
- Rule 20, per-chunk values this change touches: the pack's source row (slab.batch x R + a: slot, layer, start), its
  row count (the history length: ceil32(hi) - a) and the record row a: prepared value slots (common args), the same
  program for every chunk, slot and history length (one prepare per slab / record-buffer geometry: the served slab's
  and a state's own slab). Nothing new from the host per chunk.
- Measured after (probe, on / off ms): 51200 74.95 / 74.10, 524288 79.77 / 79.31, 786432 83.90 / 84.74, 1024000
  85.98 / 86.96; prefetch at 1M device 3.55 / 4.12 ms, host 0.52 / 1.43 ms (was 6.08 / 6.85 on); 0.0 apart. vs the
  estimate: device -2.5 ms vs on (estimate -2.0 to -2.4), host -6.3 ms (estimate -5 to -5.5).
- Gate `gate MB.4 --check` (after the last code edit, hooks clean): PASS, 7 units, 891 s. Ladder steady ms: 51200 72.1,
  131072 75.4, 139264 77.2, 262144 75.7, 524288 79.7, 786432 85.9, 1024000 86.9 (<= 88.5; MB.3 92.8, 11-19 87.4); cold
  no-rebuild PASS (0 builds / prepares / ids uploads, on vs off 0.0). `orchestrator measure --task MB.4`: rc 0.
  Report: ledgers/galaxy_8x4-full/results/MB.4_slowdown.md.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate MB.4 --spec $BRINGUP_SPEC --check` (devjob); the
  probe: `resident submit --cmd "BRINGUP_CHUNK=5120 DS41_ENGRAM_TABLE=resident BRINGUP_POS_TOKENS=agentic_cc
  DS41_DECODER_SKIP=decode scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/perf/test_mb4_history_probe.py"` (MB4_POS, MB4_SIDES); the op test:
  `resident submit --cmd "scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_ckv_bfp4_rows.py"`; C++: ./build_metal.sh (prepared_program).

## PB.T perf (attempt 1): slot isolation at chip boundaries, FAST tier + DEEP sweep (tests only)
- Expected saving per chunk (perf_guide, before any run): 0 ms device, 0 ms host: no model code changes (tt/ is
  outside this step's paths); the step adds the isolation oracle on the serving entry.
- Load reason: "needs the gate's accuracy load (40 layers, 8192, seq 64512) because the tier runs in that one load
  with the determinism check (OV-61); the served mode DECODER_SKIP=decode is a call-time switch on it".
- What (tests + one framework hint; no model code):
  - tests/packed/isolation_common.py (new): the harness. Bench = the serving runtime over the held model
    (adapter.build_runtime, all 40 layers, DECODER_SKIP=decode call-time), the engine KV cache, the host ack sink,
    runtime.compile. Every chunk goes through runtime.prefill_packed under a BuildProbe. Two worlds: packed (multi-segment
    chunks, slots reused, junk warm-up slots 0-5 used) and alone (each request one segment a chunk, same boundaries,
    a never-used slot; reference aliases share one alone run). Compared bit-exact: outputs (digest of every segment's
    rows, streams + pre, at layers 0, 1, 2, 3, 14 and the last computed one, 19), served records over [0, cap(end))
    at every layer (window ring natural rows, the 128-row snapshot of all 32 chips, compressed KV / index-key bytes,
    host engram tails), acks (layer range, start, end in order; none for a slot outside the chunk), bystander slots'
    whole slabs (every layer / config / chip). Path counters: GluedCall._index_dynamic layers (one ring),
    gather_hide regions, packed_device fallback. Host reads in ONE ttnn.to_torch a tensor (ConcatMeshToTensor, its
    chip order checked against get_device_tensors once): per-chip reads made a packed chunk 15-23 s, now 5-7 s.
  - test_packed_fast_tier.py: 4 packed chunks (8 requests: steady, replay, split in 2 chunks, 1 chip, 1 token,
    acks only, 12 empty chips, two slot reuses, a parked request, a canary slot), 18 segment checks.
  - test_packed_deep_sweep.py: (1) 6 kinds x all 24 orders of 4 of them (48 chunks, a 12-slot pool rotated);
    (2) seeded fuzz (SEED 20261121): 19 chunks, 29 requests, slots reused, a long request L to history 55296,
    6 idle-slot windows + canary. 373 segment checks.
  - core/batch.py: BATCH_MAX_SEQ (F298, dev/BREADCRUMBS): the tiers' slots are 114688 (DS41-372: runtime.compile's
    packed warm-up raises below ~112k), so the gate's accuracy load is made at 114688 instead of 64512.
- DECIDED (OV-58): replay segments are built as tt-d-gen sends a resumed request: start P - 128, write_start P,
  [0, P) written. My first schedule (write_start start + 128, as test_no_rebuild draws it) replays rows nobody wrote:
  they hold the slot's own leftovers, which legitimately differ between a reused (warm-up junk) slot and a fresh one.
  That gave 18 output / 84 served mismatches that were test artifacts. Likewise served records are compared over
  [0, cap(end)) (what decode receives), not [0, ceil32(end)).
- DECIDED (OV-58): outputs captured at 6 layers, not all 20 (the 20-layer capture made the tier ~3x over budget); a
  bit-exact difference at any layer reaches layer 19, and the served records of every layer locate it.
- Measured (resident, same code as the gate check):
  - FAST tier: 18 segments, outputs 0 / max |diff| 0.0, served 0, acks 0, bystanders 0; one-ring layers
    [2, 8, 14]; hide regions 12; fallback 0. Wall 260 s, of which 117 s is the model load inside the test (the
    gate's process has it loaded already: ~145 s there). FAIL: packed_fast_program_builds_after_warmup = 6. All 6
    come from the alone A[4096,8192) (a 1-segment packed chunk at a continuing, non-aligned start:
    attention._ring_carry_pages + 5 more, first call at each site). DS41-373, a model gap (tt/ is outside this step):
    a FIX task.
  - DEEP sweep: PASSED. 373 segments, every count 0, max |diff| 0.0; hide regions 42; the same 6 builds (recorded
    there, not gated). Wall 1092 s with a 112 s load: ~16.5 min in the gate.
  - packed_fast_long_pool_layers = 0: under DECODER_SKIP=decode the candidate indexer (layer 20) runs only its KV head,
    so the long-pool path is not on the served path at all. Recorded only.
- Rule 20: no per-chunk value touched (tests only). The tests feed the serving entry only what tt-d-gen sends
  (make_packed_input: the ids + the chunk header, made outside the build count).
- Gotchas: warm_glued_chunks raises on slots below ~112k (DS41-372); a test-file edit restarts the resident process
  (the released model's DRAM does not come back): one load an iteration.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate PB.T --spec $BRINGUP_SPEC --check` (devjob); one
  tier: `resident submit --cmd "<the gate's env> scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/packed/test_packed_fast_tier.py"` (or test_packed_deep_sweep.py).
- Gate check `gate PB.T --check` (devjob pbt_gate1, after the last code edit; hooks clean): FAIL on
  packed_fast_program_builds_after_warmup = 6 (DS41-373) only. Every isolation metric is 0; one_ring_layers 3; hide
  regions 12; packed_fast_tier_s 104.0 (the model already loaded by the determinism check; compile 9.6 s warm);
  test_no_rebuild cold strict ok (program_builds_after_warmup_serving 0, ids_uploads_per_chunk 0); determinism 0.0.
  packed_deep_* MISSING: the batch stopped at the fast tier's failure. The resident run of the same code PASSED
  (above); it was not repeated through the gate (~20 device-min for an answer already in hand).
- DECIDED (OV-58): `orchestrator measure --task PB.T` not run. It is the quick A/B of a model switch, and this step
  has neither a switch nor a model change (as PB.9C / PB.NR).
- Reports: results/PB.T_fast_tier.md, results/PB.T_deep_sweep.md (written after the gate check).
- FIX task for the owner / orchestrator: DS41-373 (warm a 1-segment continued packed kind at a 64-aligned start in
  warm_glued_chunks; consider a 1-segment path in test_no_rebuild). DS41-372 (warm_glued_chunks' far kinds raise
  below ~112k positions).

## PB.FX perf (attempt 1): no program build after warm-up for a 1-segment packed chunk at a continuing start (PACKED_WARM_CONT1 on | off)
- Expected saving per chunk (perf_guide, before any run): steady state 0 ms device, 0 ms host (warm-up only, the forward
  is unchanged); on the request path it removes 6 program builds (~0.7-1 ms host each for the Python-built ones, more
  for the kernel compiles: seconds the first time) from the first lone-segment continuing chunk after startup.
- Load reason: "needs the gate's two loads only (the 40-layer accuracy load for oracle + fast tier, the cold layers-0-20
  load for test_no_rebuild) because a build count is only honest in a fresh process after runtime.compile (F297)".
- Probe (PB.T's fast-tier log, the BuildProbe there; no extra load): the 6 builds are all "first call at this site", no
  attribute changed: to_layout @ attention.py:1138 `_ring_carry_pages`, experimental.high_bw_all_gather (`_gather_all`),
  page_copy's prepare_program + run_prepared (TILE pages gat -> mine), untilize (`to_rm`), bringup.slice_write @
  attention.py:1901 `_put_rows` (the carry into the kept tail `rtail`).
- Cause (DS41-374; DS41-373's "non-aligned start" was not it): a lone segment is `PackedChunk.trivial`
  (packed_device.py:212) and runs today's single path. After a PARTIAL (non-whole) chunk of its slot `rtail_end !=
  start`, so `_window_region` -> `cache.carry(start)` -> ring -> `_ring_carry_pages` with owned blocks (the owner
  gather). runtime.compile's single chunks always continue a whole chunk (rtail) and its packed lone segments start fresh
  or past an unwritten history (truncated carry), so nothing warmed this kind. In serving it is reachable: a request
  split over packed chunks then alone in a chunk.
- Fix (warm-up, rule 20 "cover the kind in the finite warm-up"): every program there is position-free (fixed 128-row
  carry, page-copy offsets / counts are value slots, the owner chip only an offset), so warm_glued_chunks adds ONE pair
  in slot 0: [0, c/2) then [c/2, c) (c/2 chip-aligned; PACKED_WARM_CONT1 on | off, call-time, in batch_call_settings).
  Also DS41-372: the fits-check runs before Segment() (no raise below ~112k positions; those kinds are still skipped).
- tests/test_no_rebuild.py: packed paths one4k (4096,), one1600 (1600,) (the layout, whole-to-c/2 and partial end) and
  lanes2x3k (3072, 3072) (the sibling 2-segment kind at continuing non-aligned starts); each steady / resumed / steady,
  previous chunk partial in the same slot, cold, strict.
- Per-chunk values touched (rule 20): none on the forward. The carry's start / ring slot / owner chip already reach the
  device as page-copy value slots (page_copy.PageCopy run_prepared values); the change only adds two startup chunks.
- Gate check `gate PB.FX --check` (devjob pbfx_gate1, after the last code edit, hooks clean): PASS PB.FX. test_no_rebuild
  cold strict: every path 0 builds (lanes4x2k, lanes4x1600s, lanes6mix, one4k, one1600, lanes2x3k, single8192),
  ids_uploads_per_chunk 0, on/off 0.0; fast tier: packed_fast_program_builds_after_warmup 0 (was 6), every isolation
  metric 0, one-ring layers 3, hide regions 12, packed_fast_tier_s 102.6; oracle: packed_alone_vs_today_max_abs_diff 0.0,
  digest mismatches 0; determinism 0.0.
- `orchestrator measure --task PB.FX` (devjob pbfx_measure): rc 0, min layer PCC 0.7321 (L39; = PB.MB4 / PB.9G),
  chunk_wall_ms 219.8 (spread 25.3; PB.9G 221.9: noise, as expected), host_transfers_per_layer 0.
- Not run: test_no_rebuild with PACKED_WARM_CONT1=off (a second cold load) to show the new paths fail without the fix;
  the orchestrator's off side of the A/B runs it.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate PB.FX --spec $BRINGUP_SPEC --check` (devjob).

## PB.HG perf (attempt 1): a segment's history only to the chips it owns (PACKED_HIST_OWNED on | off)

- Read first: OV-3 / OV-58 / OV-61, PACKED_RULES.md (CHIP-ALIGNED LANES, SERVING NEVER REBUILDS, NO HOST TABLES), rule 20,
  perf_guide.
- Today (PB.9G / PB.MB4): every (KV source, segment) history is all-gathered over the whole mesh (two Hamiltonian
  cycles) into the segment's own lane buffer on every chip: 4 x 2k = 4 histories into every chip (124 pieces of a chip's
  rows received a chip a source), 4 lane buffers a source (owners x R x 320 B each).
- Design (DECIDED, OV-58): tt/owned_gather.py, ONE Python-built prepared program a call: every chip sends its packed
  records of every segment of the source straight to that segment's chips (fabric unicast, routed by the fabric's own
  tables: each port core decodes the root direction of the route to every destination in the kernel, so no host routing
  replica), into the same rows of the source's ONE history buffer (owner q at rows q R + [lo, hi)); each chip receives
  exactly its own segment's 31 pieces + its own local copy. Destination map (chip -> segment, 4 bits a chip), the row
  ranges and the record buffers' addresses are common value slots (rule 20: one program a geometry). Fence: every
  destination signals "started" to every source before data moves (no write into a buffer a chip still reads); every
  wait bounded. Idle chips (no segment) get segment 0's history: the sdpa's lane map reads lane 0 there today
  (bit-identical pad rows). Tails go into place (no tail buffer, no page copies, no slice_write).
- Expected saving per chunk (before any device run; perf_guide s1-4): DRAM written by the gathers a chip 4x less at 4
  segments (31 vs 124 pieces); link traffic: unicast all-pairs ~24 piece-hops a directed link (8x4 torus,
  dimension-order) vs ~31 today at 4 segments (a constant now: independent of the segment count; worse than today at 1-2
  segments, which never take this path at 8192 except 2 x 4k); the gathers are mostly hidden already (exposed
  history_gather 2.7 ms at 139264, 5.3 ms at 1M steady, PB.MB4). Device: -0.3..-1 ms at 139264 (the tails' slice_writes
  and copies gone, less DRAM contention beside sdpa / indexer), -1..-3 ms at 1M (shorter gathers on the column, fewer
  pieces overrunning). Host: -0.5..-1 ms (no tails_input / tails_finish page copies; one program a source region
  instead of one a segment). History DRAM a chip: 4 lane buffers -> 1 a source (the lanes' buffers are never acquired).
- Next lever if the unicast's link load shows: 2D multicast rectangles (fabric_set_2d_mcast_route) instead of
  per-chip unicasts (a segment's chips are a chip-linear range: <= 3 rectangles).
- What (PACKED_HIST_OWNED=on; off = PB.9G / PB.MB4 unchanged): tt/owned_gather.py (OwnedSend, the program: home core
  + one port core a fabric direction and link, 1 + 4 x HIST_OWNED_LINKS cores of the gather column x = 11; 9 of 10 at
  2 links: the Galaxy has 2 routing planes a direction, "4 eth channels, but only 2 routing planes" in the fabric log);
  tt/history_gather.py owned_* (OwnedGroup: every segment's cache of one source; per-cache markers buf.own instead of
  the buffer's pre / ready / owner; gather / holds / prefetch owned-aware; ``owned_begin`` from GluedCall with the
  chip -> segment map, idle chips segment 0); tt/gather_hide.Hide (one item a source in owned chunks; the tails region
  sends in place; OWNED_BYTES_MS = 24e6 for the long plan: the owned send's measured rate, not the high_bw gather's
  64e6); tt/packed_glued.GluedCall.owned; tt/packed_device.bind (no lane history buffers when owned). Setting
  PACKED_HIST_OWNED (on, call-time, batch_call_settings + batch_call_apply), HIST_OWNED_LINKS (2, load-time).
- DS41-311 bit again (fast tier x2: "history_gather has no attribute owned_on" after the runner's in-place reload):
  every top-level importer of history_gather (model, compressor, sparse_attention, gather_hide, packed_device) now
  calls importlib.import_module(__package__ + ".history_gather") first, so the package attribute is the fresh module.
- Per-chunk values this change touches (rule 20) and how they reach the device: the chip -> segment map (4 bits a
  chip), every entry's row range [lo, hi) (prefetch [0, h), long-hide pieces [a, b) clipped to each entry's h, tails
  [h, local rows)), the entries' record-buffer addresses: common value slots of ONE prepared program a (history
  buffer, record buffer, R, axis) geometry (ttnn.bringup.run_prepared values; no program a layout: the op test's
  program-cache count +0 over 8 layouts x 2 ranges, the fast tier 0 builds after warm-up). The packs keep MB.4's
  ckv_bfp4_rows value slots. Nothing new is uploaded per chunk.
- Load reasons: (1) tests/packed/test_owned_send_ops.py on the runner, mesh only: "the program's bytes on every layout
  vs the host and its time vs the four gathers: op level". (2) PB.HG#test_packed_fast_tier (accuracy load): "the
  owned path inside real packed chunks vs each request alone (bit-exact outputs / served / acks): needs the model".
  (3) the gate (one accuracy + one perf load + the cold no-rebuild): the brief's ladder and oracle.
- Op test (test_owned_send_ops, 8x4): every layout (4 x 8, 4 x 7 + 4 idle, 1 + 31, six mixed, full and mid-slot
  ranges, R 64 / 2176, inside a gather region) byte-exact on every chip, nothing else written. Timing at a 1M slot's
  R = 8192 rows (80 MiB a history a chip; the model's ratio-2 sources have 16384), 4 segments x 8 chips, host-timed
  median of 5: on the gather column owned 3.52 ms vs four high_bw_all_gathers 5.57 ms; serial on the whole grid owned
  3.35 vs 2.26 (the stock op's own workers next to every link). Link-bound: ~9 GB/s a channel of multi-hop unicast.
- Fast tier (PB.HG#test_packed_fast_tier, owned on): PASS; 18 segments vs alone outputs 0 (max |diff| 0.0), served 0,
  acks 0, bystanders 0, hide regions 18, builds after warm-up 0 in 22 serving calls.
- Gate `gate PB.HG --check` (devjob pbhg_gate1, after the last code edit, hooks clean): PASS PB.HG. Oracle
  packed_alone_vs_today 0.0, lane_equiv 0.0, slot digests 0, packed_determinism 0.0; determinism 0.0; fast tier 18
  segments vs alone all 0, 0 builds; no_rebuild cold strict 0 builds on every path; single 8192 139264 100.1 ms, 1M
  127.6 ms (unchanged). owned_begin's log: 3 gathered sources x 160 MiB = 480 MiB a chip (off: that a lane).
- Ladder A/B on the same code and load (on: the gate's walls; off: the same test with DS41_PACKED_HIST_OWNED=off,
  resident run 20261121-162834), steady wall off -> on: 4 x 2k 118.6 -> 112.6 (139264), 163.2 -> 151.9 (1M); 4 x 1600s
  122.3 -> 116.7, 173.3 -> 159.9; device -4.1 / -13.6, -6.3 / -18.5 ms; resumed likewise (117.4 -> 112.6, 162.6 ->
  151.6, 122.6 -> 116.5, 172.0 -> 160.4). Exposed history_gather section off -> on: 1.13 -> 0.88, 7.18 -> 1.46, 1.74 ->
  0.66, 8.04 -> 1.65 ms; traced gather programs a chunk 15 -> 6, 32 -> 17, 18 -> 6, 36 -> 17; the indexer section
  -3.8 / -6.6 / -5.1 / -9.1 ms (less gather traffic beside the decodes / top-k). Exposed host at 1M +2..5 ms (hidden
  before behind the longer device time). Report: results/PB.HG_ladder.md.
- Gathers alone (op test, gather column, 4 x 8 chips): 139264 (2176 rows an owner) owned 1.09 vs 4 gathers 1.62 ms;
  1M (16384) 6.57 vs 10.79 ms. Whole grid serially the stock gather is faster (0.75 / 4.37 ms): link-bound unicasts.
- Expected vs measured: expected device -0.3..-1 ms (139264) / -1..-3 ms (1M); measured -4.1 / -13.6 (4 x 2k). The
  expectation missed the indexer: its decodes / top-k regions ran beside four histories' gathers (DRAM writes + link
  traffic); now beside one.
- Next lever (not built): a relay chip a (segment, column) to cut the unicasts' link bytes ~1.7-2.7x (DS41-375: a
  same-mesh 2D multicast is rooted at its sender, so a tree needs the relay).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate PB.HG --spec $BRINGUP_SPEC --check` (devjob); the
  op test: `python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "BRINGUP_CHUNK=8192
  [PBHG_ROWS=64 PBHG_TIME_ROWS=16384] scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/packed/test_owned_send_ops.py"`.

## PB.NC perf (attempt 1): ASSESSMENT, what not converting the history bf16 -> 4-bit every chunk would save at 1M (tests only)
- Read first: OV-3, OV-58, OV-61, rule 20, perf_guide s1-4, PB.MB4 / PB.HG sections above.
- Expected saving per chunk (written before any device run; perf_guide s1-4). Side B skips only
  history_gather._pack (ttnn.bringup.ckv_bfp4_rows: read the served bf16 rows, tilize, round, write 320 B records);
  the gathers / owned sends, attention and the rest are unchanged. Single 8192 at 1024000: 3 sources x (prefetch pack
  of the whole history ~8-16k rows a chip + tail pack of 2 periods): device -0.5..-1.0 ms (MB.4: the fused pack at 1M
  ~0.3-0.5 ms a source, part of it under the first engram wait, so the wall sees less), host -0.2..-0.4 ms
  (~6 run_prepared calls x 0.06 ms). Chip-aligned 4 x 2k at 1024000: 4 segments x 3 sources = 12 prefetch packs + 12
  tails: device -2..-4 ms of pack programs, but they run as compute steps of the gather-hide regions beside sdpa /
  indexer (PB.MB4), so wall -1..-3 ms; host -0.7..-1.5 ms (~24 calls). Outputs A vs B identical (records are per row).
- Load reason: "needs the ladder's executed-layers load (layers 0-20, decode, 8192, resident engram, max_seq the
  deployment in whole chunks) because the pack's cost exists only at a 1M history through every source layer; ONE
  load answers both paths and both sides".
- What (tests only, no model change, no switch): tests/perf/test_no_conversion_probe.py (BATCH_SHARED, BATCH_LAYERS
  executed, the ladder's load through test_skip_positions._runtime, 4 serving slots at the deployment context).
  Path single8192: PACKED=off, slot 0, ChunkDescriptor.single, PD.run_chunk. Path 4x2k: PACKED=on, PACKED_GLUED=on,
  ChunkDescriptor.packed of four 2048 segments at h (slot k, text from k x 4096), order rotated every run (as
  ladder_common.packed_ladder). Side B: one untimed collect run (prev + timed chunk) records every cache _pack sees;
  NoConv.prepack packs each slab [0, R) once (the original _pack_rows) into a HistoryBuffer of its own; then _pack
  returns it (nothing launched) and _owned_send (PB.HG) swaps every local record buffer for the cache last "packed"
  into it (locmap). Counter: owned-send entries read from an unfilled loc must be 0 (asserted; DS41-377).
  DS41-311: history_gather is imported through importlib.import_module so the patch hits the live module; DS41-314:
  PACKED / PACKED_GLUED set per call with packed_common.switch (snap + env), never settings.refresh().
- DECIDED (OV-58): no Setting. The brief asks for an assessment, "tests only, never a serving switch"; side B is a
  monkeypatch inside the probe. `orchestrator measure --task PB.NC` not run (as PB.9C / PB.T: no model switch).
- Gotcha: the outputs A vs B alone would not prove side B's owned sends read the pre-packed records. The local
  record buffers still hold the A side's identical bytes, so a missed swap reads stale-but-equal records. That is why
  the substitution counter was added (run 2: 3400 swapped, 0 unmapped).
- Measured (two runs, one load; report results/PB.NC_no_conversion.md). At 1024000, wall A / B ms: single 131.1 /
  131.8 and 131.7 / 130.7; 4 x 2k 150.7 / 151.0 and 151.9 / 150.2. Device saved 0.26-0.49 (single), 1.6-3.0 (4x2k).
  Pack host 0.95 ms (6 calls) / 2.55 ms (24 calls). Outputs A vs B 0.0 on both paths. vs the estimate: device as
  expected, but the wall moves at most 1-2 ms, which is noise level. The packs are already hidden, and on 4x2k B's
  device time saved turns into exposed host (the critical path is the host, not the packs).
- Per-chunk values (rule 20): none touched (no model code changed).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate PB.NC --spec $BRINGUP_SPEC --check --unit
  "PB.NC#test_no_conversion_probe"` (devjob); knobs NOCONV_POS, NOCONV_RUNS, NOCONV_PATHS, NOCONV_OUT (JSON, default
  /tmp/pb_nc_noconv.json).
- Gate `gate PB.NC --check` (devjob pbnc_gate1, after the last test edit, hooks clean): PASS PB.NC. The probe unit was
  reused (same code); determinism 0.0; fast tier passed; every noconv wall > 0; noconv max |diff| 0.0 on both paths.

## PB.AR perf (attempt 1): ANALYSIS, the sparse attention's roofline over the top-k rows (probes only)
- Read first: OV-3 / OV-58 / OV-61, rule 20, perf_guide, X.7_roofline.md, DS41-307 (the P.81 sdpa probe), DS41-68.
- Expected before any device run (from the code): per chip per compressed layer at chunk 8192 (S = 256 query rows a
  chip, 64 heads, d 512, MQA: one KV row per key for all 64 heads, [window 128 bf16 | top-k 512 bfp4 records]):
  4 x 64 x 256 x 640 x 512 = 21.5 GFLOP; DRAM reads 256 x (128 x 1024 B + 512 x 320 B) = 75.5 MB (window 33.6 +
  records 41.9). Roofline (X.7: HiFi2 297 TFLOP/s on 110 cores, 512 GB/s): compute 0.072 ms, DRAM 0.147 ms ->
  DRAM-bound at ~0.15-0.18 ms a layer, ~2.7-3.3 ms a chunk over the 18 compressed layers that run (DECODER_SKIP=
  decode: layers 2-19). Expected measured (DS41-307 scaled 160 -> 256 rows: 3 queries on the busiest core instead of 2):
  ~0.45 ms a layer, ~8 ms a chunk: ~3x the roofline, per-query bound (not DRAM). No serving change: device / host saving
  of this step 0 (analysis); the ranked list in results/PB.AR_attention_roofline.md says what each fix would save.
- Load reason: none ("mesh only: every probe is one op on synthetic tensors at the model's per-chip shapes; no model").
- DECIDED (OV-58): no Setting and no `orchestrator measure` (no model switch exists to A/B; the brief: no model change).
- What (tests only): tests/perf/test_attention_roofline_probe.py (BATCH_SHARED, mesh only, 0 model loads): (a) plain
  matmuls of the sdpa's FLOPs at HiFi2 + fp32 DEST (bmm at the sdpa's shapes, shared-in1 reuse, 2D mcast, a K 4096
  peak), (b) ttnn.bringup.sparse_sdpa at the model's shapes (S 110 / 160 / 220 / 256, top-k 128 / 256 / 512, ids
  random / clustered / shared over a 524288-row table or 32768, 4 chip-aligned lanes with kv_lane_map_runtime, the
  window-only BF16 call), (c) a gather-only probe kernel (ttnn.generic_op, the sdpa's 110 cores and ids). Knobs
  ATTN_CASES (tag prefixes), ATTN_ITERS, ATTN_OUT.
- Measured (gate run; report results/PB.AR_attention_roofline.md): S 256 sdpa 0.467 ms a layer = 46 TFLOP/s, 236 GB/s:
  32 % of the DRAM roofline (0.147 ms), 16 % of the paper HiFi2 peak; the measured matmul peak at these settings is
  147.7 TFLOP/s on 100 cores (55 % of paper); a plain bmm of the same shapes 0.527 ms (41 TFLOP/s, slower than the
  sdpa). Gather alone 0.209 ms (362 GB/s; 320 B records 267 GB/s, 1 KB window rows 457). Fit: ~47 us a query + ~21.6 us
  a 128-key chunk; time = busiest core's queries (ceil(256/110) = 3) x ~0.16 ms. Lanes / table size / id locality
  +-2 %. 8192 chunk: 8.8 ms of sparse_sdpa vs 2.8 at roofline. DS41-378.
- 4 x 2k at 1M: the op is not slower (lanes 0.466 vs 0.467; per-chip spread 3-4 % on identical work on every case);
  its +11 ms attention section at 1M vs single is outside the sdpa (gather-hide traffic in that section; inference).
- Ranked (report s6): k-split balance -1.0..-1.3 ms a chunk; records straight into the unpacker -1.0..-1.2; window-block
  reuse -0.8..-1.3; TILE q / out -0.6..-1.0; lazy rescale -0.3..-0.6.
- Gotchas: MatmulMultiCoreReuseProgramConfig refuses a broadcast in1 with batched in0 (!broadcast_batch) and N splits
  (N == per_core_N): the shared-in1 case is in0 reshaped to [1, 1, B x 64, K]. The first 2D-mcast baseline used
  in0_block_w 2 and spilled fp32 partials (48 TFLOP/s; K-51); the whole K a block with out_block_h 4: 74. bmm with a
  bf16 in1 does not fit L1 (1.8 MB of CBs): dropped.
- Per-chunk values (rule 20): none touched (no model code).
- Gate `gate PB.AR --check` (devjob pbar_gate1, after the last test edit, hooks clean): PASS PB.AR; every ar_* > 0,
  determinism_max_abs_diff 0.0.
- Re-run: `python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "BRINGUP_CHUNK=8192
  [ATTN_CASES=sdpa_,gather,mm_] scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/perf/test_attention_roofline_probe.py"`.

## PB.BM perf (attempt 1): indexer top-k by EXACT block-max pre-selection (INDEXER_BLOCK_SELECT on | off)
- Read first: OV-3 / OV-58 / OV-61, PACKED_RULES.md, rule 20, perf_guide s1-6, DS41-170 / DS41-286 / DS41-342 (top-k tie
  order), DS41-333 (the indexer at long lanes), indexer_score README (block epilogue: block_max_out needs block_keys <= 8).
- Expected saving per chunk (written before the first model run; from the op probe below, perf_guide s1-4). The
  plain indexers that run under DECODER_SKIP=decode: layers 2 / 8 / 14 (ratio 2; 256 query rows a chip at 8192).
  At 1024000 (512k keys a row): plain top-k 3.22 ms a layer (probe, pipelined) -> block path ~1.0 ms (B 32) = -2.2 ms
  a layer, -6.6 ms a chunk, minus the score writer's block max (P.16 measured +17 % of the score op at 2x2: 0 .. +1.2
  ms a layer if the writer becomes the score's bound): device -3 .. -6.6 ms a chunk, single 8192 and each chip-aligned
  4 x 2k pass alike. Host: +4 launches a layer (block_max_bf16, 2nd top-k, block_gather, block_keys): +12 launches,
  ~+0.5 .. +1 ms a chunk. At 139264 (69632 keys): the block path costs MORE (probe 1.31 vs 0.47 ms a layer: its
  fixed cost), so INDEXER_BLOCK_SELECT_MIN keeps the plain top-k below the crossover (expected ~150-200k keys); 0 there.
  4 x 2k at 1M: the top-k is a gather-hide cover region (PB.6L): a shorter top-k hides less gather (cover 1.55 -> 0.5
  ms a layer), so part of the device saving may come back as exposed gather.
- DECIDED (OV-58): the stock ttnn.gather cannot do the block gather at this size: its ROW_MAJOR kernel keeps each
  core's WHOLE input row in L1 (a 512k-key bf16 row is 1 MiB; 2 MiB at ratio 1) and reads it once a core; and
  ttnn.embedding would need the [Sq, T] scores viewed as [Sq T / B, B] rows (a copy of 256 MiB a chip). So three
  small data-movement Python ops in a new bring-up folder (ttnn/ttnn/bringup/block_select_ttnn; rule 16.3: no fork or
  library op does a per-row block gather): block_gather (64-byte DRAM reads of each chosen block), block_keys
  (positions -> key ids), block_max_bf16 (the writer's fp32 8-key maxes -> bf16 32-key maxes in one pass; 0.39 ms at
  1M vs 0.79 for stock tilize + typecast + untilize, which also cannot fold 4 blocks into one).
- Load reasons: (1) tests/perf/test_block_select_probe.py, mesh only (0 loads): "the ops' bytes and the chain's time
  at the per-chip 1M shape on synthetic scores: op level". (2) tests/test_indexer_block_select.py on the accuracy
  load with tables to 1032192 (BATCH_MAX_SEQ): "needs the full accuracy load because the real-text request runs
  layers 0-14 to 1M and its state's positions need the load's tables; one load serves every iteration".
- What (INDEXER_BLOCK_SELECT=on, call-time; off = the plain top-k, unchanged): tt/indexer.py TtIndexer.block_topk
  (score op's block_max_out at block_keys 8, no pin -> ttnn.bringup.block_max_bf16 factor 4 -> topk_large_indices of
  the 32-key block maxes -> ttnn.bringup.block_gather -> topk_large_indices of the 512 x 32 gathered scores ->
  ttnn.bringup.block_keys), use_block_select (a host decision on the chunk's largest kv_len vs
  INDEXER_BLOCK_SELECT_MIN / _MIN_PACKED), force_block_select (runtime.compile: the first warm chunk / glued layout on
  the block path, the second on the plain one: both programs built whatever the thresholds), BSEL_CHECK (tests).
  Single path: TtIndexer.__call__ / topk_of. Glued packed path: tt/packed_glued._index_dynamic (score epilogue,
  per-pass block counts from the derive, block maxes rolled with the scores' phase table). tt/glued_derive.py: the idx
  program of a plain indexer with block A3 > 1 also writes each pass's block counts (lens / 32, D_BSEL).
  tt/gather_hide.py: the top-k region's cover is BSEL_TOPK_COVER_MS (0.5) when the packed block path runs.
  Sections (profile): indexer_bmax (the fold), indexer_topk (both top-k's; the plain top-k too, on and off alike),
  indexer_gather. Plain indexers only (TtIndexer: layers 2 / 8 / 14 under decode): the candidate source's block maxes
  are pinned and the re-index layers' scores biased.
- DECIDED (OV-58): block 32 (the writer writes 8-key maxes, block_keys <= 8; the fold to 32 costs nothing extra in
  block_max_bf16 and the first top-k runs over 4x fewer blocks: probe chain 1.00 ms vs 1.44 at block 8).
- DECIDED (OV-58): every block-select op on the matmul grid (max_grid, 11 x 10): the gather-hide regions' compute
  sub-device is exactly that rectangle (tt/gather_hide.Split), so the ops are safe inside the top-k region.
- Measured, same code, on (MIN 0: block path everywhere) vs off, the gate's ladder env, 8x4 (resident logs
  20261121-222324 on, 20261121-222754 off). Single 8192 steady: 139264 wall 101.6 vs 99.7, device 86.20 vs 85.71;
  1024000 wall 122.7 vs 126.7, device 110.93 vs 115.79 (indexer_topk -8.66, indexer_bmax +1.17, indexer_gather +1.10,
  indexer (the score op, block-max epilogue) +2.02). Chip-aligned 4 x 2k steady: 139264 wall 115.6 vs 112.0, device
  100.22 vs 95.09 (indexer +4.13); 1024000 wall 152.2 vs 150.0, device 135.53 vs 134.69 (indexer +12.9, top-k -8.77).
- Why the packed path loses (tests/perf/test_bsel_score_probe.py, held perf model's indexer objects, random data, a
  1M slot's capacity): the score op with vs without block_max_out, ms a call: single ring 1.33 -> 1.67 (139264), 9.60
  -> 12.72 (1M); the glued lane_ranges ring (q chunk 32) 1.60 -> 2.92 (139264), 12.01 -> 14.57 (1M). The writer-core
  block max (one RISC-V compare a score) is the cost; in the ranges ring it exceeds what the top-k saves.
- DECIDED (OV-58): INDEXER_BLOCK_SELECT_MIN = 131072 keys (single path; crossover ~111k keys from the two points),
  INDEXER_BLOCK_SELECT_MIN_PACKED = 2^30 (never at the default: the packed ring's writer cost; the packed path stays
  built, warmed and tested: force_block_select). Next lever (not built in this attempt): the block max on the score
  op's compute kernel (reduce_block_max_row over each score tile on the FPU, the fork's MSA block-pool path) with a
  block-major [Sq / 32, NB, 32] output the writer can write in 128-byte runs, then a transpose in block_max_bf16:
  removes the +0.34..3.1 ms a layer the writer adds, which would turn the packed path into a win at 1M.
- block_gather v2 (after the A/B above): K / 4 blocks a work unit over both RISCs: 0.44 -> 0.34 ms a call (probe).
- Correctness (tests/test_indexer_block_select.py, accuracy load, real agentic text, point 32768 iteration run):
  single 24576 rows / 3 calls, packed 24576 rows: 0 mismatches; 12642 / 9433 rows differ only by 512th-value ties
  (bf16 scores tie often; the plain op's tie choice is as undefined).
- Gotchas: DS41-379 (ttnn.gather), DS41-380 (test hooks on a reloaded module), DS41-381 (one capacity a packed chunk);
  the walls need their own load after skip_positions (DS41-356 again: run the walls first in one submit); the
  resident runner refuses `--one-process` (use `A && B`).
- Rule 20, per-chunk values this change touches and how they reach the device: the valid block count (kv_len / 32):
  single path a prepared-program value slot (block_max_bf16 common arg n) and topk_large_indices' valid_length
  (runtime, not hashed); glued path the derive's per-pass block-count tensors ([1] uint32 a chip, made on the device
  from the chunk header: D_BSEL), read by block_max_bf16 (n_tensor) and topk_large_indices (valid_length_tensor). The
  scores' kv_len to the score op as before. Block ids / positions / keys are device data. The path choice (block vs
  plain) is a host decision on the chunk's length; every program depends on shapes (capacity) only and both paths are
  warmed by runtime.compile. Nothing new is uploaded per chunk.
- DECIDED (OV-58; answers the owner rule on kernel work): the compute-kernel block max is NOT built in this attempt.
  The brief's trigger ("the writer-core block max costs more than the top-k saves") holds for the packed lane_ranges
  ring only, where INDEXER_BLOCK_SELECT_MIN_PACKED keeps the plain top-k at the default; on the single path the writer
  costs +0.67 ms a layer at 1M in the model against -2.9 saved. An EXACT FPU block max needs, per score tile: a bf16
  pack (the score output's own rounding; the fp32 accumulator would reach srcA as TF32, and max(TF32) rounded to bf16
  is not always the max of the written bf16 scores), reduce_block_max_row<1> on it with a 1.0 scaler tile from the
  reader (as the MSA block-pool path), a bf16 column tile to the writer, 64-byte strip-major writes ([Sq / 32, NB x 32]:
  one query strip's 32 maxes of one block a write), and a transpose to [Sq, NB] in block_max_bf16 (a strip's blocks
  are one contiguous read). Files: ring factory (CBs, defines, validation), compute PHASE 3, writer, reader (scaler),
  device-op attributes + hash, nanobind, the fork's tests + regression baseline. Expected: packed 4 x 2k at 1M about
  -5..-7 ms device a chunk, single -2 ms more.
- Defaults' A/B (orchestrator measure --part test, then measure --task PB.BM reused them; switch on, MIN 131072,
  MIN_PACKED never): single 8192 steady 1024000 wall 122.6 ms, device 110.75 (off: 126.7 / 115.79); 139264 99.1 /
  86.02 (plain path; off 99.7 / 85.71); 4 x 2k 1024000 151.7 / 133.53, 139264 114.3 / 94.91 (plain path; off 150.0 /
  134.69, 112.0 / 95.09). test_no_rebuild cold strict: every path PASS, 0 builds after warm-up (both paths warmed).
- Gate check #1 (devjob pbbm_gate): every threshold ok except idx_block_select_set_mismatches* MISSING: the test's
  1M packed check hook used ttnn.clone inside the top-k's gather-hide region (DS41-382; whole grid across two
  sub-devices). The 139264 point had passed: single 24576 rows 0 mismatches (15564 tie rows), packed 24576 rows 0
  (9681 tie rows). Fixed in the test (block_keys over an iota on the matmul grid); unit re-run alone.
- Gate check #2 (devjob pbbm_gate2, final code, hooks clean): PASS PB.BM. idx_block_select_set_mismatches 0 (single
  and packed, 139264 and 1024000; tie rows 15564 / 9681 / 19470 / 10626 of 24576 each); packed oracle 0 / 0 / 0 / 0;
  determinism 0; no-rebuild cold strict every path 0 builds; single steady 139264 100.4 ms, 1024000 122.9; 4 x 2k
  111.5 / 151.0. Report: results/PB.BM_ladder.md.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate PB.BM --spec $BRINGUP_SPEC --check` (devjob);
  op probe: `resident submit --cmd "BRINGUP_CHUNK=8192 scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/perf/test_block_select_probe.py"`; score-epilogue probe: the walls env +
  tests/perf/test_bsel_score_probe.py; an A/B at threshold 0: DS41_INDEXER_BLOCK_SELECT_MIN=0
  DS41_INDEXER_BLOCK_SELECT_MIN_PACKED=0 with the walls / skip_positions units.
