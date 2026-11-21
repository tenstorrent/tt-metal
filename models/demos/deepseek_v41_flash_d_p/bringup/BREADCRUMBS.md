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
