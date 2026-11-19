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

## P.73 (perf, attempt 1): the STATIC_TABLES=off history gather overlapped with compute (HISTORY_GATHER_OVERLAP off | on)

- Estimate (written before the change). Plan: a sub-device split that needs no op re-gridded: compute = the capped
  11x10 grid at (0,0) (every model op already runs there, FIX.15 / FIX.36), gather = the free 12th column (10 cores; the
  gather needs 3 x links). In layer s (2, 8, 14) the TP gather stays serial (~10 % of the bytes), the SP gather (~90 %)
  runs on the gather column while the indexer's top-k runs on the compute grid; clear = the join. Only non-fabric
  compute may share the region: the ring score (axis 0) would share the gather's routers (V3 overlaps only different
  axes / no fabric). Expected device saving a chunk = sum over the 3 sources of min(SP gather, top-k): ~0.5 ms at 51k,
  ~6-9 ms at 1M (top-k ~2-3 ms a source at 1M, SP gather ~4-6 ms); host +0.3..+1 ms (2 manager switches a source).
  Far less than off - on (+30 ms device at 1M): the untilize after the gather and the remap stay serial.
- Change (HISTORY_GATHER_OVERLAP=on, default off as the owner's brief; only with STATIC_TABLES=off, INDEXER_GLUE=on):
  tt/history_overlap.py (sub-device manager made once at load, local L1 0), tt/sparse_attention.py
  (`_compressed(overlap, run)`, `history_overlapped`: the gathered rows go into the chunk memo under the key the
  attention step looks up), tt/indexer.py (`topk_of` takes a one-shot `overlap` hook; `topk_large_indices(grid=)`),
  tt/model.py (`gather_overlap` on the KV source layers 2, 8, 14, 20 that index themselves; the indexer step sets the
  hook). Off path untouched. Other boxes (no core cap): `get` is None, the serial path.
- Region (per source, per chunk): TP gather serial (as before), then load_sub_device_manager, the SP high_bw_all_gather
  on sub-device 1 (row 0, columns 0-5: 4 link workers + 2 copy cores, its own L1_SMALL semaphore pair per axis,
  DS41-216), topk_large_indices on sub-device 0 (110 cores: rows 1-9 + row 0 cols 6-7), clear = the join; the
  gather's input / output are freed only after it. Then untilize + remap as before. No fabric op in the region (the
  indexer's ring score is SP-axis: same routers as the gather).
- Split probes (tests/test_history_gather_overlap.py, standalone, 1M shapes, ms a source, serial gather + top-k ->
  region): col 11 (compute = the capped 11x10 itself): 7.28 -> 8.44 (gather 8.35 vs 5.13 serial: far from the links);
  row 0 all + compute 11x9 (99): 7.16 -> 5.30 but the top-k differs (the op splits rows by core count, ties);
  row0:3-8 / 6-11: gather 7.19, nothing hidden; row0:0-5 + 110 compute: 7.28 -> 5.45, top-k identical: shipped
  (DS41-286). At 51k: 0.58 -> 0.48 a source.
- Rule 18 (owner's call): every op <= 110 cores (top-k 110, gather 6), but inside a region 116 cores are busy at once
  (the 6 gather cores do data movement only), 3 regions a chunk, ~0.5-5.5 ms each.
- Model check (tests/test_history_gather_overlap_model.py, one load, DS41_STATIC_TABLES=off
  DS41_HISTORY_GATHER_OVERLAP=on DS41_DECODER_SKIP=decode; off = the blocks' gather_overlap flipped): layer 19's output
  on vs off bit-identical at p51200, p256000, p1024000; 3 regions a chunk (layers 2, 8, 14; layer 20 runs only its KV
  head under decode), none off. Chunk wall (synced, median of 5, one process): p51200 110.8 vs 109.5, p256000 119.1
  vs 117.0 (noise), p1024000 150.7 vs 152.8 ms.
- Overseer stop (2026-11-18, three messages): no sub-device split (116 cores busy at once is over the 110 cap, FIX.36's
  hang cause; ~2 ms of ~30 ms too little), then: no gap hunt, inject the gather where the links are quiet (engram wait,
  id readback). Done: the sub-device wiring REVERTED (patch kept at /scratch/dnijemcevic/phase_1117/
  p73_subdevice_wiring.patch); tt/history_overlap.py and tests/test_history_gather_overlap.py stay as the measured probe
  (not wired). Its on-side frozen tests had passed (measure --part test, rc 0) before the stop.
- Analysis (rt-profiler timeline, STATIC_TABLES=off, p1024000, all 40 layers, BRINGUP_PROFILE_OPS=1; ms a source, layer
  2): TP gather 2.17 (the COMPOSITE all_gather: nb x 80 = 16080 rows is not tile-aligned at odd nb, so tt/ccl._fits
  refuses high_bw), SP high_bw_all_gather 4.98, untilize 2.77 = ~10 ms; x 3 sources = the whole off - on gap (~30 ms).
  Device idle: before layer 1's engram matmul 66.6 ms under the profiler (the host's engram rows; the clean wall's
  host share at 1M is ~25-33 ms), L2 / L8 / L14 compressor 14 / 10 / 7 ms; nothing else above 4 ms. What runs in the
  engram window: nothing (the device waits); the history [0, start) is complete at chunk start, so its gather could
  be enqueued just before that wait, but only the history part (the chunk's own period is written at layer s), and
  the table must stay one tensor for sparse_sdpa: [history | chunk] needs a 0.5 GB concat or a fixed-stride table
  (high_bw_all_gather's gathered_slab_global / gathered_prefix_tensor, V3's prefix gather) plus a remap with that
  stride. The id readback window: the same idle device, the same constraint. Not built in this attempt.
- Built instead (the cheapest bit-identical cut of that ~10 ms, no op or core added): HISTORY_GATHER_OVERLAP=on now =
  the history gathered as the slab's ROW_MAJOR rows (tt/sparse_attention._compressed, ``gather_rows``): no tilize, no
  untilize, high_bw_all_gather on both axes at any row count. Standalone (test_history_gather_rows): bit-identical,
  10.04 -> 5.93 ms a source at 1M, 1.10 -> 0.58 at 51k (expected ~-12 ms device a chunk at 1M, ~-1.5 at 51k).
  Remaining ~6 ms a source = the two gathers: the engram-window move above is the next step.
- Model check (tests/test_history_gather_overlap_model.py, rows version, one load): layer 19 on vs off bit-identical
  at p51200 and p1024000; chunk wall p51200 108.9 vs 110.6, p1024000 144.0 vs 152.8 ms.
- measure --part test (change on: STATIC_TABLES=off HISTORY_GATHER_OVERLAP=on): test_determinism rc 0,
  test_decoder_skip_decode rc 0, test_skip_positions HUNG (DS41-288): dispatch TIMEOUT at p1024000 resumed, warm-up
  run 1, right after the steady kind's profiled run. Stopped all device work there (brief / overseer); the ladder,
  the final measure and the gate were NOT run. Steady points it measured (wall / device ms; P.72 off, P.72 on):
  p51200 111.5 / 82.2 (109.1 / 83.2, 104.5 / 80.7); p256000 112.3 / 91.4 (118.3 / 92.7, 108.1 / 84.2);
  p512000 121.5 / 98.2 (125.9 / 104.4, 112.4 / 88.5); p768000 132.5 / 107.3 (141.4 / 116.3, 116.2 / 92.9);
  p1024000 140.7 / 115.2 (155.8 / 128.3, 123.0 / 98.4). P.72 predates FIX.36's caps: the orchestrator's A/B is the
  real off column.
- Next (not built): the remaining ~6 ms a source at 1M is the two gathers; moving the history part [0, start) to the
  engram wait before layer 1 (the device idles there) needs a fixed-stride table (high_bw_all_gather's
  gathered_slab_global, V3's prefix gather) + the remap with that stride, and the chunk's own period gathered at layer s.
- Re-run: `DS41_STATIC_TABLES=off DS41_HISTORY_GATHER_OVERLAP=on DS41_DECODER_SKIP=decode scripts/run_safe_pytest.sh
  --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_history_gather_overlap_model.py -s`;
  standalone: `scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_history_gather_overlap.py -s`.

## P.73 (perf, attempt 2): the 07:06 hang first (OV-48): STATIC_TABLES=off capped, loop hung again

- Expected saving (unchanged from attempt 1, written before that change): HISTORY_GATHER_OVERLAP=on = the ROW_MAJOR
  history gather, ~-4 ms device a source at 1M (~-12 ms a chunk), ~-0.5 a source at 51k; host ~0. No perf code
  changed in this attempt.
- tests/test_skip_hang_loop.py: HANGLOOP_GATHER_ROWS=alt flips the sparse attentions' ``gather_rows`` every round
  (both switch values in one load; per-side determinism reference, and on vs off compared and asserted bit-identical);
  HANGLOOP_STOP_OVER=1 stops after the first round with a program over the cap.
- Audit (STATIC_TABLES=off, watch + tests/perf/test_core_audit.py BRINGUP_POS=1024000): 120-core programs on BOTH
  switch values, every chunk (DS41-289): the slab slice's composite copies (ND-sharded RM slab, nb x 80 rows not
  tile-aligned at odd nb: every skip-positions / A/B point), 2 uncapped bitcasts in _topk_rows; switch off also the
  composite all_gather's concat (120) and the padded tilize (119). FIX.36 had audited STATIC_TABLES=on only, so the
  attempt-1 hang (DS41-288) ran with these on both A/B sides.
- Fix (bit-identical, no switch, all boxes: no-ops without a cap or when aligned): own slab rows rounded up to
  lcm(32, chip rows a chunk) (tt/compressor; 32 alone trips update_padded_kv_cache's cache_seq % written_seq);
  bitcasts capped; tt/cores.to_tile; tt/ccl._composite_capped (stock's composite steps on the capped grid).
- Proof loop (loop3, 50 min budget, STATIC_TABLES=off, sides alternating): rounds 1-16 clean (8 rows_on, 8 rows_off,
  2,944 chunks), no program over 110 on any chip, determinism 0 on both sides, rows_on vs rows_off layer 19 identical
  in all 15 cross compares. Round 17 (rows_on, p256000 resumed run 10) HUNG: the DS41-283 signature (chips 0-3 /
  24-27 stopped before the 104-core ring_attention gather + indexer program; chip 25 MMIO dead at teardown). Box reset
  by the safe runner (32 boards back, 10:12). Per the brief: device work stopped. NOT done: 20 clean rounds a side, the
  STATIC_TABLES=on control run, measure --part test, the final measure, the gate.
- Reading: the capped tree still drops a chip at about the pre-FIX.36 rate (~2,400-2,900 chunks), so the 120-core
  programs were not the whole cause; FIX.36's 41 clean rounds were one sample. The rows gather itself runs on few
  cores and the hung round's last finished programs are the same as FIX.36's hang 3 (rope_chain, then the ring
  gather + indexer). Next (owner / overseer): the hardware / firmware leads of DS41-283 (which chip each time: 15, 3,
  7, 25), or a loop on STATIC_TABLES=on with the current tree to measure its rate after the caps.
- Logs: /scratch/dnijemcevic/phase_1117/p73a2/ (audit.log, coreaudit_on.log, loop2.log, loop3.log).
- Re-run: `DS41_STATIC_TABLES=off DS41_HISTORY_GATHER_OVERLAP=on DS41_DECODER_SKIP=decode HANGLOOP_GATHER_ROWS=alt
  HANGLOOP_ROUNDS=44 HANGLOOP_MINUTES=50 HANGLOOP_STOP_OVER=1 scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_skip_hang_loop.py -s` (under devjob).

## P.74 perf (attempt 1): STATIC_TABLES=off on the new tree: one whole-mesh row gather into a persistent buffer

- Owner 2026-11-18: what the served-state-only path (STATIC_TABLES=off) costs vs the per-user replicated table (on).
  The switch is STATIC_TABLES (change on = off; old = on). HISTORY_GATHER_OVERLAP (P.73) renamed HISTORY_GATHER =
  rows | rows_axes | tiles, default rows (rows_axes = P.73's row gather, tiles = the P.19 / P.72 path).
- Expected saving (written before measuring; per KV source a chunk, off side only): HISTORY_GATHER=rows vs P.73's
  rows_axes: no slab slice copy (L rows read + written: ~32 MB a chip at 1M), no TP intermediate, one whole-mesh
  gather on the torus (two Hamiltonian cycles, all four links a chip) instead of a TP then an SP line gather: device
  ~5.9 -> ~3.5 ms a source at 1M (~-7 ms a chunk over the encoder's 3 sources), ~-0.2 ms a source at 51k; host
  -2 ops and -3 allocations a source (~-0.3 ms a source). Against STATIC_TABLES=on the off side still pays the gather
  (~3-4 ms a source at 1M, ~0.3 at 51k) plus the index remap, and saves the table's slice_writes and 2.5 GiB a user a
  chip at 1M (it holds the shared buffers instead: ~1 + 0.5 GiB a chip at 1M, for all users).
- Design (tt/history_gather.py): ttnn.experimental.high_bw_all_gather straight from the served slab
  (input_batch_index = the slab's (slot, layer) row; the ND-sharded engine cache is the op's own input form, as
  deepseek_v3_d_p's MLA), gathered_dim_size = owners x L (only the first nb chunk periods move), cluster_axis=None
  for the TP-dedup cache (row-major rank r TP + c = the caches' chip-linear owner order), output = the persistent
  buffer: owner q at the FIXED row q R (R = the slab's rows), rows past L never indexed. _topk_rows takes the owner
  stride (R; nb hs on the other two paths). A third semaphore pair (axis None, DS41-216) made with the buffer.
- Departure from the brief's "ONE buffer": ONE per distinct table size (owners x slab rows). The stock op requires
  its output at exactly the worst-case shape of its input, and ttnn has no Python view of a buffer prefix of another
  volume (ttnn.view and ttnn.experimental.view both check the volume; tt_metal::view's MeshBuffer view is C++ only,
  ttnn/cpp is outside this step's paths). Ratio-1 and ratio-2 sources have R and R / 2 rows, so a model with both
  holds two buffers (1.5x the largest source's table), each shared by every chunk, user and source of its size.
- Lifetime (overseer, P.74: only the sizes gathered in the running mode): a KV source's compressed cache
  (TtServedRows.hold_history) holds the buffer of its size only while some attention reading that source runs in the
  model's DECODER_SKIP mode (TtDS41Model.history_sources: under decode not layer 20's ratio-1 source, whose readers
  are all decoder layers: no 1 GiB-at-1M ratio-1 buffer then; under off / exact / replay128 both sizes). Set at state
  build (DS41DeviceState.__init__ -> model.hold_history) and re-synced by use_decoder_skip; released in free(), the
  last holder frees it. Never per chunk (a gather on a cache without one acquires it and notes history_gather.LATE).
  So test_skip_positions' dram_state_gib_p<p> (after new_state) and dram_after include exactly the decode-mode
  buffer (ratio 2, ~0.5 GiB a chip at 1M), sized for that point's state (p + chunk), as the on side's tables are.
- One buffer serves every source of a size because a source's readers are the layers up to the next source
  (Config.kv_source_of) and the layers run in order; the memo entry is checked against the buffer's owner
  (history_gather.holds) before reuse and re-gathers instead of reading another source's rows.
- Cores: the gather is a CCL (link workers + local copy cores, a few a chip); no matmul touched.
- Checks run (device, STATIC_TABLES=off): tests/test_history_gather_overlap_model.py (HISTORY_GATHER=rows vs tiles,
  one load, DECODER_SKIP=decode, before the overseer's lifetime change): layer 19 bit-identical at p51200 and
  p1024000 (max abs diff 0.0); indicative chunk wall (one process, no profiler) rows vs tiles: p51200 103.2 vs
  105.3 ms, p1024000 129.9 vs 154.2 ms. test_decoder_skip_decode and determinism (final code, measure --part, change-on env): both rc 0;
  determinism_max_abs_diff 0.0 (state 0.0).
- Re-run: `DS41_STATIC_TABLES=off DS41_HISTORY_GATHER=rows P73_POS=51200,1024000 scripts/run_safe_pytest.sh
  --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_history_gather_overlap_model.py -s`;
  `python -m models.demos.common.bringup.orchestrator measure --task P.74 --part test_decoder_skip_decode` and
  `--part determinism` (the run names: determinism, test_decoder_skip_decode, test_contract, test_skip_positions,
  last).

## P.75 perf (attempt 1): the STATIC_TABLES=off history gather hidden in the engram wait (HISTORY_GATHER_HIDE off | on)

- Expected saving (written before the change; STATIC_TABLES=off, DECODER_SKIP=decode, sources 2 / 8 / 14). Window:
  the device idles before layer 1's engram rows (P.73: 66.6 ms under the profiler at 1M; P.74's clean host share
  25-28 ms a chunk at every position). Option (a): enqueue each source's history gather [0, c hs) a chip (c = start /
  S, complete before the chunk) at layer 1's engram wait; at layer s only the chunk's own periods [c hs, nb hs)
  (1 or 2 x 80 rows an owner) are sliced, gathered into a small persistent buffer and slice_written into the source's
  buffer at the same fixed-stride rows: the table stays ONE tensor with P.74's stride R, so _topk_rows and
  sparse_sdpa are untouched (bit-identical). Per source a chunk, device on the critical path: 1M ~3.5 -> ~0.3 ms,
  256k ~1.0 -> ~0.3, 51k ~0.3 -> ~0.3 (nothing to hide). Per chunk: ~-9 ms device at 1M, ~-2 at 256k, ~0 at 51k;
  the wall the same as long as layer 0 + the 3 history gathers fit in the engram wait (~26 ms). Host: +2 ops a
  source at layer s, +1 at layer 1 (~+0.3-0.5 ms host, the layer-1 part behind the wait). Memory: the history of
  three sources lives at once, so one buffer per source (shared by every user) instead of one per size: +2 x ~0.5
  GiB a chip at 1M. Options (b) readback and (c) ring not built: (b) is the same idle span as (a) one step earlier
  (the next chunk's start), (c) shares the SP routers with the indexer's own ring (P.73).
- Change (HISTORY_GATHER_HIDE=on, the default; active only with STATIC_TABLES=off, HISTORY_GATHER=rows; off = P.74):
  tt/history_gather.py (`prefetch`: the history [0, c hs) of a source into its own buffer, `gathered_dim_size` =
  owners c hs; `gather` takes the tail path when the buffer's prefetch matches (cache, start, version within 1 write):
  `_tail` = slab.slice_rows(c hs, nb hs) + one high_bw_all_gather into a small persistent buffer [owners m, 512]
  (m = 1 or 2 periods, 80 / 160 rows an owner at the 8x4) + ttnn.bringup.slice_write of its [1, owners, m, 512] view
  into the source buffer's [1, owners, R, 512] view (views made once at acquire, asserted no copy); anything else ->
  the P.74 full gather). tt/compressor.TtServedRows.hold_history(on, layer): key (rows, layer) = one buffer per source
  (shared by every user/state) + the two tail buffers. tt/model.py: `begin_chunk` notes the start,
  `take_engram_rows` runs `prefetch_history` (every source of `history_sources()` of the active state) before its
  wait, once a chunk. No table / index / sdpa change: P.74's stride R stays.
- Gotcha (DS41-291): the slice of the slab loses its dim-2 sharding topology, so high_bw_all_gather's full-mesh mode
  TT_FATALs; `update_tensor_topology(slab topology)` (host only). DS41-289 checked: the composite slice keys on the
  whole slab's H (engine slab 16320 rows; own slabs lcm-rounded), so the tail slice at a 16-row offset stays native and
  capped. Cores: the tail gather 10 a chip (full-mesh torus, 2 links), slice and slice_write on the capped grid.
- Model check (devjob p75_hide2, tests/test_history_gather_hide_model.py, one load, DECODER_SKIP=decode, on = prefetch,
  off = model._hide False: P.74's full gather into the same buffer): layer 19 bit-identical (max abs diff 0.0) at
  p51200 / p256000 / p1024000, each at start offset 0 and +2048 (a 2-period tail); on side every chunk 3 prefetches +
  3 tails, 0 full gathers. Indicative synced chunk wall on vs off (median of 5, one process, no profiler): p51200
  111.1 / 103.4 and +2048 104.3 / 106.5 (noise), p256000 106.3 / 108.0 and 109.8 / 111.4, p1024000 118.3 / 125.7 and
  125.4 / 129.4 ms.
- Re-run: `DS41_STATIC_TABLES=off DS41_HISTORY_GATHER_HIDE=on DS41_DECODER_SKIP=decode python -m
  models.demos.common.bringup.devjob start x -- scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/test_history_gather_hide_model.py -s` (P75_POS, P75_OFF pick points).
- measure --part test (change on: STATIC_TABLES=off HISTORY_GATHER_HIDE=on): test_decoder_skip_decode rc 0,
  test_contract rc 0 (contract_checks_failed 0, _decoder_skip 0, unwritten_records_changed 0), test_skip_positions
  rc 0 (all 6 points fit). Its steady numbers vs P.74's off side (HIDE=off; another run, so a +-3 ms wall noise):
  attention section 22.59 / 22.77 / 22.69 ms at p51200 / p256000 / p1024000 (P.74: 23.01 / 25.00 / 32.41), the
  prefetch charged to engram_rows 0.63 / 2.54 / 9.95 (0): per source a chunk ~0.14 / 0.75 / 3.2 ms moved off layer s
  into the engram wait. Device total unchanged (79.85 / 85.32 / 105.52 vs 80.17 / 84.94 / 105.41: the work moved, it
  now overlaps the host's engram time); host gap (wall - device) 29.4 / 23.4 / 17.9 (26.9 / 25.3 / 25.9); steady wall
  109.3 / 108.7 / 123.4 (107.1 / 110.2 / 131.3): ~-8 ms a chunk at 1M, ~-1.5 at 256k, noise at 51k. DRAM state +0.07
  / +0.26 / +0.99 GiB a chip (two more per-source buffers). Against STATIC_TABLES=on (FIX.36, older tree): wall
  107.2 .. 123.2 at 51k .. 1M, so off+hide at 1M is within ~0-2 ms of the table's wall; device +8 ms at 1M (now hidden).
- Final check `orchestrator measure --task P.75` (change on): the three --part runs reused; determinism rc 0
  (max_abs_diff 0.0, state 0.0); ladder rung last rc 0 (final hidden PCC 0.992, top1 / top5 1.0). The orchestrator's
  A/B (on vs off) and the gate run after this step.
- Not built: (b) the token-id readback window (the same idle span one step earlier; (a) already fills most of it at
  1M: host gap 17.9 ms left) and (c) riding the indexer's ring (shares its SP routers, P.73). No Python-built op added
  (OV-23); no fork touched or regression-run (OV-25).

## FIX.38 overseer fix (attempt 1): STATIC_TABLES=off the shipped default (owner 2026-11-18)

- Change: tt/settings.py STATIC_TABLES default "on" -> "off", reason text = owner decision 2026-11-18, P.74's A/B (off at
  1M +11.8 ms wall / +10 %, accuracy and determinism identical; the per-user replicated table, 2.5 GiB a user a chip at
  1M, removed; item 4 of the slot-state page solved). HISTORY_GATHER stays "rows" (P.74, persistent buffers) and
  HISTORY_GATHER_HIDE stays "on" (P.75 passed its gate, no A/B pending: accepted). STATIC_TABLES=on is untouched and
  still selectable (DS41_STATIC_TABLES=on). Defaults checked compatible with off: CKV_FORMAT=bf16 (bfp4_rows still
  asserts STATIC_TABLES=on, compressor.py:497), WINDOW_KV_FORMAT=bf16_tiles (off path since P.72, DS41-282).
- Docs: sparse_attention.py docstring (on = the alternative), serving_contract.md (the private table's put-back under
  encoder replay is on-only), known_issues DS41-101 (the table is the on path; its stale "default bfp4_rows" dropped).
- Gate check on the new default (X.4's units, one process + skip_positions): PASS. err_final_hidden 0.127 (limit
  0.242), top5_overlap 1.0, determinism 0.0 / 0.0 (decoder skip), contract_checks_failed 0 / 0, acks_early 0,
  unwritten_records_changed 0, steady wall p51200 105.7 ms, p1024000 120.7 ms (P.75's on side: 118.3-125.4 ms at 1M).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.38 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.

## P.76 perf (attempt 1): the indexer scores the served fp8 index rows (INDEX_K_SOURCE served | private)

- Owner 2026-11-18, item 3 of the slot-state page: the private bfloat8_b copy k8 (P.41) is right only if this
  process wrote every earlier chunk of the slot. Switch INDEX_K_SOURCE: served (change on, default) | private (P.41).
- Expected (written before the model runs; per chunk, DECODER_SKIP=decode: scored sources 2 / 8 / 14, ratio 2,
  wr = 320 local rows a W block on 8x4). At the layer: private = bfp_round of the chunk into k8 (+ the region path at
  start % S != 0); served = one unpack_fp8_rows of the chunk's W blocks (10-20 bands): device ~0, host ~0 (+0.1 ms
  op, -0.05..0.1 ms bfp_round) a source. Hidden: the history [0, c wr) decoded at layer 1's engram wait:
  ~100 / ~500 / ~2000 bands a source at 51k / 256k / 1M: ~0.03 / ~0.1 / ~0.4 ms device a
  source (0.1 / 0.3 / 1.2 ms a chunk), host ~0.2-0.38 ms a source on the main thread while the engram worker runs.
  Wall: ~0 if it fits in the engram wait (P.75 left 17.9 ms host gap at 1M). DRAM: no per-user k8 (8.9 MB a ratio-2
  source, 17.8 MB layer 20, a chip at 1M) for one shared buffer a source.
- New fork op ttnn.bringup.unpack_fp8_rows (fp8_rows_ttnn, next to pack_fp8_rows): rows [lo, hi) of a batch of the
  ND-sharded served tensor -> bfloat8_b tiles in place, e4m3 x scale in fp32 (SFPU) rounded to nearest even
  (ckv_round_bfp_face<7>). Chosen over the stock per_token_cast_back (needs an FP8_E4M3 tensor plus a separate fp32
  scale tensor, writes RM bf16 / fp32: + slices, bitcasts, tilize, bfp_round, every op over the whole history). One
  prepared program per (shapes, core count), the range 5 value slots a core (OV-23); cores = min(110, bands)
  (per-core value slots cost ~3 us host a core on 32 chips: 0.096 ms at 10 cores, 0.377 at 110).
  Cores: <= MAX_WORKER_CORES (max_grid 11 x 10).
- Gotchas: DS41-292 (copy_init does not reconfigure the unpacker between the fp8 and fp32 CBs: reconfig_data_format_srca);
  DS41-293 (ttnn's host bfp8 packer rounds just-above-tie fp32 values as ties; the device is exact RNE, the tests use a
  float64 RNE reference).
- Model (tt/index_keys.py): one bfloat8_b buffer [1, 1, crows, 128] per (crows, source layer), shared by every
  slot / state (TtServedRows acquires it at the state's build, frees with the last). keys(cache, start) decodes
  [0, hi) (hi = end of the chunk's W blocks) or, after prefetch in the same chunk (epoch moved by begin_chunk; cache
  serial, start, version within one write), only [c wr, hi); a memo reader (re-index layers) reuses it. Rows past hi
  are later positions (causally masked). TtServedRows under served: no slab, no k8; _put encodes the chunk's bf16
  keys (fp8_rows) into the served rows + mirrors only. TtDS41Model.prefetch_index_keys at the first take_engram_rows
  (index_sources(): blocks with an indexer not skipped this chunk). indexer._keys(cache, start).
- Precision: the scored keys are bfp8(fp8(key)) (decode scores the same), not bfp8(key); not bit-identical to private.
- Device tests (8x4): fork tests/unit/test_unpack_fp8_rows.py (exact, 65536 rows a chip, timing);
  tests/test_index_k_served.py (ratio 2 and 1, 6 writes incl. 32 mod 64 starts, full and prefetch + tail paths exact vs
  the host decode on 32 chips; served bytes = the private path's; a reused slot written by another cache: served exact,
  a private copy would score all-zero rows); tests/test_index_k_copy.py still passes with INDEX_K_SOURCE=private.
- Re-run: `scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_index_k_served.py
  ttnn/ttnn/bringup/fp8_rows_ttnn/tests/unit/test_unpack_fp8_rows.py -s`
- Model checks (change on, `orchestrator measure --task P.76`, all rc 0): test_decoder_skip_decode; determinism
  max_abs_diff 0.0 (state 0.0); test_contract; test_skip_positions; rung last: err_final_hidden 0.1271 (limit 0.242;
  P.75's on side 0.1268), final hidden PCC 0.992, top1 / top5 1.0, err L19 0.1232 (0.1231), L39 0.781 (0.794): the
  bfp8(fp8(key)) keys move the metrics within noise.
- Measured (skip_positions, steady, vs P.75's on side, another run): engram_rows section (where the prefetch runs)
  0.71 / 2.88 / 10.84 ms at 51k / 256k / 1M (0.63 / 2.54 / 9.95): the history decode costs +0.08 / +0.34 / +0.89 ms
  device a chunk (3 sources), inside the engram wait. Indexer section 1.38 / 4.86 / 17.58 (1.33 / 4.79 / 17.56):
  the layer-time decode of the chunk's W blocks is within noise. Wall 105.2 / 108.9 / 122.4 ms (109.3 / 108.7 /
  123.4; spread ~10 ms): no change visible. DRAM state 16.09 / 17.08 / 20.74 GiB (same: one user, the shared buffer
  replaces k8 at the same size; saving = k8 x (users - 1)).

## FIX.39 (overseer, owner review 2026-11-18): outside-written history, prefetch marker cleared at chunk start

- (A) tt/history_gather.new_chunk(): every history buffer's ``pre`` (P.75's prefetch marker) set to None; called from
  TtDS41Model.begin_chunk next to index_keys.new_chunk(). Why: gather() reused a prefetched history when (cache id,
  start) matched and ``version`` moved by <= 1, but ``version`` counts this process's writes only: a write from outside
  (the KV Manager after a store load / d2p pull) never bumps it, so a stage that starts mid-model (no prefetch before
  its sources) could reuse rows gathered before that write. The prefetch runs after begin_chunk (first engram wait),
  so the normal path is unchanged (host only; the gate's X.4 units rerun it).
- (B) nothing (FIX.38's units ran on the new default in its own --check).
- (C) tests/test_outside_history.py (one load, DECODER_SKIP=decode via use_decoder_skip): slot states A and B (own
  slabs, the same CacheSlab code as the engine's slots). A runs [0, C); every served cache of A (48 tensors: window
  snapshots, compressed KV, 192-byte index rows) copied into B as a raw DRAM image (ttnn.from_device ->
  ttnn.copy_host_to_device_tensor; no model write path, B's counters stay 0; checked B == A after the copy); then the
  replay chunk [C - 128, 2C - 128) with global_write_start C in A, then in B; every encoder layer output (0-19) and
  every served image compared. Replay: every window truncated at C - 128 in both (item 2, the window ring, not read);
  B told the prefix ids with note_tokens (host only; item 1, the engram tail). Also asserted: no prefetch marker left
  right after any begin_chunk (A).
- Measured (8x4, chunk 5120, novel prompt): shipped outside_history_max_abs_diff 0.0 (no layer, no cache differs; 28
  caches rewritten by chunk k + 1). Documented (recorded, not gated): STATIC_TABLES=on max 242, layers 2-19 differ
  (layer 2 by 1.34; its replicated AttnTable never sees the outside write: item 3); INDEX_K_SOURCE=private max 245,
  layers 2-19 (layer 2 by 0.57; the k8 copy: item 4); B without the prefix ids max 220, layers 1-19 (layer 1 by
  0.46: the engram tail, item 1); [C, 2C) without replay max 229, layers 0-19 (layer 0 by 0.80: B's ring holds no
  carry, item 2). DS41-294.
- STATIC_TABLES / INDEX_K_SOURCE switched for one variant's states on the same load: env + settings.refresh() at state
  build, plus each TtSparseAttention.static and model._hide (host flags), put back after.
- Re-run: `scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_outside_history.py`

## P.77 perf (attempt 1): 4-bit compressed-KV attention history on prefill's own path (HISTORY_FORMAT bf16 | bfp4)

- Expected saving (written before the change; STATIC_TABLES=off, HISTORY_GATHER=rows, HISTORY_GATHER_HIDE=on,
  DECODER_SKIP=decode, 3 gathered ratio-2 sources, 8x4: 32 owners, hs = 80 rows an owner a chunk period). Gathered
  bytes a chunk a source into each chip: 32 x L x 1024 B (bf16) -> 32 x L x 320 B (bfp4 records), L = nb x 80:
  p145408 (nb 30, L 2400) 78.6 -> 24.6 MB; p1024000 (nb 201, L 16080) 527 -> 165 MB. Device a source: the history
  gather (P.75's engram-wait prefetch, ~3.2 ms at 1M, ~0.45 at 142k) -> ~1.0 / ~0.15 ms, plus the per-chip pack of
  its own served rows (slice + tilize + ckv_bfp4 of L rows: ~0.3-0.5 ms at 1M, ~0.1 at 142k): ~-1.8 ms a source at
  1M (~-5 ms a chunk, inside the engram wait: the wall moves only where the wait no longer covers it), ~-0.2 ms a
  source at 142k (~-0.6 a chunk). Layer time: the tail (1-2 periods an owner) + its pack ~0 change; sparse_sdpa
  ROW_BFP4 +3-5 % a call vs BF16 (sdpa README: ~+0.01-0.02 ms a call, ~+0.3 ms a chunk over the 18 compressed
  encoder layers). Host: +2 ops (tilize, ckv_bfp4 prepared) + 1 slice a gather (6 gathers a chunk: ~+1 ms, half of
  it in the engram wait). DRAM a chip: the three per-source history buffers 3.2x smaller (1M: ~3 x 0.5 GiB -> ~3 x
  0.16 GiB, ~-1.1 GiB), + one local record buffer a slab size (R x 320 B, ~5 MB).
- Change (HISTORY_FORMAT=bfp4, the default; bf16 = the P.74 / P.75 path; only with STATIC_TABLES=off,
  HISTORY_GATHER=rows): tt/history_gather.py. The history buffers are [1, 1, owners R, 320] uint8 bfloat4_b row
  records (the format FIX.8's bfp4_rows table used); one more buffer, the local records [1, 1, R, 320] ("loc", R),
  shared by every source of that slab size. Every gather (prefetch, tail, full) first runs `_pack`: the chip's own
  served bf16 rows [lo, hi) -> slab.slice_rows -> cores.to_tile -> ttnn.bringup.ckv_bfp4 (prepared; compact at row0
  0, row0 a value slot for the tail) into the local records; the high_bw_all_gather then moves the records (the full
  path: gathered_dim_size owners x L from the local records, no batch index; the tail: a slice of the local records
  [h, local_rows) -> the small uint8 tail buffer -> slice_write into the buffer's view, as P.75). A whole-history pack
  covers [0, L) rounded up to PACK_BUCKET = 2048 rows (the prepared ckv_bfp4 is one program per row count: a growing
  L would build one each chunk); the extra rows are later slab rows no index reads. The local records take the slab's
  tensor topology (DS41-291). tt/sparse_attention.py: a uint8 history goes to the sdpa fork's ROW_BFP4 with the bf16
  region as kv_window (P.38's bfp4_rows + bf16_tiles call; index [region ids | top-k rows, offset 0], not compacted);
  without a region the [carry | chunk] rows are the kv_window (index tables.index[cls], not reached on the 4 boxes:
  every chip owns >= 128 rows of a 5120 chunk). tt/compressor.TtServedRows.hold_history keeps the format the buffers
  were acquired with (release frees the same keys). No served-format change (decode reads the bf16 rows).
- Ops: no Python-built op added; ckv_bfp4 is prepared (OV-23: prepared=True, compact where row0 is 0). Cores:
  ckv_bfp4 max_grid() (<= 110), slice / tilize sub_cores (non-matmul), the gather 10 a chip (full-mesh torus). No
  fork changed; no fork suite run (OV-25). No padding (OV-3): the tile rounding / bucket packs real slab rows.
- Device check (devjob p77a, tests/test_history_bfp4.py, one load, DECODER_SKIP=decode, p145408 / p1024000, offsets 0
  / 2048, sides hide (bfp4), full (bfp4, no prefetch), bf16): every source's records on every chip byte-exact vs the
  host bfloat4_b rule (ckv_bfp4_ttnn/tests/reference.encode_rows) of the owners' served rows [0, L), both bfp4 sides;
  layer 19 hide vs full bit-identical; hide side 3 prefetches + 3 tails. Layer 19 bfp4 vs bf16 (both device): rel L2
  0.057 at 142k, 0.22 at 1M (random ids; not a gate: the ladder judges vs the reference).
- Per source (synced isolated op, median of 5; p145408 / p1024000): gathered into each chip 24.6 / 164.7 MB (bf16
  78.6 / 526.9 MB); records gather 0.27 / 1.15 ms, pack (slice + tilize + ckv_bfp4, host dispatch included) 0.50-0.57 /
  0.53-0.77 ms, pack + gather 0.62-0.64 / 1.57-2.03 ms vs bf16 gather 0.60 / 3.43-3.47 ms. History buffer a chip
  (a source, the state's size) 23.4 / 157.8 MiB vs 75.0 / 505.0 MiB.
- measure --task P.77 --part test (change on): test_decoder_skip_decode rc 0; test_contract rc 0 (failed 0 / 0,
  unwritten_records_changed 0); test_outside_history rc 0, outside_history_max_abs_diff 0.0 (no layer, no cache
  differs); test_skip_positions rc 0: steady wall 103.9 / 106.5 / 106.8 / 118.5 ms at 51k / 145k / 262k / 1M (P.76's
  on side 105.2 / - / 108.9 / 122.4), engram_rows (where the prefetch runs) 0.52 / 0.92 / 1.41 / 5.06 ms (P.76: 0.71 /
  - / 2.88 / 10.84), attention 20.96 / 21.30 / 21.19 / 21.19 (P.75: 22.59 / - / 22.77 / 22.69), DRAM state 16.03 /
  16.49 / 16.85 / 18.40 GiB (P.76: 16.09 / - / 17.08 / 20.74).
- Re-run: `scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_history_bfp4.py -s`
  (P77_POS, P77_OFF); `python -m models.demos.common.bringup.orchestrator measure --task P.77 --part test` / `--part
  determinism` / `--part last`.
- measure --part determinism: rc 0, determinism_max_abs_diff 0.0 (state 0.0, 4 chunks x 3 runs). --part last (rung
  last, ladder): rc 0, all 103 err metrics within their limits; err_final_hidden 0.1346 (limit 0.2422; P.76's on side
  0.1271), err L19 0.1375 (0.2643; P.76 0.1232), L39 0.684 (1.533; 0.781), final hidden PCC 0.991, top1 / top5 1.0
  (limit 0.994). The 4-bit history moves the stable metrics by about +0.01 (L19 / final), within the limits.

## X.7 fix 1 (2026-11-18): profile at p145408, metrics named with `_nonlast`
- No code change. The previous attempt's run passed (both units rc 0) and wrote results/X.7.json, X.7_profile.json,
  X.7_ops.json.gz, X.7_host_p145408_nonlast.{json,pstats}. Its metrics carry `_p145408_nonlast` because the gate sets
  BRINGUP_CHUNK_LAST=0 (F249, testing/profile.py:417); tasks.yaml X.7 gates `device_ms_total_p145408` /
  `op_rows_p145408`, so the runner reports them MISSING (DS41-296). The fix belongs in tasks.yaml (owner /
  orchestrator): rename both thresholds with `_nonlast`. Not re-run: the same code would record the same names.
- Recorded (steady chunk [145408, 150528), 40 layers): device_ms_total 82.21 (experts 43.78, attention 21.54,
  ffn_residual 4.2, attn_residual 3.5, indexer 3.1, q_a 1.7, compressor 1.4, engram_rows 0.9), op_rows 794,
  profiled wall 147.1 ms, clean wall median 154.8 ms, host cProfile 161.4 ms, host_overhead 72.6 ms,
  timeline gap 143.6 ms, determinism_max_abs_diff 0.0.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate X.7 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check` (after the tasks.yaml rename).

## X.8 perf (attempt 1): chunk-size sweep at ~142k (1536 / 2560 / 5120 / 8192), default tree

- Expected (written before any run; X.7 at p145408 / 5120: device 82.2 ms, of it ~18 ms weights read once a chunk
  whatever its length, clean wall ~106 ms in skip_positions, host share ~24 ms roughly fixed a chunk (launch count
  does not depend on the chunk)). Model: device(C) ~ 18 + 64.2 x C / 5120 ms, wall(C) ~ device(C) + 24 ms.
  1536: device ~37 ms, wall ~61 ms, ~25 tok/ms (-48 % vs 5120); 2560: device ~50, wall ~74, ~35 tok/ms (-28 %);
  5120: device ~82, wall ~106, ~48 tok/ms; 8192: device ~121, wall ~145, ~57 tok/ms (+17-20 %, more if the small-M
  expert tiles gain at 256 rows a chip). Host per chunk ~0 change (same launches), so the host share falls a token
  as the chunk grows. Risk: at 1536 / 2560 a chip owns 48 / 80 rows (< 128 window rows, not tile-aligned), a path
  the 5120 tree never takes on the 4 boxes; at 8192 the L1 / DRAM of the larger chunk's buffers.
- Result: no code change (a measurement). 1536 and 2560 do not load on the 8x4: tt/model.py:1057 asserts
  chunk % (32 x 32 chips) == 0 (48 / 80 rows a chip; the same tile split in attention.py:166 WindowGeometry,
  indexer.py:144, the OWN_ROWS streams, rope / compressor / history_gather); padding each chip's rows to a tile would
  inflate the work (OV-3), so the limit is reported, DS41-297. On 8x4 the chunk is a multiple of 1024; 1536 / 2560
  load on 2x2 / 2x4 / 4x4 (4 / 8 / 16 chips), not on the 32-chip mesh. In their place (not gate units, same test
  by hand under run_safe_pytest): 2048 at p145408 and 3072 at p144384.
- Steady chunk, test_skip_positions (clean wall median, DECODER_SKIP=decode, ~142k history, 8x4):
  | chunk | p | wall ms (spread) | device ms | host ms | tok/s | vs 5120 | experts | attention | attn_res | indexer | ffn_res | q_a |
  | 2048 | 145408 | 70.7 (3.9) | 50.9 | 19.8 | 28,967 | -39 % | 24.0 | 16.5 | 3.9 | 1.7 | 2.3 | 1.2 |
  | 3072 | 144384 | 79.7 (15.5) | 60.3 | 19.4 | 38,545 | -18 % | 31.0 | 18.8 | 3.0 | 2.1 | 2.2 | 1.4 |
  | 5120 | 145408 | 108.3 (8.3) | 80.6 | 27.7 | 47,276 | - | 45.0 | 21.3 | 4.3 | 2.9 | 4.3 | 1.7 |
  | 8192 | 147456 | 150.2 (17.1) | 115.4 | 34.8 | 54,541 | +15.4 % | 65.9 | 28.2 | 6.3 | 5.5 | 4.8 | 2.4 |
- test_profile (plain, BRINGUP_CHUNK_LAST=0), device_ms_total (slowest chip a section, summed): 5120 82.05 ms
  (experts 43.7, attention 21.6, ffn_res 4.2, attn_res 3.5, indexer 3.0, q_a 1.7, compressor 1.4); 8192 118.3 ms
  (experts 65.4, attention 28.5, attn_res 6.2, indexer 5.5, ffn_res 4.8, q_a 2.4, compressor 1.9); chips agree
  within ~1 ms. Its clean wall (153.0 / 243.6 ms, spread 35 / 53, not settled) is noisier than skip_positions'.
- Determinism at 8192: max_abs_diff 0.0 (state 0.0, 4 chunks x 3 runs).
- Vs the expectation: 8192 +15.4 % tok/s (expected +17-20 %, roofline >= 20 %): device a token 15.7 -> 14.1 us
  (-10 %): experts x1.46 and attention x1.32 for x1.6 tokens (the fixed weight read amortizes), but the indexer
  grows x1.87 (scores = queries x history keys, plus top-k over larger rows) and attn_residual x1.47; host a chunk
  27.7 -> 34.8 ms (+7 ms: not fixed, it grows with the chunk). Small chunks lose less than predicted at 2048 / 3072
  (host ~19.5 ms, below 5120's 27.7). Serving chunk unchanged (5120; a larger one is an owner decision with tt-d-gen).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate X.8 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check --unit <unit>` (the 1536 /
  2560 units fail at load, DS41-297); the extra points: `BRINGUP_CHUNK=2048 DS41_DECODER_SKIP=decode SKIP_POS=145408
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_skip_positions.py`
  (3072: SKIP_POS=144384).
