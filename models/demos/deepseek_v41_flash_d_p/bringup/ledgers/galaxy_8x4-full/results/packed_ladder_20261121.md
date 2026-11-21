# Performance ladder per path at real fixed histories, galaxy 8x4 (2026-11-21, PB.9C)

Branch `dnijemcevic/model_bringup_batching`, model code at 7d755180024 (PB.8RP) + PB.9C's tests (no model change).
Gate run `gate PB.9C --check` (devjob pb9c_gate4): one executed-layers load (layers 0-20, DECODER_SKIP=decode,
BRINGUP_CHUNK=8192, serving slots at 1,048,576), agentic_cc text, the shipped resident engram table, every PB switch of
the gate on. Metrics: results/PB.9C.json.

Method (results/agentx_ladder_20261119.md's): at each history h EVERY segment sits at h, each in its own serving slot;
steady = every segment a mid-request chunk (write_start = start), resumed = every segment a request's first chunk after
a prefix hit (write_start = start + 128). Each timed run first runs the real previous chunk (every segment at h - its
length, untimed). Wall = the engine's device input (made before the clock) -> ids readback -> sync after the last
layer; warm-up until no new program and the walls settle, then 10 timed runs with program-cache misses refused, the
median; device ms = one real-time profiled run (slowest chip); exposed host = wall - device; tok/s = new tokens /
steady wall. Serving wall = the same steady chunk through the serving entry (adapter.prefill_chunk / prefill_packed)
WITH tt-d-gen's per-layer device-op ack (outbound_socket_service_sync into a metadata-only D2HStreamService drained by
a LayerAckService; one ack a layer and segment), warm-up + 5 runs, the median. The harness wall stays the headline.

No memoised history between repeats: every timed chunk makes its engine input anew and gathers every segment's
history again (begin_chunk -> history_gather.new_chunk() drops every prefetch marker each chunk, FIX.39; the untimed
previous chunk at other positions runs in between). Traced history gathers in each profiled timed chunk: 15 (4
segments, h <= 524288), 28 (786432), 32 (1024000); 18 / 33 / 36 with 5 segments. Every case fits DRAM at every history
(fits_<path>_p<h> = 1, all segments' histories resident at once). The segments' order in the chunk rotates by one every
run (no lane / row keeps a slot).

## Single request, 8192-row chunk (test_skip_positions, BRINGUP_CHUNK=8192, SKIP_POS)

No-rebuild (test_no_rebuild: fresh slots / starts, 8 draws, 16 chunks): **PASS** (0 builds after warm-up); ids uploads a chunk 0.

| history | steady wall ms | new tok/s | resumed wall ms | device ms | exposed host ms | serving wall ms (prefill_chunk + D2H ack) | earlier steady wall (tok/s) | earlier source |
|---|---|---|---|---|---|---|---|---|
| 51200 | 99.8 | 82084 | 104.1 | 84.2 | 15.6 | 107.3 | - | - |
| 131072 | 100.3 | 81675 | 106.0 | 86.5 | 13.8 | 105.1 | - | - |
| 139264 | 100.6 | 81431 | 104.8 | 86.5 | 14.1 | 104.7 | 100.4 ms (79.0k) | PB.2 gate, 2026-11-19, 8ef46676e6e |
| 262144 | 104.6 | 78317 | 106.8 | 91.3 | 13.3 | 107.3 | - | - |
| 524288 | 111.0 | 73802 | 116.2 | 99.2 | 11.8 | 113.6 | 108.7 ms (75.4k) | PB.2 gate, 2026-11-19, 8ef46676e6e |
| 786432 | 124.4 | 65852 | 128.5 | 112.5 | 11.9 | 127.9 | - | - |
| 1024000 | 131.7 | 62202 | 135.2 | 118.2 | 13.5 | 136.2 | - | - |

Serving entry's extra at 139264: serving wall 104.7 ms vs harness 100.6 ms (+4.1 ms; over the 7 histories
+2.6 .. +7.5 ms, mean +4.2). Item by item (host split below): ids readback 1.8 + un-shuffle 0.5 + reorder_ids 0.3 +
note_tokens 0.07 = 2.7 ms the harness does not do, less the harness's ids upload 0.4; the 21 D2H acks 4.3 ms of
enqueue, about 1.8 ms of it exposed (the device tail shrinks 22.8 -> 21.0 ms).

## Single request, 5120-row chunk

Placeholder: PB.9C5 runs it (one geometry a gate, F283); the overseer fills this table from PB.9C5's metrics
(test_skip_positions at BRINGUP_CHUNK=5120 records the same columns, its serving wall and its host split at 139264).

| history | steady wall ms | new tok/s | resumed wall ms | device ms | exposed host ms | serving wall ms | earlier steady wall (tok/s), agentx_ladder_20261119 @ 04c14a09d00 |
|---|---|---|---|---|---|---|---|
| 51200 | (PB.9C5) | | | | | | 74.1 ms (69.1k) |
| 131072 | (PB.9C5) | | | | | | 75.2 ms at 128000 (68.1k) |
| 139264 | (PB.9C5) | | | | | | - |
| 262144 | (PB.9C5) | | | | | | 77.2 ms at 256000 (66.3k) |
| 524288 | (PB.9C5) | | | | | | 79.5 ms at 512000 (64.4k) |
| 786432 | (PB.9C5) | | | | | | 84.9 ms at 768000 (60.3k) |
| 1024000 | (PB.9C5) | | | | | | 87.4 ms (58.6k) |

## Fixed 4 x 2048 lanes: lanes4x2k (test_packed_walls, PACKED_ROUTE=lanes, PACKED_GLUED=off; off in serving since PB.9X)

No-rebuild (PACKED_ROUTE=lanes, a 4 x 2k chunk at fresh slots / histories, 3 draws, 6 chunks): **FAIL** (153 builds after warm-up) (ttnn.slice
keyed on slot rows / pick sizes: served.py:162 slice_rows, served.py:218 _pick, history_gather; DS41-346), ids uploads
4.3 a chunk. Recorded, not hidden: this ladder does not count as serving performance.

| history | steady wall ms | new tok/s | resumed wall ms | device ms | exposed host ms | serving wall ms (prefill_packed + D2H ack) | earlier steady wall (tok/s) | earlier source |
|---|---|---|---|---|---|---|---|---|
| 51200 | 110.2 | 74338 | 137.3 | 94.5 | 15.7 | 117.3 | - | - |
| 131072 | 111.0 | 73802 | 138.4 | 99.7 | 11.3 | 117.8 | - | - |
| 139264 | 114.8 | 71359 | 143.2 | 101.9 | 12.9 | 119.5 | 109.2 ms (75.0k) | PB.6L gate (fixed lanes, then the shipped path), 2026-11-20, ecd3a562c49 |
| 262144 | 121.0 | 67702 | 144.9 | 102.1 | 18.9 | 122.1 | - | - |
| 524288 | 133.2 | 61502 | 151.2 | 116.5 | 16.7 | 135.0 | - | - |
| 786432 | 145.3 | 56380 | 162.0 | 129.6 | 15.7 | 146.7 | - | - |
| 1024000 | 155.3 | 52750 | 173.6 | 138.2 | 17.1 | 156.3 | 156.7 ms (52.3k) at 1,028,096-1,040,384 | PB.6L gate, 2026-11-20, ecd3a562c49 |

Serving entry's extra at 139264: 119.5 vs 114.8 ms (+4.7); over the ladder +1.0 .. +7.1 ms (smaller at long
histories, where the lane chunk is more device-bound); 84 D2H acks a chunk (21 layers x 4 segments).

## Glued 4 x 2048: glued4x2k (test_glued_walls, PACKED_ROUTE=glued: the serving path)

No-rebuild (PACKED_ROUTE=glued, fresh slots / histories, 3 draws steady / resumed / steady, 6 chunks): **FAIL** (2 builds after warm-up): 1
program-cache entry + 1 prepare_program @ tt/packed_glued.py:805-822 GluedSend.__call__ on the first draw (histories
385728 / 435200 / 273920 / 19008): the window send's prepared program is keyed on the ordered mix of the segments'
carry sources (a slot's served snapshot or a ring-carry temp), and the warm-up covers only some orders (DS41-360).
So this ladder does not count yet (a ladder counts only when its path's no-rebuild check passes on the same code).

| history | steady wall ms | new tok/s | resumed wall ms | device ms | exposed host ms | serving wall ms (prefill_packed + D2H ack) | earlier steady wall (tok/s) | earlier source |
|---|---|---|---|---|---|---|---|---|
| 51200 | 122.9 | 66656 | 410.2 | 98.5 | 24.4 | 129.4 | - | - |
| 131072 | 122.0 | 67148 | 416.6 | 102.4 | 19.6 | 133.1 | - | - |
| 139264 | 129.4 | 63308 | 419.0 | 104.9 | 24.5 | 136.9 | 124.2 ms (66.0k) | PB.8RP walls (lane header run glued), 2026-11-21, 7d755180024 |
| 262144 | 130.2 | 62919 | 436.9 | 110.7 | 19.5 | 137.2 | - | - |
| 524288 | 145.0 | 56497 | 477.1 | 127.5 | 17.5 | 149.8 | - | - |
| 786432 | 163.6 | 50073 | 517.7 | 144.7 | 18.9 | 174.3 | - | - |
| 1024000 | 178.2 | 45971 | 552.9 | 157.7 | 20.5 | 183.9 | - | - |

Resumed (every segment a prefix-hit first chunk): 410-553 ms, 3.1-3.4x the steady wall, device 395-536 ms. A glued
chunk with any encoder-replay segment is refused by the one-call glued path (tt/packed_glued.glued_call_ok: "an
encoder replay segment") and runs path (b) (PB.7 v1, per-segment moves; its attention and indexer are charged to the
q_a section there: 346 of 408 ms device at 139264). DS41-357. The fixed lanes take replay segments in their one call
(143 ms at 139264).

Serving entry's extra at 139264: 136.9 vs 129.4 ms (+7.5; over the ladder +4.8 .. +11.1 ms; the host
split's own runs +18.6): the D2H acks, 84 ack ops, 16.7 ms of host enqueue. The glued chunk is host-bound (its layers' enqueue alone is 110-128 ms against ~105 ms device), so the acks
land on the critical path; without any ack the serving entry equals the harness (127.4 vs 130.8 ms, inside the spread).

## Glued 5 x 1600: glued5x1600 (test_glued_walls, PACKED_ROUTE=glued)

No-rebuild (fresh slots / histories, 3 draws, 6 chunks): **PASS** (0 builds after warm-up); ids uploads 0.

| history | steady wall ms | new tok/s | resumed wall ms | device ms | exposed host ms | serving wall ms (prefill_packed + D2H ack) | earlier steady wall (tok/s) | earlier source |
|---|---|---|---|---|---|---|---|---|
| 51200 | 137.7 | 58097 | 487.7 | 103.9 | 33.8 | 147.4 | - | - |
| 131072 | 149.9 | 53369 | 495.5 | 108.2 | 41.7 | 145.6 | - | - |
| 139264 | 137.5 | 58182 | 501.7 | 111.8 | 25.7 | 150.5 | 138.7 ms (57.7k) | PB.7C walls, 2026-11-20, ce4e3bc674f |
| 262144 | 143.8 | 55633 | 519.8 | 119.4 | 24.4 | 153.6 | - | - |
| 524288 | 161.4 | 49566 | 570.7 | 142.4 | 19.0 | 167.7 | - | - |
| 786432 | 190.3 | 42039 | 621.1 | 168.3 | 22.0 | 194.7 | - | - |
| 1024000 | 211.1 | 37897 | 666.8 | 190.3 | 20.8 | 221.2 | - | - |

Resumed: 488-667 ms (DS41-357, as glued4x2k). Serving entry's extra at 139264: 150.5 vs 137.5 ms (+13.0);
over the ladder -4.3 .. +13.0 ms (the 131072 harness wall had a slow median, 149.9 ms, spread wide); 105 D2H acks a
chunk.

## The serving entry's extra host, item by item (owner 2026-11-21: PB.MB / MB.2's 5-7 ms)

Every item is the host ms of one wrapped call (perf_counter; a nested wrapped call is taken out of the outer one), the
median of 7 timed chunks, each after its real previous chunk. "device tail" = the sync after the last enqueue (the
device still running). "rest" = wall - every item (the host Python between the wrapped calls: the adapter's asserts,
slot-state lookups, descriptor routing, ack bookkeeping), measured per chunk. *_fresh: every chunk at a position not
run before (serving_fresh also in a slot of its own); *_freshpos: every segment at a history not run before, same
slots. The engram rows (hash, lookup, pack, upload) run on worker threads: their ms overlap the layers; the main
thread's wait for them is inside the layers' item. Columns of one side are medians of their own chunks, so a column's
items need not add to exactly its median wall.

**single 8192 at 139264 (test_skip_positions)**

| item | harness | serving | serving_noack | harness_fresh | serving_fresh |
|---|---|---|---|---|---|
| wall (median of 7) | 99.99 | 103.28 | 104.71 | 99.96 | 102.88 |
| begin_chunk (fields, engram tails + jobs submitted) | 0.66 | 0.59 | 0.60 | 0.62 | 0.58 |
| device tail (sync after the last enqueue) | 22.79 | 20.99 | 22.80 | 20.53 | 21.60 |
| embed (enqueue) | 0.44 | 0.35 | 0.35 | 0.45 | 0.35 |
| ids upload (from_torch) | 0.43 | - | - | 0.45 | - |
| layers (host enqueue, engram waits included) | 75.31 | 74.25 | 77.86 | 77.21 | 73.61 |
| rest: host Python between the wrapped calls (wall - the items) | 0.46 | 0.58 | 0.51 | 0.46 | 0.58 |
| D2H ack ops (enqueue) | - | 4.27 | - | - | 4.15 |
| ids readback (to_torch, mesh column 0's SP rows) | - | 1.83 | 1.90 | - | 1.78 |
| ids un-shuffle (server order -> natural, host) | - | 0.52 | 0.59 | - | 0.55 |
| note_tokens (slot id history + engram tails) | - | 0.07 | 0.07 | - | 0.07 |
| reorder_ids (natural-order ids on the device: enqueue) | - | 0.30 | 0.31 | - | 0.30 |
| engram worker jobs (off the main thread: hash, lookup, pack, upload) | 23.07 | 22.77 | 25.43 | 22.53 | 25.15 |
| of the layers: main thread waiting for engram jobs | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

Single 8192, explained: the serving entry does four things the harness does not: the ids readback (8 to_torch of mesh
column 0's SP rows, 1.8 ms), the un-shuffle to natural order (0.5), note_tokens (0.07) and the device reorder of the
embedding ids (0.3 enqueue), together 2.7 ms; it skips the harness's host ids upload (0.4). Net +2.3 ms of host before
the first layer, then the 21 D2H acks (4.3 ms of enqueue, ~0.2 ms each), about half hidden behind the device tail.
Fresh slots / positions add nothing measurable (serving_fresh 102.9 vs serving 103.3 ms; 0 builds). The layers'
enqueue varies +-3 ms run to run, which bounds the precision of these walls. PB.MB's "5-7 ms" compared the adapter's
fresh walls with PB.9F's run_chunk walls (another harness); against this ladder's harness, at fixed histories, the
serving entry costs +2.6 .. +7.5 ms (mean +4.2): ~2.3 ms of ids handling plus the exposed part of the acks.

**glued 4 x 2k at 139264 (test_glued_walls)**

| item | harness | serving | serving_noack | harness_freshpos | serving_freshpos |
|---|---|---|---|---|---|
| wall (median of 7) | 130.79 | 149.37 | 127.44 | 122.91 | 131.41 |
| begin_packed (per segment note_tokens + begin_chunk fields, engram jobs) | 1.51 | 1.64 | 1.64 | 1.63 | 1.64 |
| device tail (sync after the last enqueue) | 6.16 | 0.32 | 6.27 | 6.38 | 0.43 |
| embed (enqueue) | 0.39 | 0.32 | 0.34 | 0.39 | 0.38 |
| end_packed | 0.06 | 0.07 | 0.06 | 0.06 | 0.06 |
| glued_dev_ids (embedding ids reordered on the device: enqueue) | 0.34 | 0.28 | 0.29 | 0.31 | 0.32 |
| ids readback (from_device / to_torch, mesh column 0's SP rows) | 0.42 | 0.45 | 0.43 | 0.48 | 0.50 |
| ids un-shuffle (engine rows -> model rows, host) | 0.27 | 0.27 | 0.28 | 0.27 | 0.27 |
| layers 0-20 (host enqueue, engram waits included) | 122.25 | 127.86 | 117.43 | 112.24 | 110.47 |
| rest: host Python between the wrapped calls (wall - the items) | 0.55 | 0.82 | 0.55 | 0.53 | 0.81 |
| D2H ack ops (enqueue) | - | 16.67 | - | - | 16.57 |
| engram worker jobs (off the main thread: hash, lookup, pack, upload) | 30.85 | 31.13 | 30.45 | 27.17 | 91.69 |
| of the layers: main thread waiting for engram jobs | 0.01 | 0.02 | 0.02 | 0.02 | 28.31 |

Glued 4 x 2k, explained: the packed harness reads the engine input too, so the ids path (one batched readback 0.4-0.5
ms, un-shuffle 0.3, device ids 0.3) is the same on both sides. The serving entry's extra is the D2H acks alone: 84 ack
ops, 16.6-16.7 ms of host enqueue, on the critical path because the glued chunk is host-bound (the device tail falls
6.2 -> 0.3 ms: the device waits for the host). Without acks it equals the harness. The ack's cost was also measured
alone (tests/perf/test_d2h_ack_probe.py: 0.16-0.24 ms of host enqueue an ack, 315 acks, every record drained).
DS41-358. (serving_freshpos' engram row: one chunk's worker jobs were slow, 91.7 ms with 28.3 ms waited, inside a
wall still below the repeated side's; not reproduced in the other sides.)

## For the owner

1. No-rebuild per path, same code as the ladders: single8192 PASS, glued5x1600 PASS, glued4x2k FAIL (1 build + 1
   prepare on a fresh carry-source mix, DS41-360), lanes4x2k FAIL (153 builds; off in serving). So only the single 8192
   and glued 5 x 1.6k ladders count today.
2. Resumed chunks on the glued path (the serving path for every packed chunk) are 3.1-3.4x slower than steady: any
   encoder-replay segment sends the whole chunk to path (b) (DS41-357). Every prefix-hit request's first chunk is one.
3. The D2H ack costs ~0.2 ms of host a layer and segment: 4.3 ms a single 8192 chunk (about half hidden), 16.7 ms a
   4-segment glued chunk (host-bound: serving +4.8 .. +11.1 ms over the harness on the ladder, +18.6 in the split)
   (DS41-358).
4. Steady glued 4 x 2k vs the fixed lanes at the same histories: 122-178 vs 110-155 ms (glued +9 .. +23 ms, its host
   enqueue); single 8192: 100-132 ms (62-82k new tok/s), the fastest per token at every history.
