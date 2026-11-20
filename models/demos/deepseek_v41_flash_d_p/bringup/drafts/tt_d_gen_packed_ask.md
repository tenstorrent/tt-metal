# DeepSeek-V4.1-Flash prefill: packed chunks, what we need from tt-d-gen (draft, 2026-11-19)

Context: most agentic turns add few new tokens (Claude Code traces: p50 1.6k new tokens over p50 139k of cached
history). Today each turn pays a whole prefill chunk, so a 1.6k turn runs a mostly empty chunk. The prefill model
(tt-metal `dnijemcevic/model_bringup_batching`, galaxy 8x4) is adding **packed chunks**: one 8192-row chunk carries up
to **4 requests ("segments") of at most 2048 new tokens each**, each continuing its own slot. Projected: 4 x 2k over
~139k in ~104-120 ms a chunk (68-79k new tok/s) vs ~4 x 75 ms today. Design: `bringup/PACKED_DESIGN.md`.

Nothing changes for a request sent alone: a chunk with one segment is today's chunk, bit for bit.

## What stays as it is

- The served caches (formats, layouts, addresses, the record rule), the KV chunk table, migration, the window
  snapshot at decode's cap, PREFILL_END=cap, global_write_start (ask 1 of 2026-11-18), DECODER_SKIP=decode's acks of
  every served layer.
- Each segment reads and writes only its own slot. The model never keeps per-slot state across chunks (every chunk
  gathers each segment's history from its slot's served cache), never assumes the next chunk, and never copies between
  slots. Slots of a chunk are arbitrary and may change every chunk.

## What we need

### Ask P1: a chunk header with up to 4 segments

- Header: `{n_segments, segments[4]}`, each segment `{slot_id, actual_start, actual_end, global_write_start}` (the 4
  fields of today's header; 16 B each, 68 B in all). Segment i occupies **lane i = rows [2048 i, 2048 i + 2048)** of
  the 8192-row chunk; a shorter segment is padded inside its lane (PAD_ID, as today's tail); a lane may be empty
  (`slot_id = -1` or `n_segments < 4` with lanes filled from 0).
- Rules (as today, per segment): `actual_start` 32-aligned; `actual_end - actual_start <= 2048`; one slot at most once
  a chunk; `global_write_start` missing = `actual_start`.
- A request longer than 2048 new tokens keeps today's path: a chunk with ONE segment over all 8192 rows
  (`n_segments = 1`, `actual_end - actual_start <= 8192`). The scheduler decides; the model accepts both.
- `prefill_chunk` keeps its signature for one segment; packed chunks come through a new `prefill_packed(input,
  kv_cache, segments, ...)` (or `prefill_chunk(..., segments=[...])`).

### Ask P2: the device ids of a packed chunk

- The same uint32 device tensor as today (`[SP, 1, 8192 / SP]`, sharded over mesh rows): lane i's ids on lane i's mesh
  rows (8x4: rows 2i, 2i + 1).
- Inside a lane, **one of** (the header says which; the model handles both):
  - (a) the per-segment ring_sdpa_reshuffle order: today's server order for that segment's `actual_start`, over the
    lane's own mesh rows (a ring of 2 rows on 8x4), or
  - (b) natural order (row d of the lane = position `actual_start + d`): no reshuffle, simplest for the engine.
  The model un-shuffles each lane on the device from load-time tables; no host work.
- With ask 2 of 2026-11-18 (host ids `[start - 3, end)` per chunk): per segment, the same (its 3 ids before
  `actual_start` and its new ids). That also removes the per-chunk ids read-back (~1.5-2 ms) for packed chunks.

### Ask P3: per-segment acks

- Each served layer is acked **once per segment**: `layers_completed(g, g + 1, slot_id, actual_start, actual_end)` with
  that segment's fields, in lane order, after the layer's writes for every segment of the chunk are in DRAM (one wait
  serves the 4 acks). D2H acks: one `outbound_socket_service_sync` per layer with the 4 segments' metadata (or 4
  calls, one per segment: say which the KV Manager prefers).
- A segment whose computation ends below decode's cap of its `actual_end` (PREFILL_END=cap: `cap <= actual_start`)
  computes nothing but still gets its snapshot rewrite and its acks, as today's `_acks_only`.
- Migration: each segment's KV is in its own slot's records; the KV chunk table is unchanged.

### Ask P4: the scheduler

- Pack pending prefill segments (turns of <= 2048 new tokens, or the last <= 2048 tokens of a longer turn) up to 4 a
  chunk; a request in flight keeps one slot; any slot may be in any lane in any chunk.
- Resident slots: 4 per packed chunk (the engine's `num_users` already allows this).

## Test we can run together

- Model side, today: `tests/packed/test_packed_cpu.py` (descriptor, lanes, the CPU reference packed vs alone, bit for
  bit); next: `tests/packed/test_packed_oracle.py` (on the device: a segment packed with 3 others == alone, every layer
  and every served byte) and `tests/packed/test_packed_walls.py` (chunks driven with the header above and device ids in
  the engine layout).
- With tt-d-gen: the same oracle driven through the engine's real packed header: served bytes of each slot identical
  to the slot's request sent alone.

## What PB.3 (the first device path, 2026-11-19) learned

- Built and tested: `DS41PrefillRuntime.prefill_packed(input, kv_cache, descriptor)` (tt/runners/adapter.py), the
  descriptor = `tt/packed.ChunkDescriptor` (4 x `Segment(slot, start, end, write_start)`), the engine input in
  **lane-natural order** (P2 (b)); the reshuffle order (P2 (a)) is not built yet. If tt-d-gen can send (b), nothing
  else is needed for the input.
- Acks: the model acks every served layer once per segment (`sink.layers_completed(g, g + 1, slot, start, end)` per
  segment after the layer's sync; D2H: one `outbound_socket_service_sync` per segment, `metadata_msg` may be a list,
  one per segment). An ack naming the 4 segments at once (P3) would save 3 host calls a layer.
- A segment whose decode cap is at or below its start (PREFILL_END=cap) computes nothing, as today's `_acks_only`
  chunk: its lane is pad, its snapshot is rewritten at cap and it is acked.
- One readback of the input a chunk (the engram host step's ids), as today; P4 (host ids in the header) would remove
  it for every segment at once.

## Revision 2026-11-20 (PB.7): GLUED chunks, up to 6 variable-length segments (replaces the 4 x 2048 lanes of P1 / P2)

Why: fixed 2048-row lanes pad real AgentX turns (at p50 ~78 % of a lane chunk's rows are real) and cap a segment at
2048. Glued: segments back to back, any lengths, only the chunk's tail may be padding.

### Ask G1: the header (replaces P1)

- `{n_segments (1..6), segments[6]}`, each `{slot_id, actual_start, actual_end, global_write_start}` (16 B each,
  100 B in all), in row order. Segment i's ids occupy chunk rows `[row_i, row_i + (actual_end_i - actual_start_i))`,
  `row_0 = 0`, `row_{i+1} = row_i + actual_end_i - actual_start_i`: **back to back, no alignment, no padding between
  segments**; rows past the last segment are PAD_ID. `sum(actual_end - actual_start) <= 8192`.
- Rules per segment (as today): `actual_start` on a kv block (64; 32-aligned at least); one slot at most once a chunk;
  `global_write_start` missing = `actual_start`. No per-segment length cap other than the chunk.
- A request that does not fit the rows left is **split**: its first part ends at the last kv-block boundary that fits
  (`actual_end = (actual_start + rows_left) // 64 * 64`, so the part is a non-last chunk of the request: computed in
  full, cap = end) and the rest is a segment of a later chunk starting there (same slot, same `global_write_start`), as
  today's chunked prefill continues a long prompt. <= 63 tail rows of the chunk may stay pad. The model does not
  know the next chunk and keeps no state between the two parts beyond the slot's served caches.
- One segment over the whole chunk is today's chunk, bit for bit (the model's oracle checks it).

### Ask G2: the device ids (replaces P2)

- The same uint32 `[SP, 1, 8192 / SP]` tensor (sharded over mesh rows) in **natural order of the glued rows** (row r of
  the chunk = mesh row r // (8192 / SP), column r % (8192 / SP)): segment i's ids at rows `[row_i, row_i + len_i)`.
  No reshuffle. The model reads the header and lays the computed rows out itself (each segment's `[actual_start, cap)`
  back to back: decode's cap drops `[cap, actual_end)`, which today's chunk never computes either).
- With ask 2 of 2026-11-18 (host ids per segment `[start - 3, end)`): the same per segment; removes the ids readback.

### Ask G3: acks (as P3)

- Each served layer acked once per segment in row order, after the layer's writes of every segment are in DRAM.
  A segment with `cap <= actual_start` computes nothing and still gets its snapshot rewrite and its acks.

### Ask G4: the scheduler

- Fill a chunk in arrival order (or first-fit) with up to 6 pending segments back to back; split the one that does
  not fit (G1). Resident slots: 6 per chunk (+ the split request's slot, which stays in flight).
- DRAM (model side, measured `glued_dram_headroom_gb_6x_p1m`): six slots' history buffers at ~1M resident at once.

### Model side today (PB.7)

- `tt/packed.GluedChunk` (descriptor, engine rows vs model rows, `fill` / `split`), `prefill_packed(input, kv_cache,
  GluedChunk)` with `PACKED_GLUED=on` (call-time switch; off = the 4-lane path of P1 / P2).
- A glued chunk whose segments sit at rows k x 2048 (e.g. 4 x 2048) runs the lane path (one attention call a layer);
  any other glued chunk runs each segment's attention after moving its rows to row 0 (v1; one attention call per
  segment). v2 (per-row segment arguments in the attention forks) is the model's next step; nothing in G1-G4 changes.
- Tests: `tests/packed/test_glued_cpu.py`, `test_glued_oracle.py` (a segment's outputs and slot writes identical
  whatever it is glued with and wherever it lands, a split request == the same request chunked alone),
  `test_glued_reference.py`, `test_glued_walls.py`.
