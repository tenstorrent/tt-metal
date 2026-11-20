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
