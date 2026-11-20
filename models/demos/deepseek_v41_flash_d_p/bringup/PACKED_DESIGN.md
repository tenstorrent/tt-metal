# Packed prefill: design and interfaces (PB.1, owner 2026-11-19)

One prefill chunk of **8192 rows** carries **up to 4 requests (segments)**, each **<= 2048 new tokens**, each
continuing its own slot's cached history. Token-wise work runs once over the 8192 rows; attention is per segment.
Switch: `PACKED=off|on` (tt/settings.py, call-time, in `bringup/hooks.py batch_call_settings`); off = today.

Targets (owner, binding): agentic text, steady, serving-equivalent: **4 x 2k over ~139k each >= 67k new tok/s
(<= ~122 ms a chunk)**; **mixed 61k / 139k / 308k / 547k >= 64k (<= ~128 ms)**. No shortcuts: every chunk each
segment's history is gathered from its own slot's served cache in tt-d-gen's layout; slots are arbitrary and change
every chunk; all segments' history buffers resident at once within the DRAM budget at 1M; inputs as tt-d-gen delivers
them; the model does not know the next chunk.

Status of this file: PB.1 (CPU only). Built here: `tt/packed.py` (descriptor + index math), `reference/packed_ref.py`
(the packed forward on the CPU reference), `tests/packed/test_packed_cpu.py` (the CPU rung, gated),
`tests/packed/test_chunk8192_reference.py` (PB.2's accuracy test, not run), `drafts/tt_d_gen_packed_ask.md`.
PB.3 (v1 on the device, gate passed): `tt/packed_device.py` + `TtDS41Model.begin_packed / layer_packed`, the adapter's
`prefill_packed`, tests `test_packed_oracle.py` / `test_packed_reference.py` / `test_packed_walls.py`. v1 deviates from
s5 v1 in one point (BREADCRUMBS PPB.3, DECIDED): today's attention steps are reused unchanged by moving each lane >= 1
to lane 0's chips and back (the code computes positions as start + q R + d), so each segment's call covers the full
8192-row geometry; engine input lane-natural (s2's alternative). Measured walls in BREADCRUMBS PPB.3 (4 x 2k @ 139k:
294.5 ms with the resident engram table): v2 (s5) is what reaches the bar.
PB.3A (v2 attention, PACKED_ATTN_ONECALL=on, `tt/packed_onecall.py`): the attention steps ONCE a layer over every
lane, each chip at its lane's positions / slot caches / history (no moves): per-lane rope rows (`rope.LaneStarts`),
the window region per lane (per-chip keep mask, region index lists per start-class tuple, one prepared lane send:
neighbour sends inside a lane, a lane's first chip reading its carry in place from the slot's served snapshot), the
compressor once with every lane's served writes through lane-offset server-order rows, sparse_sdpa once (sdpa fork
`kv_lanes`: each chip its lane's history gather buffer), the indexer as ONE ring (indexer fork lanes: up to 4 key
buffers through one all-gather, each chip its lane's start / kv_len / keys) + one top-k with a per-chip
valid_length_tensor (per-segment rings when the slots' key capacities differ, DS41-328, or a candidate pool is short).
Oracle bit-identical (alone vs today, lane equivalence, digests, determinism); walls (resident table): 4 x 2k @ 139k
122.3 ms = 67.0k new tok/s; mixed (slots sized by position: per-segment rings) 150.8 ms = 54.3k. Not yet: the
lane-targeted gather (each lane's chips receive only their slot's rows; PB.3H), pad rows out of the MoE dispatch
(the block's padded-forward mode does not describe lanes as whole SP rows: PB.4), the lanes ring over unequal key
capacities.
PB.4 (PACKED_MOE=on, `tt/packed_device.MoEPadSpan`): pad rows out of the MoE dispatch without a fork change: the
head's padded-chunk words are per device, (0, r W + n_r), so SP row r has its n_r real rows as a prefix
(`ChunkDescriptor.moe_real_prefix`; exact where lanes are whole SP rows, 8x4 / 4x4; on 2x4 / 2x2 a superset: pad rows
between two lanes of one SP row stay routed). Verified lane-equivalent by construction on the block alone
(`tests/packed/test_packed_moe_block.py`: permuted / different / hot companions, other lanes, pad lanes routed or
not: A's rows bit-identical) and by the oracle with PACKED_MOE=on.

Contents: 1 geometry and the key fact, 2 the descriptor, 3 per-chunk quantities, 4 non-row-local ops, 5 attention and
the cross-chip pieces, 6 pipelining and hiding per-segment costs, 7 DRAM at 1M, 8 8192 on today's tree, 9 test plan,
10 projected time, 11 owner decisions, 12 PB.2-PB.4 estimates.

## 1. Geometry, and why lanes are whole chips

The model's rows are natural blocks (tt/attention.WindowGeometry.query_offsets): chip q = r TP + c (chip-linear)
holds chunk rows `[q R, (q + 1) R)`, R = chunk / (SP TP). At 8192:

| box | mesh | R (rows a chip) | chips a 2048 lane | lane i = chips |
|---|---|---|---|---|
| galaxy_8x4 (deploy) | 8x4 | 256 | 8 (2 SP rows) | 8i .. 8i + 7 |
| galaxy_4x4 | 4x4 | 512 | 4 (1 SP row) | 4i .. 4i + 3 |
| loudbox_2x4 | 2x4 | 1024 | 2 | 2i, 2i + 1 |
| quietbox_2x2 | 2x2 | 2048 | 1 | i |

So **every lane is a fixed set of whole chips on every box** (`ChunkDescriptor.lane_chips`, tested on all four), and
every per-chip quantity of today's chunk (start, carry, causal offsets, which history a chip's queries read) becomes a
per-chip quantity fixed by the chip's lane. That turns most of "attention per segment" into "attention once, with
per-chip arguments": each chip's queries belong to exactly one segment. Lanes are 2048-aligned and 1024-aligned
(DS41-297: 8192 % (32 x 32) == 0), ratio-2 groups never straddle a lane, and a 32-aligned segment start never leaves a
group pending (compressor docstring; `compress_groups` carry_in == 0 for ratios 1 and 2).

## 2. The chunk descriptor

`tt/packed.py`:

```
Segment(slot, start, end, write_start=None)     # one request: the engine's per-request header fields
    start        actual_start, 32-aligned (lane row 0 = position start)
    end          actual_end (engine), start < end <= start + lane rows
    write_start  global_write_start (P.51; None = start): no history row below it; > start = encoder replay
ChunkDescriptor(chunk, segments)                 # segments[i] in lane i, None = empty lane
    .single(chunk, slot, start, end, gws)        # TODAY's header: 1 lane of `chunk` rows
    .packed(chunk, [seg | None, ...])            # v1: 4 lanes of chunk / 4
```

- **Today's header maps onto it** as `ChunkDescriptor.single`: one lane covering the chunk; every function returns
  today's values (positions start + d, valid rows [0, cap - start), writes [gws, cap)); tested
  (`test_single_header_is_todays_chunk`). PACKED=on with one segment is today's chunk; `PACKED=on` with one 2k segment
  in lane 0 is today's chunk with valid rows [0, len) (bit-identical by construction where every op is row-local:
  the oracle's `packed_alone_vs_today`).
- Per row (all host tensors here, load-time tables + 4 integers a lane on the device): `row_positions` (pad rows continue
  their lane, as today's pad rows past actual_end), `row_lane`, `row_slot`, `row_valid(prefill_end)` ([start, cap) of
  each segment), `row_writes` ([max(start, gws), cap)), `causal_mask` (block-diagonal: no row reads another lane),
  `pack_ids` / `unpack_rows`.
- Per chip: `lane_chips`, `carry_sources` (a lane's first chip takes its window carry from its OWN slot; every other chip
  from its neighbour, as today), `history_slots` (whose history a chip's queries read; -1 for an empty lane).
- Per segment: `write_range`, `compress_groups` (carry_in / carry_out of a group straddling the start / end, for any
  ratio: 0 for this checkpoint's 1 and 2 at 32-aligned starts; a ratio 4 / 128 compressor would carry start % r
  tokens from the slot, the reference `State.partial` path), `window_carry` (positions [start - 128, start), start
  class min(start, 128), a replay truncates: class 0), `engram_tail` (the 3 ids before start from the slot's ids),
  `snapshot_range` (the served 128-row window snapshot at cap(actual_end)).
- **How the harness drives it** (PB.3): the bring-up harness builds a `ChunkDescriptor` per chunk and calls
  `adapter.prefill_packed(input_tensor, kv_cache, descriptor)` (new; `prefill_chunk` = `prefill_packed` with
  `ChunkDescriptor.single`), the device ids in the engine's layout (below), PAD_ID in pad rows. The walls test drives
  exactly what tt-d-gen would send (section 9).
- **The engine's device input** (today: uint32 `[SP, 1, W]`, server order of ONE start, `WindowGeometry.server_rows`):
  packed, lane i's rows go to lane i's SP rows; within a lane the order is the per-segment ring_sdpa_reshuffle order of
  that segment's start over the lane's own SP rows (8x4: a 2-row sub-ring). The model un-shuffles each lane with a
  load-time table indexed by (start mod lane W) / 32, as today's `WindowTables.srv` (one table a box). The draft ask
  proposes lane-natural order (no reshuffle) as the simpler alternative; the model takes either (a flag in the header).
- **Descriptor on the device**: 4 x {slot, start, end, gws} = 64 B. Ops whose per-chip value depends on a lane take it as
  a per-chip runtime arg (a mesh workload's per-device args; our prepared programs and forks do this already); stock ops
  that take a tensor (rotary_embedding_indexed's positions) read a per-chip position row made on the device from a
  load-time iota + the lane's start (one tiny add a chunk, shared by every layer). No per-prompt precompute, no read
  back: the descriptor is what the header already carries.

## 3. Every per-chunk quantity: today, per segment, oracle

"oracle" names the test (section 9) that covers it: CPU = test_packed_cpu.py (built), OR = test_packed_oracle.py,
REF = test_packed_reference.py, W = test_packed_walls.py.

| quantity | today (code) | packed plan | oracle |
|---|---|---|---|
| positions / RoPE | `rope.rows(start, S)` slices the load-time cos/sin table at start (tt/rope.py RopeTable; kv `from_proj(p, start)`, attention `_rope_rows`, compressor `compress_rope.rows(g0, n)`, indexer q) | per-row positions = per-chip load-time iota (offset in lane) + lane start: one position row a chunk on the device; RoPE via rotary_embedding_indexed (ROPE_IMPL=on already takes indices) or a per-chip slice offset arg | CPU (row_positions; reference kv with per-row positions == forward_chunk), OR (lane equivalence), REF |
| slot id | `adapter.state(kv, slot)`: the active DS41DeviceState bound to the slot's slabs (`model.state = st`) | 4 active states: `model.states[lane]`; every slab access by the lane's state | OR (slot digests), W (slots reassigned every chunk) |
| write_start (P.51) | `begin_chunk(write_start=)` -> `_write_from`; compressor `write_from(lo)`, kv_only skip | per segment (`write_range`); `kv_only` runs when ANY lane writes; each lane's writes clipped to its own [gws, cap) | CPU (row_writes, write_range), OR (replay segment case) |
| PREFILL_END / cap | `adapter._run`: run_end = cap(actual_end) (P.49); pad past cap; `_acks_only` when cap <= start | per segment: `run_end` per lane; a segment with cap <= start computes nothing (its lane is pad) but still gets its snapshot rewrite and its acks | CPU (row_valid cap), OR (a cap <= start segment) |
| engram 3-id tail | host: `st.tail[i]` per slot (note_tokens from the slot's id history); device path (P.87): the slot's device id history + explicit tail | per segment: hash each lane's ids with ITS slot's tail (C++ fill job over the 4 lanes in one GIL release, pad rows zero); needs ids [start - 3, start) per segment (tt-d-gen ask 2; today from the slot's id history) | CPU (engram_tail; reference engram_rows per segment == alone), OR |
| window snapshot / ring | `TtWindowCache`: ring + served 128-row snapshot at cap (P.42 / P.65 snapshot_send); `_window_policy` truncation; `rtail` (the last chip's kept tail = next chunk's carry if contiguous) | per segment: the snapshot of slot k at cap(end_k) (`snapshot_range`); the kept tail never reused across chunks unless the same slot and contiguous (slots change every chunk: read the carry from the slot's served snapshot, the `carry_owner` path) | CPU (window_carry), OR (slot digests: snapshot bytes), W |
| window carry at a segment start | chip 0's carry = kept tail or served rows (`_window_region` step 3); others by the neighbour send (tt/window_send.py) | lane-first chips: the slot's carry (served snapshot rows [start - 128, start)); inner chips: neighbour send; the send suppressed across a lane boundary (per-chip flag in the prepared program) | CPU (carry_sources), OR |
| start class (index lists) | `tables.region_index[cls]`, cls = min(start, 128), one class a chunk | per lane-first chip its own class (inner chips 128): the region index composed per chunk from the per-class load-time tables (a per-chip select, once a chunk for all window layers) | CPU, OR (a fresh segment next to continuing ones) |
| compressor carry (ratio 4 / 128) | none here: ratios 1 and 2, 32-aligned starts (compressor docstring) | `compress_groups`: carry_in = start % r from the slot's state, carry_out = run_end % r into it (only for a ratio > 32) | CPU (all ratios), OR (an odd-length segment leaves a ratio-2 group pending: CPU covers it) |
| served-cache writes | compressed KV / index keys: `TtServedRows.write` (update_padded_kv_cache, server-order rows, metadata path); window rows `cache.write` / `write_local` | per segment into its slot's slab, positions [gws, cap) of its lane's rows; batch the 4 writes into one prepared program (per-lane slot / start / valid args) | OR (slot digests vs alone), W |
| history gather (P.74 / P.75 / P.77) | `history_gather.gather / prefetch`: the source's served history of the ACTIVE slot [0, c S) over the mesh into a persistent buffer, at the first engram wait | per segment: each slot's history gathered every chunk (owner: no persistence); placement and hiding in sections 5-6 | OR, W (gather exposed / hidden) |
| index keys (P.76) | `index_keys.prefetch / keys`: the slot's served fp8 rows decoded into a bfp8 buffer | per segment (4 key buffers), decoded at the prefetch point | OR |
| ids readback | `adapter._host_ids`: one read of the engine input a chunk (~1.5-2 ms) for the engram host step | one read a chunk (the whole input, 4 lanes); gone with tt-d-gen ask 2 (host ids) | W (serving-equivalent wall includes it) |
| decoder skip / layer 20 | DECODER_SKIP=decode: layers 0-19, layer 20 `kv_only`, 21-39 skipped; every layer acked | unchanged per chunk; layer 20's kv_only writes each lane's [gws, cap) | OR, REF |
| prefetch (P.75 / P.76) | `take_engram_rows` -> `prefetch_history` / `prefetch_index_keys` of the active state | the 4 states' prefetches, ordered by need (source 2 first); section 6 | W |
| acks | `sink.layers_completed(g, g + 1, slot, start, end)` per layer / `outbound_socket_service_sync` | one ack a layer per segment (or one ack naming the 4 segments: ask) | W, contract tests |
| cands / top-k stores | `st.put(st.topk, i, start, top)`, `st.get` by (layer, start) | per lane state | OR |
| memo (INDEXER_GLUE) | `st.memo` per state, cleared every begin_chunk | per lane state | OR |

## 4. Every op that is not row-local

| op (where) | today | packed plan | oracle |
|---|---|---|---|
| sparse_sdpa / window sdpa (attention.py `_sdpa`, sparse_attention.py) | per query over [carry 128 | own rows] + top-k history rows | per chip, its lane's table (section 5); no row reads another lane (`causal_mask`) | OR lane equivalence |
| window carry / neighbour send (window_send.py) | chip q's last 128 rows -> chip q + 1 | not across a lane boundary (section 3) | OR |
| indexer score (ttnn.bringup.ring_indexer_score_dsa) | ring over SP of the slot's key shares; per query causal lens | per chip its lane's keys and causal offsets (section 5) | OR |
| top-k (topk_large_indices, candidates) | per query row; ties split by core count (DS41-286) | per row; width = the lane's key count, -inf past it; core count unchanged (no sub-device change of compute cores) | OR, determinism |
| compressor groups (ratio-2 pairs) | pairs of rows, never straddle a chip / a 32-aligned start | lanes are 2048-aligned: never straddle a lane | CPU |
| MoE planning / dispatch / combine (moe_block, fused head, TtK3TokenSplitMoE) | combine snap 2^-19: order-free sums (MOE_BLOCK.md D7); planner deterministic | **PB.4 first check**: a token's result must not depend on per-expert counts (arena capacity / overflow paths, split rows, program config by load). Pad rows: excluded from dispatch (weight 0 / no expert) so padded lanes cost no expert work and do not move the planner | OR (MoE block alone with permuted / companion rows), determinism |
| routed / shared expert matmuls | per-row results independent of M on Tensix (a tile row only meets its own A row); bfp8 / bfp4 shared exponents are per 16 values of ONE row | same | OR |
| router top-6, softplus | per row; fp32 partials in a fixed order | same | OR |
| fabric_reduce_scatter (shared expert), TP sums | fixed topology order | same (no dependence on row counts) | determinism |
| rms_norm / mHC stats / engram gate | per row | same | OR |
| bfp8 activations, bfp4 history records | shared exponent per 16 values of one row (bfp8_b), a record per row (ckv_bfp4) | same | OR |
| embedding / engram lookup | per row | per row; pad rows zero rows | CPU |
| history gather | the slot's rows | per slot (section 5) | OR digests |

Measured on the CPU rung (DS41-321): the CPU reference itself is NOT composition-independent in its routed experts:
its grouped SwiGLU runs clamp / silu / multiply on the whole [group rows, I] tensor, and torch's multithreaded
elementwise kernels give a row a 1-ulp different result depending on its place in that tensor (layer 3, expert 60:
134 rows vs 1, h 4.7e-10, experts_out 3e-8). The CPU oracle therefore runs the routed SwiGLU row by row on every side
(`packed_ref.fixed_expert_tiles`); then packed == alone == today's single header, bit for bit. The device's question
(does a token's result depend on its companions?) is PB.4's, on the device MoE.

## 5. Attention and the cross-chip pieces

Per layer the attention needs, for each chip, (a) its queries (local), (b) the window region [carry | own rows] (local
but for the carry), (c) the history table of its segment's slot (gathered over the mesh), (d) the index keys of its
slot (indexer: ring over SP), (e) top-k (local). Only (c) and (d) cross chips.

**v1 (PB.3, the brief's "attention per segment in a loop")**, built for correctness first, measured, then replaced
where the walls demand it:
- (c) per segment k: today's stock gather (high_bw_all_gather of slot k's slab, `history_gather.prefetch`) into ITS OWN
  buffer (one per (source, lane): the owner's rule "all segments' buffers resident at once"); sparse_sdpa called once
  per segment with that table; only lane k's rows of its output are kept (`causal_mask` makes the other rows pad).
  4 calls a layer: ~+0.2 ms a call x 3 extra x 18 layers = +8-10 ms (DS41-307; projection "naive").
- (d) per segment k: the ring score with slot k's key buffer; queries of other lanes masked. 4 calls: ~4 x 2.9 ms at
  139k (x 3 index layers ... projected 6.8-7.7 ms for the four).
- writes, snapshots, acks: per segment.
v1 misses the 122 ms bar by the projection (121-145 ms). It is the correctness reference for v2.

**v2 (PB.3H / PB.4 tasks, the target path): one call a layer, per-chip arguments.**
- sparse_sdpa: one call; chip q reads its lane's table. The 4 gathered tables live in 4 buffers (DS41-290: the stock
  gather cannot write a region of a bigger buffer); the sdpa fork (ttnn/ttnn/bringup/sdpa) takes up to 4 table
  tensors and a per-chip table index (runtime arg; the program is per device already). Indices unchanged (no offset).
  Removes the +8-10 ms loop.
- indexer: one ring call; each SP row carries the 4 slots' key shares around the ring, each chip scores its lane's
  slot only (fork option on ring_indexer_score_dsa: up to 4 key buffers + a per-chip selector + per-chip causal start).
  Compute = today's per chip at that slot's H (the C x H term once, not 4x); ring traffic 4x the keys (at 139k: 4 x
  ~1.2 MB a ring step, well under the links).
- window: one call (already per chip); the carry per section 3.
- gather: v2b replaces the 4 stock gathers (every chip receives every slot: 4x the bytes) by a lane-targeted gather
  (each owner sends slot k's rows only to lane k's chips: today's per-chip bytes; per-lane buffers; DRAM v2 numbers in
  section 7). A fork of the gather (the rows path is a stock op today): fabric payloads <= 6208 B, bounded waits.
Each step keeps v1 selectable (a Setting) and is accepted only by the oracle (bit-identical to v1) and the walls.

## 6. Per-segment fixed costs: pipelined and hidden from the start

Per extra segment (projection, M = measured, P = projected): history gathers 3 sources x (pack 0.07 + gather
~0.2) ms at 139k (M: DS41-315, P.87 probe), ~1.0 / source at 1M (P.77); index-key decode ~0.1 ms / source; ~150-200
extra launches (P); per-segment writes / snapshots (P); engram rows: the same 8192-row host job as one request.

Design rules (from the start, not as a later optimization):
1. **Nothing per segment waits for the host on the device's critical path.** The C++ engram job (P.87) hashes all 4
   lanes in one GIL release; the 4 segments' history and key prefetches are enqueued before layer 0 starts (they
   depend only on the slot caches, complete before the chunk, and on the descriptor) rather than at the first
   engram wait, in need order: all slots' source-2 history, then index keys of 2, then sources 8 and 14.
2. **Overlap, v1 (PB.3): same command queue, ordered by need.** Each source's gather sits in the CQ just before the
   first layer that reads it is enqueued: source 2's 4 gathers behind layer 0-1's token-wise work, 8's behind layers
   2-7, 14's behind 8-13 (`prefetch` today issues them all at layer 1's engram wait). On one CQ this only reorders: it
   removes the dependency stall (P.73: the device idled at the engram wait), not the gather time. Exposed v1 = sum of
   gathers (W reports it per segment).
3. **Overlap, v2 (PB.3H): concurrent with compute.** A gather sub-device next to its links' Ethernet cores (row 0,
   cols 0-5, DS41-286) and compute on the other cores with the SAME core count as today (DS41-286: top-k ties split
   by core count; the compute grid is 110 matmul cores already, the gather's 6 cores come out of the 10 non-matmul
   spare ones). Gathers for source 8 / 14 run during layers 2-13's token-wise work (~4-5 ms a layer at 8192: 12 layers
   cover 4 x 3 x 1 ms at 1M). Hang history to respect: P.73's sub-device attempt hung (FIX.36: two 120-core programs a
   chunk, rule 18; DS41-283 the PCIe drop that turned out to be the same); DS41-288 a dispatch timeout with
   STATIC_TABLES=off + HISTORY_GATHER_OVERLAP=on (120-core programs, capped in P.73 attempt 2). So: every program
   capped (matmuls 110), bounded waits on every semaphore, smallest shape first (one source, one segment, 1 layer on the
   runner), then the 4 x 3 schedule; the hang loop (tests/test_skip_hang_loop.py) once before the walls.
   **Built (PB.3H, PACKED_GATHER_HIDE=on, tt/gather_hide.py):** deepseek_v3_d_p's sparse_sdpa-beside-push pattern
   instead of a gather sub-device at row 0: sub-device 0 = the 11 x 10 matmul grid (every matmul and every fork with
   ``max_grid`` runs there already: no compute op changes its core count), sub-device 1 = the free 12th column (10
   cores = the full-mesh gather's 8 link workers + 2 copy cores; slower than next to the links, DS41-286, but hidden).
   Each layer's attention phase (wq_b, rope chain, sparse_sdpa, rope chain, wo_a, wo_b: no fabric op; the stock
   non-matmul ops in it moved to the matmul grid for the region, cores.nonmm_on_mm_grid, row-local) is a region
   beside prefetch gathers, and so is the index-key decode at layer 1's engram wait: each source's (segment) items are
   balanced over the regions before its reader (largest first; source 2's over layers 0-1 and the decodes, 8's over
   2-7, 14's over 8-13). Dispatch is in order, so a region's gathers go between its compute steps, never back to back
   (DS41-331). At a source layer the compressor runs first and every segment's tail is ONE combined gather beside
   q_a, kv and the indexer's gates and query. Packs before a region (own record buffers), slice_writes after; an item
   not reached by its source's table is gathered there serially. Bit-identical (the oracle's packed3 case, histories
   of whole chunks). Measured (walls, device clock stamps: the real-time profiler cannot show two sub-devices at
   once): exposed / serial gather time 0.113 at 4 x 2k @ 139k, 0.177 mixed (BREADCRUMBS PPB.3H).
4. **Batch per-segment launches.** The 4 segments' served writes of a source (compressed KV + index keys) go in one
   prepared program (per-lane slot / start / valid runtime args); the 4 snapshot writes of a layer in one
   snapshot_send program (P.65's tree, 4 owners); the per-lane carry reads folded into window_send's program. Target:
   <= +20 launches a chunk over today, not +600 (4 x 150).
5. **Engram lookups**: one job for the chunk (unchanged count); the device-engram path (P.87, ENGRAM_DEVICE) fused its
   lookup with the history prefetch: with 4 slots the fused program takes the 4 slots' (slab, rows) pairs.
6. **Host prep**: the descriptor -> per-chip args is O(lanes) integer work; per-lane un-shuffle tables are load time.
Judged by the device profile (`packed_gather_exposed_ms`, `_hidden_ms` per segment) and the exposed-host breakdown.

## 7. DRAM budget at 1M, per chip (`tt/packed.history_bytes_per_chip`, printed by the CPU test)

Gathered history (HISTORY_FORMAT=bfp4: 320 B records; ratio 2: 522k rows at 1M; 3 gathered sources: 2, 8, 14;
layer 20's is not gathered under DECODER_SKIP=decode) + decoded index keys (bfp8 [rows / SP, 128] a source):

| layout | 1 segment (today) | 4 x 1M, v1 (4 tables on every chip) | 4 x 1M, v2 (each chip its lane's table) |
|---|---|---|---|
| history tables | 0.500 GB | 2.002 GB | 0.500 GB |
| index keys | 0.027 GB | 0.106 GB | 0.106 GB |
| total a chip | 0.527 GB | **2.108 GB** | **0.607 GB** |

Chip DRAM 32 GB; used at 1M today ~9 GiB a chip (P.87: weights, experts, served caches of the engine's slots, the
tables above). v1 adds ~1.6 GB a chip at 4 x 1M: fits (~11 GiB of 32). The engine's served caches are tt-d-gen's
allocation (num_users slots); packing needs >= 4 slots resident, which serving already has. The walls test records
`packed_dram_peak_gb_4x2k_p1m` from the device allocator (all 4 buffers held at once) and the headroom.

## 8. 8192 single-request chunks on today's tree (PB.2's input)

X.8 ran 8192 on the X.7 tree (b4e05edc584): +15.4 % tok/s at ~142k, determinism 0.0. Since then (git log
b4e05edc584..HEAD -- tt/): P.78 (host hide), P.85 (replica table; replication is not on this branch, DS41-278),
F281-283 (layers 0-20, DECODER_SKIP per run), C++ engram host (P.87 groundwork), P.87 (engram device path, off),
P.83 (mhc_post fast), P.81 (grouped wo_a). P.77 / P.49 / P.75 / P.76 were already in X.7. What may not work at 8192
(R = 256 rows a chip, was 160):
- **P.83 mhc_post fast** (fork kernels mhc_post_feed_reader/sender): probed only at [160, 5120] a chip; the feed's
  packet count / row tiling at 256 rows unverified (risk: hang or wrong rows; run its op test at 256 rows first).
- **P.81 grouped_matmul (wo_a)**: `supported()` plans per shape; at M = 256 it either plans (new L1 blocks, unprobed)
  or falls back to stock (safe). Check which.
- **mm_configs**: every tuned entry is for 160 rows; 256-row shapes fall back to the stock config (safe, untuned:
  part of X.8's +44 % device).
- **P.78 host hide**: engram rows TP-sharded per chunk (sized from the ids); jobs scale with 8192; the switch interval
  and deferral tuned at 5120 (wall only).
- **MoE block**: one MoEBlock per (mesh, chunk): arena / dispatch buffers sized at 256 tokens a chip (X.8 passed on
  the X.7 tree; the fused head's L1 at 256 rows unchanged since).
- **DS41-279**: position tables end at target.seq: an 8192 test past 56320 raises it (the test does).
- **DS41-281**: programs keyed by chunk start: a first 1M pass at 8192 builds new programs (warm-up before walls).
- **ENGRAM_DEVICE=on** (off by default): engram_device's L1 sizing names 160 rows (engram_device.py:1168).
Accuracy at 8192 against the goldens (written: `tests/packed/test_chunk8192_reference.py`): the s56320 golden is
5120-chunked, but a reference row does not depend on the chunking (fixed-tile GEMMs, exact causal attention over the
state, no pending group at 32-aligned starts), so golden rows are valid targets for any chunking:
- `from0`: 8192 chunks from 0, the last [49152, 57344) with actual_end 56320; rows [51200, 56320) vs golden chunk_10,
  limits from drift-s56320.json (the rung that runs from 0 on the device) - every layer, final hidden, top-5,
  states; rows [49152, 51200) vs chunk_09 recorded as a diagnostic.
- `last`: the golden state at 51200 (the only fixed-state snapshot) + one 8192 chunk with 5120 real rows (the padded
  tail at 8192) vs chunk_10, limits from drift-last.json.
No new golden or drift is needed (CPU cost 0). A golden with a full 8192 chunk of real rows at the gated position
would need a new s65536_c8192 golden: ~2.5-3 h CPU (the s56320 golden's cost scaled by 65536 / 56320) plus its
official drift (~same): not needed for an honest limit, since `from0` runs six full 8192 chunks whose errors feed
the gated rows through every cache.

## 9. Test plan (named tests, commands, metrics)

All device tests: BATCH_SHARED = True, through the resident runner, the 8192 model (BRINGUP_CHUNK=8192):
`python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "BRINGUP_CHUNK=8192
scripts/run_safe_pytest.sh --run-all --no-precompile <file>"`; layers 0-20 (BATCH_LAYERS) except where noted.

1. **CPU rung (built, PB.1 gate)** `tests/packed/test_packed_cpu.py` (`gate PB.1 --check`): descriptor math on every
   box; the CPU reference packed vs alone (4 x 64-row lanes, 5 slots, histories 0-160, padded / odd / empty lanes).
   Metrics: `packed_cpu_cases` (28), `packed_cpu_failures` (0), `packed_cpu_lane_equiv_max_abs_diff` (0),
   `packed_cpu_alone_vs_today_max_abs_diff` (0), `packed_cpu_slot_digest_mismatches` (0),
   `packed_cpu_determinism_max_abs_diff` (0), `packed_cpu_vs_forward_chunk_rel` (diagnostic, 5.8e-7).
2. **Oracle** `tests/packed/test_packed_oracle.py` (PB.3; rungs: 1 layer, the characteristic layers [0, 1, 2, 3, 14,
   20, 21, 24], layers 0-20), short synthetic histories first, then agentic_cc sessions:
   - `packed_lane_equiv_max_abs_diff`: segment S packed with any 3 others (2 compositions, S in 2 lanes) == S alone in
     the same 8192 geometry, every layer output (its rows) and every slot write; must be 0.
   - `packed_alone_vs_today_max_abs_diff`: one segment through PACKED=on vs PACKED=off on the same request and chunk
     (both in one load: call-time switch); must be 0.
   - `packed_slot_digest_mismatches`: sha256 of every served record each segment wrote (compressed KV, index keys,
     window snapshot, every layer) vs the same request alone; must be 0.
   - `packed_determinism_max_abs_diff`: the same packed chunk twice; must be 0.
   Cases: a fresh segment (start 0) next to continuing ones (start classes differ), a replay segment (gws > start), a
   segment with cap <= start (acks only), an empty lane, a 1-row segment.
3. **Reference** `tests/packed/test_packed_reference.py` (PB.3, 40 layers once in the accept gate). The goldens hold
   one fixed-state snapshot (51200 in s56320_c5120), so every segment continues that point: 4 distinct slots each
   loaded with the golden state at 51200, lane k carrying golden rows [51200, 51200 + n_k) (n = 2048, 2048, 1600, 777:
   full and padded lanes), each segment vs golden chunk_10 rows [0, n_k): `err_seg<k>_layer_L<i>`,
   `err_seg<k>_final_hidden`, `top5_overlap_seg<k>` (the ladder's sampled rows below n_k), each with `_limit` from
   drift-last.json (rung `last`: the limits the ladder gates; no new limit). The 4 lanes then hold the same request at
   the same positions in different lanes and slots: lane equivalence at full depth for free (their outputs must be
   bit-identical; recorded `packed_ref_lane_spread_max_abs_diff`).
4. **Walls** `tests/packed/test_packed_walls.py` (PB.3 / PB.3H; DECODER_SKIP=decode, layers 0-20): steady chunks driven
   as tt-d-gen would: device ids in the engine layout + the descriptor; each lane a different agentic_cc session at its
   real history position (P.85 sessions); slots assigned at random and changed every chunk; every segment's history
   gathered from its own served cache every chunk (no carried device state). Cases (metric suffix): `4x2k_p139264`,
   `mixed` (61k / 139k / 308k / 547k), `1x2k_p139264` (1 segment + 3 empty lanes), `4xp50_p139264` (4 x ~1.6k: padded
   lanes), `4x2k_p1m`. Metrics: `packed_wall_ms_<case>`, `packed_serving_tok_s_<case>` (new tokens / (wall + ids
   readback + the device-id input path)), `packed_device_ms_<case>`, `packed_host_exposed_ms_<case>` (+ breakdown:
   launches, engram, chunk start / end, python), `packed_gather_exposed_ms_<case>`, `packed_gather_hidden_ms_<case>`
   (and `_seg<k>`, from the device profile), `packed_gather_exposed_frac_<case>`, `packed_dram_peak_gb_4x2k_p1m`,
   `packed_dram_headroom_gb_4x2k_p1m`. Table: `<results dir>/<task>_packed_walls.md`.
5. **8192 single request** `tests/packed/test_chunk8192_reference.py` (written, PB.2): section 8.

## 10. Projected chunk time (packed8k_projection.md; P = projected, not measured)

| ms a chunk | 4 x 2k @ 139k v1 | v2 | mixed v1 | mixed v2 |
|---|---|---|---|---|
| token-wise device (8192 rows) | 82-88 | 82-88 | 82-88 | 82-88 |
| indexer | 7-8 (4 calls) | 4-5 (1 call, C x H once) | 11-13 | 6-8 |
| sparse_sdpa loop delta | 8-10 | 0 | 8-10 | 0 |
| gathers exposed | 2.6 | 0-0.7 | 4.8 | 0-1.2 |
| per-segment glue (device) | 1-4.5 | 1-4.5 | 1-4.5 | 1-4.5 |
| host exposed | 16-32 | 13-17 | 15-31 | 12-16 |
| ids readback | 1.5-2 | 1.5-2 | 1.5-2 | 1.5-2 |
| **wall / new tok/s** | 121-145 / 56-68k | **104-120 / 68-79k** | 126-151 / 54-65k | **107-125 / 66-77k** |
The v2 column is what the targets need; v1 is the correctness rung.

## 11. Decisions (OV-58: owner asleep, decided by the agent; marked DECIDED in BREADCRUMBS)

- DECIDED: engine input order inside a lane = per-segment ring_sdpa_reshuffle (each lane reshuffled on its own
  chips, the order today's single-request path already uses at 2048 rows a lane); the model also accepts
  lane-natural order, so the engine may pick either (the draft asks for reshuffle first).
- DECIDED: pad rows are excluded from MoE dispatch (no expert, weight 0): changes nothing for a real row if PB.4's
  row-locality check holds, and saves the padded lanes' expert work. If PB.4 finds the device MoE is not row-local,
  pad rows go through dispatch like real ones and the oracle compares under the same padding.
  Built in PB.4 (PACKED_MOE=on): the row-locality check holds (test_packed_moe_block, oracle: 0).

## 12. Estimates PB.2-PB.4 (from this ledger)

Comparable: X.8 (8192 sweep, ~1 h, 1 load), P.83 (fork kernel to roofline, 1.4 h, ~10 probes on the runner), P.81
(fork + model, 2 attempts ~3 h), P.87 (engram device + fused gather, ~2.7 h + attempt 1), P.73-P.75 (history gather
and its hiding, 3 tasks, ~1 day with the FIX.36 hang).
- PB.2 (8192 accuracy + walls at 139k / 512k): ~1-1.5 agent h; device ~25-35 min (1 load of 40 layers for the
  accuracy test ~5 min + 7 chunks; walls layers 0-20 ~10 min; profile ~10 min). Risk: P.83 / P.81 at 256 rows.
- PB.3 (packed v1 + oracle + reference + walls): ~5-8 agent h; device ~60-90 min (oracle rungs 1 layer -> 8 layers ->
  0-20, 3-4 runner loads; 40-layer gate once ~15 min).
- PB.3H (hidden gathers, v2 sdpa / indexer forks): ~5-8 agent h; device ~60 min + hang-loop ~30 min.
- PB.4 (bulk MoE row-locality + tuning): ~3-5 agent h; device ~30-45 min.
