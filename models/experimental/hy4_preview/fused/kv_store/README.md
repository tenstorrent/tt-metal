<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# `fused/kv_store`: a paged KV store for every live conversation

The design is `fused/README.md` (sections 0, 1, 4 option 3, 7, appendix). State: 2026-11-20
00:37 UTC. (M) measured on this Galaxy, (S) simulated, (E) estimate.

## What exists

| file | what it is | evidence |
|---|---|---|
| `frames.py`, `kernels/kv_frames.hpp` | The address arithmetic: stripe row of a paged conversation -> DRAM bank and byte offset through a frame table, ONE function (`locate`), the same in Python and for kernels. A tensor's layout is a `Geometry` (two strides), so one function serves the frame-major pool, today's slot-major cache (identity table) and the workbench. | `tests/test_frames_host.py`: 400,000 random cases equal to `sm::kv_row_bank / kv_row_offset` (compiled from the owner's header), `layout.kv_row_location`, the append's stripe arithmetic, the scan's bank stream (log `mk10_frames_host`) |
| `manager.py` | `KvStore`: the pool of 8192-token frames, conversations as ordered frame lists, LRU eviction by tokens of conversations with no request in flight, the decode-row binding and the tables the kernels read, the frame pairs of a workbench prefill (`copy_in`, `copy_out`), `fork`; `PrefixIndex`: longest stored prefix of a prompt on the 4096 grid. Plain Python. | `tests/test_manager_host.py`: unit and random-operation tests, the recorded hour, the simulated hour at 4 / 5 / 6 / 8 M tokens (log `mk10_manager_host`) |
| `device.py` | Pool / cache tensors as the prefill stack allocates them, their raw (uint8) twins, `FrameTables` (the table as an L1 tensor), `Probe` (a reader of ours that reads through a table). | `tests/test_frames_device.py`, one chip (logs `mk10_frames_board_first`, `mk10_frames_board`) |
| `copy.py`, `kernels/kv_copy*.{hpp,cpp}` | `KvCopy`: frames between two tensors or inside one, bank-local, the same program on every chip, no fabric; the plan is a device tensor. | `tests/test_kv_copy.py`, one chip and 32 chips (logs `mk10_copy_board_first`, `_bytes`, `_rate`, `mk10_copy_galaxy`; numbers in `generated/hy4_preview/logs/mk10_kv_copy.json`) |
| `engine.py`, `kernels/kv_compare.cpp` | `Engine`: the store end to end with the prefill stack (reserve, copy the prefix into the workbench slot, traced passes, copy out, tables); `KvCompare`: frames of two tensors compared on the device. | `tests/test_engine.py`, 32 chips, 6 and 78 layers (logs `mk10_engine_L6`, `mk10_engine_L78`, `mk10_engine_replay_L78`); `tests/test_kv_copy.py::test_compare` (`mk10_compare_board`) |
| `PATCH.md`, `patch.py` | The change of `sm_reader.cpp`, `sm_append.cpp`, `scan_reader.cpp` and their Python to go through `kv_frames.hpp`, for their owner; as text and as data. | `tests/test_patch.py`: patched copies in the owners' programs, bit-equal to the unpatched kernels on one chip (logs `mk10_patch_board_sm`, `mk10_patch_board_scan`) |

Mock compiles of everything: `mk10_frames_mock`, `mk10_copy_mock`, `mk10_patch_mock`, `mk10_probe_mock`.

## Measured

- `kv_copy` (M, traced, 20 replays back to back; 99 slots = 78 latent + 21 index-key per frame, 14.8 MB per frame
  and chip):

  | frames | MB per chip | one chip | all 32 chips at once (per chip) |
  |---|---|---|---|
  | 1 | 14.8 | 0.092 ms, 161 GB/s | 0.097 ms, 152 GB/s |
  | 12 | 177.5 | 0.92 ms, 192 GB/s | 0.93 ms, 191 GB/s |
  | 64 | 946.5 | 4.78 ms, 198 GB/s | 4.81 ms, 197 GB/s (30.3 GB over the mesh, 6.3 TB/s) |

  Read side alone 486 GB/s, write side alone 313 GB/s, launch floor 13 us (12 frames, one chip). One awaited
  launch, not traced: 0.16 / 0.99 / 4.86 ms on one chip, 0.49 / 1.34 / 5.82 ms on 32. Bytes equal after every
  plan (pool -> workbench, workbench -> pool, inside the pool, empty plan; 64 frames in one launch; on all 32
  chips). Best configuration (the default): 4 copier cores per DRAM bank on the bank's worker row, 2 packets in
  flight. Chip by chip on the mesh (12 frames): 192 to 193 GB/s on the 15 chips whose right-hand bank column is 7,
  204 to 208 on the 17 where it is 6, with the one placement taken from a column-7 chip; with each chip's own
  placement (`placement="own"`) 190 to 196 on all. No chip is the slow one.
- A row address through the table: 22.9 ns against 21.3 ns through today's arithmetic (M, one core, 16,320
  addresses). The patched row reader: 47.3 us against 45.7 us for 2,067 rows; the patched scan: 42.1 / 41.6 us
  (identity / pool) against 42.3 us at 8 users x 8,192 stripe rows (M, one chip).
- The store in the trace simulator (S, 16 client slots, fused decode estimate, traced prefill; frames of 8192):

  | budget | tok/s | evictions | prefilled again | allocations (frames) | `kv_occupancy.md` LRU, exact tokens |
  |---|---|---|---|---|---|
  | 4 M | 122 | 124 | 11.13 M (78 %) | 263 (2,275) | 118 tok/s, 131, 11.66 M |
  | 5 M | 220 | 31 | 1.09 M (21 %) | 294 (1,108) | 225, 29, 0.62 M (run here; not in its table) |
  | 6 M | 237 | 17 | 0 | 301 (917) | 237, 16, 0 |
  | 8 M | 237 | 0 | 0 | 301 (917) | 237, 0, 0 |

  32 client slots: 6 M 95 tok/s (154 evictions, 12.01 M again; exact 111, 137, 10.07 M), 8 M 209 (39, 0.10 M; exact
  209, 37, 0.10 M). Where they differ: a conversation holds whole frames (half a frame more on average), so at a
  budget on the cliff fewer conversations fit (5 M, and 6 M at 32 slots); and the store evicts nothing when the
  eviction would not make the request fit (at 4 M it keeps more: 122 against 118 tok/s). The recorded hour itself
  (2,101 requests at their recorded times, 111 conversations never told to end): 653 to 690 allocations, 47 / 71 /
  89 / 152 evictions and 0 / 0.10 / 1.71 / 11.27 M tokens prefilled again at 8 / 6 / 5 / 4 M.

## How the stack is meant to use it

    store = KvStore.for_budget(tokens, rows=8)                      # the pool: store.pool_shape(kind) per kind
    # a request arrives
    grant = store.begin(key, prompt + max_new, shared_prefix)       # None: wait for a request to end
    #   3b (workbench): copy the prefix in, prefill from grant.restart in the workbench, copy the new rows out
    copy_in.set_pairs(store.copy_in(key, grant.restart));  copy_in()
    ...prefill stack, one contiguous user slot, unchanged...
    copy_out.set_pairs(store.copy_out(key, grant.restart, prompt)); copy_out()
    # decode
    store.bind(row, key)
    if store.table_version != written:                              # rewrite the L1 tables
        latent_tables.write(store.table(Geometry.pool(LATENT)));  index_tables.write(store.table(Geometry.pool(INDEX)))
    ...decode passes: the row reader, the append and the scan resolve rows through the tables (PATCH.md)...
    store.unbind(row);  store.end(key, cached=prompt)

- An append past the mapped frames is skipped by the kernel: reserve prompt + max output at `begin` (as above), or
  `store.reserve(key, more)` and rewrite the table before the pass that crosses a frame.
- `copy_in` / `copy_out` / a pool-internal copy are three `KvCopy` objects (different tensor pairs), each one
  program, traceable; a plan holds up to 255 frames (2 M tokens).
- At the measured rate the workbench copies of KV_STORE.md 4 (3b) are: copy-in of the median request (103K tokens,
  186 MB per chip) 1.0 ms, of the largest (697K, 1.26 GB) 6.4 ms, against 0.47 s for the traced prefill pass of a
  small append; 181 GB per chip per hour = about 1 s per hour (E from the measured rate).
- AgentX terms: this piece adds nothing to a decode pass but the table lookups (PATCH.md 5); per additional decode
  row 512 B of L1 per table core; what it bounds is the number of conversations kept, i.e. the cliff in the table
  above (6 M tokens for 16 client slots, 8 M for 32).

## End to end with the prefill stack (`engine.py`, `tests/test_engine.py`; no decode)

The prefill stack is built in process through its adapter as `tests/hy4/test_hy4_agentx_append.py` builds it
(torus mesh, L1_SMALL 1536, a 256 MB trace region, trace mode, one capture), with `1 + n` ordinary user slots:
slot 0 is the workbench, the others are reference / fixed slots. Nothing of the stack is edited. The engine
allocates the pool (zeroed with the stack's `DRAMZeroFill`), the two table tensors and its plans, and launches each
of its programs once, BEFORE `capture_trace`; after the capture it only rewrites tensors and launches programs.

Correctness (M; logs `mk10_engine_L6`, `mk10_engine_L78`, reports `mk10_engine_L6.json`, `mk10_engine_L78.json`):
five conversations of random tokens through the one workbench, 13 turns: first turns, appends that restart at
8192 / 4096 / 12288 (inside a frame), growth by a frame, a compaction (5000 tokens shared, the conversation shrinks
to one frame), an explicit eviction and the re-prefill, two LRU evictions by the manager (4 evictions in all),
frames out of order (B ends in frames [2, 3, 1], D in [0, 4, 8]). At two checkpoints each live conversation is
prefilled ALONE into ordinary slot 1 with the same passes, and its pool frames are compared with that slot on the
device (`KvCompare`: every layer, both kinds, all 32 chips, positions [0, ceil32(N))):

| | 6 layers | 78 layers |
|---|---|---|
| conversations compared (A, B, C, D; then B, D, E) | 7 of 7 equal | 7 of 7 equal |
| latent rows compared on the device / differing | 644,736 / 0 | 8,381,568 / 0 |
| index-key blocks compared / differing | 10,074 / 0 | 70,518 / 0 |
| host comparison of the same rows (bytes; decoded keys) | all 7 conversations, all layers: 0 differ | A and B, layers 0, 1, 2, 5, 39, 76, 77: 0 differ |
| control: a conversation against the slot still holding the previous one | differs, as it must (asserted) | the same |

So a conversation that left the workbench, was replaced there by others and came back attends to the same rows,
bit for bit, as if it had an ordinary slot of its own. That is also the experiment for KV_STORE.md 9 question 3.
By reading: every pass re-reads its slot from the cache tensor (`ttMLA._gather_kvpe_prefix`: the latent prefix of
the slot into the persistent scratch, extent from the chunk start; `ring_indexer_score_dsa`: the key slabs of the
slot, capped at ceil32 of the valid end; the push planner's cached plan holds row INDICES and is dropped by every
full-indexer layer, layer 0 first, so in every pass); the runtime's only per-call state is `_trace_slot_id`.
Nothing in the runtime remembers a slot's CONTENT between calls, and replacing the rows of user 0 between requests
is safe; `compile()`'s warm-up rows in slot 0 are simply overwritten or never selected.

What B can see of A: nothing. What is physically there: the workbench keeps the rows of whoever was in it, so the
frames copied out for a conversation hold stale rows beyond its own ceil32(N) (host count: at 78 layers 1,344
rows of A's last frame and 17,472 of B's over the 7 compared layers; at 6 layers 0 for D and E, whose tails
nobody had written). The stack never reads them: real query rows are causally limited to positions below their own, the key
scoring is capped at ceil32(valid end), and the latent gather's rounded-up suffix "is never consumed" (its own
comment); the bit-equality above was measured with different stale rows on the two sides (the workbench held other
conversations, slot 1 held the previous reference or zeros). Decode has to keep to the length likewise (below).

A slice of the recorded AgentX hour at 78 layers (M; log `mk10_engine_replay_L78`, every request in
`mk10_engine_replay_L78.json`): the first 120 requests of the export in recorded order, 25 conversations, random
tokens with the recorded prompt lengths and shared prefixes, lengths capped at 126,976 (workbench of 131,072
tokens; 6 conversations capped), one request at a time, decode a stub (bind a row, rewrite the tables, unbind).
Pool 183 frames = 1.5 M tokens, which this slice overflows.

| | |
|---|---|
| passes | 644 in 354.6 s = 0.551 s per traced pass; 116 of 120 requests had new tokens (55 of them one pass, 0.52 s) |
| copy-in (prefix, pool -> workbench) | 88 requests, 885 frames (10 per request, at most 15 = 222 MB per chip): mean 2.3 ms per request with new tokens, p50 2.9, max 3.9 ms |
| copy-out (new frames, workbench -> pool) | 116 requests, 374 frames: mean 2.9 ms, max 4.1 ms |
| of both, the host writing the plan tensors | 0.3 + 0.6 ms per request |
| table rewrite (two L1 tensors, 16 + 64 cores, 32 chips) | mean 3.6 ms, max 5.6 ms per binding change |
| the same 50 requests of the 2 busiest conversations on fixed slots of the stack (same 104 passes) | store path 56.39 s, fixed slots 55.94 s: +9.2 ms per request (+0.8 %), all of it copies, tables and bookkeeping (passes 55.69 s on both) |
| everything the store adds, over the slice | 1.6 s of 357 s (0.46 %) |
| evictions (LRU, by the manager) | 12; 209,565 tokens prefilled again (52 passes, about 29 s: the price of a pool this small, not of the store) |
| the busiest conversation at the end (38 requests, 108,813 tokens, 14 frames): pool against its fixed slot | 8,488,896 latent rows and 71,421 index-key blocks compared on the device, 0 differ (the other fixed conversation had been evicted from the pool) |
| DRAM per chip | prefill weights 20.94 GB; 3 slots of 131,072 tokens 0.71 GB; pool 2.71 GB; scratch and trace 0.22 GB; 24.58 GB allocated, 9.33 GB free |

The copies are bound by the host, not by the device: 10 frames are 0.8 ms of device time at the measured rate;
the rest of the 2 to 3 ms is one plan write and one untraced launch with its synchronize on 32 chips. Capturing
the three copy programs as traces and writing plan and tables in one host write would bring the 9 ms down; it
is 1.8 % of a one-pass request as it is.

## Decode on the store: what the engine needs from decode later

`engine.py` ends where a request's prompt rows are in the pool. For decode to serve from there:

1. Binding of decode rows. A request that enters decode row r: `store.bind(r, key)`; when it leaves:
   `store.unbind(r)`. Before the next decode pass, `engine.sync_tables()` (it rewrites the two L1 table tensors only
   when `store.table_version` changed). A bound conversation is never evicted; a conversation between `begin` and
   `end` is never evicted either. The decode kernels read the tables as PATCH.md describes: the row readers and
   the append get `engine.latent_tables`, the scan cores `engine.index_tables` (their core lists are constructor
   arguments of `Engine`; the defaults are the sparse-MLA reader columns and the scan grid).
2. The append reservation. The patched append skips a row whose frame is not mapped, silently. So the engine must
   map the frames of every position a pass can write BEFORE the pass: either `engine.begin(key, prompt, max_new)`
   with the request's output limit (the frames are held for the whole request: at most 4 frames for the 32,370
   output tokens the trace has), or `store.reserve(key, context + margin)` plus `sync_tables()` when a row comes
   within a pass of its last mapped position (`store.mapped_tokens(row)`; once per 8192 generated tokens per row).
   A reserve that returns False (pool full of pinned conversations) must stall that row, not the pass.
3. Decode writes the pool, not the workbench: rows `[N - 1, N + generated)` through the table (the stack's
   contract, `prefill_stack.md` 3: decode runs the last prompt token itself, which rewrites row N - 1). After the
   request `engine.end(key)` records N as what the next request can reuse: the AgentX client's next prompt
   continues the PROMPT, so the generated rows are dead; `end(key, trim=True)` gives their frames back at once.
4. Order with prefill. There is one workbench, so one prefill request at a time. A request may join a decode pass
   only after its copy-out (until then its new rows exist in the workbench only). Decode passes of OTHER
   conversations can run between the chunk passes of a prefill: they read the pool, which a prefill touches only
   in its copy-out, and that writes frames of the conversation being prefilled, which is not in decode.
5. Lengths. A pool frame holds stale rows beyond the conversation's length (rows of whoever used the workbench
   before, or of an earlier, longer turn; measured below). As in an ordinary slot, decode must carry the context
   length and never select a position at or beyond it; the scan's context word and the selection already do.
6. Nothing converts: a pool row is byte for byte the row the prefill stack wrote (same packing, chip, stripe row
   and bank), so decode reads prefill's rows in place, in another tensor.
7. More than one user per chunk pass (a later optimisation of the prefill) would need one workbench slot per user
   of the pass: `Engine` takes the workbench as user slot 0 of the stack's tensors today.

## Still needed from the prefill stack's owner (KV_STORE.md 9)

Unchanged from the study except where marked.
1. Experts: which single layout, and is the expert -> chip map stable across builds?
2. tp32: when? (6.2 M -> 8.9 M tokens.)
3. Workbench: ANSWERED here (section "End to end with the prefill stack"): nothing in the runtime remembers a
   slot's content between calls; replacing the rows of user 0 with `kv_copy` between requests gives bit-equal
   caches at 6 and 78 layers. Worth the owner's confirmation that this is a property he means to keep.
4. Frame table in the four cache ops (3a): acceptable as an optional tensor argument, and in which order? NEW: the
   table format this piece settled on is one uint32 "frame base" (bytes inside a bank) per 8192 tokens, the same
   for every chip and bank, plus one per-layer slot step; the ops could take exactly `kv_frames.hpp`.
5. Could the full-indexer layers use the push path too (0.58 GB of gather scratch back)?
6. Classes: no longer a question unless paging is dropped (the copy is 100 times above the 2 GB/s threshold).
7. Allocation of the pool as ONE tensor per kind: PARTLY ANSWERED: the allocator takes `[59,046, 1, 256, 656]`
   (757 frames, 1.33 GB per bank) and `[105,534, 1, 256, 656]` (1,353 frames, 2.38 GB per bank) on the mock (log
   `mk10_probe_mock`); on the board the largest allocated here was 64 frames. Still open: `DRAMZeroFill` over
   27 M pages, allocation order against the weights, whether the trace region is DRAM on every chip.
8. Appends restart on the 4096 grid; if the fix of the off-grid difference moves the write window, `copy_out`'s
   range changes (it copies whole frames from `restart // 8192`).
9. Rows `[N, ceil32(N))` hold pad-token KV after a prefill and `copy_out` copies whole frames: is anything else
   written beyond `ceil32(N)`?

## Where KV_STORE.md turned out different

- The copy rate was assumed at 20 GB/s per chip; it is 152 to 198 GB/s (M). The fallback condition of section 7
  (below 2 GB/s) is far away. In the engine a copy is host-bound instead: 2 to 3 ms per request whatever its size
  up to 15 frames (M), so the copies of 3b cost about 5 ms per request, 4 s in an hour of 811 requests (E).
- Section 4 expected the scan to pay for frames of 8192 (packets of 1 block instead of 3) and offered frames of
  24,576: measured, the patched scan on a frame-major pool is no slower at 8,192 stripe rows (41.6 against
  42.3 us) and 1.9 us slower at 1,024. k = 1 stands.
- The table: the study has frame NUMBERS (uint16) and a multiply-add per row, in a DRAM tensor read at launch. Built:
  frame BASES in bytes (uint32, one table per kind) in an L1 tensor: no multiply by the frame, no read at launch,
  and the same kernel reads today's slot-major storage under an identity table, so the kernels can be patched
  and checked bit for bit before the storage is re-allocated. 512 B per decode row and core, not 200 B.
- The copy plan: the study has (source offset, destination offset, bytes) per bank range (99 x 8 ranges per
  frame); built: one (source frame, destination frame) pair per frame, the ranges follow from the two tensors'
  geometry.
- "822 allocations for 4K pages, about half with 8192": in the simulated hour the store allocates 301 times (917
  frames) at 16 client slots; in the recorded hour 653 (1,647 frames).
- Not KV_STORE.md but its simulator: `agentx_model/sim.py` `_admit` takes ALL of a conversation's cached prompt
  as shared when a request has no recorded prefix (`lcp` 0), also for the three compaction turns whose prompt is
  shorter than what is cached: 107K tokens of prefill are skipped in the 16-slot hour (3 % of the 3.57 M). The
  manager test reports it (`kept ... beyond a prompt's own length`).
- Confirmed as written: chip, stripe row and bank of every row are kept by frames of 8192 and not by 4096
  (`test_bank_is_kept_by_frames_of_8192_and_not_by_4096`); thresholds of the budget sweep (220 at 5 M, 237 at 6 M).

## Not done

- End to end with DECODE: the engine stops where the prompt rows are in the pool; no decode kernel has read a
  pool the engine filled, and the manager's row binding was only exercised by a stub.
- The engine's copies and table writes are not traced (host-bound, 9 ms per request); `max_new` reservation and
  `trim` were not exercised on the board; one workbench, one prefill at a time.
- The patch is not applied (PATCH.md); `attend_program` and the layer program were not run with it; 16 rows not run.
- Partial eviction (dropping only tail frames), prefix sharing between conversations (frames are never shared;
  `fork` copies), a layer-major pool (`Geometry.contiguous(kind, frames)` with one user: supported by the
  arithmetic and the copy, would keep 3-block scan packets for adjacent frames; not needed by the measurements).
