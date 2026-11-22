# Disaggregated prefill -> decode over TT-Fabric: plan

Status (2026-11-22): phases 0-3 done; the servers run end-to-end with `--prefill-push` (section 6, phase 3). This document is the
work plan for moving the prefill server's KV hand-off from "read back to host + TCP blobs" to "prefill writes decode's
DRAM directly over TT-Fabric, TCP carries only control".

## 1. The plan being implemented

1. The decode server (host 41) allocates the new session's KV cache pages itself (`ensure_session_capacity`).
2. Its prefill request to the prefill server (host 40) carries, besides the ids, where every page lives: per decode
   chip the fabric node and DRAM bank table, per cache buffer its address, per paged block its physical block id.
3. The prefill server prefills, then writes the KV entries straight into those pages over TT-Fabric. No device-side
   synchronization with decode.
4. It replies "done" over TCP after its write program has finished. By the time decode has that reply and schedules
   the user, the pages have landed (measured below), so decode finishes the commit's host bookkeeping and replays the
   prompt's tail as it does today.

What this removes, per request at 1M tokens (`PREFILL_D_P_INTEGRATION.md` 6.2): the 5.5-11.8 s device-to-host
readback inside the prefill device lock, the 2 s / 3.35 GB TCP send, and on decode the host-to-device upload plus
the cross-stage migration of `commit_prefill_state`.

## 2. What is proven (`tests/xhost_kv/`)

Three-part unit test, run under one `tt-run` job (rank 0 = host 40, mesh 0; rank 1 = host 41, mesh 1; two-mesh MGD
`dual_bh_galaxy_8x4_hosts_40_41_two_mesh.textproto`):

* `receiver_init.py` (decode side): opens mesh 1, allocates decode-shaped pools (`comp_kv`: bf16 ROW_MAJOR
  `[blocks, 1, 64, 512]`; `idx_key_cache`: bf16 TILE `[blocks, 1, 64, 128]`) filled with a sentinel, sends addresses
  and bank tables, requests writes into random blocks, reads them back the moment "done" arrives.
* `sender_write.py` (prefill side): one core per chip, chip (r, c) of mesh 0 writes chip (r, c) of mesh 1, one fabric
  unicast write a page, the destination NoC address from the receiver's bank table.
* `receiver_verify.py`: device-less UMD read of every page, byte-exact, untouched blocks still the sentinel.

Results: 32 chip pairs x 10 rounds x 32 blocks, both pools, all byte-exact at "done" and after the job; sentinel
blocks intact. One debugging run hung (28 of 32 senders stuck in `wait_for_empty_write_slot`) and dropped a chip
off PCIe on host 40 (`tt-smi -glx_reset` brought it back); the identical run passed afterwards. Not reproduced
since: treat as an open risk (section 8).

Route map of mesh 0 -> mesh 1 (sender log): rows 0-1 (fabric chips D0-D7) leave over Z directly, rows 2-4 go N,
rows 5-7 go S (torus wrap) to an exit chip first.

## 3. Constraints found

| Constraint | Consequence |
| --- | --- |
| A two-mesh fabric's control plane all-gathers over every rank at init (`ControlPlane::init_control_plane`) | prefill and decode servers are the two ranks of ONE `tt-run`/MPI job: they start together, and one dying ends the job |
| Router settings must be identical on all ranks (payload, routing planes, tensix config, manager mode) | decode must open with the prefill's packet payload: 6208 B (`MOE_BLOCK_SPLIT_ROWS=on`, default) or 10880 B (`off`); decode today uses the 4352 B default |
| The sender kernel needs a 2D fabric (a remote mesh is reached through its exit chip); a 2D packet header has to be routed by mesh 1's routers | decode must run a 2D fabric (`FABRIC_2D_TORUS_XY`, as the prefill); it already does: the `base` profile of `deepseek_v4_flash/configs/system_configs.yaml` sets it (`FABRIC_1D_RING_X` is only the dataclass fallback in `tt/system_config.py`) |
| `STRICT_INIT` reliability (both servers' default) wants every link of the MGD up | the xhost test used `RELAXED_INIT`; check `STRICT_INIT` against the two-mesh MGD or switch both servers |
| `/big_home` is per host; tt-run hands every rank the launch host's paths | host 41 runs a checkout of this branch at the same path, built (copying single files only worked for the test) |
| Remote-mesh traffic leaves through exit chips | link balance is decided by routing, not by us; measure (phase 4) |

Correctness, independent of transport: the metal_2 prefill states do not yet match metal_1's own prefill states
(`PREFILL_D_P_INTEGRATION.md` 6.1: layer 0 matches, layers 1-20 diverge, PCC 0.5-0.8). The fabric path moves the
same values, so decode output will be wrong until that is found. Tracked separately (section 7); the push path is
validated byte-for-byte against the existing host-path commit of the same states, which does not depend on it.

## 4. What moves, from where to where

Decode targets (`tt/model.py`, `prepare_static_decode`, `commit_prefill_state`). Decode stage `s` is mesh-1 row `s`,
a `1 x 4` submesh created at `(s, 0)`; a stage's caches are replicated over its 4 chips, at one address per stage (each
submesh has its own allocator, so addresses differ between stages).

| Data | Decode buffer (per stage, every chip of it) | Pages written | Prefill source (metal_2, mesh 0) | Transform |
| --- | --- | --- | --- | --- |
| window ring, layers 0-19 | `scaches[li].kv`, bf16 TILE `[1, 1, slots*128, 512]` | 64 tiles at page `slot*64` (rows `slot*128 ..+128`) | `st.window[li]` ring blocks (`_window_rows`) | tilize 128 x 512 |
| layer 20 ring, layers 21-39 rings | same | zeros | none (`DECODER_SKIP=decode` never writes layer 20's) | decode zeroes locally |
| compressed KV, sources 2, 8, 14 (ratio 2), 20 (ratio 1) | `comp_kv`, bf16 ROW_MAJOR `[pool_blocks, 1, 64, 512]` on every stage that selects from the source | entry `e` -> page `page_row[e // 64] * 64 + e % 64` (1 KiB) | `st.compress[src].ckv`, TP-dedup slab: 32 owners, 160 tokens a chip a 640-token row block | none (bf16 rows), page by page |
| index keys, same sources | `idx_key_cache`, bf16 TILE `[pool_blocks, 1, 64, 128]` | block `b` -> 8 tiles at page `page_row[b] * 8` (2 KiB each) | `st.compress[src].index_k` (`INDEX_K_SERVED=bf16`), SP-sharded TP-replicated: 8 owners (any of a row's 4 chips) | tilize 64 x 128 a block |
| `dense_kv` | decode's own copy of the first `index_topk` entries | — | — | decode runs `_sync_dense_entries` after "done" |

Volumes for one copy at 1,044,480 tokens: ratio-2 source 522,240 entries = 510 MiB KV + 127.5 MiB keys; source 20:
1020 MiB + 255 MiB; rings 20 x 128 KiB. About 3.2 GB a copy (the TCP path's 3.35 GB). The push writes one copy per
chip of every consuming stage (x4 replicas x consuming stages), so the fabric carries several times that; at the
xhost test's link count this is the number to measure first (phase 2).

metal_2 slab facts the push kernel relies on (`deepseek_v41_flash_d_p/tt/compressor.py` `TtServedRows`,
`tt/served.py` `CacheSlab`):

* a slab is the pages `batch * rows + local_row` of the serving engine's cache tensor `[slots x layers, 1, rows,
  width]`, bf16 ROW_MAJOR, ND-sharded one 32-token record a shard: read pages through a `TensorAccessor` of that
  tensor (as `tt/page_copy.py` does), not by computing addresses;
* natural entry `e` of a source: `hs = stripe / ratio` rows per owner per block, `period = owners * hs`; block
  `e // period`, owner `(e % period) // hs`, local row `(e // period) * hs + e % hs` (the inverse of
  `d_p_bridge._compressed_rows`);
* compressed KV (dedup): `hs` = 80 (ratio 2) / 160 (ratio 1): a 64-entry decode block spans two owners, fine for 1 KiB
  row pages; index keys (replicated): `hs` = 320 / 640, so a decode key block (64 entries) always lies on one owner
  row, which can tilize it alone.

## 5. Design

### 5.1 Protocol (`tt/prefill/d_p_wire.py`, version 3)

New op `prefill_push`: `{"op": "prefill_push", "have": H, "cache": bool, "targets": {...}}` + the ids blob. Reply:
zero or more `{"progress"}`, then `{"seq_len", "base", "cached", "seconds", "push_seconds", "queued", "pushed":
{"bytes", "pages"}}` and no blobs. The existing `prefill` op stays (fallback, and the reference for byte checks).

`targets` (built by decode, section 5.2):

```
{"mesh_id": 1,
 "slot": <decode slot of the session>, "seq_len": T, "first_entry": {src: e0},
 "stages": [{"stage": s, "chips": [{"coord": [s, c], "chip_id": ..., "banks": [[x, y, off], ...]}, ...],
             "rings": {li: addr, ...},                       # layers 0-19 on this stage
             "pools": {src: {"kv": addr, "keys": addr}}}],   # sources this stage selects from
 "page_rows": {ratio: [physical block of logical block 0, 1, ...]}}
```

### 5.2 Decode side (`tt/model.py`, `demo/server.py`)

* `DeepSeekV41Model.remote_commit_targets(session_id, total, start=0) -> dict`: validates like
  `commit_prefill_state` (lengths, alignment, `start == _session_pos`), `ensure_session_capacity(total - 1)`, checks
  no target block is shared (`block_refs`) or unmapped, and returns `targets`. Fabric node ids and bank tables come
  from `mesh.get_fabric_node_id` / `ttnn.cluster.get_dram_bank_table` (computed once at startup).
* `DeepSeekV41Model.finish_remote_commit(session_id, total)`: zero the rings the push does not write (layer 20,
  layers 21-39), `_session_pos = total`, `_sync_dense_entries`. Same end state as `commit_prefill_state`.
* `demo/server.py`: `_RemotePrefiller(_V41Prefiller)` for `--prefill-backend remote --prefill-server HOST:PORT`:
  `aligned()` keeps the "leave the last 128 tokens to decode" rule; `run()` does `user.activate()`, then
  `remote_commit_targets`, `PrefillClient.prefill_push`, `finish_remote_commit`, seeds the Engram history
  (`hash_session(slot, ids, 0)`), sets `user.pos`; no local prefill model, no resident-state snapshot/reload. On an
  error reply or a dead connection the session is reset and its blocks are NOT handed to another session until the
  prefill server has replied (a late write must not land in reused pages).
* Phase 3 runs it on the scheduler thread (other users wait, as with the local prefill); phase 4 makes it async.

### 5.3 Prefill side (`demo/prefill_server.py`, `tt/prefill/d_p_bridge.py`, new `tt/prefill/d_p_push.py`)

As written (`d_p_push.push(d_p, targets, slot)`; no `DPPrefill` method):

* Sources, all DRAM interleaved so one `TensorAccessorArgs` set serves them: per kv source the slab rows of the
  owner-stripe blocks the entries reach (`CacheSlab.slice_rows`, 1 KiB rows), the index-key slab rows tilized on the
  device (`ttnn.to_layout`: an owner stripe is a multiple of 64 rows, so a 64-entry block is 8 consecutive tiles in
  decode's order), and one TILE staging tensor of the rings (read on the host by `_window_rows`, tilized by
  `from_torch`, replicated: 2.5 MiB for 20 layers).
* Sender assignment: KV pages by their owner chip (dedup: chip-linear owner); key blocks and ring tiles by one of the
  chips holding them, rotated by destination chip so the replicas share the load.
* One kernel a sending chip (core (0, 0)): a DRAM work table of 16-byte runs (source | count << 8 | destination <<
  16, first source page, destination buffer, first destination page), one a decode cache block (KV: the block's
  entries within one owner stripe, so up to 64 rows; keys: the block's 8 tiles; a ring: its 64 tiles), read `K` = 32
  at a time, a run's pages staged `K` at a time; a destination table (fabric chip, mesh, bank table) read into L1
  once. (Phase 2 had one entry a page: 11.8M at 1M tokens, 3.24 s of host planning.) One connection: a remote mesh's route is the route to this mesh's exit chip
  (`fabric_set_unicast_route`), so every decode chip is behind the same first hop (the host checks it with
  `get_eth_forwarding_direction`); the route is rewritten when the destination changes.
* `PrefillService.prefill_push`: same lock / prefix-cache flow as `prefill`, then the push instead of
  `host_states` + `send_states`.

### 5.4 Synchronization

Phase 2-3: the TCP reply goes after `synchronize_device` on mesh 0 (all sender programs finished), as tested.
Phase 4 option: fused write + atomic increment on the last packet of each (sender, destination chip) stream into a
per-session counter in decode L1/DRAM; decode waits for the expected count before `finish_remote_commit`. Cheap, and it
turns the timing assumption into a check.

## 6. Work plan

Each phase ends with a two-rank run on hosts 40 and 41.

### Phase 0: prerequisites (no new feature code)

1. Host 41: check out this branch at `/big_home/ttuser/smanoj/metal_1_prefill`, `./build_metal.sh`, copy the venv
   setup. Both ranks then run the same commit; the xhost test's per-rank path overrides go away.
2. Decode on the prefill's fabric settings, single host: a payload option on decode's mesh open (done: device setting
   `fabric_payload_bytes`, env `DEEPSEEK_V4_FABRIC_PAYLOAD`, applied in `chat_cli.py:open_mesh_device`); decode already
   opens `FABRIC_2D_TORUS_XY` with `STRICT_INIT`, as the prefill does. Run the decode server's own tests with
   `DEEPSEEK_V4_FABRIC_PAYLOAD=6208`: functional
   (`demo/test_two_users_chat.py`, `test_server_users.py`) and perf (`test_server_perf.py`) against the current
   defaults. Gate: same outputs, perf within noise. If decode cannot run 2D, stop and revisit (section 8).
3. Both servers in one job: a launch script (done: `demo/run_d_p_servers.sh`, a bash case on `OMPI_COMM_WORLD_RANK`) that starts
   `prefill_server.py` on rank 0 and the decode server on rank 1 with the two-mesh MGD; settle `STRICT_INIT` vs
   `RELAXED_INIT`. Gate: both come up, the existing TCP `prefill` path works between them (prefill_client from 41).

### Phase 1: decode-side API, tested with a synthetic sender

1. `remote_commit_targets(session_id, total, ring_layers, start=0)` / `finish_remote_commit(session_id, ids=None)`
   (section 5.2; written). `targets` is JSON-ready: per stage its chips (fabric node, bank table), ring addresses of
   `ring_layers` and pool addresses of the sources it selects from, plus `page_rows` per ratio and the page sizes.
2. Test `tests/xhost_kv/decode_commit.py` (two ranks; written): rank 1 builds the real decode model (layers 0-20),
   commits seeded random states into session B with `commit_host_prefill_state`, publishes session A's targets;
   rank 0 regenerates the states and writes them with a work-list kernel (per page: destination buffer and page id,
   streamed from a DRAM table; decode chip `chip_id` served by mesh 0's chip of the same index); rank 1 runs
   `finish_remote_commit` and compares A, per stage and chip, with the states and with B: rings (written and
   zeroed), `_gather_paged_entries` (entries and keys) and `dense_kv`. Gate: byte-exact.

### Phase 2: prefill-side push from metal_2's caches

1. `tt/prefill/d_p_push.py` (section 5.3; written).
2. Test `tests/xhost_kv/d_p_push.py` (two ranks, real models; written): rank 0 prefills a seeded random prompt with
   `DPPrefill`, pushes it into session A and replies with `host_states`; rank 1 commits the host states into session
   B with `commit_host_prefill_state` and compares A with the states and with B (`decode_commit.check`). Gate:
   byte-exact at 16k and 128k; push time and GB/s logged. 16k: PASS, 189440 pages / 210 MiB to 32 chips of 8
   stages, device 671 ms (0.33 GB/s incl. launch and the kernel's compile), 1.7 s in all (0.93 s of it slicing /
   tilizing the sources). 128k: PASS, 1479680 pages / 1610 MiB, device 168 ms (10 GB/s over the 32 senders), 1.96 s
   in all (host: 1.40 s slicing / tilizing the sources, 0.35 s building the work tables).
3. Scale: 1M tokens; measure push time vs the host path (readback + send + decode commit). Gate: faster than the host
   path, or a profile saying where the time goes (exit chips, small packets, work-list streaming). 1044480 tokens:
   PASS, byte-exact; 11755520 pages / 12760 MiB, device 1.33 s (10 GB/s). Push path 6.6 s (push 6.52 s +
   `finish_remote_commit` 0.08 s) vs host path 13.1 s (readback 3.53 s + TCP 5.48 s for 3190 MiB +
   `commit_host_prefill_state` 4.09 s). The push is host-bound: building and uploading the work tables 3.24 s (11.8M
   16-byte entries), slicing / tilizing the sources 1.77 s; next: tables of block runs instead of pages (phase 4).

### Phase 3: servers

1. Protocol v3 `prefill_push` (5.1), `PrefillService.prefill_push`, `_RemotePrefiller` (5.2, synchronous).
2. `demo/prefill_client.py --push` stays a host-only client for the error paths; the end-to-end check is a decode
   server request: same first tokens as the TCP `prefill` backend for the same prompt (and, once 6.1 is fixed, as
   decode's own local prefill).
3. Failure handling: prefill error reply, prefill crash (ends the job: document), decode request cancelled mid-push
   (blocks stay reserved until the reply).

Written: `PrefillClient.prefill_push`, `PrefillService.prefill(targets=...)`, `server.py --prefill-push` (and
`--prefill-wait`, so rank 1 waits for rank 0's prefill server), `DeepseekV4Model.abort_remote_commit`;
`reset_session` / `close_session` refuse a session with a commit in flight. A prefill error reply aborts the commit
(the turn fails, the user is reset); a cancellation during the push is held until the reply, then aborts. A prefill
crash ends the `tt-run` job (both servers). Mocked in `tests/test_remote_prefill.py`. Launch:
`demo/run_d_p_servers.sh --hf <snapshot> -- ... --prefill-push`.

End-to-end (DSpark, 5 instances, 8.6k-token prompt, 64 greedy tokens): PASS, identical replies in push and TCP modes
for the first turn, a repeat (decode prefix-cache hit) and a follow-up turn (push continued from `have` 8832).

| | push | TCP |
|---|---|---|
| first turn (8576 tokens) | 2.97 s, push 115 MiB in 0.78 s, KV load 1986 ms (first, incl. compile), 0.99 s blocking the scheduler | 1.58 s, 30 MB sent ahead (0 s blocking), KV load 1274 ms |
| follow-up (2176 tokens from 8832) | 0.30 s, push 37 MiB in 0.03 s, KV load 26 ms | 1.08 s, 9.7 MB, KV load 799 ms |

The push is synchronous, so it blocks the scheduler where TCP sends ahead; async is phase 4.1.

### Phase 4: production

1. Async remote prefill on decode (other users keep decoding; a turn waits in a "remote prefill" state).
   Written: `_RemotePrefiller.start` reserves the pages at admission (scheduler thread) and sends `prefill_push` from
   a thread of its own (`_RemoteFetch` with `targets`), the way the TCP backend sends ahead; `run` only calls
   `finish_remote_commit`. A stop is held until the reply (the connection is not cut). A push whose turn ended is
   an orphan until its reply is in: `reap` (each scheduler round; blocking before an admission, which may reset or
   evict its slot) then aborts its commit and resets the user. Only users of one instance gain (each instance has
   its own scheduler): the check runs `--instances 1 --num-users 2`, one user streaming while another's long prompt
   is pushed.
2. Prefix cache: `have > 0` pushes entries from `have` on (key blocks rewritten whole from the block holding `have /
   ratio`); `start > 0` sessions (`commit_prefill_state(start=...)` semantics).
3. Arrival counters (5.4).
4. Bandwidth: per-link accounting from the route map, spreading destinations over exit chips, larger packets for KV
   (several 1 KiB rows a packet when destination pages are contiguous within a block). Written: the work table in
   runs of a cache block (section 5.3) instead of pages, ~64x fewer entries; byte-exact at 16k (210 MiB, plan 11 ms,
   device 743 ms); 1M to do.
5. The push as a C++ op, a launch per prefill chunk:
   `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/kv_fabric_push` (`KvFabricPush`, driven by
   `FabricPush` in `tt/prefill/d_p_push.py`). `begin` takes the request's layout (sources, destinations, per layer
   the owner stripes, decode block rows and pools); `push(end, last)` after each chunk is enqueued plans the runs that
   chunk completed in C++ (entries below `end / ratio`, key blocks whose 64 rows are all written; the last launch the
   rest, the partial key block and the rings), enqueues one program a chip on the prefill command queue and streams
   each chip's runs into an H2D socket (DEVICE_PULL: the FIFO in pinned host memory, the kernel pulls a 512 B page
   of 32 runs at a time). A launch is a header page (runs, destination-table bytes), the destination table, then the
   runs; nothing a launch moves is in the program, so every launch hits the program cache. The kernel reads metal_2's
   ND-sharded slabs in place through runtime-configured accessors (no slices) and tilizes the index-key blocks in L1
   (no device tilize). Next (option B): one persistent kernel on a reserved core, gated per chunk by a semaphore,
   several packets in flight and multi-page packets.

## 7. Separate track: prefill state correctness

`PREFILL_D_P_INTEGRATION.md` 6.1: metal_2 vs metal_1 prefill states diverge from layer 1 on. Suspects in order:
Engram (layers 1, 14; tokenizer normalisation of the n-gram hash), layer 0's MoE routing (hash vs score), the mHC
residual mix. Needs a CPU reference at 16k for both. Until it is fixed, end-to-end checks compare against the TCP
`prefill` path (same states), not against decode's local prefill.

## 8. Risks and open questions

* Decode on `FABRIC_2D_TORUS_XY` + 6208 B payload (phase 0.2): decode's CCLs, D2D sockets and migration sockets were
  built and tuned on `FABRIC_1D_RING_X`. If they fail or slow down, alternatives: a 2D fabric on decode with 1D-style
  CCLs, or ask whether the two-mesh setup can carry a 1D mesh 1 (unlikely: the packet header format differs).
* The one unexplained hang + PCIe drop (section 2). If it recurs under the push: triage on both hosts before killing,
  and suspect contention through the exit chips (rows 0-1) first.
* Coupled lifecycle: a crash of either server ends both (one MPI job). Decoupling needs a tt-metal change (a
  `DistributedContext` that replays the control plane's recorded init exchanges from a file) and is out of scope here.
* DRAM contention: the push writes decode's DRAM while decode serves other users; measure decode step time during a
  1M push.
* Address stability: targets are valid only while decode keeps the session's blocks and its cache buffers; decode
  must not reallocate caches (e.g. re-run `prepare_static_decode`) with a push in flight.
