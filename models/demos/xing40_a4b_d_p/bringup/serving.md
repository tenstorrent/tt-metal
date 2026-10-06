server: /localdev/dnijemcevic/tt-d-gen @ 24c381bb6d39530b8fa59163ecfe068a629bb3f7 (main, `git pull --ff-only`: already up to date)
tt-metal: 6883be1cc5789fea0a8a40a1b2528dbe2f08cfc8 (branch dnijemcevic/ernie45_prefill)
date: 2026-10-01

Serving contract for Xing4.0-29B-A4B (`xing40_a4b_d_p`), 4x2 Blackhole p150b, SP=4 (axis 0) x TP=2 (axis 1), FABRIC_2D,
chunk 5120, max seq 56320 (11 chunks), 40 layers. Server paths are relative to tt-d-gen, tt-metal paths to tt-metal.
Numbers below use W = chunk / sp = 1280.

**Deployed geometry (2026-10-06): chunk 2048, max seq 4096, 1 slot**, to pair with the frozen Xing decode (tt-blaze
xchin/xing4-integration @ 17edc80257: 1 slot x 4096 positions, bfp8 TILE 576-wide latent; its kernels are compiled for
that cache). The values live in tt/settings.py (`SERVE_CHUNK` / `SERVE_MAX_SEQ` / `SERVE_SLOTS`, env `XING_SERVE_*`);
the contract tests run at them (tests/bringup/contract/server_rules.py SCENARIOS). The rules below are geometry-free;
their worked numbers are for the bring-up target 5120 / 56320 (W = 1280); at 2048 / 4096, W = 512. Prompt plus
generation is at most 4096 tokens until decode is rebuilt with a deeper cache.

## 1. Prefill call sequence

**Answer.** The server (prefill role) plans a request once, at admission: `base = align_down_32(resident)` where
`resident` is the reused prefix (0 on a cold admit), `n_chunks = max(1, ceil((prompt_len - base) / chunk))`. Chunk `i`
is `lo = base + i * chunk`, `actual_start = lo`, `actual_end = min(lo + chunk, prompt_len)`. If the **last** chunk's
padded end `lo + chunk` would pass `max_seq_len`, it is moved back to `lo = max_seq_len - chunk` and re-sends part of
the already-computed prefix (workaround, tt-d-gen #430). The payload is always exactly `chunk` tokens, padded with
0xFFFFFFFF. Several slots are interleaved round-robin: after each non-last chunk the slot rotates to the back of the
prefill queue, so the device sees slot A chunk 0, slot B chunk 0, slot A chunk 1, ... Every chunk carries its own
slot_id.

Alignment rules: `chunk_size % 32 == 0`, `max_seq_len % 32 == 0`, `max_seq_len >= chunk_size`,
`chunk_size % sp_factor == 0`, `kv_block_size % 32 == 0` (0 = prefix indexing off). `actual_start` is only tile (32)
aligned in general, and block (kv_block_size, 64 in every shipped config) aligned after a prefix reuse. It is
**chunk-aligned only when the request is cold** (resident = 0). Prompt length <= max_seq_len (admit FATAL).

Concrete calls (chunk 5120, max_seq 56320):

| case | calls (actual_start, actual_end) |
|---|---|
| fresh 3000-token prompt | (0, 3000) |
| fresh 52000 (> max_seq - chunk = 51200) | (0,5120) ... (46080,51200), (51200,52000): 11 chunks, all chunk-aligned |
| fresh 56320 (= max_seq) | (0,5120) ... (51200,56320) |
| follow-up / remount, resident 2944 (prefix hit of 46 blocks of 64), prompt 9000 | (2944,8064), (8064,9000) |
| remount, resident 2944, prompt 56000 | (2944,8064), (8064,13184), ..., (49024,54144), then the moved-back last chunk (51200,56000) which overlaps the previous one by [51200,54144) |
| remount, resident 5056, prompt 6100 | (5056,6100) |

Prefix reuse is live on the prefill node whenever `kv_block_size > 0`: `prepare_admission` matches the longest
cached prefix (capped below the block holding the last token), remounts an idle holder slot, or copies from a busy one
(`min_copy_tokens`, 256 in the shipped configs). The cap means `resident <= (prompt_len - 1) // 64 * 64`.

Citations:
- chunk plan: engine/src/runtime/backend_runtime.cpp:841-850 (`setup_slot_for_prefill`, `align_down_to_tile`, n_chunks)
- per-chunk range and the #430 move-back: engine/src/runtime/prefill_writer.cpp:51-72; pad fill 93-97; round-robin
  `queue_.rotate()` 103-114
- PAD_ID / TILE_ALIGNMENT: engine/include/engine/runtime/types.hpp:20-30
- alignment FATALs: engine/src/runtime/backend_runtime.cpp:66-69, 96-106; engine/src/pipeline/prefill_pipeline.cpp:34-41;
  prompt bound backend_runtime.cpp:168
- prefix reuse: backend_runtime.cpp:461-560 (match, Remount, Copy), prefix_indexer.hpp:72-78 (`reusable_prefix_cap`)
- the server's own test of a remount starting at a non-chunk offset: engine/tests/test_prefix_reuse.cpp:356-363
- shipped kv_block_size 64: models/deepseek-r1/dynamo.disagg.prefill.json:31, models/kimi-k2.7/dynamo.disagg.prefill.json:31,
  models/deepseek-r1/sglang.standalone.prefill.json:16; kv_block_size 0 in the doc example
  adapters/sglang/docs/prefill_bringup.md:58

## 2. Input

**Answer.** Per chunk the H2D stream delivers one uint32 ROW_MAJOR DRAM tensor, global shape `[sp, 1, chunk/sp]` =
`[4, 1, 1280]`, sharded over mesh axis 0 and replicated over axis 1, plus a 12-byte metadata page
`{slot_id, actual_start, actual_end}` (uint32 each). Pad value 0xFFFFFFFF after the prompt's last token. Shutdown is
the all -1 metadata sentinel.

Placement: with `sp_factor > 1` the server rotates the payload on the host (`ring_sdpa_reshuffle`) so that position
`g` lands on mesh row `(g // W) % 4`; each row receives the positions it owns **in increasing order**, densely packed
(not at local index g % W). For a chunk-aligned start this is the identity: row r gets `[start + r W, start + (r+1) W)`.
For an unaligned start, `c_start = (start // W) % 4`, `intra = start % W`, and row c_start holds two runs:

| actual_start | row 0 | row 1 | row 2 | row 3 |
|---|---|---|---|---|
| 0 | [0,1280) | [1280,2560) | [2560,3840) | [3840,5120) |
| 2944 (remount) | [5120,6400) | [6400,7680) | [2944,3840) + [7680,8064) | [3840,5120) |
| 51200 | [51200,52480) | [52480,53760) | [53760,55040) | [55040,56320) |

This is exactly the block-cyclic cache layout with period = chunk (the latent cache's row r holds `j*chunk + r*W + [0,W)`),
so the KV can be written in place; the model must apply RoPE / causal masking with these per-row position runs. Pad
positions are always the highest positions of the window (`> actual_end`), so they stay causally after every real
token, but with an unaligned start they sit at the end of row c_start's column, not at the end of the chunk.

Citations:
- metadata layout: engine/include/engine/pipeline/pipeline_types.hpp:42-54
- submit + reshuffle call: engine/src/pipeline/prefill_pipeline.cpp:130-159; payload/metadata size checks 53-72
- reshuffle semantics: engine/include/engine/runtime/ring_sdpa_reshuffle.hpp:7-68; spec oracle (encounter order,
  dense) engine/tests/test_ring_sdpa_reshuffle.cpp:44-61
- tt-metal H2D spec: models/demos/common/prefill/runners/runner_utils.py:59-99; mapper Shard(0)/Replicate
  runners/prefill_runner.py:60; metadata decode and sentinel prefill_runner.py:53, 158-168
- DeepSeek MLA handles the rotated (mid-slab) start with the same ops: models/demos/deepseek_v3_d_p/tt/mla/mla.py:973-991
  (kv_actual_isl only has to be tile-aligned)

## 3. Acks

**Answer.** Two transports. Default (`PREFILL_LAYER_ACK_D2H=0`): the runtime calls the host sink
`sink(layer_idx, request_id)` once per KV-writing layer; the runner pushes `seq = request_id * num_ack_layers + ack_idx`
into a per-rank ring, a router reorders by seq and bumps the shared-memory counter `/tt_prefill_layer_acks_<service_id>`
that the server polls. `request_id` is the runner's chunk counter. With `PREFILL_LAYER_ACK_D2H=1` the acks are device
records sent through a D2HStreamService (`LayerAckService`); the runtime must enqueue them itself from the
`d2h_service` / `metadata_msg` arguments.

The server counts acks only (no layer id on the wire). Per ack: if the slot is migrating (PD_DISAGG), it queues and
immediately issues (when the destination is armed) a migration of layer `k` = ack ordinal within the chunk over
`[actual_start, actual_end)` (plus `[0, actual_start)` once, on chunk 0 of a remount). After `layers_per_chunk` acks it
retires the head chunk: indexes every full kv_block below `actual_end` into the prefix index, sets the slot position to
`actual_end`, and on the last chunk emits PREFILL_DONE (local) or seals the burst (migrating). A wrong
`layers_per_chunk` is silent: too high never retires (hang), too low retires early.

At each ack the layer's KV for `[actual_start, actual_end)` must be final in device DRAM, because the KV manager may
read it right away, out of band of the ttnn command queue (hence DeepSeek's flush before a host ack). The ack order
must be the table's layer order 0..L-1.

For Xing: 40 acks per chunk (one per layer), so `layers_per_chunk = 40` and `PREFILL_NUM_LAYERS = 40`.

Citations:
- server ack handling: engine/src/runtime/prefill_reader.cpp:46-158 (per-ack migration 66-97, retire 99-156)
- layers_per_chunk silent failure: adapters/sglang/docs/prefill_bringup.md:50-53; models/kimi-k2.7/dynamo-disagg-prefill.json:12-13
- tt-metal sink / seq: models/demos/common/prefill/runners/prefill_runner.py:113-155; transports and ack space 597-711
  (D2H 664-684, host sink 685-706); seq modulus / reorder note 620-628
- why the host ack needs a flush: models/demos/deepseek_v3_d_p/tt/kv_ack.py:57-69
- ack contract: models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md:171-187

## 4. Migration

**Answer.** Migration is per layer, per ack. For a migrating slot each ack queues layer k over the chunk's
`[actual_start, actual_end)`, clipped to the range the decode side announced, `[dst_from, dst_to)` =
`[matched, cap)` with `cap = (prompt_len - 1) // kv_block_size * kv_block_size` (decode's own kv_block_size, 64 in
the shipped configs). So the block holding the last prompt token, and with it the pad tail, is **never migrated**: the
decode side recomputes it. If cap == 0 (prompt <= one block) or the prefix is already resident, decode prefills
locally. The range goes to the KV manager unrounded; it rounds the start down to an entry boundary and covers every
entry the range overlaps (so a partial entry would be sent whole, stale rows included; with block-aligned ranges this
does not happen). Entry granularity = the table's `chunk_n_tokens` (32 for Xing). Reads come from the first fabric node
of the source device group (column 0 of each mesh row for Xing); writes go to every replica the destination's own table
lists. Bytes are copied verbatim: no dtype or layout conversion anywhere.

Example, prompt 40000, cold decode: decode asks [0, 39936); prefill chunks 0..7 send [0,5120) ... [35840,39936)
per layer as each layer acks (the last chunk (35840,40000) is clipped to 39936); decode recomputes [39936, 40000).
Remount on prefill with resident 2944: chunk 0's acks also queue [0, 2944) once.

Lead, not verified: the slot-copy arm (`min_copy_tokens`) sends one SLOT_COPY for layers [0, N) and the KV manager
may copy only `layerBegin` (subagent reading of kv_manager control_plane / data_plane_request_builder); only matters
with kv_block_size > 0.

Citations: engine/src/runtime/prefill_reader.cpp:88-95, 160-186 (clip + enqueue_layer), 212-215 (dst range);
engine/src/runtime/backend_runtime.cpp:488-521, 715-730 (`spec.from = matched`, `spec.to = resident`);
engine/include/engine/control/prefix_indexer.hpp:72-78; engine/src/kv_manager_clients/kvm_client.cpp:354-362;
kv_manager/src/control_plane/services/migration_strategy_builder.cpp:141-163 (round down, per-entry size check);
kv_manager/include/data_plane/kv_chunk_index.hpp:32-41 (overlap), kv_manager/src/data_plane/data_plane.cpp:656-667,
691-716 (read, verbatim write to all replicas); kv_manager/src/control_plane/maps/kv_chunk_index_builder.cpp:133-164
(entries of chunk_n_tokens; reader = group.front()); copy arm backend_runtime.cpp:531, 707, data_plane.cpp:1242-1294.

## 5. KV table / entry format

**Answer.** The KV manager is model-agnostic (no 576 / MLA / dtype knowledge). At startup it FATALs unless, within a
table, config ids are contiguous, `chunk_n_tokens > 0`, num_slots is equal across configs and each (layer, config)
has one owning host; across the source and destination tables, the config count, the name-to-id map and
`chunk_n_tokens` per config must match; the local table's entries must be 64 B aligned and inside their DRAM bank.
Per command it checks the layer range, the configs on each layer, slot / max position bounds and **equal size_bytes
per entry** (a mismatch fails the command; on the prefill side a failed burst condemns the slot). Not compared:
`chunk_size_bytes`, num_layers beyond the command's bound, num_slots across tables, device-group shapes, and the
engine's kv_block_size against `chunk_n_tokens` (works only because both are multiples of 32).

What source and destination must agree on, for Xing: config name "0" (one config), chunk_n_tokens 32, entry size
36,864 B (18 bf16 tiles of 32 x 576), the byte format inside an entry (bf16 TILE, `kv_a_layernorm(latent 512) |
RoPE(k_pe 64)`, rope interleaved in checkpoint order), layer index = ack ordinal = served-layer row 0..39 (decode
kv_num_layers >= 40), slot ids in [0, max_slots) on both sides. Xing's table: one device group per mesh row holding
both TP replicas, position -> (row, local row) block-cyclic with period = the served chunk, ROUND_ROBIN_1D over the
DRAM banks.

Citations: kv_manager/src/control_plane/maps/proto_kv_chunk_table.cpp:89-91; kv_chunk_table.cpp:43-73, 94-104;
kv_chunk_table_loader.cpp:580-627; kv_table_manager.cpp:194-233, 340-369; kv_manager/src/kv_manager.cpp:265-285;
migration_strategy_builder.cpp:90-163; kv_manager/src/data_plane/data_plane_utils.hpp:121-122; table paths
kv_manager/src/config/config.cpp:233-262, 625-650; tt-metal models/demos/xing40_a4b_d_p/tt/runners/kv_contract.py:9-21,
75-107 (entry_bytes 72-73).

## 6. Slots and memory

**Answer.** Shipped prefill configs: DeepSeek-R1 disagg 16 slots, DeepSeek-R1 SGLang standalone 4, Kimi-K2.7 disagg 16,
Kimi-K2.7 example 64; runner manifests kimi27 86, glm52 28; the engine default `PREFILL_NUM_USERS` is 2 and the
server's `RuntimeConfig.max_slots` default is 64. `max_slots` must equal `PREFILL_NUM_USERS` (and the decode side's
max_slots / blaze `--n-slots`); nothing cross-checks it.

Xing's latent cache per chip: `num_users x 40 layers x (56320/4) x 576 x 2 B` = **0.604 GiB per slot**
(16.2 MB per layer-slot). The plan's non-KV footprint is about 8.4 GiB per chip (9.63 GiB total minus the two 0.60 GiB
cache lines) against a 27.2 GiB budget, so about **31 slots fit**; 16 slots = 9.7 GiB. Also allocated but unused when
serving: each layer's own geometry latent cache (40 x 16.2 MB = 0.60 GiB) and the shared ring_mla gather scratch
([1,1,56320,576] bf16 replicated, 65 MB).

Citations: models/deepseek-r1/dynamo.disagg.prefill.json:29, models/deepseek-r1/sglang.standalone.prefill.json:14,
models/kimi-k2.7/dynamo.disagg.prefill.json:29, models/kimi-k2.7/dynamo-disagg-prefill.json:38,
engine/include/engine/runtime/types.hpp:73; tt-metal prefill_runner.py:80,
models/demos/deepseek_v3_d_p/tt/runners/manifests/kimi27.json:6, glm52.json:6;
models/demos/xing40_a4b_d_p/tt/runners/kv_contract.py:56-66; geometry cache tt/attention.py:137-148, scratch 150-160;
budget bringup/plan.md:8, 127-146.

## 7. Deployment

**Answer.** Two configs must agree.

Server worker config (`runtime` / `device.prefill`), schema as models/deepseek-r1/dynamo.disagg.prefill.json:26-64:
`role: prefill`, `layers_per_chunk: 40`, `max_slots: 1` (= PREFILL_NUM_USERS = the decode's slots), `max_seq_len: 4096`
(= the decode's cache depth), `chunk_size: 2048`, `kv_block_size` (64 shipped; see section 1 and the audit: 0 until the model takes unaligned starts),
`min_copy_tokens` (256 shipped; only acts with kv_block_size > 0), `kv_num_layers: 40` (migrate range [0, 40)),
`device.prefill.sp_factor: 4`, `service_id` = PREFILL_H2D_SERVICE_ID, `ack_shm_name: /tt_prefill_layer_acks_<service_id>`.
Dynamo validates `min_disagg_tokens > kv_block_size` (adapters/dynamo/tt_dynamo/config.py:649-654).

tt-metal runner env (rank binding `global_env`, tt-run does not forward shell PREFILL_*):
`PREFILL_MODEL=xing40_a4b_d_p`, `PREFILL_SP=4`, `PREFILL_TP=2`, `PREFILL_FABRIC_MODE=2d`, `PREFILL_NUM_LAYERS=40`,
`PREFILL_CHUNK_SIZE=2048`, `PREFILL_MAX_SEQ_LEN=4096`, `PREFILL_NUM_USERS=1`, `PREFILL_H2D_SERVICE_ID=<id>`,
`PREFILL_LAYER_ACK_D2H=0`, `PREFILL_USE_TRACE=0` (no capture_trace), migration: `PREFILL_ENABLE_MIGRATION=1`
(+ `PREFILL_MIGRATION_EXPORT_TO_FILE`, table / device-map paths) or `PREFILL_MOCK_MIGRATION=1` for the read-back
test. `PREFILL_XING_LAYERS` must be unset (or equal to all 40) in serving. Keep `XING_KV_CACHE_DTYPE` unset (bfp8): bf16
makes the record 36864 B and the KV Manager's pairing with the decode table fails. These three values are the
deployment geometry (tt/settings.py SERVE_*); the bring-up target (5120 / 56320 / 16) returns when decode is rebuilt
with a deeper cache, and max_seq must then stay a multiple of both the chunk and decode's 4096 stride.

Defaults that are wrong for this box: `PREFILL_SP=8`, `PREFILL_TP=4` (prefill_runner.py:74-75), fabric
`FABRIC_2D_TORUS_XY` when PREFILL_FABRIC_MODE is unset (runner_utils.py:23-38; owner rule is FABRIC_2D),
`PREFILL_MODEL=kimi_k2_7` (adapter.py:304; the doc says deepseek_v3_d_p, ADDING_A_PREFILL_MODEL.md:206),
`PREFILL_NUM_USERS=2`, `PREFILL_H2D_SERVICE_ID=ds_prefill`, run_pipeline_prefill.sh's galaxy host list and NIC
(run_pipeline_prefill.sh:5-6), and every shipped rank binding's mesh graph descriptor (galaxy; e.g.
engine/tools/manifests/deepseek/runner_1rank_migrate.yaml:6). No 4x2 LoudBox descriptor ships in either repo for the
runner. The runner opens the mesh with its own fabric router config (`max_packet_payload_size_bytes =
FABRIC_PAYLOAD_SIZE = 3584`, RELAXED_INIT; runner_utils.py:41-53), not the spec's device_params; that combination is
untested for this model. Ordering: runner first, then worker, then router (prefill_bringup.md:78-90); the worker and
runner must share a host.

## 8. This model in the server

**Answer.** Nothing for Xing: no `models/xing*` directory, no mention of Xing/TeleChat anywhere in tt-d-gen
(`grep -rniw xing|telechat`: no hits). Shipped models: deepseek-r1, glm-5.2, kimi-k2.6, kimi-k2.7 (models/README.md:5-10).
The decode side is tt-blaze (`blaze.models.cli --model ...`, models/deepseek-r1/dynamo.disagg.decode.json:99-120),
which is not in this repo, so its KV format and RoPE layout cannot be read here. The closest family (DeepSeek-V3 MLA)
serves a bfp8 TILE kvpe cache "to align with the decode KV cache" (tt-metal
models/demos/deepseek_v3_d_p/utils/kv_cache_utils.py:1001, 1021) with the rope half in the TT de-interleaved
convention (the producer re-interleaves it, prefill_producer.py:755-760; kimi_k3.py:65 opts out). Xing stores bf16
TILE with the rope half interleaved in checkpoint order (kv_contract.py:9-10), so a DeepSeek-family decode would not
read it as is. A Xing decode implementation must exist in tt-blaze first.

## Audit

Adapter: models/demos/xing40_a4b_d_p/tt/runners/adapter.py + kv_contract.py, binding tt/attention.py
(`TtMlaAttention.bind_cache`). Ranked.

### Blocks serving

1. **Unaligned chunk starts crash rank 0.** Server: after any prefix hit (remount or copy, kv_block_size 64 and
   min_copy_tokens 256 in every shipped config) chunks start at `align_down_32(resident) + i * chunk`, e.g.
   (2944, 8064) (section 1). Model: `assert actual_start % c == 0` (adapter.py:203) and `assert start % g.chunk == 0`
   (attention.py:395). The input placement the server sends for such a start (section 2) also breaks the adapter's
   "row r holds [r chunk/4, (r+1) chunk/4)" assumption (adapter.py:15-19). Fix, short term: deploy the Xing prefill
   worker with `kv_block_size: 0` (no prefix index, no remount, no copy; the migrate range comes from the decode side's
   own kv_block_size, backend_runtime.cpp:715-730, so disagg still works). Fix, model: support the rotated start the way
   ttMLA does with the same three ops Xing already calls (rotary_embedding_indexed with the start,
   update_padded_kv_cache `kv_actual_global`, ring_mla `kv_actual_isl`; deepseek_v3_d_p/tt/mla/mla.py:973-1030), make
   the mHC / MoE / embedding path position-agnostic for the two-run row (they are per token, so likely only RoPE and
   the causal offset matter), and accept the #430 overlapping last chunk (prefill_writer.cpp:63-67). Then drop both asserts.
2. **`PREFILL_ENABLE_MIGRATION=1` raises TypeError before serving.** The runner calls
   `runtime.build_kv_chunk_table(kv_caches, table_path, first_layer_idx=..., num_my_layers=..., stage_layout=...)`
   (prefill_runner.py:786, 796-830); Xing's signature is `build_kv_chunk_table(self, kv_cache, path)` (adapter.py:212).
   Only the mock-only path (prefill_runner.py:721, 851) and the contract test (contract.py:138) call it with two
   arguments. Fix (adapter): accept `first_layer_idx`, `num_my_layers`, `stage_layout(s)` keyword arguments (single
   rank: assert first_layer_idx == 0 and the stage's base address == kvpe.buffer_address(), then build as now).
   The same signature is shared by ernie45, gemma4, glm53_flash, hy4_preview and both mimo adapters.
3. **No decode side exists for Xing** (section 8). The entry format (bf16, 36,864 B, interleaved rope) is what the
   KV manager will copy verbatim, and a per-entry size mismatch fails every migration command
   (migration_strategy_builder.cpp:152-163). Blocked on outside information: which decode, which format.

### Wrong results

4. **None found on the chunk-aligned path.** The block-cyclic table matches the server's placement (period = chunk,
   chip_of kv_contract.py:37-41 vs the reshuffle), acks carry the global layer index in layer order (adapter.py:169-176),
   the layer's KV is device-complete at the ack (`event_synchronize` before the sink, adapter.py:173-175), and pad ids
   are clamped into the vocab and sit causally after every real token (adapter.py:144-154).
5. **Config footgun, silent hang:** `PREFILL_XING_LAYERS` (adapter.py:78-88) narrower than `PREFILL_NUM_LAYERS` makes
   the runtime emit fewer acks than the runner's seq modulus, so the reorder buffer waits forever
   (prefill_runner.py:620-628, 656-663) and the server never retires a chunk. Fix: leave it unset in serving, or
   have the adapter assert `len(served_layers) == params.num_layers` outside the bring-up.
6. **`PREFILL_LAYER_ACK_D2H=1` hangs:** prefill_chunk ignores `d2h_service` / `metadata_msg` (adapter.py:190-209), so
   no device ack records are sent. Keep the host sink (the default) or emit the D2H record per layer.
7. **Pad KV is not zeroed** past actual_end (DeepSeek zeros it, kv_ack.py:57-60). Harmless today: migration stops at
   the decode cap, below the block with the last token (section 4), and the next chunk overwrites it. It matters only if
   a consumer ever reads the partial block.

### Perf / memory

8. One host `event_synchronize` per layer, 40 per chunk (adapter.py:175). Correct for a host ack; the D2H transport
   (device-ordered ack, no host sync, kv_ack.py:61-64) is the fix if the stalls show up in the chunk time.
9. 0.60 GiB per chip of geometry latent caches allocated by `setup` and unused while serving
   (adapter.py:113-115, attention.py:137-148): about one slot's worth.
10. Fabric router config in serving is `max_packet_payload_size_bytes = 3584` with RELAXED_INIT
   (runner_utils.py:41-53), not the spec's device_params that every bring-up test uses.

### Untested by models/demos/common/bringup/testing/contract.py

- runs the first ladder rung (s4096, chunk 2048) by default (contract.py:41-43), not the target 56320 / 5120;
- only chunk-aligned starts `c * chunk` (contract.py:162-166): no remount start, no reshuffled input, no #430 overlap chunk;
- one slot, chunks in order: no interleaving of slots (prefill_writer.cpp:113), no check that other slots stay intact;
- tail pad of 32 tokens, so the last chunk ends on a block boundary: no partial block;
- builds the table with two arguments (contract.py:138), so it cannot see audit item 2; no real KV manager or
  migration, no D2H acks, no runner (`open_mesh_device` fabric config, H2D service, tt-run binding).

## Questions for the owner

1. **Which decode implementation serves Xing, and what KV entry does it read?** dtype (bf16 vs bfp8_b), RoPE half
   layout (interleaved checkpoint order vs the TT half-split), 32-token entries, config name "0", TP-replicated or
   deduplicated cache, layer count. Default suggested: keep bf16 interleaved (what the prefill writes and validates
   now) until a decode side is chosen; if a DeepSeek-family tt-blaze decode is reused, it expects bfp8 TILE and the
   half-split rope (kv_cache_utils.py:1001, 1021), and the prefill cache must change to match.
2. **Prefix reuse on the prefill worker: on or off?** Default: off (`kv_block_size: 0`) until audit item 1's model
   change lands; then 64 like the shipped configs.
3. **Slot count (max_slots = PREFILL_NUM_USERS).** Default 16 (9.7 GiB of KV per chip); about 31 fit the 27.2 GiB budget.
4. **Deployment shape:** disaggregated (Dynamo prefill + decode pair, like models/deepseek-r1/dynamo.disagg.*) or a
   standalone prefill worker driven directly? Default: Dynamo disagg, mirroring DeepSeek-R1.
5. **tt-run mesh graph descriptor for the 4x2 LoudBox** (no shipped binding fits; the p150_x8 descriptor is 2x4,
   spec.yaml box.mesh note). Default: a single-rank binding with a 4x2 descriptor and PREFILL_FABRIC_MODE=2d.
