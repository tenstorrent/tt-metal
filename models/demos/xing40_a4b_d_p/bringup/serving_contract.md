# Serving contract: Xing4.0-29B-A4B prefill under tt-d-gen

Read from: tt-d-gen @ 93e77b802999fdb4303e61195373fdfd76f48b3c (main, `git pull --ff-only` 2026-10-01); launch
harness and KV checks from tt-d-gen PR #1088 @ 7d4bee4ecdcebb2d8f90b8519e471319400781cb (not on main yet);
disagg_lb @ e001a53cef8b2d3d3ef45baf901755db5d429318, which pins tt-d-gen `sshon/mistral4-disagg-rebased-260929` @
ba4c33e3. Server paths are relative to tt-d-gen, the rest to tt-metal.

Geometry: 4x2 Blackhole mesh, FABRIC_2D, SP = 4 over axis 0, TP = 2 over axis 1, chunk C = 5120, max seq M = 56320
(11 chunks), 40 layers, W = C / SP = 1280 tokens per SP row per chunk. The tests are in
`models/demos/xing40_a4b_d_p/tests/bringup/contract/`; `server_rules.py` there is the server's planner, pad and
reshuffle ported to Python, with citations.

**Deployed geometry (2026-10-06): chunk 2048, max seq 4096, 1 slot** (W = 512), to pair with the frozen Xing decode
(1 slot x 4096 positions; bringup/serving.md section 7). The values live in tt/settings.py (`SERVE_CHUNK` /
`SERVE_MAX_SEQ` / `SERVE_SLOTS`, env `XING_SERVE_*`) and the contract tests run at them (server_rules.py SCENARIOS
holds each test's prompts per geometry). The rules below are geometry-free; the worked numbers are for C = 5120,
M = 56320.

## Input

**Build.** `prefill_chunk(input, kv, slot_id=, actual_start=, actual_end=, ...)` gets one chunk from the H2D stream:
a uint32 ROW_MAJOR DRAM tensor, global shape `[4, 1, 1280]`, sharded over axis 0 and replicated over axis 1. Treat it
as already in place: row r's 1280 tokens are the positions the server sent to row r. Do no host-side reorder and no
CCL before the embedding. Map the pad id `0xFFFFFFFF` into the vocabulary before `ttnn.embedding`, for example with
`ttnn.minimum(ids, V - 1)` or `bitwise_and(V - 1)` (V = 131072). Pad tokens come after every real token, so the
causal mask keeps them out of real rows.

**Where each token lands.** Absolute position g goes to SP row `(g // 1280) % 4`. Within a row, positions are in
increasing order. The start is any multiple of 32, not only of the chunk. For example, the chunk (2944, 8064) has
`c_start = (2944 // 1280) % 4 = 2` and `intra = 2944 % 1280 = 384`:
- row 2 holds [2944, 3840), then [7680, 8064)
- row 3 holds [3840, 5120)
- row 0 holds [5120, 6400)
- row 1 holds [6400, 7680)

When the start is a multiple of 5120 this is the identity: row r holds `start + [1280 r, 1280 (r + 1))`. RoPE
positions and the causal mask follow this placement, as DeepSeek's indexed RoPE and `ring_mla(kv_actual_isl=start)`
do (deepseek_v3_d_p/tt/mla/mla.py:899-923, 960-1050).

**Calls the server makes** (`server_rules.chunk_plan`, `interleave`):
- `base = align_down_32(resident)`, where resident = the reused prefix (0 on a cold admit).
- `n = max(1, ceil((prompt_len - base) / 5120))`. Chunk i is `lo = base + 5120 i`, `actual_end = min(lo + 5120, prompt_len)`.
- If the last chunk's padded end `lo + 5120` passes 56320, it is pulled back to `lo = 56320 - 5120 = 51200`.
- Slots interleave round-robin: a non-last chunk sends its slot to the back of the queue.

Examples:
- fresh 3000 tokens: (0, 3000)
- fresh 56320 tokens: (0, 5120) ... (51200, 56320)
- follow-up of 9000 tokens over a resident 2944: (2944, 8064), (8064, 9000)
- follow-up of 56000 tokens over a resident 2944: (2944, 8064) ... (49024, 54144), then the pulled-back (51200, 56000)

**Rules:**
- metadata `{slot_id, actual_start, actual_end}`, 3 uint32 (engine/include/engine/pipeline/pipeline_types.hpp:42-46)
- chunk plan engine/src/runtime/backend_runtime.cpp:841-850; per-chunk range and pull-back
  engine/src/runtime/prefill_writer.cpp:51-72; pad fill 93-97; round-robin 103-114
- PAD_ID 0xFFFFFFFF and TILE_ALIGNMENT 32: engine/include/engine/runtime/types.hpp:20-30
- reshuffle by `kv_offset = actual_start`: engine/src/pipeline/prefill_pipeline.cpp:146-152,
  engine/include/engine/runtime/ring_sdpa_reshuffle.hpp:9-68
- `chunk % sp == 0`, `chunk % 32 == 0`, `max_seq % 32 == 0`, `max_seq >= chunk`, prompt <= max_seq:
  prefill_pipeline.cpp:37-39, backend_runtime.cpp:97-105, 168
- H2D tensor spec and mapper: models/demos/common/prefill/runners/runner_utils.py:59-67, prefill_runner.py:60

tt-metal's `prefill_producer.py` is not the server. It does not reshuffle (`_chunk_to_host_array`), fills the pad with
real tokens, starts follow-up turns at a 32-aligned end and never pulls the last chunk back. It agrees with the server
only for chunk-aligned starts. The runner test wraps it so it sends the server's payload (`producer_case.py`).

**Tests:** `test_adapter_acks.py` (in process) and `test_runner_contract.py` (through the runner) send exactly this
payload; `test_cache_starts.py` drives the hooks model at these starts.

## KV cache

**Build.** One MLA latent cache: per token, 576 values = `kv_a_layernorm(latent 512) | RoPE(k_pe 64)`, the golden's
`kv_latent`. Store it as **bfp8_b TILE** (see the owner question). The record format follows from two constraints:
- `ring_mla` reads only TILE (ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp:133).
- The server's KV checks infer the format from the 32-token record size (tools/launch_harness/tables.py:129-140,
  PR #1088): 19584 B = bfp8 TILE, 36864 B = bf16 **row-major**.

A bf16 TILE cache, which Xing uses today (tt/runners/kv_contract.py:56-66), has the row-major size, so the harness
decodes it scrambled. Use DeepSeek's cache: `init_kvpe_cache` / `allocate_mla_kvpe_cache`
(deepseek_v3_d_p/utils/kv_cache_utils.py:990-1008, `MlaKvCacheFormat.BFP8_TILE`).

**Layout per chip:**
- `[num_slots * 40, 1, 56320 / 4, 576]`, batch index = `slot * 40 + layer`.
- Block-cyclic over the SP rows with period 5120: position g lives on row `(g // 1280) % 4`, local row
  `(g // 5120) * 1280 + g % 1280`. This is the placement from "Input", so a chunk's KV is written where it was
  computed (update_padded_kv_cache).
- Replicated over the 2 TP columns. Both copies must hold identical bytes, because the KV Manager may read either.
- One private region per slot and layer. No two records may share an address.

**Memory per chip:**

| dtype | per layer and slot | per slot (40 layers) | 16 slots |
|---|---|---|---|
| bfp8_b | 14080 rows x 612 B = 8.6 MB | 0.32 GiB | 5.1 GiB |
| bf16 | 16.2 MB | 0.60 GiB | 9.7 GiB |

The plan's non-KV footprint is about 8.4 GiB against a 27.2 GiB budget (bringup/serving.md section 6). Free the
attention modules' own geometry caches when serving: 0.6 GiB per chip, unused.

**Slot count:** `PREFILL_NUM_USERS` = worker `max_slots` = the decode side's slots. Every shipped prefill config uses
16 (models/deepseek-r1/dynamo.disagg.prefill.json, models/kimi-k2.7/dynamo.disagg.prefill.json; `max_slots`); the
harness rejects a table with fewer slots (tables.py:104).

**Tests:** `test_runner_contract.py` and `test_adapter_acks.py`: the harness's geometry decodes the records into the
golden KV, slots stay isolated, replicas are identical, no address aliases.

## Attention and cache writes

**Build.** Take DeepSeek's dense chunked MLA path as is (deepseek_v3_d_p/tt/mla/mla.py):
- `update_padded_kv_cache(cache, kv, slot_idx, layer_idx, num_layers, kv_actual_global=actual_start,
  cluster_axis=0, valid_global=actual_end)`. It derives each row's write offset from the start, so the c_start row
  writes its head and its tail into two slabs, and it writes only the records holding real tokens (mla.py:1362-1405).
- `ring_mla(..., kv_actual_isl=actual_start, logical_n=min(start + 5120, capacity), kv_cache_batch_idx=slot * 40 + layer)`
  (mla.py:1010-1050).
- RoPE and the q-side positions derived from `actual_start` (mla.py:899-923).

Drop the `actual_start % chunk == 0` assert (tt/attention.py:408, tt/runners/adapter.py:203).

Cases the model must get right:
- **Any 32-aligned start.** Follow-up turns start at the reused prefix: a multiple of `kv_block_size` (64 shipped,
  32 allowed). Example: 2944 = 46 blocks of 64; the chunk (2944, 8064) spans parts of 2 slabs on row 2.
  Rules: prefix_indexer.hpp:75-78 (`reusable_prefix_cap`), backend_runtime.cpp:461-560 (remount),
  backend_runtime.cpp:66-69 (kv_block_size % 32), prefill_writer.cpp:60.
- **Writes only [actual_start, actual_end).** Rows below the start belong to the reused prefix the server already
  shipped. Never rewrite them, and never write past the slot's 56320 rows (the clamp is `valid_global`).
- **Pad rows.** After the last real token, rows `[actual_end, ceil32(actual_end))` of the last record must be zero
  before the ack. The record ships whole; DeepSeek zeroes it with `zero_padded_kv_cache` (deepseek_v3_d_p/tt/kv_ack.py:57-105).
  The harness compares only real rows (kv_dump_compare.py:27-29), so zero is DeepSeek's convention that the decode
  side reads.
- **The pulled-back last chunk** rewrites part of the previous chunk: (51200, 56000) after (49024, 54144). That
  range is shipped twice (prefill_reader.cpp:94 queues [actual_start, actual_end) per ack), so the recompute must be
  deterministic: layer-0 rows are bit-identical, deeper layers match the golden.

**Hooks interface for the part tests** (`parts.py`). Add to the bring-up model, keeping the ladder's positional calls:
- `embed(tokens, start=0)`: tokens = one chunk in natural order, PAD_ID past the real tokens, laid out as the server
  lays out a chunk at `start`.
- `layer(i, h, start, state, end=None)`: `end` = actual_end.
- `state.load_prefix` / `state.to_torch` stay in natural order.

**Tests:**
- `test_cache_starts.py`, layers 0-1, three cases: cold with a mid-record end (0, 5120) / (5120, 8017); a 64-aligned
  follow-up (2944, 8064) / (8064, 9000); a 32-aligned follow-up (1312, 6432) / (6432, 7001).
- `test_pulled_back_chunk.py`: (49024, 54144), then (51200, 56000).

Pass limits:
- rows below the start bit-identical
- `[start, end)` vs golden with kv_dump_compare's per-channel PCC (nope 0:512, pe 512:576) >= 0.97: the stricter of
  the server's prefill-golden 0.93 (launch_harness validation.py PCC_DEFAULTS) and the spec's state threshold 0.97
- pad rows exactly 0
- layer-0 rewrite bit-identical

## Acks

**Build.** One ack per layer per chunk, in layer order, with the global layer index: 40 per chunk. The server counts
acks and retires a chunk after `layers_per_chunk` of them (prefill_reader.cpp:56-101, 121). A migrating slot ships
layer k over [actual_start, actual_end) at ack k (prefill_reader.cpp:68-95). Before ack k, layer k's rows for the
chunk and the zeroed pad rows must be in DRAM. Support both transports the runner wires:
- **Host sink**, the default: `set_layer_completion_sink(sink)`, then `sink(global_layer, request_id)` after the
  layer's writes have finished. Use an event or a synchronize; the KV Manager reads DRAM out of band
  (kv_ack.py:117-128; prefill_runner.py:686-706). The shipped runner manifests leave `PREFILL_LAYER_ACK_D2H` unset
  (engine/tools/manifests/*/runner_*_migrate.yaml).
- **D2H** (`PREFILL_LAYER_ACK_D2H=1`): `prefill_chunk(..., d2h_service=, metadata_msg=)`. Enqueue
  `ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h_service, metadata=metadata_msg)` on the same
  CQ after the layer's cache write and pad zero, with no host sync (kv_ack.py:107-115; prefill_runner.py:664-685).

Keep `num_kv_cache_layers(n) = n`: every Xing layer writes KV, and the producer and `LayerAckService` count acks with
it (prefill_producer.py:98, prefill_runner.py:640-663). Server config: `layers_per_chunk: 40`, `kv_num_layers: 40`
(models/deepseek-r1/dynamo.disagg.prefill.json for the shape).

**Tests:**
- `test_adapter_acks.py` (host sink): inside every ack it reads the layer's records through the table over UMD and
  checks them against the golden, the pad rows and the replicas; acks must come one per layer, in order, with the
  request_id passed through.
- `test_runner_contract.py` (D2H): 40 acks per chunk for 17 chunks. Snapshots taken right after the acks must match
  the golden and the final bytes.

## Adapter and table

**Build.** The runner calls these (models/demos/common/prefill/adapter.py, runners/prefill_runner.py):
- adapter: `load_hf_config()`, `weight_cache_path(mesh_shape)`,
  `allocate_kv_cache(*, mesh_device, hf_config, params) -> KvCaches`, `build_runtime(*, mesh_device, hf_config, params)`,
  `l1_small_size = 24576`, `num_kv_cache_layers(n)`
- `model_config.FABRIC_PAYLOAD_SIZE` in [4352, 15232] B: the runner opens the fabric with it as the packet payload
  (runner_utils.py:41-53). Xing's 3584 (tt/runners/adapter.py:44) is below an fp32 tile, 4096 B, so the mHC
  all_reduce gets 0 pages per packet and the runner dies with a SIGFPE building reduce_scatter (seen 2026-10-01). Use
  4352, the fabric default the bring-up ran with (tt_metal/fabric/erisc_datamover_builder.hpp:460-483).
- runtime: `.mesh_device`; `.config` (`is_first_rank`, `is_last_rank`, `use_trace`, `first_layer_idx`, `num_layers`);
  `compile(kv)`; `set_layer_completion_sink(sink)`;
  `prefill_chunk(input, kv, *, slot_id, actual_start, actual_end, request_id=0, d2h_service=None, metadata_msg=None) -> None`
  (prefill_runner.py:318-327)
- `kv_migration_base_address(kv) -> int`, or `kv_migration_stages(kv, first_layer_idx, num_my_layers)`
  (prefill_runner.py:766-773)
- `build_kv_chunk_table(kv, path, *, first_layer_idx=0, num_my_layers=None, stage_layout=None) -> path`. The runner's
  migration path passes those keywords (prefill_runner.py:796-830); the mock-only path calls `(kv, path=)` (line 721).
  Xing's current `(kv_cache, path)` fails the first.

**Table rules** (DeepSeek's builder is the model: deepseek_v3_d_p/tt/runners/kv_chunk_table.py):
- format_version <= 1, compression 0 or 1, `chunk_n_tokens == 32` (tables.py:18-24)
- the 576-wide MLA cache is config 0, and its record size names its format (tables.py:129-140)
- config 0 covers exactly layers 0..39 (tables.py:95-102); `num_slots >= max_slots` (tables.py:103-105)
- every (slot, layer, position < 56320) has a record of one size (`chunk_size_bytes`), and prefill and decode sizes
  match (migration_strategy_builder.cpp:120-160)
- the layer coverage per config is the same on both sides (migration_strategy_builder.cpp:103-118)
- `max_sequence_length >= 56320` (migration_strategy_builder.cpp:120-133)
- every chip of an entry's device group holds the bytes at its address, and the device groups' chips are named in
  `fabric_node_hosts` (tables.py:31-41)

**Tests:**
- `test_runner_contract.py`: the real runner and producer, all 40 layers, 2 slots, mock migration on the
  migration-enabled path, D2H acks. It runs `tables.read_table` / `layout` on the exported `.pb`, then dumps the KV
  through the table and checks it with `kv_dump_compare` (bytecmp, pcc).
- `test_adapter_acks.py`: the same table rules on a 2-layer runtime.
- `test_runner_smoke.py`: the intake smoke prompt through the runner, the KV below `(prompt_len - 1) // 64 * 64` read through the table into the device model as decode's prefix, tail recomputed, greedy answer must contain "Paris".

## Deployment

**Runner** (one process on the LoudBox, single rank):
- Launch with `python -m models.demos.common.prefill.runners.prefill_runner`. No shipped tt-run binding or mesh
  descriptor is 4x2: the p150_x8 descriptor is 2x4.
- Environment:
  - `PREFILL_MODEL=xing40_a4b_d_p`, `PREFILL_SP=4`, `PREFILL_TP=2`, `PREFILL_NUM_LAYERS=40`
  - `PREFILL_CHUNK_SIZE=5120`, `PREFILL_MAX_SEQ_LEN=56320`, `PREFILL_NUM_USERS=16`
  - `PREFILL_FABRIC_MODE=2d`: the default is `2d_torus_xy` (runner_utils.py:23-38), never use it on this box
  - `PREFILL_H2D_SERVICE_ID=<id>`, `PREFILL_ENABLE_MIGRATION=1`, `PREFILL_MIGRATION_TABLE_PATH=<shared path>`
  - `PREFILL_USE_TRACE=0` (no trace capture is built)
  - `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0`: a KMD 2.9.0 hang pinning a large embedding upload
    (disagg_lb "mistrall 4"/README.md:88-90)
- Fabric packet payload = `FABRIC_PAYLOAD_SIZE` (see "Adapter and table"): 4352 B.
- `PREFILL_MAX_SEQ_LEN` stays a multiple of the chunk. disagg_lb's runner needs max_seq > chunk (10240 / 5120).
  Xing's 56320 / 5120 meets both.

**Server worker (prefill role), mirroring models/deepseek-r1/dynamo.disagg.prefill.json:**
- `role: prefill`, `layers_per_chunk: 40`, `kv_num_layers: 40`, `max_slots: 16`, `max_seq_len: 56320`
- `chunk_size: 5120`, `kv_block_size: 64`, `min_copy_tokens: 256`
- `device.prefill.sp_factor: 4`, `device.prefill.ack_shm_name: /tt_prefill_layer_acks_<id>`

Do not rely on `runtime.chunk_aligned_start`. It exists only on disagg_lb's pinned tt-d-gen branch
(ba4c33e3 prefill_writer.cpp `base = chunk_aligned_start ? align_down(resident, chunk) : align_down_to_tile(resident)`,
where Mistral's runner needs it). Main always starts at `align_down_to_tile(resident)`, and this contract builds for
main. That branch's migration_strategy_builder differs from main only in logging and slab record sizing.

**Tests:** `test_runner_contract.py` runs the runner with these settings (2 slots instead of 16, mock migration
instead of a KV Manager). Its runner process gets `TT_METAL_OPERATION_TIMEOUT_SECONDS=600`: the runner blocks on its
H2D socket between chunks, and the safe runner's 5 s dispatch timeout would report that wait as a hang.

## Questions for the owner

1. **KV cache format for the decode side.** No Xing decode exists in tt-d-gen or tt-blaze yet. The options are:
   - bfp8_b TILE: what ring_mla and the harness both accept, and DeepSeek's decode format
   - bf16 row-major: harness-readable, but ring_mla cannot read it
   - bf16 TILE: today's format, which the harness misreads

   Default: **bfp8_b TILE**, with the RoPE half kept interleaved in checkpoint order like the golden. Changing the
   order waits until a decode side is chosen.
2. **Slot count (`max_slots` = `PREFILL_NUM_USERS`).** Default **16**, like every shipped config: 5.1 GiB of bfp8 KV
   per chip.
3. **Prefix reuse on the prefill worker.** Default **on, `kv_block_size: 64`**, as shipped. The model must then
   accept any 64-aligned start; the tests already require 32-aligned starts.
4. **Launcher.** Default: a single-rank `python -m prefill_runner` with the 4x2 mesh opened directly. Alternatively,
   a tt-run binding with a new 4x2 LoudBox mesh graph descriptor, if the deployment tooling (disagg_lb CLI) needs one.
