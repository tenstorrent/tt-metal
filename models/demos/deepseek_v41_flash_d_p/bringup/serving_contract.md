# Serving contract: DeepSeek-V4.1-Flash prefill under tt-d-gen

Read from tt-d-gen @ a286b6221da5 (main, 2026-10-14), tt-metal's prefill runner (`models/demos/common/prefill`, this
branch) and disagg_lb @ e001a53c (it pins tt-d-gen ba4c33e3). Line numbers are at those commits; tt-d-gen paths are
relative to its root. Tests live in `models/demos/deepseek_v41_flash_d_p/tests/bringup/contract/`; the rules they apply
are ported in `server_rules.py`, the cache geometry is `caches.py`.

Numbers used below: **deploy** = galaxy_4x4, SP 4 (mesh rows) x TP 4, chunk 8192, max seq 1,048,576, W = chunk / SP =
2048 tokens per SP row per chunk. **This run** = quietbox_2x2, SP 2 x TP 2, chunk 5120, max seq 56,320 (W 2560).
**Tests** = chunk 2048, max seq 4096 (golden rung s4096, every position dumped), W 1024 on this box.

## Input

Build:
- `prefill_chunk` gets the H2D tensor: uint32 ROW_MAJOR `[SP, 1, chunk / SP]`, sharded over mesh axis 0, replicated
  over axis 1 (prefill_runner.py:79, runner_utils.make_h2d_spec), plus the 12-byte header `{slot_id, actual_start,
  actual_end}` (pipeline_types.hpp:42-53; prefill_runner.py:201-207 decodes it). The runner passes the three as
  keywords.
- Ids past `actual_end` are PAD_ID 0xFFFFFFFF (types.hpp:20, prefill_writer.cpp:93-97). Mask positions >= actual_end,
  clamp the ids before `ttnn.embedding` (K-72); the engram hash must skip them too.
- Token order is `ring_sdpa_reshuffle` with kv_offset = actual_start (prefill_pipeline.cpp:146-153,
  ring_sdpa_reshuffle.hpp:9-19, 31-68): absolute position g goes to SP row (g / W) % SP. Row c_start = (start / W) % SP
  gets [start, start + W - intra) and then the chunk's last `intra = start % W` tokens; every other row gets one whole
  W block. Identity when start % chunk == 0.
  Deploy, start 2944: row 1 = [2944, 4096) + [10240, 11136), row 2 = [4096, 6144), row 3 = [6144, 8192),
  row 0 = [8192, 10240). Tests, start 960: row 0 = [960, 1024) + [2048, 3008), row 1 = [1024, 2048).
  Every local row is its absolute position for RoPE, the 128-token window, causal masks, compressor groups and engram
  n-grams.
- Every start is a multiple of 32 and nothing more: chunk bases are `align_down_to_tile(resident)`
  (prefill_writer.cpp:60-61, types.hpp:22-30); max_seq_len is a multiple of 32 and >= chunk (backend_runtime.cpp:124-129);
  chunk % SP == 0 (prefill_pipeline.cpp:37-39). disagg_lb's tt-d-gen (ba4c33e3) adds `chunk_aligned_start=true`, which
  snaps follow-ups to chunk multiples; main has no such option. Build for main.
- Engram (layers 1, 14): the hash at positions start..start+2 needs the ids of [start - 3, start), which are not in
  the payload. Keep per slot the ids by absolute position (written from each chunk's input) and read the three before
  `start` from it. On a prefill node a slot's earlier positions are always its own: a follow-up remounts the slot that
  holds the prefix (backend_runtime.cpp:527-557), and a prefill node never copies a prefix between slots (copy needs
  `kv_owned_by_control()`, decode only: backend_runtime.hpp:175, backend_runtime.cpp:616-618). d2p is the exception
  (see Questions).
- Device-model entry points (for the part tests; positional ladder calls stay valid): `model.embed(tokens, start=0)`
  (natural-order ids, PAD_ID tail, laid out as above), `model.from_host(h, start=0)` (block-input rows, same layout),
  `model.layer(i, h, start, state, end=None)` (`end` = actual_end), `begin_chunk(tokens, start)`. Hidden rows from
  `to_host` / `logits(hidden, rows)` are device-major in that layout (natural order when start % chunk == 0).

Tests: `test_cache_starts.py` (layer 0 from `embed` with the server's ids at starts 0, 960, 1312, 2048),
`test_adapter_acks.py` (the real payload through the runtime), `test_runner_contract.py`.

## KV cache

Build these served caches, in this table order (a config is present when a served layer has it):

| config | layers (full model) | rows per 32-token record | width | content (checkpoint inference/model.py) |
|---|---|---|---|---|
| 0_window_kv | every layer, 0-39 | 32 | 512 | window KV per position: kv_norm(wkv x), RoPE, fp8 q-dq (`Attention._window_kv`) |
| 1_compress_kv_r2 | 2, 8, 14 | 16 | 512 | latent per 2-token group, RoPE at the group's first token, fp4 q-dq (`_compress_kv`) |
| 2_index_k_r2 | 2, 8, 14 | 16 | 128 | index key per group: k_norm(wk latent), RoPE, fp4 (`Indexer`) |
| 3_compress_kv_r1 | 20 | 32 | 512 | as above, ratio 1 |
| 4_index_k_r1 | 20 | 32 | 128 | as above, ratio 1 |

Reader layers (3-7, 9-13, 15-19, 21-23, ...) own only 0_window_kv. topk, candidates and the compressor's partial group
are not served: topk / candidates are recomputed every chunk, and no partial group is pending at a chunk start
(a multiple of 32) or at the decode boundary (a multiple of 64), with ratio <= 2.

Rules:
- **Per position, never overwritten.** The KV Manager ships every acked layer's [actual_start, actual_end) by token
  position (prefill_reader.cpp:121-155), record by record (migration_strategy_builder.cpp:143-166), while later
  chunks run. A follow-up starts at any 64-aligned prefix (prefix_indexer.hpp:75-78) and needs the window KV of
  [start - 127, start). The reference's 128-slot ring (spec state window_kv) can serve neither: keep the window KV of
  every position. Later chunks may rewrite a position only with the same rows.
- **The record rule.** One table entry is one address and a size in ONE DRAM bank: `noc_addr = (bank << 32) |
  offset` (kv_manager/device_io/src/noc_addr.cpp:12-18, noc_addr.hpp:15-16), `offset % 64 == 0`
  (noc_address_resolver.cpp:18-22), size = the config's chunk_size_bytes (migration_strategy_builder.cpp:154-165).
  chunk_n_tokens = 32 for every config (tools/launch_harness/tables.py:37). So each 32-token record (32 / r rows x
  width) must be contiguous in one bank. Allocate every cache in DRAM, ND-sharded with shard `[1, 1, 32 / r, width]`,
  `ShardDistributionStrategy.ROUND_ROBIN_1D` over all DRAM banks, as DeepSeek's `init_kvpe_cache`
  (deepseek_v3_d_p/utils/kv_cache_utils.py:986-1080). Never DRAM interleaved (a page per row spreads a record over
  banks).
- **One format: bf16, ROW_MAJOR.** The owner's rule is a bf16 KV cache. A 16-row record cannot be a 32-row tile, and
  kv_dump_compare reads bf16 row-major records (kv_dump_compare.py:90-93). bf16 holds the reference values exactly
  (fp8 e4m3 / fp4 e2m1 times power-of-two or e4m3 scales), so the cache adds no error. The ladder, the contract and
  serving all run on these caches (report them in `state.formats()`).
- **One region per slot.** A slot's rows of a layer never alias another slot or layer.
- **Placement.** Shard the sequence over SP rows like the input: position g on row (g / W) % SP, local row
  (g / (W * SP)) * W + g % W. A compressed row j (tokens [j r, j r + r)) sits on the row of token j r, local row =
  that token's local row / r. TP: replicate (the table's device group lists every chip holding the same bytes) or
  dedup (single-chip groups). That choice is the plan's.
- **Memory per chip per slot (bf16).** Deploy at 1M: window_kv 40 x 262,144 rows x 1 KiB = 10 GiB when replicated
  over TP (2.5 GiB deduped). compress r2 3 x 131,072 x 1 KiB = 384 MiB, index r2 96 MiB, compress r1 256 MiB,
  index r1 64 MiB. Total ~10.8 GiB per slot (~2.7 GiB deduped). This run (56,320, layers 0-3,14,20,21,24): ~0.3 GiB
  per slot.
- **Interfaces.** `KvCaches.by_config = {config name: ttnn.Tensor}` in table order. On the device model,
  `state.served(layer, length) -> {config: tensor}` returns natural-order rows of the records covering
  [0, ceil32(length)). `state.load_served(layer, tensors, length)` and `state.load_tokens(ids)` load a migrated prefix.

Tests: `test_kv_layout.py` (memory configs, table vs allocation), `test_cache_starts.py` (read back through
`state.served`).

## Attention and cache writes

Build:
- **Every start the server sends.** Cold: (0, chunk), (chunk, 2 chunk), ... A follow-up starts at
  `base = align_down_to_tile(resident)`, `resident = min(align_down(prev_end, kv_block_size), reusable_prefix_cap)`
  (backend_runtime.cpp:1238-1247, prefill_writer.cpp:60-61, prefix_indexer.hpp:75-78, prefill_reader.cpp:177).
  kv_block_size is 64 in every shipped config (models/*/dynamo.disagg.prefill.json); 32 is legal.
  Deploy: a 3000-token turn, then 9000 tokens: resident 2944, chunk (2944, 9000).
- **The pulled-back last chunk.** When lo + chunk > max_seq, lo = max_seq - chunk (prefill_writer.cpp:63-67,
  tt-d-gen #430). It recomputes positions the chunk before it wrote. Deploy, a follow-up to 1,048,576 over 2944:
  ..., (1035136, 1043328), (1040384, 1048576), so [1040384, 1043328) is computed twice. Tests: (960, 3008), then
  (2048, 4000).
- **Write only [actual_start, actual_end).** Window rows of the chunk's positions. Compressed / index rows of the
  groups that complete inside the chunk: group j once (j + 1) r <= actual_end. Rows below start (the resident prefix)
  stay untouched. A position written again gets the same bytes where the computation is the same: layer 0's window
  KV depends only on the token and its position. Deeper layers must stay within the PCC limits.
- **Chunks that cross a block.** When start % W != 0, the c_start row's local rows wrap. Cache writes, the window
  gather and the compressor follow absolute positions, not local indices.
- **Context.** The window of position p is [p - 127, p]; the part before `start` comes from the cache. Compressed rows
  t < (p + 1) // r come from the source layer's cache, including this chunk's own groups. Indexer, candidates and
  top-k run per chunk.
- **Pad rows.** Rows at or past actual_end never enter attention, the compressor (an incomplete group at the end stays
  out), the indexer or a cache. The last record ships whole, so its rows past actual_end are zero in DRAM before the
  layer's ack: window rows [end, ceil32(end)), compressed rows from end // r to the record's end. That is what
  DeepSeek does: `update_padded_kv_cache(valid_global=actual_end)` clamps the write (deepseek_v3_d_p/tt/mla/mla.py:1403-1445),
  and `zero_pad_and_ack` zeroes the pad rows before the ack (deepseek_v3_d_p/tt/kv_ack.py). Start from it.
- **Compressor state.** Reset the partial group at every chunk start (no group is pending there). The group cut by a
  request's last actual_end is recomputed by the next turn, which starts at or before it.

Test: `test_cache_starts.py`, at chunk 2048 / max 4096. cold_mid_end: (0, 2048), (2048, 3017). follow_block: 1000,
then (960, 3000). follow_tile: kv_block 32, 1330, then (1312, 3001). pulled_back: 1000, then (960, 3008),
(2048, 4000). Checked for every served cache of the gate's block type.

## Acks

Build:
- **One ack per served layer per chunk, in layer order.** The engine's layers_per_chunk = the served layers (40).
  The k-th ack of a chunk ships layer k's [actual_start, actual_end) of every config covering that layer. After
  layers_per_chunk acks the chunk retires; on the last chunk PREFILL_DONE fires at actual_end
  (prefill_reader.cpp:121-160, 177-211). Every layer owns 0_window_kv, so every layer acks: no `kv_slot_layer_ids`.
- **Before layer k's ack, its records are final in DRAM:** window rows, its compressed / index rows, pad rows zeroed.
  An early ack ships stale KV (K-71).
- **Both transports.** Host sink: `set_layer_completion_sink(sink)`, then `sink(global_layer_idx, request_id)` after
  `ttnn.synchronize_device` (prefill_runner.py:856-870). Shipped manifests leave PREFILL_LAYER_ACK_D2H unset, which
  selects the sink (engine/tools/manifests/deepseek/runner_1rank_migrate.yaml). D2H (PREFILL_LAYER_ACK_D2H=1):
  `prefill_chunk` gets `d2h_service` and `metadata_msg`; enqueue
  `ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h_service, metadata=metadata_msg)` after the
  layer's writes, on the same command queue. The runner requires `set_d2h_ack_service(service)` whenever D2H is on
  and calls it when traced (prefill_runner.py:761-767, 846-847). Model: deepseek_v3_d_p/tt/kv_ack.py
  `zero_pad_and_ack`.
- Deploy: 40 acks per chunk, so 5,120 for a 1M prompt (128 chunks). Tests: 4 per chunk (layers 0-3).

Tests: `test_adapter_acks.py` (host sink, ack timing, order), `test_runner_contract.py` (D2H, counts).

## Adapter and table

Adapter: a `PrefillModelAdapter` registered as `deepseek_v41_flash_d_p` in
models/demos/common/prefill/adapter.py `ADAPTER_PATHS` (spec `contract.adapter`). What the runner calls:
- **Class attributes.** `model_config` with `NUM_LAYERS = 40` and `FABRIC_PAYLOAD_SIZE` in [4352, 15232]
  (runner_utils.py:19-62, K-79). `l1_small_size = 24576` (the spec's device_params; the runner opens the mesh with
  the adapter's value).
- **Factory methods.** `load_hf_config()`, `weight_cache_path(mesh_shape)`,
  `build_runtime(*, mesh_device, hf_config, params)` (prefill_runner.py:682), and
  `allocate_kv_cache(*, mesh_device, hf_config, params)`, which returns a `KvCaches` with `.by_config` (688).
  `PREFILL_NUM_LAYERS = N` builds layers [0, N): runner layer k is model layer k.
- **Runtime attributes.** `.mesh_device`, `.config` (is_first_rank, is_last_rank, use_trace, first_layer_idx,
  num_layers).
- **Runtime methods.**
  - `compile(kv)` (689).
  - `prefill_chunk(tokens, kv, *, slot_id, actual_start, actual_end, request_id, d2h_service=None,
    metadata_msg=None)` (437-447).
  - `set_layer_completion_sink(sink)` (859) and `set_d2h_ack_service(service)` (762, 847).
  - `kv_migration_stages(kv, first_layer_idx, num_layers)`: one `KvCacheStage(base_addr, first_layer, count)` per
    config, in table order (936).
  - `build_kv_chunk_table(kv, path, *, first_layer_idx=0, num_my_layers=None, stage_layouts=None) -> path`
    (955-990; also called as `(kv, path=...)`, 887).

Table rules (`ttnn.experimental.disaggregation.KvChunkAddressTable`):
- **Config fields.** The configs above, in that order, with those names: they sort in table order, because
  `KvChunkAddressTable`'s name-to-config constructor gives ids in sorted name order
  (ttnn/cpp/ttnn-nanobind/disaggregation.cpp:117-122). chunk_n_tokens 32. chunk_size_bytes = (32 / r) x width x 2.
  num_layers = the served layers. num_slots = max_slots, the same in every config (kv_chunk_table.cpp:41-72).
  max_sequence_length = PREFILL_MAX_SEQ_LEN. Compression UNROLLED or STRIDED_ROWS (tables.py:37).
- **Coverage.** Config 0 (0_window_kv) covers exactly the served layers 0..N-1 (tables.py:67-74). Every other config
  covers exactly its layers: a layer covered on one side only fails the migration
  (migration_strategy_builder.cpp:107-121).
- **Matching decode.** Names, ids and chunk_n_tokens must equal the decode side's table
  (kv_chunk_table_loader.cpp:26-51).
- **Entries.** Every (slot, layer, position < max_seq, step 32) has an entry of chunk_size_bytes
  (migration_strategy_builder.cpp:143-166; size 0 means no entry, proto_kv_chunk_table.cpp:79-82).
- **Hosts and slots.** Every chip has its host (tables.py:50-54), one host per (config, layer)
  (kv_chunk_table.cpp:100-112), slots >= the worker's max_slots (tables.py:75-77).
- **Device groups.** A device group is exactly the chips holding identical bytes (the KVM reads one, writes all).

The server's KV checks (tools/launch_harness/validation.py:18-22, 55-61; README 380-396):
- Prefill KV vs golden PCC >= 0.93. Ours is max(0.93, the spec's state threshold 0.97).
- Decode receives [0, cap), cap = (prompt_len - 1) // 64 * 64, compared byte for byte. It recomputes
  [cap, prompt_len), the last block, compared by PCC >= 0.99. Deploy: a 3000-token prompt gives cap 2944.

Tests:
- `test_runner_contract.py`: the real runner, mock migration, D2H acks, 2 interleaved slots, follow-ups, a near-max
  prompt pulled back. KV read back through the table and checked with tables.py and kv_dump_compare.
- `test_runner_smoke.py`: the decode stand-in.
- `test_kv_layout.py`.

## Deployment

- **Runner.** The tt-run rank binding's global_env (template: engine/tools/manifests/deepseek/runner_1rank_migrate.yaml):
  - `PREFILL_MODEL=deepseek_v41_flash_d_p`, `PREFILL_SP=4`, `PREFILL_TP=4`, `PREFILL_NUM_LAYERS=40`,
    `PREFILL_CHUNK_SIZE=8192`, `PREFILL_MAX_SEQ_LEN=1048576`. Max seq must be > chunk (disagg_lb).
  - `PREFILL_NUM_USERS` = the worker's max_slots, `PREFILL_H2D_SERVICE_ID=ds41_prefill`.
  - `PREFILL_FABRIC_MODE`: galaxy_4x4 uses `2d_torus_xy` (spec device_params; it needs a descriptor with RING on both
    axes, runner_utils.py:36-41), every other box `2d`. Never `1d`. The runner's default is `2d_torus_xy`, so set it on
    every box.
  - Mesh graph descriptor for the box. The LoudBox 2x4 one is `p150_x8_mesh_graph_descriptor.textproto` (disagg_lb).
  - `PREFILL_USE_TRACE=0` until `capture_trace` exists.
  - `PREFILL_ENABLE_MIGRATION=1`, `PREFILL_MIGRATION_EXPORT_TO_FILE=1`, and the table / device-map paths for the KVM
    fleet.
  - `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0`: disagg_lb's fix for a KMD 2.9.0 hang pinning large file-backed
    uploads. The engram tables stay memory-mapped from the checkpoint on the host.
- **Worker.** The models/*/dynamo.disagg.prefill.json schema: role prefill, `layers_per_chunk = kv_num_layers = 40`,
  `chunk_size = 8192`, `sp_factor = 4`, `max_seq_len = 1048576`, `kv_block_size = 64`, `max_slots`, `service_id`,
  `ack_shm_name = /tt_prefill_layer_acks_<service_id>`.
- **This box (tests).** quietbox_2x2, SP 2 x TP 2, `PREFILL_FABRIC_MODE=2d`, chunk 2048 / max seq 4096, 2 slots.
  - The runner subprocess gets `TT_METAL_OPERATION_TIMEOUT_SECONDS=600` (K-80).
  - tt-d-gen is not built here and `serving.require_engine` is false, so tt-metal's prefill_producer feeds the runner
    with the server's plan and payload (K-78).
  - The runner tests serve layers 0-3, the contiguous prefix of the spec's subset.

## Questions for the owner

1. **Decode-side cache format.** No DeepSeek-V4 decode exists in tt-d-gen or blaze yet. Default: bf16 row-major per
   position, lossless, the five config names and order above. Decode must use the same names, ids and record sizes.
2. **Window KV memory at 1M.** 10 GiB per chip per slot on galaxy_4x4 when replicated over TP. Default: TP-dedup
   (2.5 GiB). The alternative, fp8 rows, is lossless for this fp8 data but not yet in the server's tools.
3. **Slots.** Default max_slots 1 at 1M on galaxy_4x4 (the window KV above per slot, on top of the weights);
   2 in the tests.
4. **Prefix reuse.** Default on, kv_block_size 64. The model handles any 32-aligned start.
5. **d2p (decode to prefill).** Default off (`d2p_enabled = false`, engine/include/engine/runtime/types.hpp:96). A
   pulled prefix arrives without its token ids, which the engram hash needs. Turning it on needs a token-id config
   in both tables.
6. **Server tooling for this model.** `tables.layout()` knows only 576 / 128-wide caches (tables.py:106-117), and
   kv_dump_compare assumes 32-row records, while the ratio-2 configs have 16. Default: our tests decode with explicit
   geometry; tt-d-gen needs a DeepSeek-V4 entry before a launch-harness run.
7. **The runner smoke in this partial run.** It cannot answer, because it needs all 40 layers, yet K.1 requires
   `smoke_runner_ok == 1`. Default: run it on the deployment Galaxy and waive that metric for this run's K.1.
