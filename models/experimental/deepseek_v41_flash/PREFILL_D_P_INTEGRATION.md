# DeepSeek-V4.1-Flash prefill: metal_2's whole-mesh prefill vs metal_1's pipelined prefill, and how to integrate

Status (2026-11-20): design written; merge of metal_2's branch (`dnijemcevic/model_bringup_batching`) into this
branch done (`smanoj/ds41_d_p_prefill`); host-path state bridge written (`tt/prefill/d_p_bridge.py`); device
validation pending. Read "The blocker" before planning server work.

## 1. The two implementations at a glance

| | metal_2: `models/demos/deepseek_v41_flash_d_p` | metal_1: `models/experimental/deepseek_v41_flash/tt/prefill` |
| --- | --- | --- |
| Mesh | one 8x4 Blackhole Galaxy mesh, `FABRIC_2D_TORUS_XY` | eight `1 x 4` submeshes (pipeline stages) of the same Galaxy |
| Parallelism | sequence-parallel (SP) over mesh rows (axis 0), tensor-parallel (TP) over mesh columns (axis 1); every chip works on every layer | pipeline over stages (five layers a stage), TP4 inside a stage; stages work on consecutive chunks |
| Chunk | 5120 tokens (640 a mesh row) | 1024 tokens (`--prefill-trace-chunk`) |
| Context | 1,044,480 deployed, 1M measured | `--prefill-max-len` (the context when unset); two-level select now written but not validated past 16384 |
| Execution | eager per chunk, cached programs, no trace; host enqueue overlaps device | one trace per stage and position tier, D2D sockets between stages, H2D packet, D2H logits |
| Attention | `ttnn.bringup.sparse_sdpa` over the window + the indexer's top-512 entries; two-level candidate select at layer 20 | `deepseek_v4_prefill_sdpa` over window + selected entries; two-level select phase 1-2 written |
| MoE | `MOE_BLOCK_V2` fused block, routed experts bfp4, expert-parallel over the whole mesh | `fused_experts_prefill`, routed experts shared in place with decode (bfp4) |
| Engram | host n-gram hash + memory-mapped table (device path behind `DS41_ENGRAM_DEVICE`) | host `EngramHostLookup`, rows over one H2D socket a device |
| Layers run | `DS41_DECODER_SKIP=decode`: layers 0-19, layer 20's KV / index head only, 21-39 skipped | `kv_only=True`: layers 0-20, no head |
| Output | KV written in place into engine-owned served caches (`tt/runners/kv_contract.py`) + per-layer acks | per-layer `PrefillAttentionState` tensors handed to `commit_prefill_state` |
| Hand-off | tt-d-gen KV Manager migrates records by address table (disaggregated decode) | `commit_prefill_state` writes rings and paged pools of the same process's decode |
| Ops | `ttnn.bringup.*` fork ops (sparse_sdpa, flat_routed_expert, moe_block, ckv_bfp4, mhc_post, update_padded_kv_cache, topk_large_indices, fabric_reduce_scatter, ...) | `ttnn.transformer.deepseek_v4_prefill_*`, `fused_experts_prefill` |

Measured (metal_2, 8x4 Galaxy, warm request, `test_full_prefill.py`, 200 chunks of 5120):

| Prompt | Wall | Throughput |
| --- | --- | --- |
| 51,200 | 0.80 s | 63.9k tok/s |
| 256,000 | 4.13 s | 62.0k tok/s |
| 1,024,000 | 17.49 s | 58.6k tok/s (device busy 15.36 s; 0 of 200 chunks host-bound) |

## 2. How metal_2's prefill works

### 2.1 Layout

The whole Galaxy is one mesh; SP = 8 rows, TP = 4 columns, both read from `mesh.shape`, so smaller meshes run the
same code. A 5120-token chunk is split into 8 row blocks of W = 640 tokens: token g of the chunk lives on mesh row
`(g / W) % SP` (`tt/attention.py:WindowGeometry`). The engine sends token ids in its own "server order"; the
runtime puts them back in natural order with `WindowGeometry.server_rows`.

Within a row, TP splits projections and attention heads over the 4 columns. KV records are either TP-dedup (each
column holds a 160-token stripe of the row's 640: window KV, compressed KV) or TP-replicated (index keys, which the
indexer on every column needs).

### 2.2 Layer kinds

From the checkpoint config (`reference/ds41_ref.py:Config`, `bringup/spec.yaml` `block_types`):

| Block type | Layers | Role |
| --- | --- | --- |
| `swa` / `swa_engram` | 0 / 1 | window-only attention (ratio 0); layer 1 adds engram rows |
| `enc_source` / `enc_source_engram` | 2, 8 / 14 | ratio-2 KV + index source (14 adds engram) |
| `enc_reuse` | 3-7, 9-13, 15-19 | read their group's compressed KV and top-k |
| `dec_candidate` | 20 | ratio-1 KV source + candidate pool (2048 blocks of 8) |
| `dec_reindex` | 24, 28, 32, 36 | own top-512 inside layer 20's candidates |
| `dec_reuse` | the rest of 21-39 | reuse the decoder sources' KV / top-k |

Each block runs the reference step graph (`block_steps`): engram, attention mHC + collapse + norm, `q_a` / `kv`,
compressor (sources), indexer (index sources), attention, residual, FFN mHC + collapse + norm, router, experts,
shared expert, `moe_add`, residual. Between layers the hidden state is a `DevH` (4 streams + the pending `pre` mix);
with `OWN_ROWS=on` each chip holds 160 whole token rows (5120 / (SP x TP)).

The residual is the mHC 4-stream hyper-connection (`tt/mhc.py`, `ttnn.bringup.mhc_post`).

### 2.3 One chunk (`tt/runners/adapter.py:DS41PrefillRuntime._run`)

1. Read the engine's uint32 ids back once (host engram input), natural order, embed.
2. `begin_chunk`: per-slot id history, engram tails, `global_write_start` (rows below it are not rewritten).
3. Layer by layer, `model.layer(i, h, start, state, end=run_end)`:
   * window K=V rows written through `update_padded_kv_cache` into the slot's 128-row ring, plus the served
     128-row snapshot at `cap` (`DS41_WINDOW_KV_KEEP=snapshot128`);
   * KV sources write compressed KV (bf16, `CKV_FORMAT`) and index keys (`INDEX_K_SERVED=fp8_row192`: Blaze
     decode's 192-byte rows; `bf16`: plain rows) into the served caches;
   * the indexer scores every visible entry, `topk_large_indices` keeps 512; layer 20 also keeps 2048 candidate
     blocks of 8 for the re-index layers;
   * one `sparse_sdpa` over `[128 carry | chunk window | gathered compressed rows]` with a 640-entry index list
     (128 window + 512 top-k) and per-head attention sinks; the window path uses stock
     `ttnn.transformer.sparse_sdpa` unless a low-bit window format routes it to `ttnn.bringup.sparse_sdpa`. History
     from other SP rows arrives through `high_bw_all_gather` (`history_gather`) into a persistent buffer, not the host;
   * MoE: sqrtsoftplus grouped top-6 of 384 experts (`moe_grouped_topk`), then the shared expert and bfp4 routed
     experts in one fused `ttnn.bringup.moe_block` call (dispatch groups = mesh columns), combine over fabric;
   * TP partial sums are exact fp32 (`fabric_reduce_scatter` / all-gather + add); the mHC mix uses
     `mhc_pre_xing` (Sinkhorn) and the fused `mhc_post`;
   * then the layer's ack (`sink.layers_completed` after a device sync, or a D2H socket op with no host sync).
4. No host round trip inside a layer (the bring-up gate checks `host_transfers_per_layer == 0`).

`DS41_PREFILL_END=cap` computes only up to `cap = actual_end // 64 * 64`; decode replays `[cap, prompt_len)`.
`DS41_DECODER_SKIP=decode` marks every chunk "not last": layers 21-39 never run in prefill and layer 20 runs only
its KV / index head (`TtDS41Block.kv_only`), because the decoder layers only read layer 20's compressed KV and
decode rebuilds their windows by replaying the tail.

### 2.4 What metal_2 leaves behind

Per slot, a `DS41DeviceState` (`tt/model.py`) bound to the engine-owned served caches:

* `window[i]`: `TtWindowCache` of every layer (the 128-row ring, position p at row p % 128);
* `compress[i]`: `TtCompressCaches` of every KV source (compressed KV + index keys, SP x TP striped);
* `to_torch(layer, length)` reads any of it back in natural order (`window_kv`, `compress_kv`, `index_k`, ...).

The serving form is the address table (`kv_contract.py`): five configs (`0_window_kv`, `1_compress_kv_r2`,
`2_index_k_r2`, `3_compress_kv_r1`, `4_index_k_r1`), 32-token records, each mapped to a (chip, DRAM bank,
address), which tt-d-gen's KV Manager reads and migrates to a disaggregated decode.

## 3. How metal_1's prefill works

* `DeepSeekV41Model.build_prefill` builds `DeepSeekV41PrefillModel` on decode's own stages: each prefill layer on its
  decode layer's `1 x 4` submesh, reading that layer's routed experts in place; the embedding, `lm_head`, D2D socket
  pairs and the Engram lookup are shared with decode.
* `DeepSeekV41TracedPrefill` captures one trace per stage per position tier; stages pass `(streams, pre_mix)` and
  compressed KV a later stage reads over D2D sockets.
* Prefill shares decode's command queue and overwrites decode's L1-resident tensors: the server snapshots them
  (`snapshot_resident_state`) and writes them back after every prefill (`reload_resident_state`).
* Output: per layer a `PrefillAttentionState` (`deepseek_v4_flash/tt/prefill/attention.py`), ROW_MAJOR bf16,
  replicated over the stage's 4 chips: `kv_tail [1, 1, 128, 512]`, and for a KV source `compressed_kv [1, 1, E, 512]`
  and `idx_keys [1, 1, E, 128]`, `E = T / ratio`.
* `commit_prefill_state(states, sid)` writes each layer's ring into decode's sliding cache (`fill_cache`) and each
  source's entries / keys into the paged pools of every stage that selects from them (TT-Fabric migration sockets
  between stages, host fallback). It reads nothing else: no compressor partial (T is a multiple of 128) and no
  candidate state (decode recomputes candidates every step).
* `--prefill-kv-only` (the server default): layers 0-20; the server always leaves the last 128 prompt tokens to
  decode, whose replay fills every layer's window (21-39 included) and gives the first token's logits.

## 4. Contrast

* **Parallel decomposition.** metal_2 puts the whole mesh on one chunk (SP x TP); per-chunk latency is low and every
  chip is busy on every layer. metal_1 pipelines layers over stages; one chunk only uses one stage at a time, and
  throughput comes from overlapping chunks. metal_1's prefill weights are decode's own (no extra DRAM); metal_2
  needs its own copy, laid out for expert parallelism over 32 chips.
* **Long context.** metal_2 is built and measured to 1M (sparse attention everywhere, history over fabric).
  metal_1's dense-window + selected-entry path was capped at 16384 entries; the two-level select that lifts that is
  written but not validated on device.
* **Trace vs eager.** metal_1 replays captured traces (fixed buffers, position tiers). metal_2 is eager with cached
  programs and hides host enqueue behind the device (0/200 chunks host-bound at 1M).
* **Hand-off contract.** Both stop after layer 20 and let decode replay a tail: metal_2 `DECODER_SKIP=decode` +
  `PREFILL_END=cap` (cap = 64-aligned); metal_1 `kv_only` + the last 128 tokens. The data decode needs is the same
  (window rings of 0-20, compressed KV + index keys of the KV sources), but metal_2 serves it as address-table
  records striped over 32 chips, while metal_1 hands over contiguous per-layer tensors on the decode stages.
* **Index-key format.** metal_2 serves fp8 192-byte rows by default (Blaze decode's format); metal_1's decode
  takes bf16 keys. The bridge runs metal_2 with `DS41_INDEX_K_SERVED=bf16`.
* **Layer 20's window.** Under `DECODER_SKIP=decode` metal_2 never writes layer 20's window ring (FIX.28); metal_1's
  kv_only prefill does. The bridge hands decode a zero ring for layer 20, as metal_1 already does for 21-39; the
  128-token decode replay rewrites it.

## 5. The blocker: one process, two allocators

A parent `MeshDevice` and each submesh created from it get their own `L1BankingAllocator`
(`tt_metal/distributed/mesh_device.cpp`, `MeshDeviceImpl::initialize`). Nothing reconciles them: a tensor
allocated on the 8x4 parent and one on a `1 x 4` stage submesh can get the same DRAM / L1 address on the same chip.
metal_1's decode lives on stage submeshes covering all 32 chips; metal_2's model needs the whole 8x4 mesh. Loading
both in one process (the obvious "swap the prefill backend" integration) would corrupt memory. Options:

1. **Disaggregated (metal_2's intended deployment).** metal_2 prefill on its own mesh / host process, KV migrated
   through the address table (tt-d-gen), metal_1 decode imports the records into its paged pools. Needs a second
   Galaxy (or a split one) and a KV import path in metal_1 decode.
2. **Split the Galaxy.** metal_2 on a 4x4 submesh (rows 0-3), metal_1 decode on 4 stages (rows 4-7). Disjoint chips,
   so separate allocators are safe. Costs: decode drops from 8 stages to 4 (10 layers a stage), metal_2 loses half
   its SP and its axis-0 torus wrap (its ring CCLs must work on a 4-row line), and 21 layers of weights on 16 chips.
3. **Shared address space.** Make a submesh allocate from its parent's allocator (or reserve each other's ranges)
   in tt-metal. Most general, a core change.
4. **Sequential, offline (validation only).** One process runs metal_2 and dumps host states; another builds
   metal_1 decode and commits them. Not a server, but it validates the whole data path without solving 1-3.

## 6. What this branch does

* Merges metal_2's branch (fork ops, `models/demos/deepseek_v41_flash_d_p`, common prefill engine). Conflicts:
  binary_ng shard-shape (metal_1's tile-aware `get_shape_dims` + metal_2's padded-volume `unrolled_Ht`), H2D socket
  (both new methods kept), SDPA test utils (metal_2's move to `tt_py_test_utils_common/` + metal_1's `q_layout`),
  rope LLK test (metal_2's file + metal_1's `is_fp32_dest_acc_en` template argument).
* `tt/prefill/d_p_bridge.py`:
  * `DPPrefill`: metal_2's runtime (adapter path, one slot, `DECODER_SKIP=decode`, `INDEX_K_SERVED=bf16`) driven
    chunk by chunk over a prompt; `host_states(T)` reads layers 0-20 back as host tensors.
  * `to_prefill_states(host, layer_devices)`: host tensors -> metal_1 `PrefillAttentionState`s on each layer's decode
    stage (ring rolled to natural order; layer 20's ring zero), ready for `commit_prefill_state`.
  * `save_host_states` / `load_host_states`: the hand-off file for option 4.
* `tests/prefill/test_d_p_bridge.py`: step 1 runs metal_2 on the 8x4 mesh and saves states; step 2 compares them
  with metal_1's own kv_only prefill states for the same prompt (per layer, per tensor PCC).

### 6.1 First result (16384 tokens, galaxy 8x4)

Step 1 passed. Step 2 fails: layer 0 matches, everything after it diverges.

| layer | kv_tail | compressed_kv | idx_keys |
|---|---|---|---|
| 0 | 0.99998 | - | - |
| 1 | 0.79 | - | - |
| 2 | 0.73 | 0.71 | 0.89 |
| 8 | 0.58 | 0.57 | 0.75 |
| 14 | 0.50 | 0.47 | 0.65 |
| 19 | 0.52 | - | - |
| 20 | (not written) | 0.67 | 0.78 |

The layer-0 match means token ids, embedding, `attn_norm`, `wkv`, RoPE and ring order agree between the two code
bases, so the bridge's layout conventions are right. The break is in the layer-0 block output, or in layer 1's
input path before `wkv`. Suspects, in order: Engram (layers 1 and 14; metal_1 hashes n-grams with its own
tokenizer normalisation), layer 0's MoE routing (hash vs score), and the mHC residual mix. Neither side has been
checked against a CPU reference at this length yet, so it is not known which one is wrong.

## 7. Next steps

1. Validate the bridge on device (section 6 test): this checks value conventions (RoPE layout of compressed KV and
   index keys, ring order) between the two code bases, which nothing else does.
2. Pick an option from section 5. Option 1 needs a metal_1 decode importer for address-table records; option 2 needs
   metal_2's CCLs on a 4-row line; option 3 is a tt-metal allocator change.
3. Server: a `--prefill-backend d_p` `_Prefiller` subclass that calls `DPPrefill` and commits through the bridge
   (its `aligned()` keeps metal_1's "leave 128 tokens to decode" rule; metal_2's chunk is 5120, prompts are run as
   whole chunks plus one partial chunk).
4. Device-side bridge: replace the host round trip with a fabric gather of the SP x TP striped records into each
   stage's paged pools (1M tokens: ~0.5 GB of ratio-2 KV a source layer through the host today).
