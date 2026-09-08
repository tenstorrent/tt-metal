# Pipeline-Parallel Prefill for Gemma 4

How to spread Gemma 4 31B prefill across several Blackhole Galaxies so that more prefill slots
fit, while the sliding-window (SWA) layers keep storing the **full** context length.

Target configuration in this document: **PP=4** — four prefill galaxies, 15 layers each.

---

## 0. The problem

Gemma 4 31B prefill runs today on **one** Blackhole Galaxy as CP8/TP4 (32 chips, 60 layers,
`max_seq_len=262144`), feeding the decode galaxies over the disaggregated KV-migration path
(`tt/runners/manifests/gemma4_binding_disagg_migration_1rank.yaml`).

It is capped at **`PREFILL_NUM_USERS=2`**. The cause is that the 50 SWA layers store the full 256K
context rather than a 1024-token window, so per-user KV dominates DRAM. Bounding those caches to
the window would cut KV ~250×, but the decode side and the migration table depend on full-context
SWA storage, so that is off the table. The remaining lever is to spread the layers over more
galaxies, which divides both KV **and** weights per galaxy.

Two findings shape everything below:

1. **The generic prefill engine already implements pipeline parallelism.** The layer split, the D2D
   activation sockets, and the multi-rank migration-table merge are all in
   `models/demos/common/prefill/`, and 2/4/8-galaxy mesh descriptors plus rank bindings are checked
   in and exercised for DeepSeek/Kimi/GLM. Gemma 4 is the only thing gated to one rank, by two
   explicit `NotImplementedError`s.
2. **The KV address table is a second, independent blocker on the slot count.** It hits protobuf's
   2 GiB message cap between **2 and 3** slots — today's config sits at 78% of it — for reasons
   unrelated to PP. §5 covers it.

---

## 1. Theory

### 1.1 Why memory, not compute, forces the split

Per-device KV per user, at `max_seq_len=262144`, CP=8, TP=4, `bfloat8_b` (1088 B per 32×32 tile =
1.0625 B/element), `seq_local = 262144/8 = 32768`:

| Layer type | Count | Per-device tensors | Elements/user | Bytes/user/layer |
|---|---|---|---|---|
| `sliding_attention` | 50 | K and V, `[users, 16/4 heads, 32768, 256]` | 2 × 4×32768×256 | **68 MiB** |
| `full_attention` | 10 | one packed `[users, 4/4 heads, 32768, 640]` | 1×32768×640 | **21.25 MiB** |

`640 = GLOBAL_HEAD_DIM 512 + GLOBAL_ROTARY_DIM 128` — one physical buffer with overlapping K and V
views, because `attention_k_eq_v` ties K and V on global layers only
(`tt/attention/global_kv_cache.py:28-30`, `tt/attention/__init__.py`).

Total per user per device = 50×68 + 10×21.25 = **3612.5 MiB ≈ 3.53 GiB**, of which the 50 SWA
layers are **94%**. A Blackhole chip has 8 DRAM banks × 3.984 GB ≈ **32 GB**
(`tt_metal/soc_descriptors/blackhole_140_arch.yaml`, `dram_bank_size: 4278190080`).

Weights are sharded on the TP axis **only** and replicated across the 8 CP rows — `column_parallel`
/ `row_parallel` pass `dims=(None, tensor_dim)` when `tp_axis == 1` (`config.py:85-95`) — so ≈30 GB
of bfp8 layer weights land as ≈7.6 GB per device. With 7.6 (weights) + 2×3.53 (KV) ≈ 14.7 GB, plus
the token embedding (262144×5376, column-parallel over TP ≈ 0.7 GB), the 32 MB trace region, and
activation scratch, 2 slots is what fits in 32 GB.

### 1.2 What pipeline parallelism means here

Split the 60 layers into `N` contiguous ranges, one per galaxy. Rank *r* holds layers
`[first_layer_idx, first_layer_idx + num_layers)` and nothing else.

```
scheduler ──H2D──▶ rank0 (L0-14) ──D2D──▶ rank1 (L15-29) ──D2D──▶ rank2 (L30-44) ──D2D──▶ rank3 (L45-59)
  tokens            embed +               15 layers            15 layers            15 layers
                    15 layers                                                       (no LM head — KV only)
```

Rank 0 owns the token embedding and the H2D socket; every other rank receives a hidden-state
activation over a D2D socket. No rank needs the final norm or the LM head — this is a KV-producing
service, and `Gemma4Model` already returns pre-norm hidden states under `prefill_weights_only=True`
(`tt/model.py:1306-1308`), which is exactly the pipeline-stage contract.

Each stage keeps its **own** KV for its **own** layers, at full context. The merged
`KvChunkAddressTable` published to the migration worker spans all 60 layers by stitching together
per-rank address ranges, so the decode side sees one logical cache exactly as it does today.

**PP is orthogonal to CP and TP.** Every rank still runs CP8/TP4 internally, so the ring-attention
halo rule (`chunk_size % (sp × 1024) == 0` — one 1024-token window per CP rank,
`tt/tt_prefill_runtime.py:72-73`) and `max_seq_len % chunk_size == 0` are unchanged. The three axes
compose: TP=4 within a mesh row, CP=8 across rows, PP=4 across galaxies.

### 1.3 Memory scaling model

For a rank holding `s` sliding and `g` full layers, per user per device:

```
KV_bytes(user, device) = s × 68 MiB + g × 21.25 MiB
```

At PP=4 the layer *count* splits 15/15/15/15, but the layer **types** do not divide evenly — full
attention sits at global indices 5, 11, 17, …, 59:

| Rank | Global layers | sliding | full | KV/user/device |
|---|---|---|---|---|
| 0 | 0–14 | 13 | 2 (5, 11) | 926.5 MiB |
| 1 | 15–29 | 12 | 3 (17, 23, 29) | 879.75 MiB |
| 2 | 30–44 | 13 | 2 (35, 41) | 926.5 MiB |
| 3 | 45–59 | 12 | 3 (47, 53, 59) | 879.75 MiB |

Worst rank **≈0.905 GiB per user per device**, a **3.9×** reduction from 3.53 GiB. Weights drop to
≈1.9 GB per device. Budget on the worst rank: 32 − (1.9 weights + 0.7 embedding on rank 0 + ~2 GB
traces/scratch) ≈ 27 GB → **~29 slots on DRAM grounds**. DRAM stops being the binding constraint;
§5 covers what replaces it.

### 1.4 Schedule, bubbles, and why no microbatching is needed

Training PP needs microbatching because one batch must traverse the whole pipeline before the next
can start. Chunked prefill does not have that problem: **the chunks are already the microbatches.**
A 256K prompt at `chunk_size=8192` is 32 chunks, and the engine's request loop
(`prefill_runner.py:348-389`) pushes them back-to-back — once chunk *c* leaves rank 0, rank 0 starts
chunk *c+1* while rank 1 works on *c*.

Correctness needs only that each rank sees chunks **in order** for a given slot, because a chunk's
KV must be written before the next chunk's attention reads it. One FIFO D2D socket per hop
guarantees exactly that. `D2D_FIFO_SIZE_BYTES` (`prefill_runner.py:59`, default 256 B, raised to
32768 in the intra-galaxy configs) bounds how far ahead a rank may run.

For `C` chunks and `N` stages with per-stage time `t`:

```
total ≈ (C + N − 1) × t        utilization ≈ C / (C + N − 1)
```

At PP=4 with C=32: 32/35 = **91%**. At PP=8: 32/39 = 82%. This is why PP=4 is a good target — the
bubble is small at 256K, and it degrades sharply for short prompts (an 8K prompt is one chunk,
utilization 1/4). Short prompts should be routed to a single-rank prefill pool or batched.

Two caveats on the ideal model:

- **Load imbalance.** Ranks 1 and 3 carry 3 full-attention layers vs 2. Full attention over the
  whole history costs more per layer than a 1024-window layer at long context, and the slowest
  stage sets the rate. `PREFILL_PP_LAYER_COUNTS` (`runner_utils.py:154`) allows a hand-tuned split
  (e.g. `16,14,16,14`) once measured.
- **D2D cost.** The activation is `[1, 1, chunk_size, hidden_size]` bf16 = 8192×5376×2 = 88 MB per
  chunk, seq-sharded over CP=8 → 11 MB per device per hop, and emb-**replicated** over TP=4 because
  Gemma 4 sets `pipeline_activation_emb_tp_sharded = False` (`tt/runners/adapters/gemma4.py:42`).
  Flipping that to TP-sharded would cut the payload 4×, but it must match the layout the decoder
  layer consumes and produces.

### 1.5 What "more slots" actually buys

`PREFILL_NUM_USERS` is the prefill galaxy's **handoff staging** capacity: how many prompts can be
resident and awaiting (or undergoing) migration to a decode galaxy at once. It is not end-to-end
serving concurrency — the decode side sizes its own slots separately. At 2 slots the prefill galaxy
stalls whenever both slots are waiting on migration, so raising it is what lets prefill run ahead of
the KV mover and keep the decode galaxies fed.

---

## 2. What already exists (reuse, do not rebuild)

Everything in this table is done and in use by other models.

| Capability | Location |
|---|---|
| Rank topology, `is_first/last_rank` | `prefill_runner.py:479-487` via `ttnn.distributed_context_get_rank/size` |
| Contiguous layer split + `PREFILL_PP_LAYER_COUNTS` + boundary snapping | `runner_utils.py:129-183` `compute_layer_split` |
| Per-rank knobs handed to the adapter | `adapter.py:45-95` `PrefillRunParams` — already carries `first_layer_idx`, `is_first_rank`, `is_last_rank` |
| D2D activation sockets over tt-fabric | `prefill_runner.py:178-248`, `ttnn.D2DStreamService.create_sender/create_receiver` |
| Activation spec + mesh mapper | `runner_utils.py:101-107` `activation_global_spec`, `prefill_runner.py:64-69` `D2D_MAPPER_CONFIG` |
| Send on non-last rank, shutdown-sentinel forwarding | `prefill_runner.py:293-338`, `:251-268` |
| Cross-rank config agreement check | `prefill_runner.py:439-467` |
| Multi-rank migration stage all-gather + merged table | `migration.py:287-334`, `prefill_runner.py:605-700` |
| 2/4/8-galaxy mesh descriptors, rank bindings, launcher | `common/prefill/runners/topology_configuration/`, `run_pipeline_prefill.sh` |

The model-side reference implementation to mirror is DeepSeek:
`models/demos/deepseek_v3_d_p/tt/tt_prefill_transformer.py:100-125,155-175` (embedding only on the
first rank, global `layer_idx = first_layer_idx + local_idx`) and
`models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py:858-887` (`kv_migration_stages`).

---

## 3. Code changes

All work is in `models/demos/gemma4/`. Ordered by dependency.

### 3.1 Thread the global layer index through the model

The key invariant: **`Gemma4DecoderLayer` must receive the GLOBAL layer index.** Everything inside
`tt/layer.py` is already written against a global index — `hf_config.layer_types[layer_idx]` (`:91`),
the state-dict prefix `model.layers.{layer_idx}` (`:98`),
`tensor_cache_path=.../layer_{layer_idx}/...` (`:108`), and
`Gemma4AttentionConfig(hf_config, layer_idx)` (`:132`). Passing a global index therefore fixes layer
type, RoPE theta (10000 sliding vs 1e6 global), and head dims in one move — **and** reuses the
existing per-global-layer weight tensorbin cache with no re-caching.

- `tt/model.py:381` — add `first_layer_idx` to `Gemma4Model.__init__`; keep `n_layers` as the rank's
  local count. Add `self._global_layer_idx(i) -> self.first_layer_idx + i`.
- `tt/model.py:556-561` — pass `layer_idx=self._global_layer_idx(i)` to `Gemma4DecoderLayer`, and
  use the global index for `Gemma4AttentionConfig(hf_config, ...)` at `:582`.
- `tt/model.py` forward loop — replace every `self.hf_config.layer_types[i]` with the global index
  at `:1144`, `:1150`, `:1216`, `:1231`, `:1242`, `:1272-1274`; likewise `_get_rope_mats` at
  `:874,888` and the `used_types` sets at `:1076`, `:1090`.
- `tt/model.py:1286`, `:1289` — **the per-layer ack must report the GLOBAL index.** Both
  `self._prefill_trace_controller.layer_ack(i)` and `on_layer_complete(i)` pass the local loop index
  today. The engine builds the ack sequence number as `seq = request_id × NUM_LAYERS + layer_idx`
  with `NUM_LAYERS` the **global 60** (`prefill_runner.py:125-127`, wired at `:778-783`), so with
  local indices every rank would emit `seq` starting at 0 and the four ranks' acks would collide in
  the router's ring. This is a real multi-rank bug, not a cosmetic one.
- `tt/model.py:627-628` — `last_kv_layer_by_type` must key off global indices (it feeds the
  spec-decode drafter, which is decode-only, but leaving it local is a latent bug).
- `tt/model.py:403-420` — `kv_shared_layer_map` must be computed in global space. 31B has
  `num_kv_shared_layers=0`; E2B/E4B do not (see §3.6).
- `tt/common.py:45-82` — add `first_layer_idx` to `create_tt_model` and forward it. Note that
  `model_args.num_hidden_layers = num_layers` at `:81-82` truncates to a prefix; it must stay the
  **global** 60 so `layer_types` lookups resolve, with the rank's count passed separately. This is
  the subtlest edit in the change set.

### 3.2 First/last-rank I/O roles in the runtime

In `tt/tt_prefill_runtime.py`:

- `:66-69` — remove the single-rank `NotImplementedError`. Keep the `(8,4)` assertion (§3.6).
- `:118-136` `make_chunk_input` — on a non-first rank, return a placeholder activation
  `[1, 1, chunk_size/sp, hidden_size]` bf16 TILE instead of token IDs. Mirror
  `models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:217-227`.
- `:138-148` `_normalize_input` — accept the activation shape on non-first ranks; it currently
  asserts a token count.
- `:182-194` `_forward` — call `transform_and_embed_prefill_inputs_device` only when
  `is_first_rank`; otherwise feed the received activation straight into `ttnn_prefill_forward`.
- `:245-289` `prefill_chunk` — **return the hidden state** when `not is_last_rank` instead of
  deallocating it (`:287-288`); the engine sends it downstream (`prefill_runner.py:321-331`). Under
  `use_trace=True` the traced replay path returns `None` at `:282`, and `self._trace_output` is the
  stage activation that must be returned instead.
- `:80-99` `_build_model` — pass `first_layer_idx`, and drop `embed_tokens.weight` from the state
  dict on non-first ranks so `Gemma4Model` skips building the embedding (≈0.7 GB/device saved).
  `_cache_completion_state` (`utils/partial_weights.py`) supplies only `layer_scalar` floats plus
  the embedding, so this is a small filter, not a new loader.

### 3.3 KV cache allocation for a layer subset with mixed geometry

`tt/runners/kv_caches.py` has two defects for PP:

- `:58` — `layer_types = tuple(hf_config.layer_types[:num_layers])` slices from index **0**, so
  every rank believes it holds layers 0..n. Must be `[first_layer_idx : first_layer_idx + n]`.
- `:61` — `Gemma4AttentionConfig(hf_config, layer_idx)` passes the **local** index where a global
  one is required.

Add `first_layer_idx` to `allocate_ring_kv_caches` and use global indices for both.

**Design decision: consolidate to 3 tensors per rank.** Today each layer gets its own tensor
(`init_ring_kv_cache` / `init_packed_ring_kv_cache` are called with the default `num_layers=1`), so
a rank holds 60 separate base addresses. The engine's stage merge all-gathers **one base address per
stage**, with `count` layers laid out contiguously from it (`migration.py:21-24` `KvCacheStage`,
`:294-334` `allgather_kv_stage_layout`) — a layout per-layer tensors cannot express.

Consolidate to **three** tensors per rank, each sized to that rank's layers *of that type*:

| Stage | Tensor | Shape | Index space |
|---|---|---|---|
| 0 | packed global KV | `[users × g, 1, 32768, 640]` | compacted full-attention |
| 1 | sliding K | `[users × s, 4, 32768, 256]` | compacted sliding |
| 2 | sliding V | `[users × s, 4, 32768, 256]` | compacted sliding |

The allocators **already** take `num_layers` and pack the batch dim user-major as
`slot = user × num_layers + layer` (`tt/attention/ring_prefill.py:120-139,160-167`), and the write op
`update_padded_kv_cache` already takes `layer_idx` / `num_layers` (`ring_prefill.py:186-204`). So
this uses existing machinery rather than new code.

Per-type compacted indices, derived from the global `layer_types`:

```python
first_global  = count of "full_attention"    in layer_types[:first_layer_idx]
first_sliding = count of "sliding_attention" in layer_types[:first_layer_idx]
g = count of "full_attention"    in layer_types[first_layer_idx : first_layer_idx + n]
s = count of "sliding_attention" in layer_types[first_layer_idx : first_layer_idx + n]
```

This is the pattern GLM-5.2 already uses for its indexer cache, whose slots are numbered in
compacted `full_indexer_rank` space and sized to the stage
(`models/demos/deepseek_v3_d_p/tt/runners/adapters/glm_5_2.py:57-105`, with the sizing assertion at
`tt_prefill_runtime.py:879-885`). `Gemma4KvCaches.global_layers` / `.sliding_layers`
(`kv_caches.py:33-39`) must return **global** indices while indexing into the compacted tensors.

### 3.4 Migration table: stage merge

In `tt/tt_prefill_runtime.py:323-337` and `tt/runners/kv_chunk_table.py`:

- Replace `kv_migration_base_address` (`:323-326`, a single address — insufficient) with
  **`kv_migration_stages`**, returning the three `KvCacheStage`s from §3.3, each with its own
  `(base_addr, first_layer, count)` in its own compacted numbering. Model on
  `models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py:858-887`, whose docstring describes exactly
  this situation ("The two caches do NOT share a layer-index space").
- `build_kv_chunk_table` (`:328-337`) must accept `first_layer_idx`, `num_my_layers`, and
  `stage_layouts`, and build the table from the **all-gathered** layouts rather than local
  `tensor.buffer_address()` / `mesh_device.get_fabric_node_id` calls (`kv_chunk_table.py:103-132`).
  Only rank 0 builds it (`prefill_runner.py:667-676`).
- `kv_chunk_table.py:65-72,84` — each config's `num_layers` stays the **global** 60 and
  `semantic_layer` must be the **global** layer index (`:141`, `:152`), so the merged table is
  numbered identically to today's single-rank table. **This preserves the decode-side contract:**
  the 36 config IDs are per-head, not per-layer (`:20-23`), so PP leaves them unchanged — which
  matters because config-id order is the src↔dst contract with the decode endpoint
  (`common/prefill/docs/PREFILL_MIGRATION_TESTING.md:528-530`).
- `iter_cache_chunk_locations` (`:32-62`) shard math must use the **per-type compacted**
  `heads_per_device` and layer count for the consolidated tensors, since `shard` is computed from
  `slot × heads_per_device + local_head` against `blocks_local`.

The migrate call is already layer-ranged — `client.migrate(..., layer_start, layer_end_exclusive,
...)` (`migration_driver.py:191-202`) — so per-stage migration is expressible with no protocol
change.

### 3.5 Config, manifests, topology

- `tt/runners/manifests/gemma4_31b.json` — leave `PREFILL_NUM_LAYERS=60`; the engine splits it.
  Raise `PREFILL_NUM_USERS` once measured.
- Add `gemma4_binding_disagg_migration_4rank.yaml` with `rank_bindings` for `mesh_id: 0..3`,
  modelled on `topology_configuration/pipeline_prefill_request_4rank.yaml`.
- **Mesh graph descriptor.** The existing 4-galaxy descriptor
  (`pipeline_prefill_4galaxy_connected_mesh_graph_descriptor.textproto`) declares each mesh as
  `dims: [8,4] dim_types: [RING, RING]` with `channels count:2 policy:RELAXED`, whereas Gemma 4's
  single-galaxy descriptor
  (`tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_x_graph_descriptor.textproto`) uses
  `[LINE, RING]` with `policy: STRICT`. A Gemma-4-matched 4-galaxy descriptor preserving
  `[LINE, RING]` per mesh plus the inter-mesh chain is likely needed. `GEMMA4_CCL_TOPOLOGY` /
  `default_ccl_topology` (`tt/ccl.py:66-109`) lets the CCL algorithm be pinned independently while
  this is settled.
- **`PREFILL_MIGRATION_TABLE_PATH` must move off `/tmp`.** The engine hard-rejects per-host paths
  when `num_ranks > 1` (`prefill_runner.py:621-627`); the current manifests use
  `/tmp/gemma4_kv_chunk_table.pb`. Point it at shared/NFS storage.
- Launch: `run_pipeline_prefill.sh <binding.yaml> <host_list>` → `ttrun.py` + MPI.

### 3.6 Relax the `(8,4)` hard-codes only where needed

Three places assert CP8/TP4: `tt/runners/adapters/gemma4.py:70-71`, `tt/tt_prefill_runtime.py:67`,
`tt/runners/kv_chunk_table.py:82`. For the 4-galaxy target every rank stays `(8,4)`, so **leave
these in place** — they are cheap guards, and relaxing them is a separate, larger change (§6.1).

Also implement `layer_split_boundaries` (`adapter.py:179-186`) for variants with
`num_kv_shared_layers > 0`: a rank must not start on a layer whose KV source lives on another rank.
For 31B (`num_kv_shared_layers=0`) it can return `None`.

---

## 4. Validation plan

1. **Single-rank regression.** `pytest models/demos/gemma4/tests/unit/test_prefill_adapter.py
   tests/unit/test_kv_caches.py tests/unit/test_global_kv_cache_layout.py
   tests/test_common_prefill_runtime.py`. The consolidation in §3.3 changes cache shapes, so
   `test_kv_caches.py` and `test_global_kv_cache_layout.py` need updating; the existing single-rank
   disagg run must still produce identical KV.
2. **Host-only split math.** New unit test: for PP=4, assert the per-rank global layer ranges,
   `(s, g)` counts, compacted first indices, and cache shapes — no device needed. Cheap, and it
   catches the whole class of off-by-one errors in §3.1/§3.3.
3. **2-galaxy compute-only**, migration off. Use
   `topology_configuration/pipeline_prefill_request_2rank.yaml` with the Gemma 4 manifest,
   `PREFILL_ENABLE_MIGRATION=0`, `PREFILL_USE_TRACE=0`. Confirms the layer split, the D2D handoff,
   and the first/last-rank roles in isolation.
4. **KV PCC across the split.** `PREFILL_MOCK_MIGRATION=1` on the runner +
   `PREFILL_PRODUCER_CHECK_PCC=1` on the producer (`PREFILL_MIGRATION_TESTING.md` Gate 1). Note
   `_read_slot_kv_and_check_pcc` is **not** adapter-dispatched and knows only merged-MLA and
   MiniMax-M3 layouts (`ADDING_A_PREFILL_MODEL.md:241`) — Gemma 4's mixed global/sliding layout needs
   a branch there, or verification via the `dst-bytes` loopback path already used by
   `gemma4_producer_loopback_migration.yaml`.
5. **Trace on.** Re-run step 3 with `PREFILL_USE_TRACE=1` and `PREFILL_LAYER_ACK_D2H=1`. This is the
   highest-risk interaction in the change set (§6.2).
6. **Migration merge**, 2 ranks then 4 ranks, table on shared storage. Verify the merged table spans
   all 60 global `semantic_layer`s with unchanged config IDs.
7. **Slot ramp.** Raise `PREFILL_NUM_USERS` 2 → 4 → 8 at PP=4, watching for allocator OOM **and**
   the address-table cap (§5).

---

## 5. The address-table cap, and how to fix it

### 5.1 Diagnosis

`UnrolledGrid.entries` is a **dense** `[slot][layer][chunk]` vector
(`tt_metal/api/internal/disaggregation/kv_chunk_address_table.hpp:88-97`), and `total_entries()` is
the **dense grid-equivalent count, independent of population** —
`Σ_configs num_slots × num_layers × num_position_chunks`
(`tt_metal/impl/internal/disaggregation/kv_chunk_address_table.cpp:382-389`). With
`chunk_n_tokens=32` and `max_seq_len=262144` → 8192 position chunks, and all 36 configs declaring
`num_layers=60` (`tt/runners/kv_chunk_table.py:65-72,84-93`):

```
total_chunks = 36 × 60 × 8192 × slots = 17,694,720 × slots
total_rows   = 36 × 60 × slots        =      2,160 × slots
estimate     = total_chunks × 48 B + total_rows × 64 × 72 B ≈ 819.5 MiB × slots
```

against a threshold just under protobuf's 2 GiB message cap
(`kv_chunk_address_table_protobuf.cpp:31-53`, `:327-336`; `kEntryWireEstimate=48`,
`kRunWireEstimate=72`, `kMaxRunStep=64`):

| slots | estimate | dual-write |
|---|---|---|
| 2 (today) | 1.60 GiB | **on** — but at 78% of the cap |
| 3 | 2.40 GiB | **off → runs-only** |
| 8 | 6.40 GiB | runs-only |

So the flip happens between **2 and 3 slots**; today's config is one slot away from it. The export
path auto-detects the strided structure (`:283-299`) and Gemma 4's block-cyclic layout compresses
perfectly (`bank = shard % num_banks`, `offset = shard // num_banks × chunk_bytes`), so tt-metal
needs no new code to *emit* runs. The problem is the consumer: a runs-only table "is unreadable by
ANY entries-only reader today"
(`tt_metal/impl/internal/disaggregation/protobuf/kv_chunk_address_table.proto:81-87`), and the
reader is the out-of-tree `migration_worker` (tt-llm-engine `disaggregation/migration/`).

Secondary cost: rank 0 materializes the dense grid in host RAM (36 × 60 × 8192 × 16 B ≈ 270 MiB per
slot) before compressing; at 8 slots that is ~2.2 GB.

### 5.2 Fixes, easiest first

**A must be answered first** — it decides whether anything else is needed. B and C are additive and
together reach ~12 slots while keeping the table readable by an entries-only worker, which is the
safe path if A comes back negative.

#### A. Confirm whether the worker already reads `STRIDED_ROWS` — zero code

The runs format exists so readers can adopt it, and the "dual-write for old readers" wording implies
a transition that may already be complete. One question to the tt-llm-engine owners settles whether
there is a blocker at all.

It can also be **tested today, at 2 slots, before any PP work**, via the canary override
`KV_CHUNK_TABLE_DUAL_WRITE_MAX_BYTES` (`kv_chunk_address_table_protobuf.cpp:44-53`, read per call
specifically so tests can re-point it). Setting it low forces `dual_write=false`, so the current
single-rank disagg run emits a runs-only table and either works or fails loudly:

```bash
KV_CHUNK_TABLE_DUAL_WRITE_MAX_BYTES=1   # forces runs-only; existing 2-slot config, no PP needed
```

The override can only *lower* the threshold. Raising it cannot keep dual-write at 8 slots — the
2 GiB limit is protobuf's own message cap, so serialization would simply fail instead.

**Do this first.** If the worker reads runs, the cap is a non-issue and B/C/D are unnecessary.

#### B. Per-config `num_layers` — small code, 24% saving, needs decode-side agreement

All 36 configs declare `num_layers=60`, but configs 0–3 only ever populate the 10 full-attention
layers and configs 4–35 only the 50 sliding layers. Since `total_entries()` is dense in
`num_layers`, declaring each config's own type count removes 24% of the grid outright:

```
total_chunks = (4 × 10 + 32 × 50) × 8192 × slots = 13,434,880 × slots
estimate ≈ 622 MiB × slots     →  threshold moves from 2 slots to 3
```

`KvChunkAddressTableConfig.num_layers` is a writable Python field
(`ttnn/cpp/ttnn-nanobind/disaggregation.cpp:79`), so the edit is confined to `_config()`
(`kv_chunk_table.py:65-72`). The cost is that `semantic_layer` becomes **per-type compacted** rather
than global, changing the src↔dst contract — the decode endpoint must publish and interpret the same
numbering (`PREFILL_MIGRATION_TESTING.md:528-530`). Cheap to implement, but a coordination item, and
3 slots alone does not reach the goal.

#### C. Per-rank tables instead of one merged table — biggest win, no wire-format dependency

Let each PP rank publish a table covering only **its own** layers to its own co-located migration
endpoint, rather than rank 0 merging all 60. The per-endpoint queue naming already anticipates this
(`/mig_ep0_*`, `/mig_ep1_*` in the manifests), and each galaxy already runs its own
`migration_endpoint` + worker pair.

```
per rank, 15 layers:           36 × 15 × 8192 × slots        →  ≈ 205 MiB × slots  →   8 slots = 1.60 GiB (dual-write ON)
combined with B (worst rank):  (4×2 + 32×13) × 8192 × slots  →  ≈ 161 MiB × slots  →  12 slots = 1.89 GiB (ON)
```

This is the key result: **at PP=4, per-rank tables keep dual-write intact to ~8–12 slots, so the
out-of-tree reader never has to change.** It scales with PP for free — the thing that inflates the
table (more slots) is the same thing PP already divides.

Cost: this is engine-level, not Gemma-4-level. `prefill_runner.py:667-676` has only rank 0 build the
table, and the stage-merge all-gather (`migration.py:287-334`) exists to produce one. Moving to
per-rank publication changes the P↔D pairing model and wants the DeepSeek/Kimi owners' input, since
they share the engine. Medium effort, highest payoff, and it removes a cross-repo dependency rather
than adding one.

#### D. Build the `StridedRowMap` directly instead of detecting it on export — new bindings

Gemma 4's layout is analytic (`iter_cache_chunk_locations`, `kv_chunk_table.py:32-62`, is a closed
form with `step = num_banks`), so the rows could be constructed straight into a `StridedRowMap` via
`install_strided_map` (`kv_chunk_address_table.hpp:166`), never materializing the dense grid. Per row
that is `step` bases + `step` strides (~128 B) instead of 8192 × 16 B = 131 KB — a ~1000× reduction
that eliminates the host-RAM cost in §5.1.

But `install_strided_map` and `StridedRowMap` are **not exposed in the Python bindings** (only the
C++ import path uses them — `kv_chunk_address_table_protobuf.cpp:562`), so this needs new nanobind
surface. And the result is still runs-only on the wire, so it **does not remove the need for A**.
Worth doing only if host RAM or table-build time becomes the binding constraint.

#### E. Coarser migration granularity (`chunk_n_tokens > 32`) — does not work

Raising `chunk_n_tokens` from 32 (`kv_chunk_table.py:70`) would divide the chunk count directly, but
a `KvCacheLocation` describes **one contiguous DRAM region**, and consecutive 32-token tiles for the
same (slot, head, layer) are *not* contiguous — they round-robin across banks
(`bank_id = shard % num_banks`, `kv_chunk_table.py:56-62`). Merging 8 tiles into one entry would
describe memory that isn't there. This is precisely the regularity the strided representation exists
to capture, so D is the correct expression of the idea; E is a dead end. Recorded so it isn't
re-proposed.

#### Recommendation

Run **A** now — one question plus a one-env-var experiment on the existing 2-slot config, and it may
close the issue entirely. If the worker cannot read runs, do **C** (optionally with **B**), which
reaches the slot target with no out-of-tree change. Keep **D** in reserve.

---

## 6. Risks and open questions

### 6.1 Bring-up order

The intra-galaxy configs (`pipeline_prefill_request_intragalaxy_{2,4}rank.yaml`, which split one
galaxy via `TT_VISIBLE_DEVICES` into 4×4 sub-meshes at SP=4/TP=4) look like the cheap first step,
but they require relaxing all three `(8,4)` assertions **and** changing the CP degree, mixing two
independent changes. **Start with 2 real galaxies at `(8,4)`**, which exercises only the PP code.

### 6.2 Trace + D2D interaction

`SubDeviceTraceController` captures the per-chunk forward (`tt_prefill_runtime.py:207-243`) and the
engine's D2D send happens outside it, but the stage activation must survive trace replay: the traced
path currently returns `None` and reuses `self._trace_input` via `ttnn.copy` (`:279`). The
non-last-rank path must return `self._trace_output`, and the receiving rank must copy the inbound
activation into its own trace input. `warmup_ack_count()` (`:318-321`) returns the rank's local layer
count — correct per rank, but it means the scheduler's expected ack total is now a sum across ranks.
The global-index requirement for acks is settled in §3.1.

### 6.3 Other open questions

- **Load balance.** Full-attention layers are unevenly distributed (2/3/2/3). Measure per-stage time
  before tuning `PREFILL_PP_LAYER_COUNTS`.
- **D2D activation width.** Gemma 4 replicates the emb dim across TP
  (`pipeline_activation_emb_tp_sharded = False`), costing 4× the necessary bytes per hop. Confirm
  whether the decoder layer's input/output is TP-sharded and flip if so.
- **Decode-side / worker changes.** Because `semantic_layer` stays global and the 36 config IDs are
  per-head, the decode endpoint's view is unchanged — **provided** §3.4 keeps global numbering (and
  note that fix B in §5.2 would change it). Worth an explicit cross-check against the decode table's
  config order.
- **Cross-endpoint verification gap.** Destination read-back after a real P→D migration is currently
  skipped (`PREFILL_MIGRATION_TESTING.md:298-300`), so PP correctness at the far end rests on the
  loopback `dst-bytes` path.
- **`num_kv_shared_layers > 0` variants** (E2B/E4B) need `layer_split_boundaries`; 31B does not.

---

## References

- `models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md` — the adapter/runtime contract
- `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md` — migration gates
- `tech_reports/Programming_Multiple_Meshes/Programming_Multiple_Meshes.md` — MGDs, rank bindings,
  sockets, and a worked 2-stage pipeline example
- `tt_metal/api/internal/disaggregation/README.md` — why the address table exists
- `models/demos/deepseek_v3_d_p/tt/tt_prefill_transformer.py`,
  `models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py` — the reference PP + migration model
