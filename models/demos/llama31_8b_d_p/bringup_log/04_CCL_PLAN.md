<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 04 — Parallelism and CCL plan

**Phase:** P4 · **Date (UTC):** 2026-09-03 · **Gate:** `G-CCL-PLAN` — **PASS**

The user requirement is explicit: **CCLs are part of the modules.** Attention, MLP and RMSNorm each
own their own collectives; `DecoderLayer` and `Model` never call one. Every `path:line` here is
machine-verified (`scripts/verify_citations.py`, `raw/G-CCL-PLAN_20260903T170527Z.log`).

New decisions in this phase: `DEC-018` (residual scheme), `DEC-019` (`MeshConfig` union),
`DEC-020` (topology + `num_links` policy), `DEC-021` (SP one-shot bootstrap).

---

## 1. The `(mesh, TP, SP)` arithmetic

Convention, read from the engine, not guessed: `mesh_shape = (SP, TP)` — rows are SP, cols are TP.
`models/demos/common/prefill/adapter.py:57` (`mesh_shape: tuple  # (sp, tp)`),
`models/demos/common/prefill/runners/runner_utils.py:78` (`sp_factor, tp_factor = mesh_shape`).

```
devices           = 32                                  measured: ttnn.get_num_devices() == 32, arch == blackhole
mesh_shape        = (SP, TP) = (4, 8)                    SP·TP = 4·8 = 32                                ✓  (DEC-002)
tp_axis           = 1  (cols)                            MeshConfig default
sp_axis           = 0  (rows)                            derived: 0 if tp_axis == 1

TP = 8:
  Q heads / chip        = 32 / 8 = 4                                                                     ✓ integral
  KV heads / chip       =  8 / 8 = 1     -> NO KV replication; TP=8 is the MAXIMUM such TP                ✓
  GQA group / chip      = 4 Q per 1 KV   == the global group 32/8 = 4                                     ✓ preserved
  hidden shard          = 4096 / 8  =  512 = 16 tiles                                                     ✓ tile-aligned
  intermediate shard    = 14336 / 8 = 1792 = 56 tiles                                                     ✓ tile-aligned
  o_proj contraction    = 4096 / 8  =  512 = 16 tiles                                                     ✓ tile-aligned
  lm_head vocab shard   = 128256 / 8 = 16032 = 501 tiles                                                  ✓ tile-aligned

SP = 4:
  S_loc                 = S / 4
  CHUNK_SIZE % (TILE_SIZE·SP = 32·4 = 128) == 0                                                           ✓
  MAX_SEQ_LEN % CHUNK_SIZE == 0            131072 % 1024 == 0                                             ✓
  kv_actual_global % 32 == 0               (update_padded_kv_cache)                                       ✓ at chunk boundaries
  KV cache seq_local    = MAX_SEQ_LEN / 4  = 32768 at 131072                                              ✓ tile-aligned
  ring-SDPA hops        = 4

num_links = 2                              Blackhole, shape[0] == 4 > 1
compute_with_storage_grid_size() = (12, 10)     measured on this machine
ring-attention CCL offset = (grid.x - 1, 0) = (11, 0)
DRAM banks = dram_grid_size().x = 8            measured
```

Tile-alignment holds for **every** admissible TP (`TP ∈ {1,2,4,8}`, `00_MODEL_CARD.md` §4.3), so no
padding path is needed anywhere — which is why the `o_proj` tile-pad machinery gpt-oss carries
(`models/demos/gpt_oss_d_p/tt/attention/weights.py:68`) is deleted for Llama (`03_OUTLINE.md` §3.8).

**Single-card phases (P5–P7) run `(1,1)`, TP=1, SP=1, `num_links=1`, no collectives at all.** Every
collective in this plan sits behind `if mesh_config.tp > 1` / `if mesh_config.sp > 1`, so the
single-card path is not merely untested-with-CCL — it never enters one.

---

## 2. The two objects, both created once per model

**`CCLManager`** (`tt/ccl.py`) — persistent CCL resources.
**`MeshConfig`** (`tt/config.py`) — the parallelism decision plus the collective wrappers.
Modules call `self.mesh_config.<collective>(t, self.ccl_manager, ...)`; they never touch
`ttnn.experimental.*` directly. Rationale, from the template's own comments: raw calls are how a
semaphore gets reused while still in flight (`BRINGUP_RECIPE.md:525-529`).

An all-reduce is **reduce-scatter + all-gather**, not `all_reduce_async`
(`models/demos/minimax_m3/config.py:94` then `:115`; identical in
`models/demos/gpt_oss_d_p/tt/config.py:102` then `:118`).

---

## 3. `MeshConfig` — the union (`DEC-019`)

Neither in-repo copy is a superset (`R-009`, Appendix F.4). Member-by-member, with the copy each
member is taken from:

| Member | Take from | Why / note |
|---|---|---|
| `__init__(mesh_shape, tp, tp_axis=1)` and the `ep_axis`/`sp_axis`/`total_devices` derivation | either — identical (`models/demos/minimax_m3/config.py:24`, `models/demos/gpt_oss_d_p/tt/config.py:22`) | `sp_axis = ep_axis = 0 if tp_axis == 1 else 1`. Llama keeps `ep_axis` as a dead alias? **No** — drop `ep_axis` (no MoE), keep `sp_axis` only. |
| `_VALIDATED_MESH_SHAPE = (4, 8)`, `_VALIDATED_TP = 8` | `models/demos/gpt_oss_d_p/tt/config.py:15`, `:16` | Already exactly this package's `DEC-002` target. |
| `_validate()` — **strict**: raise unless `tp == mesh_shape[tp_axis]` | `models/demos/gpt_oss_d_p/tt/config.py:38`, raise at `:45` | **Load-bearing for `G-MESH`.** The gate requires `MeshConfig((1,8), tp=4)` to *raise*; minimax's `_validate` (`:40`) only `logger.warning`s a mismatch (`:45-49`), so a copy of minimax alone would make `G-MESH` unfailable. The reason sub-axis TP must be rejected is in gpt-oss's own comment (`:40-43`): `shard_mapper` always shards the **entire** axis, so a smaller TP would build head counts from `tp` while the mapper still splits across all `tp_dim_size` devices. |
| the "untested shape" warning | either (`models/demos/minimax_m3/config.py:47`, `models/demos/gpt_oss_d_p/tt/config.py:50`) | Keep, **after** the strict check, so `(8,4)` stays legal-but-noisy (`DEC-002`). |
| `sp` property | `models/demos/gpt_oss_d_p/tt/config.py:55` | Minimax lacks it and reads `mesh_shape[sp_axis]` inline in `__repr__` (`:175`). Modules need `mesh_config.sp` (`models/demos/gpt_oss_d_p/tt/attention/prefill.py:184`). |
| `shard_mapper`, `column_parallel`, `row_parallel`, `sequence_parallel`, `shard_size` | either — identical (`models/demos/minimax_m3/config.py:52`/`:61`/`:65`/`:69`/`:73`; `models/demos/gpt_oss_d_p/tt/config.py:60`/`:69`/`:73`/`:77`/`:81`) | `shard_mapper` returns `ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=mesh_dims)` with `mesh_dims = (None, tensor_dim)` for `tp_axis == 1` (`models/demos/gpt_oss_d_p/tt/config.py:64`). |
| `allreduce(t, ccl, memory_config=None, pad_size=None, axis=0)` | either — identical (`models/demos/minimax_m3/config.py:77`, `models/demos/gpt_oss_d_p/tt/config.py:85`) | **Keep minimax's DRAM comment verbatim** (`:104-111`): the input must be freed between the RS and the AG or peak live memory is `tensor + scattered + gathered` and fragments DRAM under long context (it cites a real OOM). `pad_size` is unused for Llama (nothing needs padding, §1) — keep the parameter, always pass `None`/`0`. |
| `allgather(t, ccl, memory_config=None, axis=0, dim=3, linear=False)` | either — identical (`models/demos/minimax_m3/config.py:135`, `models/demos/gpt_oss_d_p/tt/config.py:138`) | `linear=True` selects `ttnn.Topology.Linear` per call (`models/demos/minimax_m3/config.py:148`). |
| `reduce_scatter(t, ccl, dim=3, axis=0, memory_config=None)` | **`models/demos/minimax_m3/config.py:155` only** | gpt-oss has no `reduce_scatter`. Needed for scheme B and for `03_OUTLINE.md` §3.9's `apply_reduce_scatter`. |
| `__repr__` | `models/demos/gpt_oss_d_p/tt/config.py:158` | Uses the `sp` property. |

Deleted from both: nothing else. `MeshConfig` stays model-agnostic; **TP is the only knob and SP is
derived**, so the `(8,4)`/TP=4 fallback for `R-004` is a parameter change (`DEC-002`).

---

## 4. Collective placement — every row justified

| Module | Where the collective sits | Which collective | Axis | Why |
|---|---|---|---|---|
| `RMSNorm` | inside `forward` | **none** under scheme A | — | The input is full-emb replicated across TP, so every chip computes the same statistic over the same 4096 values. There is nothing to exchange. The 3-op distributed form is present but `is_distributed=False` (`03_OUTLINE.md` §3.4). |
| `RMSNorm` (scheme B only, dormant) | inside `forward` | `ttnn.all_gather` of the `[1,1,32,32]` stats tensor, between `rms_norm_pre_all_gather` and `rms_norm_post_all_gather` | `cluster_axis=1` | Template `models/demos/gpt_oss_d_p/tt/rms_norm.py:67` → `:70` (`dim=3`, `cluster_axis=1`) → `:82`. **This branch has never been executed upstream** — `:33` pins `self.is_distributed = False` with the condition commented out (`R-007`, Appendix F.5). It is the one place a raw `ttnn.*` collective is allowed instead of a `MeshConfig` wrapper, and enabling it requires its own `DEC`. |
| `MLP` (dense SwiGLU) | end of `__call__`, after `down_proj` | `allreduce` (scheme A) **or** `reduce_scatter` (scheme B) | `cluster_axis = tp_axis = 1` | `down_proj` is **row-parallel**: its contraction dim (the 14336 intermediate) is sharded, so each TP chip holds a *partial sum* over `[1,1,S_loc,4096]`. A TP collective is mandatory — the only choice is whether it ends full-width or scattered. Template `models/demos/minimax_m3/tt/dense_mlp.py:99` (the `tp > 1` guard), `:105-107` (RS), `:112` (AR). |
| `Attention` | end of `attention_forward`, after `o_proj` | same choice as MLP | `cluster_axis = tp_axis = 1` | `o_proj` is row-parallel over the head dim (`4096/8 = 512` contraction per chip), so each chip again holds a partial sum. Template `models/demos/gpt_oss_d_p/tt/attention/operations.py:238` → `:252` (`axis=mesh_config.tp_axis`), called at `models/demos/gpt_oss_d_p/tt/attention/prefill.py:304`. |
| `Attention` (SP path, P8) | **inside** the attention core, replacing SDPA | `ttnn.transformer.ring_joint_scaled_dot_product_attention` — the ring reads/gathers K/V across SP *internally* via online softmax; no explicit all-gather | `cluster_axis = sp_axis = 0` | Each SP row holds `S/4` of the sequence but a token must attend the whole causal prefix. Uses `ccl_manager.ring_attention_ccl_semaphore_handles` and `ring_attention_ccl_core_grid_offset`. Template `models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:106`. |
| `Attention` (SP one-shot bootstrap, P8) | inside the attention core | 3 × `all_gather` (Q, K, V) + 1 × `reduce_scatter` + `×1/sp` | `cluster_axis = sp_axis = 0` | Only when Q and the K/V slab are the **same** length, which sliding/ring-joint rejects. `DEC-021`. Template `models/demos/gpt_oss_d_p/tt/attention/prefill.py:235-237` (the three AGs, `dim=2`), `:243` (the RS, `dim=2`), `:254` (the `1/sp` rescale). |
| `Embedding` | after the lookup | **none** | — | The table is **replicated** (`DEC-015`), and the token ids are SP-sharded on rows / replicated on cols (`models/demos/gpt_oss_d_p/tt/model.py:288-306`), so the embedding output is already the full-emb, TP-replicated residual scheme A wants. A vocab-sharded table would need an all-gather per chunk **and** a second layout to debug for a ~1 GiB DRAM saving. |
| `LM head` | after the matmul | **none on device** | — | The vocab shard is concatenated **on the host** in `process_output_prefill` (`models/demos/gpt_oss_d_p/tt/model.py:326-329`), because only the last token's logits are ever needed (prefill's product is the KV cache). `DEC-015`. |
| final `RMSNorm` | — | **none** | — | Same reason as the per-layer norms under scheme A. |
| `DecoderLayer` | **never** | — | — | Both residual adds are elementwise-local: under scheme A each chip holds the same full-width residual and the same full-width sublayer output. |
| `Model` | **never** | — | — | Every collective is inside a sublayer module, by construction. |
| KV-cache write | inside `write_kv_chunk` | **none** — but it is *mesh-aware* | `cluster_axis = sp_axis = 0` (an argument, not a collective) | `ttnn.experimental.deepseek_prefill.update_padded_kv_cache` takes `cluster_axis` and derives each chip's block-cyclic write offset **on-device** from `kv_actual_global` + the chip's SP coordinate. No data crosses chips. Template `models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:125`. |
| indexed RoPE | inside `apply_rope` | **none** — mesh-aware like the KV write | `cluster_axis = sp_axis = 0` (an argument) | `ttnn.experimental.deepseek_prefill.rotary_embedding_indexed` derives this chunk's per-chip start row on-device from `kv_actual_global` + the SP coord, using the same block-cyclic math as the cache writer. Template `models/demos/gpt_oss_d_p/tt/attention/operations.py:79-86`. |

---

## 5. Residual layout — **scheme A (replicated)** (`DEC-018`)

**Choice: A.** The residual stream is `[1, 1, S_loc, 4096]` on every TP column; attention and MLP
each close with a full **all-reduce**; the norms are single-op. `scatter_output` is a real parameter
on `MLP`, `Attention` and `attention_forward` from day one, so B is a flag, not a rewrite
(`models/demos/minimax_m3/tt/dense_mlp.py:38`).

The recipe recommends A (`BRINGUP_RECIPE.md:561`). It is **confirmed**, but the recipe's stated
reason and `R-007`'s are both partly wrong, so here is the argument that actually holds.

### 5.1 The counter-evidence first: scheme B is *not* unproven

`R-007` / Appendix F.5 say B would make Llama the first user of the dead distributed-RMSNorm branch.
That is only true of **one** variant of B. Measured in the template:

- `models/demos/minimax_m3/tt/residual.py:26` — `DEFAULT_USE_SHARDED_RESIDUAL = True`. Minimax ships
  **scheme B by default**.
- `models/demos/minimax_m3/tt/residual.py:32` — `DEFAULT_NORM_MODE = "gather_first"`, i.e. the
  per-norm all-gather runs **before** a plain single-pass `ttnn.rms_norm`; `use_distributed_norm()`
  (`:53`) is therefore `False` by default.

So B-with-`gather_first` is a shipped, exercised layout that never touches the dormant branch, and it
is mathematically **identical** to A (the norm sees the same full-width input). Appendix F.5's
argument rules out *B-with-distributed-norm* only. Any future statement that "B is unproven" should
be narrowed accordingly.

### 5.2 Why A anyway

1. **There is no traffic to win.** For Llama the two schemes cost exactly the same collectives:

   | | per sublayer | per layer | RS input → output | AG input → output |
   |---|---|---|---|---|
   | **A** (replicated) | tail all-reduce = 1 RS + 1 AG | 2 RS + 2 AG | `4096 → 512` | `512 → 4096` |
   | **B** (`gather_first`) | tail RS, plus 1 AG before the next norm | 2 RS + 2 AG | `4096 → 512` | `512 → 4096` |

   Same count, same tensor sizes, same axis. Minimax's measured win for B comes from **sharing** one
   gathered norm output across several consumers — its MoE shared expert *and* its routed experts
   read the same norm output, which is exactly what `models/demos/minimax_m3/tt/residual.py:9-11` describes ("Full width is
   reconstituted only where a column-parallel projection needs it: one all-gather per norm output,
   **shared by every consumer downstream of that norm**"). A dense Llama layer has **one** consumer
   per norm, so there is nothing to share and the win does not transfer.
2. **`G-TP-PARITY` stays a direct comparison.** Under A a module's output is `[1,1,S_loc,4096]` at
   both TP=1 and TP=8, so the P8 gate can compare device-vs-device tensors directly — the sharper
   test the recipe asks for (`BRINGUP_RECIPE.md:845-850`). Under B the multi-device output is
   `[1,1,S_loc,512]` per chip, and the parity test must gather first, putting its own correctness
   inside the measurement.
3. **The model entry point is already full-width.** The embedding is replicated (`DEC-015`), so A's
   residual is what `prepare_inputs_prefill` naturally produces. B would need a per-TP-column slice
   of the embedding output at the one place where a mistake shifts every token in the prompt.
4. **DRAM is not the binding constraint at this stage.** The residual is `S_loc·4096·2 B` = 16 MiB
   per chip at `S = 8192`; B saves 14 MiB of it. The 1 GiB replicated embedding table and the KV
   cache dominate.
5. **It removes a whole class of layout bug from the P5–P7 debugging surface**, which is the recipe's
   own reason and the only one that survives unchanged.

- **Falsifier.** P8 profiling shows the per-sublayer all-gather on the critical path (it should not —
  see 1), or 128k-context DRAM pressure makes the full-width residual binding. Either way the switch
  is `scatter_output=True` plus `gather_before_norm` in the layer — **and it must use `gather_first`,
  never the dormant distributed norm** (§5.1).
- **Blast radius.** `tt/mlp.py` tail, `tt/attention/prefill.py` tail, `tt/rms_norm.py` input,
  `tt/layer.py` add sites, `tt/embedding.py`, `G-TP-PARITY`.

---

## 6. Semaphore lifetime — the statement

> **Every CCL semaphore is allocated exactly once, in `CCLManager.__init__`, and handed out by a
> cycling getter. Never per layer, never per chunk, never per collective.**

Concretely, from `models/demos/gpt_oss_d_p/tt/ccl.py`:

| set | count | allocated at | handed out by | cycle |
|---|---|---|---|---|
| reduce-scatter ping-pong | **6** (`3 × 2`) | `:66-68` (constant at `:65`) | `get_rs_ping_pong_semaphore()` `:88` — returns a 3-slice | index `0/1`, `:92` |
| all-gather ping-pong | **4** (`2 × 2`) | `:72-74` (constant at `:71`) | `get_ag_ping_pong_semaphore()` `:95` — returns a 2-slice | index `0/1`, `:99` |
| barrier | **2** (`2 × 1`) | `:78-80` (constant at `:77`) | `get_barrier_semaphore()` `:102` | index `0/1`, `:105` |
| ring-attention (fwd/bwd pair) | **2** | `:84-86` | read directly as `ccl_manager.ring_attention_ccl_semaphore_handles` | not cycled — the ring op owns both |

Those four counts — **6 / 4 / 2 / 2** — are exactly what `G-SEMAPHORE` asserts: instantiate the
32-layer model and check the list lengths are still 6/4/2/2, **not `32 ×` them**
(`BRINGUP_RECIPE.md:855-857`).

Three properties to preserve, each with its reason:

1. **The CCL core range derives from the real device grid** —
   `mesh_device.compute_with_storage_grid_size()` at `models/demos/gpt_oss_d_p/tt/ccl.py:44`, cores
   at `:46-48`. Measured **(12, 10)** on this Blackhole, so hard-coding 8×8 would leave the
   ring-attention offset at `(7,0)` instead of `(11,0)` and overlap the SDPA cores. The offset is
   `(grid.x - 1, 0)` (`:61`).
2. **Handing out a semaphore cycles a ping-pong index**, so two *consecutive* collectives never
   share one. This is the single most common source of nondeterministic multi-device PCC failures
   (`BRINGUP_RECIPE.md:515-517`). Note the depth is **2**: inside one `allreduce`, the RS takes
   `barrier[0]` and the AG takes `barrier[1]`, so the next `allreduce`'s RS takes `barrier[0]` again
   — a one-op gap. That is the template's design, inherited unchanged; `G-RACE` (3 runs
   bit-identical) is what validates it, and if `G-RACE` fails, deepening the barrier ping-pong from
   2 to 4 is the first thing to try, not the last.
3. **`reset_global_semaphores()` deliberately does not reset the barrier or ring-attention
   semaphores** (`:129`, comment `:132-135`). One-shot prefill never reuses a `CCLManager` across
   runs. Chunked prefill *does* reuse one across `prefill_chunk` calls, and the upstream comment
   marks that as an open TODO — so **P7 must decide** whether to extend the reset, and if it does,
   that is a `DEC` with `G-RACE` as its evidence.

---

## 7. Every collective call site, with `cluster_axis`, `dim` and `topology`

Steady state = scheme A, `(4,8)`, TP=8, SP=4, `L = 32` layers, one chunk.
`ax` is `cluster_axis`. Topology is `ccl_manager.topology` throughout (= `ttnn.Topology.Ring` on the
torus, `Linear` when `PREFILL_TOPOLOGY=linear` — `DEC-020`).

| # | Call site (`path:line` of the template it mirrors) | ttnn op | `ax` | `dim` | topology | calls / chunk |
|---|---|---|---|---|---|---|
| 1 | `tt/attention/operations.py::apply_allreduce` → `MeshConfig.allreduce` **RS half** (`models/demos/minimax_m3/config.py:94`) | `ttnn.experimental.reduce_scatter_minimal_async` | **1** (`tp_axis`) | **3** | Ring | 32 |
| 2 | same → **AG half** (`models/demos/minimax_m3/config.py:115`) | `ttnn.experimental.all_gather_async` | **1** | **3** | Ring | 32 |
| 3 | `tt/mlp.py::__call__` tail → `MeshConfig.allreduce` **RS half** (`models/demos/minimax_m3/tt/dense_mlp.py:112` → `models/demos/minimax_m3/config.py:94`) | `reduce_scatter_minimal_async` | **1** | **3** | Ring | 32 |
| 4 | same → **AG half** (`models/demos/minimax_m3/config.py:115`) | `all_gather_async` | **1** | **3** | Ring | 32 |
| 5 | `tt/attention/dense_sp.py::dense_sp_attention` (P8) (`models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:106`, `dim=2` at `:126`, `cluster_axis` at `:129`) | `ttnn.transformer.ring_joint_scaled_dot_product_attention` | **0** (`sp_axis`) | **2** | Ring (`:132`) | 32 |
| **Steady-state totals** | | **64 RS + 64 AG on axis 1; 32 ring-SDPA on axis 0** | | | | |
| 6 | scheme B only — `tt/mlp.py` / `tt/attention` tail (`models/demos/minimax_m3/tt/dense_mlp.py:105`) | `reduce_scatter_minimal_async` | 1 | 3 | Ring | 0 (dormant) |
| 7 | scheme B only — `tt/rms_norm.py` stats gather (`models/demos/gpt_oss_d_p/tt/rms_norm.py:70`) | `ttnn.all_gather` (non-async) | 1 | 3 | Ring (`:77`) | 0 (dormant) |
| 8 | SP one-shot bootstrap — Q/K/V gather (`models/demos/gpt_oss_d_p/tt/attention/prefill.py:235-237`) | `all_gather_async` via `MeshConfig.allgather` | **0** | **2** | Ring | 96, bootstrap only |
| 9 | SP one-shot bootstrap — output scatter (`models/demos/gpt_oss_d_p/tt/attention/prefill.py:243`) | `reduce_scatter_minimal_async` (raw; the one raw call the template keeps) | **0** | **2** | Ring | 32, bootstrap only |
| 10 | `G-TP-PARITY` harness only — gather a sharded reference | `MeshConfig.allgather` | 1 | 3 | per `DEC-020` | test-local |

Not collectives, but mesh-aware and listed so the `cluster_axis` set is complete:

| Site | op | `cluster_axis` | note |
|---|---|---|---|
| `tt/attention/kv_cache.py::_write_one` (`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:125`) | `ttnn.experimental.deepseek_prefill.update_padded_kv_cache` | **0** (`sp_axis`) | on-device block-cyclic offset from `kv_actual_global` + the chip's SP coord; no data crosses chips |
| `tt/attention/operations.py::apply_rope` indexed branch (`models/demos/gpt_oss_d_p/tt/attention/operations.py:79`) | `ttnn.experimental.deepseek_prefill.rotary_embedding_indexed` | **0** | same block-cyclic math; picks this chunk's cos/sin rows on-device |

**`dim` discipline, stated once:** TP collectives always act on `dim=3` (the feature/hidden
dimension) because TP shards features. SP collectives always act on `dim=2` (the sequence dimension)
because SP shards the sequence. A collective with `cluster_axis=1, dim=2` — or `cluster_axis=0,
dim=3` — is a bug by construction in this model, and reviewing a diff for that pair is the cheapest
CCL review there is.

### 7.1 The one allowed raw-`ttnn` exception

`03_OUTLINE.md` §1 convention 8 forbids raw `ttnn.experimental.*` in a module. Two sites in the
templates break it and both are inherited knowingly:

- **row 7** — `ttnn.all_gather` (the non-async op) on the `[1,1,32,32]` RMSNorm stats tensor. Dormant
  under scheme A. Enabling it needs a `DEC` (`BRINGUP_RECIPE.md:530-532`).
- **row 9** — the raw `reduce_scatter_minimal_async` in the SP bootstrap. It cannot go through
  `MeshConfig.reduce_scatter` unchanged because it scatters on `dim=2` with a `1/sp` rescale, which
  the wrapper does not express. **P8 action:** route it through
  `mesh_config.reduce_scatter(t, ccl, dim=2, axis=sp_axis)` — the union wrapper *does* take `dim`
  (`models/demos/minimax_m3/config.py:155`), so only the rescale stays local. That removes the last
  raw call from the package.

---

## 8. Topology, fabric config and `num_links` (`DEC-020`)

`num_links` comes from `utils/general_utils.get_default_num_links(mesh_device)`
(`models/demos/gpt_oss_d_p/utils/general_utils.py:27`): **1** when `mesh_device.shape[0] == 1`
(`:33`), else **2** on Blackhole (`:35`).

| Mesh | Phase / gate | `num_links` | `ttnn.set_fabric_config` | `CCLManager.topology` | Mesh graph descriptor |
|---|---|---|---|---|---|
| `(1,1)` | P5–P7 unit gates | 1 | none needed (no CCL is entered) | n/a | default |
| `(1,2)`, `(1,4)`, `(1,8)` | P8 `G-TP-PARITY` | **1** | `FABRIC_1D` | `Linear` | default |
| `(4,8)` | P8 `G-MESH-KV`, `G-RACE`; P10 | **2** | `FABRIC_1D_RING` | `Ring` | `TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto` |

Rule and evidence: ring collectives need the cyclic torus route, so `Ring` topology and
`FABRIC_1D_RING` are selected together and require the torus descriptor —
`models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:121` (`PREFILL_TOPOLOGY`, default `"ring"`),
`:122` (`FABRIC_1D` if linear else `FABRIC_1D_RING`), `:161` (`Topology.Linear` if linear else
`Topology.Ring`), descriptor and `channels { count: 2 }` at
`tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto:6`
and `:8`. A single-row `(1,N)` sub-mesh is **not** a ring on this hardware, hence `Linear` there.

**Consequence P8 must not miss:** the `(1,N)` parity meshes run at `num_links = 1` and
`Topology.Linear`, so `G-TP-PARITY` proves the *sharding math*, not the 2-link ring path. Only
`G-MESH-KV` / `G-RACE` on `(4,8)` exercise `num_links=2` + `Ring`. A `(4,8)`-only failure after a
green `G-TP-PARITY` is therefore expected to be a fabric/topology issue, not a sharding one.

---

## 9. SP one-shot bootstrap: keep, gated (`DEC-021`)

The template keeps two SP attention paths and picks between them at
`models/demos/gpt_oss_d_p/tt/attention/prefill.py:191`:
`use_cache_backed_ring = cached_len > 0 or kv_cache.max_seq_len > seq_len * sp`. The bootstrap
(rows 8–9 of §7) exists only because ring-joint SDPA requires Q **shorter** than the K/V slab, which
an equal-sized one-shot request violates.

**Keep it.** Two reasons: (a) `G-MESH-KV` runs both one-shot and chunked
(`BRINGUP_RECIPE.md:858-861`), and one-shot with `max_seq_len == seq_len·sp` hits exactly that case;
(b) it is the only SP path that does not depend on the cache being correct, which makes it the
bisection tool when `G-MESH-KV` fails. Llama's version is simpler than the template's: no sinks, no
sliding window, and `_gather_seq_len` collapses to `return full_seq`
(`models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:36`).

**But it must stay off the default path.** Allocate `max_seq_len > chunk_size · sp` in the runtime so
production always takes the ring, and cover the bootstrap with an explicit test parametrisation
rather than letting a cache-sizing accident silently select it.

---

## 10. What this plan hands to P8, and the one risk it does not cover

- **`G-TP-PARITY`** — modules at `(1,1)` vs `(1,8)`, PCC ≥ 0.999 device-vs-device. Scheme A makes
  this a direct tensor comparison (§5.2).
- **`G-SEMAPHORE`** — assert **6 / 4 / 2 / 2** after building the 32-layer model.
- **`G-RACE`** — 3 runs of the KV harness, bit-identical; log all three hashes. If it fails, §6
  property 2 names the first move.
- **`G-MESH-KV`** — `(4,8)`, per-layer K/V PCC vs golden, one-shot and chunked, min-across-layers
  recorded in the README status table.
- **Not covered by any gate in this plan: `n_kv = 1` at TP=8** (`R-004`, Appendix F.6). SDPA is
  proven safe at `n_kv=1` (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:98`),
  but `update_padded_kv_cache` and `ring_joint_scaled_dot_product_attention` have **never been
  executed** at `n_kv=1`, and `G-KV` runs at TP=1 (8 KV heads) so it cannot cover it. **P8 owns
  closing this.** The pre-costed fallback is `(8,4)`/TP=4/SP=8 → 2 KV heads per chip (`DEC-002`).
  See `07_RISKS.md` `R-012` for the two further exposures this phase found.

---

## 11. Citation verification

Verified by the same run as `G-OUTLINE`: `scripts/verify_citations.py` — **380 / 380 explicit
citations verified, 0 mismatched, 0 missing files**, plus a new document-scan pass covering **140 /
140** `path:line` references in `03_OUTLINE.md` and this file. Raw log:
`raw/G-CCL-PLAN_20260903T170527Z.log` (same transcript as `raw/G-OUTLINE_20260903T170527Z.log`; the two gates share one
verifier run). The 11 wrong line numbers this caught are itemised in `03_OUTLINE.md` §8.
