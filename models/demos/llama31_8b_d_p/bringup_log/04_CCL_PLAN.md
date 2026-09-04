# 04 — Parallelism and CCL plan

Phase P4. Where every collective in the model sits, on which axis, with which `dim` and topology; the
residual-layout decision; and the semaphore-lifetime contract. Date (UTC): 2026-09-04.
Gate: `G-CCL-PLAN`.

The user requirement is explicit and it shapes everything below: **CCLs are part of the modules.**
Attention, MLP and RMSNorm own their own collectives; the layer and the model never call one.
That is what makes each module independently SP-safe, and it is the concrete difference between this
plan and `models/common/modules/mlp/mlp_2d.py:461` (`02_SURVEY.md` §2).

---

## 1. `(mesh, TP, SP)` and the arithmetic

| Field | Value | Derivation |
|---|---|---|
| mesh shape | `(4, 8)` | 32 devices, `ttnn.get_num_devices() == 32`, `get_arch_name() == 'blackhole'` (measured, `raw/G-CARD_20260904T034303Z.log`); the shape `models/demos/gpt_oss_d_p/tt/config.py:15` already pins as validated |
| TP | **8**, on the **columns** (`tp_axis = 1`) | a hard equality, not a bound — §1.1 |
| SP | **4**, on the **rows** (`sp_axis = 0`) | derived: `32 / 8 = 4`. TP is the only knob (`models/demos/minimax_m3/config.py:29-30`) |
| `num_links` | **2** at `(4,8)`; **1** on any `(1,N)` submesh | `get_default_num_links` (`models/demos/gpt_oss_d_p/utils/general_utils.py:27`): `mesh_device.shape[0] == 1 -> 1` (`:33`), else `2` on Blackhole (`:35`) |
| topology | `ttnn.Topology.Ring` at `(4,8)`; `Topology.Linear` selectable | §5 |
| chunk / max_seq_len | deferred (`DEC-004`), constrained: `CHUNK_SIZE % (SP*32) == 0` → `% 128 == 0`, `MAX_SEQ_LEN % CHUNK_SIZE == 0`, `MAX_SEQ_LEN > CHUNK_SIZE` | `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md:62`; `BRINGUP_RECIPE.md:2150` |

### 1.1 TP = 8 is an equality

**`TP == num_key_value_heads == 8`** — the upper and lower bounds collapse onto the same number.
Repeated here because every collective's `dim` depends on it (full derivation:
`00_MODEL_CARD.md` §4.1).

- **`TP <= 8`**: the packed KV cache allocates exactly **one** KV head per chip —
  `models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:95` ("Per-chip cache is one head") and the
  `torch.zeros(num_users * num_layers, 1, seq_local, head_dim)` at `:99`. At TP < 8 the model emits
  `8/TP > 1` local KV heads and the write op aborts with
  `TT_FATAL(cache_shape[1] == input_shape[1], "cache and input num-heads dim must match")`
  (`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp:230`).
- **`TP >= 8`**: above 8 the 8 KV heads no longer cover the TP group and would need replicating — its
  own `DEC`, not taken (and unreachable on 32 devices at SP >= 4).
- **SDPA does not constrain TP**: `TT_FATAL(nqh >= nkv && nqh % nkv == 0)`
  (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:98`) gives
  `4 >= 1 && 4 % 1 == 0` at TP=8.

### 1.2 Every collective's scatter width, computed

A collective on `dim=3` of a `[1, 1, S_loc, W]` tensor splits `W` across 8 devices, so `W/8` must be
a multiple of 32 (tiles are 32x32):

| collective site | `W` | `W/8` | `% 32` | ok |
|---|---|---|---|---|
| attention `o_proj` tail | 4096 | 512 | `512 = 16*32` → 0 | yes |
| MLP `down_proj` tail | 4096 | 512 | 0 | yes |
| LM-head all-gather (vocab) | 128256 | 16032 | `16032 = 501*32` → 0 | yes |
| RMSNorm stats all-gather (scheme B only) | `32*8 = 256` | 32 | 0 | yes |

**No `pad_size` is ever needed.** `MeshConfig.allreduce` carries an optional perf-padding argument
(`models/demos/minimax_m3/config.py:77`, applied at `:88`) because gpt-oss's `2880/8 = 360` is not
tile-aligned; Llama's 4096/8 = 512 is, so the argument is left `None` at every call site and the
`o_proj` pad branch (`models/demos/gpt_oss_d_p/tt/attention/weights.py:64-70`) is deleted rather
than carried. Recorded because "we do not need the padding" is a claim, and the arithmetic above is
its evidence.

---

## 2. The two objects, created once per model

### 2.1 `CCLManager` (`tt/ccl.py`) — persistent CCL resources

Template `models/demos/gpt_oss_d_p/tt/ccl.py:17`, taken essentially whole
(`03_OUTLINE.md` §2.2). Three properties are load-bearing:

1. **The CCL core range derives from the real device grid.**
   `mesh_device.compute_with_storage_grid_size()` (`models/demos/gpt_oss_d_p/tt/ccl.py:44`) — this
   Blackhole is **(12, 10)**, wider than 8x8. The ring-attention CCL workers take the **last compute
   column**, `ring_attention_ccl_core_grid_offset = (grid.x - 1, 0) = (11, 0)`
   (`models/demos/gpt_oss_d_p/tt/ccl.py:61`), because the ring op requires the CCL workers and the
   SDPA compute cores to be non-overlapping.
2. **This is the *opposite* of the SDPA program grid, and the two must not be unified.** The SDPA
   program grid stays a pinned 8x8 (`03_OUTLINE.md` §2.7): the op asserts
   `ccl_core_grid_offset.x >= program_config.compute_with_storage_grid_size.x`
   (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp:421`), so
   `11 >= 8` passes and a *derived* 12 fails. A derived grid passes **every** single-card gate and
   fails only at SP > 1 in P8.
3. **Semaphores are allocated once, in `__init__`, and cycled per call** — never per layer, never per
   chunk. §3.

### 2.2 `MeshConfig` (`tt/config.py`) — the parallelism decision and the collective wrappers

The **union** of the two in-repo copies (`03_OUTLINE.md` §2.1). Modules call
`self.mesh_config.<collective>(t, self.ccl_manager, ...)`; **never** raw `ttnn.experimental.*` inside
a module, which is how semaphore-reuse bugs get in. One exception, logged: `DEC-028`.

**An all-reduce is reduce-scatter + all-gather**, not `all_reduce_async`
(`models/demos/minimax_m3/config.py:94` then `:115`). That is the repo's converged answer and this
plan does not invent another. Re-measured on this tree, reproducing the recipe's counts exactly:

| op | uses across `minimax_m3`, `gpt_oss_d_p`, `deepseek_v3_d_p`, `tt_transformers` |
|---|---|
| `ttnn.experimental.all_gather_async` | **29** |
| `ttnn.experimental.reduce_scatter_minimal_async` | **18** |
| `ttnn.experimental.all_reduce_async` | **2** |

---

## 3. Semaphore lifetime and depth

**Statement (the form `G-CCL-PLAN` asks for):** every CCL semaphore this model uses is
**allocated once in `CCLManager.__init__`**, is **cycled per call** by a ping-pong index, and is
**never allocated per layer or per chunk**. One `CCLManager` serves all 32 layers and every chunk
of a request.

**Depth, per class** (`models/demos/gpt_oss_d_p/tt/ccl.py:63-86`):

| class | count | per call | depth | allocated at |
|---|---|---|---|---|
| reduce-scatter ping-pong | `3 * 2 = 6` | 3 | **2** | `models/demos/gpt_oss_d_p/tt/ccl.py:65` |
| all-gather ping-pong | `2 * 2 = 4` | 2 | **2** | `models/demos/gpt_oss_d_p/tt/ccl.py:71` |
| barrier | `2 * 1 = 2` | 1 | **2** | `models/demos/gpt_oss_d_p/tt/ccl.py:77` |
| ring-attention (fwd/bwd pair) | `2` | 2 | 1 (a pair, not a ring) | `models/demos/gpt_oss_d_p/tt/ccl.py:84` |

**Total: 14 global semaphores per model.** `G-SEMAPHORE` asserts exactly these four lengths at
construction, after dozens of getter cycles, and after a real multi-layer harness run — the failure
it exists to catch is `n_layers x` any of them.

**The barrier ping-pong is only 2 deep, and that is the thin one.** A reduce-scatter takes
`barrier[0]`, the all-gather that follows takes `barrier[1]`, and the *next* reduce-scatter takes
`barrier[0]` again — a **one-op gap**. Under scheme A each layer issues 4 barrier-consuming
collectives (RS+AG for attention, RS+AG for the MLP), so across 32 layers `barrier[0]` is reused 64
times with one op of separation each time. `DEC-026` records the decision to ship at depth 2 and
names deepening it to 4 as `G-RACE`'s **first** move if `G-RACE` fails — before suspecting the model.

**`reset_global_semaphores` deliberately does not reset the barrier or ring-attention semaphores**
(`models/demos/gpt_oss_d_p/tt/ccl.py:132`, an open upstream TODO whose own comment says one-shot
prefill never reuses a `CCLManager` across runs). Chunked prefill **does** reuse one across chunks.
`DEC-026` covers this too.

---

## 4. Collective placement, by module

Every row is justified by *why the tensor is incomplete at that point*, which is the only reason a
collective is ever correct.

| Module | Where the collective sits | Which collective | Why it is necessary there |
|---|---|---|---|
| `Embedding` | nowhere | **none** | the table is **replicated** (`DEC-024`), so the lookup already yields the full 4096-wide row on every TP column. A vocab-sharded table would need an all-gather; that is the alternative `DEC-024` declines. |
| `RMSNorm` | nowhere under scheme A | **none** | the input is full-emb replicated across the TP columns, so mean-square is computable locally. Under scheme B it becomes `rms_norm_pre_all_gather` → all-gather of the `[1,1,32,256]` stats → `rms_norm_post_all_gather` (`models/demos/gpt_oss_d_p/tt/rms_norm.py:67-90`), which is `DEC-028`'s one allowed raw `ttnn.all_gather`. Branch present, default off. |
| `Attention` | end of the forward, **after `o_proj`** | `allreduce` on the **TP** axis (scheme A) / `reduce_scatter` (scheme B seam, refuses until P8) | `o_proj` is **row-parallel** over the head dim: each TP chip contracts only its own 512 of the 4096 input features, so its `[1,1,S_loc,4096]` output is a **partial sum**. Without the TP collective every chip holds 1/8 of the answer. |
| `Attention` (SP path, P8) | **inside** SDPA | ring-attention halo exchange on the **SP** axis, via `ccl_manager.ring_attention_ccl_semaphore_handles` + `ring_attention_ccl_core_grid_offset` | chunk *k*'s queries live on one SP row but must attend K/V spread block-cyclically across all 4 rows. Template `models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:41`. |
| `MLP` (dense SwiGLU) | end of `__call__`, **after `down_proj`** | `allreduce` on the **TP** axis (scheme A) / `reduce_scatter` (scheme B) | identical argument: `gate/up` are column-parallel so `down_proj` is **row-parallel** over the 14336 intermediate dim, and each chip holds a partial sum over its 1792. Template `models/demos/minimax_m3/tt/dense_mlp.py:96-112`. |
| `LM head` | after the matmul, **only when logits are wanted** | `allgather` on the **TP** axis, `dim=3` | the weight is column-parallel over the vocab, so each chip holds 16032 of 128256 logits. Prefill's product is the KV cache, so this fires only for `G-MODEL`'s top-1 check — and even there `Model.process_output_prefill` can gather on the **host** instead (`models/demos/gpt_oss_d_p/tt/model.py:322`). |
| `DecoderLayer` | **never** | — | both residual adds are elementwise-local by construction: attention and MLP each hand back a tensor whose TP collective has already run, in the same layout as the residual. |
| `Model` | **never** | — | the layer loop is pure composition. `prepare_inputs_prefill` *shards* (a mapper, not a collective) and `process_output_prefill` gathers on the host. |

**Collectives go on the TP axis only** — with the single exception of the P8 SP ring path, which is
on the SP axis *by definition* (it exchanges sequence, not features). That rule is what makes every
module SP-safe, and it is precisely what `models/common/modules/mlp/mlp_2d.py` violates: its prefill
path reduce-scatters on `cluster_axis = 1` (`models/common/modules/mlp/mlp_2d.py:256`, set at `:259`)
and closes with an all-reduce on `cluster_axis = 0` (`:461`, via `:361`). With SP on the row axis that
final all-reduce sums activations belonging to **different tokens** — silently wrong, and it would
still produce a plausible PCC on a one-row mesh.

### 4.1 Two things that take a `cluster_axis` and are **not** collectives

Both appear in the call sites below with a `cluster_axis` argument, and mistaking either for a
collective is how a reader concludes the KV write needs a barrier.

- `ttnn.experimental.deepseek_prefill.update_padded_kv_cache(..., cluster_axis=sp_axis)`
  (`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:125-132`) — each chip writes **its own** shard;
  `cluster_axis` only tells the op which mesh coordinate to use for the block-cyclic row arithmetic.
- `ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(..., cluster_axis=sp_axis)`
  (`models/demos/gpt_oss_d_p/tt/attention/operations.py:79-86`) — same: it derives this chunk's
  per-chip start row on device from `kv_actual_global` plus the device's SP coordinate. No comms.

---

## 5. Every collective call site

Phases P5–P7 run at `(1,1)`, where `tp == 1` and every row below is **skipped by the
`if self.mesh_config.tp > 1` guard** (`models/demos/minimax_m3/tt/dense_mlp.py:99`). The table is
therefore the P8 wiring, written in P4 so the branch exists from the first line of P5 code.

`axis` is the `cluster_axis` passed to the ttnn op. `tp_axis = 1`, `sp_axis = 0`.

| # | Call site | Wrapper | ttnn op(s) | `dim` | `axis` | topology | `num_links` |
|---|---|---|---|---|---|---|---|
| 1 | `attention/operations.apply_allreduce`, after `o_proj` | `MeshConfig.allreduce` | `reduce_scatter_minimal_async` then `all_gather_async` | 3, then 3 | 1 (TP) | `ccl.topology` (Ring) | 2 |
| 2 | `mlp.MLP.__call__`, after `down_proj` | `MeshConfig.allreduce` | `reduce_scatter_minimal_async` then `all_gather_async` | 3, then 3 | 1 (TP) | `ccl.topology` (Ring) | 2 |
| 3 | `attention/operations.apply_reduce_scatter` (scheme B seam; **refuses** until P8) | `MeshConfig.reduce_scatter` | `reduce_scatter_minimal_async` | 3 | 1 (TP) | `ccl.topology` | 2 |
| 4 | `mlp.MLP.__call__` with `scatter_output=True` (scheme B seam; **refuses** until P8) | `MeshConfig.reduce_scatter` | `reduce_scatter_minimal_async` | 3 | 1 (TP) | `ccl.topology` | 2 |
| 5 | `lm_head.LMHead.__call__(gather=True)` | `MeshConfig.allgather` | `all_gather_async` | 3 | 1 (TP) | `ccl.topology` | 2 |
| 6 | `rms_norm.RMSNorm.forward`, `is_distributed=True` (P8, scheme B only) | **raw `ttnn.all_gather`** (`DEC-028`) | `ttnn.all_gather` | 3 | 1 (TP) | Ring | `get_default_num_links` |
| 7 | `attention/dense_sp.dense_sp_attention` — the ring halo (P8) | the op itself | `ttnn.transformer.ring_joint_scaled_dot_product_attention` | — (seq) | 0 (SP) | ring, over `ring_attention_ccl_semaphore_handles` | 2 |
| 8 | `attention/prefill` SP **bootstrap** — only when `max_seq_len == seq_len * sp` (P8) | `MeshConfig.allgather` x3 then a raw `reduce_scatter_minimal_async` | `all_gather_async` (Q, K, V) then `reduce_scatter_minimal_async` | **2** (seq) | 0 (SP) | `ccl.topology` | 2 |
| 9 | every wrapper above | — | `get_barrier_semaphore()` is passed to each of them | — | — | — | — |

Notes that the table cannot carry:

- **Rows 1–5 are the whole steady-state cost: 4 barrier-consuming collectives per layer**
  (2 all-reduces = 2 RS + 2 AG), 128 per 32-layer forward. Row 5 fires once per request at most.
- **Row 8 is a trap, not a feature.** It is the equal-length one-shot fallback
  (`models/demos/gpt_oss_d_p/tt/attention/prefill.py:234-256`), and `BRINGUP_RECIPE.md:2150` names it:
  if `max_seq_len == chunk_size` the ring op has no room and attention silently runs this *different*
  core, so P8/P10 must choose `max_seq_len > chunk_size` and **log which core actually ran**.
- **Topology and fabric config must agree or the box hangs.** A `Topology.Ring` collective on a plain
  `FABRIC_1D` fabric **hangs rather than erroring**. The pairing is
  `FabricConfig.FABRIC_1D_RING` ↔ `Topology.Ring` and `FabricConfig.FABRIC_1D` ↔ `Topology.Linear`,
  selected together — the template does exactly that from one variable
  (`models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:121-122` and `:161`). `DEC-027`.
- **A `(1,N)` submesh does not exercise the deployment fabric.** `get_default_num_links` returns 1 for
  any single-row mesh (`models/demos/gpt_oss_d_p/utils/general_utils.py:33`), so `(1,8)` parity runs
  1 link. `(2,8)` is the cheapest shape that puts the 2-link Ring transport under test, which is why
  `G-TP-PARITY` runs five shapes and not four.

---

## 6. Residual layout: scheme A (replicated), and the reason is cost equivalence

Full reasoning in `DEC-025`. Summary, because the wrong reason is more persuasive than the right one:

- **Scheme A (taken)** — the residual stream is full `[1,1,S_loc,4096]` on every TP column; attention
  and MLP each close with a **full all-reduce**; norms are single-op and local.
- **Scheme B** — the residual is `[1,1,S_loc,512]` (`4096/8`, tile-aligned); attention and MLP close
  with a **reduce-scatter only**, and each norm either all-gathers first
  (`DEFAULT_NORM_MODE = "gather_first"`, `models/demos/minimax_m3/tt/residual.py:32`) or runs the
  3-op distributed RMSNorm.
- **The wrong reason to take A:** "B is unproven, because
  `models/demos/gpt_oss_d_p/tt/rms_norm.py:33` pins `is_distributed = False` with the condition
  commented out." That branch is indeed dormant — but B does not require it:
  `models/demos/minimax_m3/tt/residual.py:26` ships **scheme B on by default** with `gather_first`.
  Only **B-with-distributed-norm** is unproven.
- **The right reason:** on a *dense* model A and B issue the **identical** collectives per layer —
  2 reduce-scatters + 2 all-gathers, same sizes, same axis. B's win in Minimax comes from sharing one
  gathered norm output across several MoE consumers
  (`models/demos/minimax_m3/tt/residual.py:9-11`), and **Llama has no such consumers**: each norm
  output feeds exactly one module. A additionally keeps `G-TP-PARITY` a direct device-vs-device
  comparison, and a replicated embedding already yields a full-width residual.

`scatter_output` is wired from day one (`models/demos/minimax_m3/tt/dense_mlp.py:38`) so switching is
a flag, not a rewrite — and any module that cannot honour it **refuses `scatter_output=True` loudly**
rather than half-wiring the scheme (`BRINGUP_RECIPE.md:1206-1208`). Rows 3, 4 and 6 of §5 are that seam.

---

## 7. What this plan does not settle

- **Whether the barrier depth of 2 is sufficient.** Argued, not measured. `G-RACE` is the measurement
  and `DEC-026` names the fix.
- **Which fabric descriptor P8 pins.** Three BH-galaxy torus descriptors exist —
  `tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto`
  (what the working template's harness uses),
  `tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_sp4_torus_xy_graph_descriptor.textproto` and
  `tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto`
  (the two the recipe names). `G-FABRIC-MATRIX` picks one by measurement; `DEC-027` records the
  default and the reason it is not yet pinned.
- **Whether `quiesce_devices()` placement is right.** P8 step 2's requirement is recorded in
  `DEC-027`'s blast radius and owned by `G-TP-PARITY`, the one gate that holds two overlapping
  submeshes at once.
- **Any collective on the embedding or the LM head under a TP-sharded vocab.** `DEC-024` replicates
  the table, so those rows are absent by decision rather than by omission.
