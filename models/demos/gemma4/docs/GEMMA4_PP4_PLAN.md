# Gemma4 prefill: pipeline parallelism (PP=4 × [8,1]) — implementation plan

> **Start with [`GEMMA4_PP4_OVERVIEW.md`](GEMMA4_PP4_OVERVIEW.md)** — what PP=4 is, how it works,
> the headline numbers and the ranked next steps. This document is the detail behind it.

**Status:** IMPLEMENTED and measured. Results, and the four things this plan got wrong, are in
[`GEMMA4_PP4_RESULTS.md`](GEMMA4_PP4_RESULTS.md) — read that alongside this. Headline: 256k in
**11.64 s / 22,517 tok/s** against the 8×4 baseline's 13.6 s / 19,287, a **1.17×** end-to-end and
**1.39×** on stage compute. The three corrections that matter to a reader of this document:
`torus_y` is the WRONG descriptor for Gemma4 (§3.1-A below is superseded — its CP collective is
`Topology.Linear` and the baseline opens plain `FABRIC_2D`); TP=1 breaks `nlp_concat_heads`, not
the matmuls (risk 1); and a later rank must CLONE the activation it receives, because
`Gemma4DecoderLayer` deallocates its input.
**Branch:** `gemma4-prefill-pr` (`b7babca5380`) in `/data/kmabee/tt-metal-2`.
**Machine:** `bh-glx-120-b03u02`, one 32-chip Blackhole galaxy.
**Baseline to beat:** 256k @ 8×4, chunk 8192 = **13.6 s device / 19271 tok/s**
(`~/debug-docs/gemma4_prefill_bringup-noissue/logs/passing_8x4_256k_20260907.log`).
**Prior art being ported:** Mistral Small 4 PP=4, commits `48bf86595ae`, `e0abf839b91`,
`7dec8e0f773`, `8068fed3321`, `c2238ee9f41` in `/data/kmabee/tt-metal-3` — measured 1.29–1.57×
throughput and 1.5–2.9× single-request latency vs. single-rank on the same model.

---

## 0. TL;DR for the implementer

Carve the one 32-chip galaxy into **four Z-connected `[8,1]` column sub-meshes**, four MPI ranks,
15 Gemma4 layers each, hidden state handed stage→stage over a ttnn `MeshSocket` on fabric.
Per stage: **SP=CP=8, TP=1**.

The pipeline engine (rank topology, layer split, H2D/D2D sockets, LayerAck, migration, shutdown)
**already exists** in `models/demos/common/prefill/runners/` and is model-agnostic. The Mistral
commits added *only* configuration on top of it. **For Gemma4 the engine side is likewise nearly
free; the work is model-side** — Gemma4's model, runtime and adapter all assume
`first_layer_idx == 0`, a single rank, and `mesh_shape == (8,4)`, and they index `layer_types`
and weight keys by *local* layer index.

Roughly: ~6 files of real model changes, 2 mesh-graph descriptors + 2 YAMLs ported, one new
TP=1 weight cache to build, and a measurement harness.

---

## 1. Why PP=4 × [8,1] is the right target for Gemma4 (the arithmetic)

### 1.1 It removes exactly the headroom the branch identified

`gemma4_branch_overview.md` §4, global layer at 4k, per layer, **4397.8 µs total**:

| bucket | µs | % |
|---|---:|---:|
| SDPA (`ring_joint`) | 1178.9 | 26.8 |
| 5 matmuls | 843.0 | 19.2 |
| **4 all-gathers + 2 reduce-scatters (TP CCL)** | **1344.8** | **30.6** |
| 8 layernorms | 475.2 | 10.8 |
| other | ~556 | 12.6 |

Those all-gathers and reduce-scatters are **TP-axis collectives**. At TP=1 they are gone —
`ccl_allreduce` and `ccl_allgather` (`models/demos/gemma4/tt/ccl.py:422`, `:490`) both
short-circuit on `mesh_config.tp <= 1` and return the tensor unchanged.

### 1.2 The trade is exactly neutral on everything else

Let a layer's device cost at TP=`t` be `W/t + X(t) + R(t)`, where `W` is TP-parallelisable work
(matmuls, SDPA math, norms), `X` is TP CCL, and `R` is the CP/ring collective, whose per-device
traffic is proportional to local KV heads and therefore scales as `1/t` too.

* single rank, 60 layers, t=4: `60·(W/4 + X + R₄)` = `15W + 60X + 60R₄`
* PP=4, 15 layers/stage, t=1: `15·(W + 0 + 4R₄)` = `15W + 60R₄`

**Speedup = 1 + 60X / (15W + 60R₄)**. Plugging the §4 numbers (X/(L−X) = 1344.8/3053) gives an
ideal ceiling of **≈1.44×**, before pipeline fill and the D2D hop. That is the same band Mistral
actually measured (1.29–1.57×), which is a good sign the model is right.

### 1.3 Memory is neutral, which is why [8,1] is even legal

Gemma4 shards weights **only on the TP axis** (`MeshConfig.shard_mapper` uses
`dims=(None, tensor_dim)` for `tp_axis=1`); the SP axis replicates. So going TP 4→1 multiplies
per-device weight bytes by 4, and PP=4 divides layers by 4. Net zero.

Measured from the existing cache (`tensor_cache_bf16_mesh8x4`): a sliding layer is 486 MB and a
global layer 654 MB *mesh-wide* at bfp8. Per device:

| | 1 rank, 8×4 (TP=4) | PP=4, [8,1] (TP=1) |
|---|---:|---:|
| layer weights / device | 60 layers ÷ 4 ≈ **7.7 GB** | 15 layers × 1 ≈ **7.7 GB** |
| KV cache / device (256k, 2 users, bfp8) | ≈ **7.3 GB** | ≈ **7.3 GB** |

Same on both. The one real delta: the **token embedding** (262144 × 5376) is TP-sharded today
(~0.7 GB/device) and unsharded at TP=1 (~2.8 GB/device) — so gate it to the first rank only, and
gate the final norm + tied LM head to the last rank. That is a correctness/cache change anyway
(§3.2, item E).

### 1.4 CP=8 is preserved

`chunk >= 1024 * cp` (the `ring_joint` halo constraint). `[8,1]` keeps CP=8, so **chunk 8192 stays
legal and stays the canonical config** — the same chunk the 13.6 s baseline uses, so the comparison
is apples-to-apples. This is why `[8,1]` beats the alternatives:

| stage shape | PP | CP | min chunk | verdict |
|---|---:|---:|---:|---|
| **`[8,1]`** | **4** | **8** | **8192** | **target.** MGD + YAML already written for Mistral; kills all TP CCL; memory-neutral |
| `[8,2]` | 2 | 8 | 8192 | fallback if TP=1 hits an L1/program-config wall. Halves TP CCL only; needs a NEW mesh-graph descriptor; balances global layers perfectly (5 per stage) |
| `[4,2]` | 4 | 4 | 4096 | **don't.** CP=4 is the measured-worse axis (4×8 = 17.5 s vs 8×4 = 13.6 s) |

### 1.5 The Gemma4-specific catch Mistral did not have: stage balance is by GLOBAL layer count

`layer_types` is 50 `sliding_attention` + 10 `full_attention`, global at indices
**5, 11, 17, 23, 29, 35, 41, 47, 53, 59** — every 6th. Sliding layers are flat in context
(3.37 → 3.49 ms across chunk depth); global layers pay the whole prefix (3.84 → 23.48 ms).

An even 15/15/15/15 split gives **2 / 3 / 2 / 3** global layers per stage. Pipeline throughput is
`1/max(stage)`, so the 3-global stages set the rate and ~20 % of the win evaporates.

10 globals do not divide by 4, so compensate with **unequal layer counts**: give the 3-global
stages fewer sliding layers. The runner already exposes this — **`PREFILL_PP_LAYER_COUNTS`**
(`runner_utils.compute_layer_split`), no code needed.

Candidate: `PREFILL_PP_LAYER_COUNTS=17,13,17,13`

| stage | layers | globals | sliding |
|---|---|---:|---:|
| 0 | 0–16 | 2 (5, 11) | 15 |
| 1 | 17–29 | 3 (17, 23, 29) | 10 |
| 2 | 30–46 | 2 (35, 41) | 15 |
| 3 | 47–59 | 3 (47, 53, 59) | 10 |

Treat the exact counts as a knob to sweep (16,14,16,14 / 17,13,17,13 / 18,12,18,12), not as a
derived truth. **Note the Mistral correction that does *not* transfer:** the Mistral docs conclude
"stage balance is a non-issue, the stage-1 outlier is layer 1". That was an MoE model with uniform
layers. For Gemma4 the 6:1 sliding/global alternation makes stage balance a real, first-order knob.

---

## 2. What already exists — do not rebuild any of this

Verified present in `/data/kmabee/tt-metal-2` on this branch (the common prefill engine is on
`main`, so the Gemma4 branch inherits it):

| Thing | Where | Notes |
|---|---|---|
| Multi-rank pipeline engine | `models/demos/common/prefill/runners/prefill_runner.py` | owns rank topology, layer split, sockets, LayerAck, migration, shutdown |
| D2D `MeshSocket` endpoints | `prefill_runner.py:162` `build_d2d_pipeline_endpoints` | sender/receiver per rank, fabric-link lease/reclaim |
| Per-rank layer split + override | `runner_utils.py:compute_layer_split` | reads `PREFILL_PP_LAYER_COUNTS` |
| Activation spec + mapper | `runner_utils.py:activation_global_spec`, `prefill_runner.py:64` | `[1,1,chunk,hidden]` bf16 TILE, `Shard(2)` on SP × `Shard(3)`-or-`Replicate` on TP |
| ttrun launcher | `models/demos/common/prefill/runners/run_pipeline_prefill.sh` | takes a rank-binding YAML |
| Producer | `models/demos/common/prefill/runners/prefill_producer.py` | unchanged for PP |
| Gemma4 `text_config` handling in the engine | `prefill_runner.py:522` | already resolves `hf_config.text_config.hidden_size` for D2D width |
| `pipeline_activation_emb_tp_sharded = False` | `models/demos/gemma4/tt/runners/adapters/gemma4.py:42` | already correct: Gemma4's activation is emb-replicated across TP |
| Adapter already forwards the rank fields | `adapters/gemma4.py:109-111` | it just rejects them one line earlier |

**Analyzers and harness from the Mistral work** — `/data/kmabee/tt-metal-3/models/demos/
deepseek_v3_d_p/tests/perf/pp4/` (`analyze_pp.py`, `analyze_layer_budget.py`, `analyze_kv_ramp.py`,
`analyze_ttft.py`, `summarize_campaign.py`, `gen_pp4_binding.py`, `probe_columns.py`). These read
runner logs / Tracy CSVs and touch no device. **They are model-agnostic** — port them as-is
(`analyze_kv_ramp.py` takes `layers_per_stage`, pass 15 not 1).

---

## 3. Work items

### 3.1 Port the topology (mechanical — copy from tt-metal-3)

**A. Mesh-graph descriptors** — neither exists in `tt-metal-2`; copy verbatim:

```
tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_4x8x1_z_chain_graph_descriptor.textproto
tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_4x8x1_z_chain_torus_y_graph_descriptor.textproto
```

`torus_y` is the one to use: `dim_types: [RING, LINE]` wraps the SP axis so Gemma4's ring
attention runs Ring, matching the 8×4 baseline (which opens `FABRIC_2D_TORUS_XY`). The plain
`z_chain` is the fallback if a torus mode refuses to route the D2D socket.
**A torus mode MUST match the descriptor's `dim_types` — a Ring collective on an axis the fabric
does not wrap hangs.**

**B. Rank-binding YAMLs** — port the templates and write Gemma4 `global_env`:

```
models/demos/common/prefill/runners/topology_configuration/
  pipeline_prefill_request_intragalaxy_4rank_8x1_torus_y.yaml            (template, from tt-metal-3)
  pipeline_prefill_request_intragalaxy_4rank_8x1_torus_y_profile.yaml    (template, from tt-metal-3)
  gemma4_pipeline_prefill_4rank_8x1_torus_y.yaml                         (NEW — Gemma4 global_env)
```

Gemma4 `global_env` (differs from Mistral's):

```yaml
global_env:
  PREFILL_FABRIC_MODE: "2d_torus_y"
  PREFILL_MODEL: "gemma4_31b"
  PREFILL_MANIFEST: "models/demos/gemma4/tt/runners/manifests/gemma4_31b_pp4.json"
  PREFILL_SP: "8"
  PREFILL_TP: "1"
  PREFILL_NUM_LAYERS: "60"
  PREFILL_CHUNK_SIZE: "8192"
  PREFILL_MAX_SEQ_LEN: "262144"
  PREFILL_NUM_USERS: "2"
  PREFILL_USE_TRACE: "1"
  PREFILL_H2D_SERVICE_ID: "gemma4_prefill"
  PREFILL_PP_LAYER_COUNTS: "17,13,17,13"
  PREFILL_PP_D2D_FIFO_BYTES: "32768"
  LOGURU_LEVEL: "INFO"
```

Also add a PP manifest `models/demos/gemma4/tt/runners/manifests/gemma4_31b_pp4.json` mirroring
`gemma4_31b.json` with `PREFILL_TP=1`. Note `PREFILL_MANIFEST` **must** live in the YAML's
`global_env`: exporting it in the shell reaches rank 0 only, and the runner's
`_assert_ranks_agree_on_config` will (correctly) abort the run.

**C. Regenerate the per-host device map — NOT OPTIONAL.**

```bash
python models/demos/gemma4/tests/perf/pp4/gen_pp4_binding.py \
  --template models/demos/common/prefill/runners/topology_configuration/gemma4_pipeline_prefill_4rank_8x1_torus_y.yaml
```

The `[8,1]` column → physical device map is **per-galaxy** and a wrong map **does not error** — it
silently builds four stages that are not columns and reports plausible, wrong numbers. A column
spans two trays, so it cannot be derived from tray discovery. `gen_pp4_binding.py` derives it live
via `create_submeshes(MeshShape(8,1))` and writes `<stem>.<hostname>.yaml`, which every runner
prefers automatically. `bh-glx-120-b03u02`'s map already exists in tt-metal-3 (in
`pipeline_prefill_request_intragalaxy_4rank_8x1_torus_y.bh-glx-120-b03u02.yaml`) and can be
cross-checked against a fresh generation. The galaxy must be idle to run it.

### 3.2 Model-side changes (the real work)

Everything below is in `models/demos/gemma4/`. The single organising idea:

> **A rank owns global layers `[first_layer_idx, first_layer_idx + num_layers)`. Every index into
> `layer_types`, every weight key `...layers.{i}...`, and every cache path must use the GLOBAL
> index. Only `self.layers[...]` uses the local one.**

**D. `tt/model.py` — `Gemma4Model`: accept a layer window.**

Add `first_layer_idx: int = 0`, `is_first_rank: bool = True`, `is_last_rank: bool = True` to
`__init__`, and define `gidx = lambda i: self.first_layer_idx + i`. Sites to fix (line numbers at
`b7babca5380`):

| line | today | change |
|---|---|---|
| 406–412 | `kv_shared_layer_map` built over `range(first_shared_idx, n_layers)` | global indices; for 31B `num_kv_shared_layers == 0`, so this is dead code — assert it stays 0 rather than porting it |
| 555 | `Gemma4DecoderLayer(..., layer_idx=i)` | `layer_idx=gidx(i)` — this is what selects the weight keys, the layer type, and the cache path |
| 573 | `Gemma4AttentionConfig(hf_config, i)` | `gidx(i)` |
| 622 | `last_kv_layer_by_type[layer_types[i]] = i` | global type, local value (it indexes `self.layers`) — be explicit |
| 866–880 | `_get_rope_mats(layer_idx)` → `layer_types[layer_idx]` | callers must pass the global index |
| 1065, 1079 | `{layer_types[i] for i in range(len(self.layers))}` | `range(first, first+n)` |
| 1133, 1139, 1205, 1220, 1231, 1261, 1263 | forward loop, `layer_types[i]` with `i = enumerate(self.layers)` | `gidx(i)` |

Also in the same file:
* **Embedding**: build `embed_tokens` only when `is_first_rank`.
* **Final norm**: `forward` currently always runs `self.norm` before the `_prefill_trace_mode`
  early-return (`model.py:1316`). A non-last rank must return the **pre-norm residual stream** —
  the next stage's layer 0 expects raw hidden. Gate: build `self.norm` only on the last rank and
  return `hidden_states` before it when `not is_last_rank`.
* **LM head**: `tie_word_embeddings: true`, so the head is only constructed when
  `embed_tokens.weight` is in the state dict. Gating the embedding to rank 0 gates this too;
  make sure rank 0 does *not* then build a head it will never use.

**E. `tt/common.py` — `create_tt_model`: thread `first_layer_idx` / `is_first_rank` /
`is_last_rank` through to `Gemma4Model`.** Default them so every existing caller is unchanged.
Note `hf_config=model_args` (a `Gemma4ModelArgs`), whose `layer_types` is the **full 60-entry**
tuple from HF even when `num_hidden_layers` has been overwritten to 15 — `__post_init__` only
truncates when `layer_types is None`. So slicing is your job, not the config's.

**F. `tt/tt_prefill_runtime.py` — `TtPrefillRuntime`: the PP branches.**
Model on `models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py`, which does all of this already.

1. Delete the `NotImplementedError` at `:64-68`. Replace the mesh-shape check with
   `sp * tp == mesh_device.get_num_devices()` and the CP/chunk arithmetic already there.
2. `_build_model`: pass the three new args to `create_tt_model`.
3. `make_chunk_input`: on a non-first rank return a **placeholder activation**, not tokens —
   `[1, 1, chunk/sp, hidden/tp]` bf16 TILE, replicated. See DeepSeek's
   `make_placeholder_activation` (`tt_prefill_runtime.py:365-392`).
4. `_forward`: on a non-first rank skip `transform_and_embed_prefill_inputs_device` entirely and
   pass the received tensor straight in as `x`.
5. `_normalize_input`: its token-count check only makes sense on the first rank; on a non-first
   rank validate the activation shape instead.
6. `prefill_chunk`: **return the output activation when `not is_last_rank`** (today it always
   returns `None`). Under trace, return the persistent `self._trace_output` — the runner's
   `_compute_and_send` forwards it with `deallocate=False` because the replay refreshes it in
   place. Mirror DeepSeek `:699` and `:756`.
7. `capture_trace`: keep `self._trace_output` alive on a non-last rank (today it is deallocated).
8. `warmup_ack_count` already returns `self.config.num_layers`, which is per-rank — correct.

**G. `tt/runners/adapters/gemma4.py`.**
* `_validate`: drop the `(8,4)` and single-rank rejections. Keep real invariants:
  `sp * tp == num_devices`, `chunk % (sp * 1024) == 0`, `max_seq_len % chunk == 0`.
* `weight_cache_path(mesh_shape)`: already keys off `mesh_shape` → resolves
  `tensor_cache_bf16_mesh8x1` for a `[8,1]` rank. No change, but verify.
* Consider implementing `layer_split_boundaries(num_layers)` → `None` (dense, unconstrained) —
  it is already the base-class default, so this is documentation only.

**H. `tt/runners/kv_caches.py` — two fixes.**
* `:58` `layer_types = tuple(hf_config.layer_types[:num_layers])` → slice
  `[first_layer_idx : first_layer_idx + num_layers]`. Add `first_layer_idx` to
  `allocate_ring_kv_caches` and pass it from the adapter.
* `:60` `local_heads = 1 if layer_type == "full_attention" else num_key_value_heads // tp`.
  The `1` is a **hardcoded TP=4 value** (`num_global_key_value_heads = 4`, 4/4 = 1). At TP=1 it
  must be 4. Replace with `config.num_key_value_heads // mesh_config.tp` for both branches —
  `Gemma4AttentionConfig` already returns 4 KV heads / head_dim 512 for `full_attention`
  (`tt/attention/__init__.py:48-51`).

**I. `tt/runners/kv_chunk_table.py` / migration.** The engine builds `KvCacheStage(base_addr,
first_layer_idx, num_my_layers)` generically (`prefill_runner.py:625`), so
`kv_migration_base_address` needs no change. `build_kv_chunk_table` may need `first_layer_idx`
awareness. **Disable migration for the perf bringup** (`PREFILL_MIGRATION_TABLE_PATH` unset) and
revisit it only once the throughput number exists.

**J. `utils/partial_weights.py` — add `load_layer_range_state(model_path, first, count)`.**
Needed only for the cold TP=1 cache build (§4). Each rank currently would need
`GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1`, which reads the whole 62 GB checkpoint into host RAM — ×4
concurrent ranks = ~248 GB against 477 GB available. A range loader drops it to ~15 GB/rank.
`load_layer_state` (single layer) already exists; generalising it is small.

---

## 4. The TP=1 weight cache

The existing cache is `tensor_cache_bf16_mesh8x4` (37 GB) with filenames tagged `_tp4_`
(e.g. `layer_0/mlp/down_proj.weight_tp4_bfp8_dtype_BFLOAT8_B_layout_TILE.tensorbin`). A `[8,1]`
rank resolves `tensor_cache_bf16_mesh8x1` with `_tp1_` names — **none of it exists**. ~37 GB more;
`/data` has 23 TB free.

**It cannot be pre-built in one process.** A single `[8,1]` mesh holding all 60 layers at TP=1 is
~31 GB of weights per chip before any KV — it will not fit. The cache must be built **per rank,
each rank writing only its own layer slice** into the shared directory. The four ranks write
disjoint `layer_N/` subtrees, so the concurrency is safe; only the completion marker would race,
and `create_tt_model` skips `mark_weight_cache_complete` whenever `num_layers is not None` (which
PP always sets), so no marker is written at all. That is fine — the runtime always supplies its
own state dict via `_cache_completion_state`.

Procedure:
1. First PP run with `GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1` (or the §3.2-J range loader).
   Expect **~15–30 min** and heavy NFS + host RAM. Watch `free -g`.
2. Verify afterwards:
   ```bash
   ls /data/kmabee/hf_cache/tt_cache/google--gemma-4-31B-it/          # expect tensor_cache_bf16_mesh8x1
   ls .../tensor_cache_bf16_mesh8x1 | wc -l                            # expect 60 layer_N dirs
   ls .../tensor_cache_bf16_mesh8x1/layer_0/mlp/                       # expect _tp1_ in the names
   grep -c "reusing legacy" <log>                                      # MUST be 0
   ```
3. Subsequent runs load in ~40–60 s.

> **This is the #1 time sink on this branch.** A mesh-mismatched cache does not error — it
> **deadlocks in layer 0** with no diagnostic. `mesh_cache_hang_debug.md` is the writeup. Before
> debugging any hang, check cache provenance first. Set `GEMMA4_WEIGHT_CACHE_MESH_ONLY=1` to make
> the legacy fallback refuse rather than silently succeed.

---

## 5. Phasing — each phase has a gate that must pass before the next

### Phase 0 — host-only, no device (hours)
Extend `tests/unit/test_prefill_adapter.py` and `tests/test_common_prefill_runtime.py` with
non-zero `first_layer_idx` / `is_first_rank=False` / `is_last_rank=False` /
`mesh_shape=(8,1)` `PrefillRunParams`. Assert the adapter accepts them, resolves the `mesh8x1`
cache path, and slices `layer_types` correctly.
**Gate:** `pytest models/demos/gemma4/tests/unit/test_prefill_adapter.py -q` green, no device.

### Phase 1 — single process, `[8,1]`, TP=1, full 60 layers at small context
Prove TP=1 works at all before adding PP. Open one `[8,1]` column (pin `TT_VISIBLE_DEVICES` to a
column from the generated binding), build **all 60 layers**, run a short context that fits
(e.g. `max_seq_len=16384`, chunk 8192 — the weights will be ~31 GB/chip so this may itself need a
reduced `PREFILL_NUM_LAYERS`; if so run 15 layers with `first_layer_idx=0`).
**Gate:** a chunk completes, output is finite, and the log shows `Fabric config: FABRIC_2D_TORUS_Y`
and no TP collectives dispatched. This is where a TP=1 L1 / matmul-program-config wall would show
up (§6, risk 1).

### Phase 2 — single process, `[8,1]`, TP=1, layer slice `[15, 30)`
Same mesh, but `first_layer_idx=15, num_layers=15, is_first_rank=False, is_last_rank=False`, fed a
placeholder activation. This validates the **entire global-index threading** (§3.2-D) with no
sockets, no MPI, no ttrun — the cheapest possible place to catch a `layer_types` off-by-15.
**Gate:** stage 1 (`first_layer_idx=15`) reports 3 global + 10 sliding layers in its log, and the
`ring_joint` shapes for those layers match a global layer (head_dim 512, packed 640), not a
sliding one. Cross-check against Phase 1's layer-0-based run.

### Phase 3 — 4-rank PP through ttrun, short context
```bash
PREFILL_MANIFEST=models/demos/gemma4/tt/runners/manifests/gemma4_31b_pp4.json \
bash models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/common/prefill/runners/topology_configuration/gemma4_pipeline_prefill_4rank_8x1_torus_y.$(hostname).yaml \
  "$(hostname):4"
```
plus the producer in a second terminal (`how_to_run.md` §4, with `PREFILL_TP=1`). Start at
`PREFILL_MAX_SEQ_LEN=32768`, `PREFILL_PRODUCER_CHUNKS=4`.
**Gate:** all four ranks log `CHUNK_COMPUTE`, the `SEND-d2d` / `RECV-d2d` pairs line up, and the
run drains through the SHUTDOWN sentinel without a hang.

### Phase 4 — 256k, then the matrix
Full `PREFILL_MAX_SEQ_LEN=262144`, 32 chunks. Then sweep `PREFILL_PP_LAYER_COUNTS`
(`15,15,15,15` / `16,14,16,14` / `17,13,17,13` / `18,12,18,12`) and read stage balance off
`PREFILL_TIMING_DIR`'s per-rank CSVs.
**Gate:** a device-time / tok/s number comparable to the 13.6 s baseline, from
`analyze_pp.py` — see §7.

---

## 6. Risks, in order, with the fallback for each

1. **TP=1 shapes / L1.** With TP=1 a device carries 32 Q heads instead of 8 and the full
   5376×21504 MLP. The head *ratios* are unchanged (sliding 2Q:1KV, global 8Q:1KV at both TP=4 and
   TP=1), so `ring_joint`'s structural constraints should hold, but SDPA's working set and the
   DRAM-sharded matmul program configs (`tt/dram_sharded.py`) are sized for TP=4 widths.
   *Fallback:* PP=2 × `[8,2]`. Halves rather than eliminates TP CCL (~1.2× instead of ~1.44×), but
   keeps every head count divisible and balances globals 5/5 exactly. Cost: one new mesh-graph
   descriptor (copy the 4x8x1 z_chain torus_y, change to two `[8,2]` meshes and one connection).
2. **Wrong `TT_VISIBLE_DEVICES` map.** Does not error; produces plausible wrong numbers.
   *Mitigation:* regenerate with `gen_pp4_binding.py` on this host, and cross-check against
   `probe_columns.py`'s raw output. Never hand-edit.
3. **Weight-cache mesh mismatch → silent layer-0 deadlock.** *Mitigation:* §4, plus
   `GEMMA4_WEIGHT_CACHE_MESH_ONLY=1` and the `grep -c "reusing legacy"` check on every log.
4. **`layer_scalar` off-by-`first_layer_idx`.** `Gemma4DecoderLayer` reads `layer_scalar` as a
   Python float and **silently defaults to 1.0** when the key is missing (`tt/layer.py:126-129`).
   The real 31B values are 0.089, 0.065, 0.992… A local-index lookup on rank 1 would take layer
   15's scalar for layer 0 — wrong numerics, no error, no PCC test on this branch to catch it
   (the CPU reference was deleted; the surviving e2e test asserts liveness and finiteness only).
   *Mitigation:* assert in `Gemma4DecoderLayer.__init__` that `layer_scalar` was **found**, not
   defaulted, whenever a non-empty state dict was supplied. Add this even though it is scope creep
   — it is the only guard against the most likely silent failure in this whole change.
5. **Torus mode vs. D2D routing.** All `FABRIC_2D_TORUS_*` are 2D fabric and should route the D2D
   MeshSocket, but this is unverified for Gemma4. *Fallback:* the plain `z_chain` (non-torus)
   descriptor with `PREFILL_FABRIC_MODE=2d` — at the cost that the SP-axis ring attention runs
   Linear, making the number not directly comparable to the 8×4 baseline. Say so if you fall back.
6. **`test_prefill_layer_perf_chunk_n` is already red at HEAD** (`prefill.py:452`, packed-RoPE
   built in `Gemma4Model.__call__` but the test hand-mirrors the layer loop). Unrelated to this
   work; do not spend time on it, and do not use it as a gate.

---

## 7. Measuring it — the four corrections that are not optional

Port the analyzers from `/data/kmabee/tt-metal-3/models/demos/deepseek_v3_d_p/tests/perf/pp4/`.
Each exists because the obvious reading of that measurement is wrong:

* **`analyze_pp.py`** — throughput comes from the **last rank's chunk-to-chunk interval**, not the
  producer's tok/s (which counts H2D pushes and so includes pipeline fill and the first chunk's
  compile).
* **`analyze_layer_budget.py`** — one row per `(op, DEVICE)`; a stage's 8 chips run
  **concurrently**, so an op's cost is the **MAX across devices, not the sum**. And
  `InboundSocketServiceSyncOperation` is ~99 % of a PP stage's device time and is **pure idle**
  (the receiver blocking on upstream) — report it separately, never fold it into compute.
* **`analyze_kv_ramp.py`** — pass `layers_per_stage=15`, not the default 1. Program order is
  chunk-outer/layer-inner, so a 15-layer stage yields 15 instances per chunk; left at 1 it
  conflates layer index with KV depth.
* **Warm-up is not cosmetic.** A multi-chunk interval **grows** as the KV cache deepens. Every
  published Mistral table discards **8** intervals. Latency cells must be re-run **warm** as a
  separate phase — cold pp4@25,600 read 6.5 s against 1.17 s warm. And `analyze_ttft.py`'s
  `E2E_CLOCK last_compute_end` is stamped at DRAIN and can absorb a multi-second shutdown stall
  (it read 9.122 s against a reconstructed 4.102 s on one 20-chunk request) — cross-check, never
  trust it alone.

Never sum the four stage reports: the sum is roughly one chunk's **latency** through the pipeline;
**throughput is `1 / max(stage)`**.

Report the comparison as: 8×4 single-rank 256k chunk8192 **13.6 s / 19271 tok/s** vs. PP=4 [8,1]
same context, same chunk, same producer.

---

## 8. Run hygiene on this galaxy (learned the hard way, don't relearn)

* **Never run a device job in a foreground tool call with a timeout.** A killed wrapper is a
  SIGKILL mid-fabric and the next mesh open dies with *"Timed out while waiting for active ethernet
  core N-N to become active again"*. Use `setsid nohup … &` and poll the log.
* **`tt-smi -r` after every hard kill.** Then confirm the chips are free:
  `for d in /dev/tenstorrent/*; do fuser "$d"; done` (empty = free).
* **`pgrep -f` / `pkill -f` match the Claude Code wrapper shell itself** (its command line contains
  the whole script text). Act on a PID, never a pattern.
* **`ttrun` forwards only what is in its `-x` list.** `TT_*` / `ARCH_*` / `TTNN_*` are automatic;
  `PREFILL_*` and `GEMMA4_*` are **not**. An unforwarded knob silently keeps its default while the
  driver reports the value you asked for — which reads as a clean "no effect" result. That happened
  four times during the Mistral work. Put every knob in the YAML's `global_env`.
* **`TT_METAL_CACHE` is per-user as well as per-host.** A fixed `/tmp` path belongs to whoever ran
  first; everyone after gets `EACCES`, from a path that appears in no env file.
* Keep run output **per-host** (`gemma4_pp4_$(hostname)`) so two galaxies never overwrite each
  other.

---

## 9. Scope note

The Mistral commits split cleanly into *durable* (topology config + analyzers) and *working
material* (shell drivers, hardcoded to one site's layout). Mirror that split here:

* **Meant to survive review:** §3.2 model changes (D–I), the mesh-graph descriptors, the YAML
  templates + manifest, the Phase-0/1/2 tests, and the `layer_scalar` assertion.
* **Working material, not for merge:** the generated `<stem>.<hostname>.yaml`, the campaign shell
  drivers, and any Tracy captures.

The model-side changes (D–I) are worth landing on their own merit even if PP turns out not to win:
they are what makes the Gemma4 prefill runtime a *general* pipeline-capable adapter instead of one
hardcoded to a single (8,4) rank, which is the shape every other model in
`models/demos/common/prefill/` already has.
