<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 03 — Package outline

**Phase:** P3 · **Date (UTC):** 2026-09-03 · **Gate:** `G-OUTLINE` — **PASS**

Every file in the planned tree gets four things: (i) a one-sentence responsibility, (ii) the public
interface signature, (iii) the input/output tensor shapes with dtype and layout, (iv) the template it
mirrors as `path:line`. **Every `path:line` in this document is machine-verified** —
`scripts/verify_citations.py` re-reads each cited file and asserts the claimed symbol is on the
claimed line (`raw/G-OUTLINE_20260903T170527Z.log`).

Inputs, not re-litigated: `DEC-001` (model = Llama-3.1-8B-Instruct), `DEC-002` (`mesh_shape = (4,8)`,
TP=8, SP=4), `DEC-003` (no `reference/`), `DEC-004` (oracle = `transformers` directly), `DEC-005`
(bundled config verbatim), `DEC-006` (copy `MeshConfig`/`CCLManager`/`utils`), `DEC-007` (Meta RoPE +
reuse `tt_transformers` helpers), `DEC-008` (import the HF→Meta key mapping).
New in this phase: `DEC-009` … `DEC-017`.

---

## 1. Conventions every file in `tt/` obeys

| # | Convention | Source |
|---|---|---|
| 1 | **Module signature:** `X(mesh_device, hf_config, state_dict, *, mesh_config, ccl_manager=None, tensor_cache_path=None, weight_dtype=ttnn.bfloat8_b)`. Everything after `state_dict` is **keyword-only**. Forward is `__call__`, except `RMSNorm` (an `nn.Module`) which uses `forward`. | `BRINGUP_RECIPE.md:459-462`; template shape `models/demos/minimax_m3/tt/dense_mlp.py:29` (M3 allows these positionally; keyword-only is the recipe's stricter form and is deliberate) |
| 2 | **`hf_config` is a normalised object**, never a raw dict and never a raw `PretrainedConfig`. Exactly one constructor, `tt/model_config.py::llama_hf_config()`, turns either into `LlamaHFConfig`. Modules do `hf_config.hidden_size`; **no module ever calls `getattr(hf_config, ..., default)`**. | `DEC-009`. Attribute style: `models/demos/minimax_m3/tt/dense_mlp.py:47`, `models/demos/gpt_oss_d_p/tt/model.py:62`. Engine boundary is an object: `models/demos/common/prefill/adapter.py:143` |
| 3 | **`rope_theta` / `rope_scaling` are read in exactly one place** — inside `llama_hf_config()`, through `models/tt_transformers/tt/common.py:165` `get_rope_theta` and `:183` `get_rope_scaling`, both of which take a **dict**. | `DEC-010` |
| 4 | **State-dict splitting is the caller's job**, via `substate(state_dict, "mlp")`. Modules receive an already-stripped sub-dict. | `models/demos/gpt_oss_d_p/utils/substate.py:15`; caller pattern `models/demos/gpt_oss_d_p/tt/layer.py:68` |
| 5 | **Weight loading goes through `ttnn.as_tensor(..., cache_file_name=get_cache_file_name(path, name))`**, and every module must build from an **empty `state_dict`** when a cache path exists ("cache-only mode"): pass `None` for the torch tensor and let `as_tensor` read the tilized file. | `models/demos/minimax_m3/tt/dense_mlp.py:58` (closure), `:62` (the `weight is None and not tensor_cache_path` branch), `:70` (`cache_file_name`); `models/demos/gpt_oss_d_p/utils/general_utils.py:11` |
| 6 | **HF `[out, in]` → ttnn `[in, out]` at load time**: `w.transpose(-1,-2).unsqueeze(0).unsqueeze(0)`. Never at runtime. | `models/demos/minimax_m3/tt/dense_mlp.py:77` |
| 7 | **Deallocate eagerly** — `t.deallocate(True)` after last use; free the big input before allocating the big output. | `models/demos/minimax_m3/config.py:112` (and the comment at `:104-111`) |
| 8 | **Collectives only via `self.mesh_config.<collective>(t, self.ccl_manager, ...)`** — never raw `ttnn.experimental.*` inside a module. | `BRINGUP_RECIPE.md:525-529`; see `04_CCL_PLAN.md` §5 for the one exception |
| 9 | **Docstring anchors:** every module names its HF anchor (`transformers.models.llama.modeling_llama.LlamaMLP`) and the `path:line` template it mirrors. | `BRINGUP_RECIPE.md:472-474` |
| 10 | **No env-var magic** beyond the `README.md` table. This package plans exactly two: `LLAMA31_8B_DELTA_PROBE` (P6.1) and `LLAMA31_8B_WEIGHTS_FROM_CACHE` (P7 harness). | `BRINGUP_RECIPE.md:475`; probe template `models/demos/gpt_oss_d_p/tt/layer.py:19` |
| 11 | **Activation dtype is `bfloat16`**; `bfloat8_b` only above `seq_len > 32*1024`. Weight dtype default `bfloat8_b` (Appendix E measured MLP 0.9995823 at bf8_b), except RMSNorm gains which are always bf16. | `models/demos/gpt_oss_d_p/tt/attention/prefill.py:106-109`; norm dtype `models/demos/gpt_oss_d_p/tt/rms_norm.py:37` |
| 12 | **Assert, do not branch, on features Llama lacks.** No bias branch (`attention_bias: false`, `mlp_bias: false`), no sinks, no sliding window, no QK-norm, no MoE, no partial rotary. | `00_MODEL_CARD.md` §3; `02_SURVEY.md` §3 |

### 1.1 The two settled design questions

**`hf_config` — object, built by one normaliser (`DEC-009`).** Modules take an object so every line
copied from `minimax_m3` / `gpt_oss_d_p` keeps working unedited, and because the engine's own
boundary is an object: `models/demos/common/prefill/adapter.py:143` declares
`def load_hf_config(self) -> "PretrainedConfig"` and the gpt-oss adapter returns an `AutoConfig`
(`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:63`). But the *bundled-config* path already
returns a dict (`models/demos/llama31_8b_d_p/tests/test_factory.py:49` `llama_config_dims()`), and `get_rope_theta` needs a dict.
So the boundary is a single function:

```python
# tt/model_config.py
@dataclass(frozen=True)
class LlamaHFConfig:
    hidden_size: int; intermediate_size: int; num_hidden_layers: int
    num_attention_heads: int; num_key_value_heads: int; head_dim: int
    vocab_size: int; max_position_embeddings: int; rms_norm_eps: float
    tie_word_embeddings: bool
    rope_theta: float                 # resolved, never None
    rope_scaling_factor: float        # rope_scaling["factor"]
    rope_orig_context_len: int        # rope_scaling["original_max_position_embeddings"]
    @property
    def gqa_group_size(self) -> int: return self.num_attention_heads // self.num_key_value_heads

def llama_hf_config(source) -> LlamaHFConfig: ...
```

`source` is a `dict` (bundled JSON, or `cfg.to_dict()`) or anything with `to_dict()`. The function
converts to a plain dict **first**, derives `head_dim` when absent
(`python_env/lib/python3.12/site-packages/transformers/models/llama/configuration_llama.py:87-88` does the same), and asserts every field is not `None`. Mechanical
rule for P5: *if a module needs a model dimension, it is a field on `LlamaHFConfig`; if it is not
there, add it there — do not reach past the object.*

**Where the RoPE parameters are read (`DEC-010`).** Only inside `llama_hf_config()`, via
`get_rope_theta(cfg_dict)` (`models/tt_transformers/tt/common.py:165`) and
`get_rope_scaling(cfg_dict)` (`:183`). Measured on this machine (transformers 5.12.1,
`raw/G-OUTLINE_20260903T170527Z.log`), and this **corrects `R-002`**:

| expression | measured result |
|---|---|
| `LlamaConfig.from_pretrained(...).rope_theta` | **raises `AttributeError: 'LlamaConfig' object has no attribute 'rope_theta'`** — the attribute does **not** exist (`R-002` says it "EXISTS and is None"; that is wrong) |
| `getattr(cfg, "rope_theta", 500000.0)` | `500000.0` — i.e. it returns the **default**, not `None` |
| `cfg.rope_scaling` | the full dict, **including** `rope_theta: 500000.0` — not `None` |
| `cfg.rope_parameters` | same dict (`rope_scaling` is an alias of it) |
| `cfg.to_dict()` | has **no** `rope_theta` and **no** `rope_scaling` key — only `rope_parameters` |
| `get_rope_theta(cfg.to_dict())` | `500000.0` ✓ |
| `get_rope_theta(raw_bundled_json)` | `500000.0` ✓ (top-level, transformers-4.42.3 layout) |

The danger is therefore *not* a silent `None` (that would raise loudly on attribute access, or return
the default under `getattr`). It is the opposite: `getattr(cfg, "rope_theta", DEFAULT)` **succeeds**
and silently substitutes a hard-coded default for the checkpoint's actual θ. For Llama-3.1-8B the
common default (10000.0) is **not** 500000.0, so that pattern would produce a wrong RoPE at every
position with no error. `get_rope_theta` is mandatory for that reason, and `llama_hf_config()`
asserts the result is not `None`.

---

## 2. The planned tree

`E` = already exists (P0/P1). Phase = when the file is created. Files marked `—` in the Gate column
are exercised by another file's gate.

```
models/demos/llama31_8b_d_p/
├── README.md                                   P9    G-CLEAN
├── BRINGUP_RECIPE.md                           E
├── __init__.py                                 E
├── conftest.py                                 E     (P1; session `state_dict` fixture + --skip-model-load)
├── configs/Llama-3.1-8B-Instruct/config.json   E     (P1; DEC-005 byte-identity)
├── bringup_log/…                               E
├── tt/
│   ├── __init__.py                             E
│   ├── config.py                               P5.1  G-MESH
│   ├── ccl.py                                  P5.1  G-MESH, G-SEMAPHORE
│   ├── model_config.py                         P5.1 + P6.2   (split — DEC-014)  G-WEIGHTS
│   ├── rms_norm.py                             P5.2  G-RMS
│   ├── rope.py                                 P5.3  G-ROPE
│   ├── mlp.py                                  P5.4  G-MLP
│   ├── attention/
│   │   ├── __init__.py                         P5.5  G-ATTN
│   │   ├── config.py                           P5.5  —
│   │   ├── weights.py                          P5.5  —
│   │   ├── operations.py                       P5.5  —
│   │   ├── prefill.py                          P5.5  G-ATTN
│   │   ├── kv_cache.py                         P5.6  G-KV
│   │   └── dense_sp.py                         P5.5 (stub) → P8 (real)  G-MESH-KV
│   ├── embedding.py                            P6.2  —
│   ├── lm_head.py                              P6.2  G-MODEL (top-1)
│   ├── layer.py                                P6.1  G-LAYER
│   ├── model.py                                P6.3  G-MODEL
│   ├── tt_prefill_runtime.py                   P7    G-CHUNK
│   └── runners/
│       ├── __init__.py                         P10   —
│       ├── kv_chunk_table.py                   P10   G-MOCK-MIG
│       ├── adapters/__init__.py                P10   —
│       ├── adapters/llama.py                   P10   G-ADAPTER
│       └── manifests/llama31_8b_d_p.json       P10   G-REQUEST
├── utils/
│   ├── __init__.py                             P5.1  —      (recipe's tree lists `utils/` but P0 step 1's mkdir omits it)
│   ├── general_utils.py                        P5.1  —
│   └── substate.py                             P5.1  —
├── scripts/
│   ├── __init__.py                             E
│   ├── verify_citations.py                     E     (extended every phase)
│   ├── generate_golden_kv_cache.py             P7    G-GOLDEN
│   └── verify_golden_kv.py                     P7    G-GOLDEN
└── tests/
    ├── __init__.py                             E
    ├── test_factory.py                         E     (P1)
    ├── unit/
    │   ├── __init__.py                          E
    │   ├── test_reference_model.py              E     G-REF
    │   ├── test_mesh_config.py                  P5.1  G-MESH        ← added, DEC-016
    │   ├── test_ccl_semaphores.py               P5.1  G-MESH (part) + G-SEMAPHORE @P8  ← added, DEC-016
    │   ├── test_rms_norm_vs_ref.py              P5.2  G-RMS
    │   ├── test_rope_vs_ref.py                  P5.3  G-ROPE
    │   ├── test_mlp_vs_ref.py                   P5.4  G-MLP
    │   ├── test_attention_vs_ref.py             P5.5  G-ATTN
    │   ├── test_kv_cache_vs_ref.py              P5.6  G-KV
    │   ├── test_decoder_layer_vs_ref.py         P6.1  G-LAYER
    │   ├── test_weight_loading.py               P6.2  G-WEIGHTS     ← added, DEC-016
    │   ├── test_model_vs_ref.py                 P6.3  G-MODEL
    │   ├── test_attention_chunked_vs_ref.py     P7    G-CHUNK
    │   └── test_tp_parity.py                    P8    G-TP-PARITY   ← added, DEC-016
    └── galaxy_prefill_kv_pcc.py                 P8    G-MESH-KV, G-RACE
```

**Deltas from the recipe's tree (`BRINGUP_RECIPE.md:396-453`), each with its reason:**

| Delta | Reason |
|---|---|
| `reference/` **removed** | `DEC-003` — recipe self-contradiction; Llama is first-class in `transformers` |
| `utils/__init__.py` **added** | `02_SURVEY.md:215-217`; `utils/` must be a package to be importable |
| `tests/unit/test_mesh_config.py`, `test_ccl_semaphores.py`, `test_weight_loading.py`, `test_tp_parity.py` **added** | `DEC-016` — the recipe defines gates `G-MESH`, `G-SEMAPHORE`, `G-WEIGHTS`, `G-TP-PARITY` but its tree contains **no file that could host them**. A gate with no test file cannot be run. |
| `tt/model_config.py` created in **P5.1**, not P6.2 | `DEC-014` — `llama_hf_config()` is a prerequisite of every P5 module (`DEC-009`). The `ModelArgs` half stays in P6.2. |
| `tests/unit/__init__.py` | already exists (P1) |

---

## 3. Per-file contracts

Notation: `S` = global sequence length; `S_loc = S/SP = S/4`; `TP = 8`; `H = 4096`; `I = 14336`;
`n_q = 32`; `n_kv = 8`; `hd = 128`; `V = 128256`; `L = 32`. Per chip: `n_q_loc = 4`, `n_kv_loc = 1`,
`H/TP = 512`, `I/TP = 1792`. All device tensors are 4-D `[b, x, s, f]` unless stated.

### 3.1 `tt/config.py` — `MeshConfig`

- **Responsibility.** Owns the parallelism decision (TP is the only knob; SP derived) and the three
  collective wrappers; nothing model-specific.
- **Interface.**
  ```python
  _VALIDATED_MESH_SHAPE = (4, 8); _VALIDATED_TP = 8            # DEC-002
  class MeshConfig:
      def __init__(self, mesh_shape, tp, tp_axis: int = 1)
      @property
      def sp(self) -> int
      def shard_mapper(self, mesh_device, tensor_dim=None, mesh_dims=None) -> ttnn.ShardTensor2dMesh
      def column_parallel(self, mesh_device)      # shard dim -1
      def row_parallel(self, mesh_device)         # shard dim -2
      def sequence_parallel(self, mesh_device)    # shard dim -3
      def shard_size(self, total_size) -> int
      def allreduce(self, tensor, ccl_manager, memory_config=None, pad_size=None, axis=0)
      def allgather(self, tensor, ccl_manager, memory_config=None, axis=0, dim=3, linear=False)
      def reduce_scatter(self, tensor, ccl_manager, dim=3, axis=0, memory_config=None)
  ```
- **Shapes.** Pure host object; no tensors. `shard_size(4096) == 512`, `shard_size(14336) == 1792`.
- **Template.** The **union** of `models/demos/minimax_m3/config.py:21` and
  `models/demos/gpt_oss_d_p/tt/config.py:19` — field-by-field table in `04_CCL_PLAN.md` §3
  (`DEC-019`). `reduce_scatter` from `models/demos/minimax_m3/config.py:155`; `sp` property from
  `models/demos/gpt_oss_d_p/tt/config.py:55`; strict `_validate` from `:44`.

### 3.2 `tt/ccl.py` — `CCLManager`

- **Responsibility.** Owns every *persistent* CCL resource — sub-device, ping-pong semaphore sets,
  barrier semaphores, the ring-attention semaphore pair, and the ring-gather scratch buffers —
  allocated **once per model**.
- **Interface.**
  ```python
  class CCLManager:
      def __init__(self, mesh_device, num_links, topology=ttnn.Topology.Ring)
      def get_rs_ping_pong_semaphore(self)   # 3 handles, cycles idx 0/1
      def get_ag_ping_pong_semaphore(self)   # 2 handles, cycles idx 0/1
      def get_barrier_semaphore(self)        # 1 handle,  cycles idx 0/1
      def get_ring_gather_buffer(self, key, n_kv, seq, head_dim, dtype)
      def reset_global_semaphores(self)
      # attributes: mesh_device, num_links, topology, compute_grid_size, ccl_cores,
      #             ccl_sub_device_id, ring_attention_ccl_core_grid_offset,
      #             ring_attention_ccl_semaphore_handles
  ```
- **Shapes.** Semaphore list lengths **6 / 4 / 2 / 2** (rs / ag / barrier / ring-attention) — the four
  constants `G-SEMAPHORE` asserts. `get_ring_gather_buffer` allocates
  `[1, n_kv, seq, head_dim]`, mapped `dims=[None, 1]` (heads on TP cols, seq replicated on SP rows).
- **Template.** `models/demos/gpt_oss_d_p/tt/ccl.py:17` — copied essentially verbatim (it is fully
  model-agnostic): `_init_subdevice` `:40`, grid from `compute_with_storage_grid_size()` `:44`,
  ring-attention offset `(grid.x-1, 0)` `:61`, `_init_semaphores` `:63`, counts `:65`/`:71`/`:77`/`:84`,
  getters `:88`/`:95`/`:102`/`:108`, `reset_global_semaphores` `:129`. **Measured on this machine:**
  `compute_with_storage_grid_size() == (12, 10)` on Blackhole, so the offset is `(11, 0)`.
- **Nothing to delete.** Unlike the other copies, this file carries no MoE baggage.

### 3.3 `tt/model_config.py` — `LlamaHFConfig` (P5.1) + `ModelArgs` (P6.2)

- **Responsibility.** The single normalisation point for model dimensions (P5.1), and real-checkpoint
  state-dict loading + weight-cache pathing (P6.2).
- **Interface.**
  ```python
  # --- P5.1 ---
  @dataclass(frozen=True)
  class LlamaHFConfig: ...                                  # §1.1
  def llama_hf_config(source) -> LlamaHFConfig              # dict | PretrainedConfig -> object
  # --- P6.2 ---
  class ModelArgs:
      def __init__(self, mesh_device, *, weights_path=None, hf_config=None)
      @staticmethod
      def load_state_dict(weights_path, convert_to_meta_format=True) -> dict
      def weight_cache_path(self, dtype) -> Path
      @staticmethod
      def get_state_dict_prefix(module_name, layer_idx) -> str
  ```
- **Shapes.** Host only. `load_state_dict` returns **291 tensors** (`9·32 + 3`, `00_MODEL_CARD.md`
  §2.1), keys converted HF→Meta via `models/tt_transformers/tt/load_checkpoints.py:193`
  `convert_hf_to_meta` (which internally calls `:451` `convert_hf_qkv_to_meta_format`, the Q/K
  `reverse_permute` at `:891` — required by `DEC-007`'s Meta RoPE).
- **Template.** `models/demos/minimax_m3/tt/model_config.py:22` (class), `:126` `load_state_dict`,
  `:212` `weight_cache_path`, `:235` `get_state_dict_prefix`. **Do not** subclass
  `models/tt_transformers/tt/model_config.py:539` — it raises without `HF_MODEL` (`:702`), `R-005`.
  **Do not** copy M3's `convert_to_meta_format` path: it calls M3's *partial*-RoPE QKV converter
  (`models/demos/minimax_m3/tt/model_config.py:19`) and Llama is full-rotary (`DEC-008`).
- **Cache path layout.** Mirror the adapter's, so P5–P8 and P10 share one cache:
  `$LLAMA31_8B_TTNN_CACHE/llama31_8b_d_p_bh_32dev/4x8` — shape from
  `models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:75`.

### 3.4 `tt/rms_norm.py` — `RMSNorm`

- **Responsibility.** Plain RMSNorm, `out = rms_norm(x) * weight`, with the distributed 3-op branch
  present but dormant (`is_distributed = False`).
- **Interface.**
  ```python
  class RMSNorm(nn.Module):
      def __init__(self, mesh_device, hf_config, state_dict, *, mesh_config,
                   tensor_cache_path=None, is_distributed=False)
      def forward(self, x) -> ttnn.Tensor
  ```
- **Shapes.**
  | tensor | shape (per chip) | dtype | layout |
  |---|---|---|---|
  | weight (torch, HF) | `[4096]` | fp32/bf16 | — |
  | weight (device) | `[1, 1, 128, 32]` (`reshape(1,1,-1,TILE_SIZE)`, `TILE_SIZE = 32` measured) | bf16 | **ROW_MAJOR**, replicated |
  | `x` in | `[1, 1, S_loc, 4096]` | bf16 | TILE |
  | out | `[1, 1, S_loc, 4096]` | bf16 | TILE |
  | (scheme B only) stats after AG | `[1, 1, 32, 32·TP] = [1,1,32,256]` | bf16 | width-sharded 1×1 |
- **Template.** `models/demos/gpt_oss_d_p/tt/rms_norm.py:17` (class), `:49` (`forward`), `:27`
  (weight reshape), `:34` (`as_tensor` ROW_MAJOR + `cache_file_name`), `:94` (plain `ttnn.rms_norm`).
- **Deletions.** The `use_gemma_norm` `+1` fold (`:22`, `:25-26`) — Llama's norm is plain
  (`00_MODEL_CARD.md` §2). `eps` comes from `hf_config.rms_norm_eps` = `1e-05`.
- **Change vs template.** `is_distributed` becomes an explicit constructor argument instead of the
  pinned literal at `:33`, so scheme B is a caller decision rather than an edit. It stays `False`
  this iteration (`DEC-018`).

### 3.5 `tt/rope.py` — llama3-scaled RoPE

- **Responsibility.** Build the Meta-interleaved cos/sin tables (per-chunk for P5–P6, whole-cache
  block-cyclic SP-sharded for P7–P8) and the replicated `[1,1,32,32]` transformation matrix.
  **Assembly of imported helpers, no new math** (`DEC-007`).
- **Interface.**
  ```python
  def build_transformation_mat(mesh_device, dtype=ttnn.bfloat16) -> ttnn.Tensor
  def build_prefill_rope(mesh_device, hf_config, *, seq_len, start_pos=0) -> list[ttnn.Tensor]
  def build_indexed_rope(mesh_device, hf_config, *, max_seq_len, chunk_size,
                         sp_axis=0, dtype=ttnn.bfloat16) -> list[ttnn.Tensor]
  ```
- **Shapes.**
  | tensor | shape (per chip) | dtype | layout |
  |---|---|---|---|
  | transformation mat | `[1, 1, 32, 32]` | bf16 | TILE, **replicated** |
  | `build_prefill_rope` cos/sin | `[1, 1, S, 128]` | bf16 | TILE, **replicated** (P5/P6 single card: `S_loc == S`) |
  | `build_indexed_rope` cos/sin | `[1, 1, max_seq_len/4, 128]` | bf16 | TILE, seq SP-sharded on rows (`dims[sp_axis]=2`), replicated on TP cols |
- **Imports (not rewrites).** `models/tt_transformers/tt/common.py:489` `precompute_freqs(dim, end,
  theta, scale_factor, orig_context_len, rope_type="llama3")`, `:437` `apply_scaling`, `:405`
  `compute_llama3_parameters`, `:525` `gather_cos_sin` (this is what produces the Meta interleaving —
  `torch.stack([cos, cos], -1).flatten(-2)`), `:534` `get_prefill_rot_mat`, `:562`
  `get_rot_transformation_mat` — **called with no argument** (`:564` hard-codes `dhead = 32`,
  `R-010`).
- **Template.** `models/demos/gpt_oss_d_p/tt/rope.py:103` (`build_transformation_mat`, copy verbatim)
  and `:115` (`build_indexed_rope`) — same block-cyclic + SP-shard structure, with gpt-oss's YaRN
  builder (`:75` `build_yarn_cos_sin`) replaced by `precompute_freqs`/`gather_cos_sin`. The two
  constraints at `:146` and `:148` (`chunk_size % (TILE_SIZE*sp) == 0`, `max_seq_len % chunk_size == 0`) are
  kept verbatim; at SP=4 that is `chunk_size % 128 == 0` (`00_MODEL_CARD.md` §4.4).
- **`block_cyclic_reorder` is imported, not copied**, from
  `models/demos/deepseek_v3_d_p/tt/mla/utils` — the precedent is gpt-oss's own
  `models/demos/gpt_oss_d_p/tt/rope.py:25`.
- **Asserts (`R-006`).** Before delegating, assert `rope_scaling["low_freq_factor"] == 1.0` and
  `["high_freq_factor"] == 4.0`, because `models/tt_transformers/tt/common.py:407-408` hard-codes
  them as local constants and would silently ignore anything else. Already implemented at
  `models/demos/llama31_8b_d_p/tests/test_factory.py:100` `rope_scaling()`; `tt/rope.py` calls the same check.

### 3.6 `tt/mlp.py` — dense SwiGLU

- **Responsibility.** `down(silu(gate(x)) * up(x))`; gate/up column-parallel, down row-parallel plus
  the TP collective.
- **Interface.**
  ```python
  class MLP:
      def __init__(self, mesh_device, hf_config, state_dict, *, mesh_config, ccl_manager=None,
                   tensor_cache_path=None, weight_dtype=ttnn.bfloat8_b, scatter_output=False)
      def __call__(self, x) -> ttnn.Tensor
  ```
  `scatter_output=False` ⇒ close with `allreduce` (scheme A). `True` ⇒ `reduce_scatter` only
  (scheme B). The parameter exists **now** so scheme B stays a flag (`DEC-018`).
- **Shapes.**
  | tensor | shape (per chip) | dtype | layout |
  |---|---|---|---|
  | `gate_proj` / `up_proj` weight | `[1, 1, 4096, 1792]` | bf8_b | TILE, col-parallel (dim −1) |
  | `down_proj` weight | `[1, 1, 1792, 4096]` | bf8_b | TILE, row-parallel (dim −2) |
  | `x` in | `[1, 1, S_loc, 4096]` | bf16 | TILE |
  | `gate`, `up` | `[1, 1, S_loc, 1792]` | bf16 | TILE |
  | `act` (fused SiLU-mul) | `[1, 1, S_loc, 1792]` | bf16 | TILE |
  | `out` pre-collective (partial sum) | `[1, 1, S_loc, 4096]` | bf16 | TILE |
  | out, scheme A (post all-reduce) | `[1, 1, S_loc, 4096]` | bf16 | TILE |
  | out, scheme B (post reduce-scatter) | `[1, 1, S_loc, 512]` | bf16 | TILE |
- **Template.** `models/demos/minimax_m3/tt/dense_mlp.py:26` (class), `:29` (`__init__`), `:38`
  (`scatter_output`), `:58` (`_load` closure), `:62-63` (cache-only branch), `:77` (HF transpose),
  `:87` (`__call__`), `:99` (`if tp > 1`), `:105-107` (reduce-scatter), `:112` (all-reduce).
- **Changes.** (a) `:92`'s clamped `swigluoai` becomes the one fused op
  `ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])` — settled in P2, in-tree
  usage `models/common/modules/mlp/mlp_1d.py:262` and the default at `:84`. (b) Drop the `zone(...)`
  profiler wrappers (`02_SURVEY.md` §3). (c) `hidden_size` comes from `hf_config.hidden_size`
  (`DEC-009`).

### 3.7 `tt/attention/config.py` — `AttentionConfig` + `ProgramConfig`

- **Responsibility.** Two frozen-ish dataclasses: the model's attention shape, and the device program
  / compute-kernel configuration.
- **Interface.**
  ```python
  @dataclass
  class AttentionConfig:
      hidden_size: int; num_heads: int; num_kv_heads: int; head_dim: int
      max_seq_len: int
      rms_norm_eps: float = 1e-5
      scaling: float | None = None          # -> head_dim ** -0.5 in __post_init__
      rotary_dim: int | None = None         # -> head_dim (full rotary)
      sequence_parallel: bool = False
      @property
      def gqa_group_size(self) -> int

  @dataclass
  class ProgramConfig:
      prefill_q_chunk_size_small: int = 32;  prefill_k_chunk_size_small: int = 32
      prefill_q_chunk_size_large: int = 256; prefill_k_chunk_size_large: int = 256
      prefill_threshold: int = 2048
      sdpa_core_grid: tuple = (8, 8)        # DEC-012 — explicit, NOT derived from the device grid
      math_fidelity: str = "HiFi4"; math_approx_mode: bool = False
      fp32_dest_acc_en: bool = False; packer_l1_acc: bool = False
      def get_prefill_sdpa_config(self, mesh_device, seq_len) -> ttnn.SDPAProgramConfig
      def get_compute_kernel_config(self, mesh_device)
  ```
- **Shapes.** Host only.
- **Template.** `models/demos/gpt_oss_d_p/tt/attention/config.py:23` (`AttentionConfig`), `:45`
  (`__post_init__`), `:52` (`gqa_group_size`), `:57` (`ProgramConfig`), `:90`
  (`get_prefill_sdpa_config`), `:102` (`get_compute_kernel_config`).
- **Deletions.** `sliding_window` (`:34`), and everything sink-related (`:38-40`'s coupling comment
  stays only as a reason the `scaling` is passed explicitly). Llama has neither
  (`00_MODEL_CARD.md` §3).
- **Two corrections to the template (`DEC-012`, `DEC-013`).**
  1. The SDPA program grid stays **8×8 by default and configurable**, and is *not* derived from
     `compute_with_storage_grid_size()`. Measured: that grid is **12×10** on this Blackhole, and the
     ring-joint SDPA op asserts
     `ccl_core_grid_offset.x >= program_config.compute_with_storage_grid_size.x`
     (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp:421`,
     taken because gpt-oss passes `use_column_major_ccl=True`,
     `models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:134`). With the CCL offset at
     `grid.x - 1 = 11`, an SDPA grid of x=12 **fails the assert**; x=8 passes. So `R-008`'s proposed
     fix would have broken P8. `ProgramConfig` therefore asserts
     `sdpa_core_grid[0] <= grid.x - 1` whenever SP > 1.
  2. `get_compute_kernel_config` takes `mesh_device` and returns
     `ttnn.init_device_compute_kernel_config(mesh_device.arch(), math_fidelity=..., ...)` instead of
     naming a class. There is nothing to branch on: `ttnn.BlackholeComputeKernelConfig` does not
     exist (`hasattr` → **False**; `ttnn/ttnn/__init__.py:305` exports only the Wormhole name), and
     where it *is* defined it is the same object (`ttnn/ttnn/types.py:61`,
     `BlackholeComputeKernelConfig = WormholeComputeKernelConfig`). So `02_SURVEY.md` row 11's
     "pick the class by arch" is a no-op.
     gpt-oss already uses the factory form in its SP path
     (`models/demos/gpt_oss_d_p/tt/attention/prefill.py:201`).

### 3.8 `tt/attention/weights.py` — projection weights

- **Responsibility.** Load, Meta-swizzle, transpose, shard and tilize `q/k/v/o_proj`. No biases, no
  sinks.
- **Interface.**
  ```python
  @dataclass(frozen=True)
  class AttentionWeights:
      wq: ttnn.Tensor; wk: ttnn.Tensor; wv: ttnn.Tensor; o_proj: ttnn.Tensor

  def load_attention_weights(mesh_device, config: AttentionConfig, state_dict, *, mesh_config,
                             weight_dtype=ttnn.bfloat8_b, tensor_cache_path=None) -> AttentionWeights
  ```
- **Shapes.**
  | weight | HF `[out, in]` | ttnn (pre-shard) | per chip @TP=8 | mapper | dtype/layout |
  |---|---|---|---|---|---|
  | `q_proj` | `[4096, 4096]` | `[1, 1, 4096, 4096]` | `[1, 1, 4096, 512]` | `column_parallel` | bf8_b / TILE |
  | `k_proj` | `[1024, 4096]` | `[1, 1, 4096, 1024]` | `[1, 1, 4096, 128]` | `column_parallel` | bf8_b / TILE |
  | `v_proj` | `[1024, 4096]` | `[1, 1, 4096, 1024]` | `[1, 1, 4096, 128]` | `column_parallel` | bf8_b / TILE |
  | `o_proj` | `[4096, 4096]` | `[1, 1, 4096, 4096]` | `[1, 1, 512, 4096]` | `row_parallel` | bf8_b / TILE |
- **Template.** `models/demos/gpt_oss_d_p/tt/attention/weights.py:23` (dataclass), `:38`
  (`load_attention_weights`), `:74-78` (substate reads and the `o_proj` transpose), `:145-146`
  (the two mappers), `:149` (`as_tensor` with `cache_file_name`).
- **Three deletions.** `wqkv_bias` (`:32`), `o_proj_bias` (`:34`), `sinks` (`:35`) — and with them
  the bias-fusion loop (`:107-115`), the sink pre-division (`:119-120`) and the `sinks_tt` load
  (`:192`). Assert the bias keys are **absent** instead of branching (`attention_bias: false`,
  `00_MODEL_CARD.md` §2).
- **Fourth deletion: the `o_proj` tile-alignment padding** (`:64-70`, `:122-128`, and its
  companions `apply_allgather_and_slice`'s slice at `models/demos/gpt_oss_d_p/tt/attention/operations.py:227` and `apply_allreduce`'s at
  `:262`). gpt-oss needs it because `2880/8 = 360` is not tile-aligned. For Llama
  `4096/TP ∈ {4096, 2048, 1024, 512}` is tile-aligned for every admissible TP
  (`00_MODEL_CARD.md` §4.3 constraint 3), so the whole path is dead code. Replace with
  `assert (hidden_size // tp) % ttnn.TILE_SIZE == 0`.
- **Fifth, and the substantive one: three separate weights, not a fused `wqkv` (`DEC-011`).**
  gpt-oss fuses (`:31`) and must therefore pre-interleave per device — `:83-100` builds
  `cat([q_0,k_0,v_0, q_1,k_1,v_1, …])` so that a naive equal split hands each chip its own Q|K|V
  triple. That loop is the most error-prone code in the file and its failure mode is invisible at
  TP=1 (the single-card gates). Three separate column-parallel weights are shard-correct **by
  construction**: 4096/8 = 512 = 4 Q heads, 1024/8 = 128 = 1 KV head, no interleave. It also makes
  the Meta `reverse_permute` a per-tensor call on `q_proj`/`k_proj` only
  (`models/tt_transformers/tt/load_checkpoints.py:451` keys off `"q_proj.weight" in key or
  "k_proj.weight" in key`), which is exactly the granularity the imported helper offers.
  Cost: one extra `ttnn.concat` per layer at runtime (§3.9). Fused QKV is the perf follow-up.

### 3.9 `tt/attention/operations.py` — primitive ops

- **Responsibility.** The small reusable tensor ops: projections, GQA head split/merge, RoPE
  application, and the TP-collective tail.
- **Interface.**
  ```python
  def apply_qkv_projection(hidden_states, weights) -> tuple[q, kv]
  def split_qkv_heads_prefill(q, kv, num_heads, num_kv_heads) -> (Q, K, V)
  def apply_rope(tensor, rope_mats, transformation_mat, is_decode_mode=False,
                 kv_actual_global=None, cluster_axis=None) -> ttnn.Tensor
  def concat_heads(tensor) -> ttnn.Tensor
  def apply_output_projection(tensor, weights, activation_dtype) -> ttnn.Tensor
  def apply_allreduce(tensor, mesh_config, ccl_manager) -> ttnn.Tensor
  def apply_reduce_scatter(tensor, mesh_config, ccl_manager) -> ttnn.Tensor      # scheme B
  ```
- **Shapes.**
  | step | in | out |
  |---|---|---|
  | `apply_qkv_projection` | `[1,1,S_loc,4096]` | `q [1,1,S_loc,512]`, `kv [1,1,S_loc,256]` (`ttnn.concat([k, v], dim=-1)`) |
  | `split_qkv_heads_prefill` | `q`, `kv` above | `Q [1,4,S_loc,128]`, `K [1,1,S_loc,128]`, `V [1,1,S_loc,128]` |
  | `apply_rope` | `[1,n,S_loc,128]` + cos/sin `[1,1,S_loc,128]` + mat `[1,1,32,32]` | same shape as input |
  | `concat_heads` | `[1,4,S_loc,128]` | `[1,1,S_loc,512]` |
  | `apply_output_projection` | `[1,1,S_loc,512]` | `[1,1,S_loc,4096]` (partial sum) |
  | `apply_allreduce` | `[1,1,S_loc,4096]` | `[1,1,S_loc,4096]` (summed) |
  | `apply_reduce_scatter` | `[1,1,S_loc,4096]` | `[1,1,S_loc,512]` |
  All bf16, TILE, except the weights (bf8_b).
- **Template.** `models/demos/gpt_oss_d_p/tt/attention/operations.py:14`
  (`apply_qkv_projection`), `:29` (`split_qkv_heads_prefill`), `:41` (`nlp_create_qkv_heads`), `:50`
  (`apply_rope`), `:87` (`rotary_embedding_llama`), `:79` (the indexed variant), `:92`
  (`concat_heads`), `:102` (`nlp_concat_heads`), `:105` (`apply_output_projection`), `:238`
  (`apply_allreduce`), `:252` (the `allreduce` call with `axis=mesh_config.tp_axis`).
- **Change forced by `DEC-011`.** `split_qkv_heads_prefill` uses the op's **separate-KV form**:
  `ttnn.experimental.nlp_create_qkv_heads(q, kv, num_heads=n_q_loc, num_kv_heads=n_kv_loc,
  transpose_k_heads=False)`. The binding documents it —
  `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads_nanobind.cpp:24`
  ("If optional ``input_kv`` tensor is provided, K and V will be created from ``input_kv`` and
  ``input`` should have shape [B, 1, S, head_dim * num_heads] instead"), argument at `:28`.
  `transpose_k_heads=False` keeps K as `[B, n_kv, S, head_dim]`, which is what SDPA wants (the
  template passes `False` at `:45`).
- **Deletions.** All bias adds (`:25` `bias=weights.wqkv_bias`, `:120` the `o_proj_bias` add), the
  padding slice inside `apply_allreduce` (`:257-269`), and the entire fused matmul+reduce-scatter
  path (`:126` `_FUSED_MM_RS_CONFIGS`, `:131` `is_shape_fused_mm_rs_supported`, `:142`
  `apply_output_projection_fused_rs`, `:214` `apply_allgather_and_slice`) — it is **gated off on
  Blackhole** anyway because the op races there (`:136`, comment `:132-135`), and this is a
  Blackhole-only package.

### 3.10 `tt/attention/prefill.py` — `attention_forward`

- **Responsibility.** The one-shot and cache-backed prefill attention pipeline.
- **Interface.**
  ```python
  def attention_forward(hidden_states, rope_mats, weights, kv_cache, config, mesh_config,
                        mesh_device, program_config, transformation_mat, ccl_manager,
                        user_id=0, batch_size=1, layer_idx=0, cached_len=0,
                        indexed_rope=False, scatter_output=False) -> ttnn.Tensor
  ```
- **Pipeline and shapes** (per chip, TP=8/SP=4, scheme A):
  ```
  hidden_states [1,1,S_loc,4096] bf16 TILE
    -> q [1,1,S_loc,512] , kv [1,1,S_loc,256]                 (3 linears + 1 concat)
    -> Q [1,4,S_loc,128] , K [1,1,S_loc,128] , V [1,1,S_loc,128]
    -> RoPE on Q and K (in place shape)
    -> write_kv_chunk(K_post_rope, V)                          (kv_cache is not None)
    -> SDPA (is_causal=True, scale=config.scaling)  -> [1,4,S_loc,128]
       or dense_sp_attention (SP>1, cache-backed)   -> [1,4,S_loc,128]
    -> concat_heads                                 -> [1,1,S_loc,512]
    -> o_proj                                       -> [1,1,S_loc,4096]  (partial sum)
    -> TP all-reduce                                -> [1,1,S_loc,4096]
  ```
- **Template.** `models/demos/gpt_oss_d_p/tt/attention/prefill.py:51` (`attention_forward`), `:34`
  (`_run_sdpa`), `:40` (`is_causal=True`), `:43` (`scale=config.scaling`), `:116` (qkv proj), `:127`
  (head split), `:143`/`:151` (RoPE on Q/K), `:168` (`write_kv_chunk`), `:184` (the SP branch),
  `:272` (the single-card SDPA), `:280` (`concat_heads`), `:302` (`o_proj`), `:304`
  (`apply_allreduce`).
- **Deletions.** `sliding_window_size=` (`:44`) and `attention_sink=` (`:45`) from the SDPA call —
  Appendix D is explicit. The `batch_size > 1` reshapes (`:120-121`, `:284-285`) stay: the runtime's
  multi-user path needs them.
- **GQA needs no on-chip KV repeat.** `ttnn.transformer.scaled_dot_product_attention` handles the
  group; the only head constraint is
  `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:98`
  `TT_FATAL(nqh >= nkv && nqh % nkv == 0, ...)` (paged variant `:326`). At TP=8: `4 >= 1 &&
  4 % 1 == 0` ✓. At TP=1 (P5): `32 >= 8 && 32 % 8 == 0` ✓.
- **The `cached_len > 0` single-device branch stays a loud `NotImplementedError`**, verbatim in
  spirit from `:266-270`: a plain `is_causal` SDPA is off by `cached_len` and silently wrong.
  Chunked cache-read is the SP ring path (P8) or the paged chunked SDPA — not a single-card path.

### 3.11 `tt/attention/kv_cache.py` — `LlamaKVCache`

- **Responsibility.** Allocate the two packed K/V caches and write one chunk into them. **The KV
  cache is the output of prefill**, so this file's correctness is the point of the whole package.
- **Interface.**
  ```python
  @dataclass
  class LlamaKVCache(KvCaches):
      k: ttnn.Tensor; v: ttnn.Tensor
      num_users: int; num_layers: int; max_seq_len: int; sp: int

  NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32

  def allocate_kv_cache(mesh_device, *, num_layers, max_seq_len, sp_axis=0, num_users=1,
                        head_dim=128, cache_dtype=ttnn.bfloat8_b) -> LlamaKVCache
  def write_kv_chunk(kv_cache, tt_k, tt_v, *, slot_idx, layer_idx, kv_actual, sp_axis) -> None
  ```
- **Shapes.**
  | tensor | shape (per chip) | dtype | layout |
  |---|---|---|---|
  | `k`, `v` (each) | `[num_users·32, 1, max_seq_len/4, 128]` | **bf8_b** (`DEC-017`) | TILE, DRAM `NdShardSpec(shard_shape=[1,1,32,128])` over **8** DRAM banks |
  | chunk written | `[1, 1, S_loc, 128]` | cast to cache dtype | TILE |
  - `num_users·num_layers` = `1·32 = 32` slots at `num_users=1`; `slot = user_id·num_layers + layer_idx`.
  - `max_seq_len` must satisfy `max_seq_len % (TILE_SIZE·sp) == 0` = `% 128 == 0` at SP=4; at
    `max_seq_len = 131072`, `seq_local = 32768` (1024 tiles).
  - Bank count is `mesh_device.dram_grid_size().x`
    (`models/demos/common/prefill/runners/migration.py:338`) — **measured 8** on this Blackhole.
- **Template.** `models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:27`
  (`NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32`), `:31` (dataclass), `:48` (`allocate_kv_cache`), `:77`
  (the tile-alignment assert), `:80` (`seq_local`), `:86-91` (`NdShardSpec`, `shard_shape` at `:87`),
  `:104` (`ReplicateTensorToMesh` — content diverges on first write), `:117` (`_write_one`), `:125`
  (`update_padded_kv_cache`), `:138` (`write_kv_chunk`), asserts `:149`/`:155-159`.
- **Only change: `head_dim` 64 → 128.** Zero gpt-oss baggage; it is already a pure GQA cache
  (`02_SURVEY.md` row 12). **Keep the block geometry exactly** — that is what lets P10 reuse the
  producer's existing packed-K/V reader (`BRINGUP_RECIPE.md:711-716`,
  `08_PREFILL_INTEGRATION.md`).
- **K is stored post-RoPE, V raw** — as in every template (`models/demos/gpt_oss_d_p/tt/attention/prefill.py:162-165` comment, write at
  `:168` after the RoPE at `:151`). The golden-KV script must match.

### 3.12 `tt/attention/dense_sp.py` — SP ring SDPA (P5 stub → P8 real)

- **Responsibility.** Cache-backed ring-joint SDPA over the block-cyclic SP KV cache. **P5 creates
  the file with a `NotImplementedError` and a docstring pointing at the template**
  (`BRINGUP_RECIPE.md:698-700`).
- **Interface (P8).**
  ```python
  def dense_sp_attention(tt_q, cache_k, cache_v, tt_k_chunk, tt_v_chunk, *, kv_actual, logical_n,
                         n_kv, cache_global, head_dim, mesh_device, ccl_manager, program_config,
                         compute_kernel_config, scale, cluster_axis, slot_idx=0, layer_idx=0,
                         num_layers=1, write_chunk=True) -> ttnn.Tensor
  ```
- **Shapes.** `tt_q [1, 4, S_loc, 128]` → out `[1, 4, S_loc, 128]`; `cache_k/v` as §3.11; the
  persistent ring-gather scratch is `[1, 1, cache_global, 128]` bf8_b.
- **Template.** `models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:41` (`dense_sp_attention`), the op
  call `:106`, `persistent_output_buffer_k/v` `:116`/`:119`, `joint_strategy="rear"` `:122`, `dim=2`
  `:126`, `multi_device_global_semaphore=...ring_attention_ccl_semaphore_handles` `:127`,
  `ccl_core_grid_offset` `:133`, `use_column_major_ccl=True` `:134`, `is_causal=True` `:135`,
  `kv_cache_batch_idx=slot_idx*num_layers+layer_idx` `:141`, `kv_actual_isl` `:142`.
- **Deletions.** `attention_sink=` (`:144`), `sliding_window_size=` (`:145`), and `_gather_seq_len`
  (`:30`) — which for Llama's full-causal attention collapses to `return full_seq` (`:36`), so the
  buffer key is simply `cache_global`.
- **Inherited constraint.** `:77-81` asserts the cache is **bf8_b**: "KV_CACHE_DTYPE=bf16 is not
  supported for chunked prefill". This is what forces `DEC-017`.

### 3.13 `tt/attention/__init__.py` — `class Attention`

- **Responsibility.** Build `AttentionConfig` + weights and dispatch `attention_forward`. No
  per-layer type logic.
- **Interface.**
  ```python
  __all__ = ["Attention", "AttentionConfig", "ProgramConfig", "AttentionWeights",
             "LlamaKVCache", "allocate_kv_cache", "write_kv_chunk"]

  class Attention:
      def __init__(self, mesh_device, config: AttentionConfig, state_dict, *, mesh_config,
                   ccl_manager, program_config, layer_idx, transformation_mats=None,
                   weight_dtype=ttnn.bfloat8_b, tensor_cache_path=None, scatter_output=False)
      def __call__(self, hidden_states, rope_mats, kv_cache=None, user_id=0, batch_size=1,
                   cached_len=0, indexed_rope=False) -> ttnn.Tensor
  ```
- **Shapes.** In `[1,1,S_loc,4096]` bf16 TILE → out `[1,1,S_loc,4096]` bf16 TILE (scheme A).
- **Template.** `models/demos/gpt_oss_d_p/tt/attention/__init__.py:28` (class), `:38` (`__init__`),
  `:87` (`load_attention_weights`), `:103` (`__call__`), `:133` (dispatch to `attention_forward`).
- **Deletions.** `layer_types` / `is_sliding` (`:47`, `:78-84`), the per-layer
  `dataclasses.replace` (`:84`), and `position_idx` (unused in prefill, `:107`).

### 3.14 `tt/embedding.py`

- **Responsibility.** Token embedding — **replicated table**, no vocab sharding, no collective
  (`DEC-015`).
- **Interface.**
  ```python
  class Embedding:
      def __init__(self, mesh_device, hf_config, state_dict, *, mesh_config,
                   tensor_cache_path=None)
      def __call__(self, tokens: ttnn.Tensor) -> ttnn.Tensor
  ```
- **Shapes.**
  | tensor | shape (per chip) | dtype | layout |
  |---|---|---|---|
  | table | `[1, 1, 128256, 4096]` | bf16 | **ROW_MAJOR**, replicated (no mapper) |
  | `tokens` in | `[1, 1, 1, S_loc]` | uint32 | ROW_MAJOR, seq SP-sharded on rows, replicated on cols |
  | out | `[1, 1, S_loc, 4096]` | bf16 | TILE |
- **Template.** `models/demos/gpt_oss_d_p/tt/model.py:77` (substate), `:84` (`as_tensor`), `:88`
  (`ROW_MAJOR_LAYOUT`), `:315` (`ttnn.embedding(..., layout=ttnn.TILE_LAYOUT, dtype=bfloat16)`),
  token SP-shard at `:288-306`.
- **Why replicated.** `BRINGUP_RECIPE.md:806-807`; the table is 128256·4096·2 B ≈ 1.05 GiB per chip,
  which fits, and sharding it costs an all-gather per chunk plus a second layout to debug. The
  TODO at `models/demos/gpt_oss_d_p/tt/model.py:82-83` is the same deferral.

### 3.15 `tt/lm_head.py`

- **Responsibility.** The last-token logits projection — needed **only** for `G-MODEL`'s top-1
  check; prefill's real product is the KV cache.
- **Interface.**
  ```python
  class LMHead:
      def __init__(self, mesh_device, hf_config, state_dict, *, mesh_config,
                   tensor_cache_path=None, weight_dtype=ttnn.bfloat8_b)
      def __call__(self, x) -> ttnn.Tensor
  ```
- **Shapes.**
  | tensor | shape | dtype | layout |
  |---|---|---|---|
  | HF `lm_head.weight` | `[128256, 4096]` (untied — `tie_word_embeddings: false`) | bf16 | — |
  | device weight | `[4096, 16032]` per chip (`128256/8`, `16032/32 = 501` tiles ✓) | bf8_b | TILE, `column_parallel` |
  | `x` in (last-token tile) | `[1, 1, 32, 4096]` | bf16 | TILE |
  | out | `[1, 1, 32, 16032]` | bf8_b | TILE |
- **Template.** `models/demos/gpt_oss_d_p/tt/model.py:127` (transpose to `[hidden, vocab]`), `:134`
  (`as_tensor`), `:141` (`column_parallel`), `:241` (the matmul), host gather at `:322`
  (`process_output_prefill`), `:326-329`.
- **Two deviations (`DEC-015`).** (a) **No power-of-2 vocab padding.** gpt-oss rounds the per-device
  vocab up to a power of two (`models/demos/gpt_oss_d_p/tt/model.py:31` `compute_per_device_vocab`, `:38`) purely so
  `ttnn.topk`'s multi-core bitonic path works for on-device sampling. This iteration has **no
  on-device sampling** (prefill runs `skip_lm_head=True`), so the plain `128256/8 = 16032` shard is
  used and the padding, `padded_vocab_size`, and the `_supports_on_device_sampling` machinery
  (`:145`) are all deleted. (b) **No device-side all-gather on the vocab shard** — the TP concat
  happens on the host in `process_output_prefill`, exactly as the template does. So the LM head
  contributes **zero** collectives (`04_CCL_PLAN.md` §4 row 6).
- **`lm_head.weight` must not be aliased to the embedding.** Llama-3.1-8B is untied
  (`00_MODEL_CARD.md` §2, `models/demos/llama31_8b_d_p/configs/Llama-3.1-8B-Instruct/config.json:33`), unlike Llama-3.2-1B/3B. Assert
  `hf_config.tie_word_embeddings is False` and that the key exists.

### 3.16 `tt/layer.py` — `DecoderLayer`

- **Responsibility.** `norm → attn → residual → norm → mlp → residual`, plus the bring-up delta
  probe.
- **Interface.**
  ```python
  _DELTA_PROBE = os.environ.get("LLAMA31_8B_DELTA_PROBE", "") != ""

  class DecoderLayer:
      def __init__(self, mesh_device, hf_config, state_dict, layer_idx, *, mesh_config, ccl_manager,
                   program_config=None, transformation_mats=None, max_seq_len=1024,
                   weight_dtype=ttnn.bfloat8_b, tensor_cache_path=None,
                   sequence_parallel=False, scatter_output=False)
      def __call__(self, hidden_states, position_embeddings=None, kv_cache=None, user_id=0,
                   batch_size=1, cached_len=0, indexed_rope=False) -> ttnn.Tensor
  ```
- **Shapes.** In and out `[1, 1, S_loc, 4096]` bf16 TILE. Under scheme A **every** intermediate on
  the residual path is full-width; nothing in this file changes width.
- **Template.** `models/demos/gpt_oss_d_p/tt/layer.py:46` (class), `:65`/`:72` (the two norms with
  `substate` + per-module cache path), `:98` (`AttentionConfig` construction from `hf_config`),
  `:111` (`Attention`), `:126` (`__call__`), `:137-140` (the `ttnn.move` guard for
  `seqlen > 32*1024`), flow `:143-175`, `:19` (`_DELTA_PROBE`), `:22` (`_delta_stats`), probe call
  sites `:158-159`/`:169-170`.
- **Deletions.** The MoE `MLP` kwargs (`:60-62`, `:82-93`), `layer_types` (`:96`, `:105`, `:119`),
  and `position_idx`.
- **Keep.** The `ttnn.move` guard, every `deallocate(True)`, and the delta probe — the recipe
  explicitly wants the probe (`BRINGUP_RECIPE.md:743-745`) and Appendix E's masking caveat makes it
  the tool that localises what a residual-dominated layer PCC hides.

### 3.17 `tt/model.py` — `Model`

- **Responsibility.** `embedding → DecoderLayer×32 → final norm → (lm_head)`, plus the three prefill
  entry points the engine calls.
- **Interface.**
  ```python
  class Model:
      def __init__(self, mesh_device, hf_config, state_dict, *, mesh_config, ccl_manager,
                   max_seq_len=128*1024, num_layers=None, weight_dtype=ttnn.bfloat8_b,
                   tensor_cache_path=None, sequence_parallel=False, scatter_output=False)
      def prepare_inputs_prefill(self, tokens, start_pos=0, batch_size=1, user_id=0, **kwargs)
      def prefill_forward(self, x, rot_mats_global=None, user_id=0, get_last_token=-1,
                          kv_cache=None, batch_size=1, skip_lm_head=True,
                          on_layer_complete=None, cached_len=0, indexed_rope=False)
      def process_output_prefill(self, tt_out, last_token_idx)
  ```
- **Shapes.** `prepare_inputs_prefill`: torch `[1, S]` int → device `[1, 1, S_loc, 4096]` bf16 TILE
  (the 3-tuple `(tokens_embd, None, None)` interface is preserved). `prefill_forward` with
  `skip_lm_head=True` returns `[1, 1, S_loc, 4096]`; with `get_last_token=i` and the LM head, a
  `[1, 1, 32, 16032]` tile slice. `process_output_prefill` returns a host `[128256]` vector.
- **Template.** `models/demos/gpt_oss_d_p/tt/model.py:41` (class), `:93` (the layer list), `:113`
  (final norm), `:179` (`_forward_layers_and_head`, including the `on_layer_complete` per-layer
  seam P10 needs), `:246` (`prefill_forward`), `:279` (`prepare_inputs_prefill`), `:287-306` (the
  SP token shard), `:322` (`process_output_prefill`).
- **Deletions.** `rot_mats_local` (`:250`, for gpt-oss's sliding layers), `use_ep_moe` /
  `ep_seq_len_per_chip` / `expert_weight_dtype`, and the sampling hooks (`:145-157`, `:166-169`).
- **Addition.** `num_layers=None` (default = `hf_config.num_hidden_layers`) so `G-MODEL` can run the
  recipe's `n_layers ∈ {2, 4}` reduced stack (`BRINGUP_RECIPE.md:781-783`) without mutating
  `hf_config` — gpt-oss's harness mutates `hf_config.num_hidden_layers` in place
  (`models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:132`), which a frozen `LlamaHFConfig`
  forbids by design.

### 3.18 `tt/tt_prefill_runtime.py` (P7)

- **Responsibility.** The chunked-prefill runtime satisfying the engine's §2 contract. **Must not own
  the KV cache** on the engine path.
- **Interface.**
  ```python
  @dataclass
  class TtPrefillRuntimeConfig:
      num_layers: int; max_seq_len: int
      mesh_shape: tuple = (4, 8); default_chunk_size: int = 8192
      additional_chunk_sizes: tuple = (); num_users: int = 1
      sp_axis: int = 0; tp_axis: int = 1
      topology: ttnn.Topology = ttnn.Topology.Ring
      cache_dtype: ttnn.DataType = ttnn.bfloat8_b
      weight_cache_path: Path | None = None
      owns_kv_cache: bool = True
      is_first_rank: bool = True; is_last_rank: bool = True; first_layer_idx: int = 0
      @property
      def sp_factor(self) -> int
      @property
      def tp_factor(self) -> int

  class TtPrefillRuntime:
      def __init__(self, mesh_device, hf_config, state_dict, config)
      def make_chunk_input(self, token_ids, chunk_size=None)
      def compile(self, kv_caches=None)
      def prefill_chunk(self, input_tensor, kv_caches=None, *, slot_id, actual_start, actual_end, ...)
      def set_layer_ack_channel(self, ...)
      def kv_migration_base_address(self)
      def build_kv_chunk_table(self, ...)
      def read_slot_kv(self, ...)
      def kv_cache_pcc_check(self, ...)
  ```
- **Shapes.** `make_chunk_input` → `[1, 1, chunk/4, 4096]` bf16 TILE per chip;
  `prefill_chunk` writes into the caches of §3.11.
- **Template.** `models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:59`
  (`TtPrefillRuntimeConfig`, `sp_factor` `:88`, `tp_factor` `:92`), `:96` (class), `:204`
  (`make_chunk_input`), `:250` (`compile`), `:288` (`prefill_chunk`), `:375`
  (`build_kv_chunk_table`).
- **Deletions.** The MoE config fields, and `_build_indexed_rope`'s YaRN specifics (replaced by
  `tt/rope.build_indexed_rope`, §3.5).

### 3.19 `utils/general_utils.py`, `utils/substate.py`, `utils/__init__.py`

- **Responsibility.** Cache-file naming, `num_links` selection, and state-dict prefix splitting.
- **Interface.**
  ```python
  def get_cache_file_name(tensor_cache_path, name) -> str | None
  def cache_file_exists(cache_file_name) -> bool
  def get_default_num_links(mesh_device) -> int
  def substate(state, key) -> dict
  def has_substate(state, key) -> bool
  def indexed_substates(state, key) -> list[dict]
  ```
- **Shapes.** None (host helpers).
- **Template.** Copied verbatim from `models/demos/gpt_oss_d_p/utils/general_utils.py:11`, `:15`,
  `:27` (single-row mesh → **1** link at `:33`; else 2 on Blackhole / 4 on Wormhole at `:35`) and
  `models/demos/gpt_oss_d_p/utils/substate.py:15`, `:37`, `:53`. 35 + 74 lines, both fully
  model-agnostic (`DEC-006`).
- **Consequence to carry into P8.** At `(4,8)` → `num_links = 2`; at the `(1,2)/(1,4)/(1,8)` parity
  meshes → `num_links = 1`. The parity tests therefore do **not** exercise the 2-link path.

### 3.20 `scripts/generate_golden_kv_cache.py`, `scripts/verify_golden_kv.py` (P7)

- **Responsibility.** Run the torch reference in fp32 on real weights and emit per-layer post-RoPE K
  and raw V; then compare a device read-back per layer.
- **Interface.** `python -m …generate_golden_kv_cache --model $HF_MODEL --prompt-file … --out DIR`;
  `python -m …verify_golden_kv --golden DIR --device-dump DIR`.
- **Shapes / layout** (copy exactly — the engine's producer read-back expects it):
  ```
  {trace_dir}/metadata.json
  {trace_dir}/kv_cache/layer_<i>.safetensors   # key_cache_layer_<i>, value_cache_layer_<i>
                                               # [1, 8, S, 128] fp32, HF layout
  ```
- **Template.** `models/demos/minimax_m3/scripts/generate_golden_kv_cache.py:27` (the "Output
  format" header), minus M3's MSA-only `index_k_cache_layer_<i>`.
- **Runnable, not blocked** — weights are staged (Appendix F.1), which **voids `R-003`**.

### 3.21 `tt/runners/` (P10)

#### `tt/runners/adapters/llama.py`
- **Responsibility.** The engine-facing adapter: one class of per-model constants plus four factory
  methods.
- **Interface.**
  ```python
  class LlamaPrefillAdapter(PrefillModelAdapter):
      name = "llama31_8b_d_p"
      model_config = "Llama-3.1-8B-Instruct"
      hf_model_default = "meta-llama/Llama-3.1-8B-Instruct"
      ttnn_cache_default = ""
      prefill_trace_default = ""
      def load_hf_config(self) -> "PretrainedConfig"
      def weight_cache_path(self, mesh_shape: tuple) -> Path | None
      def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches
      def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams)
  ```
- **Shapes.** `allocate_kv_cache` returns the §3.11 caches
  (`[32·num_users, 1, max_seq_len/4, 128]` bf8_b each); `weight_cache_path` returns
  `<cache>/llama31_8b_d_p_bh_32dev/4x8`. `load_hf_config` returns the object `llama_hf_config()`
  normalises (`DEC-009`).
- **Template.** `models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:41` (class), `:45-49`
  (the five runner defaults), `:63` (`load_hf_config`), `:75` (`weight_cache_path`), `:96`
  (`allocate_kv_cache`), `:120` (`build_runtime`). Base class
  `models/demos/common/prefill/adapter.py:104`, `PrefillRunParams` `:46`, registry `:277`.
  **Delete** `default_gate_mode` (`:50`) — MoE-router-only.

#### `tt/runners/kv_chunk_table.py`
- **Responsibility.** Build the block-cyclic KV address table the migration path reads (config
  `0..N-1` = K heads, `N..2N-1` = V heads).
- **Interface.** `def build_kv_chunk_table(kv_cache, *, mesh_device, slot_id, chunk_size, num_layers)
  -> list[...]`, called from `TtPrefillRuntime.build_kv_chunk_table`.
- **Shapes.** Host-side address/offset table; no device tensors. Bank walk must match
  `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32` and the 8 DRAM banks of §3.11.
- **Template.** `models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:375`
  (`build_kv_chunk_table`); bank count from
  `models/demos/common/prefill/runners/migration.py:337`.

#### `tt/runners/manifests/llama31_8b_d_p.json`
- **Responsibility.** The runner manifest (binding, `global_env`, layer/chunk counts) the
  disaggregated engine loads.
- **Interface.** JSON; no Python surface.
- **Shapes.** n/a. Must pin `PREFILL_SP=4` / `PREFILL_TP=8` (the engine's own defaults are 8/4 —
  `models/demos/common/prefill/runners/prefill_producer.py:83`, `:84`) and set any `PREFILL_*` var
  in `global_env`, because `tt-run` forwards only `TT_/ARCH_/WH_/TTNN_/DEEPSEEK_/MESH_` prefixes
  (Appendix B).
- **Template.** `models/demos/gpt_oss_d_p/tt/runners/manifests/` (the loopback-migration manifest
  is the closest shape).

**The P10 trap is already documented:** the producer's KV read-back branches on `ADAPTER.name`
(`models/demos/common/prefill/runners/prefill_producer.py:503`; `:534` is the plain packed-K/V
reader Llama wants, `:685` the MLA fallback it must not get) — see `08_PREFILL_INTEGRATION.md`.

### 3.22 `tests/` — per-file contracts

`hf_config` is `llama_hf_config(llama_config_dims())` in every test (`DEC-009`); `state_dict` is the
session fixture (`conftest.py`), which returns `{}` on a weightless machine. All P5/P6 tests drive
the torch reference and the TT module from **identical random weights** and compare with `comp_pcc`
(`models/common/utility_functions.py:488`).

| File | Responsibility (one sentence) | Interface / parametrisation | Input → output shapes | Template |
|---|---|---|---|---|
| `unit/test_reference_model.py` | Prove the torch oracle is deterministic and agrees with HF. | host-only; two dim sets (full, tiny) | `[1, S, 4096]` fp32 → same | delivered in P1 |
| `unit/test_mesh_config.py` | `MeshConfig` arithmetic and its refusals. | no device for the arithmetic; `mesh_device` `[(1,1)]` for `CCLManager` | asserts `sp/tp/shard_size(4096)==512/shard_size(14336)==1792`; `MeshConfig((1,8), tp=4)` **raises** | `BRINGUP_RECIPE.md:604-609` |
| `unit/test_ccl_semaphores.py` | `CCLManager` allocates its semaphores once, not per layer. | `mesh_device` target mesh + `device_params` fabric | asserts list lengths **6 / 4 / 2 / 2** after building the 32-layer model | `models/demos/gpt_oss_d_p/tt/ccl.py:66`, `:72`, `:78`, `:84` |
| `unit/test_rms_norm_vs_ref.py` | Plain RMSNorm vs torch. | `mesh_device` `[(1,1)]`, `reset_seeds`, `seq_len ∈ {32, 512, 4096}` | `[1,1,S,4096]` bf16 TILE → same | `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:149` (the parametrise shape) |
| `unit/test_rope_vs_ref.py` | Meta-convention RoPE on device vs the HF `rotate_half` path, **and** that llama3 scaling is active. | `[(1,1)]`, `seq_len ∈ {128, 512, 8192}` | `[1, n_heads, S, 128]` bf16 TILE → same; cos/sin `[1,1,S,128]` | `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:83` (`_build_cos_sin` builds **both** conventions from one frequency set — copy exactly) |
| `unit/test_mlp_vs_ref.py` | Dense SwiGLU vs torch at both weight dtypes. | `[(1,1)]`, `seq_len ∈ {32, 512, 4096}`, `weight_dtype ∈ {bf8_b, bf16}` | `[1,1,S,4096]` → `[1,1,S,4096]` | `models/tt_transformers/tests/test_mlp.py` + `models/demos/minimax_m3/tt/dense_mlp.py:87` |
| `unit/test_attention_vs_ref.py` | Full attention block (QKV → GQA split → RoPE → causal SDPA → o_proj) vs torch. | `[(1,1)]`, `seq_len ∈ {128, 512, 2048}` | `[1,1,S,4096]` → `[1,1,S,4096]` | `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:117` (`_torch_attention` — remove the sink column and the sliding term), threshold at `:258` |
| `unit/test_kv_cache_vs_ref.py` | Cache write correctness **and** absence of collateral writes. | `[(1,1)]` + fabric `device_params`; cache dtype ∈ {bf8_b, bf16} | writes `[1, n_kv, S, 128]`; reads back `[slots, 1, S, 128]` | `models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:138`; asserts modelled on `:149`, `:155-159` |
| `unit/test_decoder_layer_vs_ref.py` | One decoder layer vs HF `LlamaDecoderLayer` — **integration check only** (§5.1). | `[(1,1)]`, `seq_len ∈ {128, 512, 2048}` | `[1,1,S,4096]` → `[1,1,S,4096]` | `models/tt_transformers/tt/model_config.py:4393` (usable now that weights are staged) or the in-test layer |
| `unit/test_weight_loading.py` | No missing and no silently-unused checkpoint keys; cache-only rebuild is bit-identical. | 1 card; `requires_hf_reference` | asserts all **291** keys consumed; rebuild from `{}` + populated cache | `BRINGUP_RECIPE.md:766-772` |
| `unit/test_model_vs_ref.py` | Full stack hidden states + top-1 — **integration check only**. | `[(1,1)]`, `num_layers ∈ {2, 4, 32}`, `seq_len ∈ {128, 512}` | tokens `[1,S]` → `[1,1,S,4096]`; logits `[1,1,32,16032]` | `models/demos/gpt_oss_d_p/tt/model.py:246` |
| `unit/test_attention_chunked_vs_ref.py` | N-chunk prefill ≡ one-shot prefill. | `[(1,1)]`+, chunked vs one-shot | KV caches compared per layer | `models/demos/minimax_m3/tests/unit/` chunked test |
| `unit/test_tp_parity.py` | Multi-device module output equals single-device output. | `[(1,1)]` vs `[(1,2)]`/`[(1,4)]`/`[(1,8)]`, `device_params` `FABRIC_1D` | same shapes both sides under scheme A (`DEC-018`) → direct comparison | `BRINGUP_RECIPE.md:845-850` |
| `galaxy_prefill_kv_pcc.py` | Per-layer K/V PCC vs golden on the target mesh, one-shot and chunked, 3× for `G-RACE`. | standalone script; `(4,8)`, `FABRIC_1D_RING`, torus descriptor | reads the §3.11 caches; writes a per-layer table into `bringup_log/raw/` | `models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:44`, `:121`, `:122`, `:161` |

### 3.23 Support files

| File | Responsibility | Interface | Shapes | Template / status |
|---|---|---|---|---|
| `README.md` (P9) | Architecture table, deployment path, status table with **measured** PCC, run commands, env-var table, layout, and a "what is not implemented" section — plus the "why not TTTv2 / `models/common/models/llama3_8b`" line (§6). | Markdown | n/a | `models/demos/minimax_m3/README.md`; checklist `BRINGUP_RECIPE.md:864-889` |
| `conftest.py` | Package pytest config: `--skip-model-load` and the session `state_dict` fixture. | `pytest_addoption(parser)`; `state_dict(request) -> dict` | returns `{}` or 291 torch tensors in HF layout | **exists** (P1); mirrors `models/demos/minimax_m3/conftest.py:13`, `:17`. Does **not** redefine `mesh_device`/`reset_seeds` (`conftest.py:554`, `:34`). |
| `tests/test_factory.py` | Dimension-only config access, the real-weight skip marker, and the per-test device objects. | `llama_config_dims() -> dict`; `rope_theta(cfg)`; `rope_scaling(cfg)`; `requires_hf_reference`; `TestFactory.setup_test(mesh_device, *, tp=None, ...) -> dict` | host only | **exists** (P1); `models/demos/minimax_m3/tests/test_factory.py:45`, `:56`. **P5.1 must change one line**: `setup_test` returns `"hf_config": llama_config_dims()` (a dict) and must wrap it in `llama_hf_config()` per `DEC-009`. |
| `scripts/verify_citations.py` | Re-verify every load-bearing `path:line` in the bring-up logs, plus scan the logs for unresolvable references. | `python .../verify_citations.py`; exit 0 iff clean | n/a | **exists** (P0), extended each phase (§8) |
| `configs/Llama-3.1-8B-Instruct/config.json` | The bundled dims, byte-identical to the `tt_transformers` copy. | JSON | n/a | **exists** (P1), `DEC-005` |
| `utils/__init__.py`, `tt/__init__.py`, `tests/__init__.py`, `tests/unit/__init__.py`, `scripts/__init__.py`, `tt/runners/__init__.py`, `tt/runners/adapters/__init__.py` | Package markers only — SPDX header pair and nothing else. | n/a | n/a | `models/demos/gpt_oss_d_p/tt/__init__.py` (a **three**-line header: SPDX-FileCopyrightText, bare `#`, SPDX-License-Identifier) |

---

## 4. Per-layer tensor-shape table — real numbers at `(4,8)` / TP=8 / SP=4

`S_loc = S/4`. Scheme A residual (`DEC-018`). Weight dtype `bfloat8_b`, activation `bfloat16`,
KV cache `bfloat8_b`. Every width below is tile-aligned (`TILE_SIZE = 32`, measured).

| tensor | shape (per chip) | tiles (last dim) | dtype | layout |
|---|---|---|---|---|
| token ids | `[1, 1, 1, S_loc]` | — | uint32 | ROW_MAJOR (SP-sharded rows, replicated cols) |
| **hidden / residual** | `[1, 1, S_loc, 4096]` | 128 | bf16 | TILE (replicated across TP) |
| norm out | `[1, 1, S_loc, 4096]` | 128 | bf16 | TILE |
| norm weight | `[1, 1, 128, 32]` | 1 | bf16 | ROW_MAJOR, replicated |
| `q_proj` out | `[1, 1, S_loc, 512]` | 16 | bf16 | TILE |
| `k_proj` / `v_proj` out | `[1, 1, S_loc, 128]` | 4 | bf16 | TILE |
| `kv` concat (for the head split) | `[1, 1, S_loc, 256]` | 8 | bf16 | TILE |
| **Q** | `[1, 4, S_loc, 128]` | 4 | bf16 | TILE |
| **K, V** | `[1, 1, S_loc, 128]` | 4 | bf16 | TILE |
| cos / sin (per-chunk) | `[1, 1, S_loc, 128]` | 4 | bf16 | TILE, replicated |
| cos / sin (indexed, whole-cache) | `[1, 1, max_seq_len/4, 128]` | 4 | bf16 | TILE, seq SP-sharded |
| RoPE transformation mat | `[1, 1, 32, 32]` | 1 | bf16 | TILE, replicated |
| **KV cache** `k` and `v` (each) | `[32·num_users, 1, max_seq_len/4, 128]` | 4 | **bf8_b** | TILE, DRAM `NdShard [1,1,32,128]` over 8 banks |
| SDPA out | `[1, 4, S_loc, 128]` | 4 | bf16 | TILE |
| attn out pre-`o_proj` (concat heads) | `[1, 1, S_loc, 512]` | 16 | bf16 | TILE |
| `o_proj` out (partial sum) | `[1, 1, S_loc, 4096]` | 128 | bf16 | TILE |
| attn out post all-reduce | `[1, 1, S_loc, 4096]` | 128 | bf16 | TILE |
| MLP gate / up | `[1, 1, S_loc, 1792]` | 56 | bf16 | TILE |
| MLP SwiGLU act | `[1, 1, S_loc, 1792]` | 56 | bf16 | TILE |
| `down_proj` out (partial sum) | `[1, 1, S_loc, 4096]` | 128 | bf16 | TILE |
| MLP out post all-reduce | `[1, 1, S_loc, 4096]` | 128 | bf16 | TILE |
| residual, **scheme B** (not used) | `[1, 1, S_loc, 512]` | 16 | bf16 | TILE |
| final norm out | `[1, 1, S_loc, 4096]` | 128 | bf16 | TILE |
| logits (last-token tile) | `[1, 1, 32, 16032]` | 501 | bf8_b | TILE |

### 4.1 Per-chip weight shapes (all 291 tensors)

| weight | count | HF `[out, in]` | ttnn pre-shard | per chip @TP=8 | mapper |
|---|---|---|---|---|---|
| `input_layernorm.weight` | 32 | `[4096]` | `[1,1,128,32]` | `[1,1,128,32]` | replicate |
| `post_attention_layernorm.weight` | 32 | `[4096]` | `[1,1,128,32]` | `[1,1,128,32]` | replicate |
| `self_attn.q_proj.weight` | 32 | `[4096, 4096]` | `[1,1,4096,4096]` | `[1,1,4096,512]` | column |
| `self_attn.k_proj.weight` | 32 | `[1024, 4096]` | `[1,1,4096,1024]` | `[1,1,4096,128]` | column |
| `self_attn.v_proj.weight` | 32 | `[1024, 4096]` | `[1,1,4096,1024]` | `[1,1,4096,128]` | column |
| `self_attn.o_proj.weight` | 32 | `[4096, 4096]` | `[1,1,4096,4096]` | `[1,1,512,4096]` | row |
| `mlp.gate_proj.weight` | 32 | `[14336, 4096]` | `[1,1,4096,14336]` | `[1,1,4096,1792]` | column |
| `mlp.up_proj.weight` | 32 | `[14336, 4096]` | `[1,1,4096,14336]` | `[1,1,4096,1792]` | column |
| `mlp.down_proj.weight` | 32 | `[4096, 14336]` | `[1,1,14336,4096]` | `[1,1,1792,4096]` | row |
| `model.embed_tokens.weight` | 1 | `[128256, 4096]` | `[1,1,128256,4096]` | `[1,1,128256,4096]` | replicate |
| `model.norm.weight` | 1 | `[4096]` | `[1,1,128,32]` | `[1,1,128,32]` | replicate |
| `lm_head.weight` | 1 | `[128256, 4096]` | `[4096, 128256]` | `[4096, 16032]` | column |

`9·32 + 3 = 291` — the number `G-WEIGHTS` asserts. **No bias tensors anywhere.**
Q/K (and only Q/K) are `reverse_permute`d at load for the Meta RoPE convention.

---

## 5. Test tree ↔ gate map, with the Appendix E thresholds

| Test file | Gate | Mesh | Threshold (Appendix E revised) | Oracle measured on this box |
|---|---|---|---|---|
| `unit/test_reference_model.py` | `G-REF` | host | bit-identical ×2; cross-ref PCC ≥ 0.9999 | **PASS**, PCC 1.0 |
| `unit/test_mesh_config.py` | `G-MESH` | none + 1 card | exact asserts | — |
| `unit/test_ccl_semaphores.py` | `G-SEMAPHORE` | target mesh | list lengths **6 / 4 / 2 / 2** | — |
| `unit/test_rms_norm_vs_ref.py` | `G-RMS` | (1,1) | **PCC ≥ 0.9999** | `tt_transformers` 0.9999867 / 0.9999886 |
| `unit/test_rope_vs_ref.py` | `G-ROPE` | (1,1) | PCC ≥ 0.999 **and** scaled ≠ unscaled `inv_freq` | `precompute_freqs` vs HF: `max|Δ| = 0.0` |
| `unit/test_mlp_vs_ref.py` | `G-MLP` | (1,1) | **≥ 0.999 @bf8_b**, ≥ 0.9995 @bf16 | 0.9995823 @bf8_b, seq 512 |
| `unit/test_attention_vs_ref.py` | `G-ATTN` | (1,1) | **≥ 0.999** | 0.9996099 / 0.9996010 |
| `unit/test_kv_cache_vs_ref.py` | `G-KV` | (1,1) | ≥ 0.99 @bf8_b (record bf16); written-region-only asserts | — |
| `unit/test_decoder_layer_vs_ref.py` | `G-LAYER` | (1,1) | **≥ 0.999** — *integration check only* | 0.9999985 (see §5.1) |
| `unit/test_weight_loading.py` | `G-WEIGHTS` | 1 card | 0 missing, 0 unused of 291; cache-only rebuild bit-identical | — |
| `unit/test_model_vs_ref.py` | `G-MODEL` | (1,1) | ≥ 0.999 hidden state; 100% top-1 — *integration check only* | — |
| `unit/test_attention_chunked_vs_ref.py` | `G-CHUNK` | (1,1)+ | ≥ 0.999 chunked ≡ one-shot; ≥ 0.99 K / ≥ 0.98 V vs golden | — |
| `unit/test_tp_parity.py` | `G-TP-PARITY` | (1,1) vs (1,TP) | ≥ 0.999 device-vs-device | — |
| `galaxy_prefill_kv_pcc.py` | `G-MESH-KV`, `G-RACE` | (4,8) | per-layer min recorded; 3 runs bit-identical | — |

### 5.1 `G-LAYER` and `G-MODEL` are integration checks, never sublayer evidence

Appendix E measured `test_decoder_prefill` at **0.9999985** — *higher* than either of its own
sublayers (attention 0.9996099, MLP 0.9995823). The residual stream dominates the correlation, so a
full-layer PCC **partially launders a degraded sublayer**. Three rules follow, and they are binding
on P5/P6:

1. `G-RMS`, `G-ROPE`, `G-MLP`, `G-ATTN`, `G-KV` are the **only** evidence that a sublayer is
   correct. A passing `G-LAYER` may never be used to excuse a missing, skipped or weakened sublayer
   gate.
2. A layer PCC of ~0.9999 while a sublayer sits at ~0.99 is the **signature of this masking**, not
   proof the sublayer is fine. Treat the band between the revised gate and the recipe's original
   guess as *investigate*, never *pass*.
3. This is why the per-layer delta probe (§3.16) is mandatory: magnitude ratios localise what a
   residual-dominated PCC hides, and its output goes into `bringup_log/raw/`.

The same reasoning applies upward: `G-MODEL`'s hidden-state PCC over 32 layers is even more
residual-dominated than one layer's.

---

## 6. Why not `models/common/modules/` (TTTv2) or `models/common/models/llama3_8b/`

Recorded here so P9's `README.md` can cite it — it is the first question a reviewer asks, because
this tree **already contains a complete Llama-3.1-8B**. Not re-evaluated (Appendix F.3 settles it);
recorded with its evidence.

| Candidate | What it is | Why it is not the base |
|---|---|---|
| `models/common/modules/` (TTTv2) | A shared, unit-tested module library: `MLP1D/2D`, `RMSNorm1D/2D`, `Attention1D`, `RotarySetup1D`, `Embedding1D`, `LMHead1D`, cached `TT_CCL`. "Universal Module Contract" at `models/common/modules/README.md:38`. | (a) **`MLP2D`'s "2D" is 2D *tensor* parallelism, not TP×SP.** Its prefill path reduce-scatters on `cluster_axis=1` then closes with an all-reduce on `cluster_axis=0` (`models/common/modules/mlp/mlp_2d.py:461`). With SP on the row axis, that all-reduce would sum activations belonging to **different tokens** — silently wrong, and it would still produce a plausible PCC on a 1-row mesh. (b) There is **no `Attention2D`** — only `attention_1d.py`. (c) No chunked-prefill runtime, no KV migration, no `common/prefill` adapter, i.e. nothing for P7 or P10. (d) A different weight/config idiom (`LazyWeight`, `<Name>Config` + `from_config`) than the `(mesh_device, hf_config, state_dict, …)` convention this package must honour. |
| `models/common/models/llama3_8b/` | A complete TTTv2 Llama-3.1-8B: `Embedding1D → RotarySetup1D → TransformerBlock1D×N → RMSNorm1D → LMHead1D`, with generator and HF adaptor. | It **cannot run the target mesh**: `models/common/models/llama3_8b/model.py:890` raises `ValueError("Llama3Transformer1D only supports 1D mesh topologies.")`, guarded by `is_galaxy_cluster` at `:884`. It is N150/N300/T3K, decode/generation-oriented, with no disaggregated prefill. |

**What TTTv2 did give us**, and what it stays the reference for: the definitive answer to the
recipe's own open SwiGLU question — `models/common/modules/mlp/mlp_1d.py:84` shows
`mlp_activation_type` already **defaults to `ttnn.UnaryOpType.SILU`**, and `:262` shows the fused
`ttnn.mul(..., input_tensor_a_activations=[...])` call shape. `models/common/modules/mlp/mlp_2d.py:5` is worth reading before
writing `tt/mlp.py`.

The one-line answer for the README: *2D-mesh TP×SP on a 32-chip Galaxy plus the
disaggregated-prefill engine contract — neither of which TTTv2 has yet.*

---

## 7. What P5 must decide (deliberately left open here)

| Open item | Where it bites | What P3 already fixed |
|---|---|---|
| **KV cache dtype: bf8_b vs bf16** | `G-KV`, `G-CHUNK` thresholds | `DEC-017` sets the **default to bf8_b** and requires the bf16 number be *measured and recorded*, not assumed. The choice is effectively forced: `models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:77-81` asserts a bf8_b cache for the chunked ring path, so bf16 is a measurement-only mode that cannot ship. |
| **SDPA q/k chunk sizes** | `G-ATTN` perf, not correctness | `ProgramConfig` defaults (32/32 below 2048, 256/256 at or above) carried from the template; a change is a `DEC`. |
| **`math_fidelity` / `fp32_dest_acc_en`** | `G-ATTN`, `G-MLP` PCC | Template defaults `HiFi4` / `False`; note `models/demos/gpt_oss_d_p/tt/attention/prefill.py:200` records that the ring op **requires** `fp32_dest_acc_en=False`, so the SP path cannot raise it. |
| **`CHUNK_SIZE` and `MAX_SEQ_LEN`** | P7 | Constraints fixed: `CHUNK_SIZE % 128 == 0` and `MAX_SEQ_LEN % CHUNK_SIZE == 0` at SP=4 (`00_MODEL_CARD.md` §4.4). |

---

## 8. Citation verification

`scripts/verify_citations.py` was extended by this phase in two ways:

1. **`CITES` grew from 231 to 380 explicit entries** — every new load-bearing `path:line` in this
   document and in `04_CCL_PLAN.md`, each with the substring that must be on that line.
2. **A second pass was added (`scan_docs()`)** that extracts *every* `` `path:line` `` reference from
   `03_OUTLINE.md` and `04_CCL_PLAN.md` and asserts the file exists and the line is in range. It is
   the safety net for references that carry no needle, and it makes "extend the verifier every
   phase" cheap. **140 references scanned, 140 resolved.**

Result: **380 / 380 explicit citations verified, 0 mismatched, 0 missing files; 140 / 140 doc
references resolved** — `raw/G-OUTLINE_20260903T170527Z.log`.

**It earned its keep again.** The first draft of this document contained **10 wrong line numbers**,
all caught by pass 1 and corrected here: `gpt_oss_d_p/tt/model.py` `:37`→`:38` (`next power of 2`),
`:287`→`:288` (the SP shard branch), `:325`→`:326` (`get_device_tensors`);
`gpt_oss_d_p/tt/rope.py` `:145`→`:146` (the `TILE_SIZE*sp` constraint);
`gpt_oss_d_p/tt/config.py` `:44`→`:45` (the `raise`); `gpt_oss_d_p/tt/rms_norm.py` `:36`→`:37`
(the norm dtype); `gpt_oss_d_p/utils/general_utils.py` `:34`→`:35` (`is_blackhole`);
`gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py` `:135`→`:132` (`num_hidden_layers`);
`models/demos/minimax_m3/tt/residual.py:9` (right line, wrong needle);
`tt_transformers/tt/common.py` `:531`→`:529` (`torch.stack`).

**And it found an eleventh, inherited from P2:** `02_SURVEY.md:76` cites
`models/demos/gpt_oss_d_p/tt/model.py:252` for the `rot_mats_local` parameter; it is at **`:250`**
(`:252` is `get_last_token`). P2's verifier reported 200/200 because that particular reference was
never added to `CITES` — which is the argument for pass 2.

---

## 9. Files P8 added (and the two that were planned but never created)

Appendix F.9 states the rule these follow: **when adding a gate, add its file in the same edit**, or
the gate silently becomes a `NOT-RUN`.

| file | gate | why it did not exist before |
|---|---|---|
| `tests/unit/test_tp_parity.py` | `G-TP-PARITY` | Appendix F.9 assigned it to P5, but P5 had no multi-device mesh to parametrise, so it was never created. P8 creates it. |
| `tests/unit/test_kv_cache_tp8.py` | **`G-KV-TP8`** (new) | `R-027`'s coverage hole: the model -> cache path needs TP=8, which no earlier phase could open. |
| `tests/unit/test_sp_attention_chunked.py` | `G-CHUNK-ATTN`, **`G-SP-RING`** (new) | `G-CHUNK-ATTN` was `BLOCKED` on `dense_sp_attention` (`R-028`). `G-SP-RING` is new: the ring op alone, against fp32 torch, plus the `fp32_dest_acc_en` A/B. |
| `tests/galaxy_prefill_kv_pcc.py` | `G-MESH-KV`, `G-RACE` | §3.22 lists it under the P8 row; it is the `gpt_oss_d_p` harness minus the MoE knobs, plus `PREFILL_RUNS` (the `G-RACE` repetition on **one** `CCLManager`). |
| `tests/fabric_topology_matrix.py` | **`G-FABRIC-MATRIX`** (new) | Not planned. It exists because two P8 assumptions about this machine were wrong (`DEC-080`, `DEC-081`) and the corrections needed a reproducible measurement rather than a paragraph. |

Additions to existing files: `tests/test_factory.py` gains `TestFactory.setup_submesh` and
`parametrize_galaxy_submeshes` (`DEC-080`); `tests/unit/test_ccl_semaphores.py` gains the `(4,8)`
`G-SEMAPHORE` test; `tests/unit/test_weight_loading.py` gains the TP=8 cache-only test (`R-017`);
`tt/attention/config.py` gains `get_ring_sdpa_config` / `get_ring_compute_kernel_config`
(`DEC-083`, `DEC-084`); `tt/attention/prefill.py` gains `_run_sp_bootstrap_sdpa` (the `DEC-021`
bootstrap P4 owed to P8) and the `use_cache_backed_ring` selection; `tt/tt_prefill_runtime.py` gains
one bring-up-only logging helper, `_log_layer_error_steps` (`R-025`).

`tt/attention/dense_sp.py` stops being a stub. Nothing else in `tt/` changed, and nothing under
`tt/runners/` was written — that is P10's.
