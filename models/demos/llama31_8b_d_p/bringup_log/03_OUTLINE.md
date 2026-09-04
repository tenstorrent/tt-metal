# 03 — Package outline

Phase P3. The target file tree, each file's responsibility, its public interface, the tensor shapes
crossing it, and the template it mirrors. Written **before** any device code exists (`tt/` holds only
its `__init__.py` today). Date (UTC): 2026-09-04. Gate: `G-OUTLINE`.

Nothing here is executed. What P3 buys is that P5–P10 never has to guess a signature, a shape or an
owner — and that every gate in Appendix A has a file to live in before the phase that runs it starts
(`BRINGUP_RECIPE.md:1056` — "An unowned gate silently becomes a `NOT-RUN`").

---

## 0. The four questions P3 was required to settle

`06_GATES.md`'s P2 status line named two; reading the templates against the recipe surfaced two more.

| # | Question | Settled by |
|---|---|---|
| 1 | Does the package keep `docs/` and `scripts/__init__.py`, and does it vendor the recipe? | `DEC-017` — no to all three; the P3 tree wins over P0's literal reading |
| 2 | Is `utils/` created at all, given `DEC-013` imports the four helpers from `models/demos/gpt_oss_d_p/utils/`? | `DEC-018` — no `utils/` package in the committed tree |
| 3 | Three separate Q/K/V projections (`DEC-016`) — but then **what performs the GQA head split**? The op the survey cites requires a *fused* layout, and the template `DEC-016` cites as the three-weight pattern actually **fuses**. | `DEC-019` (corrects `DEC-016`'s evidence; keeps its choice) |
| 4 | `head_dim`, which the templates read as `hf_config.head_dim` (`models/demos/gpt_oss_d_p/tt/model.py:64`) and which **does not exist** as a key in Llama's `config.json` | `DEC-020` — the one normalised constructor derives and exposes it |

Two more numbers the outline's shape table cannot be written without, both logged here rather than
smuggled in as table cells: the KV-cache dtype (`DEC-021`) and the activation dtype ladder
(`DEC-022`).

---

## 1. The committed file tree

Derived from `BRINGUP_RECIPE.md:987-1054`, with every deviation marked `[DEV-n]` and justified in §1.1.
`(exists)` = already in the tree from P0/P1. The phase column is the phase that *creates* the file.

```
models/demos/llama31_8b_d_p/
├── README.md                                   P9   arch table, deployment path, status, run cmds, env vars
├── __init__.py                                 P0   (exists)
├── conftest.py                                 P1   (exists) session state_dict + --skip-model-load
├── configs/Llama-3.1-8B-Instruct/config.json   P0   (exists) bundled dims, byte-identical to $HF_MODEL's
├── bringup_log/                                P0   (exists) 9 log files + raw/
├── tt/
│   ├── __init__.py                             P0   (exists)
│   ├── config.py                               P5.1 MeshConfig: mappers + collective wrappers
│   ├── ccl.py                                  P5.1 CCLManager: subdevice, semaphores, ring scratch
│   ├── model_config.py                         P6.2 ModelArgs + the ONE normalised hf_config constructor
│   ├── rms_norm.py                             P5.2 plain RMSNorm (+ dormant distributed branch)
│   ├── rope.py                                 P5.3 llama3-scaled cos/sin + transformation matrix
│   ├── mlp.py                                  P5.4 dense SwiGLU, TP collective inside
│   ├── attention/
│   │   ├── __init__.py                         P5.5 class Attention
│   │   ├── config.py                           P5.5 AttentionConfig + ProgramConfig (pinned 8x8 SDPA grid)
│   │   ├── weights.py                          P5.5 load/swizzle/shard/tilize q,k,v,o
│   │   ├── operations.py                       P5.5 head split/merge, RoPE apply, CCL tail helpers
│   │   ├── prefill.py                          P5.5 attention_forward()
│   │   ├── kv_cache.py                         P5.6 LlamaKVCache, allocate_kv_cache, write_kv_chunk
│   │   └── dense_sp.py                         P5.5 NotImplementedError stub; filled in P8
│   ├── embedding.py                            P6.2 token embedding (replicated — DEC-024)
│   ├── lm_head.py                              P6.3 V/TP shard; exists for G-MODEL's top-1
│   ├── layer.py                                P6.1 DecoderLayer + the delta probe
│   ├── model.py                                P6.3 Model: embedding -> layers -> norm -> (lm_head)
│   ├── tt_prefill_runtime.py                   P7   chunked runtime to the engine's contract
│   └── runners/
│       ├── __init__.py                         P10
│       ├── kv_chunk_table.py                   P10  block-cyclic KV address table
│       ├── adapters/__init__.py                P10  [DEV-4]
│       ├── adapters/llama.py                   P10  LlamaPrefillAdapter
│       └── manifests/llama31_8b_d_p.json       P10  {"env": {"PREFILL_MODEL": "llama31_8b_d_p"}}
├── scripts/
│   ├── verify_citations.py                     P0   (exists) extended every phase
│   ├── generate_golden_kv_cache.py             P7   torch fp32 -> per-layer golden KV
│   └── verify_golden_kv.py                     P7   golden trace structural check (no ttnn)
└── tests/
    ├── __init__.py                             P0   (exists)
    ├── test_factory.py                         P1   (exists) fixtures + the ONE noise-floor definition
    ├── unit/
    │   ├── __init__.py                         P0   (exists)
    │   ├── test_reference_model.py             P1   (exists) G-REF
    │   ├── test_mesh_config.py                 P5.1 G-MESH
    │   ├── test_ccl_semaphores.py              P5.1 G-SEMAPHORE (device half in P8)
    │   ├── test_rms_norm_vs_ref.py             P5.2 G-RMS
    │   ├── test_rope_vs_ref.py                 P5.3 G-ROPE
    │   ├── test_mlp_vs_ref.py                  P5.4 G-MLP
    │   ├── test_attention_vs_ref.py            P5.5 G-ATTN
    │   ├── test_kv_cache_vs_ref.py             P5.6 G-KV
    │   ├── test_kv_cache_tp8.py                P8   G-KV-TP8
    │   ├── test_attention_chunked_vs_ref.py    P7   G-CHUNK (deltas 1-2)
    │   ├── test_sp_attention_chunked.py        P8   G-SP-RING + G-CHUNK-ATTN (delta 3)
    │   ├── test_decoder_layer_vs_ref.py        P6.1 G-LAYER
    │   ├── test_embedding_vs_ref.py            P6.2 (regression only; no Appendix A gate)
    │   ├── test_lm_head_vs_ref.py              P6.3 (regression only; no Appendix A gate)
    │   ├── test_weight_loading.py              P6.2 G-WEIGHTS (+ P8 TP=8 extension)
    │   ├── test_tp_parity.py                   P8   G-TP-PARITY
    │   ├── test_model_vs_ref.py                P6.3 G-MODEL
    │   ├── test_prefill_runtime_chunked.py     P7   G-RUNTIME
    │   ├── test_prefill_adapter.py             P10  G-ADAPTER
    │   └── test_kv_chunk_table.py              P10  G-KV-TABLE
    ├── fabric_topology_matrix.py               P8   G-FABRIC-MATRIX (subprocess-isolated sweep)
    └── galaxy_prefill_kv_pcc.py                P8   G-MESH-KV, G-RACE
```

**File count:** 57 tracked files at the end of P10 (excluding `bringup_log/` and `raw/`), of which 9
exist today (i.e. at the end of **P3**, when this document was written).

> **P9 check (`G-CLEAN`).** The delivered tree is **57** files, matching this contract
> file-for-file, and **50** of them are not `__init__.py`. P9 added **no** file: the
> generated needle-manifest it considered was dropped in favour of 20 hand-written `CITES` rows
> (`DEC-120`), precisely so this count would not move. Two figures elsewhere are P3-era and do not
> describe the delivered tree — `06_GATES.md`'s `G-OUTLINE` row says "41 files contracted (49/49
> non-`__init__` tree files)" and `DEC-109` says "the P3 tree contracts 41 files". Both are left as
> written (the ledger is append-only and they record what was true when measured); this note is the
> correction.

### 1.1 Deviations from the recipe's tree, and why

| id | Deviation | Reason |
|---|---|---|
| `[DEV-1]` | **No `BRINGUP_RECIPE.md` inside the package.** `BRINGUP_RECIPE.md:989` lists it. | The recipe for this bring-up lives in the kit (`models/demos/common/bringup/BRINGUP_RECIPE.md`) and this session may not modify anything outside the package, so a copy would immediately fork. `scripts/verify_citations.py`'s `DOC_PREFIXES["BRINGUP_RECIPE.md"]` points at the kit copy and the doc pass scans it. `DEC-002`, re-affirmed by `DEC-017`. |
| `[DEV-2]` | **No `utils/` package.** `BRINGUP_RECIPE.md:1021-1023` lists `utils/general_utils.py` + `utils/substate.py`. | `get_cache_file_name`, `cache_file_exists`, `get_default_num_links` and `substate` are **imported** from `models/demos/gpt_oss_d_p/utils/` — agent-contract rule 4 ("reuse means *import*, not copy-paste"). `DEC-013`, `DEC-018`. |
| `[DEV-3]` | **No `docs/`, no `scripts/__init__.py`.** Both were created by P0's literal step 1 (`BRINGUP_RECIPE.md:720`); neither appears in the P3 tree. | `DEC-002` flagged the recipe's self-conflict and deferred it here. `DEC-017` commits to the P3 spelling; both are deleted in **P5.1**, the first phase that touches the tree. Neither template ships them (`models/demos/gpt_oss_d_p/scripts/`, `models/demos/minimax_m3/scripts/` have no `__init__.py`). |
| `[DEV-4]` | **`tt/runners/adapters/__init__.py` added.** Not in the recipe tree. | Required for `models.demos.llama31_8b_d_p.tt.runners.adapters.llama` to be importable by the registry (`models/demos/common/prefill/adapter.py:277`). The template has it: `models/demos/gpt_oss_d_p/tt/runners/adapters/__init__.py`. Not a judgement call; an omission in the recipe's tree. |
| `[DEV-5]` | **`tt/embedding.py` and `tt/lm_head.py` are kept as separate files** even though the nearest template inlines both into `Model.__init__` (`models/demos/gpt_oss_d_p/tt/model.py:84` embedding, `:134` lm_head). | The recipe's tree lists them (`BRINGUP_RECIPE.md:1010-1011`) and P9 item 9 requires every `tt/` module to own a test. Two small files with two tests beat 90 inline lines in `model.py` that no test can reach directly. |
| `[DEV-6]` | **`tests/unit/test_ccl_semaphores.py` is created in P5.1**, not P8, and holds only the device-free/one-card half until then. | `G-MESH` (P5.1) already requires "allocates its semaphores exactly once (assert the list lengths, and again after dozens of getter cycles)" (`BRINGUP_RECIPE.md:1265-1266`) — which is `G-SEMAPHORE`'s assertion. Writing it once, in the file that owns the gate, avoids the same assertion existing twice. |

Everything else is the recipe's tree verbatim, including the deliberate absence of a `reference/`
package (`BRINGUP_RECIPE.md:1064`).

---

## 2. Per-file contract

Shapes are **per chip** at the deployment target `(4,8)`, TP=8 on the columns, SP=4 on the rows
(`00_MODEL_CARD.md` §4). `S` = the chunk's global token count, `S_loc = S/SP = S/4`. Layout is
`TILE` unless stated. `hf` = the normalised config dict (`DEC-020`).

### 2.1 `tt/config.py` — `MeshConfig`

- **Responsibility.** Owns the parallelism decision and the three collective wrappers. TP is the only
  knob; SP is derived from the other axis. No module ever calls `ttnn.experimental.*` directly.
- **Interface.**
  ```python
  class MeshConfig:
      def __init__(self, mesh_shape, tp, tp_axis: int = 1)
      @property
      def sp(self) -> int                                     # mesh_shape[sp_axis]
      def shard_mapper(self, mesh_device, tensor_dim=None, mesh_dims=None)
      def column_parallel(self, mesh_device)                  # shard dim -1
      def row_parallel(self, mesh_device)                     # shard dim -2
      def sequence_parallel(self, mesh_device)                # shard dim -3
      def shard_size(self, total_size) -> int                 # total_size // tp
      def allreduce(self, tensor, ccl_manager, memory_config=None, pad_size=None, axis=0)
      def allgather(self, tensor, ccl_manager, memory_config=None, axis=0, dim=3, linear=False)
      def reduce_scatter(self, tensor, ccl_manager, dim=3, axis=0, memory_config=None)
  ```
- **Template.** The **union** of the two in-repo copies, neither of which is a superset:
  `models/demos/minimax_m3/config.py:21` (has `reduce_scatter:155`, lacks the `sp` property and the
  sub-axis refusal) and `models/demos/gpt_oss_d_p/tt/config.py:19` (has `sp:56` and the sub-axis
  refusal at `:44`, lacks `reduce_scatter`). `_VALIDATED_MESH_SHAPE`/`_VALIDATED_TP` become `(4, 8)`
  / `8` — which is what `models/demos/gpt_oss_d_p/tt/config.py:15-16` already pins.
- **Shapes.** Host-side only; it produces `ttnn.ShardTensor2dMesh` mappers and passes tensors through.
- **Refusal contract** (`G-MESH`): `tp != mesh_shape[tp_axis]` **raises** — the gpt-oss rule at
  `models/demos/gpt_oss_d_p/tt/config.py:44`, not minimax's weaker check at
  `models/demos/minimax_m3/config.py:42-45`, which only raises on `tp > axis_size` or
  `total_devices % tp` and lets `(1,8)`/`tp=4` through with a warning (`:46-50`). See §5.1 for the recipe wording this resolves.

### 2.2 `tt/ccl.py` — `CCLManager`

- **Responsibility.** Allocates every persistent CCL resource **once** and hands it out with a
  ping-pong index, so back-to-back collectives never reuse a semaphore that may still be in flight.
- **Interface.**
  ```python
  class CCLManager:
      def __init__(self, mesh_device, num_links, topology=ttnn.Topology.Ring)
      def get_rs_ping_pong_semaphore(self)      # 3 semaphores per call, 2-deep
      def get_ag_ping_pong_semaphore(self)      # 2 semaphores per call, 2-deep
      def get_barrier_semaphore(self)           # 1 semaphore per call, 2-deep
      def get_ring_gather_buffer(self, key, n_kv, seq, head_dim, dtype)
      def reset_global_semaphores(self)
      # attributes: mesh_device, num_links, topology, compute_grid_size, ccl_cores,
      #             ccl_sub_device_id, ring_attention_ccl_semaphore_handles,
      #             ring_attention_ccl_core_grid_offset
  ```
- **Template.** `models/demos/gpt_oss_d_p/tt/ccl.py:17` (139 lines), taken essentially whole. Two
  properties are load-bearing and must not be "improved": the CCL core range derives from
  `mesh_device.compute_with_storage_grid_size()` (`models/demos/gpt_oss_d_p/tt/ccl.py:44`), and the
  ring-attention offset is `(grid.x - 1, 0)` (`:61`). On this box the grid is (12,10), so the offset
  is `x = 11` — see §2.7 for why the SDPA program grid must *not* follow it.
- **Deletions for Llama.** None. There is no `ep_axis` and no MoE-specific state in this file, and the
  ring-gather scratch stays for P8's SP path.
- **Semaphore inventory** (asserted by `G-SEMAPHORE`): `3*2 = 6` RS + `2*2 = 4` AG + `2*1 = 2` barrier
  + `2` ring-attention = **14** global semaphores, per model, for all 32 layers
  (`models/demos/gpt_oss_d_p/tt/ccl.py:65`, `:71`, `:77`, `:84`).

### 2.3 `tt/model_config.py` — `ModelArgs` + the one normalised config

- **Responsibility.** The **single** place a config value is read, a checkpoint is loaded, a key is
  renamed, or a cache path is built. P1 trap 2 exists because the templates mix dict and object
  access; this file ends the mixing.
- **Interface.**
  ```python
  class ModelArgs:
      def __init__(self, mesh_device, *, hf_config=None, max_seq_len=..., max_batch_size=1,
                   instruct=True, cache_hf=False)
      # normalised, derived, asserted non-None at construction:
      #   hidden_size 4096 · num_hidden_layers 32 · num_attention_heads 32 · num_key_value_heads 8
      #   head_dim 128 (DERIVED: hidden_size // num_attention_heads — no such key exists) · DEC-020
      #   intermediate_size 14336 · vocab_size 128256 · rms_norm_eps 1e-05
      #   rope_theta 500000.0        <- get_rope_theta(config_json_dict)
      #   rope_scaling {...}         <- get_rope_scaling(config_json_dict)
      @staticmethod
      def load_state_dict(weights_path, dummy_weights=False, convert_to_meta_format=True) -> dict
      def weight_cache_path(self, dtype) -> Path
      def get_state_dict_prefix(self, prefix, layer_idx) -> str
  ```
- **Template.** `models/demos/gpt_oss_d_p/tt/model_config.py:30` (`load_state_dict:106`,
  `weight_cache_path:157`, `get_state_dict_prefix:175`); second opinion
  `models/demos/minimax_m3/tt/model_config.py:22`. Key mapping imports
  `models/tt_transformers/tt/load_checkpoints.py:800` `map_hf_to_meta_keys` and `:451`
  `convert_hf_qkv_to_meta_format`.
- **Three things must NOT be carried across from the template:**
  1. `models/demos/gpt_oss_d_p/tt/model_config.py:76` is the `getattr(self.hf_config, "rope_theta", …)`
     trap itself (`07_RISKS.md` R-005). Theta and scaling come from
     `models/tt_transformers/tt/common.py:165` / `:183` on the **raw `config.json` dict**, asserted
     non-`None`.
  2. `models/demos/gpt_oss_d_p/tt/model_config.py:160` defaults the weight-cache root to
     `self.model_path` — i.e. **into `$HF_MODEL`**, which is `07_RISKS.md` R-003's exact hazard
     (that directory already holds `ttnn_cache/` and `P150/` from another package). This package's
     cache root is `$TT_CACHE_PATH` and it **refuses** to fall back to the checkpoint dir.
  3. `hf_config.head_dim` (`models/demos/gpt_oss_d_p/tt/model.py:65`) — absent from Llama's config
     (`00_MODEL_CARD.md` §2); `DEC-020`.
- **Cache-path contract.** `tensor_cache_<dtype>_<mesh_shape>` under `$TT_CACHE_PATH`. The dtype +
  mesh shape are both in the path because a tilized tensor is already sharded
  (`BRINGUP_RECIPE.md:1078-1080`). Note the recipe cites
  `models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:75` for this; that function encodes arch
  and `sp x tp` but **not** the dtype — `models/demos/gpt_oss_d_p/tt/model_config.py:157` is the one
  that encodes both. This package encodes both in both places.

### 2.4 `tt/rms_norm.py` — plain RMSNorm

- **Responsibility.** `out = rms_norm(x) * weight`, eps from the config. **No Gemma `+1` fold.**
- **Interface.** `class RMSNorm(nn.Module)`:
  `__init__(mesh_device, hf, state_dict, *, tensor_cache_path=None, mesh_config=None, is_distributed=False)`;
  `forward(x)`.
- **Shapes.** in `[1, 1, S_loc, 4096]` bf16 TILE → out same. Weight: torch `[4096]` reshaped to
  `(1, 1, 128, 32)` (`4096/32 = 128`), stored **bf16 `ROW_MAJOR`**, replicated
  (`models/demos/gpt_oss_d_p/tt/rms_norm.py:27`, `:34-44`).
- **Template.** `models/demos/gpt_oss_d_p/tt/rms_norm.py:17`; the dormant distributed branch at
  `:50-92` is kept behind the constructor flag (default `False` until P8), the Gemma fold at `:25-26`
  is deleted, and the eps read at `:46` becomes a dict lookup.
- **The one required change.** `models/demos/gpt_oss_d_p/tt/rms_norm.py:94` calls `ttnn.rms_norm`
  with **no** `compute_kernel_config`. This module passes one explicitly with
  `fp32_dest_acc_en=True` — measured worth ~25x of the op's error and ~7x of the module's
  (`BRINGUP_RECIPE.md:638-644`; `DEC-014`).

### 2.5 `tt/rope.py` — llama3-scaled RoPE tables

- **Responsibility.** The **only** place theta and the scaling parameters are read, and the only place
  cos/sin tables are built. Two builders, deliberately separate.
- **Interface.**
  ```python
  def llama3_freqs(hf) -> tuple[torch.Tensor, torch.Tensor]   # imports precompute_freqs(rope_type="llama3")
  def build_prefill_rope(mesh_device, hf, seq_len, start_pos=0) -> [cos, sin]   # contiguous; asserts start_pos <= seq_len
  def build_indexed_rope(mesh_device, hf, max_seq_len, sp, sp_axis) -> [cos, sin]  # whole-cache, block-cyclic
  def build_transformation_mat(mesh_device, dtype=ttnn.bfloat16) -> ttnn.Tensor    # 32x32, replicated
  def assert_llama3_factors(hf) -> None                        # closes 07_RISKS.md R-010
  ```
- **Shapes.** contiguous cos/sin `[1, 1, S, 128]` bf16 TILE, **replicated** on the mesh
  (`models/tt_transformers/tt/common.py:542-547`). Indexed cos/sin `[1, 1, max_seq_len/SP, 128]`
  bf16 TILE, sharded on the SP axis (`models/demos/gpt_oss_d_p/tt/rope.py:115`). Transformation
  matrix `[1, 1, 32, 32]` bf16 TILE, replicated.
- **Template / imports.** The scaling math is **imported**, not rewritten:
  `models/tt_transformers/tt/common.py:489` `precompute_freqs` → `:437` `apply_scaling` →
  `:405` `compute_llama3_parameters`; table assembly `:534` `get_prefill_rot_mat`; transformation
  matrix `:562` `get_rot_transformation_mat` — **called with no arguments**, because `:564`
  reassigns `dhead = 32` regardless (P1 trap 4). Structural template for the indexed builder:
  `models/demos/gpt_oss_d_p/tt/rope.py:115`.
- **Convention.** Meta / interleaved (`ttnn.experimental.rotary_embedding_llama`), matching both
  prefill templates (`models/demos/gpt_oss_d_p/tt/attention/operations.py:87`,
  `models/demos/minimax_m3/tt/attention/operations.py:93`). The Q/K `reverse_permute` lives in
  `tt/attention/weights.py`, never at runtime (`DEC-011`).
- **Two asserts that are the module's whole point.** `assert_llama3_factors` checks the config's
  `low_freq_factor`/`high_freq_factor` against the literals hard-coded at
  `models/tt_transformers/tt/common.py:407-408` (`07_RISKS.md` R-010); and the contiguous builder
  asserts `start_pos <= seq_len`, because `models/tt_transformers/tt/common.py:525`
  `gather_cos_sin` indexes a table of length `seq_len * 2` and a chunked call would read out of
  bounds (LANDMINES: "`RuntimeError: index N is out of bounds` from inside `gather_cos_sin`").

### 2.6 `tt/mlp.py` — dense SwiGLU

- **Responsibility.** `down(silu(gate(x)) * up(x))`, no biases, with the TP collective **inside** the
  module.
- **Interface.**
  `class MLP: __init__(mesh_device, hf, state_dict, *, mesh_config, ccl_manager=None, weight_dtype=ttnn.bfloat8_b, tensor_cache_path=None, scatter_output=None)`;
  `__call__(x)`.
- **Shapes.**

  | tensor | per-chip shape | dtype | layout |
  |---|---|---|---|
  | in `x` | `[1, 1, S_loc, 4096]` | bf16 | TILE |
  | `gate_proj` / `up_proj` weight | `[1, 1, 4096, 1792]` (`14336/8`) | bf8_b | TILE |
  | `gate` / `up` activation | `[1, 1, S_loc, 1792]` | bf16 | TILE |
  | `down_proj` weight | `[1, 1, 1792, 4096]` | bf8_b | TILE |
  | out (pre-collective, partial sum) | `[1, 1, S_loc, 4096]` | bf16 | TILE |
  | out (scheme A, post all-reduce) | `[1, 1, S_loc, 4096]` | bf16 | TILE |

- **Template.** `models/demos/minimax_m3/tt/dense_mlp.py:26` — structure taken whole: the
  column/row mapper split, the cache-only `_load` branch at `:58-72`, the `scatter_output` flag, and
  the TP tail at `:96-112`. Two deletions: the clamped `swigluoai` activation becomes plain
  `ttnn.silu(gate) * up`, and `swiglu_limit`/`alpha` disappear.
- **Deviation from the template.** `models/demos/minimax_m3/tt/dense_mlp.py:89-90` and `:94` pass **no**
  `compute_kernel_config` to `ttnn.linear`. This module passes one explicitly. On a matmul the flag's
  `True` is bit-identical to the default and `False` costs 96x-1168x
  (`BRINGUP_RECIPE.md:652-653`) — so the risk is inheriting an explicit `False`, and the defence is
  passing an explicit `True` everywhere.

### 2.7 `tt/attention/` — GQA, full RoPE, causal SDPA

Split exactly as `models/demos/gpt_oss_d_p/tt/attention/` (`__init__`, `config`, `weights`,
`operations`, `prefill`, `kv_cache`, `dense_sp`).

#### `attention/config.py`

- **Interface.**
  ```python
  @dataclass
  class AttentionConfig:
      hidden_size: int; num_heads: int; num_kv_heads: int; head_dim: int; max_seq_len: int
      rms_norm_eps: float = 1e-5
      scaling: float | None = None          # __post_init__: head_dim ** -0.5
      sequence_parallel: bool = False
      @property
      def gqa_group_size(self) -> int       # 32 // 8 = 4 globally; 4 // 1 = 4 per chip at TP=8

  @dataclass
  class ProgramConfig:
      sdpa_grid_x: int = 8; sdpa_grid_y: int = 8       # PINNED, never derived — see below
      prefill_q_chunk_size_small: int = 32; prefill_k_chunk_size_small: int = 32
      prefill_q_chunk_size_large: int = 256; prefill_k_chunk_size_large: int = 256
      prefill_threshold: int = 2048
      math_fidelity: str = "HiFi4"; math_approx_mode: bool = False
      fp32_dest_acc_en: bool = True; packer_l1_acc: bool = False
      def get_prefill_sdpa_config(self, mesh_device, seq_len) -> ttnn.SDPAProgramConfig
      def get_compute_kernel_config(self)
  ```
- **Template.** `models/demos/gpt_oss_d_p/tt/attention/config.py:23` / `:57`, with three fields
  **dropped** (`sliding_window`, `rotary_dim`, and the `layer_types` plumbing the caller does — Llama
  has none: `00_MODEL_CARD.md` §3) and one field **inverted**:
  `models/demos/gpt_oss_d_p/tt/attention/config.py:71` sets `fp32_dest_acc_en: bool = False` and
  carrying it forward costs two to three orders of magnitude (`BRINGUP_RECIPE.md:657-660`).
- **The pinned grid, and the assert that makes it fail early.** The SDPA program grid is an explicit
  named field defaulting to **8x8** (`models/demos/gpt_oss_d_p/tt/attention/config.py:96` does the
  same), and `__post_init__` asserts `sdpa_grid_x <= compute_grid.x - 1`. On this (12,10) box the CCL
  offset is `11` and `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp:421`
  asserts `ccl_core_grid_offset.x >= program_config.compute_with_storage_grid_size.x`: `11 >= 8`
  passes, a *derived* `12` fails — and it fails **only at SP > 1 in P8**, after every single-card gate
  has passed. The SP ring path gets its **own** program config (§P8), never a mutation of this one.

#### `attention/weights.py`

- **Interface.**
  ```python
  @dataclass(frozen=True)
  class AttentionWeights:
      q_proj: ttnn.Tensor; k_proj: ttnn.Tensor; v_proj: ttnn.Tensor; o_proj: ttnn.Tensor

  def load_attention_weights(mesh_device, config, state_dict, *, mesh_config,
                             weight_dtype=ttnn.bfloat8_b, tensor_cache_path=None) -> AttentionWeights
  ```
- **Shapes** (HF `[out, in]` transposed at load, never at runtime):

  | weight | HF shape | per-chip ttnn shape at TP=8 | mapper | dtype |
  |---|---|---|---|---|
  | `q_proj` | `[4096, 4096]` | `[1, 1, 4096, 512]` | `column_parallel` | bf8_b |
  | `k_proj` | `[1024, 4096]` | `[1, 1, 4096, 128]` | `column_parallel` | bf8_b |
  | `v_proj` | `[1024, 4096]` | `[1, 1, 4096, 128]` | `column_parallel` | bf8_b |
  | `o_proj` | `[4096, 4096]` | `[1, 1, 512, 4096]` | `row_parallel` | bf8_b |

  `4096/8 = 512` = 4 local Q heads x 128; `1024/8 = 128` = **1** local KV head x 128 — the equality
  that forces TP=8 (`00_MODEL_CARD.md` §4.1). All four are tile-aligned, so **no o_proj padding is
  needed** — unlike gpt-oss, whose `2880/8 = 360` forces the pad branch at
  `models/demos/gpt_oss_d_p/tt/attention/weights.py:64-70`. That branch is deleted, along with every
  bias tensor and the `sinks` tensor.
- **Template.** `models/demos/gpt_oss_d_p/tt/attention/weights.py:23` / `:38` for the loader shape,
  the mapper choice (`:145-146`) and the cache-only `None` branch (`:135-142`).
- **Meta swizzle.** `q_proj` and `k_proj` are `reverse_permute`d at load
  (`models/tt_transformers/tt/load_checkpoints.py:891`, via `:451`
  `convert_hf_qkv_to_meta_format`), **inside this loader**, so no path can reach the device
  un-swizzled. `v_proj` and `o_proj` are **not** permuted. `G-ATTN`'s negative control is exactly
  this permute omitted (expected ~0.9475).

#### `attention/operations.py`

- **Interface.**
  ```python
  def apply_qkv_projection(hidden_states, weights, compute_kernel_config)   # -> (q, k, v)
  def split_qkv_heads_prefill(q, k, v, num_heads, num_kv_heads)             # -> (Q, K, V) head-major
  def apply_rope(tensor, rope_mats, transformation_mat, *, kv_actual_global=None, cluster_axis=None)
  def concat_heads(tensor)
  def apply_output_projection(tensor, weights, activation_dtype, compute_kernel_config)
  def apply_allreduce(tensor, mesh_config, ccl_manager)
  def apply_reduce_scatter(tensor, mesh_config, ccl_manager)               # scheme B seam, P8
  ```
- **Shapes.** `[1,1,S_loc,4096] -> q [1,1,S_loc,512] / k,v [1,1,S_loc,128] -> Q [1,4,S_loc,128] /
  K,V [1,1,S_loc,128] -> (rope on Q,K only) -> SDPA out [1,4,S_loc,128] -> concat [1,1,S_loc,512] ->
  o_proj [1,1,S_loc,4096] partial -> all-reduce [1,1,S_loc,4096]`.
- **Head split (`DEC-019`).** `ttnn.experimental.nlp_create_qkv_heads(q, ttnn.concat([k, v], dim=3),
  num_q_heads=4, num_kv_heads=1, transpose_k_heads=False)`. The op takes Q separately with a **fused
  K|V** second tensor — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.cpp:22-32`
  is the two-tensor branch — which is what makes three separate projections compatible with it. The
  wrapper template is `models/demos/gpt_oss_d_p/tt/attention/operations.py:29`, which passes a single
  fully fused QKV instead.
- **RoPE application.** `ttnn.experimental.rotary_embedding_llama` (contiguous) or
  `ttnn.experimental.deepseek_prefill.rotary_embedding_indexed` (chunked, P7), dispatched exactly as
  `models/demos/gpt_oss_d_p/tt/attention/operations.py:78-89`. **Invariant asserted in the test:
  only Q and K are rotated.**
- **Deletions.** `apply_output_projection_fused_rs`
  (`models/demos/gpt_oss_d_p/tt/attention/operations.py:142`) and
  `is_shape_fused_mm_rs_supported` (`:131`) are **not** brought over: the fused matmul+RS op is
  Ring-only and gated off on Blackhole per its own comment
  (`models/demos/gpt_oss_d_p/tt/attention/prefill.py:288-292`), so it would be dead code in a
  functional-first iteration.

#### `attention/prefill.py`

- **Interface.**
  ```python
  def attention_forward(hidden_states, rope_mats, *, weights, kv_cache, config, mesh_config,
                        mesh_device, program_config, transformation_mat, ccl_manager,
                        user_id=0, batch_size=1, layer_idx=0, cached_len=0,
                        indexed_rope=False)
  ```
- **Pipeline.** qkv proj → head split → RoPE(Q,K) → KV-cache write → SDPA → concat heads → o_proj →
  TP collective. Template `models/demos/gpt_oss_d_p/tt/attention/prefill.py:51`.
- **The SDPA call**, with `sliding_window_size=` and `attention_sink=` **removed** (gpt-oss-only
  arguments):
  ```python
  ttnn.transformer.scaled_dot_product_attention(
      tt_q, tt_k, tt_v, is_causal=True, scale=config.scaling,
      program_config=program_config.get_prefill_sdpa_config(mesh_device, seq_len),
      compute_kernel_config=program_config.get_compute_kernel_config())
  ```
  GQA is native — no on-chip KV repeat. `nqh = 4`, `nkv = 1` per chip satisfies
  `TT_FATAL(nqh >= nkv && nqh % nkv == 0)` at
  `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:98`.
- **Three refusals, loud.** (a) `seq_len <= 1` → prefill only; (b) `cached_len > 0` on a single card
  → `NotImplementedError` naming `G-CHUNK-ATTN` and P8, mirroring
  `models/demos/gpt_oss_d_p/tt/attention/prefill.py:257-270` rather than silently running the wrong
  core; (c) `scatter_output=True` before scheme B is wired → refuse
  (`BRINGUP_RECIPE.md:1206-1208`).
- **Retained memory hygiene.** `deallocate(True)` after last use, and the input freed before the big
  output is allocated (`models/demos/minimax_m3/config.py:104-112` explains why).

#### `attention/kv_cache.py`

- **Interface.**
  ```python
  @dataclass
  class LlamaKVCache(KvCaches):        # models/demos/common/prefill/adapter.py:95
      k: ttnn.Tensor; v: ttnn.Tensor
      num_users: int; num_layers: int; max_seq_len: int; sp: int

  def allocate_kv_cache(mesh_device, *, num_layers, max_seq_len, sp_axis=0, num_users=1,
                        head_dim=128, cache_dtype=ttnn.bfloat8_b) -> LlamaKVCache
  def write_kv_chunk(kv_cache, tt_k, tt_v, *, slot_idx, layer_idx, kv_actual, sp_axis) -> None
  ```
- **Shapes.** Per-chip `[num_users * 32, 1, max_seq_len/SP, 128]` **bf8_b** TILE, DRAM
  `NdShardSpec(shard_shape=[1, 1, 32, 128])` round-robin over the DRAM banks. Batch dim is
  user-major: `slot = user_id * num_layers + layer_idx`. K is stored **post-RoPE**, V raw.
- **Template.** `models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:48` / `:117` / `:138`, changed in
  exactly two places: `head_dim` 64 → 128 and `num_layers` 36 → 32. The
  `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32` block geometry
  (`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:27`, used at `:87`) is **kept unchanged** —
  that is what lets P10 reuse the producer's existing packed-GQA read-back instead of writing a
  fourth reader (`BRINGUP_RECIPE.md:1453-1458`).
- **Asserts kept verbatim.** `batch == 1` per call
  (`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:149`) — the op ignores the leading dim and a
  batched tensor would write only `slot_idx`; `kv_actual % 32 == 0` (`:157`); slot and layer in range
  (`:155-156`); `max_seq_len % (32 * sp) == 0` (`:77`).
- **What `G-KV` at `(1,1)` cannot prove.** The model → cache path. At TP=1 the model emits 8 local KV
  heads and the write op refuses them outright
  (`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp:230`).
  `G-KV` therefore drives the op **one synthetic head at a time**; `G-KV-TP8` (P8) owns the rest.
  `07_RISKS.md` R-001.

#### `attention/dense_sp.py`

- **P5.5 content:** a module docstring pointing at
  `models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:41` `dense_sp_attention` and a single
  `NotImplementedError` naming P8 as the owner. **Filled in P8.**
- **P8 interface (pinned now so P5's seam matches):**
  `dense_sp_attention(q, cache_k, cache_v, new_k, new_v, *, kv_actual, logical_n, n_kv, cache_global, head_dim, mesh_device, ccl_manager, program_config, compute_kernel_config, scale, cluster_axis, slot_idx, layer_idx, num_layers, write_chunk=False)`.
- **The one op in this model where `fp32_dest_acc_en=False` is mandatory**, structurally:
  `use_streaming_compute = !fp32_dest_acc_en`
  (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp:1304`) and
  the chunked path needs the streaming compute (`:1306`), so `True` is refused with a `TT_FATAL`.
  This is not a regression of the package default; it is an exception with a `TT_FATAL` behind it,
  and `G-SP-RING` records both sides of the A/B.

#### `attention/__init__.py`

- **Interface.**
  `class Attention: __init__(mesh_device, config, state_dict, *, ccl_manager, mesh_config, program_config, layer_idx, transformation_mats=None, weight_dtype=ttnn.bfloat8_b, tensor_cache_path=None)`;
  `__call__(hidden_states, rope_mats, *, kv_cache=None, user_id=0, batch_size=1, cached_len=0, indexed_rope=False)`.
- **Template.** `models/demos/gpt_oss_d_p/tt/attention/__init__.py:28`, with the `is_sliding` /
  `layer_types` logic at `:77-83` and the per-layer `dataclasses.replace` at `:84` **deleted** — every
  Llama layer is full-causal, so the config is shared across layers unmodified.

### 2.8 `tt/embedding.py`

- **Responsibility.** Token ids → hidden states, replicated table (`DEC-024`, P4).
- **Interface.** `class Embedding: __init__(mesh_device, hf, state_dict, *, mesh_config, ccl_manager=None, tensor_cache_path=None)`; `__call__(token_ids)`.
- **Shapes.** table `[1, 1, 128256, 4096]` bf16 `ROW_MAJOR`, **replicated**; in `[1,1,1,S_loc]` uint32
  ROW_MAJOR; out `[1, 1, S_loc, 4096]` **bf16** TILE (not bf8_b — `DEC-022`).
- **Template.** `models/demos/gpt_oss_d_p/tt/model.py:84-91` (the replicated `as_tensor`) and
  `:315-318` (the `ttnn.embedding` + `unsqueeze_to_4D` call); the TP-sharded alternative is
  `models/demos/minimax_m3/tt/parallel_embedding.py:80`, evaluated and not taken (`DEC-024`).

### 2.9 `tt/lm_head.py`

- **Responsibility.** Exists only so `G-MODEL` can check top-1. Prefill's product is the KV cache.
- **Interface.** `class LMHead: __init__(mesh_device, hf, state_dict, *, mesh_config, ccl_manager=None, weight_dtype=ttnn.bfloat8_b, tensor_cache_path=None)`; `__call__(x, *, gather=True)`.
- **Shapes.** weight `[1, 1, 4096, 16032]` per chip (`128256/8 = 16032`, `= 501*32`, tile-aligned)
  bf8_b TILE, `column_parallel`; in `[1,1,32,4096]` (the last-token tile); out `[1,1,32,16032]`
  → optional all-gather on the TP axis → `[1,1,32,128256]`.
- **Template.** `models/demos/gpt_oss_d_p/tt/model.py:123-142` (padded vocab + column-parallel
  mapper) and `:241` (the matmul). **No vocab padding is needed here** —
  `models/demos/gpt_oss_d_p/tt/model.py:31` `compute_per_device_vocab` exists because gpt-oss's vocab
  does not divide its mesh; 128256/8 is exact.

### 2.10 `tt/layer.py`

- **Responsibility.** `norm → attn → residual → norm → mlp → residual`, plus the per-layer delta
  probe.
- **Interface.**
  `class DecoderLayer: __init__(mesh_device, hf, state_dict, layer_idx, *, ccl_manager, mesh_config, dtype=ttnn.bfloat16, tensor_cache_path=None, transformation_mats=None, max_seq_len=1024, max_local_batch_size=1, sequence_parallel=False)`;
  `__call__(hidden_states, position_embeddings=None, *, kv_cache=None, user_id=0, batch_size=1, cached_len=0, indexed_rope=False)`.
- **Shapes.** in/out `[1, 1, S_loc, 4096]` bf16 TILE. Sub-dict splitting is the caller's job via
  `substate(state_dict, "self_attn")` / `"mlp"` / `"input_layernorm"` /
  `"post_attention_layernorm"` (`models/demos/gpt_oss_d_p/utils/substate.py:15`).
- **Template.** `models/demos/gpt_oss_d_p/tt/layer.py:46`, forward `:126`. Deleted: the MoE branch
  and the `layer_types` plumbing. **Kept:** `_delta_stats` (`:22`) behind one env var, and the
  `ttnn.move(hidden_states)` guard for `seqlen > 32*1024` (`:138-140`) plus the eager
  `deallocate(True)` calls — both load-bearing under long-context DRAM pressure.
- **Env var.** `LLAMA_DELTA_PROBE` (`DEC-023`), documented in the README's table at P9. Its output
  goes to `bringup_log/raw/`. This is the one place a `try/except Exception` is allowed
  (`BRINGUP_RECIPE.md:1708-1710`) because a probe must never break a run — and it logs.

### 2.11 `tt/model.py`

- **Responsibility.** `embedding → [DecoderLayer] * 32 → final norm → (lm_head)`, and the
  engine-facing surface.
- **Interface** (matching both templates so P10 is wiring):
  ```python
  class Model:
      def __init__(self, mesh_device, hf, state_dict, *, ccl_manager, mesh_config=None,
                   dtype=ttnn.bfloat16, tensor_cache_path=None, max_local_batch_size=1,
                   max_seq_len=128*1024, n_layers=None, with_lm_head=True,
                   sequence_parallel=False)
      def prepare_inputs_prefill(self, tokens, start_pos=0, batch_size=1, user_id=0, **kw)
      def prefill_forward(self, x, rot_mats_global=None, *, user_id=0, get_last_token=-1,
                          kv_cache=None, batch_size=1, skip_lm_head=False,
                          on_layer_complete=None, cached_len=0, indexed_rope=False)
      def process_output_prefill(self, tt_out, last_token_idx)
  ```
- **Shapes.** `prepare_inputs_prefill`: torch `[1, S]` → device `[1, 1, S_loc, 4096]` bf16 TILE
  (token ids SP-sharded on dim 3 across the rows, replicated across the TP cols —
  `models/demos/gpt_oss_d_p/tt/model.py:288-306`). `prefill_forward` → `[1,1,S_loc,4096]`
  (`skip_lm_head=True`) or `[1,1,32,16032]` logits. `process_output_prefill` → torch
  `[..., 128256]`.
- **Template.** `models/demos/gpt_oss_d_p/tt/model.py:41` (`_forward_layers_and_head:179`,
  `prefill_forward:246`, `prepare_inputs_prefill:279`, `process_output_prefill:322`); second opinion
  `models/demos/minimax_m3/tt/model.py:87`. Deleted: the MoE/EP knobs, the hybrid layer schedule, and
  the optional on-device sampling (`models/demos/gpt_oss_d_p/tt/model.py:145-157`) — a decode
  feature, and decode is an explicit non-goal.
- **`n_layers` is a required parameter**, not a nicety: `G-MODEL` runs at 2 and 4 layers before 32
  (`BRINGUP_RECIPE.md:1559-1560`), and `with_lm_head=True` is the **default** so the top-1 half of
  that gate is never conditional.
- **`on_layer_complete(layer_idx)`** is preserved from the template (`:196`, called at `:210-211`): it is the seam P10's
  per-layer KV migration hooks attach to.

### 2.12 `tt/tt_prefill_runtime.py`

- **Responsibility.** The chunked-prefill runtime, written to the engine's contract from the start.
  **It does not own the KV cache** — the engine allocates it and passes it in.
- **Interface** (audited against the engine's real call site by `G-RUNTIME`, not against the doc):
  ```python
  @dataclass
  class TtPrefillRuntimeConfig:
      chunk_size: int; max_seq_len: int; first_layer_idx: int
      is_first_rank: bool; is_last_rank: bool
      # + sp_factor / tp_factor properties (models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:88, :92)

  class TtPrefillRuntime:
      def __init__(self, mesh_device, hf_config, state_dict, config)
      def compile(self, kv_caches=None) -> None
      def make_chunk_input(self, token_ids, chunk_size=None) -> ttnn.Tensor
      def prefill_chunk(self, input, kv_cache, *, slot_id, actual_start, actual_end,
                        request_id=0, d2h_service=..., metadata_msg=...)
      # optional migration hooks (P10):
      def set_layer_ack_channel(self, layer_ack_channel) -> None
      def kv_migration_base_address(self, kv_caches) -> int
      def build_kv_chunk_table(self, ...) -> bytes
  ```
- **`d2h_service` and `metadata_msg` are in the signature from day one.** The contract doc's §2 omits
  them; the engine passes both on every chunk
  (`models/demos/common/prefill/runners/prefill_runner.py:364`). A runtime written to the prose dies
  with a `TypeError` on its **first served chunk**, after the mesh is open and the weights are loaded.
- **Shapes.** `make_chunk_input(token_ids)` → `[1, 1, 1, chunk_size/SP]` uint32 ROW_MAJOR per chip;
  `prefill_chunk` writes into the caller's `LlamaKVCache` and returns the engine's expected
  per-chunk result. RoPE is the **indexed** whole-cache table built once in `__init__`
  (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:174`).
- **Template.** `models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:96`. **Do not carry `:185`** — it
  is the second live instance of the `getattr(self.hf_config, "rope_theta", 150000.0)` trap
  (`07_RISKS.md` R-005).

### 2.13 `tt/runners/` (P10)

| file | responsibility | interface | template |
|---|---|---|---|
| `adapters/llama.py` | `PrefillModelAdapter` subclass; the engine's entry point. Import-light: no torch, no ttnn, no reference model at module scope (`G-ADAPTER` measures it). | `class LlamaPrefillAdapter(PrefillModelAdapter)` with `name = "llama31_8b_d_p"`, `model_config`, `hf_model_default`, `ttnn_cache_default`, `prefill_trace_default`, `l1_small_size`, `supports_dflash = False`; methods `load_hf_config()`, `weight_cache_path(mesh_shape)`, `allocate_kv_cache(*, mesh_device, hf_config, params)`, `build_runtime(*, mesh_device, hf_config, params)` | `models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:41`, attributes `:45-49` |
| `manifests/llama31_8b_d_p.json` | `{"env": {"PREFILL_MODEL": "llama31_8b_d_p"}}` plus the mesh-graph descriptor and fabric config the deployment needs — and nothing that belongs to the caller | JSON | `models/demos/gpt_oss_d_p/tt/runners/manifests/gpt_oss_d_p.json` |
| `kv_chunk_table.py` | Block-cyclic KV address table + protobuf serialisation. Anything unimplemented (the multi-rank merge) **raises**, naming its risk id. | `build_kv_chunk_address_table(...)`, `build_and_serialize_kv_chunk_table(...)` | `models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:66`, `:179` |

`load_hf_config` must return a **mutable** config: the engine assigns `max_seq_len` on the line after
it returns (`models/demos/common/prefill/runners/prefill_runner.py:477`), so a frozen dataclass
raises `FrozenInstanceError` at runner startup. Knobs come from `params`
(`models/demos/common/prefill/adapter.py:46` `PrefillRunParams`), **never** from `os.environ`.

### 2.14 `scripts/`

| file | responsibility | interface | template |
|---|---|---|---|
| `verify_citations.py` | Re-resolve every `path:line` in the logs, the kit recipe and the package's own docstrings. Extended every phase. | `CITES` list + `DOCS` glob; exit 0 iff clean | `models/demos/common/bringup/examples/verify_citations.py` (+ `DEC-003`) |
| `generate_golden_kv_cache.py` | Run the HF reference in **fp32**, one `LlamaDecoderLayer` at a time via mmap, and write per-layer post-RoPE K / raw V. Golden stored at **fp32**, not the template's bf16. | `main()`; output `{trace_dir}/metadata.json` + `{trace_dir}/kv_cache/layer_<i>.safetensors` with `key_cache_layer_<i>` / `value_cache_layer_<i>` of shape `[1, 8, S, 128]` in **HF layout** | `models/demos/minimax_m3/scripts/generate_golden_kv_cache.py:195` |
| `verify_golden_kv.py` | Structural check over all 32 layers; per-layer min/mean PCC table. **Imports no ttnn.** | `verify_trace(trace_dir)`; non-zero exit on a zeroed or deleted layer | `models/demos/minimax_m3/scripts/verify_golden_kv.py:26` |

The trace directory comes from **`$PREFILL_TRACE_DIR`** — the engine already owns that variable, so
the package does not invent one (`BRINGUP_RECIPE.md:1590-1591`).

### 2.15 `tests/`

`tests/test_factory.py` and `conftest.py` exist (P1). Two additions are already scheduled:
`TestFactory.setup_test(mesh_device, ...)` building `MeshConfig` + `CCLManager` lands in **P5.1**, in
the same edit that creates them (`DEC-008`); and `parametrize_mesh_with_fabric`-style
submesh parametrisation lands in **P8** — but as `create_submesh` from a full `(4,8)`, **not** as the
template's top-level partial mesh (`models/demos/minimax_m3/tests/test_factory.py:89`), which
fabric-times-out on this galaxy (`BRINGUP_RECIPE.md:1708-1728`).

Every `tests/unit/test_*_vs_ref.py` has the same five-part shape, and a file missing any part is
incomplete: identical random weights on both sides, an **fp32** reference in the test file, a
**computed noise floor** via `tests/test_factory.py::quantize_like_device`, a **negative control**
that the same assertion must reject, and the recorded context (input distribution + reference dtype
policy). Template: `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py`
(`(1,1)` parametrize at `:149`, `convert_hf_qkv_to_meta_format` at `:197`,
`get_rot_transformation_mat()` at `:213`, `comp_pcc` at `:258`).

#### Per-test-file contract

Every row: the gate it owns, the mesh parametrisation, the reference it compares against, and the
**negative control** — the deliberately wrong variant the same assertion must reject. A row without a
control is not a gate test (`BRINGUP_RECIPE.md:350-355`); expected control values, where the recipe
measured one, are in brackets.

| file | gate | mesh | reference | negative control |
|---|---|---|---|---|
| `test_reference_model.py` | `G-REF` | none (host) | HF `LlamaDecoderLayer`, fp32 | five, incl. wrong theta [0.87437], GQA `repeat` [0.55396] — **exists** |
| `test_mesh_config.py` | `G-MESH` | (a) none, (b) (1,1) | exact arithmetic (`shard_size(4096)=512`, `shard_size(14336)=1792`) | sub-axis TP `MeshConfig((1,8), tp=4)` must **raise** (via `expect_error`) |
| `test_ccl_semaphores.py` | `G-SEMAPHORE` | (1,1); (4,8) in P8 | the constants 6/4/2/2 | list lengths after dozens of getter cycles must not be `n_layers x` the constant |
| `test_rms_norm_vs_ref.py` | `G-RMS` | (1,1) | in-test fp32 `rms_norm`, **fp32** weight | zero-gain probe: `max|out| = 0.0` (a Gemma `1+w` fold would return the normalised input) |
| `test_rope_vs_ref.py` | `G-ROPE` | (1,1) | HF `rotate_half` on the permuted input | HF-layout tensor into the Meta op must collapse [0.01296]; and scaled != unscaled `inv_freq` past 8192 |
| `test_mlp_vs_ref.py` | `G-MLP` | (1,1) | in-test fp32 SwiGLU | SiLU on `up` instead of `gate` must collapse [0.6462] |
| `test_attention_vs_ref.py` | `G-ATTN` | (1,1) | in-test fp32 attention, explicit causal mask, `repeat_interleave` KV | Q/K loaded **without** the Meta `reverse_permute` [0.9475] |
| `test_kv_cache_vs_ref.py` | `G-KV` | (1,1) | torch post-RoPE K / raw V | positional read-back **bit-exact** (`rtol=atol=0`), values <= 256; pad tail, other slot, earlier chunk all unchanged |
| `test_kv_cache_tp8.py` | `G-KV-TP8` | (1,8) submesh | fp32 golden + head->column identity | read column `(c+1)%8` as head `c` — must fail on **bit-equality** (it still scores PCC 0.99890) |
| `test_attention_chunked_vs_ref.py` | `G-CHUNK` | (1,1) | the one-shot producer on identical hidden states, + the fp32 golden | rope every chunk at `kv_actual_global = 0` [0.706 / 0.655] |
| `test_sp_attention_chunked.py` | `G-SP-RING`, `G-CHUNK-ATTN` | (4,8) | fp32 torch on the same values | `fp32_dest_acc_en=True` must be refused with a `TT_FATAL` (recorded verbatim) |
| `test_decoder_layer_vs_ref.py` | `G-LAYER` | (1,1) | in-test fp32 layer or `reference_decoder` | swap the two norm gains [0.9471] |
| `test_embedding_vs_ref.py` | — | (1,1) | `torch.nn.functional.embedding`, fp32 | a shifted token id must change the row |
| `test_lm_head_vs_ref.py` | — | (1,1) | fp32 matmul | rotate the vocab shard order |
| `test_weight_loading.py` | `G-WEIGHTS` (+P8 ext) | (1,1); (1,8) in P8 | the checkpoint itself, **bit-exact** through transpose + swizzle + dtype ladder | bypass `map_hf_to_meta_keys` — every key must go missing |
| `test_tp_parity.py` | `G-TP-PARITY` | (1,1) vs (1,2)/(1,4)/(1,8)/(2,8)/(4,8) | the **(1,1) device output**, not torch | rotate the reference by one TP shard [<= 0.95] |
| `test_model_vs_ref.py` | `G-MODEL` | (1,1) | HF with the same weights; in-test fp32 must match HF at PCC 1.0 | rotate the per-layer weights [0.1612]; plus a direct causality check |
| `test_prefill_runtime_chunked.py` | `G-RUNTIME` | none | an AST walk over the engine's real call site | the audit itself gets a control: a deliberately renamed parameter must be reported |
| `test_prefill_adapter.py` | `G-ADAPTER` | none | `config.json` and the registry | a heavy module in `sys.modules` after import must fail the assertion |
| `test_kv_chunk_table.py` | `G-KV-TABLE` | (4,8) | DRAM read back over UMD, **`torch.equal`** | read one head through another head's config |
| `fabric_topology_matrix.py` | `G-FABRIC-MATRIX` | (1,2)/(1,4)/(1,8)/(2,8)/(4,8) submeshes, subprocess-isolated with a timeout | each case's **stated expectation** | the cases expected to hang or fail are part of the matrix — a case that unexpectedly *passes* fails the gate |
| `galaxy_prefill_kv_pcc.py` | `G-MESH-KV`, `G-RACE` | (4,8) | fp32 golden, per layer | `G-RACE` is its own control: 3 runs in one process on one `CCLManager` must be **bit-identical** |

**`pytest.raises` is forbidden in `tests/`** by the repo's `prefer-expect-error` hook
(`.pre-commit-config.yaml:51`). Every refusal test takes the repo-root `expect_error` fixture
(`conftest.py:948`) with a **mandatory** message substring. This affects `G-MESH` (sub-axis TP),
`G-RUNTIME` (nine refusals), `G-SP-RING` (the `fp32_dest_acc_en=True` `TT_FATAL`) and the
`scatter_output=True` refusal — i.e. four gates whose whole content is a refusal.

### 2.16 `README.md` (P9)

- **Responsibility.** The package's front door, and a `G-CLEAN` deliverable rather than a courtesy.
- **Required contents** (`BRINGUP_RECIPE.md:2030-2032`): the architecture table, the deployment path,
  a **status table with measured PCC** (`G-MESH-KV`'s per-layer min K/V per run configuration), the
  run commands, the **env-var table** (four entries — `DEC-023`), a layout section, the "why not
  `models/common/`" answer (`02_SURVEY.md` §2, with both its citations), and a "what is **not**
  implemented" section (decode, perf, trace/2CQ, multi-galaxy PP, quantised weights, multi-user).
- **Template.** `models/demos/minimax_m3/README.md` (its "Status" table is the format the status
  section must follow); `models/demos/gpt_oss_d_p/README.md:67-86` for the shapes-and-notes section.
- **Shapes.** n/a.

---

## 3. Per-layer tensor shapes at `(4, 8)`, TP=8, SP=4

Filled with real numbers, per chip, per layer, one prefill chunk. `S_loc = S/4`.
Format follows `models/demos/gpt_oss_d_p/README.md:71-77`.

| tensor | shape (per chip) | dtype | layout |
|---|---|---|---|
| hidden in / residual / hidden out | `[1, 1, S_loc, 4096]` | bf16 | TILE |
| norm weight (`input_layernorm`, `post_attention_layernorm`) | `[1, 1, 128, 32]` | bf16 | ROW_MAJOR |
| `q_proj` weight / `o_proj` weight | `[1, 1, 4096, 512]` / `[1, 1, 512, 4096]` | bf8_b | TILE |
| `k_proj` / `v_proj` weight | `[1, 1, 4096, 128]` each | bf8_b | TILE |
| Q (post head split, post RoPE) | `[1, 4, S_loc, 128]` (4 of 32 Q heads) | bf16 | TILE |
| K, V (post head split; K post-RoPE) | `[1, 1, S_loc, 128]` (**1** of 8 KV heads) | bf16 | TILE |
| RoPE cos/sin, contiguous (one-shot) | `[1, 1, S, 128]`, replicated | bf16 | TILE |
| RoPE cos/sin, indexed (chunked) | `[1, 1, max_seq_len/4, 128]`, SP-sharded | bf16 | TILE |
| RoPE transformation matrix | `[1, 1, 32, 32]`, replicated | bf16 | TILE |
| KV cache (K and V, one each) | `[num_users*32, 1, max_seq_len/4, 128]`, block-cyclic, DRAM shard `[1,1,32,128]` | **bf8_b** (`DEC-021`) | TILE |
| SDPA out / attn out pre-`o_proj` | `[1, 4, S_loc, 128]` → concat → `[1, 1, S_loc, 512]` | bf16 | TILE |
| attn out post-`o_proj`, pre-collective (partial sum) | `[1, 1, S_loc, 4096]` | bf16 | TILE |
| MLP gate/up weight | `[1, 1, 4096, 1792]` each (`14336/8`) | bf8_b | TILE |
| MLP gate/up activation | `[1, 1, S_loc, 1792]` | bf16 | TILE |
| MLP down weight | `[1, 1, 1792, 4096]` | bf8_b | TILE |
| embedding table | `[1, 1, 128256, 4096]`, **replicated** (`DEC-024`) | bf16 | ROW_MAJOR |
| `lm_head` weight | `[1, 1, 4096, 16032]` (`128256/8`) | bf8_b | TILE |
| logits (last-token tile, pre-gather) | `[1, 1, 32, 16032]` | bf8_b | TILE |

Every sharded dim is a multiple of 32 (`00_MODEL_CARD.md` §4.2): `512`, `128`, `1792`, `16032`. The
residual row is `[1,1,S_loc,4096]` and **not** `[1,1,S_loc,512]` because scheme **A** is the choice
(`04_CCL_PLAN.md`, `DEC-025`); the `scatter_output` seam that would make it 512 is wired from day one
but refuses until P8.

---

## 4. Gate → owner map

Every one of the 32 gate rows in Appendix A (`BRINGUP_RECIPE.md:1743-1776`), with the thing in the
tree that owns it.
Added in the same edit as its owner; an unowned gate silently becomes a `NOT-RUN`.

| Gate | Phase | Owner in the tree | Kind |
|---|---|---|---|
| `G-CARD` | P0 | `bringup_log/00_MODEL_CARD.md` + `scripts/verify_citations.py` | doc |
| `G-REF` | P1 | `tests/unit/test_reference_model.py` | pytest |
| `G-SURVEY` | P2 | `bringup_log/02_SURVEY.md` + `scripts/verify_citations.py` | doc |
| `G-OUTLINE` | P3 | **this file** + `scripts/verify_citations.py` | doc |
| `G-CCL-PLAN` | P4 | `bringup_log/04_CCL_PLAN.md` + `scripts/verify_citations.py` | doc |
| `G-MESH` | P5.1 | `tests/unit/test_mesh_config.py` | pytest |
| `G-RMS` | P5.2 | `tests/unit/test_rms_norm_vs_ref.py` | pytest |
| `G-ROPE` | P5.3 | `tests/unit/test_rope_vs_ref.py` | pytest |
| `G-MLP` | P5.4 | `tests/unit/test_mlp_vs_ref.py` | pytest |
| `G-ATTN` | P5.5 | `tests/unit/test_attention_vs_ref.py` | pytest |
| `G-KV` | P5.6 | `tests/unit/test_kv_cache_vs_ref.py` | pytest |
| `G-LAYER` | P6.1 | `tests/unit/test_decoder_layer_vs_ref.py` | pytest |
| `G-WEIGHTS` | P6.2 | `tests/unit/test_weight_loading.py` | pytest |
| `G-MODEL` | P6.3 | `tests/unit/test_model_vs_ref.py` | pytest |
| `G-CHUNK` | P7 | `tests/unit/test_attention_chunked_vs_ref.py` | pytest |
| `G-GOLDEN` | P7 | `scripts/verify_golden_kv.py` (+ `scripts/generate_golden_kv_cache.py`) | script, exit code |
| `G-RUNTIME` | P7 | `tests/unit/test_prefill_runtime_chunked.py` (AST walk over `models/demos/common/prefill/runners/prefill_runner.py`) | pytest |
| `G-FABRIC-MATRIX` | P8 | `tests/fabric_topology_matrix.py` | harness, subprocess-isolated |
| `G-KV-TP8` | P8 | `tests/unit/test_kv_cache_tp8.py` | pytest |
| `G-SP-RING` | P8 | `tests/unit/test_sp_attention_chunked.py` | pytest |
| `G-CHUNK-ATTN` | P8 | `tests/unit/test_sp_attention_chunked.py` | pytest |
| `G-TP-PARITY` | P8 | `tests/unit/test_tp_parity.py` | pytest |
| `G-RACE` | P8 | `tests/galaxy_prefill_kv_pcc.py` (3 runs, one process, one `CCLManager`) | harness |
| `G-SEMAPHORE` | P8 | `tests/unit/test_ccl_semaphores.py` | pytest |
| `G-MESH-KV` | P8 | `tests/galaxy_prefill_kv_pcc.py` | harness |
| `G-WEIGHTS` (P8 ext) | P8 | `tests/unit/test_weight_loading.py`, TP=8 parametrisation | pytest |
| `G-ADAPTER` | P10 | `tests/unit/test_prefill_adapter.py` | pytest |
| `G-REQUEST` | P10 | `bringup_log/raw/G-REQUEST_<ts>.log` — verbatim two-terminal transcript | transcript |
| `G-MOCK-MIG` | P10 | `bringup_log/raw/G-MOCK-MIG_<ts>.log` — verbatim two-terminal transcript | transcript |
| `G-KV-TABLE` | P10 | `tests/unit/test_kv_chunk_table.py` | pytest |
| `G-LOOPBACK` | P10 | `bringup_log/raw/G-LOOPBACK_<ts>.log` (or a `DEC` scoping it out + a named risk) | transcript |
| `G-CLEAN` | P9 | `README.md` + `scripts/verify_citations.py` + the 11-item sweep recorded in `06_GATES.md` | sweep |
| per-phase regression | every | `pytest models/demos/llama31_8b_d_p -q`, 0 failed | pytest |

**32 Appendix A gate rows + 1 per-phase regression gate; 32/32 owned.** The four the recipe warns are easiest to
leave unowned — `G-MESH`, `G-SEMAPHORE`, `G-WEIGHTS`, `G-TP-PARITY` (`BRINGUP_RECIPE.md:1058-1060`) —
each have a dedicated file above, because no `test_<module>_vs_ref.py` naturally covers them.

Two files in the tree own **no** Appendix A gate and are there for the P9 test-inventory item
(`BRINGUP_RECIPE.md:2037-2038`, "every `tt/` module has a corresponding test"):
`tests/unit/test_embedding_vs_ref.py` and `tests/unit/test_lm_head_vs_ref.py`. Both are ordinary
`_vs_ref` PCC tests with a floor and a control; they are simply not gates.

---

## 5. Conventions honoured, and the two the templates do not actually follow

`BRINGUP_RECIPE.md:1066-1090` lists eight conventions as "all observed in the templates". Six are;
two are not, and pretending otherwise would make the outline's signatures wrong.

| Convention | Status here |
|---|---|
| Module signature `(mesh_device, hf_config, state_dict, ...)` then **keyword-only** `mesh_config=`, `ccl_manager=`, `tensor_cache_path=`, `weight_dtype=` | **Adopted, and it is stricter than the templates.** `models/demos/minimax_m3/tt/dense_mlp.py:29-38` takes `mesh_config` *positionally* and orders `weight_dtype` before `tensor_cache_path`; `models/demos/gpt_oss_d_p/tt/attention/__init__.py:38-50` takes an `AttentionConfig` rather than `hf_config` and marks nothing keyword-only. This package puts a `*` in every constructor. Two exceptions, both deliberate: `Attention` takes `config: AttentionConfig` (the config *is* the model-specific normalisation, and building it per layer from the raw dict would put config parsing in 32 places), and `DecoderLayer` takes `layer_idx` positionally after `state_dict`, as the template does. |
| State-dict splitting is the caller's job, via `substate` | Adopted, imported from `models/demos/gpt_oss_d_p/utils/substate.py:15` (`DEC-013`) |
| `ttnn.as_tensor(..., cache_file_name=...)`, and **every module builds from an empty `state_dict`** when a cache path exists | Adopted. `models/demos/minimax_m3/tt/dense_mlp.py:58-72` is the exact branch shape. Mesh shape **and** dtype in the path. A module with a weight that has no cache source **fails loud** rather than running weight-free. |
| HF `[out, in]` → ttnn `[in, out]` transposed **at load** | Adopted; `weight.transpose(-1,-2).unsqueeze(0).unsqueeze(0)` (`models/demos/minimax_m3/tt/dense_mlp.py:77`) |
| Explicit `compute_kernel_config` on **every** op that accepts one | Adopted, and it is a **deviation from both templates**: `models/demos/gpt_oss_d_p/tt/rms_norm.py:94` and `models/demos/minimax_m3/tt/dense_mlp.py:89-94` pass none. `fp32_dest_acc_en=True` is the package default; the SP ring op is the single exception (§2.7). |
| Deallocate eagerly; free the big input before allocating the big output | Adopted (`models/demos/minimax_m3/config.py:104-112`) |
| Docstring names the HF anchor + the template it mirrors | Adopted. Anchors: `LlamaRMSNorm`, `LlamaMLP`, `LlamaAttention`, `LlamaDecoderLayer`, `LlamaModel`, `LlamaForCausalLM` (all in `transformers.models.llama.modeling_llama`). |
| No env-var magic beyond the README's table | Adopted. The package's total env-var surface is planned as **four**: `HF_MODEL`, `TT_CACHE_PATH`, `LLAMA_DELTA_PROBE` (`DEC-023`), `PREFILL_TOPOLOGY` (P8; the name `models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:121` already uses). `$PREFILL_TRACE_DIR` and the other `PREFILL_*` variables belong to the engine, not to this package. P9 item 6 generates the list by grep, not by hand. |

### 5.1 One recipe sentence the outline had to resolve rather than obey

`BRINGUP_RECIPE.md:1262-1264` states both that "sub-axis TP (e.g. `MeshConfig((1,8), tp=4)`)
**raises**" and that "`MeshConfig` accepts any shape whose TP **divides** the column axis". `4`
divides `8`, so the two halves of that sentence contradict each other. The outline takes the
**refusal** as the binding requirement (it is the one stated as a gate assertion, and it is what
`models/demos/gpt_oss_d_p/tt/config.py:44` implements): `tp` must **equal** `mesh_shape[tp_axis]`, and
`_VALIDATED_*` only warns. Recorded as a finding, not as a deviation — the gate assertion is
satisfied either way, and the divisibility reading is the one that would produce a wrong tensor.

---

## 6. What P3 deliberately does not settle

- **`CHUNK_SIZE` / `MAX_SEQ_LEN`.** Still `DEC-004`'s deferral, still `07_RISKS.md` R-004; the
  constraint (`% 128 == 0`, `MAX_SEQ_LEN % CHUNK_SIZE == 0`, `MAX_SEQ_LEN > CHUNK_SIZE`) is what the
  shape table above is parameterised on. P7 picks the numbers.
- **Program-config block sizes** for the matmuls. The templates tune them per M-tile
  (`models/demos/gpt_oss_d_p/tt/attention/operations.py:126-128`); a functional-first iteration takes
  the op defaults and each `DEC` for a tuned block belongs to the phase that measures it.
- **`num_users`.** 1 for the bring-up; multi-user prefill loops `slot_idx + b` at the call site
  (`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:145-152`) and is not in this iteration.
- **Whether the distributed RMSNorm branch is ever switched on.** It is only reachable under residual
  scheme B, and P4 takes scheme A. The branch exists, defaults off, and P8 owns the question.
- **Anything about decode, perf, trace/2CQ, multi-galaxy PP or quantised weights** — explicit
  non-goals (`BRINGUP_RECIPE.md:16-17`).
