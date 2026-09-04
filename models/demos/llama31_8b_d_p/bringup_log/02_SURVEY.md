# 02 — Repo survey: reuse vs write fresh

Phase P2. One row per model component, each with a decision (**import** / **adapt** / **write**),
a `path:line` citation, and a reason. This is the phase that keeps the package small: rule 4 of the
agent contract is *reuse before writing*, and **reuse means import, not copy-paste** — a copy-paste
is a `DEC`. Date (UTC): 2026-09-04. Gate: `G-SURVEY`.

## 0. What was read, in the recipe's order

`models/demos/minimax_m3/README.md`, `models/demos/gpt_oss_d_p/README.md`,
`models/demos/minimax_m3/tt/dense_mlp.py`, `models/demos/gpt_oss_d_p/tt/ccl.py`,
`models/demos/gpt_oss_d_p/tt/config.py`, `models/demos/gpt_oss_d_p/tt/rms_norm.py`,
`models/demos/gpt_oss_d_p/tt/layer.py`, `models/demos/gpt_oss_d_p/tt/attention/__init__.py`,
`models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py`, `models/tt_transformers/tests/test_mlp.py`
— plus, for the rows below, the rest of `models/demos/gpt_oss_d_p/tt/attention/`,
`models/demos/minimax_m3/config.py`, `models/common/modules/mlp/mlp_2d.py` and
`models/common/models/llama3_8b/model.py`.

Definitions used in the table:

- **import** — the package calls the existing symbol; no equivalent code is written here.
- **adapt** — a new file in this package, structurally modelled on a named template, differing for a
  reason stated in the row. Adaptation is *not* copy-paste: where a helper can be imported it is.
- **write** — no template exists; the row says why.

## 1. The table

| # | Component | Decision | Source `path:line` | Why | `DEC` |
|---|---|---|---|---|---|
| 1 | **MeshConfig** (parallelism + collective wrappers) | adapt | `models/demos/minimax_m3/config.py:21` (`allreduce:77`, `allgather:135`, `reduce_scatter:155`); `models/demos/gpt_oss_d_p/tt/config.py:19` (`_VALIDATED_MESH_SHAPE = (4, 8)` at `:15`, `_VALIDATED_TP = 8` at `:16`) | **Neither in-repo copy is a superset.** M3's has `reduce_scatter`; gpt-oss's does not, but gpt-oss's already pins exactly our `(4,8)`/TP=8 target. P4 builds the union. Cannot be imported: both live inside their own package's namespace and each hard-codes its model's validated shapes | P4 |
| 2 | **CCLManager** (sub-device, ping-pong semaphores, scratch) | adapt | `models/demos/gpt_oss_d_p/tt/ccl.py:17`, semaphore getters `:88`/`:95`/`:102`, ring-gather scratch `:108`, `reset_global_semaphores:129` and its barrier caveat at `:132` | 139 lines, fully commented, itself mirroring `models/demos/minimax_m3/tt/ccl.py`. Two properties must be preserved rather than re-derived: the CCL core range comes from `compute_with_storage_grid_size()` (this box is (12,10)), and semaphores are allocated **once** and cycled. The barrier ping-pong is only 2 deep — `G-RACE`'s first move if it fails | P4 |
| 3 | **`num_links` / cache-file naming** | **import** | `models/demos/gpt_oss_d_p/utils/general_utils.py:11` `get_cache_file_name`, `:15` `cache_file_exists`, `:27` `get_default_num_links` | 35 lines, model-agnostic, already arch-aware (`is_blackhole()` → 2 links; single-row mesh → 1, which has a P8 gate consequence). The P3 tree says "copy from gpt_oss_d_p/utils"; importing is strictly better and the recipe's own rule 4 prefers it | `DEC-013` |
| 4 | **`substate()` state-dict splitter** | **import** | `models/demos/gpt_oss_d_p/utils/substate.py:15` (`has_substate:37`, `indexed_substates:53`) | Pure dict manipulation, no model assumptions. Same reasoning as row 3 | `DEC-013` |
| 5 | **RMSNorm (plain)** | adapt | `models/demos/gpt_oss_d_p/tt/rms_norm.py:17`; distributed branch `:50-92`; single-pass call `:94`. Second opinion: `models/demos/minimax_m3/tt/rms_norm.py:30` | Llama's norm is plain, exactly the gpt-oss `use_gemma_norm=False` branch — so the *math* is a match. Three changes: drop the dead Gemma fold (Llama has no `use_gemma_norm` key), **add an explicit `compute_kernel_config`** (the template's `ttnn.rms_norm` at `:94` passes none, which §2.4 measures as ~25x error), and take a normalised dict `hf_config` rather than `hf_config.rms_norm_eps` (P1 trap 2) | `DEC-014` |
| 6 | **RoPE tables (llama3 scaling)** | adapt + import the math | `models/tt_transformers/tt/common.py:489` `precompute_freqs`, `:437` `apply_scaling` (llama3), `:534` `get_prefill_rot_mat`; theta/scaling readers `:165`/`:183`. Structural template: `models/demos/gpt_oss_d_p/tt/rope.py:36` (YaRN) | The **scaling math is imported**, not rewritten — `apply_scaling(..., rope_type="llama3")` is the repo's llama3 implementation and `G-REF` measured it bit-identical to this package's transcription (`max|Δ| = 0.0`). `tt/rope.py` is a thin adapter because gpt-oss's builder is YaRN-specific end to end. **Caveat found while reading:** `compute_llama3_parameters` hard-codes `low_freq_factor = 1` / `high_freq_factor = 4` (`models/tt_transformers/tt/common.py:407-408`) instead of reading them from the config; correct for Llama-3.1 (whose config says 1.0/4.0), silently wrong for any llama3-scaled model that differs → `07_RISKS.md` R-010 | `DEC-015` |
| 7 | **RoPE transformation matrix** | **import** | `models/tt_transformers/tt/common.py:562` `get_rot_transformation_mat`; call site to copy: `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:213` | Model-agnostic 32×32 matrix. **Call it with no arguments** — `:564` reassigns `dhead = 32` immediately, so the parameter is a lie (P1 trap 4) | — |
| 8 | **Indexed RoPE (chunked prefill)** | **import** | `ttnn.experimental.deepseek_prefill.rotary_embedding_indexed`, used at `models/demos/minimax_m3/tt/attention/operations.py:85`; table builder `models/demos/gpt_oss_d_p/tt/rope.py:115` `build_indexed_rope` | A C++ device op; nothing to write. The contiguous builder has a `start_pos <= seq_len` ceiling and chunked prefill needs the indexed one (LANDMINES: `RuntimeError: index N is out of bounds` from `gather_cos_sin`). P7 | — |
| 9 | **Q/K/V + O projection weights** | adapt | `models/demos/gpt_oss_d_p/tt/attention/weights.py:23` `AttentionWeights`, `:38` `load_attention_weights` | Same three-weight column-parallel + row-parallel shape, minus **every bias** (`attention_bias: false`) and minus the sinks tensor. Deletion, not addition — see `07_RISKS.md` R-009 for why deletions are the risk here | P5.5 |
| 10 | **HF→Meta QKV swizzle / key mapping** | **import** | `models/tt_transformers/tt/load_checkpoints.py:451` `convert_hf_qkv_to_meta_format`, `:800` `map_hf_to_meta_keys`, `:891` `reverse_permute` | The canonical llama-family helpers, and Llama is the model they were written for. `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:197` shows the exact usage. `fuse_qkv_meta:494` is available but not used — three separate projections this iteration | `DEC-016` |
| 11 | **GQA head split** | **import** the op, adapt the wrapper | `models/demos/gpt_oss_d_p/tt/attention/operations.py:29` `split_qkv_heads_prefill` (over `ttnn.experimental.nlp_create_qkv_heads`) | Head counts come from the config; the wrapper is 20 lines around a ttnn op. At TP=8 the local split is 4 Q / 1 KV (`00_MODEL_CARD.md` §4.2) | P5.5 |
| 12 | **RoPE application on device** | **import** the op | `ttnn.experimental.rotary_embedding_llama`, wrapped at `models/demos/gpt_oss_d_p/tt/attention/operations.py:50` `apply_rope` | Device op + a Meta-format table. The convention split (HF halves vs Meta interleaved) is settled in `DEC-011`: the reference is HF-convention, the device gets `reverse_permute`d weights and interleaved tables | — |
| 13 | **SDPA (prefill, causal, one-shot)** | **import** the op, adapt the program config | `ttnn.transformer.scaled_dot_product_attention`, called at `models/demos/gpt_oss_d_p/tt/attention/prefill.py:34` `_run_sdpa`; program config `models/demos/gpt_oss_d_p/tt/attention/config.py:90` `get_prefill_sdpa_config` | The op is fused and shared. Two things do **not** transfer: the sinks/sliding arguments (Llama has neither), and the program grid — it must be **pinned at 8×8**, never derived from this (12,10) device grid, or SP>1 fails an assert that every single-card gate passes (P5.5, LANDMINES) | P5.5 |
| 14 | **Ring SDPA (SP path)** | **import** the op, adapt the caller | `ttnn.transformer.ring_joint_scaled_dot_product_attention`, called at `models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:106` | P8 only; `tt/attention/dense_sp.py` is a `NotImplementedError` stub until then. The one op in this model where `fp32_dest_acc_en=False` is **mandatory** rather than a preference | P8 |
| 15 | **KV cache: allocation + chunk write** | adapt | `models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:48` `allocate_kv_cache`, `:117` `_write_one`, `:138` `write_kv_chunk`; the op is `ttnn.experimental.deepseek_prefill.update_padded_kv_cache` (`:125`) | Block-cyclic, SP-sharded, one KV head per chip (`:95`, `:99`) — the constraint that forces `TP == 8`. Adapted only for `head_dim = 128` and Llama's layer count. `write_kv_chunk` handed a multi-head tensor writes **only head 0** (LANDMINES), so the per-head slicing must survive the adaptation | P5.6 |
| 16 | **Dense SwiGLU MLP** | adapt | `models/demos/minimax_m3/tt/dense_mlp.py:26` (`_load` cache-only branch `:58-72`, TP tail `:99-112`) | The single best template: column-parallel gate/up, row-parallel down, **the TP collective inside the module**, and a `scatter_output` flag wired from day one. Two deletions: M3's clamped `swigluoai` activation becomes plain `silu` (`hidden_act`), and its `swiglu_limit`/`alpha` config disappears. `models/demos/gpt_oss_d_p/tt/mlp.py:38` is **not** a candidate — it is a MoE router + expert-parallel wrapper (`:5-10`) | P5.4 |
| 17 | **Embedding** | adapt | `models/demos/minimax_m3/tt/parallel_embedding.py:80` `TtParallelEmbedding`; vocab split helper `models/demos/gpt_oss_d_p/tt/model.py:31` `compute_per_device_vocab` | Replicated vs TP-sharded vocab is a P4/P6 `DEC`; `128256/8 = 16032` is tile-aligned either way (`00_MODEL_CARD.md` §4.2). M3's supports a 2D (vocab-on-SP) split this model does not need | P6.2 |
| 18 | **LM head** | adapt | `models/demos/gpt_oss_d_p/tt/model.py:179` `_forward_layers_and_head` | Prefill's product is the KV cache, so the head exists only for `G-MODEL`'s top-1 check. `V/TP` shard + all-gather when logits are wanted | P6.3 |
| 19 | **Decoder layer** | adapt | `models/demos/gpt_oss_d_p/tt/layer.py:46` `DecoderLayer`, forward `:126`; per-layer delta probe `:22` `_delta_stats` | Structure is identical (norm → attn → residual → norm → mlp → residual); the MoE call becomes a dense MLP call, and the sliding/full alternation disappears. `_delta_stats` is worth keeping: Appendix B's first move for "a *step* in the per-layer PCC curve" | P6.1 |
| 20 | **Model** | adapt | `models/demos/gpt_oss_d_p/tt/model.py:41` `Model` (`prefill_forward:246`, `prepare_inputs_prefill:279`); second opinion `models/demos/minimax_m3/tt/model.py:87` | The engine-facing surface (`prepare_inputs_prefill` / `prefill_forward` / `process_output_prefill`) is what P10 needs; the layer loop is generic. Llama drops the hybrid layer schedule both templates carry | P6.3 |
| 21 | **`ModelArgs` / normalised `hf_config`** | adapt | `models/demos/gpt_oss_d_p/tt/model_config.py:30` (`load_state_dict:106`, `weight_cache_path:157`, `get_state_dict_prefix:175`) | Adapted, **not** copied: `:76` is the `getattr(self.hf_config, "rope_theta", …)` trap itself (`07_RISKS.md` R-005). This is the "ONE normalised hf_config constructor" P1 trap 2 requires — dict in, dims out, theta/scaling through the tt_transformers readers | P6.2 |
| 22 | **Weight tilizing cache** | **import** + convention | `models/demos/gpt_oss_d_p/utils/general_utils.py:11`; mesh-shape-and-dtype-in-path convention at `models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:75` `weight_cache_path`; completeness check `models/demos/minimax_m3/tt/weight_cache.py:49` `weight_cache_is_complete` | `ttnn.as_tensor(..., cache_file_name=...)` does the work. The convention is the load-bearing part: a tilized tensor is already sharded, so a cache written at another mesh shape is wrong at ours — `07_RISKS.md` R-003 | P6.2 |
| 23 | **Chunked-prefill runtime** | adapt | `models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py` (585 lines); second opinion `models/demos/minimax_m3/tt/tt_prefill_runtime.py` | Satisfies the engine's §2 contract, which is model-agnostic; the model-specific parts are RoPE setup and the attention call. Note `:185` carries the same `getattr(..., "rope_theta", 150000.0)` trap — do not carry it forward | P7 |
| 24 | **Prefill adapter (`common/prefill`)** | adapt | `models/demos/common/prefill/adapter.py:104` `PrefillModelAdapter` (+ `KvCaches:95`); reference subclass `models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:41` | The base class is **imported**; only the subclass is written. Registry resolves the name lazily, so the adapter module must stay import-cheap (`G-ADAPTER`) | P10 |
| 25 | **KV chunk address table** | adapt | `models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:66` `build_kv_chunk_address_table`, `:179` `build_and_serialize_kv_chunk_table` | Block-cyclic address arithmetic parameterised by layers/users/chunk bytes. Gated on **bit-equality**, never PCC (§2.5) | P10 |
| 26 | **Golden-KV generation + verification** | adapt | `models/demos/minimax_m3/scripts/generate_golden_kv_cache.py:195` `main`, `models/demos/minimax_m3/scripts/verify_golden_kv.py:26` `verify_trace`; gpt-oss has its own pair under `models/demos/gpt_oss_d_p/scripts/` | Host-only, imports no ttnn. The model-specific part is the HF loop; the trace format and the verifier are reusable. `G-GOLDEN` requires the streamed driver to match HF's own loop bit-exactly | P7 |
| 27 | **Per-module test shape** | adapt | `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py` (identical random weights `:12`, `(1,1)` parametrize `:149`, `comp_pcc` `:258`); `models/tt_transformers/tests/test_mlp.py` | The recipe's canonical gate-test shape. The kit's own `examples/module_test_vs_ref.py` is **advertised but not shipped** (`07_RISKS.md` R-008), so this file is the substitute — the recipe names it too | — |
| 28 | **PCC / allclose helpers** | **import** | `models/common/utility_functions.py:488` `comp_pcc`, `:476` `comp_allclose`, `:1043` `is_blackhole` | Every test in the tree uses them | — |
| 29 | **Noise-floor helpers** | copy (one definition) | `models/demos/common/bringup/examples/noise_floor.py:33`, `:53` → `tests/test_factory.py` | The kit's `examples/` is documentation, not an importable package (no `__init__.py`); its README says to copy it. This is the copy-paste rule 4 requires a `DEC` for | `DEC-007` |
| 30 | **Mesh graph descriptors / fabric** | **import** (config, not code) | `tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_sp4_torus_xy_graph_descriptor.textproto`, `tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto` | Both present. A Ring topology on a plain `FABRIC_1D` fabric **hangs** rather than erroring, so the torus descriptor is not optional at P8 | P8 |

Decision split over the 30 rows: **import 9** (rows 3, 4, 7, 8, 10, 12, 22, 28, 30), **adapt 16**
(1, 2, 5, 9, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27), **import-the-op-adapt-the-caller 4**
(6, 11, 13, 14), **copy-with-a-`DEC` 1** (29), **write 0**.

**There is no "write" row.** Every component of a dense GQA transformer already exists in this tree
in some form; what this bring-up does is compose and *delete*. That is the finding of the survey,
and it is why `07_RISKS.md` R-009 is about deletion rather than about missing kernels.

## 2. `models/common/` (TTTv2) — evaluated, and why it is **not** the base

The first question a reviewer asks. Answered with evidence, not taste:

- **`MLP2D`'s "2D" is 2D *tensor* parallelism, not TP × SP.** Its prefill path reduce-scatters on
  `cluster_axis = 1` (`models/common/modules/mlp/mlp_2d.py:256` `_reduce_scatter_axis1`, set at
  `:259`) and closes with an all-reduce on `cluster_axis = 0`
  (`models/common/modules/mlp/mlp_2d.py:461`, inside `_all_reduce_tg` called at `:361`). With SP on
  the row axis, that final all-reduce would sum activations belonging to **different tokens** —
  silently wrong, and it would still produce a plausible PCC on a one-row mesh. The tempting
  shortcut *"an MLP is token-pointwise, so SP looks like DP to it"* holds for the math and **not**
  for this module's collectives. Exactly the bug class a single-row gate cannot see.
- **There is no `Attention2D`.** `models/common/modules/attention/` contains `Attention1D`
  (`models/common/modules/attention/attention_1d.py:319`) and nothing else, and
  `models/common/models/llama3_8b/model.py:890` raises
  `ValueError("Llama3Transformer1D only supports 1D mesh topologies.")` when `num_devices == 32`
  — i.e. on precisely this machine.
- **No chunked-prefill runtime and no `models/demos/common/prefill` adapter** anywhere under
  `models/common/`, so P7 and P10 would be greenfield against it.

So: `models/demos/minimax_m3/tt/dense_mlp.py` is the MLP template — it collectives on the **TP axis
only**, which is what makes it SP-safe — and `models/demos/gpt_oss_d_p/tt/attention/` is the
attention template. **P9 requirement:** the package `README.md` must carry this answer.

`models/common/models/llama3_8b/` is still worth one thing: it is a complete, independently-written
Llama-3.1-8B, so it is a **third opinion** on layer wiring when a PCC is ambiguous. Same for
`models/demos/llama3_70b_galaxy/tt/llama_ccl.py` and
`models/demos/llama3_70b_galaxy/tt/distributed_norm.py` on llama-specific CCL placement — decode, and
1D, but llama.

## 3. What this package will **NOT** bring over

The anti-bloat control, one line each. Every item exists in a template that is otherwise the right
structural answer, which is exactly why it needs listing (`00_MODEL_CARD.md` §3).

| Not brought over | From | Why not |
|---|---|---|
| MoE / routed experts | `models/demos/gpt_oss_d_p/tt/moe/`, `models/demos/minimax_m3/tt/moe/` | Llama has no experts; every layer is dense |
| Router (`topk` + softmax) | `models/demos/gpt_oss_d_p/tt/moe/router.py`, `models/demos/minimax_m3/tt/topk.py` | No routing decision to make |
| EP dispatch / combine | `models/demos/deepseek_v3_d_p/tt/moe/` | No expert parallelism; EP=1 is not a degenerate case worth carrying |
| Shared expert | `models/demos/minimax_m3/tt/moe/tt_minimax_moe.py` | M3-only concept |
| `unified_routed_expert_ffn` + `swigluoai` activation | `models/demos/minimax_m3/tt/moe/activation.py` | Llama's activation is plain `silu`; the clamped OAI variant has different math |
| Attention sinks | `models/demos/gpt_oss_d_p/tt/attention/` | No `sinks` tensor in the checkpoint; a sink left at zeros still changes the softmax denominator, so this must be *removed*, not defaulted |
| Sliding window + `layer_types` alternation | `models/demos/gpt_oss_d_p/tt/attention/config.py`, `models/demos/gpt_oss_d_p/tt/layer.py` | Every Llama layer is full-causal |
| QK-norm | `models/demos/minimax_m3/tt/attention/` | Absent from the config |
| Partial RoPE (`rotary_dim < head_dim`) | `models/demos/minimax_m3/tt/attention/config.py` | Llama is full rotary |
| YaRN + mscale | `models/demos/gpt_oss_d_p/tt/rope.py:36` | Llama uses `llama3` piecewise scaling; carrying YaRN's `attention_factor` would scale every table by a wrong constant |
| MLA (latent attention) | `models/demos/deepseek_v3_d_p/` | Llama is plain GQA |
| Sparse / MSA attention + indexer | `models/demos/minimax_m3/tt/attention/msa.py` | Dense causal attention only |
| MXFP4 weight loader | `models/demos/gpt_oss_d_p/tt/moe/weights.py`, `models/demos/gpt_oss_d_p/tests/unit/test_mxfp4_loader.py` | Plain bf16 safetensors |
| Bias handling on every projection | `models/demos/gpt_oss_d_p/tt/attention/weights.py` | `attention_bias` and `mlp_bias` are both false; every bias branch is dead code here |
| Decode / paged attention / trace / 2CQ | `models/tt_transformers`, `models/demos/llama3_70b_galaxy` | Explicit non-goals for this iteration |

## 4. Genuinely missing from the repo

Nothing. Every component maps to an existing symbol or an existing template. The two places where
the map is thinnest are recorded as risks rather than as inventions:

- `07_RISKS.md` **R-007** — `update_padded_kv_cache` and `rotary_embedding_indexed` have no
  Llama-shaped (head_dim 128, full rotary, GQA 32/8) exercise upstream.
- `07_RISKS.md` **R-009** — no in-repo attention template is dense + bias-free + full-RoPE, so
  `tt/attention/` is an adaptation-by-deletion.
- `07_RISKS.md` **R-010** — `compute_llama3_parameters` hard-codes the low/high frequency factors.

No kernel needs to be written for this model.
