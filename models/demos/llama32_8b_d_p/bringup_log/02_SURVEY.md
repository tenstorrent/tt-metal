<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 02 — Repo survey: reuse vs write fresh

**Phase:** P2 · **Date (UTC):** 2026-09-03 · **Gate:** `G-SURVEY` — **PASS**

Every row carries a decision and a `path:line`. **Every citation in this document was verified
mechanically** — a script re-reads each cited file and asserts the claimed symbol is on the claimed
line: **200 citations, 200 verified, 0 mismatched** (`raw/G-SURVEY_20260903T162611Z.log`, check [5]).

That check earned its keep twice. It found **five wrong line numbers in the recipe** (§6) and
**five wrong line numbers in this document's own first draft** — the
`gpt_oss_d_p/tt/attention/weights.py` dataclass fields were off by two, the `prefill.py` SDPA kwargs
off by one, and `gpt_oss_d_p/tt/model.py`'s `ttnn.embedding` was cited at `:310` when it is at
`:315`. All ten are corrected here. An unverified `path:line` is worth *less* than no citation,
because it reads as authoritative; P3+ should keep the verifier in the loop
(`models/demos/llama32_8b_d_p/scripts/verify_citations.py` — extend its `CITES` list each phase).

Decision vocabulary:

| Decision | Means |
|---|---|
| **import** | `from models... import X`. No new code. |
| **adapt** | Copy into `llama32_8b_d_p/` and modify, because it is not importable across sibling demo packages or carries features Llama lacks. Requires a `DEC`. |
| **write** | New code. Requires either a `DEC` or a stated "no equivalent exists" finding. |

---

## 1. What was read

Recipe P2 step 1 (`BRINGUP_RECIPE.md:365-368`) names eight files. All read, plus the substrate the
recipe does not mention (§5):

| File | Lines | Read |
|---|---|---|
| `models/demos/minimax_m3/README.md` | 88 | ✅ (Layout `:71`, Status `:29`, env-var table `:62-67`, deployment path `:19`) |
| `models/demos/gpt_oss_d_p/README.md` | ~95 | ✅ (mesh `:6`, reuse-vs-fresh `:49-66`, shapes table `:67-86`, thresholds `:90-91`) |
| `models/demos/minimax_m3/tt/dense_mlp.py` | 113 | ✅ in full |
| `models/demos/gpt_oss_d_p/tt/ccl.py` | 139 | ✅ |
| `models/demos/gpt_oss_d_p/tt/config.py` | 159 | ✅ |
| `models/demos/gpt_oss_d_p/tt/rms_norm.py` | 99 | ✅ |
| `models/demos/gpt_oss_d_p/tt/layer.py` | 175 | ✅ |
| `models/demos/gpt_oss_d_p/tt/attention/__init__.py` | 150 | ✅ |
| `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py` | ~262 | ✅ |
| `models/tt_transformers/tests/test_mlp.py` | ~135 | ✅ |

---

## 2. The reuse-vs-write table

`TT` = `models/tt_transformers`, `GO` = `models/demos/gpt_oss_d_p`, `M3` = `models/demos/minimax_m3`,
`DS` = `models/demos/deepseek_v3_d_p`, `CP` = `models/demos/common/prefill`,
`CM` = `models/common`.

| # | Component | Decision | Source `path:line` | Why | DEC |
|---|---|---|---|---|---|
| 1 | **RMSNorm** (plain, no `+1` fold) | **adapt** | `GO/tt/rms_norm.py:17` (class), `:49` (`forward`), `:93-99` (plain `ttnn.rms_norm` branch, call at `:94`), `:27` (weight reshaped to `(1,1,-1,TILE_SIZE)`), `:34-44` (`as_tensor` ROW_MAJOR + `cache_file_name`) | Structurally exactly Llama's norm. Delete the `use_gemma_norm` `+1`-fold affordance (`:20-27`) — Llama has no fold (card §2). Keep the `is_distributed` branch, dormant. | `DEC-006` |
| 2 | **Distributed RMSNorm** (scheme B only) | **adapt, dormant** | `GO/tt/rms_norm.py:50` (branch), `:67` `ttnn.rms_norm_pre_all_gather`, `:70-78` `ttnn.all_gather(dim=3, cluster_axis=1)`, `:82-90` `ttnn.rms_norm_post_all_gather` | Carried but left `False` until P8 (recipe `:613`). ⚠️ **It is dead code upstream**: `GO/tt/rms_norm.py:33` pins `self.is_distributed = False  # self.mesh_config.tp > 1`. Llama would be its first real user. | `R-007` |
| 3 | **RoPE — llama3 scaled cos/sin** | **import** | `TT/tt/common.py:489` `precompute_freqs(dim, end, theta, scale_factor, orig_context_len, rope_type="llama3")`; `:437` `apply_scaling`; `:405` `compute_llama3_parameters`; `:534` `get_prefill_rot_mat(head_dim, mesh_device, seq_len, theta, scale_factor, orig_context_len, start_pos=0)`; `:525` `gather_cos_sin` | Importable across packages (`tt_transformers` is a library, not a sibling demo) and **validated against HF in P1 to `max|Δ| = 0.0`** on both cos and sin (`01_REFERENCE.md` §5). Do not rewrite. | `DEC-007` |
| 4 | **RoPE θ extraction** | **import** | `TT/tt/common.py:165` `get_rope_theta(config: dict, default=None)`; comment `:160-163`; sibling `get_rope_scaling` `:183` | Mandatory, not optional: under transformers 5.12.1 `cfg.rope_theta` is **`None`**. `R-002`. | — |
| 5 | **RoPE transformation matrix** (Meta path) | **import** | `TT/tt/common.py:562` `get_rot_transformation_mat(dhead=32)` | Needed by `ttnn.experimental.rotary_embedding_llama`. ⚠️ Call with **no argument**: `:564` hard-codes `dhead = 32`, overwriting the parameter. `R-010`. | — |
| 6 | **RoPE apply on device** | **adapt** | `GO/tt/attention/operations.py:50-52` `apply_rope(...)`; `:87-89` `ttnn.experimental.rotary_embedding_llama(t, rope_mats[0], rope_mats[1], trans_mat, is_decode_mode=)`; `:79-86` the indexed variant `ttnn.experimental.deepseek_prefill.rotary_embedding_indexed` (taken when `kv_actual_global is not None`, `:78`). Bindings: `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama_nanobind.cpp:18`; HF alternative `.../rotary_embedding_hf/rotary_embedding_hf_nanobind.cpp:18` | Take the **Meta** op: both prefill templates use it (`GO/.../operations.py:87`, `M3/tt/attention/operations.py:93`), so the surrounding scaffolding assumes it. Cost: Q/K weights must be `reverse_permute`d at load. | `DEC-007` |
| 7 | **HF→Meta Q/K permute** | **import** | `TT/tt/load_checkpoints.py:451` `convert_hf_qkv_to_meta_format(loaded_weights, head_dim)`; `:891` `reverse_permute(tensor, n_heads, dim1, dim2)` — body `:892` `tensor.view(n_heads, 2, dim1//n_heads//2, dim2).transpose(1,2).reshape(dim1, dim2)`; inverse `permute` `:895` | Already imported by the template test at `GO/tests/unit/test_attention_vs_ref.py:34`, so the import path is proven. | `DEC-008` |
| 8 | **Q/K/V + O projections** | **adapt** | `GO/tt/attention/weights.py:23` (`AttentionWeights` dataclass; fields `wqkv:31`, `wqkv_bias:32`, `o_proj:33`, `o_proj_bias:34`, `sinks:35`), `:38-46` `load_attention_weights(...)`, `:64-70` (o_proj tile padding) | Three separate weights, column/row-parallel. **Delete** `wqkv_bias` (`:32`), `o_proj_bias` (`:34`) and `sinks` (`:35`) — Llama has `attention_bias: false` (card `C:5`) and no sinks. ⚠️ GO **fuses** into `wqkv` (`:31`); recipe's `DEC-014` example prefers three separate matmuls. P3 must decide. | P3 |
| 9 | **GQA head split** | **adapt** | `GO/tt/attention/operations.py:29` `split_qkv_heads_prefill(xqkv_fused, num_heads, num_kv_heads)`, `:41` `ttnn.experimental.nlp_create_qkv_heads`; merge `:92` `concat_heads`, `:102` `nlp_concat_heads` | Model-agnostic given `(num_heads, num_kv_heads)`. Signature is fused-QKV-shaped, so it couples to row 8. | P3 |
| 10 | **SDPA — prefill, causal, GQA** | **import (op)** / **adapt (call site)** | Op: `ttnn.transformer.scaled_dot_product_attention`, binding `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp:337`, docstring shapes `:314-316`. Call site: `GO/tt/attention/prefill.py:34` `_run_sdpa`, call `:36-49` | **Recipe's open question at `:680-681` is RESOLVED: GQA is native, no on-chip KV repeat.** The only head constraint is `sdpa_device_operation.cpp:97-101` (non-paged) and `:325-329` (paged) — `TT_FATAL(nqh >= nkv && nqh % nkv == 0)`; heads read at `:61-62`; K/V counts must match each other `:89`/`:317`. At TP=8: `4 >= 1 && 4 % 1 == 0` ✓. **Delete** `sliding_window_size=` (`prefill.py:44`) and `attention_sink=` (`:45`) from the call. Keep `is_causal=True` (`:40`), `scale=config.scaling` (`:43`), `program_config=` (`:46`), `compute_kernel_config=` (`:47`). | — |
| 11 | **SDPA program config** | **adapt + fix** | `GO/tt/attention/config.py:90` `get_prefill_sdpa_config(mesh_device, seq_len)`, hard-coded `ttnn.CoreCoord(8, 8)` at `:96`; `:102` `get_compute_kernel_config()` returning `ttnn.WormholeComputeKernelConfig` at `:103` | Copy but **fix**: derive the grid from `mesh_device.compute_with_storage_grid_size()` and pick the kernel-config class by arch (`CM/utility_functions.py:1043` `is_blackhole`). Blackhole is wider than 8×8, and the ring-SDPA assert couples this grid to the CCL offset. `R-008`. | P3 |
| 12 | **KV cache alloc + chunk write** | **adapt** | `GO/tt/attention/kv_cache.py:27` `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32`; `:31` `GptOssKVCache(KvCaches)`; `:48-57` `allocate_kv_cache(mesh_device, *, num_layers, max_seq_len, sp_axis=0, num_users=1, head_dim=64, cache_dtype=ttnn.bfloat8_b)`; `:86-91` `NdShardSpec` with `shard_shape=[1,1,NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim]` at `:87`; `:82-85` core ranges; `:117` `_write_one`, `:125-133` `update_padded_kv_cache`; `:138` `write_kv_chunk` | **Zero gpt-oss baggage — pure GQA.** Only `head_dim=64` → 128. Keeping the block geometry is what lets P10 reuse the producer's existing reader instead of writing a fourth (recipe `:711-716`). | — |
| 13 | **KV write op** | **import** | `ttnn.experimental.deepseek_prefill.update_padded_kv_cache`; call sites `DS/tt/mla/mla.py:1204`, `:1215` (in `_update_kv_cache` `:1188`), `DS/tt/mla/indexer.py:628`; op test `DS/tests/op_unit_tests/test_deepseek_prefill_update_padded_kv_cache.py:123` | Constraint `kv_actual_global % 32 == 0` (`CP/docs/PREFILL_MIGRATION_TESTING.md`). | — |
| 14 | **Dense SwiGLU MLP** | **adapt** | `M3/tt/dense_mlp.py:26` (class), `:29-39` (`__init__`, `scatter_output` at `:38`), `:58` nested `_load` with the **cache-only branch at `:62-63`**, `:74-85` (HF `[out,in]` → transpose), `:87` `__call__`, `:99-112` (the TP collective: RS at `:105-107`, AR at `:112`) | Structurally *exactly* Llama's MLP: gate/up column-parallel, down row-parallel + TP collective. **One change:** `:92` `swiglu(gate, up, self.swiglu_cfg)` is M3's *clamped swigluoai* (cfg `:50-53`) — Llama needs plain SiLU-gated SwiGLU. Also drop the `zone(...)` profiler wrappers unless `M3/utils/profiler_utils.py` is ported. | `DEC-006` |
| 15 | **The SwiGLU activation itself** | **import (op)** | `ttnn.UnaryOpType.SILU` (confirmed present at runtime); one-op form `ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])` — in-tree usage `CM/modules/mlp/mlp_1d.py:262` and `:350`, `CM/modules/mlp/mlp_2d.py:336` and `:434`; **default is already SILU**: `CM/modules/mlp/mlp_1d.py:84` `mlp_activation_type: ttnn.UnaryOpType = ttnn.UnaryOpType.SILU` | **Recipe's open question at `:660-661` is RESOLVED.** The fused form *is* available and plain SILU on operand `a` is exactly Llama's `silu(gate) * up`. Prefer it over `ttnn.silu(gate)` + `ttnn.mul` (one op, one intermediate fewer). Contrast M3's 3-op alpha-scaled workaround at `M3/tt/moe/activation.py:50-59`, needed only because swigluoai has an `alpha` — Llama does not. | P3 |
| 16 | **Embedding** | **adapt** | `GO/tt/model.py:315` (`ttnn.embedding` inside `prepare_inputs_prefill` `:279`); vocab helper `GO/tt/model.py:31` `compute_per_device_vocab(vocab_size, num_tp)` | Replicated table for the first pass (recipe `:806-807`). `vocab_size` 128256 is tile-friendly (128256/32 = 4008). | P3 |
| 17 | **LM head** | **write (thin) / defer** | `GO/tt/model.py:241` (lm_head matmul inside `_forward_layers_and_head` `:179`); TTTv2 alternative `CM/modules/lm_head/lm_head_1d.py` | Prefill's product is the **KV cache**, so logits are needed only for `G-MODEL`'s top-1 check. ⚠️ Llama-3.1-8B is **untied** (`C:33`), unlike Llama-3.2-1B/3B — `lm_head.weight` is a distinct `[128256, 4096]` tensor and must not be aliased to the embedding. | P3 |
| 18 | **Decoder layer** | **adapt** | `GO/tt/layer.py:46` (class), `:126-136` (`__call__`), flow `:143-175`; `ttnn.move` long-seq guard `:137-140` (`if seqlen > 32*1024`); delta probe `:19` `_DELTA_PROBE`, `:22` `_delta_stats`, call sites `:158-159`, `:169-170` | Skeleton is Llama-shaped. **Delete:** the MoE `MLP` kwargs (`:60-62`, `:82-93` — GO has *no* dense branch, every layer is MoE) and the `layer_types`/`sliding_window` plumbing (`:96`, `:105`, `:119`). **Keep** the `ttnn.move` guard and the eager `deallocate(True)` (recipe `:747-748`). **Keep the delta probe** — recipe `:743-745` explicitly wants a bring-up probe. | P3 |
| 19 | **Model** | **adapt** | `GO/tt/model.py:41` (class), `:279` `prepare_inputs_prefill(tokens, start_pos=0, trace_enabled=False, batch_size=1, user_id=0, **kwargs)` (SP shard path `:287-306`), `:246` `prefill_forward`, `:322` `process_output_prefill(tt_out, last_token_idx)` (host TP concat `:325-329`), `:179` `_forward_layers_and_head` | The three prefill entry points are near-Llama-ready. Drop the `rot_mats_local` parameter (`:252`, for GO's sliding layers). | P3 |
| 20 | **CCL manager** | **adapt** | `GO/tt/ccl.py:17` (class), `:18` `__init__(mesh_device, num_links, topology=ttnn.Topology.Ring)`; `_init_subdevice` `:40`, `compute_with_storage_grid_size()` **`:44`**, `ccl_cores` `:46-48`, `SubDevice` `:50-54`, `ccl_sub_device_id` `:55`; ring-attention offset `:61` `(compute_grid_size.x - 1, 0)`; `_init_semaphores` `:63`; semaphore lists `:66-68` (rs, **6** = 3×2 per `:65`), `:72-74` (ag, **4** = 2×2 per `:71`), `:78-80` (barrier, **2**), `:84-86` (ring-attention, **2**); getters `:88`, `:95`, `:102`, `:108`, `:129` | Fully model-agnostic; the grid **is** derived correctly (unlike row 11). `:129 reset_global_semaphores` deliberately does *not* reset barrier/ring sems (`:132-135`). These four counts are what `G-SEMAPHORE` asserts. | `DEC-006` |
| 21 | **Mesh config** | **adapt — union of two** | `M3/config.py:21` (class), `:24` `__init__(mesh_shape, tp, tp_axis=1)`, `:40` `_validate`, `:52` `shard_mapper`, `:61` `column_parallel`, `:65` `row_parallel`, `:69` `sequence_parallel`, `:73` `shard_size`, `:77` `allreduce`, `:135` `allgather`, **`:155` `reduce_scatter`**, `:17-18` `_VALIDATED_MESH_SHAPE=(8,4)`/`_VALIDATED_TP=4`. And `GO/tt/config.py:19` (class), `:22` `__init__`, `:38` `_validate` (strict `raise` `:44-48`), **`:55-56` `sp` property**, `:60` `shard_mapper`, `:69`/`:73`/`:77`/`:81`, `:85` `allreduce`, `:138` `allgather`, `:15-16` `_VALIDATED_MESH_SHAPE=(4,8)`/`_VALIDATED_TP=8` | ⚠️ **Neither copy is a superset.** M3 has `reduce_scatter`, GO does not. GO has the `sp` property and strict validation, M3 does not. Take the **union**. GO's `_VALIDATED_*` already equals our `DEC-002` target. `R-009`. | `DEC-006` |
| 22 | **`num_links` / cache-file helpers** | **adapt (copy)** | `GO/utils/general_utils.py:11` `get_cache_file_name(tensor_cache_path, name)` (body `:12`), `:15` `cache_file_exists`, `:27` `get_default_num_links(mesh_device)` — body `:32-34`: single-row mesh → **1**, else **2 on Blackhole / 4 on Wormhole** | 35 lines, model-agnostic. Recipe says copy (`:434`). At `(4,8)` on Blackhole → **2**, matching `channels { count: 2 }` in the galaxy descriptor. | `DEC-006` |
| 23 | **State-dict prefix splitter** | **adapt (copy)** | `GO/utils/substate.py:15` `substate(state, key)`, `:37` `has_substate`, `:53` `indexed_substates` | 74 lines, model-agnostic. Recipe says copy (`:435`, `:462-463`). | `DEC-006` |
| 24 | **Weight loading / HF→Meta key mapping** | **import** | `TT/tt/load_checkpoints.py:18` `load_hf_state_dict(ckpt_dir)`; `:46` `load_hf_state_dict_filtered`; `:193` `convert_hf_to_meta(state_dict, head_dim, n_heads=None, n_kv_heads=None)` (pipeline `:194-197`); `:201` `convert_hf_to_meta_no_qkv_permute`; `:800` `map_hf_to_meta_keys`, rules **`:806-826`**; `:626` `replace_keys(state_dict, replacements)` (patterns are **not** regex); `:830` `map_meta_to_hf_keys`; `:494` `fuse_qkv_meta`; `:474` `fuse_mlp_meta` | Importable library code, exercised by every `tt_transformers` llama test. The mapping is what `G-WEIGHTS` checks. Worked example: `model.layers.N.self_attn.q_proj.weight` → (`model.` stripped, `:809`) → `layers.N.attention.wq.weight` (`self_attn`→`attention` `:814`, `q_proj`→`wq` `:819`). | `DEC-008` |
| 25 | **ModelArgs / weight-cache path** | **adapt** | `M3/tt/model_config.py:22` (class), `:25` `__init__`, `:126` `load_state_dict(weights_path, dummy_weights=False, convert_to_meta_format=True)`, `:212` `weight_cache_path(dtype)`, `:235` `get_state_dict_prefix(prefix, layer_idx)` | Mostly reusable, but `:161` `_load_text_backbone_safetensors` is M3-VL-specific (strips `language_model.*`) and its `convert_to_meta_format` path calls M3's **partial**-RoPE QKV converter — Llama needs the full-rotary one from row 7. ⚠️ **Do not** subclass `TT/tt/model_config.py:539 ModelArgs`: it raises without `HF_MODEL` (`:702`). `R-005`. | `DEC-004` |
| 26 | **Weight tilizing cache ("cache-only mode")** | **adapt** | `M3/tt/dense_mlp.py:58-72` — the `_load` closure; the branch is `:62-63` `if weight is None and not tensor_cache_path: return None`, then `ttnn.as_tensor(..., cache_file_name=get_cache_file_name(...))` at `:64-72` (`cache_file_name` at `:70`) | The exact shape the recipe requires every module to have (`:464-466`). Contrast the fail-loud rule: `M3/tt/mlp.py` raises rather than running bias-free (Appendix B). | — |
| 27 | **Chunked-prefill runtime** | **adapt** | `GO/tt/tt_prefill_runtime.py:96` (class), `:100` `__init__(mesh_device, hf_config, state_dict, config)`, `:59` `TtPrefillRuntimeConfig` (`sp_factor` `:88`, `tp_factor` `:92`), `:46` `resolve_chunk_sizes`, `:204` `make_chunk_input(token_ids, chunk_size=None)`, `:250` `compile(kv_caches=None)`, `:288` `prefill_chunk(input_tensor, kv_caches=None, *, slot_id, actual_start, actual_end, ...)`, `:359` `set_layer_ack_channel`, `:370` `kv_migration_base_address`, `:375` `build_kv_chunk_table`, `:407` `read_slot_kv`, `:505` `kv_cache_pcc_check`; `_build_model` `:126`, `_allocate_kv_cache` `:160`, `_build_indexed_rope` `:174` | Near-fully reusable; it already satisfies the engine's §2 contract. Baggage: `_build_indexed_rope`'s YaRN specifics and the MoE config fields. | P7 |
| 28 | **Prefill adapter** | **adapt** | `GO/tt/runners/adapters/gpt_oss.py:41` (class), the five runner defaults `:45-49` (`name`, `model_config`, `hf_model_default`, `ttnn_cache_default = ""`, `prefill_trace_default = ""`), test metadata `:53-58`, `:63` `load_hf_config`, `:75` `weight_cache_path(mesh_shape)`, `:96` `allocate_kv_cache(*, mesh_device, hf_config, params)`, `:120` `build_runtime(*, mesh_device, hf_config, params)`, `:31` `GptOssKvCaches`. Base: `CP/adapter.py:104` `PrefillModelAdapter`, `:46` `PrefillRunParams`, `:277` `ADAPTER_PATHS` | Delete `default_gate_mode` (`:50`) — MoE-router-only. Every other attribute is a per-model constant. | P10 |
| 29 | **Producer KV read-back branch** | **adapt shared code** | `CP/runners/prefill_producer.py:503` `_read_slot_kv_and_check_pcc(table, device_map, slot_id, real_len, trace_dir)`; branches `:507-508` (`minimax_m3`), `:509-510` (`gpt_oss_d_p`), `:511` fallback → MLA; impls `_gpt_oss` `:534`, `_m3` `:598`, `_mla` `:685`; dispatcher called from `:836` | Recipe `:494-516` is **correct**: the MLA fallback is wrong for Llama and would silently check the wrong bytes. `_read_slot_kv_and_check_pcc_gpt_oss` is the plain packed-K/V block-cyclic GQA reader — add `llama32_8b_d_p` to that branch (row 12 keeps the geometry). | P10 |
| 30 | **Golden-KV generation** | **adapt** | `M3/scripts/generate_golden_kv_cache.py` — header `:5-48`, output layout `:27-34`, usage from `:36` | Layout copies verbatim minus M3's MSA-only `index_k_cache_layer_N` (`:32-34`); Llama is exactly the K/V-only case. **`BLOCKED` on `R-003`** (needs real weights). | P7 |
| 31 | **Test factory / conftest / bundled config** | **adapt** | `M3/tests/test_factory.py:45` (`TestFactory`), `:56` `setup_test(mesh_device, use_real_weights=True, dtype=ttnn.bfloat8_b)` (returns dict `:79-87`; `MeshConfig` `:67`, `CCLManager` `:70`), `:25` `minimax_config_dims`, `:35-38` `requires_hf_reference`, `:22` `_CONFIG_JSON`, `:89` `parametrize_mesh_with_fabric`, `:190` `compare_tensors`. `M3/conftest.py:12` `pytest_addoption`, `:13` `--skip-model-load`, `:16-17` `state_dict` fixture. `M3/configs/MiniMax-M3/config.json` | ⚠️ **Exists only in M3** — `gpt_oss_d_p` has no `test_factory.py`, no `conftest.py`, no `configs/`. **Delivered in P1.** | `DEC-005` |
| 32 | **Per-module test convention** | **adapt** | `GO/tests/unit/test_attention_vs_ref.py:30` (`comp_pcc` import), `:83` `_build_cos_sin`, `:104` `_rotate_half`, `:109` `_rope_hf`, `:117` `_torch_attention`, `:149` `@parametrize("mesh_device", [(1,1)], indirect=True)`, `:150-157` layer parametrize, `:158` seq_len, `:159` test fn, **`:258` `comp_pcc(ref_out, out, 0.99)`**, log `:259`, assert `:260`. Also imports `get_rot_transformation_mat` (`:33`) and `convert_hf_qkv_to_meta_format` (`:34`) | The model answer (recipe `:588-592`). `_build_cos_sin` builds **both** cos/sin conventions from one frequency set — copy that structure exactly. Drop the sliding/sinks arguments. | — |
| 33 | **PCC / test utilities** | **import** | `CM/utility_functions.py:488` `comp_pcc(golden, calculated, pcc=0.99, rtol=1e-05, atol=1e-04)`, `:476` `comp_allclose`, `:741` `comp_allclose_and_pcc`, `:1043` `is_blackhole` | Library code, imported by every model test. Already used by `tests/unit/test_reference_model.py`. | — |
| 34 | **`mesh_device` / `reset_seeds` fixtures** | **import** | repo-root `conftest.py:554` `mesh_device(request, silicon_arch_name, device_params)` (decorator `:553`), `:34` `reset_seeds` (decorator `:33`, seeds `213919` at `:35-37`); `:43` `function_level_defaults` | Both recipe line numbers **confirmed correct**. Must **not** be redefined (recipe `:329-330`) — done. | — |
| 35 | **SP ring SDPA** | **adapt** | `GO/tt/attention/dense_sp.py:41-65` `dense_sp_attention(...)`; `ttnn.transformer.ring_joint_scaled_dot_product_attention` `:106-145` — `persistent_output_buffer_k/v` `:115-120`, `joint_strategy="rear"` `:122`, `dim=2` `:126`, `multi_device_global_semaphore=ccl_manager.ring_attention_ccl_semaphore_handles` `:127`, `ccl_core_grid_offset=...ring_attention_ccl_core_grid_offset` `:133`, `use_column_major_ccl=True` `:134`, `is_causal=True` `:135`, `kv_cache_batch_idx=slot_idx*num_layers+layer_idx` `:141`, `kv_actual_isl` `:142` | P8 only; stub with `NotImplementedError` in P5. `:30` `_gather_seq_len` exists only to size the sliding halo — for Llama's full-causal it collapses to `return full_seq`. Drop `attention_sink` (`:144`) / `sliding_window_size` (`:145`). | P8 |
| 36 | **Residual-layout policy** | **adapt (if scheme B)** | `M3/tt/residual.py:26` `DEFAULT_USE_SHARDED_RESIDUAL = True`, `:32-33` norm modes, `:36` `use_sharded_residual()` (env `M3_SHARDED_RESIDUAL`), `:44` `norm_mode()`, `:53` `use_distributed_norm()`, `:59` `gather_before_norm()` | Exists **only** in M3; `gpt_oss_d_p` is unconditionally replicated-residual. Recipe recommends scheme A (`:561`), so this is P4's call. Note the env vars need a `README.md` table row (P9 item 6). | P4 |
| 37 | **Second-opinion Galaxy Llama CCL** | **read only** | `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:25` `TT_CCL`, `tt/llama_attention.py:11` `TtLlamaAttention(LightweightModule)`, `tt/distributed_norm.py:10` `DistributedNorm(LightweightModule)` | Decode-oriented; consulted for llama-specific CCL placement only (recipe `:359`), not imported. | — |
| 38 | **Shared CCL substrate** | **read only** | `DS/tt/tt_ccl.py:60` `TT_CCL`; helpers `get_tt_ccl` `:36`, `create_global_semaphores` `:54`, `default_topology` `:357`, `per_axis_topology` `:387`, `get_num_links` `:455`. Also `CM/modules/tt_ccl.py` | A third CCL manager. Not imported — `CCLManager` (row 20) is the prefill-package lineage. Reinforces `R-009`. | — |

---

## 3. What we will **NOT** bring over

This is the anti-bloat control (recipe `:376-378`). One line each.

| Not bringing | Where it lives | Why not |
|---|---|---|
| **MoE / router / experts / dispatch-combine / EP=32** | `GO/tt/moe*`, `GO/tt/layer.py:60-62`, `:82-93`; `M3/tt/moe/`; `DS` MoE | Llama-3.1-8B has a **dense FFN on every layer**; no `num_local_experts` / `router` key exists in the config. |
| **Attention sinks** | `GO/tt/attention/weights.py:35` (`sinks`), `GO/tt/attention/prefill.py:45`, `GO/tt/attention/dense_sp.py:144`, the `scaling` coupling comment `GO/tt/attention/config.py:38-39` | No `sinks` key in the config. Also `sdpa_device_operation.cpp:409-412` requires `sink_shape[1] == q_shape[1]`, i.e. it is a real extra tensor to get right for nothing. |
| **Sliding-window / `layer_types` alternation** | `GO/tt/attention/config.py:34`, `GO/tt/attention/__init__.py:77-84`, `GO/tt/layer.py:96`, `:105`, `:119`, `GO/tt/attention/prefill.py:44`, `GO/tt/attention/dense_sp.py:30-39` (`_gather_seq_len`), `:145` | All 32 Llama layers are full-causal; no `sliding_window` / `layer_types` key. `_gather_seq_len` collapses to `return full_seq`. |
| **QK-norm** | Qwen3 / Gemma-3 lineage; `TT/tt/load_checkpoints.py:823-824` maps `q_norm`/`k_norm` | Absent from the config. Never add it. |
| **Gemma `+1` RMSNorm fold** | `GO/tt/rms_norm.py:20-27` (`use_gemma_norm`) | Llama's norm is plain `rms(x) * w`. |
| **Partial RoPE** (`rotary_dim < head_dim`) | `DS` MLA; `M3/tt/model_config.py` imports `convert_hf_qkv_to_meta_format_partial` | No `partial_rotary_factor` key; Llama rotates the full 128. **Consequence:** do not copy M3's partial QKV converter — use `TT/tt/load_checkpoints.py:451`. |
| **MLA / merged latent KV cache** | `DS/tt/mla/`, Kimi | Llama has plain packed K/V. **This one bites in P10:** `CP/runners/prefill_producer.py:511`'s default branch is the MLA reader and is *wrong for Llama*. |
| **MSA / sparse attention** | `M3` | Absent. Also drop M3's `index_k_cache_layer_N` from the golden-KV layout. |
| **MXFP4 / block-quantised loaders** | `GO` | Llama-3.1-8B ships plain bf16 safetensors (`C:34 torch_dtype`). |
| **Bias tensors anywhere** | `GO/tt/attention/weights.py:32` (`wqkv_bias`), `:34` (`o_proj_bias`); the bias adds at `GO/tt/attention/operations.py:25` (`ttnn.linear(..., bias=weights.wqkv_bias)`), `:120` (`ttnn.add(out, weights.o_proj_bias, ...)`) and `:202` | `attention_bias: false` (`C:5`), `mlp_bias: false` (`C:18`). **Assert absence** rather than branching on it. |
| **Tied embeddings** | Llama-3.2-1B/3B tie; `CM/modules/lm_head` | `tie_word_embeddings: false` (`C:33`) — `lm_head.weight` is its own tensor. Do not alias it. |
| **Decode / trace / 2CQ / perf / multi-galaxy PP / quantised weights** | `llama3_70b_galaxy`, `CM/models/llama3_8b`, everywhere | Explicit non-goals for this iteration (recipe `:15-16`). |
| **M3's profiler `zone(...)` decoration** | `M3/utils/profiler_utils.py`, wrapped through `M3/tt/dense_mlp.py:88`, `:91`, `:93`, `:104`, `:111` | Functional-first iteration; carrying it means porting a second utility module for no correctness gain. |

---

## 4. Nothing genuinely missing from the repo

Every component in §2 has a source. No kernel needs inventing. The two "write" rows that are not
adaptations are the thin LM head (row 17, only needed for `G-MODEL`'s top-1 check) and the
`tt/rope.py` wrapper (row 3/5, which is *assembly* of imported helpers plus the `R-006` assertions,
not new math). No `07_RISKS.md` entry is needed for a missing capability — all eleven open risks are
about *this tree's* quirks, not gaps.

---

## 5. What the recipe's "where to look" table **omits**

Recipe `:352-361` lists eight locations. It misses a whole shared module library, and — more
pointedly — **an existing Llama-3.1-8B implementation in this tree.**

| Location | What it is | Verdict for this package |
|---|---|---|
| `models/common/modules/` | **TTTv2**: a shared, unit-tested module library — `mlp/mlp_1d.py` + `mlp_2d.py`, `attention/attention_1d.py`, `rmsnorm/rmsnorm_{1d,2d}.py`, `rope/rope_1d.py`, `embedding/embedding_1d.py`, `lm_head/lm_head_1d.py`, `moe/`, `sampling/`, `tt_ccl.py`, `lazy_weight.py`, `lazy_buffer.py`. Documented in `models/common/modules/README.md` (the "Universal Module Contract" at `:38`, inventory table `:49-60`). | **Not a usable base**, but a mandatory *op-level* reference. Reasons it cannot be the base: (a) there is **no `Attention2D`** — only `attention_1d.py` — so the 2D-mesh path is incomplete for a full model; (b) no chunked-prefill runtime, no KV-migration, no adapter, i.e. nothing for P7/P10; (c) a different weight/config idiom (`LazyWeight`, `<Name>Config` + `from_config`) than the `(mesh_device, hf_config, state_dict, …)` convention the recipe mandates (`:457-460`). **What it did give us:** the definitive answer to the recipe's own open SwiGLU question — `mlp_1d.py:84` shows `mlp_activation_type` **defaults to `ttnn.UnaryOpType.SILU`** and `:262`/`:350` show the fused `ttnn.mul(..., input_tensor_a_activations=[...])` call shape. `mlp_2d.py:5` is the TG/Galaxy 2D-mesh MLP and is worth reading before writing `tt/mlp.py`. |
| `models/common/models/llama3_8b/` | A **complete TTTv2 Llama-3.1-8B**: `model.py` (`Llama3Transformer1D`), `executor.py`, `generator.py`, `hf_adaptor.py`, `README.md`. Its docstring names the exact stack: `Embedding1D → RotarySetup1D → TransformerBlock1D×N (RMSNorm1D, Attention1D, RMSNorm1D, MLP1D) → RMSNorm1D → LMHead1D`. | **Cannot run our target mesh** — `model.py:890` raises `ValueError("Llama3Transformer1D only supports 1D mesh topologies.")` (guarded by `is_galaxy_cluster` at `:884-890`), i.e. N150/N300/T3K only, and it is decode/generation-oriented with no disaggregated prefill. **But it is the closest existing Llama in the tree** and should be consulted for: the RoPE setup, the weight/`hf_adaptor` plumbing, and the per-module program configs. Worth an explicit "why not this" line in the package `README.md`, because the first reviewer question will be "there is already a llama3_8b — why a new package?" The answer: *2D-mesh TP×SP on a 32-chip Galaxy plus the disaggregated-prefill engine contract*, neither of which TTTv2 has yet. |

**Recommendation to the recipe author:** add both rows to the P2 table, and add a "why not TTTv2 /
`models/common/models/llama3_8b`" line to the P9 `README.md` checklist.

---

## 6. Recipe citations re-checked — **5 of 31 are wrong**

Verified by reading each file. Reported so the recipe can be fixed.

### Wrong

| Recipe claim | Line | Actual | Severity |
|---|---|---|---|
| `compute_llama3_parameters:405` is called "with `factor`, `low_freq_factor`, `high_freq_factor`, `original_max_position_embeddings` straight from `config.json:rope_scaling`" | `:620-624` | **The claim is false, not just the number.** `TT/tt/common.py:405` is `compute_llama3_parameters(freqs, scale_factor, orig_context_len)` — three args. `:407-408` are `low_freq_factor = 1` / `high_freq_factor = 4`, **local constants**. Only `factor` and `original_max_position_embeddings` are threaded through. Harmless for Llama-3.x (whose config *is* 1.0/4.0) but silently wrong for anything else. | **high** — `R-006` |
| `models/tt_transformers/tt/attention.py:643-716` implements both RoPE conventions | `:633-636` | The four implementations span **`:641-723`** (`_mllama_rope_decode` `:641-650`, `_mllama_rope_fused_qk_decode` `:653-661`, `_hf_rope_decode` `:663-681`, `_mllama_rope_prefill` `:683-700`, `_hf_rope_prefill` `:702-723`). The **dispatch** is at `:159-173`, which is the more useful anchor. `use_hf_rope` default `False` at `:623` is **correct**. | low |
| `gpt_oss_d_p/tt/attention/__init__.py:29` is the `Attention` class | `:687` | `class Attention:` is at **`:28`**. | low |
| `gpt_oss_d_p/tt/ccl.py:17` "itself mirroring `minimax_m3/tt/ccl.py:9`" | `:510` | Both correct, but the two files are the same class at an ~8-line offset (ring-attention offset GO `:61` vs M3 `:53`; semaphore handles GO `:84` vs M3 `:76`) — worth saying, since a reader diffing them will otherwise think they diverge. | info |
| `MeshConfig` "owns … the three collectives `allreduce`, `allgather`, `reduce_scatter`", template `minimax_m3/config.py:21` | `:518-523` | All four M3 line numbers correct (`:21`, `:77`, `:135`, `:155`). **Omission:** `gpt_oss_d_p/tt/config.py:19` has its own `MeshConfig` **without** `reduce_scatter`, and its `_VALIDATED_MESH_SHAPE`/`_VALIDATED_TP` at `:15-16` already equal the recipe's own `(4,8)`/TP=8 target. A reader following the recipe will not learn that the two copies differ. | medium — `R-009` |

### Also worth noting (correct, but incomplete in a way that matters)

| Recipe text | Line | Note |
|---|---|---|
| "the `reference_*` accessors … are the canonical llama oracles in this repo" | `:294-300` | All seven line numbers correct, but every one is a method on `ModelArgs`, which **raises without `HF_MODEL`** (`TT/tt/model_config.py:702`). On a machine with no checkpoint the recipe's *preferred* option is unreachable. `R-005`, `DEC-004`. |
| `get_rot_transformation_mat:562` | `:624`, `:630` | Correct line, but `:564` hard-codes `dhead = 32`, overwriting the argument. Call it with none. `R-010`. |
| "keep the `is_distributed` branch … but leave it `False` until P8" | `:613` | Right call, but the branch is **already dead upstream** (`GO/tt/rms_norm.py:33` pins it `False` with the condition commented out), so it has never been exercised. `R-007`. |
| "copy the two-line header style from `gpt_oss_d_p/tt/__init__.py`" | `:223-224` | The header is **three** lines (SPDX-FileCopyrightText, a bare `#`, SPDX-License-Identifier). Followed as-is. |
| P0 step 1 lists `reference` in the skeleton | `:222` | Contradicts `:301-304` and `:404-405`. `DEC-003`. |

### Confirmed correct (26)

`TT/tt/load_checkpoints.py` `convert_hf_qkv_to_meta_format:451`, `map_hf_to_meta_keys:800`,
`reverse_permute:891`, `fuse_qkv_meta:494`;
`TT/tt/common.py` `precompute_freqs:489`, `apply_scaling:437`, `compute_llama3_parameters:405`
(line right, claim wrong), `get_prefill_rot_mat:534`, `get_rot_transformation_mat:562`;
`TT/tt/model_config.py` `reference_transformer:4037`, `reference_decoder:4393`,
`reference_attention:4410`, `reference_mlp:4365`, `reference_rms_norm:4167`,
`reference_embedding:4379`, `reference_lm_head:4027`, `use_hf_rope` default `False` at `:623`;
root `conftest.py:554` (`mesh_device`) and `:34` (`reset_seeds`);
`GO/tt/model.py:41`; `GO/tt/tt_prefill_runtime.py:96`; `GO/tt/runners/adapters/gpt_oss.py:41` and
`:45-49`; `GO/tt/attention/kv_cache.py:27` (`NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32`) and `:87`
(`shard_shape`); `GO/tt/ccl.py:17`; `M3/tt/ccl.py:9`; `M3/config.py:21`, `:77`, `:135`, `:155`;
`CP/adapter.py:104`, `:46`, `:277`; `CP/runners/prefill_producer.py:503` and the `:507-511`
branches; `M3/tt/model_config.py:22`; `GO/tt/attention/operations.py:87`;
`GO/README.md:90-91` (the threshold sentence) and `:6` (the mesh).

---

## 7. What P3 inherits

Decided in P2, so P3 does not re-litigate:

1. **Four components are copies, not imports** — `MeshConfig`, `CCLManager`,
   `utils/general_utils.py`, `utils/substate.py` (`DEC-006`). Equivalents exist but only inside
   sibling *demo* packages, which the templates deliberately do not cross-import
   (`GO/README.md:46`). `MeshConfig` must be the **union** of M3's and GO's (`R-009`).
2. **`tt_rope`, key mapping, and PCC utils are genuine imports** from `models/tt_transformers` and
   `models/common` — library code, not demo code (`DEC-007`, `DEC-008`).
3. **GQA needs no on-chip KV repeat** — settles recipe `:680-681`.
4. **The SwiGLU activation is one fused op** — settles recipe `:660-661`:
   `ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])`.
5. **`kv_cache.py` keeps gpt-oss's `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32`** and shard shape, so
   P10 reuses the existing producer reader.

Left open for P3, deliberately:

| Open question | Where it bites | Inputs already gathered |
|---|---|---|
| **Fused `wqkv` or three separate q/k/v weights?** | `tt/attention/weights.py`, `operations.py` | GO fuses (`weights.py:31`) and its head split assumes it (`operations.py:29`, `:41` `nlp_create_qkv_heads`). The recipe's own `DEC-014` *example* argues for three separate. Choosing "separate" means also replacing the head split. |
| **`hf_config` as a dict or an HF config object?** | every module signature | The templates pass an **object** — `M3/tt/dense_mlp.py:47` does `hf_config.hidden_size`. `llama_config_dims()` returns a **dict**, and `TT/tt/common.py:165 get_rope_theta` takes a **dict**. P3 must pick one and adapt at the boundary; a silent mix is how `None` dims get in. |
| **Residual scheme A or B** | every module tail + the norm | P4's gate. Recipe recommends A (`:561`); `R-007` (the distributed-norm branch is dead upstream) is a further argument for A this iteration. |
| **KV cache dtype (bf8_b vs bf16)** | `G-KV`, `G-CHUNK` thresholds | GO defaults `bfloat8_b` (`kv_cache.py:56`). Recipe requires the PCC cost be *measured*, not assumed (`:707-709`). |
| **SDPA grid + compute-kernel-config for Blackhole** | `tt/attention/config.py` | `R-008`: GO hard-codes `CoreCoord(8, 8)` (`:96`) and a `WormholeComputeKernelConfig` (`:103`). Derive from `compute_with_storage_grid_size()`; branch on `is_blackhole()` (`CM/utility_functions.py:1043`). |

Also for P3's file tree: the outline at `:396-453` is sound, minus `reference/` (`DEC-003`). Add
`utils/__init__.py` (the outline lists `utils/` but P0 step 1's `mkdir` list omits it) and note that
`tests/test_factory.py`, `conftest.py` and `configs/` already exist from P1.
