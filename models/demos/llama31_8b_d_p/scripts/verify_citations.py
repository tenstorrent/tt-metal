# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Re-verify every load-bearing `path:line` citation used by a bring-up's code and logs.

Copy this into `<your package>/scripts/` and set `PKG` below. Each entry in `CITES` is
`(path_relative_to_repo_root, line_number, substring_that_must_be_on_that_line)`; the script reads
the file and reports the true line numbers of any needle whose claimed line is wrong.

Run:
    python <your package>/scripts/verify_citations.py

Exit 0 iff every citation verifies. **Extend `CITES` in every phase.** In the bring-up this came
from it caught five wrong line numbers in the recipe itself and five more in a survey's own first
draft; an unverified `path:line` is worth less than no citation, because it reads as authoritative.
Two failure modes it exists to catch: a cited file grows and every line below the edit shifts, and a
bare basename silently resolves to a *different* package's file of the same name.
"""

import os
import re
import sys

# repo root = four levels up from <package>/scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

TT = "models/tt_transformers"
GO = "models/demos/gpt_oss_d_p"
M3 = "models/demos/minimax_m3"
DS = "models/demos/deepseek_v3_d_p"
CP = "models/demos/common/prefill"
CM = "models/common"
# Your package, relative to the repo root. The only line you must edit.
PKG = "models/demos/llama31_8b_d_p"
LL = PKG
# P8: the ring-joint SDPA device op — cited by tt/attention/dense_sp.py and tt/attention/config.py.
RJ = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp"
# P5.4-P5.6: the recipe itself, CONTENT-checked. Pass 2 already scans it for outbound refs, but it
# only range-checks refs *into* it — and this session found that four of its own first-draft refs
# into the recipe were in range and wrong (R-016). Every recipe ref this package makes now has a
# needle here, so a recipe edit that shifts a section is a MISMATCH rather than a silent lie.
RCP = "models/demos/common/bringup/BRINGUP_RECIPE.md"

# P10 — the engine's own files and the two contract documents. Named here rather than spelled out in
# every CITES row, because this phase's whole subject is the boundary between this package and them.
ADP = f"{CP}/adapter.py"
RNR = f"{CP}/runners/prefill_runner.py"
PRD = f"{CP}/runners/prefill_producer.py"
MIG = f"{CP}/runners/migration.py"
DOC_ADD = f"{CP}/docs/ADDING_A_PREFILL_MODEL.md"
DOC_MIG = f"{CP}/docs/PREFILL_MIGRATION_TESTING.md"
# The kit recipe. Not vendored into the package (`DEC-002`), so it is named once here.
RECIPE = "models/demos/common/bringup/BRINGUP_RECIPE.md"
GOX_TBL = f"{GO}/tt/runners/kv_chunk_table.py"
GOX_ADP = f"{GO}/tt/runners/adapters/gpt_oss.py"
GOX_RT = f"{GO}/tt/tt_prefill_runtime.py"
# P6: the installed `transformers` Llama modeling file. Cited because `DEC-052`'s claim about
# `output_hidden_states` is a claim about THIS version's code, not about transformers in general.
HFL = "python_env/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py"

# (file, line, substring that MUST appear on that line)

# ---------------------------------------------------------------------------------------------
# CITES — populate this as you write. One entry per path:line claim you make in code or docs:
#     (f"{PKG}/tt/mlp.py", 42, "def __call__"),
# The needle is a substring that must appear ON that line. When a file shifts, this tells you
# which citations moved instead of letting them rot into confident nonsense.
#
# In the bring-up this came from it reached 661 entries and caught wrong line numbers in the
# recipe itself as well as in agents' first drafts. An unverified path:line is worse than no
# citation, because it reads as authoritative.
# ---------------------------------------------------------------------------------------------
CITES = [
    # --- P0: the model card's non-config.json claims -----------------------------------------
    (f"{GO}/tt/attention/kv_cache.py", 95, "Per-chip cache is one head"),
    (
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/"
        "update_padded_kv_cache_device_operation.cpp",
        230,
        "cache and input num-heads dim must match",
    ),
    ("ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp", 98, "nqh >= nkv && nqh % nkv == 0"),
    (f"{CP}/docs/PREFILL_MIGRATION_TESTING.md", 62, "CHUNK_SIZE % (SP*32) == 0"),
    (f"{TT}/model_params/Llama-3.1-8B-Instruct/config.json", 13, '"hidden_act": "silu"'),
    # --- P1: the reference strategy and the transformers-5.12.1 traps -------------------------
    (f"{TT}/tt/common.py", 165, "def get_rope_theta"),
    (f"{TT}/tt/common.py", 183, "def get_rope_scaling"),
    (f"{TT}/tt/common.py", 437, "def apply_scaling"),
    (f"{TT}/tt/common.py", 489, "def precompute_freqs"),
    (f"{TT}/tt/common.py", 534, "def get_prefill_rot_mat"),
    (f"{TT}/tt/common.py", 562, "def get_rot_transformation_mat(dhead=32)"),
    (f"{TT}/tt/common.py", 564, "dhead = 32"),
    (f"{TT}/tt/model_config.py", 702, "Please set HF_MODEL"),
    (f"{TT}/tt/model_config.py", 4027, "def reference_lm_head"),
    (f"{TT}/tt/model_config.py", 4037, "def reference_transformer"),
    (f"{TT}/tt/model_config.py", 4167, "def reference_rms_norm"),
    (f"{TT}/tt/model_config.py", 4365, "def reference_mlp"),
    (f"{TT}/tt/model_config.py", 4379, "def reference_embedding"),
    (f"{TT}/tt/model_config.py", 4393, "def reference_decoder"),
    (f"{TT}/tt/model_config.py", 4410, "def reference_attention"),
    (f"{GO}/tt/model_config.py", 76, 'getattr(self.hf_config, "rope_theta"'),
    (f"{GO}/tt/tt_prefill_runtime.py", 185, 'rope_theta=getattr(self.hf_config, "rope_theta"'),
    (f"{M3}/tests/test_factory.py", 25, "def minimax_config_dims"),
    (f"{M3}/tests/test_factory.py", 35, "requires_hf_reference"),
    (f"{M3}/conftest.py", 13, "--skip-model-load"),
    (f"{M3}/conftest.py", 16, 'scope="session"'),
    ("conftest.py", 34, "def reset_seeds"),
    ("conftest.py", 554, "def mesh_device"),
    (f"{M3}/tests/unit/test_reference_model.py", 1, "SPDX-FileCopyrightText"),
    # --- P2: the survey's reuse/write decisions -----------------------------------------------
    (f"{M3}/config.py", 21, "class MeshConfig"),
    (f"{M3}/config.py", 77, "def allreduce"),
    (f"{M3}/config.py", 135, "def allgather"),
    (f"{M3}/config.py", 155, "def reduce_scatter"),
    (f"{GO}/tt/config.py", 15, "_VALIDATED_MESH_SHAPE = (4, 8)"),
    (f"{GO}/tt/config.py", 16, "_VALIDATED_TP = 8"),
    (f"{GO}/tt/config.py", 19, "class MeshConfig"),
    (f"{GO}/tt/ccl.py", 17, "class CCLManager"),
    (f"{GO}/tt/ccl.py", 132, "does NOT reset the barrier"),
    (f"{GO}/tt/rms_norm.py", 33, "self.is_distributed = False"),
    (f"{GO}/tt/attention/config.py", 71, "fp32_dest_acc_en: bool = False"),
    (f"{GO}/tt/attention/config.py", 103, "ttnn.WormholeComputeKernelConfig"),
    (f"{GO}/tt/runners/adapters/gpt_oss.py", 75, "def weight_cache_path"),
    (f"{M3}/tt/dense_mlp.py", 47, "hf_config.hidden_size"),
    (f"{M3}/tt/residual.py", 26, "DEFAULT_USE_SHARDED_RESIDUAL = True"),
    (f"{M3}/tt/residual.py", 32, 'DEFAULT_NORM_MODE = "gather_first"'),
    (f"{CM}/modules/mlp/mlp_2d.py", 461, "cluster_axis=0"),
    (f"{CM}/models/llama3_8b/model.py", 890, "only supports 1D mesh topologies"),
    (f"{TT}/tt/load_checkpoints.py", 451, "def convert_hf_qkv_to_meta_format"),
    (f"{TT}/tt/load_checkpoints.py", 494, "def fuse_qkv_meta"),
    (f"{TT}/tt/load_checkpoints.py", 800, "def map_hf_to_meta_keys"),
    (f"{TT}/tt/load_checkpoints.py", 891, "def reverse_permute"),
    (f"{GO}/utils/substate.py", 15, "def substate"),
    (f"{GO}/utils/substate.py", 37, "def has_substate"),
    (f"{GO}/utils/substate.py", 53, "def indexed_substates"),
    ("ttnn/ttnn/types.py", 61, "BlackholeComputeKernelConfig = WormholeComputeKernelConfig"),
    # --- P2 (continued): the survey's per-component citations ---------------------------------
    (f"{GO}/utils/general_utils.py", 11, "def get_cache_file_name"),
    (f"{GO}/utils/general_utils.py", 15, "def cache_file_exists"),
    (f"{GO}/utils/general_utils.py", 27, "def get_default_num_links"),
    (f"{GO}/tt/rms_norm.py", 17, "class RMSNorm"),
    (f"{GO}/tt/rms_norm.py", 46, "self.eps = hf_config.rms_norm_eps"),
    (f"{GO}/tt/rms_norm.py", 94, "tt_output = ttnn.rms_norm("),
    (f"{GO}/tt/rope.py", 36, "def yarn_inv_freq"),
    (f"{GO}/tt/rope.py", 115, "def build_indexed_rope"),
    (f"{GO}/tt/mlp.py", 21, "from models.demos.deepseek_v3_d_p"),
    (f"{GO}/tt/mlp.py", 38, "class MLP"),
    (f"{GO}/tt/layer.py", 22, "def _delta_stats"),
    (f"{GO}/tt/layer.py", 46, "class DecoderLayer"),
    (f"{GO}/tt/layer.py", 126, "def __call__"),
    (f"{GO}/tt/model.py", 31, "def compute_per_device_vocab"),
    (f"{GO}/tt/model.py", 41, "class Model"),
    (f"{GO}/tt/model.py", 179, "def _forward_layers_and_head"),
    (f"{GO}/tt/model.py", 246, "def prefill_forward"),
    (f"{GO}/tt/model.py", 279, "def prepare_inputs_prefill"),
    (f"{GO}/tt/model_config.py", 30, "class ModelArgs"),
    (f"{GO}/tt/model_config.py", 106, "def load_state_dict"),
    (f"{GO}/tt/model_config.py", 157, "def weight_cache_path"),
    (f"{GO}/tt/model_config.py", 175, "def get_state_dict_prefix"),
    (f"{GO}/tt/ccl.py", 88, "def get_rs_ping_pong_semaphore"),
    (f"{GO}/tt/ccl.py", 95, "def get_ag_ping_pong_semaphore"),
    (f"{GO}/tt/ccl.py", 102, "def get_barrier_semaphore"),
    (f"{GO}/tt/ccl.py", 108, "def get_ring_gather_buffer"),
    (f"{GO}/tt/ccl.py", 129, "def reset_global_semaphores"),
    (f"{GO}/tt/attention/config.py", 23, "class AttentionConfig"),
    (f"{GO}/tt/attention/config.py", 90, "def get_prefill_sdpa_config"),
    (f"{GO}/tt/attention/weights.py", 23, "class AttentionWeights"),
    (f"{GO}/tt/attention/weights.py", 38, "def load_attention_weights"),
    (f"{GO}/tt/attention/operations.py", 29, "def split_qkv_heads_prefill"),
    (f"{GO}/tt/attention/operations.py", 50, "def apply_rope"),
    (f"{GO}/tt/attention/prefill.py", 34, "def _run_sdpa"),
    (f"{GO}/tt/attention/kv_cache.py", 48, "def allocate_kv_cache"),
    (f"{GO}/tt/attention/kv_cache.py", 99, "torch.zeros(num_users * num_layers, 1, seq_local, head_dim)"),
    (f"{GO}/tt/attention/kv_cache.py", 117, "def _write_one"),
    (f"{GO}/tt/attention/kv_cache.py", 125, "update_padded_kv_cache("),
    (f"{GO}/tt/attention/kv_cache.py", 138, "def write_kv_chunk"),
    (f"{GO}/tt/attention/dense_sp.py", 106, "ring_joint_scaled_dot_product_attention"),
    (f"{GO}/tt/runners/adapters/gpt_oss.py", 41, "class GptOssPrefillAdapter"),
    (f"{GO}/tt/runners/kv_chunk_table.py", 66, "def build_kv_chunk_address_table"),
    (f"{GO}/tt/runners/kv_chunk_table.py", 179, "def build_and_serialize_kv_chunk_table"),
    (f"{GO}/tests/unit/test_attention_vs_ref.py", 149, 'parametrize("mesh_device", [(1, 1)]'),
    (f"{GO}/tests/unit/test_attention_vs_ref.py", 197, "convert_hf_qkv_to_meta_format"),
    (f"{GO}/tests/unit/test_attention_vs_ref.py", 213, "get_rot_transformation_mat()"),
    (f"{GO}/tests/unit/test_attention_vs_ref.py", 258, "comp_pcc"),
    (f"{GO}/README.md", 26, "hoist the shared prefill scaffolding"),
    (f"{M3}/tt/dense_mlp.py", 26, "class DenseMLP"),
    (f"{M3}/tt/dense_mlp.py", 58, "def _load"),
    (f"{M3}/tt/rms_norm.py", 30, "class RMSNorm"),
    (f"{M3}/tt/parallel_embedding.py", 80, "class TtParallelEmbedding"),
    (f"{M3}/tt/weight_cache.py", 49, "def weight_cache_is_complete"),
    (f"{M3}/tt/model.py", 87, "class Model"),
    (f"{M3}/tt/attention/operations.py", 85, "rotary_embedding_indexed"),
    (f"{M3}/scripts/generate_golden_kv_cache.py", 195, "def main"),
    (f"{M3}/scripts/verify_golden_kv.py", 26, "def verify_trace"),
    (f"{TT}/tt/common.py", 407, "low_freq_factor = 1"),
    (f"{TT}/tt/common.py", 408, "high_freq_factor = 4"),
    (f"{CM}/modules/attention/attention_1d.py", 319, "class Attention1D"),
    (f"{CM}/modules/mlp/mlp_2d.py", 256, "def _reduce_scatter_axis1"),
    (f"{CM}/modules/mlp/mlp_2d.py", 259, "cluster_axis = 1"),
    (f"{CM}/modules/mlp/mlp_2d.py", 361, "self._all_reduce_tg"),
    (f"{CM}/utility_functions.py", 476, "def comp_allclose"),
    (f"{CM}/utility_functions.py", 488, "def comp_pcc"),
    (f"{CM}/utility_functions.py", 1043, "def is_blackhole"),
    (f"{CP}/adapter.py", 95, "class KvCaches"),
    (f"{CP}/adapter.py", 104, "class PrefillModelAdapter"),
    # --- P3: the outline's per-file contracts (interfaces, shapes, templates) -----------------
    (f"{GO}/tt/config.py", 44, "if self.tp != tp_dim_size"),
    (f"{GO}/tt/config.py", 56, "def sp(self)"),
    (f"{M3}/config.py", 42, "if self.tp > tp_dim_size"),
    (f"{M3}/config.py", 44, "self.total_devices % self.tp"),
    (f"{M3}/config.py", 104, "Free the full-size input"),
    (f"{GO}/tt/ccl.py", 44, "compute_with_storage_grid_size()"),
    (f"{GO}/tt/ccl.py", 61, "ring_attention_ccl_core_grid_offset = (compute_grid_size.x - 1, 0)"),
    (f"{GO}/tt/ccl.py", 65, "rs_n_sems = 3 * 2"),
    (f"{GO}/tt/ccl.py", 71, "ag_n_sems = 2 * 2"),
    (f"{GO}/tt/ccl.py", 77, "barrier_ns_sems = 2 * 1"),
    (f"{GO}/tt/ccl.py", 84, "ring_attention_ccl_semaphore_handles"),
    (f"{GO}/tt/rms_norm.py", 25, "if self.use_gemma_norm"),
    (f"{GO}/tt/rms_norm.py", 27, "reshape((1, 1, -1, ttnn.TILE_SIZE))"),
    (f"{GO}/tt/rms_norm.py", 34, "self.tt_weight = ttnn.as_tensor("),
    (f"{GO}/tt/rms_norm.py", 50, "if self.is_distributed"),
    (f"{TT}/tt/common.py", 405, "def compute_llama3_parameters"),
    (f"{TT}/tt/common.py", 525, "def gather_cos_sin"),
    (f"{TT}/tt/common.py", 542, "cos_gathereds = ttnn.from_torch("),
    (f"{TT}/tt/common.py", 547, "ReplicateTensorToMesh"),
    (f"{M3}/tt/dense_mlp.py", 29, "def __init__"),
    (f"{M3}/tt/dense_mlp.py", 77, "transpose(-1, -2).unsqueeze(0).unsqueeze(0)"),
    (f"{M3}/tt/dense_mlp.py", 89, "gate = ttnn.linear(x, self.gate_proj"),
    (f"{M3}/tt/dense_mlp.py", 94, "out = ttnn.linear(act, self.down_proj"),
    (f"{M3}/tt/dense_mlp.py", 96, "down is row-parallel"),
    (f"{M3}/tt/dense_mlp.py", 112, "self.mesh_config.allreduce"),
    (f"{GO}/tt/attention/config.py", 57, "class ProgramConfig"),
    (f"{GO}/tt/attention/config.py", 96, "ttnn.CoreCoord(8, 8)"),
    (f"{GO}/tt/attention/weights.py", 64, "o_proj padding"),
    (f"{GO}/tt/attention/weights.py", 83, "qkv_list = []"),
    (f"{GO}/tt/attention/weights.py", 96, "torch.cat([wq, wk, wv], dim=-1)"),
    (f"{GO}/tt/attention/weights.py", 100, "qkv_cat = torch.cat(qkv_list"),
    (f"{GO}/tt/attention/weights.py", 135, "else:"),
    (f"{GO}/tt/attention/weights.py", 145, "col_mesh_mapper = mesh_config.column_parallel"),
    (f"{GO}/tt/attention/weights.py", 146, "row_mesh_mapper = mesh_config.row_parallel"),
    (f"{GO}/tt/attention/operations.py", 25, "ttnn.linear(hidden_states, weights.wqkv"),
    (f"{GO}/tt/attention/operations.py", 41, "ttnn.experimental.nlp_create_qkv_heads("),
    (f"{GO}/tt/attention/operations.py", 78, "if kv_actual_global is not None"),
    (f"{GO}/tt/attention/operations.py", 126, "_FUSED_MM_RS_CONFIGS"),
    (f"{GO}/tt/attention/operations.py", 131, "def is_shape_fused_mm_rs_supported"),
    (f"{GO}/tt/attention/operations.py", 142, "def apply_output_projection_fused_rs"),
    (f"{GO}/tt/attention/prefill.py", 51, "def attention_forward"),
    (f"{GO}/tt/attention/prefill.py", 106, "if seq_len > 32 * 1024"),
    (f"{GO}/tt/attention/prefill.py", 107, "activation_dtype = ttnn.bfloat8_b"),
    (f"{GO}/tt/attention/prefill.py", 257, "elif cached_len > 0"),
    (f"{GO}/tt/attention/prefill.py", 266, "raise NotImplementedError("),
    (f"{GO}/tt/attention/prefill.py", 292, "use_fused_rs = ("),
    (f"{GO}/tt/attention/kv_cache.py", 27, "NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32"),
    (f"{GO}/tt/attention/kv_cache.py", 56, "cache_dtype=ttnn.bfloat8_b"),
    (f"{GO}/tt/attention/kv_cache.py", 72, "bf8 matches the DeepSeek substrate"),
    (f"{GO}/tt/attention/kv_cache.py", 77, "assert ("),
    (f"{GO}/tt/attention/kv_cache.py", 87, "shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim]"),
    (f"{GO}/tt/attention/kv_cache.py", 145, "One user per call"),
    (f"{GO}/tt/attention/kv_cache.py", 149, "assert tt_k.shape[0] == 1 and tt_v.shape[0] == 1"),
    (f"{GO}/tt/attention/kv_cache.py", 155, "slot_idx {slot_idx} out of range"),
    (f"{GO}/tt/attention/kv_cache.py", 157, "assert ("),
    (f"{GO}/tt/attention/dense_sp.py", 41, "def dense_sp_attention"),
    (f"{GO}/tt/attention/__init__.py", 28, "class Attention"),
    (f"{GO}/tt/attention/__init__.py", 38, "def __init__"),
    (f"{GO}/tt/attention/__init__.py", 77, "Sliding vs full for this layer"),
    (f"{GO}/tt/attention/__init__.py", 84, "dataclasses.replace(config, sliding_window="),
    (f"{GO}/tt/layer.py", 19, 'os.environ.get("GPT_OSS_DELTA_PROBE"'),
    (f"{GO}/tt/layer.py", 138, "if seqlen > 32 * 1024"),
    (f"{GO}/tt/layer.py", 140, "hidden_states = ttnn.move(hidden_states)"),
    (f"{GO}/tt/model.py", 64, "self.head_dim = hf_config.head_dim"),
    (f"{GO}/tt/model.py", 84, "self.embedding_weight = ttnn.as_tensor("),
    # --- P4: the CCL plan's placement, semaphore and topology claims -------------------------
    (f"{GO}/tt/model.py", 82, "Shard it across"),
    (f"{GO}/tt/ccl.py", 106, "return self.barrier_semaphore[cur_idx]"),
    (f"{GO}/tt/ccl.py", 134, "TODO(P5): reset them here too"),
    (f"{M3}/config.py", 102, "barrier_semaphore=ccl_manager.get_barrier_semaphore()"),
    (f"{M3}/config.py", 115, "gathered = ttnn.experimental.all_gather_async("),
    (f"{M3}/config.py", 124, "barrier_semaphore=ccl_manager.get_barrier_semaphore()"),
    (f"{M3}/config.py", 29, "tp_axis: which mesh axis is TP"),
    (f"{M3}/config.py", 88, "ttnn.pad(tensor"),
    (f"{M3}/config.py", 94, "reduce_scatter_minimal_async("),
    (f"{GO}/tt/rms_norm.py", 60, "tt_gathered_stats_memory_config = ttnn.create_sharded_memory_config("),
    (f"{GO}/tt/rms_norm.py", 67, "ttnn.rms_norm_pre_all_gather("),
    (f"{GO}/tt/rms_norm.py", 70, "tt_gathered_stats = ttnn.all_gather("),
    (f"{GO}/tt/rms_norm.py", 74, "cluster_axis=1,"),
    (f"{GO}/tt/rms_norm.py", 82, "ttnn.rms_norm_post_all_gather("),
    (f"{GO}/tt/attention/prefill.py", 234, "full_seq_len = seq_len * sp"),
    (f"{GO}/tt/attention/prefill.py", 235, "mesh_config.allgather(tt_q, ccl_manager, axis=mesh_config.sp_axis, dim=2)"),
    (f"{GO}/tt/attention/kv_cache.py", 132, "cluster_axis=sp_axis"),
    (f"{GO}/tt/attention/operations.py", 79, "rotary_embedding_indexed("),
    (f"{M3}/tt/parallel_embedding.py", 80, "class TtParallelEmbedding"),
    (f"{M3}/tt/residual.py", 9, "one all-gather per"),
    (f"{M3}/tt/residual.py", 26, "DEFAULT_USE_SHARDED_RESIDUAL = True"),
    (f"{M3}/tt/residual.py", 32, 'DEFAULT_NORM_MODE = "gather_first"'),
    (f"{M3}/tt/dense_mlp.py", 38, "scatter_output=None"),
    ("tt_metal/api/tt-metalium/mesh_device.hpp", 305, "void quiesce_devices()"),
    ("tt_metal/api/tt-metalium/mesh_device.hpp", 307, "create_submesh("),
    (
        "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto",
        1,
        "",
    ),
    (
        "tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_sp4_torus_xy_graph_descriptor.textproto",
        1,
        "",
    ),
    (
        "tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto",
        1,
        "",
    ),
    (f"{GO}/tt/model.py", 123, "sampling_splits = mesh_device.shape[1]"),
    (f"{GO}/tt/model.py", 134, "self.lm_head_weight = ttnn.as_tensor("),
    (f"{GO}/tt/model.py", 145, "_supports_on_device_sampling"),
    (f"{GO}/tt/model.py", 196, "per-layer KV migration / validation"),
    (f"{GO}/tt/model.py", 211, "on_layer_complete(i)"),
    (f"{GO}/tt/model.py", 241, "logits = ttnn.matmul(hidden_states, self.lm_head_weight"),
    (f"{GO}/tt/model.py", 288, "if self.sequence_parallel"),
    (f"{GO}/tt/model.py", 313, "bf16 (not bf8) so the residual stream keeps full dynamic range"),
    (f"{GO}/tt/model.py", 315, "tokens_embd = ttnn.embedding("),
    (f"{GO}/tt/model.py", 318, "unsqueeze_to_4D"),
    (f"{GO}/tt/model.py", 322, "def process_output_prefill"),
    (f"{GO}/tt/model_config.py", 160, "else Path(self.model_path)"),
    (f"{GO}/tt/tt_prefill_runtime.py", 88, "def sp_factor"),
    (f"{GO}/tt/tt_prefill_runtime.py", 92, "def tp_factor"),
    (f"{GO}/tt/tt_prefill_runtime.py", 96, "class TtPrefillRuntime"),
    (f"{GO}/tt/tt_prefill_runtime.py", 174, "def _build_indexed_rope"),
    (f"{GO}/tests/galaxy_prefill_kv_pcc.py", 121, 'os.getenv("PREFILL_TOPOLOGY"'),
    (f"{GO}/tests/galaxy_prefill_kv_pcc.py", 161, "ttnn.Topology.Linear if _linear else ttnn.Topology.Ring"),
    (f"{GO}/tests/test_kv_cache_table.py", 126, '"fabric_config": ttnn.FabricConfig.FABRIC_1D'),
    (f"{M3}/tests/test_factory.py", 89, "def parametrize_mesh_with_fabric"),
    (f"{M3}/tt/model_config.py", 22, "class ModelArgs"),
    (
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.cpp",
        22,
        "if (input_tensor_kv.has_value())",
    ),
    (
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.cpp",
        32,
        "Head dims must be the same for Q and K, V",
    ),
    (".pre-commit-config.yaml", 51, "prefer-expect-error"),
    # --- P5.1-P5.3: the citations the device modules and their gates make -------------------
    (".pre-commit-config.yaml", 53, "language: pygrep"),
    (".pre-commit-config.yaml", 55, "entry:"),
    (".pre-commit-config.yaml", 56, "Override on the same line"),
    ("ttnn/ttnn/__init__.py", 305, "WormholeComputeKernelConfig"),
    (f"{GO}/tt/ccl.py", 24, "_ping_pong_buffer_cache = {}"),
    (f"{GO}/tt/ccl.py", 25, "_ping_pong_buffer_indices = {}"),
    (f"{GO}/tt/ccl.py", 50, "_worker_sub_device = ttnn.SubDevice("),
    (f"{GO}/tt/ccl.py", 55, "self.ccl_sub_device_id = ttnn.SubDeviceId(0)"),
    (f"{DS}/tt/tt_ccl.py", 67, "self.sub_device_crs = ttnn.CoreRangeSet("),
    (f"{DS}/tt/mla/utils.py", 65, "def block_cyclic_reorder"),
    (f"{GO}/tt/moe/tt_gpt_oss_moe.py", 104, "subdevice_id=None"),
    (f"{M3}/tt/moe/tt_minimax_moe.py", 117, "subdevice_id=None"),
    (f"{GO}/tt/attention/config.py", 70, "math_approx_mode: bool = False"),
    (f"{CM}/models/llama32_1b/model.py", 1026, "math_approx_mode=False"),
    (f"{GO}/tt/rms_norm.py", 89, "stats=tt_gathered_stats,"),
    (f"{GO}/utils/general_utils.py", 33, "if mesh_device.shape[0] == 1"),
    (f"{M3}/tests/test_factory.py", 45, "class TestFactory"),
    (f"{M3}/tests/test_factory.py", 56, "def setup_test"),
    (f"{GO}/tests/unit/test_attention_vs_ref.py", 83, "def _build_cos_sin"),
    (f"{TT}/tt/model_config.py", 623, "use_hf_rope"),
    (f"{M3}/tt/attention/operations.py", 93, "rotary_embedding_llama"),
    (f"{GO}/tt/attention/operations.py", 87, "rotary_embedding_llama"),
    # This package's own P5 files, so a later phase's edit cannot silently move a cited line.
    (f"{LL}/tt/config.py", 52, "def default_compute_kernel_config"),
    (f"{LL}/tt/ccl.py", 46, "class CCLManager"),
    (f"{LL}/tt/rms_norm.py", 39, "class RMSNorm"),
    (f"{LL}/tt/rope.py", 64, "def assert_llama3_factors"),
    ("conftest.py", 948, "def expect_error"),
    (f"{GO}/README.md", 71, "| tensor | shape | dtype | layout |"),
    (f"{TT}/tt/load_checkpoints.py", 494, "def fuse_qkv_meta"),
    (f"{CP}/runners/prefill_runner.py", 364, "d2h_service=d2h_service, metadata_msg=metadata_msg"),
    (f"{CP}/runners/prefill_runner.py", 477, "hf_config.max_seq_len = MAX_SEQ_LEN"),
    (f"{CP}/adapter.py", 46, "class PrefillRunParams"),
    (f"{CP}/adapter.py", 277, "ADAPTER_PATHS = {"),
    (f"{GO}/tt/runners/adapters/gpt_oss.py", 45, 'name = "gpt_oss_d_p"'),
    (f"{GO}/tt/runners/adapters/gpt_oss.py", 49, "prefill_trace_default"),
    (f"{GO}/tt/runners/kv_chunk_table.py", 66, "def build_kv_chunk_address_table"),
    (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp",
        421,
        "args.ccl_core_grid_offset.x >=",
    ),
    (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp",
        1304,
        "use_streaming_compute = !fp32_dest_acc_en",
    ),
    (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp",
        1306,
        "!kv_pad_rotation_enabled || use_streaming_compute",
    ),
    # --- P5.4-P5.6 (this session): every NEW path:line claim in tt/mlp.py, tt/attention/* and the
    # three gate tests. Promoted into CITES rather than left to pass 2, because pass 2 only checks
    # that a doc ref's line is IN RANGE — it cannot tell a right line from a wrong one. Four refs
    # in this session's first draft of tt/attention/operations.py were "resolved" by pass 2 while
    # pointing at the wrong lines (they carried a +209 offset from a `cat -n a.py b.py` read); only
    # the out-of-range ones were caught. Content-checking is what CITES is for.
    # P5.4 — tt/mlp.py
    (f"{M3}/tt/dense_mlp.py", 92, "swiglu(gate, up, self.swiglu_cfg)"),
    (f"{M3}/tt/dense_mlp.py", 99, "if self.mesh_config.tp > 1"),
    (f"{M3}/tt/dense_mlp.py", 100, "if self.scatter_output"),
    (f"{M3}/tt/dense_mlp.py", 109, "out = scattered"),
    ("ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp", 1469, 'nb::arg("input_tensor_a_activations")'),
    (
        "ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp",
        2082,
        'bind_binary_operation_with_fast_approx<"multiply">',
    ),
    # P5.5 — tt/attention/{config,weights,operations,prefill,__init__,dense_sp}.py
    (f"{GO}/tt/attention/config.py", 38, "softmax scale 1/sqrt(head_dim)"),
    (f"{GO}/tt/attention/config.py", 62, "prefill_q_chunk_size_small: int = 32"),
    (f"{GO}/tt/attention/config.py", 66, "prefill_threshold: int = 2048"),
    (f"{GO}/tt/attention/config.py", 102, "def get_compute_kernel_config"),
    (f"{GO}/tt/attention/config.py", 108, ")"),
    (f"{GO}/tt/attention/operations.py", 14, "def apply_qkv_projection"),
    (f"{GO}/tt/attention/operations.py", 46, "memory_config=ttnn.DRAM_MEMORY_CONFIG"),
    (f"{GO}/tt/attention/operations.py", 88, "is_decode_mode=is_decode_mode"),
    (f"{GO}/tt/attention/operations.py", 105, "def apply_output_projection"),
    (f"{GO}/tt/attention/operations.py", 121, "return out"),
    (f"{GO}/tt/attention/operations.py", 132, "RACES on"),
    (f"{GO}/tt/attention/operations.py", 135, "Remove this gate once the fused-op sync is fixed"),
    (f"{GO}/tt/attention/operations.py", 258, "local_hidden = hidden_size // mesh_config.tp"),
    (f"{GO}/tt/attention/operations.py", 269, "tensor = tensor_sliced"),
    (f"{GO}/tt/attention/prefill.py", 162, "post-RoPE K + raw V"),
    (f"{GO}/tt/attention/prefill.py", 165, "write_kv_chunk casts its own copy"),
    (f"{GO}/tt/attention/prefill.py", 168, "write_kv_chunk("),
    (f"{GO}/tt/attention/prefill.py", 270, ")"),
    (f"{GO}/tt/attention/prefill.py", 300, "apply_allgather_and_slice"),
    (f"{GO}/tt/attention/__init__.py", 79, 'layer_types[layer_idx] == "sliding_attention"'),
    (f"{GO}/tt/attention/__init__.py", 81, "(layer_idx % 2) == 0"),
    (f"{GO}/tt/attention/dense_sp.py", 30, "def _gather_seq_len"),
    (f"{GO}/tt/attention/dense_sp.py", 38, "return max(halo, ttnn.TILE_SIZE)"),
    (f"{GO}/tt/attention/dense_sp.py", 138, "matching update_padded_kv_cache's write"),
    (f"{GO}/tt/attention/dense_sp.py", 141, "kv_cache_batch_idx=slot_idx * num_layers + layer_idx"),
    (f"{GO}/tt/attention/weights.py", 70, "o_proj_cache_suffix"),
    (f"{GO}/tt/attention/weights.py", 133, "o_proj_bias = torch.cat"),
    (f"{GO}/tt/attention/weights.py", 142, "sinks_for_sdpa = None"),
    (f"{TT}/tt/load_checkpoints.py", 458, 'elif "q_proj.weight" in key or "k_proj.weight" in key'),
    (f"{TT}/tt/load_checkpoints.py", 895, "def permute(tensor, n_heads, dim1, dim2)"),
    (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp",
        108,
        "q_chunk_size % tt::constants::TILE_WIDTH == 0",
    ),
    (RJ, 421, "ccl_core_grid_offset.x >="),
    (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp",
        1304,
        "use_streaming_compute",
    ),
    (
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.cpp",
        12,
        "const Tensor& input_tensor_q",
    ),
    (
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.cpp",
        22,
        "if (input_tensor_kv.has_value())",
    ),
    (
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/" "nlp_create_qkv_heads_nanobind.cpp",
        30,
        'nb::arg("num_heads")',
    ),
    (f"{GO}/tests/unit/test_attention_vs_ref.py", 117, "def _torch_attention"),
    (f"{GO}/tests/unit/test_attention_vs_ref.py", 169, '"q": torch.randn(NQ * HEAD_DIM, HIDDEN) * 0.02'),
    # P5.6 — tt/attention/kv_cache.py and its gate
    (f"{GO}/tt/attention/kv_cache.py", 135, "src.deallocate(True)"),
    (f"{GO}/tt/attention/kv_cache.py", 152, ")"),
    (f"{M3}/tests/unit/test_kv_cache_write_vs_ref.py", 10, "post-RoPE K"),
    (f"{M3}/tests/unit/test_kv_cache_write_vs_ref.py", 98, '"q": torch.rand(NQ * HEAD_DIM, HIDDEN) * 0.02'),
    (f"{M3}/tests/unit/test_kv_cache_write_vs_ref.py", 128, "half = ROTARY_DIM // 2"),
    (f"{M3}/tests/unit/test_kv_cache_write_vs_ref.py", 135, "ref_index_k = ref_index_k[..., src]"),
    # `expect_error`'s `message` is a **regex**, not a substring — `DEC-045`. This is the line.
    ("conftest.py", 962, "pytest.raises(error, match=message)"),
    # --- P5.4-P5.6: every `BRINGUP_RECIPE.md:N` this package cites, content-checked (R-016) ------
    (RCP, 244, "A gate with no raw log did not happen"),
    (RCP, 467, "does not describe a **fused** kernel's interior"),
    (RCP, 613, "separate error budgets per stage"),
    (RCP, 617, "standalone probe"),
    (RCP, 652, "0.9925392"),
    (RCP, 653, "0.9917529"),
    (RCP, 654, "38.7x worse"),
    (RCP, 655, "107.6x worse"),
    (RCP, 657, "already enables"),
    (RCP, 660, "two to three orders of magnitude"),
    (RCP, 791, "a test of the model"),
    (RCP, 793, "say so in the gate block"),
    (RCP, 1207, "make any module that cannot honour it"),
    (RCP, 1249, "Give it one, reachable home here"),
    (RCP, 1357, "input_tensor_a_activations"),
    (RCP, 1411, "11 >= 8"),
    (RCP, 1415, "explicit named field defaulting to"),
    (RCP, 1418, "at construction"),
    (RCP, 1429, "0.9475"),
    (RCP, 1456, "Matching it is what lets"),
    (RCP, 54, "G-MOCK-MIG"),
    (RCP, 690, "encode positions as values"),
    (RCP, 1484, "must be `PASS` before P6"),
    (RCP, 924, "dense SwiGLU"),
    (RCP, 2070, "GQA + RoPE + causal SDPA"),
    (RCP, 687, "positional read-back"),
    (RCP, 2072, "decoder layer (integration check)"),
    (RCP, 494, "per-layer step"),
    (RCP, 2117, "never *raise* one after seeing"),
    # --- P6: layer / model assembly + weight loading (G-LAYER, G-WEIGHTS, G-MODEL) -----------
    # Every reference P6's five modules and five test files make, CONTENT-checked. Pass 2 only
    # range-checks a doc ref (`07_RISKS.md` R-016), and this session found the recipe had grown by
    # 31 lines since P5 committed (`git show cbb38d0aa7a --stat`) while the prose refs written
    # against the older numbering still reported `resolved`. Hence the rule P6 works to: a recipe
    # reference is trustworthy only if it is in this list.
    (RCP, 1068, "**Module signature.**"),
    (RCP, 1070, "`mesh_config=`, `ccl_manager=`, `tensor_cache_path=`, `weight_dtype=`"),
    (RCP, 1074, "cache_file_name=get_cache_file_name"),
    (RCP, 1076, "empty `state_dict` when a cache path exists"),
    (RCP, 1078, "mesh shape and the dtype in the cache path"),
    (RCP, 1502, "Do keep a bring-up probe"),
    (RCP, 1504, "its output belongs in"),
    (RCP, 1506, "re-allocation guard for long sequences"),
    (RCP, 1507, "load-bearing for long-context DRAM pressure"),
    (RCP, 1043, "test_decoder_layer_vs_ref.py"),
    (RCP, 1510, "in-test fp32 torch layer"),
    (RCP, 1514, "integration checks and may never substitute"),
    (RCP, 1518, "a dozen other causes also move"),
    (RCP, 1527, "attenuated in"),
    (RCP, 1531, "forbids"),
    (RCP, 926, "map_hf_to_meta_keys"),
    (RCP, 1537, "weight_cache_path(dtype)"),
    (RCP, 1540, "a replicated table is fine for a first pass"),
    (RCP, 1543, "loads the real checkpoint and asserts (a) every"),
    (RCP, 1544, "no silently-unused weights and no missing weights"),
    (RCP, 1545, "cache-only rebuild"),
    (RCP, 1547, "Q/K Meta swizzle and dtype ladder"),
    (RCP, 1548, "bypass `map_hf_to_meta_keys`"),
    (RCP, 1550, "cache-only at TP > 1 is a P8 extension"),
    (RCP, 1556, "prepare_inputs_prefill"),
    (RCP, 1559, "reduced layer count"),
    (RCP, 1560, "hidden-state PCC"),
    (RCP, 1562, "with_lm_head=True"),
    (RCP, 1563, "record the per-layer hidden-state PCC curve"),
    (RCP, 1511, "PCC \u2265 0.999** and **\u2264 8x the computed floor"),
    (RCP, 1512, "0.9471"),
    (RCP, 1564, "from layer 3 onward"),
    (RCP, 1568, "admissible only with self-checks"),
    (f"{GO}/tt/layer.py", 19, "GPT_OSS_DELTA_PROBE"),
    (f"{GO}/tt/layer.py", 22, "def _delta_stats"),
    (f"{GO}/tt/layer.py", 46, "class DecoderLayer"),
    (f"{GO}/tt/layer.py", 96, "layer_types"),
    (f"{GO}/tt/layer.py", 138, "32 * 1024"),
    (f"{GO}/tt/model.py", 31, "def compute_per_device_vocab"),
    (f"{GO}/tt/model.py", 41, "class Model"),
    (f"{GO}/tt/model.py", 64, "hf_config.head_dim"),
    (f"{GO}/tt/model.py", 84, "self.embedding_weight = ttnn.as_tensor"),
    (f"{GO}/tt/model.py", 123, "sampling_splits"),
    (f"{GO}/tt/model.py", 145, "_supports_on_device_sampling"),
    (f"{GO}/tt/model.py", 179, "def _forward_layers_and_head"),
    (f"{GO}/tt/model.py", 195, "on_layer_complete"),
    (f"{GO}/tt/model.py", 210, "if on_layer_complete is not None"),
    (f"{GO}/tt/model.py", 236, "if skip_lm_head"),
    (f"{GO}/tt/model.py", 240, "self.norm(hidden_states)"),
    (f"{GO}/tt/model.py", 241, "self.lm_head_weight"),
    (f"{GO}/tt/model.py", 246, "def prefill_forward"),
    (f"{GO}/tt/model.py", 279, "def prepare_inputs_prefill"),
    (f"{GO}/tt/model.py", 288, "sequence_parallel"),
    (f"{GO}/tt/model.py", 313, "bf16 (not bf8)"),
    (f"{GO}/tt/model.py", 315, "ttnn.embedding"),
    (f"{GO}/tt/model.py", 322, "def process_output_prefill"),
    (f"{GO}/tt/model.py", 326, "get_device_tensors"),
    (f"{GO}/tt/model_config.py", 30, "class ModelArgs"),
    (f"{GO}/tt/model_config.py", 106, "def load_state_dict"),
    (f"{GO}/tt/model_config.py", 138, "AutoModelForCausalLM.from_pretrained"),
    (f"{GO}/tt/model_config.py", 157, "def weight_cache_path"),
    (f"{GO}/tt/model_config.py", 160, "Path(self.model_path)"),
    (f"{GO}/tt/model_config.py", 175, "def get_state_dict_prefix"),
    (f"{M3}/tt/model.py", 87, "class Model"),
    (f"{M3}/tt/model_config.py", 22, "class ModelArgs"),
    (f"{TT}/tt/load_checkpoints.py", 895, "def permute"),
    # `DEC-052`'s two lines: `LlamaModel.forward` builds a `last_hidden_state` and
    # `LlamaForCausalLM` consumes it without re-exposing it, which is why `output_hidden_states`'
    # last element (the POST-final-norm stream) is the only thing on offer from the causal-LM head.
    (HFL, 421, "hidden_states = self.norm(hidden_states)"),
    (HFL, 484, "outputs.last_hidden_state"),
    # --- P7: the chunked runtime, the golden KV scripts, and the engine's REAL call site -----
    # Every one of these is load-bearing: the runtime's whole design rests on the claim that the
    # engine's call site is wider than its contract doc, and `G-RUNTIME` re-derives that claim
    # from the source at test time. These entries make a shift in either file a MISMATCH rather
    # than a silent lie (`R-016`).
    (f"{CP}/runners/prefill_runner.py", 286, "runtime.prefill_chunk"),
    (f"{CP}/runners/prefill_runner.py", 295, ")"),
    (f"{CP}/runners/prefill_runner.py", 301, "runtime.config.is_last_rank"),
    (f"{CP}/runners/prefill_runner.py", 303, "runtime.config.use_trace"),
    (f"{CP}/runners/prefill_runner.py", 397, "PREFILL_TRACE_DIR"),
    (f"{CP}/runners/prefill_runner.py", 477, "hf_config.max_seq_len = MAX_SEQ_LEN"),
    (f"{CP}/runners/prefill_runner.py", 501, "runtime.compile(kv_caches)"),
    (f"{CP}/runners/prefill_runner.py", 570, "runtime.build_kv_chunk_table"),
    (f"{CP}/runners/prefill_runner.py", 616, "kv_migration_base_address"),
    (f"{CP}/runners/prefill_runner.py", 619, "raise RuntimeError"),
    (f"{CP}/runners/prefill_runner.py", 644, "runtime.build_kv_chunk_table"),
    (f"{CP}/runners/prefill_runner.py", 699, "runtime.build_kv_chunk_table"),
    (f"{CP}/runners/prefill_runner.py", 745, "runtime.config.use_trace"),
    (f"{CP}/runners/prefill_runner.py", 746, "runtime.set_d2h_ack_service"),
    (f"{CP}/runners/prefill_runner.py", 752, "runtime.set_layer_completion_sink"),
    (f"{CP}/runners/prefill_runner.py", 773, "capture_trace"),
    # Shifted +183 by P10's edit to this file (`DEC-104` renamed the packed-GQA reader and added
    # `_PACKED_GQA_MODELS` above the dispatcher). The needle is content-checked, so the verifier
    # caught the shift rather than silently resolving in range.
    (f"{CP}/runners/prefill_producer.py", 694, "_read_slot_kv_and_check_pcc_mla"),
    (f"{CP}/docs/ADDING_A_PREFILL_MODEL.md", 101, "The runtime interface"),
    (f"{CP}/docs/ADDING_A_PREFILL_MODEL.md", 116, "chunk_size, max_seq_len, first_layer_idx"),
    (f"{CP}/docs/ADDING_A_PREFILL_MODEL.md", 124, "def make_chunk_input"),
    (f"{CP}/docs/ADDING_A_PREFILL_MODEL.md", 129, "def prefill_chunk"),
    (f"{GO}/tt/tt_prefill_runtime.py", 12, "owns_kv_cache=True"),
    (f"{GO}/tt/tt_prefill_runtime.py", 46, "def resolve_chunk_sizes"),
    (f"{GO}/tt/tt_prefill_runtime.py", 63, "default_chunk_size"),
    (f"{GO}/tt/tt_prefill_runtime.py", 88, "def sp_factor"),
    (f"{GO}/tt/tt_prefill_runtime.py", 92, "def tp_factor"),
    (f"{GO}/tt/tt_prefill_runtime.py", 96, "class TtPrefillRuntime"),
    (f"{GO}/tt/tt_prefill_runtime.py", 174, "def _build_indexed_rope"),
    (f"{GO}/tt/tt_prefill_runtime.py", 194, "def _resolve_kv"),
    (f"{GO}/tt/tt_prefill_runtime.py", 204, "def make_chunk_input"),
    (f"{GO}/tt/tt_prefill_runtime.py", 231, "reshape(sp, 1, s_local)"),
    (f"{M3}/tt/tt_prefill_runtime.py", 49, "chunk_size: int = 5120"),
    (f"{M3}/tt/tt_prefill_runtime.py", 96, "max_seq_len % config.chunk_size"),
    (f"{M3}/scripts/generate_golden_kv_cache.py", 29, "metadata.json"),
    (f"{M3}/scripts/generate_golden_kv_cache.py", 143, 'default="bfloat16"'),
    (f"{M3}/scripts/verify_golden_kv.py", 26, "def verify_trace"),
    (f"{GO}/scripts/generate_golden_kv_cache.py", 111, 'default="bfloat16"'),
    (f"{GO}/scripts/verify_golden_kv.py", 111, "key_tensor"),
    # The golden's K/V come out of HF's own cache update, which happens AFTER RoPE on K and with V
    # untouched — the claim `scripts/generate_golden_kv_cache.py` rests on.
    (HFL, 270, "past_key_values.update"),
    (HFL, 399, "create_causal_mask"),
    (HFL, 408, "position_embeddings = self.rotary_emb"),
    # --- P7: the recipe's own P7 section, content-checked (`R-017`) --------------------------
    (RCP, 16, "Non-goals for this iteration"),
    (RCP, 1587, "Store the golden at"),
    (RCP, 1592, "host-only** structural and content check"),
    (RCP, 1647, "A chunked prefill differs from a one-shot in exactly three places"),
    (RCP, 1656, "Gate `G-CHUNK`"),
    (RCP, 1659, "one head at a time"),
    (RCP, 1663, "chunked vs one-shot: PCC"),
    (RCP, 1668, "assert the layer-0 ratio against a **complete** floor"),
    (RCP, 1674, "negative control: rope every chunk at"),
    (RCP, 1682, "Gate `G-GOLDEN`"),
    (RCP, 1687, "Gate `G-RUNTIME`"),
    (RCP, 1692, "Delta 3 cannot run here"),
    (RCP, 1695, "refuse** the unsupported single-card configuration"),
    (RCP, 1875, "signature is incomplete"),
    (RCP, 52, "G-GOLDEN"),
    (RCP, 52, "G-RUNTIME"),
    (RCP, 53, "G-CHUNK-ATTN"),
    # The remaining `build_kv_chunk_table` call sites and the layer-ack one, cited by the G-RUNTIME
    # gate block and `R-024`'s table.
    (f"{CP}/runners/prefill_runner.py", 655, "runtime.build_kv_chunk_table"),
    (f"{CP}/runners/prefill_runner.py", 674, "runtime.build_kv_chunk_table"),
    (f"{CP}/runners/prefill_runner.py", 768, "runtime.set_layer_ack_channel"),
    # --- P8: the fabric, the submesh rule, and the ring op's own asserts ----------------------
    # Every one of these is load-bearing for a P8 verdict, so it is CONTENT-checked here rather
    # than left to the doc pass's range check (`07_RISKS.md` R-016).
    ("tt_metal/api/tt-metalium/mesh_device.hpp", 305, "void quiesce_devices();"),
    ("tt_metal/api/tt-metalium/mesh_device.hpp", 307, "create_submesh("),
    ("tt_metal/fabric/topology_mapper.cpp", 540, "mapping_result.success"),
    ("tt_metal/fabric/topology_mapper.cpp", 544, "mapping_result.error_message"),
    ("tt_metal/fabric/mesh_graph.cpp", 447, "requires_more_connectivity"),
    ("tt_metal/fabric/mesh_graph.cpp", 450, "FabricConfig can only restrict topology"),
    ("tt_metal/fabric/fabric.cpp", 171, "forwarding_direction.has_value()"),
    ("tt_metal/fabric/fabric.cpp", 172, "Could not find any forwarding direction"),
    ("tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp", 200, "Fabric Router Sync: Timeout after"),
    (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp",
        1304,
        "use_streaming_compute = !fp32_dest_acc_en",
    ),
    (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp",
        1306,
        "!kv_pad_rotation_enabled || use_streaming_compute",
    ),
    (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp",
        1307,
        "kv_actual_isl requires the ring-joint streaming compute path",
    ),
    (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp",
        274,
        "new_actual_isl <= chunk_capacity",
    ),
    (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp",
        275,
        "KV-pad-aware rotation expects current valid Q to fit in one fixed chunk",
    ),
    (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp",
        421,
        "args.ccl_core_grid_offset.x >= args.program_config.value().compute_with_storage_grid_size.x",
    ),
    # The single-galaxy mesh descriptors, and the proof that the two the recipe names are not
    # single-galaxy (`DEC-071`, `R-033`).
    (
        "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto",
        6,
        "dim_types: [ RING, RING ]",
    ),
    (
        "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto",
        7,
        "host_topology   { dims: [ 1, 1 ] }",
    ),
    (
        "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto",
        6,
        "device_topology { dims: [ 8, 4 ] }",
    ),
    (
        "tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_sp4_torus_xy_graph_descriptor.textproto",
        8,
        "host_topology   { dims: [ 4, 1 ] }",
    ),
    (
        "tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto",
        7,
        "host_topology   { dims: [ 4, 1 ] }",
    ),
    # The templates P8 ports from.
    (f"{GO}/tt/attention/dense_sp.py", 41, "def dense_sp_attention("),
    (f"{GO}/tt/attention/dense_sp.py", 30, "def _gather_seq_len("),
    (f"{GO}/tt/attention/dense_sp.py", 138, "Fold the layer into the cache batch index"),
    (f"{GO}/tt/attention/prefill.py", 191, "use_cache_backed_ring = cached_len > 0 or kv_cache.max_seq_len"),
    (f"{GO}/tt/attention/prefill.py", 195, "ttnn.CoreCoord(grid.x - 1, grid.y)"),
    (f"{GO}/tt/attention/prefill.py", 234, "full_seq_len = seq_len * sp"),
    (f"{GO}/tt/attention/prefill.py", 254, "ttnn.multiply(tt_sdpa_out, 1.0 / sp)"),
    (f"{GO}/utils/general_utils.py", 33, "if mesh_device.shape[0] == 1:"),
    (f"{GO}/tests/galaxy_prefill_kv_pcc.py", 26, "single_bh_galaxy_torus_xy_graph_descriptor.textproto"),
    ("models/demos/minimax_m3/README.md", 48, "single_bh_galaxy_mesh_graph_descriptor.textproto"),
    ("models/demos/minimax_m3/tests/test_factory.py", 89, "def parametrize_mesh_with_fabric"),
    ("models/demos/deepseek_v3_d_p/tt/mla/utils.py", 65, "def block_cyclic_reorder("),
    ("models/demos/deepseek_v3_d_p/tt/mla/utils.py", 89, "local row lr on chip c holds global position"),
    # The recipe passages P8's gate blocks and decisions quote.
    (RCP, 90, "The Ring topology P8 needs the torus"),
    (RCP, 1708, "open **submeshes**, never a top-level partial mesh"),
    (RCP, 1730, "overlapping submeshes need `quiesce_devices()`"),
    (RCP, 1742, "A hang is not contained"),
    (RCP, 1749, "The first diagnosis of that hang was wrong"),
    (RCP, 1766, "`get_default_num_links` returns **1** for any"),
    (RCP, 1796, "G-FABRIC-MATRIX`** — the (mesh, topology, links, axis) sweep"),
    (RCP, 1799, "G-KV-TP8`** — the model \u2192 cache path at TP=8"),
    (RCP, 1806, "G-SP-RING`** — `dense_sp_attention` **alone**"),
    (RCP, 1809, "G-CHUNK-ATTN`** — the P7 blocker, now runnable"),
    (RCP, 1827, "G-TP-PARITY`** — for each module"),
    (RCP, 1832, "the multi-device output is a token slice"),
    (RCP, 1841, "G-RACE`** — run the full-model KV PCC harness"),
    (RCP, 1846, "G-SEMAPHORE`** — assert `CCLManager` allocates its CCL state once"),
    (RCP, 1849, "G-MESH-KV`** — `tests/galaxy_prefill_kv_pcc.py` on the target mesh"),
    (RCP, 1853, "G-WEIGHTS` (P8 extension)"),
    (RCP, 1785, "The ring op requires `fp32_dest_acc_en=False`, structurally"),
    (RCP, 1692, "Delta 3 cannot run here"),
    # --- P10: every load-bearing reference this phase makes, content-checked -------------------
    # The engine's adapter contract. Pass 2 only range-checks (`R-016`), and these are the
    # signatures and line numbers the adapter, the runtime and two gates are written against.
    (ADP, 46, "class PrefillRunParams:"),
    (ADP, 95, "class KvCaches(ABC):"),
    (ADP, 104, "class PrefillModelAdapter(ABC):"),
    (ADP, 116, "hf_model_default: str  # config.json dir; PREFILL_HF_MODEL overrides"),
    (ADP, 143, "def load_hf_config(self)"),
    (ADP, 148, "def default_sparse_kv_cache_format(self)"),
    (ADP, 161, "def weight_cache_path(self, mesh_shape: tuple)"),
    (ADP, 166, "def allocate_kv_cache(self, *, mesh_device"),
    (ADP, 173, "def layer_split_boundaries(self, num_layers: int)"),
    (ADP, 183, "def build_runtime(self, *, mesh_device"),
    (ADP, 291, '"llama31_8b_d_p": "models.demos.llama31_8b_d_p.tt.runners.adapters.llama:LlamaPrefillAdapter"'),
    # The engine's runner: every call site the runtime and the adapter are written to.
    (RNR, 62, "MODEL_CFG = ADAPTER.model_config"),
    (RNR, 80, 'KV_ONLY_LAST_LAYER = os.environ.get("PREFILL_KV_ONLY_LAST_LAYER", "1") == "1"'),
    (RNR, 286, "out = runtime.prefill_chunk("),
    (RNR, 303, "if runtime.config.use_trace:"),
    (RNR, 377, "str(ADAPTER.weight_cache_path(GLOBAL_MESH_SHAPE))"),
    (RNR, 472, "mesh_device = open_mesh_device("),
    (RNR, 477, "hf_config.max_seq_len = MAX_SEQ_LEN"),
    (RNR, 489, "num_links=2 if is_blackhole() else 1"),
    (RNR, 501, "runtime.compile(kv_caches)"),
    (RNR, 516, "d2d_activation_width = hf_config.hidden_size"),
    (RNR, 570, "runtime.build_kv_chunk_table(kv_caches, path=_mock_table_path)"),
    (RNR, 613, '_multi_cache_runtime = hasattr(runtime, "kv_migration_stages")'),
    (RNR, 617, "kv_stages = [KvCacheStage(runtime.kv_migration_base_address(kv_caches)"),
    (RNR, 746, "runtime.set_d2h_ack_service(d2h_service)"),
    (RNR, 752, "runtime.set_layer_completion_sink("),
    (RNR, 768, "runtime.set_layer_ack_channel(ack_channel)"),
    # The producer: the read-back branch P10 generalised, and the two facts the gate rests on.
    (PRD, 89, 'ADAPTER = get_adapter(os.environ.get("PREFILL_MODEL", DEFAULT_MODEL))'),
    (PRD, 508, '_PACKED_GQA_MODELS = ("gpt_oss_d_p", "llama31_8b_d_p")'),
    (PRD, 517, "if ADAPTER.name in _PACKED_GQA_MODELS:"),
    (PRD, 542, "def _read_slot_kv_and_check_pcc_packed_gqa("),
    (PRD, 694, "def _read_slot_kv_and_check_pcc_mla("),
    # The packed-GQA reader's own internals, which `G-ADAPTER` and `G-KV-TABLE` are written against.
    (PRD, 128, 'table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH"'),
    (PRD, 531, "from models.demos.minimax_m3.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK"),
    (PRD, 535, "loc = table.lookup(layer, pos, slot_id, config_id)"),
    (PRD, 537, "raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)"),
    (PRD, 550, "mc = ADAPTER.model_config"),
    (PRD, 552, 'rotary_dim = getattr(mc, "ROTARY_DIM", head_dim)'),
    (PRD, 565, "for layer in range(NUM_LAYERS):"),
    (PRD, 588, 'g_k = h.get_tensor(f"key_cache_layer_{layer}")'),
    (PRD, 903, "def _load_token_pool(trace_dir, num_tokens: int) -> list:"),
    (PRD, 1068, "CHECK_PCC=1 but LayerAck channel missing"),
    (PRD, 1115, "_drain_layer_acks(ack_channel, NUM_LAYERS * stats.total_pushes)"),
    # The two engine documents, at the passages `08_PREFILL_INTEGRATION.md` section 4 quotes.
    (DOC_ADD, 129, "def prefill_chunk(self, input_tensor, kv_cache, *, slot_id"),
    (DOC_ADD, 146, "def build_kv_chunk_table(self, kv_cache, path: str)"),
    (DOC_ADD, 241, "layout needs a branch in `_read_slot_kv_and_check_pcc`"),
    (DOC_MIG, 544, "**none** | nothing is decoded"),
    # The shared migration helpers, and the template this phase imports rather than copies.
    (MIG, 220, "def serialize_kv_chunk_table("),
    (MIG, 240, "def serialize_prebuilt_kv_chunk_table(*, table, path: str)"),
    (MIG, 294, "def allgather_kv_stage_layout(mesh_device, kv_base_addr"),
    (GOX_TBL, 66, "def build_kv_chunk_address_table("),
    (GOX_TBL, 138, "base_addr = tensor.buffer_address()"),
    (GOX_ADP, 41, "class GptOssPrefillAdapter(PrefillModelAdapter):"),
    (GOX_RT, 370, "def kv_migration_base_address(self, kv_caches) -> int:"),
    # This package's own P10 files.
    (f"{PKG}/tt/runners/adapters/llama.py", 60, "class Llama31_8BConfig:"),
    (f"{PKG}/tt/runners/kv_chunk_table.py", 80, "def assert_single_rank_stage("),
    # The recipe's P10 instructions this phase followed or deviated from.
    (RCP, 1905, '`{ "env": { "PREFILL_MODEL": "llama31_8b_d_p" } }`'),
    (RCP, 1898, "never** from `os.environ`"),
    (RCP, 1957, "G-ADAPTER`** — the checklist at the end of"),
    (RCP, 1977, "`PASS` = every chunk accepted and served"),
    (RCP, 1981, "PREFILL_STANDALONE_CHUNKED_PCC` = 0.93"),
    (RCP, 1986, "G-KV-TABLE`** — the address table on its own"),
    (RCP, 1992, "G-LOOPBACK`** (= the doc's **Gate 2**)"),
    # `runner_utils.py` is cited by basename in `06_GATES.md` and `tests/unit/test_prefill_adapter.py`.
    # A basename resolves only if some CITES row or doc ref names the full path, so these two rows are
    # what make those abbreviated refs resolvable AND content-checked.
    (f"{CP}/runners/runner_utils.py", 37, "fabric_config = ttnn.FabricConfig.FABRIC_1D if sp <= 8"),
    (f"{CP}/runners/runner_utils.py", 41, "max_payload_size=model_cfg.FABRIC_PAYLOAD_SIZE,"),
]

# ---------------------------------------------------------------------------------------------
# P9 addition (`DEC-120`) — the recipe lines P9 re-pointed 247 prose refs onto.
# Pass 2 only RANGE-checks a prose `path:line` (`R-016`), so a ref that drifts with an out-of-band
# recipe edit stays green while pointing at unrelated text — which is exactly what P9 found across
# 247 refs (`R-053`). Every line P9 re-pointed *to* is content-checked here, so the next drift is
# loud on the first line rather than silent on all of them. §1.6: "Put every load-bearing reference
# in `CITES`, including references *into the recipe itself*."
CITES += [
    (RECIPE, 80, "BH Galaxy mesh descriptors live in"),
    (RECIPE, 193, "Which gates in an external component's ladder are"),
    (RECIPE, 260, "Commit the raw logs, and check that you actually did"),
    (RECIPE, 273, "if you hand-built the skeleton, you do not have it"),
    (RECIPE, 1282, "Run it on both random weights and the real layer-0 norm gain"),
    (RECIPE, 1287, "Running both is the resolution"),
    (RECIPE, 1340, "Also assert the llama3 scaling actually took"),
    (RECIPE, 1349, "a test that passes with scaling silently disabled is worthless"),
    (RECIPE, 1428, "Negative control: Q/K weights loaded *without* the"),
    (RECIPE, 1595, "compare a device KV read-back"),
    (RECIPE, 1598, "reporting min/mean PCC per layer for K and V."),
    (RECIPE, 1641, "the chunked-vs-one-shot equivalence test"),
    (RECIPE, 1642, "for **deltas 1 and 2 only**"),
    (RECIPE, 1650, "| delta | what changes | owned by |"),
    (RECIPE, 1654, "read back out of the cache"),
    (RECIPE, 1663, "chunked vs one-shot: PCC"),
    (RECIPE, 1670, 'Record the cache-write-only ("storage") floor beside it'),
    (RECIPE, 1675, "the mutual **K** PCC must collapse"),
    (RECIPE, 2071, "cache **primitive**: write correctness"),
    (RECIPE, 2082, "collectives are exact"),
]

# P9 (`DEC-121`, `R-055`): the one line that makes the reference adapter 64.6x more expensive
# than this one. Content-checked because `G-CLEAN` item 8's whole finding rests on it, and
# because P9 first wrote it as `:26` — off by one, which is precisely `R-053`'s failure mode.
CITES += [
    (f"{GO}/tt/runners/adapters/gpt_oss.py", 25, "from models.common.utility_functions import is_blackhole"),
]

# P9 (`DEC-124`): the four references `README.md` rests its two central answers on — why not
# `models/common/`, and why TP is an equality. The README is the one file a reader is guaranteed
# to open, and pass 2 only range-checks it.
CITES += [
    (f"{CM}/modules/mlp/mlp_2d.py", 461, "cluster_axis=0"),
    (f"{CM}/models/llama3_8b/model.py", 890, "only supports 1D mesh topologies"),
    (f"{GO}/tt/attention/kv_cache.py", 95, "Per-chip cache is one head"),
    (f"{M3}/README.md", 48, "single_bh_galaxy_mesh_graph_descriptor.textproto"),
]


# DOCS — every markdown file whose `path:line` references should be range-checked.
# Include your recipe and your README: leaving the recipe out is how a stale citation survived
# a whole run in the original.
DOCS = [
    f"{PKG}/README.md",
    f"{PKG}/bringup_log/00_MODEL_CARD.md",
    f"{PKG}/bringup_log/01_REFERENCE.md",
    f"{PKG}/bringup_log/02_SURVEY.md",
    f"{PKG}/bringup_log/03_OUTLINE.md",
    f"{PKG}/bringup_log/04_CCL_PLAN.md",
    f"{PKG}/bringup_log/05_DECISIONS.md",
    f"{PKG}/bringup_log/06_GATES.md",
    f"{PKG}/bringup_log/07_RISKS.md",
    f"{PKG}/bringup_log/08_PREFILL_INTEGRATION.md",
    # The recipe itself: §1.6 says leaving it out is how a stale citation survived a whole run.
    # It lives in the kit, not in the package, for this bring-up (the package does not vendor it).
    "models/demos/common/bringup/BRINGUP_RECIPE.md",
]


# P6 addition (Appendix F.7 / the original run’s DEC-035; not a decision in this package’s log): the package's **own Python docstrings** carry as many
# load-bearing `path:line` refs as the logs do, and none of them were checked. They are also where
# citation shadowing bites hardest: `tt/layer.py`, `tt/model.py` and `tt/embedding.py` now shadow
# gpt-oss files of the same basename, so a bare `model.py:211` in a docstring is genuinely ambiguous
# and pass 2's AMBIGUOUS handling (line must be in range for *every* candidate) is exactly the right
# check for it. Globbed rather than listed so a new file cannot be added without being scanned.
DOCS += sorted(
    os.path.relpath(str(path), ROOT)
    # P7 addition: `scripts/*.py`. The two golden-KV scripts carry as many load-bearing `path:line`
    # refs as any module and were the only Python in the package pass 2 could not see (Appendix F.7
    # says extend the verifier every phase). `verify_citations.py` itself is matched by the glob and
    # is harmless: its own citations are tuples, not backtick-quoted refs, so the regex skips them.
    for pattern in (
        "tt/*.py",
        "tt/*/*.py",
        "tests/*.py",
        "tests/unit/*.py",
        "utils/*.py",
        "scripts/*.py",
        "conftest.py",
    )
    for path in __import__("pathlib").Path(os.path.join(ROOT, PKG)).glob(pattern)
)
# package-relative shorthands used in the logs
DOC_PREFIXES = {
    # The recipe is NOT vendored into this package (the P3 tree lists it there; this bring-up reads
    # it from the kit instead — `DEC-002`), so the shorthand resolves to the kit copy.
    "BRINGUP_RECIPE.md": "models/demos/common/bringup/BRINGUP_RECIPE.md",
    "00_MODEL_CARD.md": f"{PKG}/bringup_log/00_MODEL_CARD.md",
    "01_REFERENCE.md": f"{PKG}/bringup_log/01_REFERENCE.md",
    "02_SURVEY.md": f"{PKG}/bringup_log/02_SURVEY.md",
    "03_OUTLINE.md": f"{PKG}/bringup_log/03_OUTLINE.md",
    "04_CCL_PLAN.md": f"{PKG}/bringup_log/04_CCL_PLAN.md",
    "05_DECISIONS.md": f"{PKG}/bringup_log/05_DECISIONS.md",
    "06_GATES.md": f"{PKG}/bringup_log/06_GATES.md",
    "07_RISKS.md": f"{PKG}/bringup_log/07_RISKS.md",
}
_REF = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|cpp|hpp|md|json|textproto|yaml)):(\d+)(?:-(\d+))?`")

# P9 addition (the original run’s DEC-120; not a decision in this package’s log): `02_SURVEY.md` and the recipe write citations with the *same* one- and
# two-letter aliases this script defines above (`GO/tt/ccl.py:55`, `TT/tt/common.py:489`) — 86 of
# them. Every one of those resolved before this map existed, but only by falling through to the
# ambiguous-basename path, i.e. by luck: the day a second `ccl.py` is cited anywhere the ref flips
# to a failure that has nothing to do with the ref being wrong. Expanding the alias makes them
# LITERAL, which is what a `path:line` is supposed to be.
_ALIASES = {"TT/": TT, "GO/": GO, "M3/": M3, "DS/": DS, "CP/": CP, "CM/": CM, "LL/": LL}


# P5: the logs also use abbreviated forms — a bare basename (`common.py:564`, continuing an earlier
# full citation) or a partial path (`gpt_oss_d_p/tt/config.py:55`). Resolving them instead of
# reporting them "unresolved" is what makes pass 2 cover the decision log and the gate ledger, where
# the shorthand is the norm. Ambiguous basenames are REPORTED, not silently dropped.
# P6 addition: the package's own root, so a package-relative ref (tt/config.py line 134,
# tests/unit/test_reference_model.py line 136 — the shorthand every file in this package uses for its
# own siblings) resolves LITERALLY instead of falling through to the ambiguous-basename path. Before
# this, tt/config.py line 134 was matched against gpt_oss_d_p/tt/attention/config.py (108 lines) and
# reported out of range — a false positive from citation shadowing, and it must be listed FIRST so
# a package-local file wins over a same-named file elsewhere in the tree (original run’s DEC-035).
_PARTIAL_PREFIXES = (
    f"{PKG}/",
    "models/demos/",
    "models/",
    "python_env/lib/python3.12/site-packages/",
    "",
)


def _basename_index():
    """basename -> the set of full repo paths that basename could mean, from CITES + the docs."""
    index = {}
    candidates = {path for path, _, _ in CITES}
    for doc in DOCS:
        full = os.path.join(ROOT, doc)
        if not os.path.isfile(full):
            continue
        for m in _REF.finditer(open(full, errors="replace").read()):
            candidates.add(m.group(1))
    for cand in candidates:
        if "/" in cand and os.path.isfile(os.path.join(ROOT, cand)):
            index.setdefault(os.path.basename(cand), set()).add(cand)
    return index


def _resolve(path, index):
    """Return (resolved_path, note). `note` is non-empty when the resolution was not literal."""
    path = DOC_PREFIXES.get(path, path)
    for alias, real in _ALIASES.items():
        if path.startswith(alias):
            expanded = real + "/" + path[len(alias) :]
            if os.path.isfile(os.path.join(ROOT, expanded)):
                return expanded, f"alias {alias} -> {expanded}"
    if os.path.isfile(os.path.join(ROOT, path)):
        return path, ""
    stripped = path.lstrip("./")
    for prefix in _PARTIAL_PREFIXES:
        cand = prefix + stripped
        if os.path.isfile(os.path.join(ROOT, cand)):
            return cand, f"partial path -> {cand}"
    hits = index.get(os.path.basename(path), set())
    if len(hits) == 1:
        only = next(iter(hits))
        return only, f"basename -> {only}"
    if len(hits) > 1:
        # A bare basename shared by several real files (`model_config.py:19`) carries less
        # information than a full path. Rather than drop it, require the line to be IN RANGE for
        # EVERY candidate: then whichever file the author meant, the reference resolves.
        return sorted(hits), f"AMBIGUOUS basename, {len(hits)} candidates"
    return None, ""


def scan_docs():
    ok = bad = 0
    failures = []
    refs = set()
    index = _basename_index()
    for doc in DOCS:
        full = os.path.join(ROOT, doc)
        if not os.path.isfile(full):
            continue
        for m in _REF.finditer(open(full, errors="replace").read()):
            path, lo, hi = m.group(1), int(m.group(2)), m.group(3)
            resolved, note = _resolve(path, index)
            if resolved is None:
                bad += 1
                failures.append(f"DOC UNRESOLVED  {doc}: {path}:{lo}" + (f"  ({note})" if note else ""))
                continue
            hi_i = int(hi) if hi else lo
            for cand in resolved if isinstance(resolved, list) else [resolved]:
                refs.add((doc, cand, lo, hi_i))
    for doc, path, lo, hi in sorted(refs):
        target = os.path.join(ROOT, path)
        if not os.path.isfile(target):
            bad += 1
            failures.append(f"DOC UNRESOLVED  {doc}: {path}:{lo}")
            continue
        n = sum(1 for _ in open(target, errors="replace"))
        if hi > n:
            bad += 1
            failures.append(f"DOC OUT OF RANGE  {doc}: {path}:{lo}-{hi} (file has {n} lines)")
        else:
            ok += 1
    print(f"doc refs scanned  : {ok + bad}")
    print(f"  resolved        : {ok}")
    print(f"  unresolved      : {bad}")
    if failures:
        print("\nDOC FAILURES:")
        for f in failures:
            print("  " + f)
    return bad


# ---------------------------------------------------------------------------------------------
# Pass 3 (P8 addition, §1.6 "extend it in every phase"): **every cited raw-log artefact must
# exist.**
#
# Passes 1 and 2 both key on `path:line`, so a bare `` `G-FOO_<timestamp>.log` `` under `raw/` — which is
# precisely how the ledger cites its evidence — was scanned by **neither**. That is not a
# hypothetical gap: this pass found two dangling references on its first run, both written by an
# earlier session (`07_RISKS.md` R-042). And the recipe's definition of a gate is exactly this
# artefact: "A gate with no raw log did not happen" (`BRINGUP_RECIPE.md:244`), reinforced by
# Appendix C item 2. A ledger row citing a file that is not there is a `PASS` with no evidence.
#
# `.log.gz` is accepted for `.log`, because oversized logs are gzipped rather than trimmed
# (`LANDMINES.md`, `07_RISKS.md` R-040) and compression is lossless, so the evidence is the same
# bytes.
# ---------------------------------------------------------------------------------------------
_ARTEFACT = re.compile(r"`(raw/[A-Za-z0-9_.\-]+\.(?:log|json))(?:\.gz)?`")


def scan_artefacts():
    ok = bad = 0
    failures = []
    seen = set()
    for doc in DOCS:
        full = os.path.join(ROOT, doc)
        if not os.path.isfile(full) or "/bringup_log/" not in doc.replace(os.sep, "/"):
            continue
        base = os.path.dirname(full)
        for m in _ARTEFACT.finditer(open(full, errors="replace").read()):
            rel = m.group(1)
            if (doc, rel) in seen:
                continue
            seen.add((doc, rel))
            target = os.path.join(base, rel)
            if os.path.isfile(target) or os.path.isfile(target + ".gz"):
                ok += 1
            else:
                bad += 1
                failures.append(f"ARTEFACT MISSING  {doc}: {rel}")
    print(f"raw artefacts cited: {ok + bad}")
    print(f"  present         : {ok}")
    print(f"  missing         : {bad}")
    if failures:
        print("\nARTEFACT FAILURES:")
        for f in failures:
            print("  " + f)
    return bad


# ---------------------------------------------------------------------------------------------
# Pass 4 (P9, `DEC-120`) — pin the recipe, because pass 2 cannot check a prose ref's *content*.
#
# `R-017` fired three times in this run: the kit's recipe was edited while a phase session was
# live, and every prose `BRINGUP_RECIPE.md:N` ref shifted. Pass 1 catches only the ~120 refs that
# are in `CITES`; pass 2 range-checks the rest and reports them `resolved` while they point at
# unrelated text. P9 measured the damage: 247 of 317 refs had drifted, none of them reported.
#
# This pass cannot tell whether a ref is *right*. It tells you when the ground moved, which is the
# one thing nothing else in this script does. On a mismatch: re-validate every prose recipe ref
# (P9 did it by mapping each ref through `git blame` + `difflib` against the recipe version the
# citing line was written against), then update the fingerprint in the same commit.
RECIPE_SHA256 = "b08634f97719ebea22620db845e83893f07e438576ddcdeea002f5ead1e990fd"
RECIPE_LINES = 2235


def check_recipe_fingerprint():
    import hashlib

    full = os.path.join(ROOT, RECIPE)
    raw = open(full, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    n = raw.decode(errors="replace").count("\n")
    print(f"recipe fingerprint: {RECIPE} ({n} lines)")
    if got == RECIPE_SHA256 and n == RECIPE_LINES:
        print("  MATCHES the version every prose ref was validated against")
        return 0
    print(f"  CHANGED  recorded {RECIPE_SHA256[:16]}... / {RECIPE_LINES} lines")
    print(f"           found    {got[:16]}... / {n} lines")
    print("  Every prose `BRINGUP_RECIPE.md:N` ref in this package is now UNVERIFIED: pass 2")
    print("  range-checks them, so they will still report `resolved` while pointing elsewhere.")
    print("  Re-validate them and update RECIPE_SHA256 / RECIPE_LINES in the same commit (R-017).")
    return 1


def main():
    ok = bad = missing = 0
    failures = []
    for path, lineno, needle in CITES:
        full = os.path.join(ROOT, path)
        if not os.path.isfile(full):
            missing += 1
            failures.append(f"MISSING FILE  {path}")
            continue
        with open(full, errors="replace") as f:
            lines = f.readlines()
        if lineno > len(lines):
            bad += 1
            failures.append(f"OUT OF RANGE  {path}:{lineno} (file has {len(lines)} lines)")
            continue
        line = lines[lineno - 1]
        if needle in line:
            ok += 1
        else:
            bad += 1
            # look for the needle nearby to report the true line
            near = [i + 1 for i, l in enumerate(lines) if needle in l]
            failures.append(
                f"MISMATCH      {path}:{lineno} expected {needle!r}\n"
                f"                got: {line.rstrip()!r}\n"
                f"                needle actually on lines: {near[:6]}"
            )
    print(f"citations checked : {len(CITES)}")
    print(f"  verified        : {ok}")
    print(f"  mismatched      : {bad}")
    print(f"  missing files   : {missing}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
    doc_bad = scan_docs()
    artefact_bad = scan_artefacts()
    fingerprint_bad = check_recipe_fingerprint()
    return 0 if bad == 0 and missing == 0 and doc_bad == 0 and artefact_bad == 0 and fingerprint_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
