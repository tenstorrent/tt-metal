# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Re-verify every load-bearing `path:line` citation used by the bring-up logs.

Powers check [5] of gate `G-SURVEY`. Each entry in `CITES` is
`(path_relative_to_repo_root, line_number, substring_that_must_be_on_that_line)`; the script reads
the file and reports the true line numbers of any needle whose claimed line is wrong.

Run:
    python models/demos/llama31_8b_d_p/scripts/verify_citations.py

Exit 0 iff every citation verifies. **Extend `CITES` in every later phase** — P2 found five wrong
line numbers in BRINGUP_RECIPE.md and five in the survey's own first draft, and an unverified
`path:line` is worth less than no citation because it reads as authoritative.
"""

import os
import re
import sys

# repo root = four levels up from models/demos/llama31_8b_d_p/scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

TT = "models/tt_transformers"
GO = "models/demos/gpt_oss_d_p"
M3 = "models/demos/minimax_m3"
DS = "models/demos/deepseek_v3_d_p"
CP = "models/demos/common/prefill"
CM = "models/common"
LL = "models/demos/llama31_8b_d_p"

# (file, line, substring that MUST appear on that line)
CITES = [
    # row 1-2 RMSNorm
    (f"{GO}/tt/rms_norm.py", 17, "class RMSNorm"),
    (f"{GO}/tt/rms_norm.py", 49, "def forward"),
    (f"{GO}/tt/rms_norm.py", 33, "is_distributed = False"),
    (f"{GO}/tt/rms_norm.py", 67, "rms_norm_pre_all_gather"),
    (f"{GO}/tt/rms_norm.py", 94, "ttnn.rms_norm"),
    (f"{GO}/tt/rms_norm.py", 27, "TILE_SIZE"),
    # row 3-5 RoPE
    (f"{TT}/tt/common.py", 489, "def precompute_freqs"),
    (f"{TT}/tt/common.py", 437, "def apply_scaling"),
    (f"{TT}/tt/common.py", 405, "def compute_llama3_parameters"),
    (f"{TT}/tt/common.py", 407, "low_freq_factor = 1"),
    (f"{TT}/tt/common.py", 408, "high_freq_factor = 4"),
    (f"{TT}/tt/common.py", 534, "def get_prefill_rot_mat"),
    (f"{TT}/tt/common.py", 562, "def get_rot_transformation_mat"),
    (f"{TT}/tt/common.py", 525, "def gather_cos_sin"),
    (f"{TT}/tt/common.py", 165, "def get_rope_theta"),
    (f"{TT}/tt/common.py", 183, "def get_rope_scaling"),
    # row 6 rope apply
    (f"{GO}/tt/attention/operations.py", 87, "rotary_embedding_llama"),
    (f"{M3}/tt/attention/operations.py", 93, "rotary_embedding"),
    (
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama_nanobind.cpp",
        18,
        "rotary_embedding_llama",
    ),
    (
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/rotary_embedding_hf_nanobind.cpp",
        18,
        "rotary_embedding_hf",
    ),
    # row 7, 24 key mapping
    (f"{TT}/tt/load_checkpoints.py", 451, "def convert_hf_qkv_to_meta_format"),
    (f"{TT}/tt/load_checkpoints.py", 891, "def reverse_permute"),
    (f"{TT}/tt/load_checkpoints.py", 895, "def permute"),
    (f"{TT}/tt/load_checkpoints.py", 18, "def load_hf_state_dict"),
    (f"{TT}/tt/load_checkpoints.py", 46, "def load_hf_state_dict_filtered"),
    (f"{TT}/tt/load_checkpoints.py", 193, "def convert_hf_to_meta"),
    (f"{TT}/tt/load_checkpoints.py", 201, "def convert_hf_to_meta_no_qkv_permute"),
    (f"{TT}/tt/load_checkpoints.py", 800, "def map_hf_to_meta_keys"),
    (f"{TT}/tt/load_checkpoints.py", 830, "def map_meta_to_hf_keys"),
    (f"{TT}/tt/load_checkpoints.py", 626, "def replace_keys"),
    (f"{TT}/tt/load_checkpoints.py", 494, "def fuse_qkv_meta"),
    (f"{TT}/tt/load_checkpoints.py", 474, "def fuse_mlp_meta"),
    (f"{TT}/tt/load_checkpoints.py", 809, '"model."'),
    (f"{TT}/tt/load_checkpoints.py", 814, "self_attn"),
    (f"{TT}/tt/load_checkpoints.py", 819, "q_proj"),
    # row 8 weights
    (f"{GO}/tt/attention/weights.py", 23, "class AttentionWeights"),
    (f"{GO}/tt/attention/weights.py", 31, "wqkv"),
    (f"{GO}/tt/attention/weights.py", 32, "wqkv_bias"),
    (f"{GO}/tt/attention/weights.py", 33, "o_proj"),
    (f"{GO}/tt/attention/weights.py", 34, "o_proj_bias"),
    (f"{GO}/tt/attention/weights.py", 35, "sinks"),
    (f"{GO}/tt/attention/weights.py", 38, "def load_attention_weights"),
    # row 9 head split
    (f"{GO}/tt/attention/operations.py", 29, "def split_qkv_heads_prefill"),
    (f"{GO}/tt/attention/operations.py", 41, "nlp_create_qkv_heads"),
    (f"{GO}/tt/attention/operations.py", 92, "def concat_heads"),
    (f"{GO}/tt/attention/operations.py", 102, "nlp_concat_heads"),
    (f"{GO}/tt/attention/operations.py", 50, "def apply_rope"),
    # row 10 SDPA
    ("ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp", 337, "scaled_dot_product_attention"),
    ("ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp", 98, "nqh >= nkv"),
    ("ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp", 326, "nqh >= nkv"),
    (f"{GO}/tt/attention/prefill.py", 34, "def _run_sdpa"),
    (f"{GO}/tt/attention/prefill.py", 44, "sliding_window_size"),
    (f"{GO}/tt/attention/prefill.py", 45, "attention_sink"),
    (f"{GO}/tt/attention/prefill.py", 40, "is_causal=True"),
    (f"{GO}/tt/attention/prefill.py", 43, "scale=config.scaling"),
    # row 11 program config
    (f"{GO}/tt/attention/config.py", 23, "class AttentionConfig"),
    (f"{GO}/tt/attention/config.py", 34, "sliding_window"),
    (f"{GO}/tt/attention/config.py", 90, "def get_prefill_sdpa_config"),
    (f"{GO}/tt/attention/config.py", 96, "CoreCoord(8, 8)"),
    (f"{GO}/tt/attention/config.py", 102, "def get_compute_kernel_config"),
    (f"{GO}/tt/attention/config.py", 103, "WormholeComputeKernelConfig"),
    # row 12 kv cache
    (f"{GO}/tt/attention/kv_cache.py", 27, "NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32"),
    (f"{GO}/tt/attention/kv_cache.py", 31, "class GptOssKVCache"),
    (f"{GO}/tt/attention/kv_cache.py", 48, "def allocate_kv_cache"),
    (f"{GO}/tt/attention/kv_cache.py", 87, "shard_shape"),
    (f"{GO}/tt/attention/kv_cache.py", 117, "def _write_one"),
    (f"{GO}/tt/attention/kv_cache.py", 138, "def write_kv_chunk"),
    # row 13 kv write op
    (f"{DS}/tt/mla/mla.py", 1188, "def _update_kv_cache"),
    (f"{DS}/tt/mla/indexer.py", 628, "update_padded_kv_cache"),
    # row 14 MLP
    (f"{M3}/tt/dense_mlp.py", 26, "class DenseMLP"),
    (f"{M3}/tt/dense_mlp.py", 38, "scatter_output"),
    (f"{M3}/tt/dense_mlp.py", 58, "def _load"),
    (f"{M3}/tt/dense_mlp.py", 62, "tensor_cache_path"),
    (f"{M3}/tt/dense_mlp.py", 70, "cache_file_name"),
    (f"{M3}/tt/dense_mlp.py", 87, "def __call__"),
    (f"{M3}/tt/dense_mlp.py", 92, "swiglu"),
    (f"{M3}/tt/dense_mlp.py", 112, "allreduce"),
    (f"{M3}/tt/dense_mlp.py", 47, "hf_config.hidden_size"),
    (f"{GO}/tt/attention/operations.py", 25, "bias=weights.wqkv_bias"),
    (f"{GO}/tt/attention/operations.py", 120, "o_proj_bias"),
    (f"{GO}/tt/attention/operations.py", 202, "o_proj_bias"),
    (f"{GO}/tt/attention/dense_sp.py", 144, "attention_sink=attention_sink"),
    (f"{GO}/tt/attention/dense_sp.py", 145, "sliding_window_size"),
    (f"{GO}/tt/attention/config.py", 38, "Sinks are stored pre-divided"),
    ("ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp", 409, "sink_shape[1] == q_shape[1]"),
    # row 15 activation
    (f"{CM}/modules/mlp/mlp_1d.py", 84, "mlp_activation_type"),
    (f"{CM}/modules/mlp/mlp_1d.py", 262, "input_tensor_a_activations"),
    (f"{CM}/modules/mlp/mlp_2d.py", 336, "input_tensor_a_activations"),
    (f"{CM}/modules/mlp/mlp_2d.py", 5, "TG"),
    # row 16/19 model
    (f"{GO}/tt/model.py", 41, "class Model"),
    (f"{GO}/tt/model.py", 31, "def compute_per_device_vocab"),
    (f"{GO}/tt/model.py", 179, "def _forward_layers_and_head"),
    (f"{GO}/tt/model.py", 246, "def prefill_forward"),
    (f"{GO}/tt/model.py", 236, "if skip_lm_head:"),
    (f"{GO}/tt/model.py", 240, "self.norm(hidden_states)"),
    (f"{GO}/tt/model.py", 279, "def prepare_inputs_prefill"),
    (f"{GO}/tt/model.py", 315, "ttnn.embedding"),
    (f"{GO}/tt/model.py", 322, "def process_output_prefill"),
    # row 18 layer
    (f"{GO}/tt/layer.py", 19, "_DELTA_PROBE"),
    (f"{GO}/tt/layer.py", 22, "def _delta_stats"),
    (f"{GO}/tt/layer.py", 46, "class DecoderLayer"),
    (f"{GO}/tt/layer.py", 96, "layer_types"),
    (f"{GO}/tt/layer.py", 138, "32 * 1024"),
    (f"{GO}/tt/layer.py", 140, "ttnn.move"),
    # row 20 ccl
    (f"{GO}/tt/ccl.py", 17, "class CCLManager"),
    (f"{GO}/tt/ccl.py", 40, "_init_subdevice"),
    (f"{GO}/tt/ccl.py", 44, "compute_with_storage_grid_size"),
    (f"{GO}/tt/ccl.py", 61, "ring_attention_ccl_core_grid_offset"),
    (f"{GO}/tt/ccl.py", 63, "_init_semaphores"),
    (f"{GO}/tt/ccl.py", 129, "reset_global_semaphores"),
    (f"{M3}/tt/ccl.py", 9, "class CCLManager"),
    # row 21 mesh config
    (f"{M3}/config.py", 21, "class MeshConfig"),
    (f"{M3}/config.py", 17, "_VALIDATED_MESH_SHAPE"),
    (f"{M3}/config.py", 40, "_validate"),
    (f"{M3}/config.py", 77, "def allreduce"),
    (f"{M3}/config.py", 135, "def allgather"),
    (f"{M3}/config.py", 155, "def reduce_scatter"),
    (f"{GO}/tt/config.py", 19, "class MeshConfig"),
    (f"{GO}/tt/config.py", 15, "_VALIDATED_MESH_SHAPE"),
    (f"{GO}/tt/config.py", 16, "_VALIDATED_TP"),
    (f"{GO}/tt/config.py", 85, "def allreduce"),
    (f"{GO}/tt/config.py", 138, "def allgather"),
    # row 22-23 utils
    (f"{GO}/utils/general_utils.py", 11, "def get_cache_file_name"),
    (f"{GO}/utils/general_utils.py", 27, "def get_default_num_links"),
    (f"{GO}/utils/substate.py", 15, "def substate"),
    # row 25 model_config
    (f"{M3}/tt/model_config.py", 22, "class ModelArgs"),
    (f"{M3}/tt/model_config.py", 126, "def load_state_dict"),
    (f"{M3}/tt/model_config.py", 212, "def weight_cache_path"),
    (f"{M3}/tt/model_config.py", 235, "def get_state_dict_prefix"),
    (f"{TT}/tt/model_config.py", 539, "class ModelArgs"),
    (f"{TT}/tt/model_config.py", 702, "Please set HF_MODEL"),
    (f"{TT}/tt/model_config.py", 623, "use_hf_rope"),
    (f"{TT}/tt/model_config.py", 4037, "def reference_transformer"),
    (f"{TT}/tt/model_config.py", 4393, "def reference_decoder"),
    (f"{TT}/tt/model_config.py", 4410, "def reference_attention"),
    (f"{TT}/tt/model_config.py", 4365, "def reference_mlp"),
    (f"{TT}/tt/model_config.py", 4167, "def reference_rms_norm"),
    (f"{TT}/tt/model_config.py", 4379, "def reference_embedding"),
    (f"{TT}/tt/model_config.py", 4027, "def reference_lm_head"),
    # row 27 runtime
    (f"{GO}/tt/tt_prefill_runtime.py", 96, "class TtPrefillRuntime"),
    (f"{GO}/tt/tt_prefill_runtime.py", 204, "def make_chunk_input"),
    (f"{GO}/tt/tt_prefill_runtime.py", 250, "def compile"),
    (f"{GO}/tt/tt_prefill_runtime.py", 288, "def prefill_chunk"),
    (f"{GO}/tt/tt_prefill_runtime.py", 375, "def build_kv_chunk_table"),
    # row 28-29 adapter / producer
    (f"{GO}/tt/runners/adapters/gpt_oss.py", 41, "class GptOssPrefillAdapter"),
    (f"{GO}/tt/runners/adapters/gpt_oss.py", 45, 'name = "gpt_oss_d_p"'),
    (f"{GO}/tt/runners/adapters/gpt_oss.py", 50, "default_gate_mode"),
    (f"{CP}/adapter.py", 104, "class PrefillModelAdapter"),
    (f"{CP}/adapter.py", 46, "class PrefillRunParams"),
    (f"{CP}/adapter.py", 277, "ADAPTER_PATHS"),
    (f"{CP}/adapter.py", 57, "mesh_shape"),
    (f"{CP}/runners/prefill_producer.py", 503, "def _read_slot_kv_and_check_pcc"),
    (f"{CP}/runners/prefill_producer.py", 534, "_read_slot_kv_and_check_pcc_gpt_oss"),
    (f"{CP}/runners/prefill_producer.py", 685, "_read_slot_kv_and_check_pcc_mla"),
    (f"{CP}/runners/runner_utils.py", 78, "mesh_shape"),
    (f"{CP}/runners/prefill_producer.py", 83, "PREFILL_SP"),
    (f"{CP}/runners/prefill_producer.py", 84, "PREFILL_TP"),
    # row 30 golden
    (f"{M3}/scripts/generate_golden_kv_cache.py", 27, "Output format"),
    # row 31 test factory
    (f"{M3}/tests/test_factory.py", 45, "class TestFactory"),
    (f"{M3}/tests/test_factory.py", 56, "def setup_test"),
    (f"{M3}/tests/test_factory.py", 25, "def minimax_config_dims"),
    (f"{M3}/tests/test_factory.py", 22, "_CONFIG_JSON"),
    (f"{M3}/conftest.py", 13, "--skip-model-load"),
    (f"{M3}/conftest.py", 17, "def state_dict"),
    # row 32 test convention
    (f"{GO}/tests/unit/test_attention_vs_ref.py", 30, "comp_pcc"),
    (f"{GO}/tests/unit/test_attention_vs_ref.py", 83, "def _build_cos_sin"),
    (f"{GO}/tests/unit/test_attention_vs_ref.py", 117, "def _torch_attention"),
    (f"{GO}/tests/unit/test_attention_vs_ref.py", 149, "mesh_device"),
    (f"{GO}/tests/unit/test_attention_vs_ref.py", 258, "0.99"),
    (f"{GO}/tests/unit/test_attention_vs_ref.py", 33, "get_rot_transformation_mat"),
    (f"{GO}/tests/unit/test_attention_vs_ref.py", 34, "convert_hf_qkv_to_meta_format"),
    # row 33-34 utils / fixtures
    (f"{CM}/utility_functions.py", 488, "def comp_pcc"),
    (f"{CM}/utility_functions.py", 476, "def comp_allclose"),
    (f"{CM}/utility_functions.py", 1043, "def is_blackhole"),
    ("conftest.py", 554, "def mesh_device"),
    ("conftest.py", 34, "def reset_seeds"),
    # row 35 dense_sp
    (f"{GO}/tt/attention/dense_sp.py", 30, "def _gather_seq_len"),
    (f"{GO}/tt/attention/dense_sp.py", 41, "def dense_sp_attention"),
    (f"{GO}/tt/attention/dense_sp.py", 135, "is_causal"),
    # row 36 residual
    (f"{M3}/tt/residual.py", 26, "DEFAULT_USE_SHARDED_RESIDUAL"),
    (f"{M3}/tt/residual.py", 36, "def use_sharded_residual"),
    (f"{M3}/tt/residual.py", 44, "def norm_mode"),
    (f"{M3}/tt/residual.py", 53, "def use_distributed_norm"),
    (f"{M3}/tt/residual.py", 59, "def gather_before_norm"),
    # row 37-38 second opinions
    ("models/demos/llama3_70b_galaxy/tt/llama_ccl.py", 25, "class TT_CCL"),
    ("models/demos/llama3_70b_galaxy/tt/llama_attention.py", 11, "class TtLlamaAttention"),
    ("models/demos/llama3_70b_galaxy/tt/distributed_norm.py", 10, "class DistributedNorm"),
    (f"{DS}/tt/tt_ccl.py", 60, "class TT_CCL"),
    # section 5: TTTv2
    (f"{CM}/modules/README.md", 38, "Universal Module Contract"),
    ("models/common/models/llama3_8b/model.py", 890, "only supports 1D mesh topologies"),
    ("models/common/models/llama3_8b/model.py", 884, "is_galaxy_cluster"),
    # section 6: recipe corrections
    (f"{GO}/tt/attention/__init__.py", 28, "class Attention"),
    (f"{TT}/tt/attention.py", 641, "def _mllama_rope_decode"),
    (f"{TT}/tt/attention.py", 702, "def _hf_rope_prefill"),
    (f"{GO}/README.md", 6, "4×8 Blackhole Galaxy"),
    (f"{GO}/README.md", 91, "0.999"),
    (f"{GO}/README.md", 46, "not** imported"),
    (f"{GO}/tests/galaxy_prefill_kv_pcc.py", 44, "ROWS, COLS = 4, 8"),
    (f"{GO}/tests/galaxy_prefill_kv_pcc.py", 154, "mesh_shape"),
    (f"{CP}/docs/PREFILL_MIGRATION_TESTING.md", 62, "CHUNK_SIZE"),
    ("tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto", 6, "8, 4"),
    ("tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto", 8, "count: 2"),
    # ===================== P3 / P4 additions =====================
    # --- gpt-oss model.py: embedding, lm_head, prefill entry points (03_OUTLINE 3.14/3.15/3.17) ---
    (f"{GO}/tt/model.py", 38, "next power of 2"),
    (f"{GO}/tt/model.py", 62, "vocab_size"),
    (f"{GO}/tt/model.py", 77, "model.embed_tokens"),
    (f"{GO}/tt/model.py", 84, "as_tensor"),
    (f"{GO}/tt/model.py", 88, "ROW_MAJOR_LAYOUT"),
    (f"{GO}/tt/model.py", 93, "self.layers"),
    (f"{GO}/tt/model.py", 113, "self.norm"),
    (f"{GO}/tt/model.py", 127, "lm_head"),
    (f"{GO}/tt/model.py", 134, "as_tensor"),
    (f"{GO}/tt/model.py", 141, "column_parallel"),
    (f"{GO}/tt/model.py", 145, "_supports_on_device_sampling"),
    (f"{GO}/tt/model.py", 241, "lm_head_weight"),
    (f"{GO}/tt/model.py", 250, "rot_mats_local"),
    (f"{GO}/tt/model.py", 288, "sequence_parallel"),
    (f"{GO}/tt/model.py", 326, "get_device_tensors"),
    # --- gpt-oss attention: weights / operations / prefill details ---
    (f"{GO}/tt/attention/weights.py", 68, "padded_local_hidden"),
    (f"{GO}/tt/attention/weights.py", 74, "q_proj"),
    (f"{GO}/tt/attention/weights.py", 78, "o_proj"),
    (f"{GO}/tt/attention/weights.py", 83, "qkv_list"),
    (f"{GO}/tt/attention/weights.py", 100, "qkv_cat"),
    (f"{GO}/tt/attention/weights.py", 119, "sinks"),
    (f"{GO}/tt/attention/weights.py", 145, "column_parallel"),
    (f"{GO}/tt/attention/weights.py", 146, "row_parallel"),
    (f"{GO}/tt/attention/weights.py", 149, "as_tensor"),
    (f"{GO}/tt/attention/operations.py", 14, "def apply_qkv_projection"),
    (f"{GO}/tt/attention/operations.py", 45, "transpose_k_heads=False"),
    (f"{GO}/tt/attention/operations.py", 79, "rotary_embedding_indexed"),
    (f"{GO}/tt/attention/operations.py", 105, "def apply_output_projection"),
    (f"{GO}/tt/attention/operations.py", 126, "_FUSED_MM_RS_CONFIGS"),
    (f"{GO}/tt/attention/operations.py", 131, "def is_shape_fused_mm_rs_supported"),
    (f"{GO}/tt/attention/operations.py", 136, "blackhole"),
    (f"{GO}/tt/attention/operations.py", 142, "def apply_output_projection_fused_rs"),
    (f"{GO}/tt/attention/operations.py", 214, "def apply_allgather_and_slice"),
    (f"{GO}/tt/attention/operations.py", 227, "ttnn.slice"),
    (f"{GO}/tt/attention/operations.py", 238, "def apply_allreduce"),
    (f"{GO}/tt/attention/operations.py", 252, "axis=mesh_config.tp_axis"),
    (f"{GO}/tt/attention/operations.py", 262, "ttnn.slice"),
    (f"{GO}/tt/attention/prefill.py", 51, "def attention_forward"),
    (f"{GO}/tt/attention/prefill.py", 106, "32 * 1024"),
    (f"{GO}/tt/attention/prefill.py", 116, "apply_qkv_projection"),
    (f"{GO}/tt/attention/prefill.py", 127, "split_qkv_heads_prefill"),
    (f"{GO}/tt/attention/prefill.py", 143, "apply_rope"),
    (f"{GO}/tt/attention/prefill.py", 151, "apply_rope"),
    (f"{GO}/tt/attention/prefill.py", 168, "write_kv_chunk"),
    (f"{GO}/tt/attention/prefill.py", 184, "mesh_config.sp"),
    (f"{GO}/tt/attention/prefill.py", 191, "use_cache_backed_ring"),
    (f"{GO}/tt/attention/prefill.py", 200, "fp32_dest_acc_en=False"),
    (f"{GO}/tt/attention/prefill.py", 201, "init_device_compute_kernel_config"),
    (f"{GO}/tt/attention/prefill.py", 235, "allgather"),
    (f"{GO}/tt/attention/prefill.py", 243, "reduce_scatter_minimal_async"),
    (f"{GO}/tt/attention/prefill.py", 254, "1.0 / sp"),
    (f"{GO}/tt/attention/prefill.py", 272, "_run_sdpa"),
    (f"{GO}/tt/attention/prefill.py", 280, "concat_heads"),
    (f"{GO}/tt/attention/prefill.py", 302, "apply_output_projection"),
    (f"{GO}/tt/attention/prefill.py", 304, "apply_allreduce"),
    (f"{GO}/tt/attention/config.py", 45, "__post_init__"),
    (f"{GO}/tt/attention/config.py", 52, "def gqa_group_size"),
    (f"{GO}/tt/attention/config.py", 57, "class ProgramConfig"),
    (f"{GO}/tt/attention/kv_cache.py", 77, "assert"),
    (f"{GO}/tt/attention/kv_cache.py", 80, "seq_local"),
    (f"{GO}/tt/attention/kv_cache.py", 104, "ReplicateTensorToMesh"),
    (f"{GO}/tt/attention/kv_cache.py", 125, "update_padded_kv_cache"),
    (f"{GO}/tt/attention/kv_cache.py", 149, "assert tt_k.shape[0] == 1"),
    (f"{GO}/tt/attention/dense_sp.py", 36, "return full_seq"),
    (f"{GO}/tt/attention/dense_sp.py", 77, "bfloat8_b"),
    (f"{GO}/tt/attention/dense_sp.py", 106, "ring_joint_scaled_dot_product_attention"),
    (f"{GO}/tt/attention/dense_sp.py", 116, "persistent_output_buffer_k"),
    (f"{GO}/tt/attention/dense_sp.py", 119, "persistent_output_buffer_v"),
    (f"{GO}/tt/attention/dense_sp.py", 122, "joint_strategy"),
    (f"{GO}/tt/attention/dense_sp.py", 126, "dim=2"),
    (f"{GO}/tt/attention/dense_sp.py", 127, "ring_attention_ccl_semaphore_handles"),
    (f"{GO}/tt/attention/dense_sp.py", 129, "cluster_axis"),
    (f"{GO}/tt/attention/dense_sp.py", 132, "topology"),
    (f"{GO}/tt/attention/dense_sp.py", 133, "ccl_core_grid_offset"),
    (f"{GO}/tt/attention/dense_sp.py", 134, "use_column_major_ccl=True"),
    (f"{GO}/tt/attention/dense_sp.py", 141, "kv_cache_batch_idx"),
    (f"{GO}/tt/attention/dense_sp.py", 142, "kv_actual_isl"),
    (f"{GO}/tt/attention/__init__.py", 38, "def __init__"),
    (f"{GO}/tt/attention/__init__.py", 47, "layer_types"),
    (f"{GO}/tt/attention/__init__.py", 84, "dataclasses.replace"),
    (f"{GO}/tt/attention/__init__.py", 87, "load_attention_weights"),
    (f"{GO}/tt/attention/__init__.py", 103, "def __call__"),
    (f"{GO}/tt/attention/__init__.py", 133, "attention_forward"),
    # --- gpt-oss layer / rope / ccl / config ---
    (f"{GO}/tt/layer.py", 65, "RMSNorm"),
    (f"{GO}/tt/layer.py", 68, "substate"),
    (f"{GO}/tt/layer.py", 72, "post_attention_layernorm"),
    (f"{GO}/tt/layer.py", 98, "AttentionConfig"),
    (f"{GO}/tt/layer.py", 111, "Attention"),
    (f"{GO}/tt/layer.py", 126, "def __call__"),
    (f"{GO}/tt/rope.py", 25, "block_cyclic_reorder"),
    (f"{GO}/tt/rope.py", 75, "def build_yarn_cos_sin"),
    (f"{GO}/tt/rope.py", 103, "def build_transformation_mat"),
    (f"{GO}/tt/rope.py", 115, "def build_indexed_rope"),
    (f"{GO}/tt/rope.py", 146, "TILE_SIZE * sp"),
    (f"{GO}/tt/rope.py", 148, "max_seq_len % chunk_size"),
    (f"{GO}/tt/ccl.py", 46, "CoreRangeSet"),
    (f"{GO}/tt/ccl.py", 65, "3 * 2"),
    (f"{GO}/tt/ccl.py", 66, "rs_ping_pong_semaphores"),
    (f"{GO}/tt/ccl.py", 71, "2 * 2"),
    (f"{GO}/tt/ccl.py", 72, "ag_ping_pong_semaphores"),
    (f"{GO}/tt/ccl.py", 77, "barrier_ns_sems"),
    (f"{GO}/tt/ccl.py", 78, "barrier_semaphore"),
    (f"{GO}/tt/ccl.py", 84, "ring_attention_ccl_semaphore_handles"),
    (f"{GO}/tt/ccl.py", 88, "def get_rs_ping_pong_semaphore"),
    (f"{GO}/tt/ccl.py", 92, "rs_ping_pong_idx"),
    (f"{GO}/tt/ccl.py", 95, "def get_ag_ping_pong_semaphore"),
    (f"{GO}/tt/ccl.py", 99, "ag_ping_pong_idx"),
    (f"{GO}/tt/ccl.py", 102, "def get_barrier_semaphore"),
    (f"{GO}/tt/ccl.py", 105, "barrier_idx"),
    (f"{GO}/tt/ccl.py", 108, "def get_ring_gather_buffer"),
    (f"{GO}/tt/ccl.py", 132, "NOTE"),
    (f"{GO}/tt/config.py", 38, "def _validate"),
    (f"{GO}/tt/config.py", 45, "raise ValueError"),
    (f"{GO}/tt/config.py", 50, "logger.warning"),
    (f"{GO}/tt/config.py", 60, "def shard_mapper"),
    (f"{GO}/tt/config.py", 64, "mesh_dims"),
    (f"{GO}/tt/config.py", 69, "def column_parallel"),
    (f"{GO}/tt/config.py", 73, "def row_parallel"),
    (f"{GO}/tt/config.py", 77, "def sequence_parallel"),
    (f"{GO}/tt/config.py", 81, "def shard_size"),
    (f"{GO}/tt/config.py", 102, "reduce_scatter_minimal_async"),
    (f"{GO}/tt/config.py", 118, "all_gather_async"),
    (f"{GO}/tt/config.py", 158, "def __repr__"),
    (f"{GO}/tt/rms_norm.py", 22, "use_gemma_norm"),
    (f"{GO}/tt/rms_norm.py", 37, "dtype=ttnn.bfloat16"),
    (f"{GO}/tt/rms_norm.py", 70, "all_gather"),
    (f"{GO}/tt/rms_norm.py", 77, "Topology.Ring"),
    (f"{GO}/tt/rms_norm.py", 82, "rms_norm_post_all_gather"),
    (f"{GO}/utils/general_utils.py", 15, "def cache_file_exists"),
    (f"{GO}/utils/general_utils.py", 33, "shape[0] == 1"),
    (f"{GO}/utils/general_utils.py", 35, "is_blackhole"),
    (f"{GO}/utils/substate.py", 37, "def has_substate"),
    (f"{GO}/utils/substate.py", 53, "def indexed_substates"),
    (f"{GO}/tt/runners/adapters/gpt_oss.py", 63, "def load_hf_config"),
    (f"{GO}/tt/runners/adapters/gpt_oss.py", 75, "def weight_cache_path"),
    (f"{GO}/tt/tt_prefill_runtime.py", 59, "class TtPrefillRuntimeConfig"),
    (f"{GO}/tt/tt_prefill_runtime.py", 88, "def sp_factor"),
    (f"{GO}/tt/tt_prefill_runtime.py", 92, "def tp_factor"),
    (f"{GO}/tests/galaxy_prefill_kv_pcc.py", 121, "PREFILL_TOPOLOGY"),
    (f"{GO}/tests/galaxy_prefill_kv_pcc.py", 122, "set_fabric_config"),
    (f"{GO}/tests/galaxy_prefill_kv_pcc.py", 132, "num_hidden_layers"),
    (f"{GO}/tests/galaxy_prefill_kv_pcc.py", 161, "Topology.Linear"),
    # --- minimax_m3: MeshConfig internals, dense MLP, residual ---
    (f"{M3}/config.py", 24, "def __init__"),
    (f"{M3}/config.py", 46, "_VALIDATED_MESH_SHAPE"),
    (f"{M3}/config.py", 52, "def shard_mapper"),
    (f"{M3}/config.py", 61, "def column_parallel"),
    (f"{M3}/config.py", 65, "def row_parallel"),
    (f"{M3}/config.py", 69, "def sequence_parallel"),
    (f"{M3}/config.py", 73, "def shard_size"),
    (f"{M3}/config.py", 94, "reduce_scatter_minimal_async"),
    (f"{M3}/config.py", 104, "Free the full-size input"),
    (f"{M3}/config.py", 115, "all_gather_async"),
    (f"{M3}/config.py", 148, "Topology.Linear"),
    (f"{M3}/config.py", 175, "sp = self.mesh_shape"),
    (f"{M3}/tt/dense_mlp.py", 29, "def __init__"),
    (f"{M3}/tt/dense_mlp.py", 77, "transpose(-1, -2)"),
    (f"{M3}/tt/dense_mlp.py", 99, "self.mesh_config.tp > 1"),
    (f"{M3}/tt/dense_mlp.py", 105, "reduce_scatter"),
    (f"{M3}/tt/residual.py", 9, "Full width is reconstituted"),
    (f"{M3}/tt/residual.py", 26, "DEFAULT_USE_SHARDED_RESIDUAL = True"),
    (f"{M3}/tt/residual.py", 32, 'DEFAULT_NORM_MODE = "gather_first"'),
    (f"{M3}/tt/model_config.py", 19, "convert_hf_qkv_to_meta_format_partial"),
    # --- ttnn ops / bindings measured or read in P3 ---
    (
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads_nanobind.cpp",
        24,
        "input_kv",
    ),
    (
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads_nanobind.cpp",
        28,
        'nb::arg("input_kv")',
    ),
    ("ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp", 419, "COL_MAJOR"),
    (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp",
        421,
        "ccl_core_grid_offset.x >=",
    ),
    (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp",
        425,
        "ccl_core_grid_offset.y >=",
    ),
    ("ttnn/ttnn/types.py", 61, "BlackholeComputeKernelConfig = WormholeComputeKernelConfig"),
    (f"{CP}/runners/migration.py", 337, "def get_num_dram_banks"),
    (f"{CP}/runners/migration.py", 338, "dram_grid_size"),
    (f"{CP}/adapter.py", 143, "def load_hf_config"),
    (f"{CM}/modules/mlp/mlp_2d.py", 461, "cluster_axis=0"),
    (f"{TT}/tt/common.py", 529, "torch.stack"),
    (f"{TT}/tt/load_checkpoints.py", 452, "Convert HuggingFace QKV weights to Meta format"),
    ("python_env/lib/python3.12/site-packages/transformers/models/llama/configuration_llama.py", 84, "head_dim"),
    (
        "python_env/lib/python3.12/site-packages/transformers/models/llama/configuration_llama.py",
        87,
        "if self.head_dim is None",
    ),
    (
        "python_env/lib/python3.12/site-packages/transformers/models/llama/configuration_llama.py",
        88,
        "num_attention_heads",
    ),
    # --- this package's own P1 deliverables (cited by 03_OUTLINE) ---
    # was :49 before black collapsed UPSTREAM_CONFIG_JSON -- see DEC-027
    ("models/demos/llama31_8b_d_p/tests/test_factory.py", 47, "def llama_config_dims"),
    ("models/demos/llama31_8b_d_p/tests/test_factory.py", 100, "def rope_scaling"),
    ("models/demos/llama31_8b_d_p/configs/Llama-3.1-8B-Instruct/config.json", 33, "tie_word_embeddings"),
    # --- P5.1-P5.3 (this phase): the templates copied/mirrored, and the traps asserted ---
    (f"{GO}/tt/config.py", 56, "def sp"),  # 04_CCL_PLAN.md section 3 says :55 (the @property line) -- corrected in P5
    (f"{GO}/tt/config.py", 55, "@property"),
    (f"{GO}/tt/config.py", 22, "def __init__"),
    (f"{GO}/tt/config.py", 33, "ep_axis"),
    (f"{GO}/tt/ccl.py", 50, "ttnn.SubDevice("),
    (f"{GO}/tt/ccl.py", 24, "_ping_pong_buffer_cache"),
    (f"{GO}/tt/ccl.py", 55, "ccl_sub_device_id"),
    (f"{GO}/tt/rms_norm.py", 18, "def __init__"),
    (f"{GO}/tt/rms_norm.py", 22, "use_gemma_norm"),
    (f"{GO}/tt/rms_norm.py", 34, "ttnn.as_tensor"),
    (f"{GO}/tt/rms_norm.py", 70, "ttnn.all_gather"),
    (f"{GO}/tt/model_config.py", 76, 'getattr(self.hf_config, "rope_theta"'),
    (f"{GO}/tt/tt_prefill_runtime.py", 185, 'getattr(self.hf_config, "rope_theta"'),
    (f"{TT}/tt/common.py", 501, "freqs = 1.0 / (theta"),
    (f"{TT}/tt/common.py", 504, "apply_scaling(freqs"),
    (f"{TT}/tt/common.py", 564, "dhead = 32"),
    (f"{DS}/tt/mla/utils.py", 65, "def block_cyclic_reorder"),
    (f"{DS}/tt/mla/utils.py", 89, "local row lr on chip c holds global position"),
    # --- this package's own P5 deliverables ---
    (f"{LL}/tt/config.py", 34, "class MeshConfig"),
    (f"{LL}/tt/config.py", 177, "def reduce_scatter"),
    (f"{LL}/tt/ccl.py", 30, "class CCLManager"),
    (f"{LL}/tt/config.py", 48, "self.sp_axis = 0 if tp_axis == 1 else 1"),
    ("models/tt_transformers/tests/test_rms_norm.py", 80, "torch.rand(1, 1, 32"),
    ("models/tt_transformers/tests/test_rms_norm.py", 104, "pcc=0.9999"),
    ("conftest.py", 948, "def expect_error"),
    (f"{TT}/tt/common.py", 536, "seq_len * 2"),
    (f"{TT}/tt/common.py", 538, "gather_cos_sin(torch.arange(start_pos"),
    (".pre-commit-config.yaml", 51, "prefer-expect-error"),
    (".pre-commit-config.yaml", 55, "pytest\\.raises"),
    (f"{LL}/tt/model_config.py", 57, "class LlamaHFConfig"),
    (f"{LL}/tt/model_config.py", 90, "def llama_hf_config"),
    (f"{LL}/tt/rms_norm.py", 66, "class RMSNorm"),
    (f"{LL}/tt/rope.py", 71, "def build_transformation_mat"),
    (f"{LL}/tt/rope.py", 88, "def build_prefill_rope"),
    (f"{LL}/tt/rope.py", 108, "assert start_pos <= seq_len"),
    (f"{LL}/tt/rope.py", 59, "def _assert_llama3_scaling"),
    (f"{LL}/tt/rope.py", 124, "def build_indexed_rope"),
    (f"{LL}/tt/rope.py", 175, "def build_meta_cos_sin"),
    (f"{LL}/utils/general_utils.py", 16, "def get_cache_file_name"),
    (f"{LL}/utils/substate.py", 20, "def substate"),
    # --- P5.4-P5.6 (tt/mlp.py, tt/attention/*, tests) --------------------------------------
    # MLP: the fused SiLU-mul and its in-tree precedent
    (f"{CM}/modules/mlp/mlp_1d.py", 259, "ttnn.mul"),
    (f"{CM}/modules/mlp/mlp_1d.py", 84, "SILU"),
    (f"{M3}/tt/dense_mlp.py", 48, "use_sharded_residual"),
    (f"{M3}/tt/dense_mlp.py", 62, "not tensor_cache_path"),
    (f"{M3}/tt/dense_mlp.py", 89, "ttnn.linear"),
    (f"{LL}/tt/mlp.py", 59, "def default_compute_kernel_config"),
    (f"{LL}/tt/mlp.py", 82, "class MLP"),
    (f"{LL}/tt/mlp.py", 188, "def __call__"),
    (f"{LL}/tt/config.py", 134, "deallocate"),
    # attention/config.py + the two corrections it encodes (DEC-012, DEC-013)
    (f"{GO}/tt/attention/config.py", 23, "class AttentionConfig"),
    (f"{GO}/tt/attention/config.py", 34, "sliding_window"),
    (f"{GO}/tt/attention/config.py", 57, "class ProgramConfig"),
    (f"{GO}/tt/attention/config.py", 71, "fp32_dest_acc_en"),
    (f"{GO}/tt/attention/config.py", 90, "get_prefill_sdpa_config"),
    (f"{GO}/tt/attention/config.py", 102, "get_compute_kernel_config"),
    (f"{LL}/tt/attention/config.py", 45, "class AttentionConfig"),
    (f"{LL}/tt/attention/config.py", 88, "class ProgramConfig"),
    (f"{LL}/tt/attention/config.py", 99, "sdpa_core_grid"),
    (f"{LL}/tt/attention/config.py", 106, "fp32_dest_acc_en"),
    (f"{LL}/tt/attention/config.py", 126, "def assert_sdpa_grid_fits"),
    (f"{LL}/tt/attention/config.py", 165, "def get_compute_kernel_config"),
    (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp",
        421,
        "ccl_core_grid_offset",
    ),
    ("ttnn/ttnn/__init__.py", 305, "WormholeComputeKernelConfig"),
    ("ttnn/ttnn/types.py", 61, "BlackholeComputeKernelConfig"),
    # attention/weights.py: three separate weights (DEC-011) + the Meta swizzle (DEC-033)
    (f"{GO}/tt/attention/weights.py", 23, "class AttentionWeights"),
    (f"{GO}/tt/attention/weights.py", 31, "wqkv"),
    (f"{GO}/tt/attention/weights.py", 35, "sinks"),
    (f"{GO}/tt/attention/weights.py", 38, "def load_attention_weights"),
    (f"{GO}/tt/attention/weights.py", 87, "torch.chunk"),
    (f"{LL}/tt/attention/weights.py", 51, "class AttentionWeights"),
    (f"{LL}/tt/attention/weights.py", 60, "def _meta_swizzle"),
    (f"{LL}/tt/attention/weights.py", 71, "def load_attention_weights"),
    # attention/operations.py
    (f"{GO}/tt/attention/operations.py", 14, "def apply_qkv_projection"),
    (f"{GO}/tt/attention/operations.py", 25, "wqkv_bias"),
    (f"{GO}/tt/attention/operations.py", 29, "def split_qkv_heads_prefill"),
    (f"{GO}/tt/attention/operations.py", 41, "nlp_create_qkv_heads"),
    (f"{GO}/tt/attention/operations.py", 117, "typecast"),
    (f"{GO}/tt/attention/operations.py", 120, "o_proj_bias"),
    (f"{GO}/tt/attention/operations.py", 131, "def is_shape_fused_mm_rs_supported"),
    (f"{GO}/tt/attention/operations.py", 136, "blackhole"),
    (f"{GO}/tt/attention/operations.py", 214, "def apply_allgather_and_slice"),
    (f"{GO}/tt/attention/operations.py", 238, "def apply_allreduce"),
    (f"{LL}/tt/attention/operations.py", 40, "def apply_qkv_projection"),
    (f"{LL}/tt/attention/operations.py", 61, "def split_qkv_heads_prefill"),
    (f"{LL}/tt/attention/operations.py", 84, "def apply_rope"),
    (f"{LL}/tt/attention/operations.py", 120, "def concat_heads"),
    (f"{LL}/tt/attention/operations.py", 125, "def apply_output_projection"),
    (f"{LL}/tt/attention/operations.py", 138, "def apply_allreduce"),
    (f"{LL}/tt/attention/operations.py", 149, "def apply_reduce_scatter"),
    (
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/" "nlp_create_qkv_heads_nanobind.cpp",
        24,
        "input_kv",
    ),
    (
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/" "nlp_create_qkv_heads_nanobind.cpp",
        28,
        "input_kv",
    ),
    # attention/prefill.py + the GQA guard that makes the on-chip KV repeat unnecessary
    (f"{GO}/tt/attention/prefill.py", 34, "def _run_sdpa"),
    (f"{GO}/tt/attention/prefill.py", 44, "sliding_window_size"),
    (f"{GO}/tt/attention/prefill.py", 45, "attention_sink"),
    (f"{GO}/tt/attention/prefill.py", 51, "def attention_forward"),
    (f"{GO}/tt/attention/prefill.py", 200, "fp32_dest_acc_en=False"),
    (f"{LL}/tt/attention/prefill.py", 66, "def _run_sdpa"),
    (f"{LL}/tt/attention/prefill.py", 81, "def attention_forward"),
    (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp",
        98,
        "nqh % nkv == 0",
    ),
    # attention/kv_cache.py: head_dim 64 -> 128 is the only delta (Appendix F.6)
    (f"{GO}/tt/attention/kv_cache.py", 27, "NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32"),
    (f"{GO}/tt/attention/kv_cache.py", 48, "def allocate_kv_cache"),
    (f"{GO}/tt/attention/kv_cache.py", 87, "shard_shape"),
    (f"{GO}/tt/attention/kv_cache.py", 125, "update_padded_kv_cache"),
    (f"{GO}/tt/attention/kv_cache.py", 138, "def write_kv_chunk"),
    (f"{LL}/tt/attention/kv_cache.py", 53, "NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32"),
    (f"{LL}/tt/attention/kv_cache.py", 56, "LLAMA_HEAD_DIM = 128"),
    (f"{LL}/tt/attention/kv_cache.py", 60, "class LlamaKVCache"),
    (f"{LL}/tt/attention/kv_cache.py", 79, "def allocate_kv_cache"),
    (f"{LL}/tt/attention/kv_cache.py", 149, "def _write_one"),
    (f"{LL}/tt/attention/kv_cache.py", 171, "def write_kv_chunk"),
    (f"{CP}/runners/migration.py", 338, "dram_grid_size"),
    # attention/dense_sp.py stub + the constraints P8 inherits
    (f"{GO}/tt/attention/dense_sp.py", 30, "_gather_seq_len"),
    (f"{GO}/tt/attention/dense_sp.py", 36, "return full_seq"),
    (f"{GO}/tt/attention/dense_sp.py", 41, "def dense_sp_attention"),
    (f"{GO}/tt/attention/dense_sp.py", 134, "use_column_major_ccl"),
    (f"{LL}/tt/attention/dense_sp.py", 43, "def dense_sp_attention"),
    # attention/__init__.py
    (f"{GO}/tt/attention/__init__.py", 28, "class Attention"),
    (f"{GO}/tt/attention/__init__.py", 47, "layer_types"),
    (f"{GO}/tt/attention/__init__.py", 103, "def __call__"),
    (f"{LL}/tt/attention/__init__.py", 53, "def attention_config_from_hf"),
    (f"{LL}/tt/attention/__init__.py", 71, "class Attention"),
    (f"{LL}/tt/attention/__init__.py", 142, "def __call__"),
    # the Meta swizzle helpers and the oracle input distributions the gates quote
    (f"{TT}/tt/load_checkpoints.py", 451, "def convert_hf_qkv_to_meta_format"),
    (f"{TT}/tests/test_mlp.py", 96, "torch.randn"),
    (f"{TT}/tests/test_mlp.py", 109, "bfloat8_b"),
    (f"{TT}/tests/test_attention_prefill.py", 162, "torch.rand"),
    (f"{TT}/tests/test_rms_norm.py", 77, "reference_rms_norm"),
    (f"{GO}/tests/unit/test_attention_vs_ref.py", 83, "_build_cos_sin"),
    (f"{GO}/tests/unit/test_attention_vs_ref.py", 197, "convert_hf_qkv_to_meta_format"),
    (f"{GO}/tests/unit/test_kv_cache_vs_ref.py", 114, "blockcyclic_reorder_positions_inverse"),
    # the P5.4-P5.6 test files themselves
    (f"{LL}/tests/unit/test_rope_vs_ref.py", 60, "def _meta_to_hf_layout"),
    (f"{LL}/tests/unit/test_rope_vs_ref.py", 65, "def _hf_to_meta_layout"),
    (f"{LL}/tests/unit/test_rope_vs_ref.py", 140, "def test_wrong_convention_fails"),
    (f"{LL}/tests/unit/test_mlp_vs_ref.py", 67, "def quantize_like_device"),
    (f"{LL}/tests/unit/test_mlp_vs_ref.py", 78, "def err_ratio"),
    (f"{LL}/tests/unit/test_mlp_vs_ref.py", 145, "def test_mlp_vs_ref"),
    (f"{LL}/tests/unit/test_mlp_vs_ref.py", 186, "def test_fp32_dest_acc_is_load_bearing"),
    (f"{LL}/tests/unit/test_mlp_vs_ref.py", 238, "def test_silu_is_on_the_gate_branch"),
    (f"{LL}/tests/unit/test_attention_vs_ref.py", 184, "def test_attention_vs_ref"),
    (f"{LL}/tests/unit/test_attention_vs_ref.py", 236, "def test_unswizzled_qk_weights_fail"),
    (f"{LL}/tests/unit/test_attention_vs_ref.py", 404, "def test_sdpa_kernel_error_is_the_dominant_term"),
    (f"{LL}/tests/unit/test_attention_vs_ref.py", 469, "def test_qkv_and_rope_stage_is_at_the_floor"),
    (f"{LL}/tests/unit/test_kv_cache_vs_ref.py", 96, "def test_kv_cache_write_read_vs_ref"),
    (f"{LL}/tests/unit/test_kv_cache_vs_ref.py", 188, "def test_kv_cache_readback_is_positionally_exact"),
    (f"{LL}/tests/unit/test_kv_cache_vs_ref.py", 254, "def test_writes_touch_only_their_own_region"),
    (f"{LL}/tests/unit/test_kv_cache_vs_ref.py", 341, "def test_dram_shard_geometry_at_head_dim_128"),
    # ---------------------------------------------------------------------------------
    # P6 — layer / model assembly, weight loading. Added by P6 per Appendix F.7 ("extend the
    # verifier every phase"). These are the refs P7/P10 will cite, and they are FULLY QUALIFIED
    # because `tt/layer.py`, `tt/model.py` and `tt/embedding.py` now shadow gpt-oss files that
    # earlier logs cite by bare filename (`DEC-035`).
    # ---------------------------------------------------------------------------------
    # tt/layer.py — DecoderLayer + the delta probe
    (f"{LL}/tt/layer.py", 61, "LLAMA31_8B_DELTA_PROBE"),
    (f"{LL}/tt/layer.py", 65, "_MOVE_GUARD_SEQ_LEN"),
    (f"{LL}/tt/layer.py", 68, "def _delta_stats"),
    (f"{LL}/tt/layer.py", 96, "class DecoderLayer"),
    (f"{LL}/tt/layer.py", 99, "def __init__"),
    (f"{LL}/tt/layer.py", 211, "def __call__"),
    (f"{GO}/tt/layer.py", 46, "class DecoderLayer"),
    (f"{GO}/tt/layer.py", 19, "GPT_OSS_DELTA_PROBE"),
    (f"{GO}/tt/layer.py", 22, "def _delta_stats"),
    (f"{GO}/tt/layer.py", 35, "except Exception"),
    (f"{GO}/tt/layer.py", 138, "32 * 1024"),
    # tt/model.py — Model, the three prefill entry points, the G-WEIGHTS accessors
    (f"{LL}/tt/model.py", 55, "class Model"),
    (f"{LL}/tt/model.py", 190, "def consumed_state_dict_keys"),
    (f"{LL}/tt/model.py", 217, "def named_device_tensors"),
    (f"{LL}/tt/model.py", 244, "def prepare_inputs_prefill"),
    (f"{LL}/tt/model.py", 303, "def _forward_layers_and_head"),
    (f"{LL}/tt/model.py", 381, "def prefill_forward"),
    (f"{LL}/tt/model.py", 437, "def process_output_prefill"),
    (f"{GO}/tt/model.py", 41, "class Model"),
    (f"{GO}/tt/model.py", 31, "def compute_per_device_vocab"),
    (f"{GO}/tt/model.py", 134, "as_tensor"),
    (f"{GO}/tt/model.py", 211, "on_layer_complete(i)"),
    (f"{GO}/tt/model.py", 236, "if skip_lm_head:"),
    (f"{GO}/tt/model.py", 240, "self.norm(hidden_states)"),
    (f"{GO}/tt/model.py", 279, "def prepare_inputs_prefill"),
    (f"{GO}/tt/model.py", 320, "return tokens_embd, None, None"),
    (f"{M3}/tt/model.py", 504, "def prepare_inputs_prefill"),
    (f"{M3}/tt/model.py", 599, "rot_mats_global = ["),
    (f"{M3}/tt/model.py", 610, "raise NotImplementedError"),
    # tt/embedding.py, tt/lm_head.py
    (f"{LL}/tt/embedding.py", 35, "class Embedding"),
    (f"{LL}/tt/embedding.py", 87, "def __call__"),
    (f"{LL}/tt/lm_head.py", 42, "class LMHead"),
    (f"{LL}/tt/lm_head.py", 127, "def __call__"),
    # tt/model_config.py part 2 — ModelArgs
    (f"{LL}/tt/model_config.py", 245, "def state_dict_uses_meta_keys"),
    (f"{LL}/tt/model_config.py", 257, "class ModelArgs"),
    (f"{LL}/tt/model_config.py", 298, "def load_state_dict"),
    (f"{LL}/tt/model_config.py", 335, "def expected_state_dict_keys"),
    (f"{LL}/tt/model_config.py", 348, "def audit_state_dict_keys"),
    (f"{LL}/tt/model_config.py", 364, "def weight_cache_path"),
    (f"{LL}/tt/model_config.py", 400, "def get_state_dict_prefix"),
    (f"{M3}/tt/model_config.py", 214, "TT_CACHE_PATH"),
    (f"{GO}/tt/runners/adapters/gpt_oss.py", 75, "def weight_cache_path"),
    # tests/test_factory.py — the promoted helpers (DEC-046)
    (f"{LL}/tests/test_factory.py", 212, "def quantize_like_device"),
    (f"{LL}/tests/test_factory.py", 230, "def err_ratio"),
    # tests/unit/test_decoder_layer_vs_ref.py — G-LAYER
    (f"{LL}/tests/unit/test_decoder_layer_vs_ref.py", 118, "def _torch_rms_norm"),
    (f"{LL}/tests/unit/test_decoder_layer_vs_ref.py", 140, "def _torch_layer"),
    (f"{LL}/tests/unit/test_decoder_layer_vs_ref.py", 167, "def _quantise_layer_state"),
    (f"{LL}/tests/unit/test_decoder_layer_vs_ref.py", 231, "def test_decoder_layer_vs_ref"),
    (f"{LL}/tests/unit/test_decoder_layer_vs_ref.py", 280, "def test_swapped_norms_fail"),
    (
        f"{LL}/tests/unit/test_decoder_layer_vs_ref.py",
        316,
        "def test_residual_masking_tracks_the_delta_to_stream_ratio",
    ),
    (
        f"{LL}/tests/unit/test_decoder_layer_vs_ref.py",
        396,
        "def test_real_weights_show_the_residual_dominating",
    ),
    (f"{LL}/tests/unit/test_decoder_layer_vs_ref.py", 452, "def test_promoted_helpers_match_the_p5_copies"),
    # tests/unit/test_weight_loading.py — G-WEIGHTS
    (f"{LL}/tests/unit/test_weight_loading.py", 90, "def test_no_missing_and_no_unused_keys"),
    (f"{LL}/tests/unit/test_weight_loading.py", 148, "def test_cache_only_rebuild_is_bit_identical"),
    (f"{LL}/tests/unit/test_weight_loading.py", 197, "def test_device_weights_match_the_checkpoint"),
    (f"{LL}/tests/unit/test_weight_loading.py", 268, "def test_meta_renaming_is_caught_by_the_audit"),
    (f"{LL}/tests/unit/test_weight_loading.py", 317, "def test_weight_cache_path_carries_the_mesh_shape"),
    # tests/unit/test_model_vs_ref.py — G-MODEL
    (f"{LL}/tests/unit/test_model_vs_ref.py", 88, "MAX_LAYER_ERROR_STEP"),
    (f"{LL}/tests/unit/test_model_vs_ref.py", 98, "def _hf_model"),
    (f"{LL}/tests/unit/test_model_vs_ref.py", 123, "def _hf_run"),
    (f"{LL}/tests/unit/test_model_vs_ref.py", 149, "def _torch_stack"),
    (f"{LL}/tests/unit/test_model_vs_ref.py", 262, "def test_model_vs_hf_reduced_depth"),
    (f"{LL}/tests/unit/test_model_vs_ref.py", 309, "def test_full_stack_per_layer_pcc_curve"),
    (f"{LL}/tests/unit/test_model_vs_ref.py", 398, "def test_rotated_layer_weights_fail"),
    (
        f"{LL}/tests/unit/test_model_vs_ref.py",
        438,
        "def test_get_last_token_slice_matches_the_full_sequence",
    ),
    (f"{LL}/tests/unit/test_model_vs_ref.py", 471, "def test_hf_reference_is_causal"),
    (f"{LL}/tests/unit/test_model_vs_ref.py", 501, "def test_in_test_torch_reference_agrees_with_hf"),
    # transformers 5.12.1 — the reference's own shape (DEC-051)
    (
        "python_env/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py",
        332,
        "return hidden_states",
    ),
    (
        "python_env/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py",
        375,
        "def forward",
    ),
    (
        "python_env/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py",
        421,
        "self.norm(hidden_states)",
    ),
    # --- P7 additions (chunked prefill + golden KV). Each of these is a claim P7's docstrings and
    # gate blocks make, and each would rot silently: a moved TT_FATAL, a renamed template hook, or
    # a template default that flips.
    (
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/"
        "update_padded_kv_cache_device_operation.cpp",
        230,
        "cache and input num-heads dim must match",
    ),
    (f"{LL}/tt/attention/kv_cache.py", 130, "torch.zeros(num_users * num_layers, 1, seq_local, head_dim)"),
    (f"{LL}/tt/attention/prefill.py", 218, "elif cached_len > 0:"),
    (f"{LL}/tt/attention/prefill.py", 195, "if config.sequence_parallel and mesh_config.sp > 1:"),
    (f"{LL}/tt/attention/dense_sp.py", 43, "def dense_sp_attention("),
    (f"{GO}/tt/tt_prefill_runtime.py", 46, "def resolve_chunk_sizes("),
    (f"{GO}/tt/tt_prefill_runtime.py", 185, 'getattr(self.hf_config, "rope_theta", 150000.0)'),
    (f"{GO}/tt/tt_prefill_runtime.py", 555, "half * (m % 2) + (m // 2)"),
    (f"{GO}/tt/runners/adapters/gpt_oss.py", 132, "default_chunk_size=params.chunk_size"),
    (f"{M3}/scripts/generate_golden_kv_cache.py", 180, "ids = tokenizer.apply_chat_template("),
]


# Phase 2: scan the bring-up logs for every `path:line` reference and check it resolves.
# Explicit CITES above assert *what* is on the line; this pass is the safety net that catches a
# reference to a file that moved or a line past EOF, for the many refs that carry no needle.
DOCS = [
    "models/demos/llama31_8b_d_p/bringup_log/03_OUTLINE.md",
    "models/demos/llama31_8b_d_p/bringup_log/04_CCL_PLAN.md",
    # P5 added the decision log and the gate ledger: both now carry load-bearing `path:line` refs.
    "models/demos/llama31_8b_d_p/bringup_log/05_DECISIONS.md",
    "models/demos/llama31_8b_d_p/bringup_log/06_GATES.md",
    "models/demos/llama31_8b_d_p/bringup_log/07_RISKS.md",
]
# P6 addition (Appendix F.7 / `DEC-035`): the package's **own Python docstrings** carry as many
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
    for path in __import__("pathlib").Path(os.path.join(ROOT, "models/demos/llama31_8b_d_p")).glob(pattern)
)
# package-relative shorthands used in the logs
DOC_PREFIXES = {
    "BRINGUP_RECIPE.md": "models/demos/llama31_8b_d_p/BRINGUP_RECIPE.md",
    "00_MODEL_CARD.md": "models/demos/llama31_8b_d_p/bringup_log/00_MODEL_CARD.md",
    "01_REFERENCE.md": "models/demos/llama31_8b_d_p/bringup_log/01_REFERENCE.md",
    "02_SURVEY.md": "models/demos/llama31_8b_d_p/bringup_log/02_SURVEY.md",
    "03_OUTLINE.md": "models/demos/llama31_8b_d_p/bringup_log/03_OUTLINE.md",
    "04_CCL_PLAN.md": "models/demos/llama31_8b_d_p/bringup_log/04_CCL_PLAN.md",
    "05_DECISIONS.md": "models/demos/llama31_8b_d_p/bringup_log/05_DECISIONS.md",
    "06_GATES.md": "models/demos/llama31_8b_d_p/bringup_log/06_GATES.md",
    "07_RISKS.md": "models/demos/llama31_8b_d_p/bringup_log/07_RISKS.md",
}
_REF = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|cpp|hpp|md|json|textproto|yaml)):(\d+)(?:-(\d+))?`")


# P5: the logs also use abbreviated forms — a bare basename (`common.py:564`, continuing an earlier
# full citation) or a partial path (`gpt_oss_d_p/tt/config.py:55`). Resolving them instead of
# reporting them "unresolved" is what makes pass 2 cover the decision log and the gate ledger, where
# the shorthand is the norm. Ambiguous basenames are REPORTED, not silently dropped.
# P6 addition: the package's own root, so a package-relative ref (`tt/config.py:134`,
# `tests/unit/test_reference_model.py:136` — the shorthand every file in this package uses for its
# own siblings) resolves LITERALLY instead of falling through to the ambiguous-basename path. Before
# this, `tt/config.py:134` was matched against `gpt_oss_d_p/tt/attention/config.py` (108 lines) and
# reported out of range — a false positive from citation shadowing, and it must be listed FIRST so
# a package-local file wins over a same-named file elsewhere in the tree (`DEC-035`).
_PARTIAL_PREFIXES = (
    "models/demos/llama31_8b_d_p/",
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
    return 0 if bad == 0 and missing == 0 and doc_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
