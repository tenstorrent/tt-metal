# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TP=32 model configuration for MiniMax-M3 on the Blackhole Galaxy mesh.

This module is the single source of truth for the tensor-parallel (TP=32)
bring-up of the MiniMax-M3 *text* backbone on a Blackhole Galaxy (32 chips,
mesh_shape ``(1, 32)``, FABRIC_1D, Linear topology).

It holds:
  * the static model dimensions parsed from ``config.json`` ``text_config``,
  * the dtype discipline (bf16 activations, bf16/bf8 weights, fp32 dest acc),
  * the mesh open/close helpers,
  * the TP=32 sharding-recipe mesh-mapper helpers (column-parallel,
    row-parallel, replicate, vocab-shard) used by every block + weight loader.

There are NO torch ops here — this is pure configuration + ttnn mesh mappers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import ttnn

# ---------------------------------------------------------------------------
# Device / mesh constants (bh_galaxy)
# ---------------------------------------------------------------------------
MESH_ROWS = 1
MESH_COLS = 32
TP = MESH_COLS  # tensor-parallel degree == number of columns
MESH_SHAPE = (MESH_ROWS, MESH_COLS)
FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_1D
CCL_TOPOLOGY = ttnn.Topology.Linear

# ---------------------------------------------------------------------------
# Dtype discipline (per the framework's bring-up defaults)
#   - activations: bf16
#   - weights: bf16 default, bf8_b allowed for large matmul weights
#   - dest accumulation: fp32 (math fidelity HiFi for norms / fp32-sensitive ops)
# bf16 is the bring-up default; escalate to fp32 surgically only on PCC miss.
# ---------------------------------------------------------------------------
ACT_DTYPE = ttnn.bfloat16
WEIGHT_DTYPE = ttnn.bfloat16
WEIGHT_DTYPE_LARGE = ttnn.bfloat8_b  # for big projection / expert matmuls
NORM_WEIGHT_DTYPE = ttnn.bfloat16


def default_compute_kernel_config() -> ttnn.WormholeComputeKernelConfig:
    """fp32 dest-acc compute config (HiFi4) used as the bring-up default.

    Blackhole reuses the Wormhole compute-kernel-config struct. fp32 dest
    accumulation is the dtype-discipline default; surgically downgrade per
    block only after a measured PCC win.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


# ---------------------------------------------------------------------------
# MiniMax-M3 text_config dimensions (config.json, verified this session)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TextConfig:
    hidden_size: int = 6144
    num_hidden_layers: int = 60
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 128
    vocab_size: int = 200064

    # partial RoPE
    rotary_dim: int = 64  # head_dim * partial_rotary_factor (128 * 0.5)
    partial_rotary_factor: float = 0.5
    rope_theta: float = 5e6

    # norms
    rms_norm_eps: float = 1e-6
    use_gemma_norm: bool = True  # gamma is applied as (1.0 + weight)
    qk_norm_type: str = "per_head"  # RMSNorm over head_dim (128) per head

    # MoE / MLP
    num_dense_layers: int = 3  # layers 0..2 dense, 3..59 MoE
    dense_intermediate_size: int = 12288
    moe_intermediate_size: int = 3072
    shared_intermediate_size: int = 3072
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    routed_scaling_factor: float = 2.0
    n_shared_experts: int = 1
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0

    # sparse / lightning indexer (layers 3..59)
    sparse_block_size: int = 128
    sparse_topk_blocks: int = 16
    sparse_index_dim: int = 128
    sparse_num_index_heads: int = 4
    sparse_local_block: int = 1
    sparse_init_block: int = 0

    @property
    def gqa_n_rep(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads  # 16

    @property
    def q_proj_out(self) -> int:
        return self.num_attention_heads * self.head_dim  # 8192

    @property
    def kv_proj_out(self) -> int:
        return self.num_key_value_heads * self.head_dim  # 512


# ---------------------------------------------------------------------------
# HF checkpoint key prefixes (confirmed in model.safetensors.index.json)
# ---------------------------------------------------------------------------
TEXT_MODEL_PREFIX = "language_model.model"
LM_HEAD_KEY = "language_model.lm_head.weight"
EMBED_TOKENS_KEY = f"{TEXT_MODEL_PREFIX}.embed_tokens.weight"
FINAL_NORM_KEY = f"{TEXT_MODEL_PREFIX}.norm.weight"


def layer_key(layer_idx: int, suffix: str) -> str:
    """Build a full HF weight key for a text decoder layer.

    e.g. ``layer_key(0, "self_attn.q_proj.weight")`` ->
    ``language_model.model.layers.0.self_attn.q_proj.weight``.
    """
    return f"{TEXT_MODEL_PREFIX}.layers.{layer_idx}.{suffix}"


# ---------------------------------------------------------------------------
# Mesh lifecycle
# ---------------------------------------------------------------------------
def open_mesh() -> "ttnn.MeshDevice":
    """Open the bh_galaxy mesh once (set fabric first). Reuse across all blocks."""
    ttnn.set_fabric_config(FABRIC_CONFIG)
    return ttnn.open_mesh_device(ttnn.MeshShape(MESH_ROWS, MESH_COLS))


def close_mesh(mesh: "ttnn.MeshDevice") -> None:
    ttnn.close_mesh_device(mesh)


# ---------------------------------------------------------------------------
# TP=32 sharding-recipe mesh-mapper helpers
#
#   column-parallel  : shard OUTPUT dim (-1)  -> q/k/v/gate/up projections
#   row-parallel      : shard INPUT  dim (-2)  -> o_proj/down_proj/expert-w2
#                       (all-reduce / reduce-scatter the partials after)
#   replicate         : norms, biases, small tables
#   vocab-shard       : lm_head shards the vocab dim (-1); embedding replicates
# ---------------------------------------------------------------------------
def column_parallel_mapper(mesh):
    """Shard a weight's OUTPUT dim across the mesh (column-parallel)."""
    return ttnn.ShardTensorToMesh(mesh, dim=-1)


def row_parallel_mapper(mesh):
    """Shard a weight's INPUT dim across the mesh (row-parallel)."""
    return ttnn.ShardTensorToMesh(mesh, dim=-2)


def replicate_mapper(mesh):
    """Replicate a tensor on every device (norms / biases / small tables)."""
    return ttnn.ReplicateTensorToMesh(mesh)


def vocab_shard_mapper(mesh):
    """Shard the vocab dim (-1) for the lm_head output projection."""
    return ttnn.ShardTensorToMesh(mesh, dim=-1)


def shard_dim_mapper(mesh, dim: int):
    """Generic shard along an explicit dim."""
    return ttnn.ShardTensorToMesh(mesh, dim=dim)


# Composers for reading sharded / replicated results back to torch for PCC.
def concat_composer(mesh, dim: int):
    """Reassemble a sharded mesh tensor along ``dim`` for host PCC."""
    return ttnn.ConcatMeshToTensor(mesh, dim=dim)


@dataclass
class ModelConfig:
    """Bundle the text config + dtype discipline for the bring-up session."""

    text: TextConfig = field(default_factory=TextConfig)
    mesh_shape: tuple = MESH_SHAPE
    tp: int = TP
    fabric: object = FABRIC_CONFIG
    topology: object = CCL_TOPOLOGY
    act_dtype: object = ACT_DTYPE
    weight_dtype: object = WEIGHT_DTYPE
    norm_weight_dtype: object = NORM_WEIGHT_DTYPE

    def compute_kernel_config(self):
        return default_compute_kernel_config()
