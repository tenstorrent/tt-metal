# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.5-27B-FP8 model configuration.

Provides:
- Qwen35ModelArgs: ModelArgs subclass with Qwen3.5-specific overrides
- DRAM-sharded memory and matmul program configs for TP=4
- FP8 block-wise dequantization (128x128 blocks)
- Weight loading with HF->tt key remapping and mesh tensor caching
"""

import json
import math
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs

# ── GDN Architecture Constants (not in HF config) ──────────────────────────
# These are fixed for Qwen3.5-27B and not derivable from the standard HF config fields.

GDN_Nk = 16  # Key heads
GDN_Dk = 128  # Key head dim
GDN_Nv = 48  # Value heads
GDN_Dv = 128  # Value head dim
GDN_CONV_KERNEL_SIZE = 4

GDN_QKV_DIM = GDN_Nk * GDN_Dk + GDN_Nk * GDN_Dk + GDN_Nv * GDN_Dv  # 10240
GDN_Z_DIM = GDN_Nv * GDN_Dv  # 6144
GDN_KEY_DIM = GDN_Nk * GDN_Dk  # 2048
GDN_VALUE_DIM = GDN_Nv * GDN_Dv  # 6144

# Partial RoPE: only 64 of 256 head dims are rotated
ROPE_DIM = 64


# ── Hardware Constants ──────────────────────────────────────────────────────

TILE_SIZE = 32
DRAM_CORES = 8
DRAM_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(DRAM_CORES - 1, 0))})


# ── Helper Functions ────────────────────────────────────────────────────────


def _roundup(a, b):
    return b * math.ceil(a / b)


def _find_largest_divisor(n, max_div=8):
    for d in range(max_div, 0, -1):
        if n % d == 0:
            return d
    return 1


def _find_grid(n_tiles, target=32):
    max_r, max_c = 8, 8
    possible = [k for k in range(1, max_r * max_c + 1) if n_tiles % k == 0]
    possible.sort(key=lambda x: abs(x - target))
    for cores in possible:
        for rows in range(1, max_r + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_c:
                    return rows, cols
    raise ValueError(f"Cannot find grid for {n_tiles} tiles")


# ── DRAM-Sharded Config Builders ───────────────────────────────────────────


def create_dram_sharded_mem_config(k, n):
    """Create a WIDTH_SHARDED DRAM memory config for weight matrices."""
    padded_n = _roundup(n, TILE_SIZE * DRAM_CORES)
    shard_spec = ttnn.ShardSpec(
        DRAM_GRID,
        (k, padded_n // DRAM_CORES),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )


def create_dram_sharded_matmul_program_config(m, k, n, num_cores=None):
    """Create a DRAM-sharded matmul program config."""
    m_tiles = math.ceil(m / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)
    n_padded = _roundup(n, TILE_SIZE * DRAM_CORES)
    n_tiles = n_padded // TILE_SIZE

    if num_cores is None:
        rows, cols = _find_grid(k_tiles)
        num_cores = rows * cols

    k_tiles_per_core = k_tiles // num_cores
    if k_tiles_per_core == 0:
        k_tiles_per_core = k_tiles
        num_cores = 1
    in0_block_w = _find_largest_divisor(k_tiles_per_core)
    per_core_N = n_tiles // num_cores if n_tiles >= num_cores else 1

    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=m_tiles,
        per_core_N=per_core_N,
        fused_activation=None,
    )


def create_activation_shard_config(k):
    """Create a WIDTH_SHARDED L1 activation config."""
    k_tiles = k // TILE_SIZE
    rows, cols = _find_grid(k_tiles)
    num_cores = rows * cols
    width_per_core = k // num_cores
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, width_per_core),
        core_grid=ttnn.CoreGrid(x=cols, y=rows),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# ── 2D Matmul Config Builder (Prefill) ────────────────────────────────────


def _get_out_subblock_w(per_core_n, out_subblock_h):
    """Find largest out_subblock_w such that h*w <= 4 (FP32 DST limit on WH)."""
    for w in range(min(per_core_n, 4 // out_subblock_h), 0, -1):
        if per_core_n % w == 0:
            return w
    return 1


def create_prefill_matmul_program_config(m, k, n, grid_size=(8, 8)):
    """Create a 2D matmul program config for prefill (compute-bound).

    Inputs/outputs are DRAM interleaved. M is parallelized over grid_y,
    N over grid_x. Following the tech report pattern (Section 4.3.2.1).
    """
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid_size[1]))
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid_size[0]))

    out_subblock_h = 1
    out_subblock_w = _get_out_subblock_w(per_core_N, out_subblock_h)

    # in0_block_w: how many K-tiles to process at a time; higher = better throughput
    k_tiles = math.ceil(k / TILE_SIZE)
    in0_block_w = min(4, max(1, k_tiles // grid_size[0]))

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


# ── Compute Kernel Configs ─────────────────────────────────────────────────

COMPUTE_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

COMPUTE_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


# ── FP8 Dequantization ────────────────────────────────────────────────────


def dequant_fp8_block(weight_fp8, scale_inv, block_size=128):
    """Dequantize a block-wise FP8 weight tensor to bfloat16."""
    out_f, in_f = weight_fp8.shape
    weight_bf16 = weight_fp8.to(torch.bfloat16).reshape(out_f // block_size, block_size, in_f // block_size, block_size)
    weight_bf16 = weight_bf16 * scale_inv[:, None, :, None].to(torch.bfloat16)
    return weight_bf16.reshape(out_f, in_f)


# ── Qwen35ModelArgs ────────────────────────────────────────────────────────


class Qwen35ModelArgs(ModelArgs):
    """ModelArgs subclass for Qwen3.5-27B-FP8.

    Overrides:
    - rms_norm_add_unit_offset = True (GemmaRMSNorm format)
    - GDN-specific constants not in standard HF config
    - DRAM-sharded mem/program configs for all weight matmuls
    - Partial RoPE (64 of 256 dims)
    - KV head replication when n_kv_heads < num_devices (e.g. TP=8 with 4 KV heads)
    """

    # Register local HF params so dummy_weights can find the config
    LOCAL_HF_PARAMS = {
        **ModelArgs.LOCAL_HF_PARAMS,
        "Qwen3.5-27B-FP8": "models/demos/qwen35_27b/model_params/Qwen3.5-27B-FP8",
    }

    def _set_params_from_dict(self, config):
        """Override to handle n_kv_heads < num_devices (KV head replication).

        The parent ModelArgs asserts n_kv_heads % num_devices == 0. For TP=8 with
        4 KV heads, we temporarily set n_kv_heads to num_devices so the assertion
        passes, then restore the real value after super().__init__().
        """
        super()._set_params_from_dict(config)
        # Save the real KV head count before the parent assertion checks it
        self._real_n_kv_heads = self.n_kv_heads
        if self.num_devices > 0 and self.n_kv_heads % self.num_devices != 0:
            # Bump to satisfy parent assertion; we handle replication ourselves
            self.n_kv_heads = self.num_devices

    def __init__(self, mesh_device, **kwargs):
        super().__init__(mesh_device, **kwargs)

        # Restore real n_kv_heads after parent init
        if hasattr(self, "_real_n_kv_heads"):
            self.n_kv_heads = self._real_n_kv_heads

        # ── Qwen3.5-specific overrides ──────────────────────────────────
        self.rms_norm_add_unit_offset = True
        self.rope_dim = ROPE_DIM
        self.rope_theta = 10_000_000.0
        self.use_qk_fused = False  # Custom partial RoPE, not framework fused QK

        # KV head replication: each device gets at least 1 KV head
        tp = self.num_devices
        self.n_local_kv_heads = max(1, self.n_kv_heads // tp)
        self.kv_replication = tp > self.n_kv_heads  # True for TP=8 with 4 KV heads

        # GDN constants (not in HF config)
        self.gdn_nk = GDN_Nk
        self.gdn_dk = GDN_Dk
        self.gdn_nv = GDN_Nv
        self.gdn_dv = GDN_Dv
        self.gdn_conv_kernel_size = GDN_CONV_KERNEL_SIZE

        # TP-derived dimensions
        self.gdn_nk_tp = GDN_Nk // tp
        self.gdn_nv_tp = GDN_Nv // tp
        self.gdn_qkv_dim_tp = GDN_QKV_DIM // tp
        self.gdn_z_dim_tp = GDN_Z_DIM // tp
        self.gdn_qkvz_dim_tp = (GDN_QKV_DIM + GDN_Z_DIM) // tp
        self.gdn_value_dim_tp = GDN_VALUE_DIM // tp
        self.gdn_key_dim_tp = GDN_KEY_DIM // tp
        self.attn_out_dim_tp = (self.n_heads * self.head_dim) // tp

        # Per-device KV dim (1 KV head per device regardless of TP)
        kv_dim_per_device = self.n_local_kv_heads * self.head_dim

        # ── DRAM-sharded weight memory configs ─────────────────────────
        # Column-parallel: [hidden, out_dim_tp]
        self.gdn_qkvz_weight_memcfg = create_dram_sharded_mem_config(self.dim, self.gdn_qkvz_dim_tp)
        self.attn_qg_weight_memcfg = create_dram_sharded_mem_config(self.dim, self.n_local_heads * self.head_dim * 2)
        self.attn_k_weight_memcfg = create_dram_sharded_mem_config(self.dim, kv_dim_per_device)
        self.attn_v_weight_memcfg = create_dram_sharded_mem_config(self.dim, kv_dim_per_device)
        self.mlp_w1_weight_memcfg = create_dram_sharded_mem_config(self.dim, self.hidden_dim // tp)
        self.mlp_w3_weight_memcfg = create_dram_sharded_mem_config(self.dim, self.hidden_dim // tp)

        # Row-parallel: [input_dim_tp, hidden]
        self.gdn_out_weight_memcfg = create_dram_sharded_mem_config(self.gdn_value_dim_tp, self.dim)
        self.attn_wo_weight_memcfg = create_dram_sharded_mem_config(self.attn_out_dim_tp, self.dim)
        self.mlp_w2_weight_memcfg = create_dram_sharded_mem_config(self.hidden_dim // tp, self.dim)

        # ── DRAM-sharded matmul program configs (decode, m=1) ──────────
        M = 1
        self.gdn_qkvz_progcfg = create_dram_sharded_matmul_program_config(M, self.dim, self.gdn_qkvz_dim_tp)
        self.gdn_out_progcfg = create_dram_sharded_matmul_program_config(M, self.gdn_value_dim_tp, self.dim)
        self.attn_qg_progcfg = create_dram_sharded_matmul_program_config(
            M, self.dim, self.n_local_heads * self.head_dim * 2
        )
        self.attn_k_progcfg = create_dram_sharded_matmul_program_config(M, self.dim, kv_dim_per_device)
        self.attn_v_progcfg = create_dram_sharded_matmul_program_config(M, self.dim, kv_dim_per_device)
        self.attn_wo_progcfg = create_dram_sharded_matmul_program_config(M, self.attn_out_dim_tp, self.dim)

        # ── 2D matmul program configs (prefill, m=seq_len) ─────────
        # These are factory methods since M varies with seq_len.
        self._prefill_grid = (8, 8)

        def _prefill_cfg(seq_len, k, n):
            return create_prefill_matmul_program_config(seq_len, k, n, grid_size=self._prefill_grid)

        self.prefill_progcfg = _prefill_cfg  # General factory: (seq_len, k, n) -> config

        # ── Activation shard configs ───────────────────────────────────
        self.act_shard_hidden = create_activation_shard_config(self.dim)
        self.act_shard_gdn_value = create_activation_shard_config(self.gdn_value_dim_tp)
        self.act_shard_attn_out = create_activation_shard_config(self.attn_out_dim_tp)

        # KV cache height-shard config for paged_update_cache
        self.kv_update_shard_cfg = ttnn.create_sharded_memory_config(
            shape=(TILE_SIZE, self.head_dim),
            core_grid=ttnn.CoreGrid(x=8, y=4),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # On-device sampling config: enable force_argmax for greedy decode
        # P150x4: 4 devices, linear topology, 1 link
        self.model_config["SAMPLING_AG_CONFIG"] = {
            "allow_force_argmax": True,
            "num_links": 1,
            "topology": self.ccl_topology() or ttnn.Topology.Linear,
        }

    def get_state_dict_prefix(self, module_name, layer_num, is_vision=False):
        """Map framework module names to Qwen3.5 state dict key prefixes."""
        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""
        module_map = {
            "MLP": "feed_forward",
            "Attention": "attention",
            "TtGatedDeltaNet": "attention",
            "Qwen35Attention": "attention",
            "TransformerBlock": "",
            "": "",
        }
        mapped = module_map.get(module_name, module_name.lower())
        return layer_prefix + mapped


# ── Weight Preparation Helpers ─────────────────────────────────────────────


def prepare_gdn_qkv(sd, prefix, tp):
    """Interleave Q/K/V heads for clean TP sharding along dim 0."""
    qkv_w = sd[prefix + "linear_attn.in_proj_qkv.weight"]
    q_part = qkv_w[:GDN_KEY_DIM, :]
    k_part = qkv_w[GDN_KEY_DIM : 2 * GDN_KEY_DIM, :]
    v_part = qkv_w[2 * GDN_KEY_DIM :, :]

    q_per = GDN_Nk // tp
    v_per = GDN_Nv // tp

    shards = []
    for s in range(tp):
        q_s = q_part[s * q_per * GDN_Dk : (s + 1) * q_per * GDN_Dk, :]
        k_s = k_part[s * q_per * GDN_Dk : (s + 1) * q_per * GDN_Dk, :]
        v_s = v_part[s * v_per * GDN_Dv : (s + 1) * v_per * GDN_Dv, :]
        shards.append(torch.cat([q_s, k_s, v_s], dim=0))
    return torch.cat(shards, dim=0)


def prepare_attn_qg(sd, prefix, n_heads, head_dim, tp):
    """Return Q+gate weight for TP sharding.

    HF q_proj is per-head interleaved: [Q_h0(hd), gate_h0(hd), Q_h1(hd), gate_h1(hd), ...].
    ShardTensorToMesh(dim=-1) on the transposed weight naturally groups contiguous heads
    per device, preserving the per-head [Q, gate] layout expected by forward_decode.
    """
    return sd[prefix + "attention.wqkv.weight"]


def prepare_conv_taps(sd, prefix, tp):
    """Prepare conv taps interleaved for TP sharding."""
    cw = sd[prefix + "linear_attn.conv1d.weight"].float()
    q_per = GDN_Nk // tp
    v_per = GDN_Nv // tp
    taps = []
    for j in range(GDN_CONV_KERNEL_SIZE):
        tap = cw[:, 0, j]
        q_tap = tap[:GDN_KEY_DIM]
        k_tap = tap[GDN_KEY_DIM : 2 * GDN_KEY_DIM]
        v_tap = tap[2 * GDN_KEY_DIM :]
        shards = []
        for s in range(tp):
            q_s = q_tap[s * q_per * GDN_Dk : (s + 1) * q_per * GDN_Dk]
            k_s = k_tap[s * q_per * GDN_Dk : (s + 1) * q_per * GDN_Dk]
            v_s = v_tap[s * v_per * GDN_Dv : (s + 1) * v_per * GDN_Dv]
            shards.append(torch.cat([q_s, k_s, v_s]))
        taps.append(torch.cat(shards))
    return taps


# ── Mesh Tensor Helpers ────────────────────────────────────────────────────


def replicate_kv_weight(weight, n_kv_heads, tp, head_dim):
    """Replicate KV weight for TP > n_kv_heads.

    Input:  [n_kv_heads * head_dim, hidden]  (e.g. [1024, 5120] for 4 KV heads)
    Output: [tp * head_dim, hidden]           (e.g. [2048, 5120] for TP=8)

    Each KV head is replicated ceil(tp/n_kv_heads) times so each device gets 1 head.
    Device d gets KV head (d * n_kv_heads) // tp.
    """
    if tp <= n_kv_heads:
        return weight  # No replication needed
    chunks = weight.reshape(n_kv_heads, head_dim, -1)
    parts = []
    for d in range(tp):
        kv_idx = (d * n_kv_heads) // tp
        parts.append(chunks[kv_idx])
    return torch.cat(parts, dim=0).reshape(tp * head_dim, -1)


def _shard_w(torch_tensor, mesh, dim, memory_config, cache_path):
    """Convert torch weight to sharded mesh tensor. Transposes [out, in] -> [in, out]."""
    w = torch_tensor.to(torch.bfloat16).T.contiguous()
    return ttnn.as_tensor(
        w,
        dtype=ttnn.bfloat8_b,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        cache_file_name=cache_path,
    )


def _replicate(torch_tensor, mesh, cache_path):
    """Small tensor (norms, biases) -> replicated on all devices."""
    if torch_tensor.dim() == 1:
        torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)
    elif torch_tensor.dim() == 2:
        torch_tensor = torch_tensor.unsqueeze(0)
    return ttnn.as_tensor(
        torch_tensor.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def _shard_small(torch_tensor, mesh, cache_path):
    """Small per-head tensor -> sharded across devices."""
    if torch_tensor.dim() == 1:
        torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)
    elif torch_tensor.dim() == 2:
        torch_tensor = torch_tensor.unsqueeze(0)
    return ttnn.as_tensor(
        torch_tensor.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


# ── State Dict Loading ────────────────────────────────────────────────────


def load_qwen35_state_dict(model_path):
    """Load Qwen3.5-27B-FP8 weights with FP8 dequantization and key remapping."""
    from safetensors import safe_open

    model_path = Path(model_path)
    HF_PREFIX = "model.language_model."

    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    file_to_keys = {}
    for key, filename in weight_map.items():
        file_to_keys.setdefault(filename, []).append(key)

    raw = {}
    for filename, keys in file_to_keys.items():
        filepath = model_path / filename
        with safe_open(str(filepath), framework="pt") as sf:
            for key in keys:
                if key in sf.keys():
                    raw[key] = sf.get_tensor(key)

    # Dequantize FP8
    dequantized = {}
    for key, tensor in raw.items():
        if key.endswith(".weight_scale_inv"):
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = key + "_scale_inv"
            if scale_key in raw:
                dequantized[key] = dequant_fp8_block(tensor, raw[scale_key])
            else:
                dequantized[key] = tensor.to(torch.bfloat16)
        else:
            dequantized[key] = tensor

    # Key remapping: HF keys -> tt_transformers conventions
    state_dict = {}
    for key, tensor in dequantized.items():
        if key.startswith(HF_PREFIX):
            short = key[len(HF_PREFIX) :]
        elif key.startswith("model."):
            short = key[len("model.") :]
        else:
            short = key

        if "embed_tokens" in short:
            state_dict["tok_embeddings.weight"] = tensor
            continue
        if short == "norm.weight":
            state_dict["norm.weight"] = tensor
            continue
        if short == "lm_head.weight" or key == "lm_head.weight":
            state_dict["output.weight"] = tensor
            continue

        if short.startswith("layers."):
            parts = short.split(".", 2)
            layer_idx = int(parts[1])
            rest = parts[2]
            prefix = f"layers.{layer_idx}."

            SHARED = {
                "input_layernorm.weight": "attention_norm.weight",
                "post_attention_layernorm.weight": "ffn_norm.weight",
                "mlp.gate_proj.weight": "feed_forward.w1.weight",
                "mlp.up_proj.weight": "feed_forward.w3.weight",
                "mlp.down_proj.weight": "feed_forward.w2.weight",
            }
            if rest in SHARED:
                state_dict[prefix + SHARED[rest]] = tensor
                continue

            if rest.startswith("linear_attn."):
                state_dict[short] = tensor
                continue

            ATTN_MAP = {
                "self_attn.q_proj.weight": "attention.wqkv.weight",
                "self_attn.k_proj.weight": "attention.wk.weight",
                "self_attn.v_proj.weight": "attention.wv.weight",
                "self_attn.o_proj.weight": "attention.wo.weight",
                "self_attn.q_norm.weight": "attention.q_norm.weight",
                "self_attn.k_norm.weight": "attention.k_norm.weight",
            }
            if rest in ATTN_MAP:
                state_dict[prefix + ATTN_MAP[rest]] = tensor
                continue

            state_dict[short] = tensor
        else:
            state_dict[short] = tensor

    logger.info(f"Loaded Qwen3.5 state dict: {len(state_dict)} keys")
    return state_dict


# ── Weight Loading to Mesh ─────────────────────────────────────────────────


def load_weights_to_mesh(state_dict, mesh, cache_dir, args):
    """Load all Qwen3.5 layer weights as mesh tensors.

    Args:
        state_dict: Dequantized state dict from load_qwen35_state_dict()
        mesh: ttnn mesh device
        cache_dir: Path for ttnn.as_tensor caching
        args: Qwen35ModelArgs instance

    Returns:
        all_layer_weights: dict[layer_idx] -> dict of mesh tensors
        embed: torch embedding table (CPU)
        final_norm: mesh tensor (replicated)
        lm_head: mesh tensor (sharded)
    """
    os.makedirs(cache_dir, exist_ok=True)
    tp = args.num_devices
    n_layers = args.n_layers
    layer_types = args.layer_types

    embed = state_dict["tok_embeddings.weight"].to(torch.bfloat16)

    final_norm = _replicate(
        1.0 + state_dict["norm.weight"].to(torch.bfloat16),
        mesh,
        os.path.join(cache_dir, "final_norm"),
    )

    lm_key = "output.weight" if "output.weight" in state_dict else "tok_embeddings.weight"
    lm_head = _shard_w(
        state_dict[lm_key],
        mesh,
        dim=-1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_path=os.path.join(cache_dir, "lm_head"),
    )

    all_layer_weights = {}
    for i in range(n_layers):
        p = f"layers.{i}."
        ld = os.path.join(cache_dir, f"layer_{i:02d}")
        os.makedirs(ld, exist_ok=True)
        tw = {}

        # Norms (offset format: add 1.0)
        tw["attn_norm"] = _replicate(
            1.0 + state_dict[p + "attention_norm.weight"].to(torch.bfloat16),
            mesh,
            os.path.join(ld, "attn_norm"),
        )
        tw["ffn_norm"] = _replicate(
            1.0 + state_dict[p + "ffn_norm.weight"].to(torch.bfloat16),
            mesh,
            os.path.join(ld, "ffn_norm"),
        )

        # MLP
        tw["w1"] = _shard_w(
            state_dict[p + "feed_forward.w1.weight"],
            mesh,
            dim=-1,
            memory_config=args.mlp_w1_weight_memcfg,
            cache_path=os.path.join(ld, "w1"),
        )
        tw["w3"] = _shard_w(
            state_dict[p + "feed_forward.w3.weight"],
            mesh,
            dim=-1,
            memory_config=args.mlp_w3_weight_memcfg,
            cache_path=os.path.join(ld, "w3"),
        )
        tw["w2"] = _shard_w(
            state_dict[p + "feed_forward.w2.weight"],
            mesh,
            dim=0,
            memory_config=args.mlp_w2_weight_memcfg,
            cache_path=os.path.join(ld, "w2"),
        )

        if layer_types[i] == "linear_attention":
            # Fused QKV+Z
            qkv_reordered = prepare_gdn_qkv(state_dict, p, tp)
            z_weight = state_dict[p + "linear_attn.in_proj_z.weight"]
            qkv_per = args.gdn_qkv_dim_tp
            z_per = args.gdn_z_dim_tp
            fused_parts = []
            for d in range(tp):
                fused_parts.append(
                    torch.cat(
                        [
                            qkv_reordered[d * qkv_per : (d + 1) * qkv_per, :],
                            z_weight[d * z_per : (d + 1) * z_per, :],
                        ],
                        dim=0,
                    )
                )
            qkvz_fused = torch.cat(fused_parts, dim=0)
            tw["qkvz"] = _shard_w(
                qkvz_fused, mesh, dim=-1, memory_config=args.gdn_qkvz_weight_memcfg, cache_path=os.path.join(ld, "qkvz")
            )

            # Fused A+B projection
            a_w = state_dict[p + "linear_attn.in_proj_a.weight"]
            b_w = state_dict[p + "linear_attn.in_proj_b.weight"]
            a_per = args.gdn_nv_tp
            b_per = args.gdn_nv_tp
            ab_parts = []
            for d in range(tp):
                ab_parts.append(
                    torch.cat(
                        [
                            a_w[d * a_per : (d + 1) * a_per, :],
                            b_w[d * b_per : (d + 1) * b_per, :],
                        ],
                        dim=0,
                    )
                )
            ab_fused = torch.cat(ab_parts, dim=0)
            tw["ab"] = _shard_w(
                ab_fused, mesh, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_path=os.path.join(ld, "ab")
            )

            # Out: row-parallel
            tw["out"] = _shard_w(
                state_dict[p + "linear_attn.out_proj.weight"],
                mesh,
                dim=0,
                memory_config=args.gdn_out_weight_memcfg,
                cache_path=os.path.join(ld, "out"),
            )

            # Per-head params
            tw["A_log"] = _shard_small(state_dict[p + "linear_attn.A_log"].float(), mesh, os.path.join(ld, "A_log"))
            tw["dt_bias"] = _shard_small(
                state_dict[p + "linear_attn.dt_bias"].float(), mesh, os.path.join(ld, "dt_bias")
            )
            tw["norm_w"] = _replicate(
                state_dict[p + "linear_attn.norm.weight"].float(), mesh, os.path.join(ld, "norm_w")
            )

            # Conv taps
            taps = prepare_conv_taps(state_dict, p, tp)
            tw["conv_taps"] = [
                _shard_small(taps[j], mesh, os.path.join(ld, f"conv_tap_{j}")) for j in range(GDN_CONV_KERNEL_SIZE)
            ]
        else:
            # Full attention
            qg_reordered = prepare_attn_qg(state_dict, p, args.n_heads, args.head_dim, tp)
            tw["wqkv"] = _shard_w(
                qg_reordered,
                mesh,
                dim=-1,
                memory_config=args.attn_qg_weight_memcfg,
                cache_path=os.path.join(ld, "wqkv"),
            )
            tw["wk"] = _shard_w(
                state_dict[p + "attention.wk.weight"],
                mesh,
                dim=-1,
                memory_config=args.attn_k_weight_memcfg,
                cache_path=os.path.join(ld, "wk"),
            )
            tw["wv"] = _shard_w(
                state_dict[p + "attention.wv.weight"],
                mesh,
                dim=-1,
                memory_config=args.attn_v_weight_memcfg,
                cache_path=os.path.join(ld, "wv"),
            )
            tw["wo"] = _shard_w(
                state_dict[p + "attention.wo.weight"],
                mesh,
                dim=0,
                memory_config=args.attn_wo_weight_memcfg,
                cache_path=os.path.join(ld, "wo"),
            )
            tw["q_norm"] = _replicate(
                state_dict[p + "attention.q_norm.weight"].to(torch.bfloat16), mesh, os.path.join(ld, "q_norm_v2")
            )
            tw["k_norm"] = _replicate(
                state_dict[p + "attention.k_norm.weight"].to(torch.bfloat16), mesh, os.path.join(ld, "k_norm_v2")
            )

        all_layer_weights[i] = tw
        logger.info(f"  Layer {i:2d}/{n_layers} ({layer_types[i][:6]})")

    return all_layer_weights, embed, final_norm, lm_head
