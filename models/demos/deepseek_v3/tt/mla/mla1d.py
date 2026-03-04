# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import math
import os
from pathlib import Path
from typing import Sequence

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import models.experimental.ops.descriptors as descriptors
import models.experimental.ops.descriptors.composite as composite
import ttnn
from models.common.utility_functions import nearest_y
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.rms_norm.rms_norm import RMSNorm
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllBroadcastAsyncConfig,
    AllGatherAsyncConfig,
    AllToAllAsyncGenericConfig,
    ConcatConfig,
    FromWeightConfig,
    KvCacheConfig,
    LinearConfig,
    MeshDeviceStub,
    PermuteConfig,
    ReshardConfig,
    SavedWeight,
    SliceConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import (
    USERS_PER_ROW,
    dequantize,
    even_int_div,
    get_mesh_coords,
    get_state_dicts,
    shard_and_save,
    sub_state_dicts,
)
from models.demos.deepseek_v3.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)
from models.tt_transformers.tt.common import PagedAttentionConfig


def pad_n_to_dram_banks(n, tile_size=32, num_dram_banks=12):
    """Pad n dimension to be divisible by tile_size * num_dram_banks (default 32 and 12 respectively)."""
    lcm = tile_size * num_dram_banks
    remainder = n % lcm
    if remainder == 0:
        return n
    return n + (lcm - remainder)


def pad_batch_to_dram_banks(batch, num_banks=12):
    """Pad batch dimension to be divisible by number of DRAM banks (default 12)."""
    if batch % num_banks == 0:
        return batch
    return ((batch + num_banks - 1) // num_banks) * num_banks


def build_prefill_matmul_program_config(seq_len, k, n, batch=1, tile_h=32, tile_w=32):
    """Build MatmulMultiCoreReuseMultiCastProgramConfig for prefill matmuls.

    Handles both unbatched (batch=1, fuse_batch=True) and batched (batch>1, fuse_batch=False)
    cases for interleaved in0 + DRAM sharded in1.

    Args:
        seq_len: Sequence length (M dimension)
        k: Inner dimension (K)
        n: Output dimension (N)
        batch: Batch dimension (1 for unbatched, >1 for batched)
        tile_h: Tile height (default 32)
        tile_w: Tile width (default 32)

    Returns:
        MatmulMultiCoreReuseMultiCastProgramConfig
    """
    M_tiles = even_int_div(seq_len, tile_h)
    K_tiles = even_int_div(k, tile_w)
    N_tiles = even_int_div(n, tile_w)

    # grid_x splits N dimension; grid_y splits M dimension
    grid_x = 1
    for x in range(min(8, N_tiles), 0, -1):
        if N_tiles % x == 0:
            grid_x = x
            break

    grid_y = 1
    for y in range(min(8, M_tiles), 0, -1):
        if M_tiles % y == 0:
            grid_y = y
            break

    grid_size = (grid_x, grid_y)

    in0_block_h = M_tiles // grid_y
    out_block_h = in0_block_h
    out_block_w = N_tiles // grid_x
    per_core_M = in0_block_h
    per_core_N = out_block_w

    # in0_block_w (inner dim block) must divide K_tiles.
    # Target: keep in1 CB tiles (in0_block_w * out_block_w * 2) under ~256 tiles
    # while maximizing in0_block_w to minimize inner loop iterations.
    # BF8 is 1088 Bytes per tile, so 256 tiles ≈ 272 KB
    # This is just under a quarter of the WH CB budget, thereby leaving room for in0, interm0, and output CBs.
    max_in1_cb_tiles = 256
    in0_block_w = K_tiles
    while in0_block_w > 1:
        if K_tiles % in0_block_w == 0 and in0_block_w * out_block_w * 2 <= max_in1_cb_tiles:
            break
        in0_block_w -= 1

    # Ensure in0 CB is also reasonable
    while in0_block_w > 1 and in0_block_h * in0_block_w * 2 > max_in1_cb_tiles:
        in0_block_w = in0_block_w // 2
        if K_tiles % in0_block_w != 0:
            while in0_block_w > 1 and K_tiles % in0_block_w != 0:
                in0_block_w -= 1

    # Determine out_block_h to make sure the matmul will fit in L1.
    max_l1_usage = 1200 * 1024  # ~1.2 MB target
    actual_out_block_h = out_block_h
    for candidate_h in range(out_block_h, 0, -1):
        if out_block_h % candidate_h != 0:
            continue
        in0_cb = candidate_h * in0_block_w * 2 * 2048  # bf16 double-buffered
        in1_cb = out_block_w * in0_block_w * 2 * 1088  # bf8b double-buffered
        interm0_cb = candidate_h * out_block_w * 4096  # fp32
        output_cb = candidate_h * out_block_w * 2048  # bf16
        total = in0_cb + in1_cb + interm0_cb + output_cb
        if total <= max_l1_usage:
            actual_out_block_h = candidate_h
            break

    # Subblock calculation (h * w <= 4, hardware limit)
    out_subblock_w = 1
    for sw in range(min(out_block_w, 4), 0, -1):
        if out_block_w % sw == 0:
            out_subblock_w = sw
            break
    max_sh = 4 // out_subblock_w
    out_subblock_h = 1
    for sh in range(min(actual_out_block_h, max_sh), 0, -1):
        if actual_out_block_h % sh == 0:
            out_subblock_h = sh
            break

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=actual_out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=(batch == 1),
    )


def _deepseek_kvdbg_enabled() -> bool:
    return os.getenv("DEEPSEEK_KVDBG", "").lower() in ("1", "true", "yes", "y")


class MLA1D(AbstractModule):
    """
    Multi-Latent Attention Module for 1D tensor parallelism.
    """

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        num_shards = mesh_device.shape[0]
        weight_block_height, weight_block_width = hf_config.quantization_config["weight_block_size"]

        dim = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        kv_lora_rank = hf_config.kv_lora_rank
        qk_nope_head_dim = hf_config.qk_nope_head_dim
        qk_rope_head_dim = hf_config.qk_rope_head_dim
        v_head_dim = hf_config.v_head_dim
        q_lora_rank = hf_config.q_lora_rank
        q_head_dim = qk_nope_head_dim + qk_rope_head_dim

        # Norm weights
        norm_weight_configs = {
            ttnn_name: RMSNorm.convert_weights(
                hf_config,
                sub_state_dicts(state_dicts, f"{hf_name}."),
                output_path / ttnn_name,
                mesh_device,
            )
            for hf_name, ttnn_name in [("q_a_layernorm", "q_norm"), ("kv_a_layernorm", "kv_norm")]
        }

        # DRAM sharding configuration
        num_dram_banks = mesh_device.dram_grid_size().x

        # Create DRAM WIDTH sharded memory config for wq_b
        # wq_b: k=q_lora_rank, n=num_heads*q_head_dim (sharded by TP)
        wq_b_k = q_lora_rank  # 1536
        wq_b_n = even_int_div(num_heads * q_head_dim, mesh_device.shape[1])  # 3072 per device
        wq_b_n_padded = pad_n_to_dram_banks(
            wq_b_n, tile_size=32, num_dram_banks=num_dram_banks
        )  # 3072 (already aligned)
        wq_b_shard_shape = [wq_b_k, wq_b_n_padded // num_dram_banks]
        wq_b_dram_shard_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
        )
        wq_b_dram_shard_spec = ttnn.ShardSpec(wq_b_dram_shard_grid, wq_b_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        wq_b_dram_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, wq_b_dram_shard_spec
        )

        # Create DRAM WIDTH sharded memory config for wo
        # wo: k=num_heads*v_head_dim, n=dim
        wo_k = num_heads * v_head_dim  # 16384
        wo_n = even_int_div(dim, mesh_device.shape[1])  # 896
        wo_n_padded = pad_n_to_dram_banks(wo_n, tile_size=32, num_dram_banks=num_dram_banks)  # 1152
        wo_shard_shape = [wo_k, wo_n_padded // num_dram_banks]
        wo_dram_shard_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
        )
        wo_dram_shard_spec = ttnn.ShardSpec(wo_dram_shard_grid, wo_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        wo_dram_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, wo_dram_shard_spec
        )

        # Regular non-split weights with DRAM sharded configs
        wq_b_weight = dequantize(
            get_state_dicts(
                state_dicts, "q_b_proj.weight", (num_heads * q_head_dim, q_lora_rank), dtype=torch.float8_e4m3fn
            ),
            get_state_dicts(state_dicts, "q_b_proj.weight_scale_inv", dtype=torch.float32),
            (1, weight_block_height, weight_block_width),
        )
        wo_weight = dequantize(
            get_state_dicts(state_dicts, "o_proj.weight", (dim, num_heads * v_head_dim), dtype=torch.float8_e4m3fn),
            get_state_dicts(state_dicts, "o_proj.weight_scale_inv", dtype=torch.float32),
            (1, weight_block_height, weight_block_width),
        )

        linear_weight_configs = {
            "wq_b": {
                "input_tensor_b": cls._convert_weight(
                    output_path / "wq_b.input_tensor_b",
                    wq_b_weight,
                    (0, -1),
                    mesh_device,
                    wq_b_dram_memory_config,
                    (0, 0, 0),  # n=3072 already aligned (multiple of 384), no padding needed
                ),
            },
            "wo": {
                "input_tensor_b": cls._convert_weight(
                    output_path / "wo.input_tensor_b",
                    wo_weight,
                    (0, -1),
                    mesh_device,
                    wo_dram_memory_config,
                    (0, 0, 0),  # Pad n from 896 to 1152 (multiple of 384)
                ),
            },
        }

        # Fused wq_a and wkv_a weights: concatenated along output dimension
        # Output order: [q_lora_rank | kv_lora_rank | qk_rope_head_dim]
        wq_a_weight = dequantize(
            get_state_dicts(state_dicts, "q_a_proj.weight", (q_lora_rank, dim), dtype=torch.float8_e4m3fn),
            get_state_dicts(state_dicts, "q_a_proj.weight_scale_inv", dtype=torch.float32),
            (1, weight_block_height, weight_block_width),
        )
        wkv_a_weight = dequantize(
            get_state_dicts(
                state_dicts,
                "kv_a_proj_with_mqa.weight",
                (kv_lora_rank + qk_rope_head_dim, dim),
                dtype=torch.float8_e4m3fn,
            ),
            get_state_dicts(state_dicts, "kv_a_proj_with_mqa.weight_scale_inv", dtype=torch.float32),
            (1, weight_block_height, weight_block_width),
        )
        # Concatenate: [num_shards, q_lora_rank + kv_lora_rank + qk_rope_head_dim, dim]
        wq_kv_a_weight = torch.cat([wq_a_weight, wkv_a_weight], dim=-2)

        # Create DRAM WIDTH sharded memory config for wq_kv_a
        # wq_kv_a: k=dim, n=q_lora_rank+kv_lora_rank+qk_rope_head_dim
        qkv_a_k = even_int_div(dim, mesh_device.shape[1])  # 896
        qkv_a_n = q_lora_rank + kv_lora_rank + qk_rope_head_dim  # 2112
        qkv_a_n_padded = pad_n_to_dram_banks(qkv_a_n, tile_size=32, num_dram_banks=num_dram_banks)  # 2304
        qkv_a_shard_shape = [qkv_a_k, qkv_a_n_padded // num_dram_banks]
        qkv_a_dram_shard_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
        )
        qkv_a_dram_shard_spec = ttnn.ShardSpec(
            qkv_a_dram_shard_grid, qkv_a_shard_shape, ttnn.ShardOrientation.ROW_MAJOR
        )
        qkv_a_dram_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, qkv_a_dram_shard_spec
        )

        fused_weight_configs = {
            "wq_kv_a": {
                "input_tensor_b": cls._convert_weight(
                    output_path / "wq_kv_a.input_tensor_b",
                    wq_kv_a_weight,
                    (0, -2),  # Shard along input dim
                    mesh_device,
                    qkv_a_dram_memory_config,
                    (0, 0, 0),
                ),
            },
        }

        # wkv_b (Needs Special handling!!)
        torch_weights = dequantize(
            get_state_dicts(
                state_dicts,
                f"kv_b_proj.weight",
                shape=(num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank),
                dtype=torch.float8_e4m3fn,
            ),
            get_state_dicts(state_dicts, f"kv_b_proj.weight_scale_inv", dtype=torch.float32),
            (1, weight_block_height, weight_block_width),
        ).reshape(num_shards, num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)

        torch_weights_k = torch_weights[..., :qk_nope_head_dim, :].transpose(
            -2, -1
        )  # [num_shards, num_heads, kv_lora_rank, qk_nope_head_dim]
        torch_weights_v = torch_weights[..., qk_nope_head_dim:, :]  # [num_shards, num_heads, v_head_dim, kv_lora_rank]

        # Create DRAM HEIGHT sharded memory config for wkv_b1 (batch sharding)
        # wkv_b1: batch=num_heads_local, k=qk_nope_head_dim, n=kv_lora_rank
        # After transpose in _convert_weight: k=qk_nope_head_dim, n=kv_lora_rank
        num_heads_local = even_int_div(num_heads, mesh_device.shape[1])
        wkv_b1_batch = num_heads_local
        wkv_b1_batch_padded = pad_batch_to_dram_banks(wkv_b1_batch, num_dram_banks)
        wkv_b1_k = qk_nope_head_dim  # 128
        wkv_b1_n = kv_lora_rank  # 512
        batches_per_dram_bank_b1 = wkv_b1_batch_padded // num_dram_banks
        wkv_b1_shard_shape = [batches_per_dram_bank_b1 * wkv_b1_k, wkv_b1_n]
        wkv_b1_dram_shard_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
        )
        wkv_b1_dram_shard_spec = ttnn.ShardSpec(
            wkv_b1_dram_shard_grid, wkv_b1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR
        )
        wkv_b1_dram_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, wkv_b1_dram_shard_spec
        )

        # Create DRAM HEIGHT sharded memory config for wkv_b2 (batch sharding)
        # wkv_b2: batch=num_heads, k=kv_lora_rank, n=v_head_dim
        # After transpose in _convert_weight: k=kv_lora_rank, n=v_head_dim
        wkv_b2_batch = num_heads  # Full num_heads after all_gather
        wkv_b2_batch_padded = pad_batch_to_dram_banks(wkv_b2_batch, num_dram_banks)
        wkv_b2_k = kv_lora_rank  # 512
        wkv_b2_n = v_head_dim  # 128
        batches_per_dram_bank_b2 = wkv_b2_batch_padded // num_dram_banks
        wkv_b2_shard_shape = [batches_per_dram_bank_b2 * wkv_b2_k, wkv_b2_n]
        wkv_b2_dram_shard_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
        )
        wkv_b2_dram_shard_spec = ttnn.ShardSpec(
            wkv_b2_dram_shard_grid, wkv_b2_shard_shape, ttnn.ShardOrientation.ROW_MAJOR
        )
        wkv_b2_dram_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, wkv_b2_dram_shard_spec
        )

        return {
            **norm_weight_configs,
            **linear_weight_configs,
            **fused_weight_configs,
            "wkv_b1": {
                "input_tensor_b": cls._convert_weight(
                    output_path / "wkv_b1.input_tensor_b",
                    torch_weights_k,
                    (0, -3),
                    mesh_device,
                    wkv_b1_dram_memory_config,
                    (0, 0, 0),  # Pad batch from 16 to 24
                ),
            },
            "wkv_b2": {
                "input_tensor_b": cls._convert_weight(
                    output_path / "wkv_b2.input_tensor_b",
                    torch_weights_v,
                    (0, None),
                    mesh_device,
                    wkv_b2_dram_memory_config,
                    (0, 0, 0),  # Pad batch from 128 to 132
                ),
            },
        }

    @classmethod
    def _convert_weight(
        cls,
        path: Path,
        torch_metaweight: torch.Tensor,
        dims: tuple[int | None, int | None],
        mesh_device: ttnn.MeshDevice,
        memory_config: ttnn.MemoryConfig,
        padding_needed: tuple[int, int, int] = (0, 0, 0),
    ) -> SavedWeight:
        return shard_and_save(
            path,
            torch_metaweight.transpose(-2, -1),
            shard_dims=dims,
            mesh_device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            padding_needed=padding_needed,
        )

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelPrefillConfig:
        """Prefill model config for an MLP with 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for prefill mode
        """
        grid_size = mesh_device.compute_with_storage_grid_size()

        # Extract dimensions from HF config
        num_heads = hf_config.num_attention_heads
        q_lora_rank = hf_config.q_lora_rank
        kv_lora_rank = hf_config.kv_lora_rank
        qk_nope_head_dim = hf_config.qk_nope_head_dim
        qk_rope_head_dim = hf_config.qk_rope_head_dim
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        v_head_dim = hf_config.v_head_dim
        mscale = hf_config.rope_scaling["mscale"]
        rope_factor = hf_config.rope_scaling["factor"]

        mesh_shape = list(mesh_device.shape)
        num_heads_local = even_int_div(num_heads, mesh_shape[1])

        input_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Compute kernel config for prefill matmuls (LoFi for performance)
        prefill_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Fused wq_a and wkv_a config
        wq_kv_a_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=prefill_compute_kernel_config,
        )

        wq_b_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=prefill_compute_kernel_config,
        )

        wkv_b1_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=prefill_compute_kernel_config,
        )

        wkv_b2_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=prefill_compute_kernel_config,
        )

        wo_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=prefill_compute_kernel_config,
        )

        # FlashMLA
        q_chunk_size = 128  # TODO: Make dynamic?
        k_chunk_size = 128  # TODO: Make dynamic?

        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        flash_mla_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        scale = qk_head_dim**-0.5
        if rope_factor > 1.0:
            mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
            scale = scale * mscale * mscale

        flash_mla_config = {
            "head_dim_v": kv_lora_rank,
            "scale": scale,
            "program_config": sdpa_program_config,
            "compute_kernel_config": flash_mla_compute_kernel_config,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "attn_mask": None,
            "is_causal": True,
        }

        # Norms
        q_norm_config = RMSNorm.prefill_model_config(
            hf_config,
            mesh_device,
        )
        kv_norm_config = RMSNorm.prefill_model_config(
            hf_config,
            mesh_device,
        )

        # Set up CCLs

        # Fused wq_kv_a: AG + local reduce (since sub-tile RS not supported for new shapes)
        wq_kv_a_ag_config = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_device.shape),
            cluster_axis=1,
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        wq_kv_a_r_config = {
            "dims": [1],
            "output": None,
            "compute_kernel_config": ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        }

        # WO
        wo_ag_config = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_device.shape),
            cluster_axis=1,
            dim=2,  # Changed from dim=1 to dim=2 to gather after permute in prefill
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        wkv_b2_ag_config = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_device.shape),
            cluster_axis=1,
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return {
            "num_heads": num_heads,
            "num_heads_local": num_heads_local,
            "q_lora_rank": q_lora_rank,
            "kv_lora_rank": kv_lora_rank,
            "qk_nope_head_dim": qk_nope_head_dim,
            "qk_rope_head_dim": qk_rope_head_dim,
            "qk_head_dim": qk_head_dim,
            "v_head_dim": v_head_dim,
            "input_memory_config": input_memory_config,
            "wq_kv_a": wq_kv_a_config,
            "wq_b": wq_b_config,
            "wkv_b1": wkv_b1_config,
            "wkv_b2": wkv_b2_config,
            "wo": wo_config,
            "flash_mla": flash_mla_config,
            "q_norm": q_norm_config,
            "kv_norm": kv_norm_config,
            "wq_kv_a_ag_prefill": wq_kv_a_ag_config,
            "wq_kv_a_r_prefill": wq_kv_a_r_config,
            "wkv_b2_ag_prefill": wkv_b2_ag_config,
            "wo_ag_prefill": wo_ag_config,
            "mesh_device": mesh_device,
        }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelDecodeConfig:
        """Generate decode operator configuration for this MLP layer.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for decode mode
        """
        grid_size = mesh_device.compute_with_storage_grid_size()
        num_cores = grid_size.x * grid_size.y

        # Extract dimensions from HF config
        num_heads = hf_config.num_attention_heads
        q_lora_rank = hf_config.q_lora_rank
        kv_lora_rank = hf_config.kv_lora_rank
        qk_nope_head_dim = hf_config.qk_nope_head_dim
        qk_rope_head_dim = hf_config.qk_rope_head_dim
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        v_head_dim = hf_config.v_head_dim
        mscale = hf_config.rope_scaling["mscale"]
        rope_factor = hf_config.rope_scaling["factor"]

        mesh_shape = list(mesh_device.shape)
        num_heads_local = even_int_div(num_heads, mesh_shape[1])
        hidden_size_per_device = even_int_div(hf_config.hidden_size, mesh_shape[1])

        input_memory_config = ttnn.create_sharded_memory_config(
            shape=(
                ttnn.core.roundup(USERS_PER_ROW, ttnn.TILE_SIZE),
                hidden_size_per_device,
            ),
            core_grid=ttnn.CoreGrid(y=7, x=4),
            strategy=ttnn.ShardStrategy.WIDTH,
        )

        # DRAM sharding constants
        num_dram_banks = mesh_device.dram_grid_size().x
        tile_size = 32
        dim = hf_config.hidden_size

        # Compute kernel config for all matmuls
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # =====================================================================
        # qkv_a (wq_kv_a): m=32, k=896, n=2112 (pads to 2304)
        # in0_core_grid=(7,1), out_core_grid=(8,1), WIDTH sharding
        # =====================================================================
        qkv_a_n = q_lora_rank + kv_lora_rank + qk_rope_head_dim  # 2112
        qkv_a_n_padded = pad_n_to_dram_banks(qkv_a_n, tile_size=32, num_dram_banks=num_dram_banks)  # 2304
        qkv_a_in0_core_grid = ttnn.CoreGrid(y=1, x=7)
        qkv_a_out_core_grid = ttnn.CoreGrid(y=1, x=8)

        # Program config for qkv_a
        qkv_a_num_in0_cores = qkv_a_in0_core_grid.x * qkv_a_in0_core_grid.y
        qkv_a_num_out_cores = qkv_a_out_core_grid.x * qkv_a_out_core_grid.y
        qkv_a_in0_block_w = even_int_div(hidden_size_per_device, qkv_a_num_in0_cores * tile_size)  # 896 // 7 // 32 = 4
        qkv_a_per_core_M = math.ceil(USERS_PER_ROW / tile_size)  # ceil(32 / 32) = 1
        qkv_a_per_core_N = even_int_div(qkv_a_n_padded, qkv_a_num_out_cores * tile_size)  # 2304 // 8 // 32 = 9

        qkv_a_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=qkv_a_in0_block_w,
            per_core_M=qkv_a_per_core_M,
            per_core_N=qkv_a_per_core_N,
            fused_activation=None,
        )

        # in0 L1 WIDTH sharded memory config for qkv_a
        wq_kv_a_in0_memory_config = {
            "core_grid": qkv_a_in0_core_grid,
            "strategy": ttnn.ShardStrategy.WIDTH,
            "orientation": ttnn.ShardOrientation.ROW_MAJOR,
        }

        # Output L1 WIDTH sharded memory config for qkv_a (using padded n)
        qkv_a_out_memory_config = ttnn.create_sharded_memory_config(
            [1, 1, ttnn.core.roundup(USERS_PER_ROW, tile_size), qkv_a_n_padded],
            core_grid=qkv_a_out_core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

        wq_kv_a_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=qkv_a_out_memory_config,
            compute_kernel_config=compute_kernel_config,
            program_config=qkv_a_program_config,
        )

        # =====================================================================
        # wq_b: m=32, k=1536, n=3072 (already aligned)
        # in0_core_grid=(8,2), out_core_grid=(8,2), WIDTH sharding
        # =====================================================================
        wq_b_n = num_heads_local * qk_head_dim  # 16 * 192 = 3072
        wq_b_n_padded = pad_n_to_dram_banks(
            wq_b_n, tile_size=32, num_dram_banks=num_dram_banks
        )  # 3072 (already aligned)
        wq_b_in0_core_grid = ttnn.CoreGrid(y=2, x=8)
        wq_b_out_core_grid = ttnn.CoreGrid(y=2, x=8)

        # Program config for wq_b
        wq_b_num_in0_cores = wq_b_in0_core_grid.x * wq_b_in0_core_grid.y
        wq_b_num_out_cores = wq_b_out_core_grid.x * wq_b_out_core_grid.y
        wq_b_in0_block_w = even_int_div(q_lora_rank, wq_b_num_in0_cores * tile_size)  # 1536 // 16 // 32 = 3
        wq_b_per_core_M = math.ceil(USERS_PER_ROW / tile_size)  # ceil(32 / 32) = 1
        wq_b_per_core_N = even_int_div(wq_b_n_padded, wq_b_num_out_cores * tile_size)  # 3072 // 16 // 32 = 6

        wq_b_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=wq_b_in0_block_w,
            per_core_M=wq_b_per_core_M,
            per_core_N=wq_b_per_core_N,
            fused_activation=None,
        )

        # in0 L1 WIDTH sharded memory config for wq_b
        wq_b_in0_memory_config = {
            "core_grid": wq_b_in0_core_grid,
            "strategy": ttnn.ShardStrategy.WIDTH,
            "orientation": ttnn.ShardOrientation.ROW_MAJOR,
        }

        # Output L1 WIDTH sharded memory config for wq_b (using padded n)
        wq_b_out_memory_config = ttnn.create_sharded_memory_config(
            [1, 1, ttnn.core.roundup(USERS_PER_ROW, tile_size), wq_b_n_padded],
            core_grid=wq_b_out_core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

        wq_b_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=wq_b_out_memory_config,
            compute_kernel_config=compute_kernel_config,
            program_config=wq_b_program_config,
        )

        # =====================================================================
        # wkv_b1: batch=16 (pads to 24), m=32, k=128, n=512
        # core_grid=(3,4), HEIGHT (batch) sharding
        # =====================================================================
        wkv_b1_batch = num_heads_local  # 16
        wkv_b1_batch_padded = pad_batch_to_dram_banks(wkv_b1_batch)  # 24
        wkv_b1_m = ttnn.core.roundup(USERS_PER_ROW, tile_size)  # roundup(32, 32) = 32
        wkv_b1_k = qk_nope_head_dim  # 128
        wkv_b1_n = kv_lora_rank  # 512

        # Program config for wkv_b1 (batched DRAM sharded)
        wkv_b1_in0_block_w = even_int_div(wkv_b1_k, tile_size)  # 128 // 32 = 4
        wkv_b1_per_core_M = even_int_div(wkv_b1_m, tile_size)  # 32 // 32 = 1
        wkv_b1_per_core_N = even_int_div(wkv_b1_n, tile_size)  # 512 // 32 = 16

        wkv_b1_program_config = ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
            in0_block_w=wkv_b1_in0_block_w,
            per_core_M=wkv_b1_per_core_M,
            per_core_N=wkv_b1_per_core_N,
            fused_activation=None,
        )

        # Output L1 HEIGHT sharded memory config for wkv_b1
        # Get optimal DRAM bank-to-worker core assignment
        optimal_worker_cores = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        wkv_b1_batches_per_core = wkv_b1_batch_padded // num_dram_banks  # 24 // 12 = 2
        wkv_b1_out_shard_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in optimal_worker_cores]
        )
        wkv_b1_out_shard_shape = [wkv_b1_batches_per_core * wkv_b1_m, wkv_b1_n]  # [2 * 32, 512] = [64, 512]
        wkv_b1_out_shard_spec = ttnn.ShardSpec(
            wkv_b1_out_shard_grid, wkv_b1_out_shard_shape, ttnn.ShardOrientation.ROW_MAJOR
        )
        wkv_b1_out_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, wkv_b1_out_shard_spec
        )

        # Input L1 HEIGHT sharded memory config for wkv_b1 in0 (activations)
        # Place activations on optimal DRAM bank-to-worker cores for batched matmul
        wkv_b1_in0_shard_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in optimal_worker_cores]
        )
        wkv_b1_in0_shard_shape = [wkv_b1_batches_per_core * wkv_b1_m, wkv_b1_k]  # [2 * 32, 128] = [64, 128]
        wkv_b1_in0_shard_spec = ttnn.ShardSpec(
            wkv_b1_in0_shard_grid, wkv_b1_in0_shard_shape, ttnn.ShardOrientation.ROW_MAJOR
        )
        wkv_b1_in0_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, wkv_b1_in0_shard_spec
        )

        wkv_b1_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=wkv_b1_out_memory_config,
            compute_kernel_config=compute_kernel_config,
            program_config=wkv_b1_program_config,
        )

        # =====================================================================
        # wkv_b2: batch=128 (pads to 132), m=4, k=512, n=128, tile_h=4
        # core_grid=(3,4), HEIGHT (batch) sharding
        # =====================================================================
        wkv_b2_batch = num_heads  # 128
        wkv_b2_batch_padded = pad_batch_to_dram_banks(wkv_b2_batch)  # 132

        # wkv_b2 is meant to run with tiny tiles as K=4
        # However, this is pending functionality for the preceding transpose to output tensors in a tiny tile shape
        # For now, this is kept as a full tile instead
        wkv_b2_m = 32  # m dimension
        wkv_b2_k = kv_lora_rank  # 512
        wkv_b2_n = v_head_dim  # 128
        wkv_b2_tile_h = 32

        # Program config for wkv_b2 (batched DRAM sharded)
        wkv_b2_in0_block_w = even_int_div(wkv_b2_k, tile_size)  # 512 // 32 = 16
        wkv_b2_per_core_M = even_int_div(wkv_b2_m, wkv_b2_tile_h)  # 32 // 32 = 1
        wkv_b2_per_core_N = even_int_div(wkv_b2_n, tile_size)  # 128 // 32 = 4

        wkv_b2_program_config = ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
            in0_block_w=wkv_b2_in0_block_w,
            per_core_M=wkv_b2_per_core_M,
            per_core_N=wkv_b2_per_core_N,
            fused_activation=None,
        )

        # Output L1 HEIGHT sharded memory config for wkv_b2
        # Reuse optimal_worker_cores from wkv_b1
        wkv_b2_batches_per_core = wkv_b2_batch_padded // num_dram_banks  # 132 // 12 = 11
        wkv_b2_out_shard_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in optimal_worker_cores]
        )
        wkv_b2_out_shard_shape = [wkv_b2_batches_per_core * wkv_b2_m, wkv_b2_n]  # [11 * 32, 128] = [352, 128]
        wkv_b2_out_shard_spec = ttnn.ShardSpec(
            wkv_b2_out_shard_grid, wkv_b2_out_shard_shape, ttnn.ShardOrientation.ROW_MAJOR
        )
        wkv_b2_out_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, wkv_b2_out_shard_spec
        )

        wkv_b2_output_tile = ttnn.Tile((wkv_b2_tile_h, tile_size))

        # Input L1 HEIGHT sharded memory config for wkv_b2 in0 (activations)
        # Place activations on optimal DRAM bank-to-worker cores for batched matmul
        wkv_b2_in0_shard_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in optimal_worker_cores]
        )
        wkv_b2_in0_shard_shape = [wkv_b2_batches_per_core * wkv_b2_m, wkv_b2_k]  # [11 * 4, 512] = [44, 512]
        wkv_b2_in0_shard_spec = ttnn.ShardSpec(
            wkv_b2_in0_shard_grid, wkv_b2_in0_shard_shape, ttnn.ShardOrientation.ROW_MAJOR
        )
        wkv_b2_in0_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, wkv_b2_in0_shard_spec
        )

        wkv_b2_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=wkv_b2_out_memory_config,
            compute_kernel_config=compute_kernel_config,
            program_config=wkv_b2_program_config,
            output_tile=wkv_b2_output_tile,
        )

        # =====================================================================
        # wo: m=32, k=16384, n=896 (pads to 1152)
        # in0_core_grid=(8,1), out_core_grid=(6,1), WIDTH sharding
        # =====================================================================
        wo_k = num_heads * v_head_dim  # 16384
        wo_n = dim // mesh_device.shape[1]  # 896
        wo_n_padded = pad_n_to_dram_banks(wo_n, tile_size=32, num_dram_banks=num_dram_banks)  # 1152
        wo_in0_core_grid = ttnn.CoreGrid(y=2, x=8)
        wo_out_core_grid = ttnn.CoreGrid(y=2, x=6)

        # Program config for wo
        wo_num_in0_cores = wo_in0_core_grid.x * wo_in0_core_grid.y
        wo_num_out_cores = wo_out_core_grid.x * wo_out_core_grid.y
        wo_in0_block_w = even_int_div(wo_k, wo_num_in0_cores * tile_size)  # 16384 // 16 // 32 = 32
        wo_per_core_M = math.ceil(USERS_PER_ROW / tile_size)  # ceil(32 / 32) = 1
        wo_per_core_N = even_int_div(wo_n_padded, wo_num_out_cores * tile_size)  # 1152 // 12 // 32 = 3

        wo_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=even_int_div(wo_in0_block_w, 2),
            per_core_M=wo_per_core_M,
            per_core_N=wo_per_core_N,
            fused_activation=None,
        )

        # in0 L1 WIDTH sharded memory config for wo
        wo_in0_memory_config = {
            "core_grid": wo_in0_core_grid,
            "strategy": ttnn.ShardStrategy.WIDTH,
            "orientation": ttnn.ShardOrientation.ROW_MAJOR,
        }

        # Output L1 WIDTH sharded memory config for wo (using padded n)
        wo_out_memory_config = ttnn.create_sharded_memory_config(
            [1, 1, ttnn.core.roundup(USERS_PER_ROW, tile_size), wo_n_padded],
            core_grid=wo_out_core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

        wo_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=wo_out_memory_config,
            compute_kernel_config=compute_kernel_config,
            program_config=wo_program_config,
        )

        # Resharding for q_rope
        q_rope_shape = (1, USERS_PER_ROW, num_heads_local, qk_rope_head_dim)
        q_rope_shard_height = nearest_y(q_rope_shape[2], ttnn.TILE_SIZE)
        q_rope_shard_width = q_rope_shape[3]
        q_rope_num_cores = q_rope_shape[1]
        q_rope_core_grid = ttnn.num_cores_to_corerangeset(q_rope_num_cores, grid_size, row_wise=True)
        q_rope_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(q_rope_shard_height, q_rope_shard_width),
            core_grid=q_rope_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        q_rope_permute_config = PermuteConfig(
            dims=(1, 0, 2, 3),
        )
        q_rope_slice_config = SliceConfig(
            memory_config=q_rope_mem_cfg,
        )
        q_concat_config = ConcatConfig(
            dim=-1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        q_rope_out_reshard_config = ReshardConfig(
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Resharding for kv_rope
        kv_rope_shape = (1, USERS_PER_ROW, 1, qk_rope_head_dim)
        kv_rope_shard_height = nearest_y(kv_rope_shape[2], ttnn.TILE_SIZE)
        kv_rope_shard_width = kv_rope_shape[3]
        kv_rope_num_cores = kv_rope_shape[1]
        kv_rope_core_grid = ttnn.num_cores_to_corerangeset(kv_rope_num_cores, grid_size, row_wise=True)
        kv_rope_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(kv_rope_shard_height, kv_rope_shard_width),
            core_grid=kv_rope_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        kv_rope_reshard_config = ReshardConfig(
            memory_config=kv_rope_mem_cfg,
        )
        kv_rope_permute_config = PermuteConfig(
            dims=(0, 2, 1, 3),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        kv_rope_out_reshard_config = ReshardConfig(
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # KV concat
        kv_concat_config = ConcatConfig(
            dim=-1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Resharding for kvpe
        kvpe_shape = (1, even_int_div(USERS_PER_ROW, mesh_shape[1]), 1, kv_lora_rank + qk_rope_head_dim)
        kvpe_shard_height = nearest_y(kvpe_shape[2], ttnn.TILE_SIZE)
        kvpe_shard_width = kvpe_shape[3]
        kvpe_num_cores = kvpe_shape[1]
        kvpe_core_grid = ttnn.num_cores_to_corerangeset(kvpe_num_cores, grid_size, row_wise=True)
        kvpe_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(kvpe_shard_height, kvpe_shard_width),
            core_grid=kvpe_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        kvpe_reshard_config = ReshardConfig(
            memory_config=kvpe_mem_cfg,
        )

        # FlashMLA
        q_chunk_size = 0  # Unused in decode mode
        k_chunk_size = 128  # TODO: Make dynamic?

        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        flash_mla_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        q_num_cores = num_cores
        q_num_cores = min(even_int_div(USERS_PER_ROW, mesh_shape[1]) * num_heads, q_num_cores)
        block_height = nearest_y(
            (even_int_div(USERS_PER_ROW, mesh_shape[1]) * num_heads) // q_num_cores, ttnn.TILE_SIZE
        )
        block_width = kv_lora_rank + qk_rope_head_dim

        q_core_grid = ttnn.num_cores_to_corerangeset(q_num_cores, grid_size, row_wise=True)
        q_mem_config = ttnn.create_sharded_memory_config(
            shape=(block_height, block_width),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        flash_mla_out_mem_config = ttnn.create_sharded_memory_config(
            shape=(block_height, kv_lora_rank),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        scale = qk_head_dim**-0.5
        if rope_factor > 1.0:
            mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
            scale = scale * mscale * mscale

        flash_mla_reshard_config = ReshardConfig(
            memory_config=q_mem_config,
        )
        flash_mla_config = {
            "head_dim_v": kv_lora_rank,
            "scale": scale,
            "program_config": sdpa_program_config,
            "compute_kernel_config": flash_mla_compute_kernel_config,
            "memory_config": flash_mla_out_mem_config,
        }
        flash_mla_out_reshard_config = ReshardConfig(
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Norms
        q_norm_config = RMSNorm.decode_model_config(
            hf_config,
            mesh_device,
        )
        kv_norm_config = RMSNorm.decode_model_config(
            hf_config,
            mesh_device,
        )

        # Set up CCLs

        # Fused wq_kv_a: AG + local reduce
        wq_kv_a_ag_config = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            num_workers_per_link=1,
        )
        wq_kv_a_r_config = {
            "dims": [1],
            "output": None,
            "compute_kernel_config": ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        }

        # Q all-to-all
        wq_a2a_config = AllToAllAsyncGenericConfig(
            cluster_axis=1,
            in_dim=2,
            out_dim=1,
        )

        wq_a2a_reshard_out_mem_config = ttnn.create_sharded_memory_config(
            shape=(USERS_PER_ROW, num_heads, kv_lora_rank + qk_rope_head_dim),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
        )
        wq_a2a_reshard_config = ReshardConfig(
            memory_config=wq_a2a_reshard_out_mem_config,
        )  # 1,4,128,576, height sharded 8x8 [32,576]

        # Slice configs for fused wq_kv_a output: [q_lora_rank | kv_lora_rank | qk_rope_head_dim]
        # Q and KV nope use non-overlapping core grids so they can run parallel norms
        # via composite.launch (which requires non-overlapping core ranges).
        num_q_cores = 16
        num_kv_nope_cores = 16
        shard_height = ttnn.core.roundup(USERS_PER_ROW, ttnn.TILE_SIZE)

        # Q slice: WIDTH sharded on 4x4 grid (0,0)-(3,3) -> 16 cores, shard [32, 96]
        q_slice_mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
                [shard_height, q_lora_rank // num_q_cores],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        q_slice_config = SliceConfig(
            memory_config=q_slice_mem_config,
        )
        # KV nope slice: WIDTH sharded on 2x8 grid (5,0)-(6,7) -> 16 cores, shard [32, 32]
        kv_nope_slice_mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 7))}),
                [shard_height, kv_lora_rank // num_kv_nope_cores],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        kv_nope_slice_config = SliceConfig(
            memory_config=kv_nope_slice_mem_config,
        )
        # KV rope slice: interleaved since it goes through permute/reshard anyway
        kv_rope_slice_config = SliceConfig(
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        flash_mla_a2a_config = AllToAllAsyncGenericConfig(
            cluster_axis=1,
            in_dim=1,
            out_dim=2,
        )

        # WO
        wo_ag_config = AllBroadcastAsyncConfig(
            cluster_axis=1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return {
            "num_heads": num_heads,
            "q_lora_rank": q_lora_rank,
            "kv_lora_rank": kv_lora_rank,
            "qk_nope_head_dim": qk_nope_head_dim,
            "qk_rope_head_dim": qk_rope_head_dim,
            "qk_head_dim": qk_head_dim,
            "v_head_dim": v_head_dim,
            "input_memory_config": input_memory_config,
            "wq_kv_a": wq_kv_a_config,
            "wq_b": wq_b_config,
            "wkv_b1": wkv_b1_config,
            "wkv_b2": wkv_b2_config,
            "wo": wo_config,
            "q_rope_permute": q_rope_permute_config,
            "q_rope_slice": q_rope_slice_config,
            "q_rope_out_reshard": q_rope_out_reshard_config,
            "q_concat": q_concat_config,
            "kv_rope_reshard": kv_rope_reshard_config,
            "kv_rope_permute": kv_rope_permute_config,
            "kv_rope_out_reshard": kv_rope_out_reshard_config,
            "kv_concat": kv_concat_config,
            "kvpe_reshard": kvpe_reshard_config,
            "flash_mla_reshard": flash_mla_reshard_config,
            "flash_mla": flash_mla_config,
            "flash_mla_out_reshard": flash_mla_out_reshard_config,
            "q_norm": q_norm_config,
            "kv_norm": kv_norm_config,
            "wq_kv_a_ag_decode": wq_kv_a_ag_config,
            "wq_kv_a_r_decode": wq_kv_a_r_config,
            "q_slice_decode": q_slice_config,
            "kv_nope_slice_decode": kv_nope_slice_config,
            "kv_rope_slice_decode": kv_rope_slice_config,
            "wq_a2a_decode": wq_a2a_config,
            "wq_a2a_reshard_decode": wq_a2a_reshard_config,
            "flash_mla_a2a_decode": flash_mla_a2a_config,
            "wo_ag_decode": wo_ag_config,
            "mesh_device": mesh_device,
            "wq_kv_a_in0_memory_config": wq_kv_a_in0_memory_config,
            "wq_b_in0_memory_config": wq_b_in0_memory_config,
            "wo_in0_memory_config": wo_in0_memory_config,
            "wkv_b1_in0_memory_config": wkv_b1_in0_memory_config,
            "wkv_b2_in0_memory_config": wkv_b2_in0_memory_config,
            "wkv_b2_in0_tile": wkv_b2_output_tile,  # 4x32 tiny tile for wkv_b2 in0
        }

    @classmethod
    def get_valid_paged_config(
        cls, max_seq_len: int, batch_size_per_row: int, dp_factor: int, block_size: int = ttnn.TILE_SIZE
    ) -> PagedAttentionConfig:
        """Get a valid paged attention configuration for MLA.

        This function also calculates max_num_blocks such that each user will have max_seq_len available.
        For DP, the max_num_blocks is divided by the batch size of the DP shard, not the total batch size.

        Args:
            max_seq_len: Maximum sequence length
            batch_size_per_row: Batch size per row of the model
            block_size: Block size for paged attention (default is TILE_SIZE)

        Returns:
            A PagedAttentionConfig object with valid parameters
        """
        assert max_seq_len % block_size == 0, f"max_seq_len {max_seq_len} must be divisible by block_size {block_size}."
        assert (
            block_size % ttnn.TILE_SIZE == 0
        ), f"block_size {block_size} must be a multiple of TILE_SIZE {ttnn.TILE_SIZE}."

        batch_per_shard = even_int_div(batch_size_per_row, dp_factor)
        max_num_blocks = even_int_div(
            max_seq_len * batch_per_shard, block_size
        )  # Such that each user will have max_seq_len available

        return PagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )

    @classmethod
    def create_page_table(
        cls,
        paged_config: PagedAttentionConfig,
        mesh_device: ttnn.MeshDevice,
        page_table: torch.Tensor | None = None,
        batch_size: int = USERS_PER_ROW,
    ) -> ttnn.Tensor:
        """Helper function to allocate the page table for MLA on device.

        When doing DP, this function replicates the page table across DP shards.
        Assumptions:
            - If user X on DP shard 1 is on position N, with page id P,
                and if user X on DP shard 2 is also on position N, it will also be on page id P.
                As such, the max_num_blocks is only cut by the batch size of the DP shard, not the total batch size.

        Args:
            page_table: A torch tensor version of the page table
            paged_config: PagedAttentionConfig containing page table configuration
            mesh_device: TTNN mesh device

        Returns:
            Device-allocated version of the page table representing the page table
        """  # TODO: update docs
        if page_table is None:
            max_num_blocks = paged_config.max_num_blocks
            _, dp_factor = mesh_device.shape
            batch_per_shard = even_int_div(batch_size, dp_factor)

            page_table = torch.randperm(max_num_blocks, dtype=torch.int32)  # Randperm not necessary, but more rigorous
            page_table = page_table.reshape(batch_per_shard, even_int_div(max_num_blocks, batch_per_shard))
        assert page_table.numel() == paged_config.max_num_blocks

        return ttnn.from_torch(
            page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        paged_config: PagedAttentionConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL,
        caches: Sequence[torch.Tensor] | None = None,
        kv_cache_override: KvCacheConfig | None = None,
    ) -> ModelState:
        if kv_cache_override is None:
            kvpe_dim = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim
            cache_shape = (paged_config.max_num_blocks, 1, paged_config.block_size, kvpe_dim)
        else:
            kv_cache_shape = kv_cache_override.kv_cache_shape
            cache_shape = (
                kv_cache_shape[0],
                kv_cache_shape[1],
                kv_cache_shape[2],
                kv_cache_shape[3],
            )

        assert caches is None or len(caches) == mesh_device.shape[0]
        # Store CCL object for runtime semaphore initialization
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "mesh_shape": mesh_device.shape,
            "kvpe_cache": cls._convert_cache(caches, cache_shape, mesh_device),
            "ccl": ccl,
        }

    @classmethod
    def _convert_cache(
        cls,
        caches: tuple[torch.Tensor, ...] | None,
        cache_shape: tuple[int, ...],
        mesh_device: ttnn.MeshDevice,
    ) -> ttnn.Tensor:
        if caches is None:
            # ttnn.zeros doesn't accept a mesh_mapper, so we need to pass correct shape per device. It replicates the tensor across the devices.
            return ttnn.zeros(
                shape=cache_shape,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            return ttnn.as_tensor(
                torch.concatenate(tuple(caches)),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, 0),
            )

    @classmethod
    def forward_decode(
        cls,
        x: ttnn.Tensor,
        position_idxs: ttnn.Tensor,
        row_idx: int | None,
        cfg: RunDecodeConfig,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Forward pass of MLA in decode mode.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            cfg: RunConfig containing weights and op configurations
            position_idxs: List of position indices for the current batch
            rope_tensors: Dictionary containing RoPE tensors
            page_table: Page table tensor for paged attention
        Returns:
            Output tensor after MLA computation

        """
        _, mla_tp_factor = mesh_shape = cfg["mesh_shape"]

        num_heads = cfg["num_heads"]
        num_heads_local = even_int_div(num_heads, mla_tp_factor)
        q_lora_rank = cfg["q_lora_rank"]
        kv_lora_rank = cfg["kv_lora_rank"]
        qk_nope_head_dim = cfg["qk_nope_head_dim"]
        qk_rope_head_dim = cfg["qk_rope_head_dim"]
        qk_head_dim = cfg["qk_head_dim"]
        v_head_dim = cfg["v_head_dim"]

        kvpe_cache = cfg["kvpe_cache"]
        ccl = cfg["ccl"]

        bsz = x.shape[2]
        scale = 1.0 / mla_tp_factor

        # Fused Linear + AR: wq_kv_a (wq_a + wkv_a)

        tt_q, tt_kv_nope, tt_kv_rope = cls._fwd_decode_wq_kv_a(
            x,
            cfg,
            ccl,
            bsz,
            q_lora_rank,
            kv_lora_rank,
            qk_rope_head_dim,
        )

        # Norm and Rope

        tt_q, tt_kvpe = cls._fwd_decode_norm_and_rope(tt_q, tt_kv_nope, tt_kv_rope, cfg, rope_tensors)

        # Paged Update Cache

        cls._fwd_decode_paged_update_cache(kvpe_cache, tt_kvpe, position_idxs, page_table, mesh_shape, row_idx)

        # Q Rope + Nope

        tt_q = cls._fwd_decode_q_rope_nope(
            tt_q,
            cfg,
            rope_tensors,
            bsz,
            num_heads_local,
            qk_head_dim,
            qk_nope_head_dim,
        )

        # All To All before FlashMLA

        tt_q = cls._fwd_decode_all_to_all_pre_flash_mla(tt_q, cfg)

        # Flash MLA

        attn_out = cls._fwd_decode_flash_mla(tt_q, kvpe_cache, page_table, position_idxs, cfg)

        # Wkv_b2

        v_out = cls._fwd_decode_wkv_b2(attn_out, cfg)

        # AG + Reshape

        v_out = cls._fwd_decode_ag_reshape(v_out, cfg, ccl, bsz, num_heads, v_head_dim)

        # WO

        out = cls._fwd_decode_wo(v_out, cfg)

        return out

    @staticmethod
    def _debug_prefill_page_table(
        page_table: ttnn.Tensor,
        local_batch_idx: int,
        global_batch_idx: int,
        row_idx: int | None,
        col_idx: int,
        block_size: int,
        max_blocks: int,
        mesh_device: ttnn.MeshDevice | None,
    ) -> None:
        """Log page table row and bounds for prefill KV cache writes."""
        if not _deepseek_kvdbg_enabled():
            return
        try:
            if page_table is None:
                logger.warning(
                    "KVDBG deepseek prefill: page_table=None global={} local={} row={} col={}",
                    global_batch_idx,
                    local_batch_idx,
                    row_idx,
                    col_idx,
                )
                return
            if isinstance(page_table, ttnn.Tensor):
                if mesh_device is None:
                    logger.warning(
                        "KVDBG deepseek prefill: mesh_device=None global={} local={} row={} col={} page_table_shape={}",
                        global_batch_idx,
                        local_batch_idx,
                        row_idx,
                        col_idx,
                        list(page_table.shape),
                    )
                    return
                mesh_composer = ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=0)
                pt_torch = ttnn.to_torch(page_table, mesh_composer=mesh_composer)
            else:
                pt_torch = page_table
            if isinstance(pt_torch, torch.Tensor):
                pt_cpu = pt_torch.detach().cpu()
                if pt_cpu.ndim >= 2 and local_batch_idx < pt_cpu.shape[0]:
                    row = pt_cpu[local_batch_idx]
                    flat = row.flatten()
                    min_v = int(flat.min().item()) if flat.numel() else None
                    max_v = int(flat.max().item()) if flat.numel() else None
                    if max_blocks is not None and flat.numel():
                        oob_mask = (flat < 0) | (flat >= max_blocks)
                        oob_count = int(oob_mask.sum().item())
                    else:
                        oob_mask = None
                        oob_count = 0
                    if oob_mask is not None and oob_count > 0:
                        idxs = oob_mask.nonzero(as_tuple=False).flatten().tolist()[:16]
                        vals = flat[oob_mask][:16].tolist()
                        logger.error(
                            "KVDBG deepseek prefill OOB user global={} local={} row={} col={} row_shape={} "
                            "block_size={} max_blocks={} min={} max={} oob={} sample_idx_val={}",
                            global_batch_idx,
                            local_batch_idx,
                            row_idx,
                            col_idx,
                            list(row.shape),
                            block_size,
                            max_blocks,
                            min_v,
                            max_v,
                            oob_count,
                            list(zip(idxs, vals)),
                        )
                else:
                    logger.warning(
                        "KVDBG deepseek prefill: row index out of range global={} local={} row={} col={} table_shape={}",
                        global_batch_idx,
                        local_batch_idx,
                        row_idx,
                        col_idx,
                        list(pt_cpu.shape),
                    )
            else:
                logger.warning(
                    "KVDBG deepseek prefill: unexpected page_table type global={} local={} row={} col={} page_table={}",
                    global_batch_idx,
                    local_batch_idx,
                    row_idx,
                    col_idx,
                    pt_torch,
                )
        except Exception as exc:
            logger.warning("KVDBG deepseek prefill failed: {}", exc)

    @classmethod
    def forward_prefill(
        cls,
        x: ttnn.Tensor,
        batch_idx: int,
        row_idx: int | None,
        cfg: RunPrefillConfig,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Forward pass of MLA in prefill mode.

        Args:
            x: Input tensor
            cfg: RunConfig containing weights and op configurations
            batch_idx: Batch index for cache updates (wrt to global batch size)
            rope_tensors: Dictionary containing RoPE tensors
            page_table: Page table tensor for paged attention
            row_idx: Row index in the mesh

        Returns:
            Output tensor after MLP computation
        """
        mesh_shape = cfg["mesh_shape"]

        sdpa_dp_factor = mla_tp_factor = mesh_shape[1]

        num_heads = cfg["num_heads"]
        num_heads_local = even_int_div(num_heads, mla_tp_factor)
        q_lora_rank = cfg["q_lora_rank"]
        kv_lora_rank = cfg["kv_lora_rank"]
        qk_nope_head_dim = cfg["qk_nope_head_dim"]
        qk_rope_head_dim = cfg["qk_rope_head_dim"]
        qk_head_dim = cfg["qk_head_dim"]
        v_head_dim = cfg["v_head_dim"]

        kvpe_cache = cfg["kvpe_cache"]
        ccl = cfg["ccl"]

        seq_len = x.shape[2]

        # Fused Linear + AR: wq_kv_a (wq_a + wkv_a)

        tt_q, tt_kv_nope, tt_kv_rope = cls._fwd_prefill_wq_kv_a(
            x,
            cfg,
            ccl,
            seq_len,
            q_lora_rank,
            kv_lora_rank,
            qk_rope_head_dim,
        )

        # Q path: norm + wq_b (interleaved in0 + DRAM WIDTH sharded in1)
        tt_q = RMSNorm.forward_prefill(tt_q, cfg["q_norm"])
        wq_b_program_config = build_prefill_matmul_program_config(
            seq_len, k=q_lora_rank, n=num_heads_local * qk_head_dim
        )
        tt_q = ttnn.linear(tt_q, **cfg["wq_b"], program_config=wq_b_program_config)

        tt_q = ttnn.reshape(tt_q, (1, seq_len, num_heads_local, qk_head_dim))
        tt_q = ttnn.permute(tt_q, (0, 2, 1, 3))  # [1, num_heads_local, seq_len, qk_head_dim]

        tt_q_nope = ttnn.slice(tt_q, [0, 0, 0, 0], [1, num_heads_local, seq_len, qk_nope_head_dim])
        tt_q_rope = ttnn.slice(tt_q, [0, 0, 0, qk_nope_head_dim], [1, num_heads_local, seq_len, qk_head_dim])
        ttnn.deallocate(tt_q)

        # wkv_b1 (interleaved in0 + DRAM HEIGHT sharded in1)
        # Pad batch dimension to match DRAM sharded weight layout
        num_heads_local_padded = pad_batch_to_dram_banks(num_heads_local)

        wkv_b1_program_config = build_prefill_matmul_program_config(
            seq_len, k=qk_nope_head_dim, n=kv_lora_rank, batch=num_heads_local_padded
        )
        tt_q_nope = ttnn.linear(
            tt_q_nope, **cfg["wkv_b1"], program_config=wkv_b1_program_config
        )  # [1, num_heads_local_padded, seq_len, kv_lora_rank]

        # Q RoPE
        tt_q_rope = ttnn.experimental.rotary_embedding_llama(
            tt_q_rope,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=False,
        )

        # Q ready for FlashMLA
        tt_q = ttnn.concat([tt_q_nope, tt_q_rope], dim=-1)

        # KV Norm
        tt_kv_nope = RMSNorm.forward_prefill(tt_kv_nope, cfg["kv_norm"])

        # KV RoPE
        tt_kv_rope = ttnn.experimental.rotary_embedding_llama(
            tt_kv_rope,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=False,
        )

        tt_kvpe = ttnn.concat([tt_kv_nope, tt_kv_rope], dim=-1)

        ttnn.deallocate(tt_kv_nope)
        ttnn.deallocate(tt_kv_rope)

        tt_kvpe_fp16 = tt_kvpe
        tt_kvpe = ttnn.typecast(tt_kvpe_fp16, dtype=kvpe_cache.dtype)
        ttnn.deallocate(tt_kvpe_fp16)

        # Update KVPE Cache
        batch_size_per_dp_shard = even_int_div(USERS_PER_ROW, sdpa_dp_factor)
        local_batch_idx = batch_idx % batch_size_per_dp_shard  # Local batch index within the DP shard
        col_idx = batch_idx // batch_size_per_dp_shard  # Which DP shard the batch belongs to
        if _deepseek_kvdbg_enabled():
            cls._debug_prefill_page_table(
                page_table=page_table,
                local_batch_idx=local_batch_idx,
                global_batch_idx=batch_idx,
                row_idx=row_idx,
                col_idx=col_idx,
                block_size=kvpe_cache.shape[2],
                max_blocks=kvpe_cache.shape[0],
                mesh_device=cfg.get("mesh_device", None),
            )

        ttnn.experimental.paged_fill_cache(
            kvpe_cache,
            tt_kvpe,
            page_table=page_table,
            batch_idx=local_batch_idx,
            mesh_coords=set(get_mesh_coords(mesh_shape, row_idx, col_idx)),
        )

        # FlashMLA
        attn_out = ttnn.transformer.flash_mla_prefill(
            tt_q,
            tt_kvpe,
            **cfg["flash_mla"],
        )  # [1, num_heads_local, seq_len, kv_lora_rank]
        ttnn.deallocate(tt_q)
        ttnn.deallocate(tt_kvpe)

        # DP wkv_b2 to match decode weights
        v_out = ttnn.experimental.all_gather_async(
            attn_out, **ccl.populate_all_gather_runtime_args(cfg["wkv_b2_ag_prefill"])
        )  # [1, num_heads, seq_len, v_head_dim] # wkv_b2_ag_prefill

        # wkv_b2 (interleaved in0 + DRAM HEIGHT sharded in1)
        # Pad batch dimension to match DRAM sharded weight layout
        num_heads_padded = pad_batch_to_dram_banks(num_heads)

        wkv_b2_program_config = build_prefill_matmul_program_config(
            seq_len, k=kv_lora_rank, n=v_head_dim, batch=num_heads_padded
        )
        v_out = ttnn.linear(
            v_out, **cfg["wkv_b2"], program_config=wkv_b2_program_config
        )  # [1, num_heads_padded, seq_len, v_head_dim]
        ttnn.deallocate(attn_out)

        # Slice off batch padding from wkv_b2 output

        # Permute BEFORE all_gather to avoid large tensor permute at 32K+ seq_len
        v_out = ttnn.permute(v_out, (0, 2, 1, 3))  # [1, seq_len, num_heads_local, v_head_dim]

        # wo matmul (interleaved in0 + DRAM WIDTH sharded in1)
        # Chunk the sequence dimension if needed to avoid OOM/hang in all_gather for large sequences
        # Strategy: Process each chunk independently to keep all_gather buffers small
        wo_k = num_heads * v_head_dim
        dim = x.shape[3]

        SEQ_LEN_CHUNK_SIZE = 8192
        if seq_len > SEQ_LEN_CHUNK_SIZE:
            num_heads = v_out.shape[2]
            v_head_dim = v_out.shape[3]
            # Use ceiling division instead of even_int_div to handle non-multiples of 8192
            num_chunks = (seq_len + SEQ_LEN_CHUNK_SIZE - 1) // SEQ_LEN_CHUNK_SIZE

            # Pad seq_len to be a multiple of SEQ_LEN_CHUNK_SIZE if needed
            padded_seq_len = num_chunks * SEQ_LEN_CHUNK_SIZE
            if seq_len != padded_seq_len:
                # Pad the sequence dimension (dim=1)
                v_out = ttnn.pad(v_out, padding=((0, 0), (0, padded_seq_len - seq_len), (0, 0), (0, 0)), value=0.0)

            wo_chunk_program_config = build_prefill_matmul_program_config(SEQ_LEN_CHUNK_SIZE, k=wo_k, n=dim)

            output_chunks = []
            hidden_dim = num_heads * v_head_dim
            for chunk_idx in range(num_chunks):
                start = chunk_idx * SEQ_LEN_CHUNK_SIZE
                end = start + SEQ_LEN_CHUNK_SIZE
                v_chunk = ttnn.slice(v_out, (0, start, 0, 0), (1, end, num_heads, v_head_dim))
                v_chunk = ttnn.reshape(v_chunk, (1, 1, SEQ_LEN_CHUNK_SIZE, hidden_dim))
                out_chunk = ttnn.linear(
                    v_chunk, **cfg["wo"], program_config=wo_chunk_program_config
                )  # [1, 1, chunk_size, dim]
                output_chunks.append(out_chunk)
                ttnn.deallocate(v_chunk)

            ttnn.deallocate(v_out)

            out = ttnn.concat(output_chunks, dim=2)
            output_dim = out.shape[3]
            for chunk in output_chunks:
                ttnn.deallocate(chunk)

            # Trim padding if we added any
            if seq_len != padded_seq_len:
                out = ttnn.slice(out, (0, 0, 0, 0), (1, 1, seq_len, output_dim))
        else:
            # For non-chunked case: [1, seq_len, num_heads, v_head_dim] -> [1, 1, seq_len, hidden_dim]
            v_out = ttnn.reshape(v_out, (1, 1, seq_len, num_heads * v_head_dim))
            wo_program_config = build_prefill_matmul_program_config(seq_len, k=wo_k, n=dim)
            out = ttnn.linear(v_out, **cfg["wo"], program_config=wo_program_config)  # [1, 1, seq_len, dim]
            ttnn.deallocate(v_out)

        return out

    @classmethod
    def _fwd_decode_wq_kv_a(
        cls,
        x: ttnn.Tensor,
        cfg: RunDecodeConfig,
        ccl: CCL,
        bsz: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        # Shard in0 to L1 WIDTH sharded for qkv_a matmul
        in0_memory_config = ttnn.create_sharded_memory_config(
            x.shape,
            **cfg["wq_kv_a_in0_memory_config"],
        )
        x = ttnn.to_memory_config(x, memory_config=in0_memory_config)

        # Fused wq_kv_a matmul
        # 1,1,32,896, width sharded 7x4 [32,32]
        tt_q_kv = ttnn.linear(x, **cfg["wq_kv_a"])
        # 1,1,32,2112 (q_lora_rank + kv_lora_rank + qk_rope_head_dim = 1536 + 512 + 64)

        # AR using AG + local reduce (since sub-tile RS not supported for new shapes)
        tt_q_kv = ttnn.experimental.all_gather_async(
            tt_q_kv, **ccl.populate_all_gather_runtime_args(cfg["wq_kv_a_ag_decode"])
        )  # [1, num_devices, bsz, q_lora_rank + kv_lora_rank + qk_rope_head_dim]
        tt_q_kv = ttnn.experimental.fast_reduce_nc(
            tt_q_kv,
            **cfg["wq_kv_a_r_decode"],
        )  # [1, 1, bsz, q_lora_rank + kv_lora_rank + qk_rope_head_dim]

        # Slice into three parts: tt_q, tt_kv_nope, tt_kv_rope
        tt_q = ttnn.slice(tt_q_kv, [0, 0, 0, 0], [1, 1, bsz, q_lora_rank], **cfg["q_slice_decode"])
        tt_kv_nope = ttnn.slice(
            tt_q_kv, [0, 0, 0, q_lora_rank], [1, 1, bsz, q_lora_rank + kv_lora_rank], **cfg["kv_nope_slice_decode"]
        )
        tt_kv_rope = ttnn.slice(
            tt_q_kv,
            [0, 0, 0, q_lora_rank + kv_lora_rank],
            [1, 1, bsz, q_lora_rank + kv_lora_rank + qk_rope_head_dim],
            **cfg["kv_rope_slice_decode"],
        )
        ttnn.deallocate(tt_q_kv)
        return tt_q, tt_kv_nope, tt_kv_rope

    @classmethod
    def _fwd_decode_norm_and_rope(
        cls,
        tt_q: ttnn.Tensor,
        tt_kv_nope: ttnn.Tensor,
        tt_kv_rope: ttnn.Tensor,
        cfg: RunDecodeConfig,
        rope_tensors: dict,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        # Parallel Q and KV Norms
        # Q: 1,1,32,1536, width sharded 8x2 [32,96]
        # KV: 1,1,32,512 8x2 [32,32]
        q_norm_desc = descriptors.rms_norm(tt_q, program_config=RMSNorm._get_pc(tt_q.memory_config()), **cfg["q_norm"])
        kv_norm_desc = descriptors.rms_norm(
            tt_kv_nope, program_config=RMSNorm._get_pc(tt_kv_nope.memory_config()), **cfg["kv_norm"]
        )
        results = composite.launch([q_norm_desc, kv_norm_desc])
        tt_q = results[0][0]
        tt_kv_nope = results[1][0]
        # Q: 1,1,32,1536, width sharded 8x2 [32,96]
        # KV: 1,1,32,512 8x2 [32,32]

        tt_kv_nope = ttnn.to_memory_config(tt_kv_nope, memory_config=ttnn.L1_MEMORY_CONFIG)
        # 1,1,32,512 L1 interleaved

        # KV RoPE
        # 1,1,32,64 1x2 [32,32]
        # TODO: merge the following two once illia has his pr
        tt_kv_rope = ttnn.transpose(
            tt_kv_rope, 1, 2
        )  # [1, bsz, 1, qk_rope_head_dim]        # 1,32,1,64 interleaved | should be: 4x8 [32,64]
        tt_kv_rope = ttnn.to_memory_config(tt_kv_rope, **cfg["kv_rope_reshard"])
        tt_kv_rope = ttnn.experimental.rotary_embedding_llama(
            tt_kv_rope,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=True,
        )
        # TODO: remove the to memory config after illia's pr is merged
        tt_kv_rope = ttnn.to_memory_config(tt_kv_rope, memory_config=ttnn.L1_MEMORY_CONFIG)
        # 1,32,1,64 4x8 [32,64]
        tt_kv_rope = ttnn.transpose(
            tt_kv_rope, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG
        )  # [1, 1, bsz, qk_rope_head_dim]
        # 1,1,32,64 L1 interleaved

        tt_kvpe = ttnn.concat([tt_kv_nope, tt_kv_rope], **cfg["kv_concat"])
        # 1,1,32,576 L1 interleaved
        tt_kvpe = ttnn.transpose(tt_kvpe, 1, 2)
        # 1,32,1(32),576 L1 interleaved
        tt_kvpe = ttnn.mesh_partition(tt_kvpe, dim=1, cluster_axis=1, **cfg["kvpe_reshard"])
        # 1,4,1(32),576 height sharded 1x4 [32,576]
        ttnn.deallocate(tt_kv_nope)
        ttnn.deallocate(tt_kv_rope)
        return tt_q, tt_kvpe

    @classmethod
    def _fwd_decode_paged_update_cache(
        cls,
        kvpe_cache: ttnn.Tensor,
        tt_kvpe: ttnn.Tensor,
        position_idxs: ttnn.Tensor,
        page_table: ttnn.Tensor,
        mesh_shape: tuple[int, int],
        row_idx: int | None,
    ) -> None:
        # Update KVPE Cache
        # 1,4,1(32),576 height sharded 1x4 [32,576]
        ttnn.experimental.paged_update_cache(
            kvpe_cache,
            tt_kvpe,
            update_idxs_tensor=position_idxs,
            page_table=page_table,
            mesh_coords=set(get_mesh_coords(mesh_shape, row_idx)),
        )

    @classmethod
    def _fwd_decode_q_rope_nope(
        cls,
        tt_q: ttnn.Tensor,
        cfg: RunDecodeConfig,
        rope_tensors: dict,
        bsz: int,
        num_heads_local: int,
        qk_head_dim: int,
        qk_nope_head_dim: int,
    ) -> ttnn.Tensor:
        # Shard in0 to L1 WIDTH sharded for wq_b matmul
        wq_b_in0_memory_config = ttnn.create_sharded_memory_config(
            tt_q.shape,
            **cfg["wq_b_in0_memory_config"],
        )
        tt_q = ttnn.to_memory_config(tt_q, memory_config=wq_b_in0_memory_config)
        # 1,1,32,1536, width sharded 8x2 [32,96]
        tt_q = ttnn.linear(tt_q, **cfg["wq_b"])
        # 1,1,32,3072, L1 interleaved
        # Reshape
        tt_q = ttnn.untilize(
            tt_q,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))}),
                    (1, 3072),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        tt_q = ttnn.experimental.view(tt_q, (1, bsz, num_heads_local, qk_head_dim))
        tt_q = ttnn.to_memory_config(tt_q, memory_config=ttnn.L1_MEMORY_CONFIG)
        tt_q = ttnn.tilize_with_zero_padding(tt_q)

        # 1,32,16,192 L1 interleaved
        tt_q_nope = ttnn.slice(tt_q, [0, 0, 0, 0], [1, bsz, num_heads_local, qk_nope_head_dim])
        # 1,32,16,192 L1 interleaved
        tt_q_rope = ttnn.slice(
            tt_q, [0, 0, 0, qk_nope_head_dim], [1, bsz, num_heads_local, qk_head_dim], **cfg["q_rope_slice"]
        )
        # 1,32,16,128 L1 interleaved

        # Q Rope: wkv_b1
        # 1,32,16,192 L1 interleaved
        tt_q_nope = ttnn.transpose(tt_q_nope, 1, 2)  # [1, num_heads_local, bsz, qk_nope_head_dim]
        # 1,16,32,128 L1 interleaved

        # Shard activations on optimal DRAM bank-to-worker cores for batched matmul
        tt_q_nope = ttnn.to_memory_config(tt_q_nope, memory_config=cfg["wkv_b1_in0_memory_config"])

        tt_q_nope = ttnn.linear(tt_q_nope, **cfg["wkv_b1"])  # [1, num_heads_local_padded, bsz, kv_lora_rank]
        tt_q_nope = ttnn.to_memory_config(tt_q_nope, memory_config=ttnn.L1_MEMORY_CONFIG)

        # 1,16,32,512 L1 interleaved
        tt_q_nope = ttnn.transpose(tt_q_nope, 1, 2)  # [1, bsz, num_heads_local, kv_lora_rank]
        # 1,32,16,512 L1 interleaved

        # Q RoPE
        # 1,32,16,64 height sharded 8x4 [32,64]
        tt_q_rope = ttnn.experimental.rotary_embedding_llama(
            tt_q_rope,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=True,
        )
        # 1,32,16,64 width sharded 8x4 [32,64]
        tt_q_rope = ttnn.to_memory_config(tt_q_rope, **cfg["q_rope_out_reshard"])
        # 1,32,16,64 L1 interleaved

        # Concat Q Nope and Q Rope
        # 1,32,16,512 L1 interleaved | # 1,32,16,64 L1 interleaved
        tt_q = ttnn.concat([tt_q_nope, tt_q_rope], **cfg["q_concat"])
        # 1,32,16,576 L1 interleaved
        return tt_q

    @classmethod
    def _fwd_decode_all_to_all_pre_flash_mla(cls, tt_q: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        # 1,32,16,576 L1 interleaved
        tt_q = ttnn.experimental.all_to_all_async_generic(tt_q, **cfg["wq_a2a_decode"], **cfg["flash_mla_reshard"])
        # 1,4,128,576 L1 height sharded 8x9 [32,576]
        return tt_q

    @classmethod
    def _fwd_decode_flash_mla(
        cls,
        tt_q: ttnn.Tensor,
        kvpe_cache: ttnn.Tensor,
        page_table: ttnn.Tensor,
        position_idxs: ttnn.Tensor,
        cfg: RunDecodeConfig,
    ) -> ttnn.Tensor:
        # 1,4,128,576 L1 height sharded 8x9 [32,576]
        attn_out = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            tt_q,
            kvpe_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=position_idxs,
            **cfg["flash_mla"],
        )  #  [1, bsz_local, num_heads, kv_lora_rank]
        ttnn.deallocate(tt_q)
        # 1,4,128,512 height sharded 8x9 [32,512]
        return attn_out

    @classmethod
    def _fwd_decode_wkv_b2(cls, attn_out: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        # 1,4,128,512 height sharded 8x9 [32,512]
        attn_out = ttnn.to_memory_config(attn_out, **cfg["flash_mla_out_reshard"])
        # 1,4,128,512 L1 interleaved
        # wkv_b2: DP
        attn_out = ttnn.transpose(attn_out, 1, 2)  # [1, num_heads, bsz, kv_lora_rank]
        # 1,128,4,512 L1 interleaved

        # Shard activations on optimal DRAM bank-to-worker cores for batched matmul
        attn_out = ttnn.to_memory_config(attn_out, memory_config=cfg["wkv_b2_in0_memory_config"])
        v_out = ttnn.linear(attn_out, **cfg["wkv_b2"])  # [1, num_heads_padded, bsz, v_head_dim]
        v_out = ttnn.to_memory_config(v_out, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Slice off padding from wkv_b2 output

        # 1,128,4,128 L1 interleaved = [1, num_heads, bsz, v_head_dim]
        v_out = ttnn.transpose(v_out, 1, 2)
        # 1,4,128,128 L1 interleaved = [1, bsz, num_heads, v_head_dim]
        v_out = ttnn.to_memory_config(
            v_out,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
                    (128, 128),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        return v_out

    @classmethod
    def _fwd_decode_ag_reshape(
        cls,
        v_out: ttnn.Tensor,
        cfg: RunDecodeConfig,
        ccl: CCL,
        bsz: int,
        num_heads: int,
        v_head_dim: int,
    ) -> ttnn.Tensor:
        mesh_shape = cfg["mesh_shape"]
        # 1,4,128,128 L1 interleaved
        # Reshape
        v_out = ttnn.untilize(v_out)
        v_out = ttnn.experimental.view(v_out, (1, 1, bsz // mesh_shape[1], num_heads * v_head_dim))
        # All_gather
        v_out = ttnn.to_memory_config(v_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        v_out = ttnn.all_broadcast(v_out, **cfg["wo_ag_decode"])
        v_out = ttnn.concat(v_out, dim=2)
        v_out = ttnn.tilize(v_out)
        # 1,1,32,16384 L1 interleaved
        return v_out

    @classmethod
    def _fwd_decode_wo(cls, v_out: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        # 1,1,32,16384 L1 interleaved
        # Shard in0 to L1 WIDTH sharded for wo matmul
        wo_in0_memory_config = ttnn.create_sharded_memory_config(
            v_out.shape,
            **cfg["wo_in0_memory_config"],
        )
        v_out = ttnn.to_memory_config(v_out, memory_config=wo_in0_memory_config)
        out = ttnn.linear(v_out, **cfg["wo"])  # [1, 1, bsz, dim]
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        # 1,1,32,896 width sharded 7x4 [32,32]
        return out

    @classmethod
    def _fwd_prefill_wq_kv_a(
        cls,
        x: ttnn.Tensor,
        cfg: RunPrefillConfig,
        ccl: CCL,
        seq_len: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        # Fused wq_kv_a matmul (interleaved in0 + DRAM WIDTH sharded in1)
        dim = x.shape[3]
        qkv_a_n = q_lora_rank + kv_lora_rank + qk_rope_head_dim
        wq_kv_a_program_config = build_prefill_matmul_program_config(seq_len, k=dim, n=qkv_a_n)
        tt_q_kv = ttnn.linear(x, **cfg["wq_kv_a"], program_config=wq_kv_a_program_config)

        # AR using AG + local reduce (since sub-tile RS not supported for new shapes)
        tt_q_kv = ttnn.experimental.all_gather_async(
            tt_q_kv, **ccl.populate_all_gather_runtime_args(cfg["wq_kv_a_ag_prefill"])
        )  # [1, num_devices, seq_len, q_lora_rank + kv_lora_rank + qk_rope_head_dim]
        tt_q_kv = ttnn.experimental.fast_reduce_nc(
            tt_q_kv, **cfg["wq_kv_a_r_prefill"]
        )  # [1, 1, seq_len, q_lora_rank + kv_lora_rank + qk_rope_head_dim]

        # Slice into three parts: tt_q, tt_kv_nope, tt_kv_rope
        tt_q = ttnn.slice(tt_q_kv, [0, 0, 0, 0], [1, 1, seq_len, q_lora_rank])
        tt_kv_nope = ttnn.slice(tt_q_kv, [0, 0, 0, q_lora_rank], [1, 1, seq_len, q_lora_rank + kv_lora_rank])
        tt_kv_rope = ttnn.slice(
            tt_q_kv,
            [0, 0, 0, q_lora_rank + kv_lora_rank],
            [1, 1, seq_len, q_lora_rank + kv_lora_rank + qk_rope_head_dim],
        )
        ttnn.deallocate(tt_q_kv)
        return tt_q, tt_kv_nope, tt_kv_rope
