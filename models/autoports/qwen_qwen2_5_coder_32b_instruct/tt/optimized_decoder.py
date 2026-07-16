# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-device optimized Qwen2.5-Coder-32B decoder layer.

This module is deliberately independent of ``FunctionalDecoder``.  It keeps the
same public prefill/decode and cache semantics while making the performance
contracts explicit: packed QKV, phase-specific decode weight placement,
width-sharded L1 decode activations, sharded RMSNorm, phase-specific matmul
geometry, explicit compute/SDPA configs, and large 2D prefill matmuls.

``OptimizationConfig`` profiles exist for focused A/B evidence.  They are real
runtime paths, not functional fallbacks; the final default is selected only
after real-weight PCC and traced warmed-decode measurements.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

import ttnn
from models.common.lightweightmodule import LightweightModule

HF_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
EMITTED_BATCH = 32
EMITTED_PREFILL_SEQUENCE = 17
EMITTED_CACHE_LENGTH = 128
REPRESENTATIVE_LAYER = 32
TILE_SIZE = 32


def _config_value(config, name: str):
    value = getattr(config, name, None)
    if value is None and getattr(config, "text_config", None) is not None:
        value = getattr(config.text_config, name, None)
    return value


def _rope_theta(config) -> float:
    value = _config_value(config, "rope_theta")
    if value is not None:
        return float(value)
    parameters = _config_value(config, "rope_parameters")
    if isinstance(parameters, Mapping) and parameters.get("rope_theta") is not None:
        return float(parameters["rope_theta"])
    raise ValueError("Qwen2.5 config does not define rope_theta")


def _state_tensor(state_dict: Mapping[str, torch.Tensor], layer_idx: int, suffix: str) -> torch.Tensor:
    candidates = (
        f"model.layers.{layer_idx}.{suffix}",
        f"model.language_model.layers.{layer_idx}.{suffix}",
        f"language_model.model.layers.{layer_idx}.{suffix}",
        f"layers.{layer_idx}.{suffix}",
        suffix,
    )
    for key in candidates:
        if key in state_dict:
            return state_dict[key]
    raise KeyError(f"Missing Qwen2.5 decoder weight {suffix!r}; tried {candidates}")


def _to_device_tensor(
    tensor: torch.Tensor,
    mesh_device,
    *,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
    )


def _grid_for_cores(num_cores: int) -> ttnn.CoreGrid:
    grids = {
        8: (8, 1),
        16: (8, 2),
        20: (5, 4),
        32: (8, 4),
        40: (8, 5),
        64: (8, 8),
        80: (8, 10),
    }
    if num_cores not in grids:
        raise ValueError(f"No clean rectangular grid registered for {num_cores} cores")
    x, y = grids[num_cores]
    return ttnn.CoreGrid(x=x, y=y)


def _largest_divisor(value: int, limit: int | None = None) -> int:
    if value < 1:
        raise ValueError(f"value must be positive, got {value}")
    start = value if limit is None else min(value, limit)
    for candidate in range(start, 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _subblock_width(per_core_n: int, *, fp32_dest_acc: bool) -> int:
    register_tiles = 4 if fp32_dest_acc else 8
    for width in range(min(register_tiles, per_core_n), 0, -1):
        if per_core_n % width == 0:
            return width
    return 1


def _dram_weight_memory_config(mesh_device, *, k: int, n: int) -> ttnn.MemoryConfig:
    dram_grid_size = mesh_device.dram_grid_size()
    dram_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
            )
        }
    )
    dram_cores = dram_grid_size.x * dram_grid_size.y
    padded_n = math.ceil(n / (TILE_SIZE * dram_cores)) * TILE_SIZE * dram_cores
    shard_spec = ttnn.ShardSpec(
        dram_grid,
        (k, padded_n // dram_cores),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def _width_sharded_memory_config(*, rows: int, width: int, num_cores: int) -> ttnn.MemoryConfig:
    if width % num_cores != 0:
        raise ValueError(f"width={width} must divide evenly across {num_cores} cores")
    return ttnn.create_sharded_memory_config(
        (rows, width // num_cores),
        _grid_for_cores(num_cores),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _width_sharded_memory_config_on_device(mesh_device, *, rows: int, width: int, num_cores: int) -> ttnn.MemoryConfig:
    """Width sharding for advisor core sets that are not clean rectangles."""

    device_grid = mesh_device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, device_grid, row_wise=True)
    shard_width = math.ceil(width / (TILE_SIZE * num_cores)) * TILE_SIZE
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, (rows, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _rowwise_core_range_set(*, start: int, count: int, grid_width: int) -> ttnn.CoreRangeSet:
    ranges = set()
    cursor = start
    remaining = count
    while remaining:
        x = cursor % grid_width
        y = cursor // grid_width
        run = min(remaining, grid_width - x)
        ranges.add(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x + run - 1, y)))
        cursor += run
        remaining -= run
    return ttnn.CoreRangeSet(ranges)


def _decode_matmul_program(
    *,
    rows: int,
    k: int,
    n: int,
    num_cores: int,
    in0_block_w: int | None = None,
    fused_activation=None,
):
    k_tiles = k // TILE_SIZE
    n_tiles = math.ceil(n / TILE_SIZE)
    if k_tiles % num_cores != 0:
        raise ValueError(f"K tiles={k_tiles} must divide across {num_cores} decode cores")
    shard_k_tiles = k_tiles // num_cores
    if in0_block_w is None:
        in0_block_w = _largest_divisor(shard_k_tiles, 8)
    if shard_k_tiles % in0_block_w != 0:
        raise ValueError(f"in0_block_w={in0_block_w} must divide input shard K tiles={shard_k_tiles}")
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=math.ceil(rows / TILE_SIZE),
        per_core_N=math.ceil(n_tiles / num_cores),
        fused_activation=fused_activation,
    )


def _prefill_matmul_program(*, rows: int, k: int, n: int, fp32_dest_acc: bool):
    m_tiles = math.ceil(rows / TILE_SIZE)
    # For larger logical M, TTNN's auto-selected multi-core program chunks the
    # work without the per-core output CB overflow that a single fused-batch
    # config would create.  This is a device path, not a functional fallback.
    if m_tiles > 32:
        return None
    grid_y = min(10, max(1, m_tiles))
    while grid_y > 1 and (k // TILE_SIZE) % grid_y != 0:
        grid_y -= 1
    grid_x = 8
    per_core_m = math.ceil(m_tiles / grid_y)
    per_core_n = math.ceil(math.ceil(n / TILE_SIZE) / grid_x)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=_subblock_width(per_core_n, fp32_dest_acc=fp32_dest_acc),
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def _advisor_1d_program(*, rows: int, k: int, n: int, num_cores: int, fp32_dest_acc: bool):
    """Provisional 1D-mcast seed; replaced with final-IR fields after capture."""

    grid = _grid_for_cores(num_cores)
    k_tiles = k // TILE_SIZE
    n_tiles = math.ceil(n / TILE_SIZE)
    per_core_n = math.ceil(n_tiles / num_cores)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=_largest_divisor(k_tiles, 16),
        out_subblock_h=1,
        out_subblock_w=_subblock_width(per_core_n, fp32_dest_acc=fp32_dest_acc),
        per_core_M=math.ceil(rows / TILE_SIZE),
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _advisor_final_program(*, grid: tuple[int, int], in0_block_w: int, per_core_n: int, out_subblock_w: int):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _compute_kernel(math_fidelity, *, fp32_dest_acc: bool):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc,
        packer_l1_acc=True,
    )


@dataclass(frozen=True)
class OptimizationConfig:
    """Named cumulative decode/prefill contract for controlled experiments."""

    name: str = "bfp8_hifi2_dram_32c"
    attention_weight_dtype: object = ttnn.bfloat8_b
    mlp_gate_up_weight_dtype: object = ttnn.bfloat8_b
    mlp_down_weight_dtype: object = ttnn.bfloat8_b
    kv_cache_dtype: object = ttnn.bfloat8_b
    attention_math_fidelity: object = ttnn.MathFidelity.HiFi2
    mlp_math_fidelity: object = ttnn.MathFidelity.HiFi2
    qkv_cores: int = 32
    output_cores: int = 32
    mlp_gate_up_cores: int = 32
    mlp_down_cores: int = 32
    qkv_in0_block_w: int | None = None
    output_in0_block_w: int | None = None
    # The Qwen gate/up output shard is wide enough that larger K blocks exceed
    # Blackhole's 1.5 MiB L1 circular-buffer budget even on 32 cores.
    mlp_gate_up_in0_block_w: int | None = 1
    mlp_down_in0_block_w: int | None = 1
    packed_mlp: bool = False
    fused_kv_update: bool = True
    explicit_sdpa_program: bool = True
    sharded_residual: bool = True
    weight_strategy: str = "dram_sharded"
    advisor_layout: bool = False

    @classmethod
    def named(cls, name: str) -> "OptimizationConfig":
        cores16 = dict(qkv_cores=16, output_cores=16, mlp_gate_up_cores=16, mlp_down_cores=16)
        profiles = {
            "bf16_hifi4_dram_32c": cls(
                name="bf16_hifi4_dram_32c",
                attention_weight_dtype=ttnn.bfloat16,
                mlp_gate_up_weight_dtype=ttnn.bfloat16,
                mlp_down_weight_dtype=ttnn.bfloat16,
                kv_cache_dtype=ttnn.bfloat16,
                attention_math_fidelity=ttnn.MathFidelity.HiFi4,
                mlp_math_fidelity=ttnn.MathFidelity.HiFi4,
                qkv_cores=32,
                output_cores=32,
                mlp_gate_up_cores=32,
                mlp_down_cores=32,
            ),
            "bf16_hifi4_unfused_cache_32c": cls(
                name="bf16_hifi4_unfused_cache_32c",
                attention_weight_dtype=ttnn.bfloat16,
                mlp_gate_up_weight_dtype=ttnn.bfloat16,
                mlp_down_weight_dtype=ttnn.bfloat16,
                kv_cache_dtype=ttnn.bfloat16,
                attention_math_fidelity=ttnn.MathFidelity.HiFi4,
                mlp_math_fidelity=ttnn.MathFidelity.HiFi4,
                fused_kv_update=False,
            ),
            "bfp8_hifi2_unfused_cache_32c": cls(
                name="bfp8_hifi2_unfused_cache_32c",
                fused_kv_update=False,
            ),
            "bfp8_hifi2_dram_32c": cls(),
            "bfp8_lofi_dram_32c": cls(
                name="bfp8_lofi_dram_32c",
                attention_math_fidelity=ttnn.MathFidelity.LoFi,
                mlp_math_fidelity=ttnn.MathFidelity.LoFi,
                qkv_cores=32,
                output_cores=32,
                mlp_gate_up_cores=32,
                mlp_down_cores=32,
            ),
            "bf16_attention_bfp8_mlp_32c": cls(
                name="bf16_attention_bfp8_mlp_32c",
                attention_weight_dtype=ttnn.bfloat16,
                kv_cache_dtype=ttnn.bfloat16,
                attention_math_fidelity=ttnn.MathFidelity.HiFi4,
            ),
            "bfp8_attention_bf16_mlp_32c": cls(
                name="bfp8_attention_bf16_mlp_32c",
                mlp_gate_up_weight_dtype=ttnn.bfloat16,
                mlp_down_weight_dtype=ttnn.bfloat16,
                mlp_math_fidelity=ttnn.MathFidelity.HiFi4,
            ),
            "bfp8_weights_bf16_cache_32c": cls(
                name="bfp8_weights_bf16_cache_32c",
                kv_cache_dtype=ttnn.bfloat16,
            ),
            "bf16_weights_bfp8_cache_32c": cls(
                name="bf16_weights_bfp8_cache_32c",
                attention_weight_dtype=ttnn.bfloat16,
                mlp_gate_up_weight_dtype=ttnn.bfloat16,
                mlp_down_weight_dtype=ttnn.bfloat16,
                attention_math_fidelity=ttnn.MathFidelity.HiFi4,
                mlp_math_fidelity=ttnn.MathFidelity.HiFi4,
            ),
            "bfp8_hifi2_dram_16c": cls(name="bfp8_hifi2_dram_16c", **cores16),
            "bfp8_lofi_dram_16c": cls(
                name="bfp8_lofi_dram_16c",
                attention_math_fidelity=ttnn.MathFidelity.LoFi,
                mlp_math_fidelity=ttnn.MathFidelity.LoFi,
                **cores16,
            ),
            "bfp4_mlp_lofi_dram_16c": cls(
                name="bfp4_mlp_lofi_dram_16c",
                mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
                mlp_down_weight_dtype=ttnn.bfloat4_b,
                mlp_math_fidelity=ttnn.MathFidelity.LoFi,
                **cores16,
            ),
            "bfp4_gate_up_lofi_dram_16c": cls(
                name="bfp4_gate_up_lofi_dram_16c",
                mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
                mlp_math_fidelity=ttnn.MathFidelity.LoFi,
                **cores16,
            ),
            "bfp4_attention_lofi_dram_16c": cls(
                name="bfp4_attention_lofi_dram_16c",
                attention_weight_dtype=ttnn.bfloat4_b,
                attention_math_fidelity=ttnn.MathFidelity.LoFi,
                **cores16,
            ),
            "packed_mlp_bfp4_lofi_dram_16c": cls(
                name="packed_mlp_bfp4_lofi_dram_16c",
                mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
                mlp_math_fidelity=ttnn.MathFidelity.LoFi,
                packed_mlp=True,
                **cores16,
            ),
            "bfp4_mlp_lofi_dram_32c": cls(
                name="bfp4_mlp_lofi_dram_32c",
                mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
                mlp_down_weight_dtype=ttnn.bfloat4_b,
                mlp_math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            "bfp4_gate_up_lofi_dram_32c": cls(
                name="bfp4_gate_up_lofi_dram_32c",
                mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
                mlp_math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            "bfp4_attention_lofi_dram_32c": cls(
                name="bfp4_attention_lofi_dram_32c",
                attention_weight_dtype=ttnn.bfloat4_b,
                attention_math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            "packed_mlp_bfp4_lofi_dram_32c": cls(
                name="packed_mlp_bfp4_lofi_dram_32c",
                mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
                mlp_math_fidelity=ttnn.MathFidelity.LoFi,
                packed_mlp=True,
            ),
            "packed_mlp_bfp8_hifi2_dram_32c": cls(
                name="packed_mlp_bfp8_hifi2_dram_32c",
                packed_mlp=True,
            ),
            "packed_mlp_bfp8_hifi2_dram_gate40c": cls(
                name="packed_mlp_bfp8_hifi2_dram_gate40c",
                mlp_gate_up_cores=40,
                packed_mlp=True,
            ),
            "advisor_packed_bfp8_hifi2_1d": cls(
                name="advisor_packed_bfp8_hifi2_1d",
                weight_strategy="advisor_interleaved",
                advisor_layout=True,
                packed_mlp=True,
                qkv_in0_block_w=2,
                output_in0_block_w=2,
                mlp_gate_up_in0_block_w=2,
                mlp_down_in0_block_w=2,
            ),
        }
        try:
            return profiles[name]
        except KeyError as error:
            raise ValueError(f"Unknown optimization profile {name!r}; choose from {sorted(profiles)}") from error


class OptimizedDecoder(LightweightModule):
    """One independently implemented optimized Qwen2.5-Coder-32B layer."""

    def __init__(
        self,
        *,
        mesh_device,
        layer_idx: int,
        batch: int,
        max_cache_len: int,
        hidden_size: int,
        attention_width: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        optimization_config: OptimizationConfig,
        input_norm,
        post_attention_norm,
        qkv_weight,
        prefill_qkv_weight,
        qkv_bias,
        output_weight,
        prefill_output_weight,
        gate_weight,
        prefill_gate_weight,
        up_weight,
        prefill_up_weight,
        gate_up_weight,
        prefill_gate_up_weight,
        down_weight,
        prefill_down_weight,
        rotary_cos,
        rotary_sin,
        position_indices,
    ):
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.batch = batch
        self.max_cache_len = max_cache_len
        self.hidden_size = hidden_size
        self.attention_width = attention_width
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.scale = 1.0 / math.sqrt(head_dim)
        self.optimization_config = optimization_config

        self.input_norm = input_norm
        self.post_attention_norm = post_attention_norm
        self.qkv_weight = qkv_weight
        self.prefill_qkv_weight = prefill_qkv_weight
        self.qkv_bias = qkv_bias
        self.output_weight = output_weight
        self.prefill_output_weight = prefill_output_weight
        self.gate_weight = gate_weight
        self.prefill_gate_weight = prefill_gate_weight
        self.up_weight = up_weight
        self.prefill_up_weight = prefill_up_weight
        self.gate_up_weight = gate_up_weight
        self.prefill_gate_up_weight = prefill_gate_up_weight
        self.down_weight = down_weight
        self.prefill_down_weight = prefill_down_weight
        self.rotary_cos = rotary_cos
        self.rotary_sin = rotary_sin
        self.position_indices = position_indices

        rows = TILE_SIZE * math.ceil(batch / TILE_SIZE)
        self.decode_rows = rows
        cfg = optimization_config
        if cfg.advisor_layout:
            norm_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 0))})
            self.decode_residual_cores = 80
            self.decode_residual_mem_config = _width_sharded_memory_config_on_device(
                mesh_device, rows=rows, width=hidden_size, num_cores=80
            )
            self.decode_norm_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(norm_grid, (rows, 480), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.decode_qkv_input_mem_config = _width_sharded_memory_config_on_device(
                mesh_device, rows=rows, width=hidden_size, num_cores=40
            )
            self.decode_qkv_output_mem_config = _width_sharded_memory_config_on_device(
                mesh_device,
                rows=rows,
                width=(num_heads + 2 * num_kv_heads) * head_dim,
                num_cores=75,
            )
            self.decode_mlp_input_mem_config = self.decode_residual_mem_config
            self.decode_gate_up_output_mem_config = _width_sharded_memory_config_on_device(
                mesh_device, rows=rows, width=2 * intermediate_size, num_cores=108
            )
            self.decode_mlp_intermediate_mem_config = _width_sharded_memory_config_on_device(
                mesh_device, rows=rows, width=intermediate_size, num_cores=108
            )
            self.decode_down_input_mem_config = _width_sharded_memory_config_on_device(
                mesh_device, rows=rows, width=intermediate_size, num_cores=87
            )
            self.decode_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[11, 1],
                subblock_w=3,
                block_h=1,
                block_w=15,
                inplace=False,
            )
        elif cfg.sharded_residual:
            if len({cfg.qkv_cores, cfg.output_cores, cfg.mlp_down_cores}) != 1:
                raise ValueError("QKV, output, and MLP-down must share one coherent residual core count")
            residual_cores = cfg.qkv_cores
            self.decode_residual_cores = residual_cores
            self.decode_residual_mem_config = _width_sharded_memory_config(
                rows=rows, width=hidden_size, num_cores=residual_cores
            )
            self.decode_norm_mem_config = self.decode_residual_mem_config
            self.decode_mlp_input_mem_config = _width_sharded_memory_config(
                rows=rows, width=hidden_size, num_cores=cfg.mlp_gate_up_cores
            )
            self.decode_mlp_intermediate_mem_config = _width_sharded_memory_config(
                rows=rows, width=intermediate_size, num_cores=cfg.mlp_down_cores
            )
            block_w = hidden_size // TILE_SIZE // residual_cores
            self.decode_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[
                    _grid_for_cores(residual_cores).x,
                    _grid_for_cores(residual_cores).y,
                ],
                subblock_w=_largest_divisor(block_w, 4),
                block_h=rows // TILE_SIZE,
                block_w=block_w,
                inplace=False,
            )
        else:
            self.decode_residual_cores = 1
            self.decode_residual_mem_config = ttnn.DRAM_MEMORY_CONFIG
            self.decode_norm_mem_config = self.decode_residual_mem_config
            self.decode_mlp_input_mem_config = ttnn.DRAM_MEMORY_CONFIG
            self.decode_mlp_intermediate_mem_config = ttnn.DRAM_MEMORY_CONFIG
            self.decode_norm_program_config = None

        self.attention_compute_config = _compute_kernel(
            cfg.attention_math_fidelity,
            fp32_dest_acc=cfg.attention_math_fidelity == ttnn.MathFidelity.HiFi4,
        )
        self.mlp_compute_config = _compute_kernel(
            cfg.mlp_math_fidelity,
            fp32_dest_acc=cfg.mlp_math_fidelity == ttnn.MathFidelity.HiFi4,
        )
        self.norm_compute_config = _compute_kernel(ttnn.MathFidelity.HiFi2, fp32_dest_acc=True)

        if cfg.advisor_layout:
            self.decode_qkv_program_config = _advisor_final_program(
                grid=(11, 7),
                in0_block_w=cfg.qkv_in0_block_w,
                per_core_n=3,
                out_subblock_w=3,
            )
            self.decode_output_program_config = _advisor_final_program(
                grid=(11, 8),
                in0_block_w=cfg.output_in0_block_w,
                per_core_n=2,
                out_subblock_w=2,
            )
            self.decode_gate_up_program_config = _advisor_final_program(
                grid=(11, 10),
                in0_block_w=cfg.mlp_gate_up_in0_block_w,
                per_core_n=16,
                out_subblock_w=8,
            )
            self.decode_down_program_config = _advisor_final_program(
                grid=(11, 8),
                in0_block_w=cfg.mlp_down_in0_block_w,
                per_core_n=2,
                out_subblock_w=2,
            )
            program_factory = None
        elif cfg.weight_strategy == "dram_sharded":
            program_factory = _decode_matmul_program
        elif cfg.weight_strategy == "advisor_1d":
            program_factory = _advisor_1d_program
        else:
            program_factory = None
        common_kwargs = {"rows": rows}
        if program_factory is None and not cfg.advisor_layout:
            self.decode_qkv_program_config = None
            self.decode_output_program_config = None
            self.decode_gate_up_program_config = None
            self.decode_down_program_config = None
        elif program_factory is _decode_matmul_program:
            self.decode_qkv_program_config = program_factory(
                **common_kwargs,
                k=hidden_size,
                n=(num_heads + 2 * num_kv_heads) * head_dim,
                num_cores=cfg.qkv_cores,
                in0_block_w=cfg.qkv_in0_block_w,
            )
            self.decode_output_program_config = program_factory(
                **common_kwargs,
                k=attention_width,
                n=hidden_size,
                num_cores=cfg.output_cores,
                in0_block_w=cfg.output_in0_block_w,
            )
            self.decode_gate_up_program_config = program_factory(
                **common_kwargs,
                k=hidden_size,
                n=intermediate_size * (2 if cfg.packed_mlp else 1),
                num_cores=cfg.mlp_gate_up_cores,
                in0_block_w=cfg.mlp_gate_up_in0_block_w,
            )
            self.decode_down_program_config = program_factory(
                **common_kwargs,
                k=intermediate_size,
                n=hidden_size,
                num_cores=cfg.mlp_down_cores,
                in0_block_w=cfg.mlp_down_in0_block_w,
            )
        elif program_factory is not None:
            fp32_attention = cfg.attention_math_fidelity == ttnn.MathFidelity.HiFi4
            fp32_mlp = cfg.mlp_math_fidelity == ttnn.MathFidelity.HiFi4
            self.decode_qkv_program_config = program_factory(
                **common_kwargs,
                k=hidden_size,
                n=(num_heads + 2 * num_kv_heads) * head_dim,
                num_cores=cfg.qkv_cores,
                fp32_dest_acc=fp32_attention,
            )
            self.decode_output_program_config = program_factory(
                **common_kwargs,
                k=attention_width,
                n=hidden_size,
                num_cores=cfg.output_cores,
                fp32_dest_acc=fp32_attention,
            )
            self.decode_gate_up_program_config = program_factory(
                **common_kwargs,
                k=hidden_size,
                n=intermediate_size * (2 if cfg.packed_mlp else 1),
                num_cores=cfg.mlp_gate_up_cores,
                fp32_dest_acc=fp32_mlp,
            )
            self.decode_down_program_config = program_factory(
                **common_kwargs,
                k=intermediate_size,
                n=hidden_size,
                num_cores=cfg.mlp_down_cores,
                fp32_dest_acc=fp32_mlp,
            )

        device_grid = mesh_device.compute_with_storage_grid_size()
        decode_grid = ttnn.num_cores_to_corerangeset(num_heads, device_grid, row_wise=True)
        cache_update_grid = ttnn.num_cores_to_corerangeset(batch, device_grid, row_wise=True)
        value_update_grid = _rowwise_core_range_set(start=batch, count=batch, grid_width=device_grid.x)
        self.decode_compute_core_grid = decode_grid
        self.decode_heads_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(decode_grid, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.decode_kv_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cache_update_grid, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.decode_value_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(value_update_grid, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.decode_concat_input_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cache_update_grid, [64, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.decode_sdpa_program_config = (
            ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=False,
                q_chunk_size=0,
                k_chunk_size=0,
            )
            if cfg.explicit_sdpa_program
            else None
        )
        self.prefill_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=64,
            k_chunk_size=64,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int = EMITTED_BATCH,
        max_cache_len: int = EMITTED_CACHE_LENGTH,
        optimization_config: OptimizationConfig | None = None,
        **kwargs,
    ) -> "OptimizedDecoder":
        if kwargs:
            raise TypeError(f"Unsupported OptimizedDecoder kwargs: {sorted(kwargs)}")
        num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
        if num_devices != 1:
            raise ValueError(f"OptimizedDecoder requires a 1x1 mesh, got {num_devices} devices")
        if batch < 1 or batch > EMITTED_BATCH:
            raise ValueError(f"batch must be in [1, {EMITTED_BATCH}], got {batch}")
        if max_cache_len < 1:
            raise ValueError(f"max_cache_len must be positive, got {max_cache_len}")

        hidden_size = int(_config_value(hf_config, "hidden_size"))
        num_layers = int(_config_value(hf_config, "num_hidden_layers"))
        num_heads = int(_config_value(hf_config, "num_attention_heads"))
        num_kv_heads = int(_config_value(hf_config, "num_key_value_heads"))
        head_dim = int(_config_value(hf_config, "head_dim") or hidden_size // num_heads)
        intermediate_size = int(_config_value(hf_config, "intermediate_size"))
        rms_norm_eps = float(_config_value(hf_config, "rms_norm_eps"))
        advertised_context = int(_config_value(hf_config, "max_position_embeddings"))
        attention_width = num_heads * head_dim
        expected_contract = {
            "hidden_size": (hidden_size, 5120),
            "num_hidden_layers": (num_layers, 64),
            "num_attention_heads": (num_heads, 40),
            "num_key_value_heads": (num_kv_heads, 8),
            "head_dim": (head_dim, 128),
            "intermediate_size": (intermediate_size, 27648),
        }
        mismatches = [
            f"{name}={actual} (expected {expected})"
            for name, (actual, expected) in expected_contract.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError(f"{HF_MODEL} config does not match the translated IR: {', '.join(mismatches)}")
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"layer_idx must be in [0, {num_layers}), got {layer_idx}")
        if max_cache_len > advertised_context:
            raise ValueError(f"max_cache_len={max_cache_len} exceeds max_position_embeddings={advertised_context}")
        if str(_config_value(hf_config, "hidden_act")) != "silu":
            raise ValueError("The translated Qwen2.5 decoder requires hidden_act='silu'")
        if _rope_theta(hf_config) != 1_000_000.0:
            raise ValueError(f"The translated IR requires rope_theta=1000000.0, got {_rope_theta(hf_config)}")

        q = _state_tensor(state_dict, layer_idx, "self_attn.q_proj.weight").to(torch.bfloat16)
        k = _state_tensor(state_dict, layer_idx, "self_attn.k_proj.weight").to(torch.bfloat16)
        v = _state_tensor(state_dict, layer_idx, "self_attn.v_proj.weight").to(torch.bfloat16)
        q_bias = _state_tensor(state_dict, layer_idx, "self_attn.q_proj.bias").to(torch.bfloat16)
        k_bias = _state_tensor(state_dict, layer_idx, "self_attn.k_proj.bias").to(torch.bfloat16)
        v_bias = _state_tensor(state_dict, layer_idx, "self_attn.v_proj.bias").to(torch.bfloat16)
        o = _state_tensor(state_dict, layer_idx, "self_attn.o_proj.weight").to(torch.bfloat16)
        gate = _state_tensor(state_dict, layer_idx, "mlp.gate_proj.weight").to(torch.bfloat16)
        up = _state_tensor(state_dict, layer_idx, "mlp.up_proj.weight").to(torch.bfloat16)
        down = _state_tensor(state_dict, layer_idx, "mlp.down_proj.weight").to(torch.bfloat16)
        input_norm = _state_tensor(state_dict, layer_idx, "input_layernorm.weight").to(torch.bfloat16)
        post_attention_norm = _state_tensor(state_dict, layer_idx, "post_attention_layernorm.weight").to(torch.bfloat16)

        kv_width = num_kv_heads * head_dim
        expected_shapes = {
            "q": (attention_width, hidden_size),
            "k": (kv_width, hidden_size),
            "v": (kv_width, hidden_size),
            "q_bias": (attention_width,),
            "k_bias": (kv_width,),
            "v_bias": (kv_width,),
            "o": (hidden_size, attention_width),
            "gate": (intermediate_size, hidden_size),
            "up": (intermediate_size, hidden_size),
            "down": (hidden_size, intermediate_size),
            "input_norm": (hidden_size,),
            "post_attention_norm": (hidden_size,),
        }
        tensors = {
            "q": q,
            "k": k,
            "v": v,
            "q_bias": q_bias,
            "k_bias": k_bias,
            "v_bias": v_bias,
            "o": o,
            "gate": gate,
            "up": up,
            "down": down,
            "input_norm": input_norm,
            "post_attention_norm": post_attention_norm,
        }
        for name, expected in expected_shapes.items():
            if tuple(tensors[name].shape) != expected:
                raise ValueError(f"{name} weight has shape {tuple(tensors[name].shape)}, expected {expected}")

        cfg = optimization_config or OptimizationConfig.named(
            "advisor_packed_bfp8_hifi2_1d" if batch == EMITTED_BATCH else "packed_mlp_bfp8_hifi2_dram_gate40c"
        )
        if cfg.advisor_layout and batch != EMITTED_BATCH:
            raise ValueError(
                f"advisor_packed_bfp8_hifi2_1d is the captured batch-{EMITTED_BATCH} geometry; got batch={batch}"
            )
        qkv = torch.cat((q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)), dim=-1)
        qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0).reshape(1, 1, 1, -1)
        gate_t = gate.transpose(0, 1).contiguous()
        up_t = up.transpose(0, 1).contiguous()
        gate_up = torch.cat((gate_t, up_t), dim=-1) if cfg.packed_mlp else None

        rotary = Qwen2RotaryEmbedding(hf_config)
        positions = torch.arange(max_cache_len, dtype=torch.long).unsqueeze(0)
        rope_probe = torch.zeros((1, 1, max_cache_len, head_dim), dtype=torch.bfloat16)
        cos, sin = rotary(rope_probe, positions)
        cos = cos.to(torch.bfloat16).unsqueeze(1)
        sin = sin.to(torch.bfloat16).unsqueeze(1)

        qkv_n = (num_heads + 2 * num_kv_heads) * head_dim
        strategy_is_dram = cfg.weight_strategy == "dram_sharded"
        norm_mem_config = ttnn.DRAM_MEMORY_CONFIG

        def weight_mem(k_dim: int, n_dim: int):
            return (
                _dram_weight_memory_config(mesh_device, k=k_dim, n=n_dim)
                if strategy_is_dram
                else ttnn.DRAM_MEMORY_CONFIG
            )

        return cls(
            mesh_device=mesh_device,
            layer_idx=layer_idx,
            batch=batch,
            max_cache_len=max_cache_len,
            hidden_size=hidden_size,
            attention_width=attention_width,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
            optimization_config=cfg,
            input_norm=_to_device_tensor(input_norm, mesh_device, memory_config=norm_mem_config),
            post_attention_norm=_to_device_tensor(post_attention_norm, mesh_device, memory_config=norm_mem_config),
            qkv_weight=_to_device_tensor(
                qkv,
                mesh_device,
                dtype=cfg.attention_weight_dtype,
                memory_config=weight_mem(hidden_size, qkv_n),
            ),
            prefill_qkv_weight=_to_device_tensor(qkv, mesh_device, dtype=cfg.attention_weight_dtype),
            qkv_bias=_to_device_tensor(qkv_bias, mesh_device),
            output_weight=_to_device_tensor(
                o.transpose(0, 1),
                mesh_device,
                dtype=cfg.attention_weight_dtype,
                memory_config=weight_mem(attention_width, hidden_size),
            ),
            prefill_output_weight=_to_device_tensor(o.transpose(0, 1), mesh_device, dtype=cfg.attention_weight_dtype),
            gate_weight=(
                None
                if cfg.packed_mlp
                else _to_device_tensor(
                    gate_t,
                    mesh_device,
                    dtype=cfg.mlp_gate_up_weight_dtype,
                    memory_config=weight_mem(hidden_size, intermediate_size),
                )
            ),
            prefill_gate_weight=(_to_device_tensor(gate_t, mesh_device, dtype=cfg.mlp_gate_up_weight_dtype)),
            up_weight=(
                None
                if cfg.packed_mlp
                else _to_device_tensor(
                    up_t,
                    mesh_device,
                    dtype=cfg.mlp_gate_up_weight_dtype,
                    memory_config=weight_mem(hidden_size, intermediate_size),
                )
            ),
            prefill_up_weight=(_to_device_tensor(up_t, mesh_device, dtype=cfg.mlp_gate_up_weight_dtype)),
            gate_up_weight=(
                _to_device_tensor(
                    gate_up,
                    mesh_device,
                    dtype=cfg.mlp_gate_up_weight_dtype,
                    memory_config=weight_mem(hidden_size, 2 * intermediate_size),
                )
                if gate_up is not None
                else None
            ),
            prefill_gate_up_weight=None,
            down_weight=_to_device_tensor(
                down.transpose(0, 1),
                mesh_device,
                dtype=cfg.mlp_down_weight_dtype,
                memory_config=weight_mem(intermediate_size, hidden_size),
            ),
            prefill_down_weight=_to_device_tensor(down.transpose(0, 1), mesh_device, dtype=cfg.mlp_down_weight_dtype),
            rotary_cos=_to_device_tensor(cos, mesh_device),
            rotary_sin=_to_device_tensor(sin, mesh_device),
            position_indices=_to_device_tensor(
                torch.arange(max_cache_len, dtype=torch.int32),
                mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
            ),
        )

    def _validate_hidden_states(self, hidden_states, expected_seq_len: int | None = None) -> int:
        shape = tuple(hidden_states.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != self.batch or shape[3] != self.hidden_size:
            raise ValueError(f"hidden_states must have shape [1, {self.batch}, seq, {self.hidden_size}], got {shape}")
        seq_len = int(shape[2])
        if expected_seq_len is not None and seq_len != expected_seq_len:
            raise ValueError(f"expected seq_len={expected_seq_len}, got {seq_len}")
        if seq_len < 1 or seq_len > self.max_cache_len:
            raise ValueError(f"seq_len must be in [1, {self.max_cache_len}], got {seq_len}")
        return seq_len

    def _validate_caches(self, key_cache, value_cache) -> None:
        expected = (self.batch, self.num_kv_heads, self.max_cache_len, self.head_dim)
        if tuple(key_cache.shape) != expected or tuple(value_cache.shape) != expected:
            raise ValueError(
                f"key/value caches must both have shape {expected}; got "
                f"{tuple(key_cache.shape)} and {tuple(value_cache.shape)}"
            )
        expected_dtype = self.optimization_config.kv_cache_dtype
        if key_cache.dtype != expected_dtype or value_cache.dtype != expected_dtype:
            raise ValueError(
                f"optimized profile {self.optimization_config.name} requires {expected_dtype} caches; "
                f"got {key_cache.dtype} and {value_cache.dtype}"
            )

    def _decode_input(self, hidden_states):
        hidden_states = ttnn.permute(hidden_states, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)
        if self.optimization_config.sharded_residual:
            return ttnn.to_memory_config(hidden_states, self.decode_residual_mem_config)
        return hidden_states

    def _decode_norm(self, hidden_states, weight):
        if self.optimization_config.advisor_layout:
            hidden_states = ttnn.to_memory_config(hidden_states, self.decode_norm_mem_config)
        return ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=weight,
            program_config=self.decode_norm_program_config,
            memory_config=self.decode_norm_mem_config,
            compute_kernel_config=self.norm_compute_config,
        )

    def _qkv_forward(self, hidden_states, *, mode: str, seq_len: int):
        cfg = self.optimization_config
        if mode == "decode":
            if cfg.advisor_layout:
                hidden_states = ttnn.to_memory_config(hidden_states, self.decode_qkv_input_mem_config)
                output_mem = self.decode_qkv_output_mem_config
            else:
                output_mem = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if cfg.sharded_residual else ttnn.L1_MEMORY_CONFIG
            qkv = ttnn.linear(
                hidden_states,
                self.qkv_weight,
                dtype=ttnn.bfloat16,
                program_config=self.decode_qkv_program_config,
                compute_kernel_config=self.attention_compute_config,
                memory_config=output_mem,
            )
            # DRAM-sharded matmul's bias reader assumes per-bank sharded bias
            # pages; feeding the interleaved packed bias silently permutes its
            # 224 tiles.  Keep the projection fused, but apply bias separately.
            return ttnn.add(
                qkv,
                self.qkv_bias,
                dtype=ttnn.bfloat16,
                memory_config=output_mem,
            )
        rows = self.batch * math.ceil(seq_len / TILE_SIZE) * TILE_SIZE
        program = _prefill_matmul_program(
            rows=rows,
            k=self.hidden_size,
            n=(self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            fp32_dest_acc=cfg.attention_math_fidelity == ttnn.MathFidelity.HiFi4,
        )
        return ttnn.linear(
            hidden_states,
            self.prefill_qkv_weight,
            bias=self.qkv_bias,
            dtype=ttnn.bfloat16,
            program_config=program,
            compute_kernel_config=self.attention_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _mlp_forward(self, hidden_states, *, mode: str, seq_len: int):
        cfg = self.optimization_config
        if mode == "decode":
            if cfg.advisor_layout:
                hidden_states = ttnn.to_memory_config(hidden_states, self.decode_mlp_input_mem_config)
            elif cfg.sharded_residual and cfg.mlp_gate_up_cores != self.decode_residual_cores:
                hidden_states = ttnn.to_memory_config(hidden_states, self.decode_mlp_input_mem_config)
            program = self.decode_gate_up_program_config
            bf16_gate_up = cfg.mlp_gate_up_weight_dtype == ttnn.bfloat16
            output_mem = (
                self.decode_gate_up_output_mem_config
                if cfg.advisor_layout
                else (ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if cfg.sharded_residual else ttnn.L1_MEMORY_CONFIG)
            )
        else:
            program = _prefill_matmul_program(
                rows=self.batch * math.ceil(seq_len / TILE_SIZE) * TILE_SIZE,
                k=self.hidden_size,
                n=self.intermediate_size,
                fp32_dest_acc=cfg.mlp_math_fidelity == ttnn.MathFidelity.HiFi4,
            )
            output_mem = ttnn.DRAM_MEMORY_CONFIG

        gate_weight = self.gate_weight if mode == "decode" else self.prefill_gate_weight
        up_weight = self.up_weight if mode == "decode" else self.prefill_up_weight
        gate_up_weight = self.gate_up_weight if mode == "decode" else self.prefill_gate_up_weight

        if cfg.packed_mlp and mode == "decode":
            gate_up = ttnn.matmul(
                hidden_states,
                gate_up_weight,
                dtype=ttnn.bfloat16,
                program_config=program,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=output_mem,
            )
            if cfg.advisor_layout:
                # final_ir.mlir spills the 108-core packed output to
                # interleaved L1 before its two split/slice consumers.
                gate_up = ttnn.sharded_to_interleaved(gate_up, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
            gate, up = ttnn.split(gate_up, self.intermediate_size, dim=-1)
            gated = ttnn.mul(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                dtype=ttnn.bfloat16,
                memory_config=(
                    self.decode_mlp_intermediate_mem_config if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
                ),
            )
        else:
            gate = ttnn.matmul(
                hidden_states,
                gate_weight,
                dtype=ttnn.bfloat16,
                program_config=program,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=output_mem,
            )
            if mode == "decode" and bf16_gate_up:
                # Two BF16 27,648-wide outputs cannot coexist with the second
                # DRAM-sharded matmul's circular buffers in 1.5 MiB L1.
                gate = ttnn.sharded_to_interleaved(gate, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16)
            up = ttnn.matmul(
                hidden_states,
                up_weight,
                dtype=ttnn.bfloat16,
                program_config=program,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=output_mem,
            )
            gated = ttnn.mul(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                dtype=ttnn.bfloat16,
                memory_config=(
                    self.decode_mlp_intermediate_mem_config if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
                ),
            )

        if mode == "decode":
            if cfg.advisor_layout:
                gated = ttnn.to_memory_config(gated, self.decode_down_input_mem_config)
            down_program = self.decode_down_program_config
            down_mem = (
                self.decode_residual_mem_config
                if cfg.advisor_layout
                else (ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if cfg.sharded_residual else ttnn.L1_MEMORY_CONFIG)
            )
        else:
            down_program = _prefill_matmul_program(
                rows=self.batch * math.ceil(seq_len / TILE_SIZE) * TILE_SIZE,
                k=self.intermediate_size,
                n=self.hidden_size,
                fp32_dest_acc=cfg.mlp_math_fidelity == ttnn.MathFidelity.HiFi4,
            )
            down_mem = ttnn.DRAM_MEMORY_CONFIG
        return ttnn.matmul(
            gated,
            self.down_weight if mode == "decode" else self.prefill_down_weight,
            dtype=ttnn.bfloat16,
            program_config=down_program,
            compute_kernel_config=self.mlp_compute_config,
            memory_config=down_mem,
        )

    def prefill_forward(self, hidden_states, key_cache, value_cache):
        seq_len = self._validate_hidden_states(hidden_states)
        self._validate_caches(key_cache, value_cache)
        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_compute_config,
        )
        fused_qkv = self._qkv_forward(normed, mode="prefill", seq_len=seq_len)
        fused_qkv = ttnn.reshape(
            fused_qkv,
            [self.batch, seq_len, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim],
        )
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos = ttnn.slice(self.rotary_cos, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        sin = ttnn.slice(self.rotary_sin, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        query = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.slice(
            query,
            [0, 0, 0, 0],
            [self.batch, self.num_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.slice(
            key,
            [0, 0, 0, 0],
            [self.batch, self.num_kv_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        key_fill = key if key.dtype == key_cache.dtype else ttnn.typecast(key, key_cache.dtype)
        value_fill = value if value.dtype == value_cache.dtype else ttnn.typecast(value, value_cache.dtype)
        for user_id in range(self.batch):
            key_user = ttnn.slice(
                key_fill,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                [1, 1, 1, 1],
            )
            value_user = ttnn.slice(
                value_fill,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                [1, 1, 1, 1],
            )
            ttnn.fill_cache(key_cache, key_user, user_id)
            ttnn.fill_cache(value_cache, value_user, user_id)

        attention = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            program_config=self.prefill_sdpa_program_config,
            compute_kernel_config=self.attention_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.transformer.concatenate_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.reshape(attention, [1, self.batch, seq_len, self.attention_width])
        output_program = _prefill_matmul_program(
            rows=self.batch * math.ceil(seq_len / TILE_SIZE) * TILE_SIZE,
            k=self.attention_width,
            n=self.hidden_size,
            fp32_dest_acc=self.optimization_config.attention_math_fidelity == ttnn.MathFidelity.HiFi4,
        )
        attention = ttnn.matmul(
            attention,
            self.prefill_output_weight,
            dtype=ttnn.bfloat16,
            program_config=output_program,
            compute_kernel_config=self.attention_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states = ttnn.add(residual, attention, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        residual = hidden_states
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_compute_config,
        )
        hidden_states = self._mlp_forward(hidden_states, mode="prefill", seq_len=seq_len)
        return ttnn.add(residual, hidden_states, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward(self, hidden_states, key_cache, value_cache, *, current_pos: int):
        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        self._validate_caches(key_cache, value_cache)
        if current_pos < 0 or current_pos >= self.max_cache_len:
            raise ValueError(f"current_pos must be in [0, {self.max_cache_len}), got {current_pos}")

        hidden_states = self._decode_input(hidden_states)
        residual = hidden_states
        normed = self._decode_norm(hidden_states, self.input_norm)
        fused_qkv = self._qkv_forward(normed, mode="decode", seq_len=1)
        if self.optimization_config.sharded_residual:
            fused_qkv = ttnn.sharded_to_interleaved(fused_qkv, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            overlap_qk_coregrid=not self.optimization_config.fused_kv_update,
            memory_config=self.decode_heads_mem_config,
        )
        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.to_memory_config(value, self.decode_value_mem_config)
        cos = ttnn.slice(
            self.rotary_cos,
            [0, 0, current_pos, 0],
            [1, 1, current_pos + 1, self.head_dim],
            [1, 1, 1, 1],
        )
        sin = ttnn.slice(
            self.rotary_sin,
            [0, 0, current_pos, 0],
            [1, 1, current_pos + 1, self.head_dim],
            [1, 1, 1, 1],
        )
        query = ttnn.experimental.rotary_embedding(query, cos, sin, 0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, 0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.slice(
            query,
            [0, 0, 0, 0],
            [1, self.batch, self.num_heads, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.slice(
            key,
            [0, 0, 0, 0],
            [1, self.batch, self.num_kv_heads, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.to_memory_config(key, self.decode_kv_mem_config)
        position = ttnn.slice(self.position_indices, [current_pos], [current_pos + 1], [1])
        update_indices = ttnn.repeat(position, ttnn.Shape([self.batch]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.optimization_config.fused_kv_update:
            ttnn.experimental.paged_fused_update_cache(
                key_cache,
                key,
                value_cache,
                value,
                update_idxs_tensor=update_indices,
                page_table=None,
            )
        else:
            ttnn.experimental.paged_update_cache(
                key_cache,
                key,
                update_idxs_tensor=update_indices,
                share_cache=False,
                page_table=None,
            )
            ttnn.experimental.paged_update_cache(
                value_cache,
                value,
                update_idxs_tensor=update_indices,
                share_cache=False,
                page_table=None,
            )

        attention = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            is_causal=True,
            cur_pos_tensor=update_indices,
            scale=self.scale,
            program_config=self.decode_sdpa_program_config,
            compute_kernel_config=self.attention_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.to_memory_config(attention, self.decode_concat_input_mem_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(
            attention,
            num_heads=self.num_heads,
            sub_core_grids=self.decode_compute_core_grid,
        )
        attention = ttnn.slice(
            attention,
            [0, 0, 0, 0],
            [1, 1, self.batch, self.attention_width],
            [1, 1, 1, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if not self.optimization_config.advisor_layout:
            attention = ttnn.to_memory_config(attention, self.decode_residual_mem_config)
        attention = ttnn.matmul(
            attention,
            self.output_weight,
            dtype=ttnn.bfloat16,
            program_config=self.decode_output_program_config,
            compute_kernel_config=self.attention_compute_config,
            memory_config=(
                self.decode_residual_mem_config
                if self.optimization_config.advisor_layout
                else (
                    ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
                    if self.optimization_config.sharded_residual
                    else ttnn.L1_MEMORY_CONFIG
                )
            ),
        )
        hidden_states = ttnn.add(
            residual,
            attention,
            dtype=ttnn.bfloat16,
            memory_config=self.decode_residual_mem_config,
        )
        residual = hidden_states
        hidden_states = self._decode_norm(hidden_states, self.post_attention_norm)
        hidden_states = self._mlp_forward(hidden_states, mode="decode", seq_len=1)
        hidden_states = ttnn.add(
            residual,
            hidden_states,
            dtype=ttnn.bfloat16,
            memory_config=self.decode_residual_mem_config,
        )
        return ttnn.permute(hidden_states, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, hidden_states, key_cache, value_cache, *, mode: str, current_pos: int | None = None):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, key_cache, value_cache)
        if mode == "decode":
            if current_pos is None:
                raise ValueError("decode mode requires current_pos")
            return self.decode_forward(hidden_states, key_cache, value_cache, current_pos=current_pos)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")

    def with_geometry(self, **changes) -> OptimizationConfig:
        """Return a config clone for test harnesses; it does not mutate device state."""

        return replace(self.optimization_config, **changes)
