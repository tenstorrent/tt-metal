# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Optimized TTNN decoder layer for meta-llama/Llama-3.2-1B-Instruct.

This module keeps the functional decoder's paged prefill/decode contract while
switching the local MLP and precision policy to the optimized TTNN path:

* BF16 activations, residuals, and norms
* BFP8 attention weights and BFP4 MLP weights
* BFP8 paged KV cache, with prefill K/V fill tensors explicitly cast to cache dtype
* Attention1D paged SDPA/FlashDecode and DRAM-sharded decode QKV/WO matmuls
* Local optimized MLP using DRAM-sharded decode matmuls and 2D prefill matmuls

``from_state_dict`` is the host setup boundary. The hot ``prefill_forward`` and
``decode_forward`` methods accept TTNN tensors and do not call torch,
``ttnn.from_torch``, or ``ttnn.to_torch``.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import ttnn

from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.functional_decoder import (
    MODEL_ID,
    PagedAttentionConfig,
    _layer_prefix,
    _reverse_permute,
    _state_tensor,
)
from models.common.lightweightmodule import LightweightModule
from models.common.modules.attention.attention_1d import Attention1D, Attention1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig, _create_sharded_norm_program_config
from models.common.tensor_utils import TILE_SIZE
from models.common.utility_functions import is_blackhole


MAX_QKV_PREFILL_SEQ_LEN = 2048
MAX_ATTENTION_WO_PREFILL_SEQ_LEN = 1024


@dataclass(frozen=True)
class OptimizedDecoderPrecisionPolicy:
    """Named precision policy for the optimized single-layer decoder."""

    attention_weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    mlp_ff1_ff3_weight_dtype: ttnn.DataType = ttnn.bfloat4_b
    mlp_ff2_weight_dtype: ttnn.DataType = ttnn.bfloat4_b
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    activation_dtype: ttnn.DataType = ttnn.bfloat16
    residual_dtype: ttnn.DataType = ttnn.bfloat16
    norm_weight_dtype: ttnn.DataType = ttnn.bfloat16
    lm_head_weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    mul_dtype: ttnn.DataType = ttnn.bfloat16
    use_qk_fused_decode: bool = False

    def to_dict(self) -> dict[str, str | bool]:
        return {
            key: value if isinstance(value, bool) else dtype_to_config_name(value)
            for key, value in asdict(self).items()
        }


_DTYPE_ALIASES = {
    "bfloat16": ttnn.bfloat16,
    "bf16": ttnn.bfloat16,
    "BFLOAT16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bf8": ttnn.bfloat8_b,
    "bfp8": ttnn.bfloat8_b,
    "BFLOAT8_B": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
    "bf4": ttnn.bfloat4_b,
    "bfp4": ttnn.bfloat4_b,
    "BFLOAT4_B": ttnn.bfloat4_b,
    "float32": ttnn.float32,
    "fp32": ttnn.float32,
    "FLOAT32": ttnn.float32,
}


def dtype_to_config_name(dtype: ttnn.DataType) -> str:
    if dtype == ttnn.bfloat16:
        return "bfloat16"
    if dtype == ttnn.bfloat8_b:
        return "bfloat8_b"
    if dtype == ttnn.bfloat4_b:
        return "bfloat4_b"
    if dtype == ttnn.float32:
        return "float32"
    raise ValueError(f"unsupported dtype for precision config: {dtype}")


def dtype_from_config_name(value: str | ttnn.DataType | None) -> ttnn.DataType | None:
    if value is None or isinstance(value, ttnn.DataType):
        return value
    normalized = str(value).strip()
    if normalized in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[normalized]
    lowered = normalized.lower()
    if lowered in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[lowered]
    raise ValueError(f"unsupported dtype in precision config: {value!r}")


def precision_policy_from_config(config: dict[str, Any]) -> OptimizedDecoderPrecisionPolicy:
    policy_config = config.get("policy", config.get("precision_policy", config))
    defaults = OptimizedDecoderPrecisionPolicy()
    values: dict[str, Any] = {}
    for key, default_value in asdict(defaults).items():
        raw_value = policy_config.get(key, default_value)
        values[key] = raw_value if isinstance(default_value, bool) else dtype_from_config_name(raw_value)
    return OptimizedDecoderPrecisionPolicy(**values)


def _compute_kernel_config_hifi2_fp16() -> ttnn.WormholeComputeKernelConfig:
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _compute_kernel_config_hifi4() -> ttnn.WormholeComputeKernelConfig:
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _find_largest_divisor(n: int, max_divisor: int = 8) -> int:
    for value in range(max_divisor, 0, -1):
        if n % value == 0:
            return value
    return 1


def _find_grid(n_tiles: int, *, target_cores: int = 32, max_rows: int = 8, max_cols: int = 8) -> tuple[int, int]:
    max_cores = max_rows * max_cols
    possible_cores = [cores for cores in range(1, max_cores + 1) if n_tiles % cores == 0]
    possible_cores.sort(key=lambda cores: (abs(cores - target_cores), -cores))

    for cores in possible_cores:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols
    raise AssertionError(f"cannot find grid for {n_tiles} tiles within {max_rows}x{max_cols}")


def _find_grid_k_n(
    k_tiles: int,
    n_tiles: int,
    *,
    target_cores: int = 32,
    max_rows: int = 8,
    max_cols: int = 8,
) -> tuple[int, int]:
    max_cores = max_rows * max_cols
    possible_cores = [cores for cores in range(1, max_cores + 1) if k_tiles % cores == 0 and n_tiles % cores == 0]
    possible_cores.sort(key=lambda cores: (abs(cores - target_cores), -cores))

    for cores in possible_cores:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols
    raise AssertionError(f"cannot find grid for K={k_tiles}, N={n_tiles} tiles")


def _find_prefill_grid(row_tiles: int, col_tiles: int, max_rows: int = 8, max_cols: int = 8) -> tuple[int, int]:
    cols = next((value for value in range(max_cols, 0, -1) if col_tiles % value == 0), None)
    rows = next((value for value in range(max_rows, 0, -1) if row_tiles % value == 0), None)
    assert cols is not None and rows is not None
    return rows, cols


def _get_out_subblock_w(per_core_n: int, out_subblock_h: int = 1) -> int:
    out_subblock_w = 4
    while out_subblock_w > 1:
        if out_subblock_w * out_subblock_h <= 4 and per_core_n % out_subblock_w == 0:
            break
        out_subblock_w -= 1
    return out_subblock_w


def _core_grid_for_width(k: int, *, target_cores: int = 32, tile_size: int = TILE_SIZE) -> ttnn.CoreGrid:
    rows, cols = _find_grid(k // tile_size, target_cores=target_cores)
    return ttnn.CoreGrid(x=cols, y=rows)


def _core_grid_for_k_n(
    k: int,
    n: int,
    *,
    target_cores: int = 32,
    tile_size: int = TILE_SIZE,
) -> ttnn.CoreGrid:
    rows, cols = _find_grid_k_n(k // tile_size, n // tile_size, target_cores=target_cores)
    return ttnn.CoreGrid(x=cols, y=rows)


def _dram_matmul_config(
    *,
    m: int,
    k: int,
    n: int,
    num_cores: int,
    tile_size: int = TILE_SIZE,
) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_find_largest_divisor(k // (tile_size * num_cores)),
        per_core_M=math.ceil(m / tile_size),
        per_core_N=math.ceil(n / (tile_size * num_cores)),
        fused_activation=None,
    )


def _matmul_2d_config(
    *,
    m: int,
    k: int,
    n: int,
    grid_size: tuple[int, int],
    tile_size: int = TILE_SIZE,
    in0_block_w: int | None = None,
    fuse_batch: bool = False,
    per_core_m: int | None = None,
    per_core_n: int | None = None,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    if per_core_m is None:
        per_core_m = math.ceil(m / (tile_size * grid_size[1]))
    if per_core_n is None:
        per_core_n = math.ceil(n / (tile_size * grid_size[0]))

    out_subblock_h = 1
    out_subblock_w = _get_out_subblock_w(per_core_n, out_subblock_h)
    if in0_block_w is None:
        in0_block_w = _find_largest_divisor(k // (tile_size * grid_size[1]))

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=fuse_batch,
    )


def _attention_prefill_qkv_prg_config(
    *,
    seq_len: int,
    qkv_size: int,
    dram_shard_grid_width: int,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    return _matmul_2d_config(
        m=min(seq_len, MAX_QKV_PREFILL_SEQ_LEN),
        k=2048,
        n=qkv_size,
        grid_size=(8, 10) if is_blackhole() else (8, 8),
        in0_block_w=8,
        fuse_batch=seq_len <= MAX_QKV_PREFILL_SEQ_LEN,
        per_core_m=max(1, 8 if seq_len >= MAX_QKV_PREFILL_SEQ_LEN else math.ceil(seq_len / TILE_SIZE / 8)),
        per_core_n=math.ceil(qkv_size / (TILE_SIZE * dram_shard_grid_width)),
    )


def _attention_prefill_wo_prg_config(
    *,
    seq_len: int,
    dim: int,
    k_dim: int,
    dram_shard_grid_width: int,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    return _matmul_2d_config(
        m=min(seq_len, MAX_ATTENTION_WO_PREFILL_SEQ_LEN),
        k=k_dim,
        n=dim,
        grid_size=_find_prefill_grid(8, k_dim // TILE_SIZE),
        in0_block_w=8,
        fuse_batch=seq_len <= MAX_ATTENTION_WO_PREFILL_SEQ_LEN,
        per_core_n=math.ceil(dim / (TILE_SIZE * dram_shard_grid_width)),
    )


def _create_dram_sharded_mem_config(
    *,
    k: int,
    n: int,
    dram_grid: ttnn.CoreRangeSet,
    tile_size: int = TILE_SIZE,
    dram_cores: int = 12,
) -> ttnn.MemoryConfig:
    padded_n = math.ceil(n / (tile_size * dram_cores)) * (tile_size * dram_cores)
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def _mesh_mapper_config(mesh_device: ttnn.MeshDevice, shard_dim: int) -> ttnn.MeshMapperConfig:
    return ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(shard_dim)],
        mesh_shape_override=ttnn.MeshShape([mesh_device.get_num_devices()]),
    )


@dataclass(frozen=True)
class _OptimizedMLPConfig:
    dim: int
    hidden_dim: int
    max_batch_size: int
    mesh_device: ttnn.MeshDevice
    decode_input_memcfg: ttnn.MemoryConfig
    decode_w1_w3_output_memcfg: ttnn.MemoryConfig
    decode_w2_input_memcfg: ttnn.MemoryConfig
    decode_residual_memcfg: ttnn.MemoryConfig
    decode_w1_w3_prg_config: ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
    decode_w2_prg_config: ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
    prefill_w1_w3_prg_config: Any
    prefill_w2_prg_config: Any
    ff1_3_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig
    ff2_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig
    linear_dtype: ttnn.DataType
    mul_dtype: ttnn.DataType
    prefill_len_cutoff: int


class _OptimizedLlamaMLP(LightweightModule):
    """Local optimized Llama MLP path for this autoport checkout."""

    def __init__(
        self,
        *,
        gate_weight: LazyWeight,
        up_weight: LazyWeight,
        down_weight: LazyWeight,
        config: _OptimizedMLPConfig,
    ) -> None:
        super().__init__()
        self.gate_weight_lazy = gate_weight
        self.up_weight_lazy = up_weight
        self.down_weight_lazy = down_weight
        self.config = config
        self._device_weights_loaded = False

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        prefix: str,
        mesh_device: ttnn.MeshDevice,
        dim: int,
        hidden_dim: int,
        max_batch_size: int,
        decode_residual_memcfg: ttnn.MemoryConfig,
        policy: OptimizedDecoderPrecisionPolicy,
        cache_dir: Path | None = None,
        cache_prefix: str = "",
    ) -> "_OptimizedLlamaMLP":
        gate = _state_tensor(state_dict, prefix, "mlp.gate_proj.weight").transpose(0, 1).contiguous()
        up = _state_tensor(state_dict, prefix, "mlp.up_proj.weight").transpose(0, 1).contiguous()
        down = _state_tensor(state_dict, prefix, "mlp.down_proj.weight").transpose(0, 1).contiguous()

        num_devices = mesh_device.get_num_devices()
        if hidden_dim % num_devices != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must divide num_devices {num_devices}")

        tile_padded_batch_rows = TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE)
        per_device_hidden_dim = hidden_dim // num_devices
        decode_grid = _core_grid_for_k_n(dim, per_device_hidden_dim, target_cores=32)
        decode_mlp2_grid = _core_grid_for_k_n(per_device_hidden_dim, dim, target_cores=32)

        decode_input_memcfg = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, dim // decode_grid.num_cores),
            decode_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        decode_w1_w3_output_memcfg = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, per_device_hidden_dim // decode_grid.num_cores),
            decode_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        decode_w2_input_memcfg = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, per_device_hidden_dim // decode_mlp2_grid.num_cores),
            decode_mlp2_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        decode_w1_w3_prg_config = _dram_matmul_config(
            m=tile_padded_batch_rows,
            k=dim,
            n=per_device_hidden_dim,
            num_cores=decode_grid.num_cores,
        )
        decode_w2_prg_config = _dram_matmul_config(
            m=tile_padded_batch_rows,
            k=per_device_hidden_dim,
            n=dim,
            num_cores=decode_mlp2_grid.num_cores,
        )

        prefill_len_cutoff = 512 if is_blackhole() else 1024
        dram_shard_grid_width = 8 if not is_blackhole() else mesh_device.dram_grid_size().x
        prefill_rows = 8
        w1_w3_grid_size = _find_prefill_grid(prefill_rows, dim // TILE_SIZE)
        w2_grid_size = _find_prefill_grid(prefill_rows, per_device_hidden_dim // TILE_SIZE)

        @lru_cache
        def prefill_w1_w3_prg_config(seq_len: int):
            return _matmul_2d_config(
                m=min(seq_len, prefill_len_cutoff),
                k=dim,
                n=per_device_hidden_dim,
                grid_size=w1_w3_grid_size,
                per_core_n=math.ceil(per_device_hidden_dim / (TILE_SIZE * dram_shard_grid_width)),
            )

        @lru_cache
        def prefill_w2_prg_config(seq_len: int):
            return _matmul_2d_config(
                m=min(seq_len, prefill_len_cutoff),
                k=per_device_hidden_dim,
                n=dim,
                grid_size=w2_grid_size,
                per_core_n=math.ceil(dim / (TILE_SIZE * dram_shard_grid_width)),
            )

        dram_grid_size = mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
                )
            }
        )
        w1_w3_memcfg = _create_dram_sharded_mem_config(
            k=dim,
            n=per_device_hidden_dim,
            dram_grid=dram_grid,
            dram_cores=dram_grid_size.x,
        )
        w2_memcfg = _create_dram_sharded_mem_config(
            k=per_device_hidden_dim,
            n=dim,
            dram_grid=dram_grid,
            dram_cores=dram_grid_size.x,
        )

        def cache_name(name: str) -> tuple[Path, str] | None:
            if cache_dir is None:
                return None
            return cache_dir, f"{cache_prefix}_{name}" if cache_prefix else name

        gate_lazy = LazyWeight(
            source=gate,
            dtype=policy.mlp_ff1_ff3_weight_dtype,
            device=mesh_device,
            mesh_mapper_config=_mesh_mapper_config(mesh_device, -1),
            layout=ttnn.TILE_LAYOUT,
            memory_config=w1_w3_memcfg,
            cache_dir_weight_name=cache_name("mlp_gate_dram_sharded"),
        )
        up_lazy = LazyWeight(
            source=up,
            dtype=policy.mlp_ff1_ff3_weight_dtype,
            device=mesh_device,
            mesh_mapper_config=_mesh_mapper_config(mesh_device, -1),
            layout=ttnn.TILE_LAYOUT,
            memory_config=w1_w3_memcfg,
            cache_dir_weight_name=cache_name("mlp_up_dram_sharded"),
        )
        down_lazy = LazyWeight(
            source=down,
            dtype=policy.mlp_ff2_weight_dtype,
            device=mesh_device,
            mesh_mapper_config=_mesh_mapper_config(mesh_device, -2),
            layout=ttnn.TILE_LAYOUT,
            memory_config=w2_memcfg,
            cache_dir_weight_name=cache_name("mlp_down_dram_sharded"),
        )

        config = _OptimizedMLPConfig(
            dim=dim,
            hidden_dim=hidden_dim,
            max_batch_size=max_batch_size,
            mesh_device=mesh_device,
            decode_input_memcfg=decode_input_memcfg,
            decode_w1_w3_output_memcfg=decode_w1_w3_output_memcfg,
            decode_w2_input_memcfg=decode_w2_input_memcfg,
            decode_residual_memcfg=decode_residual_memcfg,
            decode_w1_w3_prg_config=decode_w1_w3_prg_config,
            decode_w2_prg_config=decode_w2_prg_config,
            prefill_w1_w3_prg_config=prefill_w1_w3_prg_config,
            prefill_w2_prg_config=prefill_w2_prg_config,
            ff1_3_compute_kernel_cfg=_compute_kernel_config_hifi2_fp16(),
            ff2_compute_kernel_cfg=_compute_kernel_config_hifi2_fp16(),
            linear_dtype=policy.activation_dtype,
            mul_dtype=policy.mul_dtype,
            prefill_len_cutoff=prefill_len_cutoff,
        )
        return cls(gate_weight=gate_lazy, up_weight=up_lazy, down_weight=down_lazy, config=config)

    def load_device_weights(self) -> None:
        if self._device_weights_loaded:
            return
        self.gate_weight = self.gate_weight_lazy.get_device_weight()
        self.up_weight = self.up_weight_lazy.get_device_weight()
        self.down_weight = self.down_weight_lazy.get_device_weight()
        self._device_weights_loaded = True

    def decode_forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        self.load_device_weights()
        cfg = self.config
        # The advisor tracer cannot inspect a symbolic tensor's memory config. The
        # capture-only flag is default-off and makes the phase declaration explicit.
        if getattr(self, "_advisor_capture", False) or hidden_states.memory_config() != cfg.decode_input_memcfg:
            hidden_states = ttnn.to_memory_config(hidden_states, cfg.decode_input_memcfg)

        gate = ttnn.linear(
            hidden_states,
            self.gate_weight,
            dtype=cfg.linear_dtype,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
            program_config=cfg.decode_w1_w3_prg_config,
            memory_config=cfg.decode_w1_w3_output_memcfg,
        )
        up = ttnn.linear(
            hidden_states,
            self.up_weight,
            dtype=cfg.linear_dtype,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
            program_config=cfg.decode_w1_w3_prg_config,
            memory_config=cfg.decode_w1_w3_output_memcfg,
        )
        ttnn.deallocate(hidden_states)

        fused = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=cfg.mul_dtype,
            memory_config=cfg.decode_w1_w3_output_memcfg,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        if getattr(self, "_advisor_capture", False) or fused.memory_config() != cfg.decode_w2_input_memcfg:
            fused = ttnn.to_memory_config(fused, cfg.decode_w2_input_memcfg)

        out = ttnn.linear(
            fused,
            self.down_weight,
            dtype=cfg.linear_dtype,
            compute_kernel_config=cfg.ff2_compute_kernel_cfg,
            program_config=cfg.decode_w2_prg_config,
            memory_config=cfg.decode_residual_memcfg,
        )
        ttnn.deallocate(fused)
        return out

    def prefill_forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        self.load_device_weights()
        cfg = self.config
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        seq_len = hidden_states.shape[-2]

        if seq_len >= cfg.prefill_len_cutoff:
            if seq_len % cfg.prefill_len_cutoff != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {cfg.prefill_len_cutoff}")
            hidden_states = ttnn.reshape(hidden_states, [1, seq_len // cfg.prefill_len_cutoff, cfg.prefill_len_cutoff, -1])

        gate = ttnn.linear(
            hidden_states,
            self.gate_weight,
            dtype=cfg.linear_dtype,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
            program_config=cfg.prefill_w1_w3_prg_config(seq_len),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            hidden_states,
            self.up_weight,
            dtype=cfg.linear_dtype,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
            program_config=cfg.prefill_w1_w3_prg_config(seq_len),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden_states)

        fused = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=cfg.mul_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        out = ttnn.linear(
            fused,
            self.down_weight,
            dtype=cfg.linear_dtype,
            compute_kernel_config=cfg.ff2_compute_kernel_cfg,
            program_config=cfg.prefill_w2_prg_config(seq_len),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(fused)

        original_shape = out.shape
        return ttnn.reshape(out, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1]))


class OptimizedDecoder(LightweightModule):
    """Single optimized Llama-3.2-1B decoder layer with paged KV cache."""

    def __init__(
        self,
        *,
        attention_norm: RMSNorm1D,
        attention: Attention1D,
        post_attention_norm: RMSNorm1D,
        mlp: _OptimizedLlamaMLP,
        decode_residual_memcfg: ttnn.MemoryConfig,
        mesh_device: ttnn.MeshDevice,
        hf_config: Any,
        layer_idx: int,
        page_block_size: int,
        max_seq_len: int,
        max_batch_size: int,
        precision_policy: OptimizedDecoderPrecisionPolicy,
    ) -> None:
        super().__init__()
        self.attention_norm = attention_norm
        self.attention = attention
        self.post_attention_norm = post_attention_norm
        self.mlp = mlp
        self.decode_residual_memcfg = decode_residual_memcfg
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        self.page_block_size = page_block_size
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.precision_policy = precision_policy

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: ttnn.MeshDevice,
        page_block_size: int = 64,
        max_seq_len: int | None = None,
        max_batch_size: int = 1,
        precision_policy: OptimizedDecoderPrecisionPolicy | None = None,
        cache_path: str | Path | None = None,
        materialize: bool = True,
        use_vllm_paged_kv_cache: bool = False,
        **_: Any,
    ) -> "OptimizedDecoder":
        hidden_size = int(hf_config.hidden_size)
        intermediate_size = int(hf_config.intermediate_size)
        n_heads = int(hf_config.num_attention_heads)
        n_kv_heads = int(getattr(hf_config, "num_key_value_heads", n_heads))
        head_dim = int(getattr(hf_config, "head_dim", hidden_size // n_heads) or (hidden_size // n_heads))
        if hidden_size != 2048 or intermediate_size != 8192 or n_heads != 32 or n_kv_heads != 8 or head_dim != 64:
            raise ValueError(
                f"{MODEL_ID} optimized decoder expected hidden=2048 intermediate=8192 heads=32 "
                f"kv_heads=8 head_dim=64, got hidden={hidden_size} intermediate={intermediate_size} "
                f"heads={n_heads} kv_heads={n_kv_heads} head_dim={head_dim}"
            )
        if bool(getattr(hf_config, "attention_bias", False)) or bool(getattr(hf_config, "mlp_bias", False)):
            raise ValueError(f"{MODEL_ID} optimized decoder only supports bias-free Llama layers")

        policy = precision_policy or OptimizedDecoderPrecisionPolicy()
        max_seq_len = int(max_seq_len or hf_config.max_position_embeddings)
        prefix = _layer_prefix(state_dict, layer_idx)
        cache_dir = Path(cache_path) if cache_path is not None else None
        cache_prefix = f"layer{layer_idx}"

        q_raw = _state_tensor(state_dict, prefix, "self_attn.q_proj.weight")
        k_raw = _state_tensor(state_dict, prefix, "self_attn.k_proj.weight")
        v_raw = _state_tensor(state_dict, prefix, "self_attn.v_proj.weight")
        o_raw = _state_tensor(state_dict, prefix, "self_attn.o_proj.weight")

        q_meta = _reverse_permute(q_raw, n_heads, n_heads * head_dim, hidden_size).transpose(0, 1).contiguous()
        k_meta = _reverse_permute(k_raw, n_kv_heads, n_kv_heads * head_dim, hidden_size).transpose(0, 1).contiguous()
        v_meta = v_raw.transpose(0, 1).contiguous()
        o_tt = o_raw.transpose(0, 1).contiguous()
        wqkv = torch.cat([q_meta, k_meta, v_meta], dim=-1).unsqueeze(0).unsqueeze(0)
        wo = o_tt.unsqueeze(0).unsqueeze(0)

        def lazy_weight(name: str, tensor: torch.Tensor, dtype: ttnn.DataType) -> LazyWeight:
            return LazyWeight(
                source=tensor,
                dtype=dtype,
                cache_dir_weight_name=(cache_dir, f"{cache_prefix}_{name}") if cache_dir is not None else None,
            )

        blocks_per_user = (max_seq_len + page_block_size - 1) // page_block_size
        paged_attention_config = PagedAttentionConfig(
            block_size=page_block_size,
            max_num_blocks=blocks_per_user * max_batch_size,
        )
        dram_shard_grid_width = 8 if not is_blackhole() else mesh_device.dram_grid_size().x

        @lru_cache
        def prefill_qkv_prg_config(seq_len: int) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
            return _attention_prefill_qkv_prg_config(
                seq_len=seq_len,
                qkv_size=head_dim * (2 * n_kv_heads + n_heads),
                dram_shard_grid_width=dram_shard_grid_width,
            )

        @lru_cache
        def prefill_wo_prg_config(seq_len: int) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
            return _attention_prefill_wo_prg_config(
                seq_len=seq_len,
                dim=hidden_size,
                k_dim=(n_heads * head_dim) // mesh_device.get_num_devices(),
                dram_shard_grid_width=dram_shard_grid_width,
            )

        attention = Attention1D.from_config(
            Attention1DConfig(
                wqkv=lazy_weight("wqkv_optimized", wqkv, policy.attention_weight_dtype),
                wo=lazy_weight("wo_optimized", wo, policy.attention_weight_dtype),
                mesh_device=mesh_device,
                dim=hidden_size,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                paged_attention_config=paged_attention_config,
                use_vllm_paged_kv_cache=use_vllm_paged_kv_cache,
                kv_cache_dtype=policy.kv_cache_dtype,
                activation_dtype=policy.activation_dtype,
                use_qk_fused=policy.use_qk_fused_decode,
                li_qkv_decode_compute_kernel_cfg=_compute_kernel_config_hifi2_fp16(),
                sdpa_decode_compute_kernel_cfg=_compute_kernel_config_hifi2_fp16(),
                li_o_decode_compute_kernel_cfg=_compute_kernel_config_hifi2_fp16(),
                prefill_xqkv_prg_config=prefill_qkv_prg_config,
                prefill_wo_prg_config=prefill_wo_prg_config,
                li_qkv_prefill_compute_kernel_cfg=_compute_kernel_config_hifi2_fp16(),
                sdpa_prefill_compute_kernel_cfg=_compute_kernel_config_hifi4(),
                li_o_prefill_compute_kernel_cfg=_compute_kernel_config_hifi2_fp16(),
            )
        )
        advisor_chain = os.environ.get("MD_LLAMA32_ADVISOR_CHAIN", "off")
        attention._advisor_keep_sdpa_l1 = advisor_chain == "sdpa_concat"
        attention._advisor_rotary_k_dram = advisor_chain == "rotary_k_dram"
        attention._advisor_concat_output_dram = advisor_chain == "concat_output_dram"
        decode_input_residual_memcfg = attention.config.decode_residual_memcfg
        decode_residual_memcfg = decode_input_residual_memcfg

        if advisor_chain == "residual_chain_64":
            decode_residual_memcfg = ttnn.create_sharded_memory_config(
                (TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE), hidden_size // 64),
                ttnn.CoreGrid(x=8, y=8),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            attention.config.decode_attn_output_prg_config = _dram_matmul_config(
                m=TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE),
                k=hidden_size,
                n=hidden_size,
                num_cores=64,
            )
            attention.config.decode_residual_memcfg = decode_residual_memcfg

        mlp = _OptimizedLlamaMLP.from_state_dict(
            state_dict,
            prefix=prefix,
            mesh_device=mesh_device,
            dim=hidden_size,
            hidden_dim=intermediate_size,
            max_batch_size=max_batch_size,
            decode_residual_memcfg=decode_residual_memcfg,
            policy=policy,
            cache_dir=cache_dir,
            cache_prefix=cache_prefix,
        )
        if advisor_chain == "residual_chain_64":
            advisor_mlp_wide_memcfg = ttnn.create_sharded_memory_config(
                (TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE), intermediate_size // 64),
                ttnn.CoreGrid(x=8, y=8), ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
            )
            mlp.config = replace(
                mlp.config,
                decode_input_memcfg=decode_residual_memcfg,
                decode_w1_w3_output_memcfg=advisor_mlp_wide_memcfg,
                decode_w2_input_memcfg=advisor_mlp_wide_memcfg,
                decode_residual_memcfg=decode_residual_memcfg,
                decode_w1_w3_prg_config=_dram_matmul_config(
                    m=TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE), k=hidden_size,
                    n=intermediate_size, num_cores=64,
                ),
                decode_w2_prg_config=_dram_matmul_config(
                    m=TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE), k=intermediate_size,
                    n=hidden_size, num_cores=64,
                ),
            )

        norm_eps = float(hf_config.rms_norm_eps)
        norm_compute_cfg = _compute_kernel_config_hifi2_fp16()
        attention_norm = RMSNorm1D.from_config(
            RMSNorm1DConfig(
                weight=lazy_weight(
                    "input_layernorm_optimized",
                    _state_tensor(state_dict, prefix, "input_layernorm.weight"),
                    policy.norm_weight_dtype,
                ),
                mesh_device=mesh_device,
                eps=norm_eps,
                max_batch_size=max_batch_size,
                prefill_distributed=False,
                decode_memory_config=decode_input_residual_memcfg,
                compute_kernel_config=norm_compute_cfg,
            )
        )
        post_attention_norm = RMSNorm1D.from_config(
            RMSNorm1DConfig(
                weight=lazy_weight(
                    "post_attention_layernorm_optimized",
                    _state_tensor(state_dict, prefix, "post_attention_layernorm.weight"),
                    policy.norm_weight_dtype,
                ),
                mesh_device=mesh_device,
                eps=norm_eps,
                max_batch_size=max_batch_size,
                prefill_distributed=False,
                decode_memory_config=mlp.config.decode_input_memcfg,
                compute_kernel_config=norm_compute_cfg,
            )
        )
        if advisor_chain == "residual_chain_64":
            post_attention_norm.config.decode_memory_config = decode_residual_memcfg
            post_attention_norm.config.decode_program_config = _create_sharded_norm_program_config(
                hidden_size,
                ttnn.CoreGrid(x=8, y=8),
                TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE),
            )

        decoder = cls(
            attention_norm=attention_norm,
            attention=attention,
            post_attention_norm=post_attention_norm,
            mlp=mlp,
            decode_residual_memcfg=decode_input_residual_memcfg,
            mesh_device=mesh_device,
            hf_config=hf_config,
            layer_idx=layer_idx,
            page_block_size=page_block_size,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            precision_policy=policy,
        )
        if materialize:
            decoder.load_device_weights()
        return decoder

    @property
    def kv_cache(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        self.attention.load_device_weights()
        return self.attention.kv_cache

    @property
    def decode_input_memcfg(self) -> ttnn.MemoryConfig:
        return self.attention.config.decode_input_memcfg

    def load_device_weights(self) -> None:
        self.attention_norm.load_device_weights()
        self.attention.load_device_weights()
        self.post_attention_norm.load_device_weights()
        self.mlp.load_device_weights()

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor,
        user_id: int = 0,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
    ) -> ttnn.Tensor:
        residual = hidden_states
        normed = self.attention_norm.prefill_forward(hidden_states)
        attn_out = self.attention.prefill_forward(
            normed,
            rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
        )
        hidden_states = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=self.precision_policy.residual_dtype)

        residual = hidden_states
        normed = self.post_attention_norm.prefill_forward(hidden_states)
        mlp_out = self.mlp.prefill_forward(normed)
        return ttnn.add(residual, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=self.precision_policy.residual_dtype)

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        hidden_states = ttnn.to_memory_config(hidden_states, self.decode_residual_memcfg)
        residual = hidden_states
        normed = self.attention_norm.decode_forward(hidden_states)
        attn_out = self.attention.decode_forward(normed, current_pos, rot_mats, page_table=page_table)
        hidden_states = ttnn.add(
            residual,
            attn_out,
            memory_config=self.attention.config.decode_residual_memcfg,
            dtype=self.precision_policy.residual_dtype,
        )

        residual = hidden_states
        normed = self.post_attention_norm.decode_forward(hidden_states)
        mlp_out = self.mlp.decode_forward(normed)
        return ttnn.add(
            residual,
            mlp_out,
            memory_config=self.mlp.config.decode_residual_memcfg,
            dtype=self.precision_policy.residual_dtype,
        )

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str, **kwargs: Any) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
