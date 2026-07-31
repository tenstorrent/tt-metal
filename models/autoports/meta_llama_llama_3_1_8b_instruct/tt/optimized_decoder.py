# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optimized TTNN decoder layer for meta-llama/Llama-3.1-8B-Instruct.

This is the decoder-only optimized stage for the repo-local autoport pipeline.
It preserves the functional decoder's public prefill/decode contract while using
an explicit single-chip optimization policy:

* BF16 activations and RMSNorm weights.
* BFP8 attention weights and BFP4 MLP weights.
* BFP8 paged KV cache by default.
* DRAM-interleaved prefill activations with explicit 2D matmul configs.
* Width-sharded L1 decode residuals with DRAM-sharded decode matmuls.

Setup may consume PyTorch state-dict tensors. ``prefill_forward`` and
``decode_forward`` are TTNN-only hot paths.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import torch
import ttnn

from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import (
    MODEL_DIR_NAME,
    MODEL_ID,
    PagedAttentionConfig,
    _get_layer_tensor,
    _require_llama31_8b_config,
    _reverse_permute,
)
from models.common.lightweightmodule import LightweightModule
from models.common.modules.attention.attention_1d import Attention1D, Attention1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig
from models.common.modules.rmsnorm.rmsnorm_1d import _create_sharded_norm_program_config
from models.common.tensor_utils import TILE_SIZE, get_padded_hidden_dim, zeros_like_paged_cache
from models.common.utility_functions import is_blackhole


@dataclass(frozen=True)
class OptimizedDecoderPolicy:
    """Named dtype/fidelity policy for this optimized decoder stage."""

    name: str = "llama31_8b_single_chip_bfp8_attn_bfp4_mlp_decode_v1"
    activation_dtype: ttnn.DataType = ttnn.bfloat16
    attention_weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    mlp_gate_up_dtype: ttnn.DataType = ttnn.bfloat4_b
    mlp_down_dtype: ttnn.DataType = ttnn.bfloat4_b
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    mlp_mul_dtype: ttnn.DataType = ttnn.bfloat8_b
    mlp_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi


def _compute_kernel_config_lofi() -> ttnn.WormholeComputeKernelConfig:
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


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
    for candidate in range(max_divisor, 0, -1):
        if n % candidate == 0:
            return candidate
    return 1


def _find_grid(n_tiles: int, *, target: int = 32, max_rows: int = 8, max_cols: int = 8) -> tuple[int, int]:
    max_cores = max_rows * max_cols
    possible_cores = [cores for cores in range(1, max_cores + 1) if n_tiles % cores == 0]
    possible_cores.sort(key=lambda cores: abs(cores - target))

    for cores in possible_cores:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols

    raise AssertionError(f"Cannot find grid for {n_tiles} tiles within {max_rows}x{max_cols}")


def _find_prefill_grid(row_tiles: int, col_tiles: int, max_rows: int = 8, max_cols: int = 8) -> tuple[int, int]:
    cols = next((idx for idx in range(max_cols, 0, -1) if col_tiles % idx == 0), None)
    rows = next((idx for idx in range(max_rows, 0, -1) if row_tiles % idx == 0), None)
    if rows is None or cols is None:
        raise AssertionError(f"Cannot find prefill grid for row_tiles={row_tiles}, col_tiles={col_tiles}")
    return rows, cols


def _get_out_subblock_w(per_core_n: int, out_subblock_h: int = 1) -> int:
    out_subblock_w = 4
    while out_subblock_w > 1:
        if out_subblock_w * out_subblock_h <= 4 and per_core_n % out_subblock_w == 0:
            break
        out_subblock_w -= 1
    return out_subblock_w


def _core_grid_for_tiles(n_tiles: int, *, target: int = 32) -> ttnn.CoreGrid:
    rows, cols = _find_grid(n_tiles, target=target)
    return ttnn.CoreGrid(x=cols, y=rows)


def _width_sharded_l1_memcfg(width: int, core_grid: ttnn.CoreGrid, *, rows: int) -> ttnn.MemoryConfig:
    return ttnn.create_sharded_memory_config(
        (rows, width // core_grid.num_cores),
        core_grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _dram_matmul_config(
    *,
    m: int,
    k: int,
    n: int,
    num_cores: int,
    fused_activation=None,
) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_find_largest_divisor(k // (TILE_SIZE * num_cores)),
        per_core_M=math.ceil(m / TILE_SIZE),
        per_core_N=math.ceil(n / (TILE_SIZE * num_cores)),
        fused_activation=fused_activation,
    )


def _matmul_2d_config(
    *,
    m: int,
    k: int,
    n: int,
    grid_size: tuple[int, int],
    in0_block_w: int | None = None,
    fuse_batch: bool = False,
    fused_activation=None,
    per_core_m: int | None = None,
    per_core_n: int | None = None,
    out_subblock_w: int | None = None,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    if per_core_m is None:
        per_core_m = math.ceil(m / (TILE_SIZE * grid_size[1]))
    if per_core_n is None:
        per_core_n = math.ceil(n / (TILE_SIZE * grid_size[0]))
    if in0_block_w is None:
        in0_block_w = _find_largest_divisor(k // (TILE_SIZE * grid_size[1]))

    out_subblock_h = 1
    out_subblock_w = out_subblock_w or _get_out_subblock_w(per_core_n, out_subblock_h)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=fuse_batch,
    )


def _dram_sharded_weight_memcfg(k: int, n: int, mesh_device: ttnn.MeshDevice) -> ttnn.MemoryConfig:
    dram_grid_size = mesh_device.dram_grid_size()
    dram_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
    )
    padded_n = math.ceil(n / (TILE_SIZE * dram_grid_size.x)) * (TILE_SIZE * dram_grid_size.x)
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_n // dram_grid_size.x), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def _norm_weight(
    state_dict: dict[str, torch.Tensor],
    *,
    hf_config: Any,
    layer_idx: int,
    name: str,
    mesh_device: ttnn.MeshDevice,
    max_batch_size: int,
    cache_dir: Path | None,
) -> RMSNorm1D:
    weight = _get_layer_tensor(state_dict, layer_idx, f"{name}.weight")
    source = weight.reshape(1, 1, hf_config.hidden_size // TILE_SIZE, TILE_SIZE)
    lazy_weight = LazyWeight(
        source=source,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_dir_weight_name=(cache_dir, name) if cache_dir is not None else None,
    )
    advisor_cores = int(os.environ.get("LLAMA31_ADVISOR_RESIDUAL_CORES", "0"))
    decode_grid = _core_grid_for_tiles(hf_config.hidden_size // TILE_SIZE, target=advisor_cores) if advisor_cores else None
    decode_memcfg = (
        _width_sharded_l1_memcfg(
            hf_config.hidden_size, decode_grid, rows=TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE)
        )
        if decode_grid else None
    )
    decode_program = (
        _create_sharded_norm_program_config(
            hf_config.hidden_size, decode_grid, TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE), TILE_SIZE
        )
        if decode_grid else None
    )
    return RMSNorm1D.from_config(
        RMSNorm1DConfig(
            weight=lazy_weight,
            mesh_device=mesh_device,
            eps=hf_config.rms_norm_eps,
            max_batch_size=max_batch_size,
            decode_in_sharded=True,
            decode_out_sharded=True,
            prefill_distributed=False,
            decode_memory_config=decode_memcfg,
            decode_program_config=decode_program,
        )
    )


class _OptimizedMLP(LightweightModule):
    """Self-contained optimized 1D Llama SwiGLU MLP for the autoport stage."""

    def __init__(
        self,
        *,
        gate: LazyWeight,
        up: LazyWeight,
        down: LazyWeight,
        dim: int,
        hidden_dim: int,
        max_batch_size: int,
        mesh_device: ttnn.MeshDevice,
        activation_dtype: ttnn.DataType,
        mul_dtype: ttnn.DataType,
        compute_kernel_config: ttnn.WormholeComputeKernelConfig,
        prefill_len_cutoff: int = 1024,
        decode_core_grid: ttnn.CoreGrid | None = None,
    ) -> None:
        super().__init__()
        self.gate_lazy = gate
        self.up_lazy = up
        self.down_lazy = down
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.max_batch_size = max_batch_size
        self.mesh_device = mesh_device
        self.activation_dtype = activation_dtype
        self.mul_dtype = mul_dtype
        self.prefill_len_cutoff = prefill_len_cutoff
        self.decode_rows = TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE)
        self.decode_core_grid = decode_core_grid or _core_grid_for_tiles(dim // TILE_SIZE, target=32)
        self.decode_input_memcfg = _width_sharded_l1_memcfg(dim, self.decode_core_grid, rows=self.decode_rows)
        self.decode_hidden_memcfg = _width_sharded_l1_memcfg(hidden_dim, self.decode_core_grid, rows=self.decode_rows)
        self.decode_output_memcfg = _width_sharded_l1_memcfg(dim, self.decode_core_grid, rows=self.decode_rows)
        self.decode_gate_up_prg_config = _dram_matmul_config(
            m=self.decode_rows,
            k=dim,
            n=hidden_dim,
            num_cores=self.decode_core_grid.num_cores,
        )
        self.decode_down_prg_config = _dram_matmul_config(
            m=self.decode_rows,
            k=hidden_dim,
            n=dim,
            num_cores=self.decode_core_grid.num_cores,
        )
        self.ff1_3_compute_kernel_cfg = compute_kernel_config
        self.ff2_compute_kernel_cfg = compute_kernel_config
        self._loaded = False

        dram_grid_width = 8 if not is_blackhole() else mesh_device.dram_grid_size().x
        prefill_rows = 8
        gate_up_grid = _find_prefill_grid(prefill_rows, dim // TILE_SIZE)
        down_grid = _find_prefill_grid(prefill_rows, hidden_dim // TILE_SIZE)

        @lru_cache
        def gate_up_prefill_prg_config(seq_len: int) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
            long_prefill = seq_len >= self.prefill_len_cutoff
            return _matmul_2d_config(
                m=min(seq_len, self.prefill_len_cutoff),
                k=dim,
                n=hidden_dim,
                grid_size=gate_up_grid,
                in0_block_w=4 if long_prefill else None,
                out_subblock_w=2 if long_prefill else None,
                per_core_n=math.ceil(hidden_dim / (TILE_SIZE * dram_grid_width)),
            )

        @lru_cache
        def down_prefill_prg_config(seq_len: int) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
            long_prefill = seq_len >= self.prefill_len_cutoff
            return _matmul_2d_config(
                m=min(seq_len, self.prefill_len_cutoff),
                k=hidden_dim,
                n=dim,
                grid_size=down_grid,
                in0_block_w=4 if long_prefill else None,
                out_subblock_w=2 if long_prefill else None,
                per_core_n=math.ceil(dim / (TILE_SIZE * dram_grid_width)),
            )

        self.prefill_gate_up_prg_config: Callable[[int], ttnn.MatmulMultiCoreReuseMultiCastProgramConfig] = (
            gate_up_prefill_prg_config
        )
        self.prefill_down_prg_config: Callable[[int], ttnn.MatmulMultiCoreReuseMultiCastProgramConfig] = (
            down_prefill_prg_config
        )

    def load_device_weights(self) -> None:
        if self._loaded:
            return
        self.gate = self.gate_lazy.get_device_weight()
        self.up = self.up_lazy.get_device_weight()
        self.down = self.down_lazy.get_device_weight()
        self._loaded = True

    def prefill_forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self.load_device_weights()
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        seq_len = x.shape[-2]
        reshaped = False
        if seq_len >= self.prefill_len_cutoff:
            if seq_len % self.prefill_len_cutoff != 0:
                raise ValueError(
                    f"seq_len ({seq_len}) must be divisible by prefill_len_cutoff ({self.prefill_len_cutoff})"
                )
            x = ttnn.reshape(x, [1, seq_len // self.prefill_len_cutoff, self.prefill_len_cutoff, -1])
            reshaped = True

        gate = ttnn.linear(
            x,
            self.gate,
            dtype=self.activation_dtype,
            compute_kernel_config=self.ff1_3_compute_kernel_cfg,
            program_config=self.prefill_gate_up_prg_config(seq_len),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            self.up,
            dtype=self.activation_dtype,
            compute_kernel_config=self.ff1_3_compute_kernel_cfg,
            program_config=self.prefill_gate_up_prg_config(seq_len),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        hidden = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=self.mul_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        out = ttnn.linear(
            hidden,
            self.down,
            dtype=self.activation_dtype,
            compute_kernel_config=self.ff2_compute_kernel_cfg,
            program_config=self.prefill_down_prg_config(seq_len),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)
        if reshaped:
            out = ttnn.reshape(out, [1, 1, seq_len, -1])
        return out

    def decode_forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self.load_device_weights()
        x = ttnn.to_memory_config(x, self.decode_input_memcfg)

        gate = ttnn.linear(
            x,
            self.gate,
            dtype=self.activation_dtype,
            compute_kernel_config=self.ff1_3_compute_kernel_cfg,
            program_config=self.decode_gate_up_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            self.up,
            dtype=self.activation_dtype,
            compute_kernel_config=self.ff1_3_compute_kernel_cfg,
            program_config=self.decode_gate_up_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        hidden = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=self.mul_dtype,
            memory_config=self.decode_hidden_memcfg,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        out = ttnn.linear(
            hidden,
            self.down,
            dtype=self.activation_dtype,
            compute_kernel_config=self.ff2_compute_kernel_cfg,
            program_config=self.decode_down_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)
        return ttnn.to_memory_config(out, self.decode_output_memcfg)

    def forward(self, x: ttnn.Tensor, *, mode: str) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(x)
        if mode == "decode":
            return self.decode_forward(x)
        raise ValueError(f"Unknown MLP mode {mode!r}; expected 'prefill' or 'decode'.")


class OptimizedDecoder(LightweightModule):
    """Single-layer optimized TTNN implementation of the target HF Llama decoder."""

    def __init__(
        self,
        *,
        input_layernorm: RMSNorm1D,
        self_attn: Attention1D,
        post_attention_layernorm: RMSNorm1D,
        mlp: _OptimizedMLP,
        policy: OptimizedDecoderPolicy,
    ) -> None:
        super().__init__()
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp
        self.policy = policy
        self.decode_residual_memcfg = input_layernorm.config.decode_memory_config

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: ttnn.MeshDevice,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        page_block_size: int = 64,
        max_num_blocks: int | None = None,
        policy: OptimizedDecoderPolicy | None = None,
        weight_dtype: ttnn.DataType | None = None,
        activation_dtype: ttnn.DataType | None = None,
        kv_cache_dtype: ttnn.DataType | None = None,
        mlp_gate_up_dtype: ttnn.DataType | None = None,
        mlp_down_dtype: ttnn.DataType | None = None,
        cache_dir: str | Path | None = None,
        **kwargs,
    ) -> "OptimizedDecoder":
        if kwargs:
            raise TypeError(f"Unexpected OptimizedDecoder.from_state_dict kwargs: {sorted(kwargs)}")
        _require_llama31_8b_config(hf_config)
        if mesh_device.get_num_devices() != 1:
            raise ValueError("OptimizedDecoder is the single-chip stage; use a 1x1 MeshDevice.")

        base_policy = policy or OptimizedDecoderPolicy()
        policy = OptimizedDecoderPolicy(
            name=base_policy.name,
            activation_dtype=activation_dtype or base_policy.activation_dtype,
            attention_weight_dtype=weight_dtype or base_policy.attention_weight_dtype,
            mlp_gate_up_dtype=mlp_gate_up_dtype or base_policy.mlp_gate_up_dtype,
            mlp_down_dtype=mlp_down_dtype or base_policy.mlp_down_dtype,
            kv_cache_dtype=kv_cache_dtype or base_policy.kv_cache_dtype,
            mlp_mul_dtype=base_policy.mlp_mul_dtype,
            mlp_math_fidelity=base_policy.mlp_math_fidelity,
        )

        max_seq_len = int(max_seq_len or hf_config.max_position_embeddings)
        if max_num_blocks is None:
            max_num_blocks = max(1, (max_batch_size * max_seq_len + page_block_size - 1) // page_block_size)
        paged_attention_config = PagedAttentionConfig(block_size=page_block_size, max_num_blocks=max_num_blocks)
        cache_path = Path(cache_dir) if cache_dir is not None else None

        dim = hf_config.hidden_size
        head_dim = hf_config.head_dim
        n_heads = hf_config.num_attention_heads
        n_kv_heads = hf_config.num_key_value_heads
        q_size = n_heads * head_dim
        kv_size = n_kv_heads * head_dim
        qkv_size = q_size + 2 * kv_size

        wq_raw = _get_layer_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        wk_raw = _get_layer_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        wv_raw = _get_layer_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        wo_raw = _get_layer_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")

        wq = _reverse_permute(wq_raw, n_heads, q_size, dim).transpose(-2, -1)
        wk = _reverse_permute(wk_raw, n_kv_heads, kv_size, dim).transpose(-2, -1)
        wv = wv_raw.transpose(-2, -1)
        wqkv = torch.cat([wq, wk, wv], dim=-1).unsqueeze(0).unsqueeze(0)
        wo = wo_raw.transpose(-2, -1).unsqueeze(0).unsqueeze(0)

        dram_grid_width = 8 if not is_blackhole() else mesh_device.dram_grid_size().x
        attention_prefill_grid = _find_prefill_grid(8, dim // TILE_SIZE)

        @lru_cache
        def attention_xqkv_prefill_prg_config(seq_len: int) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
            return _matmul_2d_config(
                m=seq_len,
                k=dim,
                n=qkv_size,
                grid_size=attention_prefill_grid,
                in0_block_w=4,
                out_subblock_w=4,
                per_core_m=max(1, 8 if seq_len >= 2048 else math.ceil(seq_len / TILE_SIZE / 8)),
                per_core_n=math.ceil(qkv_size / (TILE_SIZE * dram_grid_width)),
                fuse_batch=seq_len <= 2048,
            )

        @lru_cache
        def attention_wo_prefill_prg_config(seq_len: int) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
            return _matmul_2d_config(
                m=min(seq_len, 1024),
                k=dim,
                n=dim,
                grid_size=attention_prefill_grid,
                in0_block_w=4,
                out_subblock_w=4,
                per_core_n=math.ceil(dim / (TILE_SIZE * dram_grid_width)),
                fuse_batch=seq_len <= 1024,
            )

        input_layernorm = _norm_weight(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            name="input_layernorm",
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            cache_dir=cache_path,
        )
        post_attention_layernorm = _norm_weight(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            name="post_attention_layernorm",
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            cache_dir=cache_path,
        )

        advisor_residual_cores = int(os.environ.get("LLAMA31_ADVISOR_RESIDUAL_CORES", "0"))
        advisor_residual_grid = (
            _core_grid_for_tiles(dim // TILE_SIZE, target=advisor_residual_cores)
            if advisor_residual_cores else None
        )
        advisor_residual_memcfg = (
            _width_sharded_l1_memcfg(
                dim, advisor_residual_grid, rows=TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE)
            )
            if advisor_residual_grid else None
        )
        self_attn = Attention1D.from_config(
            Attention1DConfig(
                wqkv=LazyWeight(
                    source=wqkv,
                    dtype=policy.attention_weight_dtype,
                    cache_dir_weight_name=(cache_path, "self_attn_wqkv_optimized") if cache_path else None,
                ),
                wo=LazyWeight(
                    source=wo,
                    dtype=policy.attention_weight_dtype,
                    cache_dir_weight_name=(cache_path, "self_attn_wo_optimized") if cache_path else None,
                ),
                mesh_device=mesh_device,
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                qkv_size=qkv_size,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                paged_attention_config=paged_attention_config,
                kv_cache=(
                    LazyWeight(source=zeros_like_paged_cache(paged_attention_config, n_kv_heads, head_dim), dtype=policy.kv_cache_dtype),
                    LazyWeight(source=zeros_like_paged_cache(paged_attention_config, n_kv_heads, head_dim), dtype=policy.kv_cache_dtype),
                ),
                kv_cache_dtype=policy.kv_cache_dtype,
                wqkv_dtype=policy.attention_weight_dtype,
                wo_dtype=policy.attention_weight_dtype,
                activation_dtype=policy.activation_dtype,
                decode_residual_memcfg=advisor_residual_memcfg,
                scale=head_dim**-0.5,
                prefill_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                prefill_xqkv_prg_config=attention_xqkv_prefill_prg_config,
                prefill_wo_prg_config=attention_wo_prefill_prg_config,
            )
        )

        hidden_dim = hf_config.intermediate_size
        padded_hidden_dim = get_padded_hidden_dim(hidden_dim, mesh_device.get_num_devices(), TILE_SIZE)
        if padded_hidden_dim != hidden_dim:
            raise ValueError(f"Unexpected padded hidden dim {padded_hidden_dim}; Llama 3.1 8B should be tile aligned")
        gate = _get_layer_tensor(state_dict, layer_idx, "mlp.gate_proj.weight").transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        up = _get_layer_tensor(state_dict, layer_idx, "mlp.up_proj.weight").transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        down = _get_layer_tensor(state_dict, layer_idx, "mlp.down_proj.weight").transpose(-2, -1).unsqueeze(0).unsqueeze(0)

        gate_up_memcfg = _dram_sharded_weight_memcfg(dim, hidden_dim, mesh_device)
        down_memcfg = _dram_sharded_weight_memcfg(hidden_dim, dim, mesh_device)
        one_device_mesh_mapper_width = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(-1)],
            mesh_shape_override=ttnn.MeshShape([mesh_device.get_num_devices()]),
        )
        one_device_mesh_mapper_height = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(-2)],
            mesh_shape_override=ttnn.MeshShape([mesh_device.get_num_devices()]),
        )
        mlp_decode_grid = _core_grid_for_tiles(
            dim // TILE_SIZE, target=int(os.environ.get("LLAMA31_ADVISOR_RESIDUAL_CORES", "32"))
        )
        if policy.mlp_math_fidelity == ttnn.MathFidelity.LoFi:
            mlp_compute_kernel_cfg = _compute_kernel_config_lofi()
        elif policy.mlp_math_fidelity == ttnn.MathFidelity.HiFi2:
            mlp_compute_kernel_cfg = _compute_kernel_config_hifi2_fp16()
        elif policy.mlp_math_fidelity == ttnn.MathFidelity.HiFi4:
            mlp_compute_kernel_cfg = _compute_kernel_config_hifi4()
        else:
            raise ValueError(f"Unsupported MLP math fidelity: {policy.mlp_math_fidelity}")
        mlp = _OptimizedMLP(
            gate=LazyWeight(
                source=gate,
                dtype=policy.mlp_gate_up_dtype,
                device=mesh_device,
                mesh_mapper_config=one_device_mesh_mapper_width,
                layout=ttnn.TILE_LAYOUT,
                memory_config=gate_up_memcfg,
                cache_dir_weight_name=(cache_path, "mlp_gate_optimized") if cache_path else None,
            ),
            up=LazyWeight(
                source=up,
                dtype=policy.mlp_gate_up_dtype,
                device=mesh_device,
                mesh_mapper_config=one_device_mesh_mapper_width,
                layout=ttnn.TILE_LAYOUT,
                memory_config=gate_up_memcfg,
                cache_dir_weight_name=(cache_path, "mlp_up_optimized") if cache_path else None,
            ),
            down=LazyWeight(
                source=down,
                dtype=policy.mlp_down_dtype,
                device=mesh_device,
                mesh_mapper_config=one_device_mesh_mapper_height,
                layout=ttnn.TILE_LAYOUT,
                memory_config=down_memcfg,
                cache_dir_weight_name=(cache_path, "mlp_down_optimized") if cache_path else None,
            ),
            dim=dim,
            hidden_dim=hidden_dim,
            max_batch_size=max_batch_size,
            mesh_device=mesh_device,
            activation_dtype=policy.activation_dtype,
            mul_dtype=policy.mlp_mul_dtype,
            compute_kernel_config=mlp_compute_kernel_cfg,
            decode_core_grid=mlp_decode_grid,
        )

        return cls(
            input_layernorm=input_layernorm,
            self_attn=self_attn,
            post_attention_layernorm=post_attention_layernorm,
            mlp=mlp,
            policy=policy,
        )

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
        residual = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = self.input_layernorm.prefill_forward(residual)
        hidden_states = self.self_attn.prefill_forward(
            hidden_states,
            rot_mats=rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
        )
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm.prefill_forward(hidden_states)
        hidden_states = self.mlp.prefill_forward(hidden_states)
        return ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        residual = ttnn.to_memory_config(hidden_states, self.decode_residual_memcfg)
        hidden_states = self.input_layernorm.decode_forward(residual)
        hidden_states = ttnn.to_memory_config(hidden_states, self.self_attn.config.decode_input_memcfg)
        hidden_states = self.self_attn.decode_forward(
            hidden_states,
            current_pos=current_pos,
            rot_mats=rot_mats,
            page_table=page_table,
        )
        if os.environ.get("LLAMA31_ADVISOR_SKIP_ATTN_OUTPUT_RESHARD", "0") != "1":
            hidden_states = ttnn.to_memory_config(hidden_states, self.decode_residual_memcfg)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=self.decode_residual_memcfg)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm.decode_forward(hidden_states)
        hidden_states = self.mlp.decode_forward(hidden_states)
        if os.environ.get("LLAMA31_ADVISOR_SKIP_MLP_OUTPUT_RESHARD", "0") != "1":
            hidden_states = ttnn.to_memory_config(hidden_states, self.decode_residual_memcfg)
        return ttnn.add(residual, hidden_states, memory_config=self.decode_residual_memcfg)

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str, **kwargs) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"Unknown decoder mode {mode!r}; expected 'prefill' or 'decode'.")
