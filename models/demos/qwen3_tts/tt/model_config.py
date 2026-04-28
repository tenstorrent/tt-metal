# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Configuration classes for Qwen3-TTS TTNN implementation.
"""

from dataclasses import dataclass
from typing import Tuple

import ttnn


def _mesh_or_skip_width_shard(device) -> bool:
    """Skip width-sharding only for multi-device tensor-parallel MeshDevice setups.

    On a single Blackhole/Wormhole chip the demo may still open the device as a
    1-device MeshDevice (`ttnn.open_mesh_device(mesh_shape=(1,1))`); in that case
    we should still apply width-sharding for matmul activations. Only true
    multi-chip meshes (TP) need to skip this path.
    """
    if device.__class__.__name__ != "MeshDevice":
        return False
    try:
        num_devices = int(device.get_num_devices())
    except Exception:
        try:
            num_devices = len(device.get_devices())
        except Exception:
            num_devices = 1
    return num_devices > 1


def matmul_core_grid_from_device(device):
    gs = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(x=gs.x, y=gs.y)


def _already_width_sharded(x: ttnn.Tensor) -> bool:
    """Skip resharding only when in0 is already L1_WIDTH_SHARDED.

    Both the layout AND buffer type must match; otherwise (e.g.
    DRAM_WIDTH_SHARDED, or generic is_sharded()) we still want to re-shard
    into L1 so the matmul kernel reads from on-chip memory.
    """
    mc = x.memory_config()
    return (
        mc.is_sharded()
        and mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        and mc.buffer_type == ttnn.BufferType.L1
    )


_TILE = 32

# One-shot diagnostic logging. Helps verify whether the width-sharding wrapper is
# actually being exercised at runtime (e.g. during trace capture).
import os as _os

_QWEN3_TTS_MM_DEBUG = _os.environ.get("QWEN3_TTS_MM_DEBUG", "0") == "1"
_DEBUG_LIMIT = 8
_debug_seen = 0


def _dbg(msg: str) -> None:
    global _debug_seen
    if not _QWEN3_TTS_MM_DEBUG:
        return
    if _debug_seen >= _DEBUG_LIMIT:
        return
    _debug_seen += 1
    print(f"[qwen3_tts mm-shard] {msg}", flush=True)


def _largest_divisor_le(value: int, cap: int) -> int:
    """Largest divisor of `value` that is <= `cap`."""
    cap = max(1, min(int(cap), int(value)))
    for d in range(cap, 0, -1):
        if value % d == 0:
            return d
    return 1


def interleaved_to_width_sharded_activation(x: ttnn.Tensor, device) -> ttnn.Tensor:
    """INTERLEAVED (L1 or DRAM) TILE [B,1,S,H] -> L1 WIDTH_SHARDED for matmul in0.

    Picks `num_cores` as the largest divisor of W_tiles that fits the device grid,
    so the resulting shard is always tile-aligned (avoids the silent fallback that
    leaves the input as L1_INTERLEAVED).
    """
    if _mesh_or_skip_width_shard(device):
        _dbg(f"skip mesh shape={tuple(x.shape)}")
        return x
    if _already_width_sharded(x):
        _dbg(f"already_sharded shape={tuple(x.shape)}")
        return x
    if len(tuple(x.shape)) != 4:
        _dbg(f"skip non-4D shape={tuple(x.shape)}")
        return x
    # Width-sharded matmul kernel forbids batched in0 (e.g. [B, num_heads, S, D]
    # for SDPA Q @ K^T / attn @ V). Detect the batched case via the leading
    # dims excluding the row dim: if their product > 1, this is a batched
    # matmul input and we must keep it interleaved to avoid:
    #   "Input A memory layout must not be WIDTH_SHARDED" (matmul_program_config).
    _shp = tuple(x.shape)
    _batch_dims = _shp[:-2]
    _batch_prod = 1
    for _d in _batch_dims:
        _batch_prod *= int(_d)
    if _batch_prod > 1:
        _dbg(f"skip batched-matmul shape={_shp}")
        return x
    mc_in = x.memory_config()
    if mc_in.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        _dbg(f"skip non-interleaved layout={mc_in.memory_layout} shape={tuple(x.shape)}")
        return x
    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=mc_in)
        mc_in = x.memory_config()

    x_l1 = x
    if mc_in.buffer_type == ttnn.BufferType.DRAM:
        x_l1 = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)

    # Use padded_shape (storage shape) so decode tensors with logical seq=1
    # (storage tile-padded to 32) still hit the width-sharded path.
    try:
        padded_shape = tuple(x_l1.padded_shape)
    except Exception:
        padded_shape = tuple(x_l1.shape)
    height = 1
    for d in padded_shape[:-1]:
        height *= int(d)
    width = int(padded_shape[-1])

    if width % _TILE != 0 or height % _TILE != 0:
        _dbg(f"skip non-tile-aligned padded={padded_shape} h={height} w={width}")
        return x_l1

    width_tiles = width // _TILE

    compute_grid = device.compute_with_storage_grid_size()
    grid_cores = int(compute_grid.x) * int(compute_grid.y)

    num_cores = _largest_divisor_le(width_tiles, grid_cores)
    if num_cores < 1:
        _dbg(f"skip no-divisor padded={padded_shape} w_tiles={width_tiles} cores={grid_cores}")
        return x_l1

    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
    shard_w = width // num_cores
    if shard_w % _TILE != 0:
        _dbg(f"skip shard_w!=TILE padded={padded_shape} num_cores={num_cores}")
        return x_l1

    mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, [height, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )

    try:
        out = ttnn.to_memory_config(x_l1, mem_cfg)
    except Exception as exc:
        _dbg(f"to_memory_config raised padded={padded_shape} num_cores={num_cores} err={exc}")
        return x_l1

    out_mc = out.memory_config()
    is_l1_width_sharded = (
        out_mc.is_sharded()
        and out_mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        and out_mc.buffer_type == ttnn.BufferType.L1
    )
    if not is_l1_width_sharded:
        _dbg(
            f"to_memory_config not L1_WIDTH_SHARDED padded={padded_shape} "
            f"got_layout={out_mc.memory_layout} got_buffer={out_mc.buffer_type}"
        )
        if out is not x_l1:
            ttnn.deallocate(out)
        return x_l1

    _dbg(f"L1_WIDTH_SHARDED ok padded={padded_shape} num_cores={num_cores} " f"shard_h={height} shard_w={shard_w}")

    if x_l1 is not x and out is not x_l1:
        ttnn.deallocate(x_l1)
    return out


def linear_width_sharded_in0(x: ttnn.Tensor, weight: ttnn.Tensor, *, device, **kwargs) -> ttnn.Tensor:
    """ttnn.linear with width-sharded activation (in0) where supported."""
    if len(x.shape) == 2:
        m, k = (int(x.shape[0]), int(x.shape[1]))
        x4 = ttnn.reshape(x, [1, 1, m, k])
        out4 = linear_width_sharded_in0(x4, weight, device=device, **kwargs)
        return ttnn.reshape(out4, [m, int(out4.shape[-1])])
    if len(x.shape) == 3:
        b, s, h = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
        x4 = ttnn.reshape(x, [b, 1, s, h])
        out4 = linear_width_sharded_in0(x4, weight, device=device, **kwargs)
        return ttnn.reshape(out4, [b, s, int(out4.shape[-1])])
    if len(x.shape) != 4:
        return ttnn.linear(x, weight, **kwargs)
    xs = interleaved_to_width_sharded_activation(x, device)
    out = ttnn.linear(xs, weight, **kwargs)
    if xs is not x:
        ttnn.deallocate(xs)
    return out


def matmul_width_sharded_in0(a: ttnn.Tensor, b: ttnn.Tensor, *, device, **kwargs) -> ttnn.Tensor:
    """ttnn.matmul with width-sharded first operand where supported."""
    if len(a.shape) == 2:
        m, k = (int(a.shape[0]), int(a.shape[1]))
        a4 = ttnn.reshape(a, [1, 1, m, k])
        out4 = matmul_width_sharded_in0(a4, b, device=device, **kwargs)
        return ttnn.reshape(out4, [m, int(out4.shape[-1])])
    if len(a.shape) != 4:
        return ttnn.matmul(a, b, **kwargs)
    a_s = interleaved_to_width_sharded_activation(a, device)
    out = ttnn.matmul(a_s, b, **kwargs)
    if a_s is not a:
        ttnn.deallocate(a_s)
    return out


@dataclass
class Qwen3TTSTalkerConfig:
    """Configuration for the Qwen3-TTS Talker model."""

    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    text_vocab_size: int = 151936
    audio_vocab_size: int = 3072
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    num_code_groups: int = 16
    mrope_section: Tuple[int, int, int] = (24, 20, 20)
    mrope_interleaved: bool = True

    # TTNN specific settings
    tile_size: int = 32

    @property
    def qkv_size(self) -> int:
        """Combined size of Q, K, V projections."""
        return (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim


@dataclass
class Qwen3TTSCodePredictorConfig:
    """Configuration for the Qwen3-TTS Code Predictor model."""

    hidden_size: int = 1024
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    vocab_size: int = 2048
    max_position_embeddings: int = 65536
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    num_code_groups: int = 16

    # TTNN specific settings
    tile_size: int = 32

    @property
    def qkv_size(self) -> int:
        """Combined size of Q, K, V projections."""
        return (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim


def get_compute_kernel_config():
    """Returns compute kernel config for high-fidelity computations."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def get_compute_kernel_config_hifi4():
    """Returns compute kernel config for highest fidelity computations."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
