# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Configuration classes for Qwen3-TTS TTNN implementation.
"""

from dataclasses import dataclass
from typing import Tuple

import ttnn


def _mesh_or_skip_width_shard(device) -> bool:
    """Tensor-parallel MeshDevice uses different layouts; skip activation width-shard."""
    return device.__class__.__name__ == "MeshDevice"


def matmul_core_grid_from_device(device):
    gs = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(x=gs.x, y=gs.y)


def _already_width_sharded(x: ttnn.Tensor) -> bool:
    """Only skip resharding when in0 is actually WIDTH_SHARDED (not generic is_sharded())."""
    mc = x.memory_config()
    return mc.is_sharded() and mc.memory_layout() == ttnn.TensorMemoryLayout.WIDTH_SHARDED


def interleaved_to_width_sharded_activation(x: ttnn.Tensor, device) -> ttnn.Tensor:
    """INTERLEAVED (L1 or DRAM) TILE [B,1,S,H] -> L1 WIDTH_SHARDED for matmul in0."""
    if _mesh_or_skip_width_shard(device):
        return x
    if _already_width_sharded(x):
        return x
    if len(tuple(x.shape)) != 4:
        return x
    mc_in = x.memory_config()
    # L1/DRAM interleaved both use INTERLEAVED layout; do not use ttnn.is_sharded() here — it can
    # skip valid inputs and leave MatmulDeviceOperation reporting in0:l1_interleaved.
    if mc_in.memory_layout() != ttnn.TensorMemoryLayout.INTERLEAVED:
        return x
    if x.layout() != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=mc_in)
        mc_in = x.memory_config()
    # Width-sharded matmul wants L1-backed shards; if in0 stays in DRAM, Tracy often
    # still classifies the op as in0:dram_interleaved. Copy to L1 interleaved first.
    x_l1 = x
    if mc_in.buffer_type == ttnn.BufferType.DRAM:
        x_l1 = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    shape = tuple(x_l1.shape)
    mem_cfg = ttnn.create_sharded_memory_config_(
        shape,
        matmul_core_grid_from_device(device),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
        tile_layout=True,
    )
    out = ttnn.interleaved_to_sharded(x_l1, mem_cfg, keep_l1_aligned=True)
    if x_l1 is not x:
        ttnn.deallocate(x_l1)
    return out


def linear_width_sharded_in0(x: ttnn.Tensor, weight: ttnn.Tensor, *, device, **kwargs) -> ttnn.Tensor:
    """ttnn.linear with width-sharded activation (in0) where supported."""
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
