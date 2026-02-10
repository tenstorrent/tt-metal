# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style Rotary Position Embedding (RoPE) setup for 1D-topology devices.

RotarySetup1D pre-computes cos/sin rotation matrices and transformation matrices
at init time, then provides efficient lookup methods at runtime:
  - get_both_trans_mats(): decode + prefill transformation matrices
  - get_rot_idxs(position_idxs): convert position indices to device tensor
  - get_rot_mats(position_idxs): look up cos/sin rotation matrices by position

Note: torch is required at construction time for computing rotation matrices.
      Runtime methods (get_rot_idxs, get_rot_mats, get_both_trans_mats) use only
      ttnn ops and torch for lightweight index manipulation.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import nearest_32
from ttnn import replicate_tensor_to_mesh_mapper

# =============================================================================
# Pure-torch RotaryEmbedding hierarchy (reference implementations)
# These compute the cos/sin lookup tables needed by RotarySetup1D.
# Copied from models/tt_transformers/tt/rope.py to avoid TTTv1 dependencies.
# =============================================================================


class RotaryEmbedding(nn.Module):
    """Base rotary embedding (no scaling)."""

    def __init__(self, dim: int, max_position_embeddings: int, base: float, device: Optional[Any] = None) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
        self.max_seq_len_cached = None

    @staticmethod
    def permute_to_meta_format(cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = cos[:, : cos.shape[1] // 2]
        cos = torch.stack((cos, cos), dim=-1).flatten(-2)
        sin = sin[:, : sin.shape[1] // 2]
        sin = torch.stack((sin, sin), dim=-1).flatten(-2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return cos, sin

    def _set_cos_sin_cache(self, seq_len: int, device: Any, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        self.register_buffer("freqs_cis", torch.complex(cos.float(), sin.float()), persistent=False)
        cos, sin = self.permute_to_meta_format(cos, sin)
        self.register_buffer("cos_cached", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.to(dtype), persistent=False)


class ScaledRotaryEmbedding(RotaryEmbedding, ABC):
    def __init__(
        self, dim: int, max_position_embeddings: int, base: float, factor: float, device: Optional[Any] = None
    ) -> None:
        self.scaling_factor = factor
        super().__init__(dim, max_position_embeddings, base, device)

    @abstractmethod
    def apply_scaling(self, freqs: torch.Tensor) -> torch.Tensor:
        pass

    def _set_cos_sin_cache(self, seq_len: int, device: Any, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        freqs = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))
        t = torch.arange(seq_len * 2.0)
        freqs = self.apply_scaling(freqs)
        freqs = torch.outer(t, freqs).float()
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self.register_buffer("freqs_cis", torch.complex(cos.float(), sin.float()), persistent=False)
        cos, sin = _gather_cos_sin(torch.arange(seq_len), cos, sin)
        self.register_buffer("cos_cached", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.to(dtype), persistent=False)


class YarnRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        factor: float,
        original_max_position_embeddings: int,
        beta_fast: float,
        beta_slow: float,
        mscale: float,
        mscale_all_dim: float,
        truncate: bool = True,
        device: Optional[Any] = None,
    ) -> None:
        self.scaling_factor = factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.truncate = truncate
        super().__init__(dim, max_position_embeddings, base, device)

    @staticmethod
    def yarn_find_correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    @staticmethod
    def yarn_find_correction_range(
        low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int, truncate: bool = True
    ) -> Tuple[float, float]:
        low = YarnRotaryEmbedding.yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = YarnRotaryEmbedding.yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def yarn_get_mscale(scale: float, mscale: float) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    @staticmethod
    def yarn_linear_ramp_mask(min: float, max: float, dim: int) -> torch.Tensor:
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        return torch.clamp(linear_func, 0, 1)

    def _set_cos_sin_cache(self, seq_len: int, device: Any, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        dim = self.dim
        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        low, high = YarnRotaryEmbedding.yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
            self.truncate,
        )
        inv_freq_mask = 1.0 - YarnRotaryEmbedding.yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        _mscale = float(
            YarnRotaryEmbedding.yarn_get_mscale(self.scaling_factor, self.mscale)
            / YarnRotaryEmbedding.yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * _mscale
        sin = emb.sin() * _mscale
        cos, sin = self.permute_to_meta_format(cos, sin)
        self.register_buffer("cos_cached", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.to(dtype), persistent=False)


class LinearScaledRotaryEmbedding(ScaledRotaryEmbedding):
    def apply_scaling(self, freqs: torch.Tensor) -> torch.Tensor:
        return freqs / self.scaling_factor


class LlamaRotaryEmbedding(ScaledRotaryEmbedding):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        factor: float,
        original_max_position_embeddings: int,
        low_freq_factor: float,
        high_freq_factor: float,
        device: Optional[Any] = None,
    ) -> None:
        self.orig_context_len = original_max_position_embeddings
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        super().__init__(dim, max_position_embeddings, base, factor, device)

    def apply_scaling(self, freqs: torch.Tensor) -> torch.Tensor:
        low_freq_wavelen = self.orig_context_len / self.low_freq_factor
        high_freq_wavelen = self.orig_context_len / self.high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / self.scaling_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (self.orig_context_len / wavelen - self.low_freq_factor) / (
                    self.high_freq_factor - self.low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / self.scaling_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


class Phi3RotaryEmbedding(ScaledRotaryEmbedding):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        original_max_position_embeddings: int,
        long_factor: List[int],
        short_factor: List[int],
        device: Optional[Any] = None,
    ) -> None:
        self.orig_context_len = original_max_position_embeddings
        self.long_factor = long_factor
        self.short_factor = short_factor
        scale = 1024 * 128 / self.orig_context_len
        scaling_factor = 1.0 if scale <= 1.0 else math.sqrt(1 + math.log(scale) / math.log(self.orig_context_len))
        super().__init__(dim, max_position_embeddings, base, scaling_factor, device)

    def apply_scaling(self, freqs: torch.Tensor) -> torch.Tensor:
        if self.max_seq_len_cached > self.orig_context_len:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32)
        assert freqs.shape[-1] == ext_factors.shape[-1]
        return freqs / ext_factors

    def _set_cos_sin_cache(self, seq_len: int, device: Any, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        inv_freq_shape = torch.arange(0, self.dim, 2).float().to(device) / self.dim
        self.inv_freq = 1.0 / (self.base**inv_freq_shape)
        self.inv_freq = self.apply_scaling(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.scaling_factor
        sin = emb.sin() * self.scaling_factor
        cos, sin = self.permute_to_meta_format(cos, sin)
        self.register_buffer("cos_cached", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.to(dtype), persistent=False)


# =============================================================================
# Helper functions (copied from TTTv1 common.py and rope.py)
# =============================================================================


def _gather_cos_sin(position_ids, cos, sin):
    """Gather cos/sin values for given position IDs and format for Meta-style RoPE."""
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


def _get_rot_transformation_mat(dhead):
    """Build the rotation transformation matrix for RoPE (TILE_SIZE x TILE_SIZE)."""
    dhead = 32  # ROPE op uses a single tile
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def _rotary_embedding_factory(dim, max_position_embeddings, base, rope_scaling=None):
    """Create the appropriate RotaryEmbedding variant based on rope_scaling config."""
    if rope_scaling is None:
        return RotaryEmbedding(dim, max_position_embeddings, base)
    if rope_scaling.rope_type.value == "linear":
        cls = LinearScaledRotaryEmbedding
    elif rope_scaling.rope_type.value == "llama3":
        cls = LlamaRotaryEmbedding
    elif rope_scaling.rope_type.value == "yarn":
        cls = YarnRotaryEmbedding
    elif rope_scaling.rope_type.value == "longrope":
        cls = Phi3RotaryEmbedding
    else:
        raise ValueError(f"Invalid rope_scaling: {rope_scaling}")
    return cls(
        dim=dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
        **rope_scaling.model_dump(exclude_none=True),
    )


def _compute_cos_sin_matrices(head_dim, max_seq_len, rope_theta, rope_scaling=None):
    """Compute cos/sin lookup tables as torch tensors [1, 1, max_seq_len, head_dim]."""
    rotary_emb = _rotary_embedding_factory(
        dim=head_dim, max_position_embeddings=max_seq_len, base=rope_theta, rope_scaling=rope_scaling
    )
    return rotary_emb.cos_cached, rotary_emb.sin_cached


def _cos_sin_to_device(cos_torch, sin_torch, device, datatype):
    """Convert cos/sin torch tensors to TTNN device tensors (replicated)."""
    cos_matrix = ttnn.from_torch(
        cos_torch,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=datatype,
        mesh_mapper=replicate_tensor_to_mesh_mapper(device),
    )
    sin_matrix = ttnn.from_torch(
        sin_torch,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=datatype,
        mesh_mapper=replicate_tensor_to_mesh_mapper(device),
    )
    return cos_matrix, sin_matrix


# =============================================================================
# Config dataclass
# =============================================================================


@dataclass
class RotarySetup1DConfig:
    """
    Configuration for RotarySetup1D.

    Simple usage:
        config = RotarySetup1DConfig(
            device=mesh_device, batch_size=32, head_dim=128,
            max_seq_len=8192, rope_theta=500000.0,
        )

    With Llama-3.x scaling:
        config = RotarySetup1DConfig(
            device=mesh_device, batch_size=32, head_dim=128,
            max_seq_len=8192, rope_theta=500000.0,
            rope_scaling=RopeScalingLlama3(rope_type=RopeScalingType.LLAMA3, factor=8.0, ...),
        )
    """

    # Required
    device: Any  # ttnn.MeshDevice or single device
    batch_size: int
    head_dim: int
    max_seq_len: int
    rope_theta: float

    # Optional
    rope_scaling: Optional[Any] = None  # RopeScaling from common.py
    use_qk_fused: bool = False
    datatype: ttnn.DataType = ttnn.bfloat16


# =============================================================================
# RotarySetup1D - RoPE for 1D-topology devices (non-TG)
# =============================================================================


class RotarySetup1D(LightweightModule):
    """
    Rotary position embedding setup for non-TG (1D) devices.

    Pre-computes cos/sin tables and transformation matrices at init time.
    Provides efficient position-based lookup at runtime.

    Simple API:
        rope = RotarySetup1D(
            device=mesh_device, batch_size=32, head_dim=128,
            max_seq_len=8192, rope_theta=500000.0,
        )
        trans_mats = rope.get_both_trans_mats()
        rot_idxs = rope.get_rot_idxs(position_ids)
        cos_sin = rope.get_rot_mats(position_ids)

    Power API:
        config = RotarySetup1DConfig(...)
        rope = RotarySetup1D.from_config(config)
    """

    def __init__(
        self,
        device: Any,
        batch_size: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        rope_scaling: Optional[Any] = None,
        use_qk_fused: bool = False,
        datatype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        """
        Simple API - computes all rotation matrices from parameters.

        Args:
            device: MeshDevice or single device.
            batch_size: Batch size for decode mode.
            head_dim: Head dimension (e.g., 64, 128).
            max_seq_len: Maximum sequence length for cos/sin tables.
            rope_theta: RoPE theta base frequency.
            rope_scaling: Optional RoPE scaling config (e.g., RopeScalingLlama3).
            use_qk_fused: If True, double batch for fused QK rotary + paged cache ops.
            datatype: Data type for TTNN tensors (default bfloat16).
        """
        super().__init__()
        config = RotarySetup1DConfig(
            device=device,
            batch_size=batch_size,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_qk_fused=use_qk_fused,
            datatype=datatype,
        )
        self._init_from_config(config)

    @classmethod
    def from_config(cls, config: RotarySetup1DConfig):
        """Power API - construct from config dataclass."""
        instance = object.__new__(cls)
        super(RotarySetup1D, instance).__init__()
        instance._init_from_config(config)
        return instance

    def _init_from_config(self, config: RotarySetup1DConfig) -> None:
        """Shared initialization logic."""
        self.config = config
        self.use_qk_fused = config.use_qk_fused
        self.original_batch_size = config.batch_size
        self.head_dim = config.head_dim
        self.device = config.device
        self.is_mesh_device = isinstance(config.device, ttnn._ttnn.multi_device.MeshDevice)

        # Resolve batch_size_per_device_group (no TG branch for 1D)
        doubled_batch_size = config.batch_size * 2 if config.use_qk_fused else config.batch_size
        self.batch_size_per_device_group = doubled_batch_size

        # Resolve core_grid
        self.core_grid = config.device.compute_with_storage_grid_size()

        # Compute cos/sin matrices on device
        cos_torch, sin_torch = _compute_cos_sin_matrices(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.cos_matrix, self.sin_matrix = _cos_sin_to_device(
            cos_torch,
            sin_torch,
            config.device,
            config.datatype,
        )

        # Batch grid for sharded memory configs
        self.batch_grid = ttnn.num_cores_to_corerangeset(
            self.batch_size_per_device_group, self.core_grid, row_wise=True
        )

        # Decode transformation matrix (height-sharded across batch cores)
        trans_mat = _get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(
            1,
            1,
            self.batch_size_per_device_group,
            1,
        )
        trans_mat_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.transformation_mat = ttnn.from_torch(
            trans_mat,
            device=config.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=config.datatype,
            memory_config=trans_mat_mem_config,
            mesh_mapper=replicate_tensor_to_mesh_mapper(config.device),
        )

        # Prefill transformation matrix (DRAM, head_dim x head_dim)
        prefill_trans_mat = _get_rot_transformation_mat(dhead=config.head_dim)
        self.transformation_mat_prefill = ttnn.from_torch(
            prefill_trans_mat,
            device=config.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=config.datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_to_mesh_mapper(config.device),
        )

    def get_both_trans_mats(self) -> Dict[str, ttnn.Tensor]:
        """Return both decode and prefill transformation matrices."""
        return {"decode": self.transformation_mat, "prefill": self.transformation_mat_prefill}

    def get_rot_idxs(self, position_idxs: torch.Tensor, on_host: bool = False) -> ttnn.Tensor:
        """
        Convert position indices to a TTNN rotation index tensor.

        Args:
            position_idxs: 1D torch tensor of position indices, shape [batch].
            on_host: If True, return tensor on host (for later device transfer).

        Returns:
            ttnn tensor of shape [1, padded_batch], dtype uint32.
        """
        assert isinstance(position_idxs, torch.Tensor), "Position ids must be a torch tensor"
        assert len(position_idxs.shape) == 1, "position idxs must be a [batch] tensor"

        if self.use_qk_fused:
            position_idxs = position_idxs.repeat(2)
            assert (
                position_idxs.shape[0] == self.batch_size_per_device_group
            ), "Position idxs must match batch_size_per_device_group"

        batch = position_idxs.shape[0]
        position_idxs = position_idxs.reshape(1, batch)
        assert torch.min(position_idxs) >= 0, "Position idxs must be non-negative"

        # Pad to nearest tile boundary
        pad_size = nearest_32(batch) - batch
        position_idxs = torch.nn.functional.pad(position_idxs, (0, pad_size), "constant", 0)

        if on_host:
            rot_idxs = ttnn.as_tensor(
                position_idxs,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=replicate_tensor_to_mesh_mapper(self.device),
            )
        else:
            rot_idxs = ttnn.as_tensor(
                position_idxs,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_tensor_to_mesh_mapper(self.device),
            )

        return rot_idxs

    def get_rot_mats(
        self, position_idxs: Union[torch.Tensor, ttnn.Tensor], return_rot_idxs: bool = False
    ) -> Union[List[ttnn.Tensor], Tuple[List[ttnn.Tensor], ttnn.Tensor]]:
        """
        Look up cos/sin rotation matrices for given position indices.

        Args:
            position_idxs: Position indices as torch.Tensor [batch] or ttnn.Tensor [1, batch].
            return_rot_idxs: If True, also return the processed rotation index tensor.

        Returns:
            [cos, sin] rotation matrices (height-sharded), or ([cos, sin], rot_idxs) if return_rot_idxs=True.
        """
        # Convert torch tensor to ttnn if needed
        if isinstance(position_idxs, torch.Tensor):
            rot_idxs = self.get_rot_idxs(position_idxs)
        else:
            rot_idxs = position_idxs
            assert len(rot_idxs.shape) == 2 and rot_idxs.shape[0] == 1, "rot_idxs must be a [1, batch] tensor"

        # Send to device if needed
        if rot_idxs.device != self.device:
            rot_idxs = ttnn.to_device(rot_idxs, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Embedding lookup for cos/sin
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)

        # Reshape: [1, batch, head_dim] -> [1, batch, 1[32], head_dim]
        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)

        # Transpose: [1, 1, batch, head_dim] -> [1, batch, 1[32], head_dim]
        cos = ttnn.transpose(cos, 1, 2)
        sin = ttnn.transpose(sin, 1, 2)

        # Trim to actual batch size if not tile-aligned
        if self.batch_size_per_device_group % ttnn.TILE_SIZE != 0:
            cos = cos[:, : self.batch_size_per_device_group, :, :]
            sin = sin[:, : self.batch_size_per_device_group, :, :]

        # Shard to batch cores
        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        cos = ttnn.interleaved_to_sharded(cos, mem_config)
        sin = ttnn.interleaved_to_sharded(sin, mem_config)

        if return_rot_idxs:
            return [cos, sin], rot_idxs
        return [cos, sin]

    # [INFO] this is the entry point for TTTv1 model_config.py and will retire with TTTv1
    @classmethod
    def from_model_args(
        cls,
        device: Any,
        args,
        model_name: str = "unknown",
    ):
        """Factory method for backward compatibility with ModelArgs.

        Args:
            device: MeshDevice or single device.
            args: Model arguments (ModelArgs instance).
            model_name: Model name for logging (default "unknown").
        """
        if args.is_galaxy:
            raise ValueError("RotarySetup1D cannot be used for Galaxy devices.")

        from models.tt_transformers.tt.common import rope_scaling_model_factory

        rope_scaling = None
        if hasattr(args, "rope_scaling_params") and args.rope_scaling_params is not None:
            rope_scaling = rope_scaling_model_factory(
                args.rope_scaling_params, getattr(args, "original_max_context_len", None)
            )

        return cls(
            device=device,
            batch_size=args.max_batch_size,
            head_dim=args.head_dim,
            max_seq_len=args.max_seq_len,
            rope_theta=args.rope_theta,
            rope_scaling=rope_scaling,
            use_qk_fused=getattr(args, "use_qk_fused", False),
            datatype=ttnn.bfloat16,
        )
